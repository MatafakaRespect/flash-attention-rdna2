// fa_decode.hip — FlashAttention decode с KV-кэшем под gfx1030, inference-путь.
// Особенности: split-KV (параллелизм по контексту -> сотни воркгрупп, загрузка GPU), PAGED KV
// (block_table + page_size), rotary на лету (neox / interleaved) для q, дозапись new_k/new_v в кэш
// (с rotary на k), GQA, sliding-window, head_dim {64,128,256}, dtype {fp16,bf16,fp32}.
// Decode КВ-bandwidth-bound (1 запрос) => математика во fp32 после конверсии загрузки: точно и просто,
// bf16 конвертится лосслес inline (мантисса 7⊂любая fp32). Никакого кванта.
// Референс структуры: flash_split.hip (split-KV) + flash_decode.hip (онлайн-softmax по тайлам).
// Сборка гейта: hipcc fa_decode.hip -o fa_decode.exe --offload-arch=gfx1030 -O3 -ffast-math -DFA_STANDALONE
#include "fa_common.hpp"

enum FaDtype { FA_F16=0, FA_BF16=1, FA_F32=2 };

struct DecParams {
    const void* q;              // [batch, NH, HD] новый токен (после/до rotary — см. apply_rotary_q)
    const void* Kc; const void* Vc;   // paged KV-кэш: [num_blocks, page, NKV, HD] (канон flash-attn)
    const int* block_table;     // [batch, max_blocks]
    void* o;                    // [batch, NH, HD]
    float* scratch;             // [batch*NH*nsplit*(HD+2)]
    const int* ctx_len;         // [batch] длина контекста (вкл. новый токен)
    const float* cos; const float* sin;  // [HD/2] для позиции нового токена (rotary q)
    int batch, NH, gqa, HD, page, max_blocks, nsplit, window;
    int page_shift, page_mask;  // page = 1<<page_shift (степень двойки) -> div/mod через сдвиг/маску
    int rotary;                 // 0 нет, 1 neox, 2 interleaved
    float scale;
    float sc_q, sc_k, sc_v;     // bf16 pow2-масштабы (по умолч. 1)
    // --- поля совместимости с flash-attn ---
    const int* leftpad;         // cache_leftpad[batch]: с какой позиции начинается кэш. nullptr => 0
    float* lse_out;             // return_softmax_lse: [batch, NH] натуральный log-sum-exp. nullptr => не писать
    float softcap;              // <=0 => выключен: s = softcap*tanh(s/softcap)
    const float* alibi;         // [NH] наклоны ALiBi или nullptr
    int nq;                     // сколько новых токенов за вызов (спекулятивка / chunked prefill); 1 = обычный декод
    int append;                 // 1 => new_k/new_v дописываются в кэш на позиции ctx_len..ctx_len+nq-1
    int rotary_dim;             // сколько первых измерений крутит rotary (0 => все HD)
    const int* rotary_pos;      // rotary_seqlens: позиция для rotary вместо ctx_len. nullptr => ctx_len
};

template<class DT> __device__ __forceinline__ float ld1(const typename DT::T* p,float sc){
    if constexpr(DT::is_bf16) return bf16_to_f(*p)*sc;
    else if constexpr(DT::is_fp32) return *p;
    else return (float)(*p)*sc;
}
// PSTRIDE = HD+2: [0]=m,[1]=l,[2..HD+1]=acc
// grid (batch, NH, nsplit), block 256.
// Переработка (старая версия — fd_partial_v0 ниже, оставлена для сравнения на стенде).
// Что было не так, по замеру и по коду:
//  1) индекс страницы читался из глобалки НА КАЖДУЮ позицию КАЖДЫМ потоком: в фазе V это
//     128 потоков × 256 позиций = 32768 чтений block_table на тайл вместо 256.
//  2) накопление V вели 128 потоков из 256, и каждый брал по 2 байта вместо 16.
//  3) редукция — два дерева по LDS, 18 __syncthreads на тайл.
// Здесь: block_table поднят в LDS один раз, V накапливают ВСЕ 256 потоков по 16 байт
// (у каждого 8 своих измерений и своя подгруппа позиций, частичные суммы сводятся один раз
// в самом конце), редукции — DPP внутри волны + один барьер на 16 частичных.
// G — сколько q-голов одной kv-головы обслуживает воркгруп (переиспользование KV).
// Без него каждая kv-голова читается gqa раз разными воркгрупами: полезных байт столько же,
// а выданных — в gqa раз больше, и упор идёт в кэш, а не в полосу. С G=gqa K и V читаются раз,
// а дотов и накоплений просто G штук на ту же прочитанную строку.
template<class DT,int HD,int G>
__global__ __launch_bounds__(256)
void fd_partial(DecParams p)
{
    typedef typename DT::T T;
    // Несколько новых токенов за вызов (спекулятивное декодирование, chunked prefill): ось x
    // сетки несёт пару (элемент батча, номер нового токена). Токен с номером tk стоит на позиции
    // ctx-nq+tk, поэтому у него свой конец контекста — так получается причинность между новыми
    // токенами без отдельного ядра. K/V при этом читаются каждым токеном заново; за точность
    // это не платит ничем, за полосу платит, и для nq в единицы токенов это приемлемо.
    const int bx=blockIdx.x, kvh=blockIdx.y, sp=blockIdx.z, t=threadIdx.x;
    const int b=bx/p.nq, tk=bx%p.nq;
    const int h0=kvh*G;
    // ctx_len[b] — длина кэша ДО дозаписи (ровно cache_seqlens у flash-attn). Свой ключ токена
    // появляется в кэше только в режиме дозаписи, поэтому конец контекста зависит от неё.
    const int qpos = p.ctx_len[b] + tk;              // позиция нового токена в последовательности
    const int ctx  = p.ctx_len[b] + (p.append ? (tk+1) : 0);
    // cache_leftpad: ключи до lp в кэше не наши, начинаем контекст с них. Одно чтение на воркгруп.
    const int lp = p.leftpad? p.leftpad[b] : 0;
    const int chunk=(ctx-lp+p.nsplit-1)/p.nsplit;
    const int start=lp+sp*chunk, end=min(start+chunk,ctx);
    constexpr bool F32 = DT::is_fp32;
    typedef typename std::conditional<F32,float,_Float16>::type QT;
    constexpr int NDG = HD/8;          // групп измерений (по 8 измерений на поток)
    constexpr int NRG = 256/NDG;       // групп позиций, идущих параллельно (floor!)
    const int dl = t % NDG, rg = t / NDG, d0 = dl*8;
    // При head_dim, не дающем целого 256/NDG (например 112 -> 18 групп по 14 потоков = 252),
    // хвостовые потоки не входят ни в одну группу: они обязаны молчать, иначе пишут за границу
    // mrg и вдобавок повторно складывают чужие позиции в накопление V.
    const bool vact = (rg < NRG);
    const int NKV = p.NH/p.gqa;

    __shared__ QT qs[G][HD];
    __shared__ float ptile[G][256];
    __shared__ int   btile[256];       // индексы блоков для позиций тайла — читаются раз
    __shared__ float wred[G][16];      // частичные максимумы от полуволн, строка на голову
    __shared__ float wsum[G][16];      // частичные суммы — ОТДЕЛЬНО: иначе запись суммы
                                       // затирает максимум, который другие потоки ещё читают
    __shared__ float mrg[NRG][HD];     // сведение частичных сумм V (переиспользуется по головам)
    // загрузка q + rotary (позиция нового токена = ctx-1)
    #pragma unroll
    for(int g=0;g<G;g++){
        const T* qg=(const T*)p.q+((size_t)(b*p.nq+tk)*p.NH+h0+g)*HD;
        for(int d=t; d<HD; d+=256) qs[g][d]=(QT)ld1<DT>(qg+d,p.sc_q);
    }
    __syncthreads();
    // rotary: cos/sin — ТАБЛИЦА [seqlen_ro, rotary_dim/2] (как у flash-attn), строка берётся по
    // позиции токена. rotary_dim может быть меньше head_dim — тогда хвост измерений не крутится.
    const int rdim = p.rotary_dim>0 ? p.rotary_dim : HD;
    if(p.rotary && t<rdim/2){
        const int rp=(p.rotary_pos? p.rotary_pos[b] : p.ctx_len[b])+tk;
        int d=t; const size_t ro=(size_t)rp*(rdim/2)+d;
        float c=p.cos[ro], s=p.sin[ro];
        #pragma unroll
        for(int g=0;g<G;g++){
            if(p.rotary==1){ float x=qs[g][d], y=qs[g][d+rdim/2]; qs[g][d]=(QT)(x*c-y*s); qs[g][d+rdim/2]=(QT)(y*c+x*s); }
            else { float x=qs[g][2*d], y=qs[g][2*d+1]; qs[g][2*d]=(QT)(x*c-y*s); qs[g][2*d+1]=(QT)(y*c+x*s); }
        }
    }
    __syncthreads();

    float acc[G][8];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<8;i++) acc[g][i]=0.0f;

    const int lo = p.window>=0 ? (qpos - p.window + 1) : 0;   // нижняя граница окна от позиции запроса
    float m[G], l[G];
    #pragma unroll
    for(int g=0;g<G;g++){ m[g]=-1e30f; l[g]=0.0f; }

    for(int base=start; base<end; base+=256){
        // индекс блока для каждой позиции тайла — ОДНО чтение глобалки на позицию
        { int pos=base+t;
          btile[t] = (pos<end)? p.block_table[(size_t)b*p.max_blocks + (pos>>p.page_shift)] : 0; }
        __syncthreads();
        const int pos=base+t;
        const bool valid = pos<end && (p.window<0 || pos>=lo);
        float s[G];
        #pragma unroll
        for(int g=0;g<G;g++) s[g]=-1e30f;
        if(valid){
            int blk=btile[t], slot=pos&p.page_mask;
            const T* kp=(const T*)p.Kc + (((size_t)blk*p.page + slot)*NKV + kvh)*HD;
            if constexpr(F32){
                #pragma unroll
                for(int g=0;g<G;g++){ float dd=0;
                    #pragma unroll 4
                    for(int i=0;i<HD;i++) dd+=(float)qs[g][i]*ld1<DT>(kp+i,1.0f);
                    s[g]=dd*p.scale; }
            } else if constexpr(DT::is_bf16){
                // строку K конвертируем ОДИН раз в регистры, потом G дотов по ней
                half2_t kk[HD/2]; float sck=p.sc_k;
                #pragma unroll
                for(int i=0;i<HD/8;i++){ uint4 r=*(const uint4*)(kp+i*8); const uint16_t* w=(const uint16_t*)&r;
                    #pragma unroll
                    for(int mm=0;mm<4;mm++){ uint32_t b0=((uint32_t)w[2*mm])<<16,b1=((uint32_t)w[2*mm+1])<<16;
                        float f0,f1; __builtin_memcpy(&f0,&b0,4); __builtin_memcpy(&f1,&b1,4);
                        kk[i*4+mm]=half2_t{(_Float16)(f0*sck),(_Float16)(f1*sck)}; } }
                #pragma unroll
                for(int g=0;g<G;g++){ const half2_t* q2=(const half2_t*)qs[g]; float dd=0;
                    #pragma unroll
                    for(int i=0;i<HD/2;i++) dd=__builtin_amdgcn_fdot2(q2[i],kk[i],dd,false);
                    s[g]=dd*p.scale/(p.sc_q*p.sc_k); }
            } else {
                half2_t kk[HD/2];
                const half2_t* k2=(const half2_t*)kp;
                #pragma unroll
                for(int i=0;i<HD/2;i++) kk[i]=k2[i];      // строка K в регистрах, дальше G дотов
                #pragma unroll
                for(int g=0;g<G;g++){ const half2_t* q2=(const half2_t*)qs[g]; float dd=0;
                    #pragma unroll
                    for(int i=0;i<HD/2;i++) dd=__builtin_amdgcn_fdot2(q2[i],kk[i],dd,false);
                    s[g]=dd*p.scale; }
            }
        }
        // softcap и alibi — как во forward: softcap сжимает скор до s=c*tanh(s/c), alibi
        // добавляет наклон по расстоянию (позиция запроса = ctx-1, ключа = pos).
        if(valid && (p.softcap>0.0f || p.alibi)){
            #pragma unroll
            for(int g=0;g<G;g++){
                if(p.softcap>0.0f) s[g]=p.softcap*tanhf(s[g]/p.softcap);
                if(p.alibi)        s[g]+=p.alibi[h0+g]*(float)(pos-qpos);
            }
        }
        // ---- max и сумма по тайлу: DPP внутри 16 линий + один барьер на 16 частичных ----
        #pragma unroll
        for(int g=0;g<G;g++){ float lm=red16_max(s[g]); if((t&15)==0) wred[g][t>>4]=lm; }
        __syncthreads();
        float nm[G], so[G];
        #pragma unroll
        for(int g=0;g<G;g++){
            float tmax=-1e30f;
            #pragma unroll
            for(int r=0;r<16;r++) tmax=fmaxf(tmax,wred[g][r]);
            nm[g]=fmaxf(m[g],tmax); so[g]=__expf(m[g]-nm[g]);
            float pv=valid?__expf(s[g]-nm[g]):0.0f;
            ptile[g][t]=pv;
            float ls=red16_sum(pv);
            if((t&15)==0) wsum[g][t>>4]=ls;
        }
        __syncthreads();
        #pragma unroll
        for(int g=0;g<G;g++){
            float tsum=0.0f;
            #pragma unroll
            for(int r=0;r<16;r++) tsum+=wsum[g][r];
            l[g]=l[g]*so[g]+tsum; m[g]=nm[g];
            #pragma unroll
            for(int i=0;i<8;i++) acc[g][i]*=so[g];
        }

        // ---- вклад тайла в O: все 256 потоков, чтение по 16 байт, строка V — одна на G голов ----
        const int tn=min(end-base,256);
        if(vact) for(int pp=rg; pp<tn; pp+=NRG){
            int gp=base+pp, blk=btile[pp], slot=gp&p.page_mask;
            const T* vp=(const T*)p.Vc + (((size_t)blk*p.page + slot)*NKV + kvh)*HD + d0;
            float vv[8];
            if constexpr(F32){
                #pragma unroll
                for(int i=0;i<8;i++) vv[i]=(float)vp[i];
            } else if constexpr(DT::is_bf16){
                uint4 r=*(const uint4*)vp; const uint16_t* wd=(const uint16_t*)&r;
                #pragma unroll
                for(int i=0;i<8;i++){ uint32_t u=((uint32_t)wd[i])<<16; float f;
                    __builtin_memcpy(&f,&u,4); vv[i]=f*p.sc_v; }
            } else {
                half8_t a=*(const half8_t*)vp;
                #pragma unroll
                for(int i=0;i<8;i++) vv[i]=(float)a[i];
            }
            #pragma unroll
            for(int g=0;g<G;g++){ float w=ptile[g][pp];
                #pragma unroll
                for(int i=0;i<8;i++) acc[g][i]+=w*vv[i]; }
        }
        __syncthreads();
    }
    // ---- свести NRG частичных сумм, по одной голове за раз (буфер общий) ----
    #pragma unroll
    for(int g=0;g<G;g++){
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++) if(vact) mrg[rg][d0+i]=acc[g][i];
        __syncthreads();
        float* out=p.scratch + (size_t)(((size_t)(b*p.nq+tk)*p.NH+h0+g)*p.nsplit+sp)*(HD+2);
        if(t==0){ out[0]=(start<end)?m[g]:-1e30f; out[1]=l[g]; }
        for(int d=t; d<HD; d+=256){
            float a=0.0f;
            #pragma unroll
            for(int r=0;r<NRG;r++) a+=mrg[r][d];
            out[2+d]=a/p.sc_v;             // unfold bf16 V-масштаба (=1 по умолч.)
        }
    }
}

// grid (batch, NH), block HD. Слить nsplit срезов.
template<int HD>
__global__ void fd_combine(void* o,const float* scratch,int nsplit,int NH,int dtype,float* lse_out){
    const int b=blockIdx.x, h=blockIdx.y, t=threadIdx.x;
    const float* base=scratch + (size_t)(b*NH+h)*nsplit*(HD+2);
    float gm=-1e30f;
    for(int s=0;s<nsplit;s++) gm=fmaxf(gm, base[(size_t)s*(HD+2)]);
    float acc=0.0f, l=0.0f;
    for(int s=0;s<nsplit;s++){ float w=__expf(base[(size_t)s*(HD+2)]-gm);
        acc += w*base[(size_t)s*(HD+2)+2+t]; l += w*base[(size_t)s*(HD+2)+1]; }
    float r = l>0? acc/l : 0.0f;
    // return_softmax_lse: онлайн-softmax тут ведётся по натуральному основанию (__expf),
    // значит полный logsumexp строки = gm + ln(l). Пишет один поток, только если попросили.
    if(lse_out && t==0) lse_out[(size_t)b*NH+h] = l>0? (gm+__logf(l)) : -INFINITY;
    size_t idx=((size_t)b*NH+h)*HD+t;
    if(dtype==FA_F32) ((float*)o)[idx]=r;
    else if(dtype==FA_BF16) ((__hip_bfloat16*)o)[idx]=(__hip_bfloat16)r;
    else ((_Float16*)o)[idx]=(_Float16)r;
}

// дозапись new_k/new_v в paged-кэш на позицию pos_new=ctx-1, с rotary на k. grid(batch,NKV), block HD.
template<class DT,int HD>
__global__ void fd_append(DecParams p,const void* new_k,const void* new_v){
    typedef typename DT::T T;
    const int bx=blockIdx.x, kvh=blockIdx.y, t=threadIdx.x;
    const int b=bx/p.nq, tk=bx%p.nq;
    const int pos=p.ctx_len[b]+tk;   // новые токены встают сразу за концом кэша
    int blk=p.block_table[(size_t)b*p.max_blocks + (pos>>p.page_shift)], slot=pos&p.page_mask;
    const int NKV=p.NH/p.gqa;
    const T* nk=(const T*)new_k+((size_t)(b*p.nq+tk)*NKV+kvh)*HD;
    const T* nv=(const T*)new_v+((size_t)(b*p.nq+tk)*NKV+kvh)*HD;
    T* kc=(T*)p.Kc + (((size_t)blk*p.page+slot)*NKV+kvh)*HD;
    T* vc=(T*)p.Vc + (((size_t)blk*p.page+slot)*NKV+kvh)*HD;
    // rotary на k (во float, обратно в T)
    __shared__ float ks[HD];
    if(t<HD) ks[t]=ld1<DT>(nk+t,1.0f);
    __syncthreads();
    const int rdim = p.rotary_dim>0 ? p.rotary_dim : HD;
    if(p.rotary && t<rdim/2){
        const int rp=(p.rotary_pos? p.rotary_pos[b] : p.ctx_len[b])+tk;
        int d=t; const size_t ro=(size_t)rp*(rdim/2)+d;
        float c=p.cos[ro], s=p.sin[ro];
        if(p.rotary==1){ float x=ks[d],y=ks[d+rdim/2]; ks[d]=x*c-y*s; ks[d+rdim/2]=y*c+x*s; }
        else { float x=ks[2*d],y=ks[2*d+1]; ks[2*d]=x*c-y*s; ks[2*d+1]=y*c+x*s; } }
    __syncthreads();
    if(t<HD){ if constexpr(DT::is_bf16){ kc[t]=(__hip_bfloat16)ks[t]; vc[t]=(__hip_bfloat16)ld1<DT>(nv+t,1.0f); }
              else if constexpr(DT::is_fp32){ kc[t]=ks[t]; vc[t]=(*(nv+t)); }
              else { kc[t]=(_Float16)ks[t]; vc[t]=(_Float16)ld1<DT>(nv+t,1.0f); } }
}

// Воркгрупов = batch * (NH/gqa) * nsplit. При GQA-переиспользовании их в gqa раз меньше,
// поэтому nsplit поднимаем на gqa — иначе на 72 CU просто нечего класть.
// Воркгрупов = batch * (NH/gqa) * nsplit. Цель — занять карту (~288 воркгрупов на 72 CU)
// и не дробить мельче: каждый лишний срез это ещё одна строка в scratch и работа для fd_combine.
static int calc_nsplit(int ctx,int gqa=1,int NH=32,int batch=1){
    const int NKV = NH/(gqa>0?gqa:1);
    int groups = batch*(NKV>0?NKV:1);
    int n = (288 + groups - 1)/groups;
    int maxn=(ctx+255)/256; if(maxn<1) maxn=1;
    if(n>maxn) n=maxn;
    if(n<1) n=1; if(n>512) n=512;
    return n;
}

// ---- host launch ----
template<class DT,int HD>
static void launch_dec(DecParams& p,int dtype,hipStream_t st){
    const int NKV=p.NH/p.gqa;
    // Один воркгруп на kv-голову обслуживает G=gqa q-голов => K и V читаются раз, а не gqa раз.
    // Воркгрупов при этом в gqa раз меньше, поэтому split по контексту поднимаем во столько же,
    // иначе карта недогружена и выигрыш съедается простоем.
    #define DEC_G(GG) do{ dim3 gp(p.batch*p.nq,NKV,p.nsplit); fd_partial<DT,HD,GG><<<gp,256,0,st>>>(p); }while(0)
    switch(p.gqa){
        case 8: DEC_G(8); break;
        case 4: DEC_G(4); break;
        case 2: DEC_G(2); break;
        default: DEC_G(1); break;
    }
    #undef DEC_G
    fd_combine<HD><<<dim3(p.batch*p.nq,p.NH),HD,0,st>>>(p.o,p.scratch,p.nsplit,p.NH,dtype,p.lse_out);
}
template<class DT> static void append_hd(DecParams& p,const void* nk,const void* nv,hipStream_t st){
    int NKV=p.NH/p.gqa; dim3 g(p.batch*p.nq,NKV);
    #define APP(HD) case HD: fd_append<DT,HD><<<g,((HD)<64?64:(HD)),0,st>>>(p,nk,nv); break;
    switch(p.HD){ APP(16) APP(32) APP(48) APP(64) APP(80) APP(96)
                  APP(112) APP(128) APP(160) APP(192) APP(224) APP(256)
                  default: printf("fa_decode append: head_dim=%d не инстанцирован\n",p.HD); abort(); }
    #undef APP
}
extern "C" void fa_decode_launch(DecParams p,int dtype,hipStream_t st,const void* new_k,const void* new_v){
    if(p.sc_q==0.0f)p.sc_q=1.0f; if(p.sc_k==0.0f)p.sc_k=1.0f; if(p.sc_v==0.0f)p.sc_v=1.0f;
    if(p.nq<1) p.nq=1;
    p.append = new_k? 1 : 0;
    p.page_shift=__builtin_ctz(p.page); p.page_mask=p.page-1;   // page — степень двойки
    if(new_k){ if(dtype==FA_F16) append_hd<DT_F16>(p,new_k,new_v,st);
               else if(dtype==FA_BF16) append_hd<DT_BF16>(p,new_k,new_v,st);
               else append_hd<DT_F32>(p,new_k,new_v,st);
        HIP_CHECK(hipDeviceSynchronize()); }   // кэш дозаписан до внимания
    #define DECHD(DTt) do{ switch(p.HD){ \
        case 16: launch_dec<DTt, 16>(p,dtype,st);break; case 32: launch_dec<DTt, 32>(p,dtype,st);break; \
        case 48: launch_dec<DTt, 48>(p,dtype,st);break; case 64: launch_dec<DTt, 64>(p,dtype,st);break; \
        case 80: launch_dec<DTt, 80>(p,dtype,st);break; case 96: launch_dec<DTt, 96>(p,dtype,st);break; \
        case 112:launch_dec<DTt,112>(p,dtype,st);break; case 128:launch_dec<DTt,128>(p,dtype,st);break; \
        case 160:launch_dec<DTt,160>(p,dtype,st);break; case 192:launch_dec<DTt,192>(p,dtype,st);break; \
        case 224:launch_dec<DTt,224>(p,dtype,st);break; case 256:launch_dec<DTt,256>(p,dtype,st);break; \
        default: printf("fa_decode: head_dim=%d не инстанцирован\n",p.HD); abort(); } }while(0)
    if(dtype==FA_F16) DECHD(DT_F16); else if(dtype==FA_BF16) DECHD(DT_BF16); else DECHD(DT_F32);
    #undef DECHD
}

#ifdef FA_STANDALONE
#include "fa_decode_gate.inc"
#endif
