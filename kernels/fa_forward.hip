// fa_forward.hip — точный FlashAttention-2 forward под gfx1030 (RX 6800 XT), inference-путь.
// ГЛАВНЫЙ РЫЧАГ vs чемпион (flash_prefill3, memory-bound 94% HBM): GQA-REUSE — один воркгруп
// обслуживает G q-голов ОДНОЙ kv-головы, K/V грузятся раз и переиспользуются => глоб-трафик K/V ÷G.
// Это переносит ядро из memory-bound в compute-bound (потолок структуры ~30 TFLOP/s на полных клоках).
//
// Дизайн (закладка оптимизаций СРАЗУ):
//  * блок = (q-тайл BR, группа из G голов, батч). 256 потоков = 16 ty × 16 tx, t=ty*16+tx.
//  * обе матрицы (S=QKᵀ, O=PV) — регистровый fdot2-тайлинг; softmax-редукция по ключам =
//    внутриволновой __shfl_xor(width=16) БЕЗ LDS; exp2f; НЕ материализует S×S.
//  * K/V half8 (128-бит) коалесц. загрузки; Q-тайлы резидентны в LDS (грузятся раз).
//  * lazy-rescale: пропуск corr-домножения acco когда max не сдвинулся (лосслес).
//  * dtype-generic <fp16,bf16>: bf16 идёт через ТОТ ЖЕ v_dot2 — per-tile pow2-масштаб приводит bf16
//    в диапазон fp16 БЕЗ потери мантиссы (7 бит bf16 ⊂ 10 бит fp16), unfold степенью двойки точен.
//  * фичи: causal(bottom-right при SQ≠SK), sliding-window, softcap, alibi, GQA, батч, произв. страйды
//    (dim contiguous), varlen (cu_seqlens), выход O и LSE(fp32), dropout(Philox, p=0 бесплатно).
// fp32-путь — отдельное ядро (fa_forward_fp32) ниже, корректность 1e-5 (fdot2 только fp16).
// Сборка гейта: hipcc fa_forward.hip -o fa_forward.exe --offload-arch=gfx1030 -O3 -ffast-math -DFA_STANDALONE
#include "fa_common.hpp"

// ---- страйды тензора (dim contiguous, stride_d=1) ----
struct Strides { long long b,h,s; };   // batch, head, seq (в элементах)

struct FwdParams {
    const void* Q; const void* K; const void* V; void* O; float* LSE;   // LSE опц.
    Strides sQ,sK,sV,sO;
    int SQ, SK, HD, NH, gqa;
    float scale, softcap;               // softcap<=0 => off
    // Окно в семантике flash-attn: разрешены ключи [qq+delta-win_left .. qq+delta+win_right],
    // отрицательное значение = границы с этой стороны нет. window_size=(-1,-1) => обе <0;
    // causal = win_right==0. Раньше здесь было одно поле window (только левое окно с правой
    // границей на диагонали), и двусторонние окна не выражались вовсе.
    int   win_left, win_right;
    const float* alibi_slopes;          // [NH] или nullptr
    int   causal;                       // bottom-right при SQ!=SK
    float dropout_p; unsigned long long philox_seed;
    const int* cu_q; const int* cu_k;   // varlen (nullptr => dense)
    int batch;
    // bf16-range pow2-масштабы (лосслес): fp16-значение = bf16-значение * sc_*. По умолч. 1.
    // Мантисса bf16 (7 бит) ⊂ fp16 (10) — конвертация точна; sc двигает экспоненту в диапазон fp16.
    float sc_q, sc_k, sc_v;
    // --- поля совместимости с flash-attn, все выключаются нулём/nullptr ---
    // Чанкованное внимание (attention_chunk у FA-3, схема Llama-4): запрос позиции i видит
    // только ключи своего чанка [i/chunk*chunk, (i/chunk+1)*chunk-1], поверх causal/window.
    // 0 => выключено. Читается ТОЛЬКО под FEAT, поэтому плотный путь не меняется ни на инструкцию.
    int attn_chunk;
    // seqused_q/seqused_k (varlen): фактически используемая длина последовательности, короче,
    // чем даёт cu_seqlens. nullptr => берём всю длину из cu_seqlens.
    const int* seqused_q; const int* seqused_k;
};

// ========================= главное ядро (fp16/bf16) =========================
template<class DT,int HD,int BR,int BC,int G,bool CAUSAL,bool FEAT>
__global__ __launch_bounds__(256)
void fa_forward_k(FwdParams p)
{
    typedef typename DT::T T;
    constexpr bool SCALED = DT::is_bf16;   // bf16: inline convert + per-tensor pow2 scale (лосслес, скорость fp16)
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
    constexpr float LOG2E=1.442695041f;

    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z;
    const int hg=blockIdx.y;
    const int h0=hg*G, kvh=h0/p.gqa;

    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        // seqused_*: последовательность в буфере длиннее, чем реально используется (FA-3).
        // Одно скалярное чтение на воркгруп, вне всех циклов.
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR;
    if(qb>=SQ) return;
    const int delta=SK-SQ;
    const float scale2=p.scale*LOG2E;
    // bf16 pow2-масштабы: конверсия bf16->fp16 домножает на sc_*, unfold — деление (точно, pow2)
    const float scq = SCALED? p.sc_q : 1.0f, sck = SCALED? p.sc_k : 1.0f, scv = SCALED? p.sc_v : 1.0f;
    const float uqk = SCALED? 1.0f/(scq*sck) : 1.0f;   // unfold Q·K
    const float uv  = SCALED? 1.0f/scv       : 1.0f;   // unfold в хранении P

    // ---- LDS раскладка (ручной пул, ≤64КБ) ----
    constexpr int QS_B = G*HH*BR*4;
    constexpr int KS_B = HH*BC*4;
    constexpr int VS_B = BCH*HD*4;
    constexpr int PS_B = (BR<=HD)?0:(BC*BR*2);      // Ps алиасит Ks если BR<=HD
    __shared__ char pool[QS_B+KS_B+VS_B+PS_B];
    half2_t* Qs=(half2_t*)pool;
    half2_t* Ks=(half2_t*)(pool+QS_B);
    half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
    _Float16* Ps=(BR<=HD)?(_Float16*)Ks:(_Float16*)(pool+QS_B+KS_B+VS_B);
    half2_t*  Pp=(half2_t*)Ps;   // тот же буфер, вид «пары ключей в одном dword»
    #define QSA(g,dh,q) Qs[((g)*HH+(dh))*BR+(q)]
    // Свизл kc по dh. Запись K в LDS идёт линиями, у которых меняется dh при одном kc
    // (так надо для коалесцированного чтения из глобалки), и адрес (dh)*BC+kc клал все
    // 16 линий в ОДИН банк — 16-кратный конфликт в стадии загрузки, а это 18% времени.
    // XOR разводит их по 16 банкам; чтение в GEMM1 (фиксированный dh, разные kc) остаётся
    // бесконфликтным. Паддинг строки сюда не влезает — до лимита 64 КБ не хватает 256 байт.
    #define KSA(dh,kc)  Ks[(dh)*BC+((kc)^((dh)&31))]
    // Тот же приём для записи V: линии меняют dg с шагом 8 dword => 4 банка на 16 линий.
    // XOR по старшим битам d разводит запись на 16 банков, а чтение в GEMM2 (линии меняют tx
    // внутри одного e) остаётся бесконфликтным — там XOR идёт на константу.
    #define VSA(kh,d)   Vs[(kh)*HD+((d)^(((d)>>4)&7))]
    #define PSA(kc,q)   Ps[(kc)*BR+(q)]
    #define PPA(kh,q)   Pp[(kh)*BR+(q)]

    // inline-загрузка 8 элементов (128-бит) с конверсией bf16->fp16*sc
    auto load8=[&](const T* pt,bool valid,float sc,_Float16 o[8]){
        if constexpr(SCALED){
            uint4 raw=valid?*(const uint4*)pt:uint4{0,0,0,0}; const uint16_t* w=(const uint16_t*)&raw;
            #pragma unroll
            for(int m=0;m<8;m++){ uint32_t b=((uint32_t)w[m])<<16; float f; __builtin_memcpy(&f,&b,4); o[m]=(_Float16)(f*sc); }
        } else {
            half8_t a=valid?*(const half8_t*)pt:half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<8;m++) o[m]=a[m];
        }
    };

    // ---- загрузка Q-тайлов (резидент) ----
    #pragma unroll
    for(int g=0; g<G; g++){
        const T* Qb=(const T*)p.Q + qrow0 + (long long)(h0+g)*p.sQ.h;
        for(int e=t; e<BR*(HD/8); e+=256){
            int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q; _Float16 o[8];
            load8(Qb+(long long)sq*p.sQ.s+dg*8, sq<SQ, scq, o);
            #pragma unroll
            for(int m=0;m<4;m++) QSA(g,dg*4+m,q)=half2_t{o[2*m],o[2*m+1]};
        }
    }

    float acco[G][QPT][DPT]; float rm[G][QPT], rl[G][QPT];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<QPT;i++){ rm[g][i]=-1e30f; rl[g][i]=0.0f;
            #pragma unroll
            for(int e=0;e<DPT;e++) acco[g][i][e]=0.0f; }
    __syncthreads();

    int kmax = CAUSAL? (((qb+BR-1)+delta < SK-1)?(qb+BR-1)+delta:SK-1) : SK-1;
    // ранний старт/конец по окну: тайлы целиком вне окна пропускаем. Нижняя граница берётся
    // по ПЕРВОЙ строке тайла, верхняя — по последней: так они режут тайл для всех его строк.
    int kb0=0;
    if(FEAT && p.win_left>=0){ int lo=qb+delta-p.win_left; if(lo<0) lo=0; kb0=(lo/BC)*BC; }
    if(FEAT && p.win_right>=0){ int hi=qb+BR-1+delta+p.win_right; if(hi<kmax) kmax=hi; }
    // то же для чанкованного внимания: у первой строки тайла чанк начинается раньше всех,
    // у последней — кончается позже всех, значит эти две границы режут тайлы для всего блока.
    if(FEAT && p.attn_chunk>0){
        int p0=qb+delta;         if(p0<0) p0=0;      // деление отрицательных в C урезает к нулю
        int p1=qb+BR-1+delta;    if(p1<0) p1=0;
        int lo=(p0/p.attn_chunk)*p.attn_chunk;
        if(lo>kb0) kb0=(lo/BC)*BC;
        int hi=(p1/p.attn_chunk+1)*p.attn_chunk-1;
        if(hi<kmax) kmax=hi;
    }

    for(int kb=kb0; kb<=kmax; kb+=BC){
        const bool diag = CAUSAL && ((kb+BC-1)>=qb+delta);
        __syncthreads();
        // K (транспон в Ks[dh][kc]) — развёрнутая загрузка, несколько 128-бит чтений в полёте (MLP)
        constexpr int NK=(BC*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0; it<NK; it++){ int e=t+it*256; if(e<BC*(HD/8)){
            int dg=e%(HD/8), kc=e/(HD/8); int sk=kb+kc; _Float16 o[8];
            load8((const T*)p.K+krow0+(long long)kvh*p.sK.h+(long long)sk*p.sK.s+dg*8, sk<SK, sck, o);
            #pragma unroll
            for(int m=0;m<4;m++) KSA(dg*4+m,kc)=half2_t{o[2*m],o[2*m+1]}; } }
        // V (Vs[kh][d], 2 ряда упакованы в half2)
        constexpr int NV=(BCH*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0; it<NV; it++){ int e=t+it*256; if(e<BCH*(HD/8)){
            int dg=e%(HD/8), kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1; int d0=dg*8; _Float16 a[8],b[8];
            load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s0*p.sV.s+d0, s0<SK, scv, a);
            load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s1*p.sV.s+d0, s1<SK, scv, b);
            #pragma unroll
            for(int m=0;m<8;m++) VSA(kh,d0+m)=half2_t{a[m],b[m]}; } }
        __syncthreads();

        // GEMM1 (K жив, все G голов)
        float accs[G][QPT][KPT];
        #pragma unroll
        for(int g=0;g<G;g++)
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) accs[g][i][j]=0.0f;
        #pragma unroll 4
        for(int dh=0; dh<HH; dh++){
            half2_t kv[KPT];
            // Ключи потока: (j/2)*32 + 2tx + (j%2) вместо tx*KPT+j. Шаг между линиями 2 dword
            // вместо KPT=4 => 16 разных банков вместо 8, и пары ключей остаются соседними,
            // так что упаковка P в half2 ниже продолжает работать.
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=KSA(dh,(j>>1)*32+2*tx+(j&1));
            #pragma unroll
            for(int g=0;g<G;g++){
                half2_t qv[QPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) qv[i]=QSA(g,dh,ty*QPT+i);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int j=0;j<KPT;j++) accs[g][i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[g][i][j],false);
            }
        }
        __syncthreads();  // K мёртв -> Ps алиас

        #pragma unroll
        for(int g=0; g<G; g++){
            const int h=h0+g;
            const float aslope = (FEAT && p.alibi_slopes)? p.alibi_slopes[h] : 0.0f;
            #pragma unroll
            for(int i=0;i<QPT;i++){
                const int qq=qb+ty*QPT+i;
                float sj[KPT], lmax=-1e30f;
                #pragma unroll
                for(int j=0;j<KPT;j++){
                    const int key=kb+(j>>1)*32+2*tx+(j&1);   // та же раскладка, что у kv[]
                    float a=accs[g][i][j]; if constexpr(SCALED) a*=uqk;   // истинный Q·K
                    float sv;
                    if constexpr(FEAT){
                        if(p.softcap>0.0f){ float x=a*p.scale; sv=p.softcap*tanhf(x/p.softcap)*LOG2E; }
                        else sv=a*scale2;
                        if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*LOG2E;
                        bool masked = CAUSAL ? false : (key>=SK);
                        if(CAUSAL && diag && key>qq+delta) masked=true;
                        if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
                        if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
                        if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0;
                            int cs=(pp/p.attn_chunk)*p.attn_chunk;
                            if(key<cs||key>=cs+p.attn_chunk) masked=true; }
                        if(masked) sv=-1e30f;
                    } else {
                        sv=a*scale2;
                        // CAUSAL: маска только в диагональном тайле (хвост key>=SK покрыт причинностью).
                        if constexpr(CAUSAL){ if(diag && key>qq+delta) sv=-1e30f; }
                        else { if(key>=SK) sv=-1e30f; }
                    }
                    sj[j]=sv; lmax=fmaxf(lmax,sv);
                }
                lmax=red16_max(lmax);                 // DPP вместо ds_bpermute
                float mnew=fmaxf(rm[g][i],lmax);
                float corr = (mnew>rm[g][i])? __builtin_amdgcn_exp2f(rm[g][i]-mnew) : 1.0f;  // lazy-rescale
                float lsum=0.0f;
                #pragma unroll
                for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
                lsum=red16_sum(lsum);                 // DPP вместо ds_bpermute
                rl[g][i]=rl[g][i]*corr+lsum; rm[g][i]=mnew;
                if(corr!=1.0f){
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]*=corr;
                }
                if(FEAT && p.dropout_p>0.0f){ float ik=1.0f/(1.0f-p.dropout_p);
                    #pragma unroll
                    for(int j=0;j<KPT;j++){ int key=kb+(j>>1)*32+2*tx+(j&1);
                        unsigned long long idx=(((unsigned long long)h*SQ+qq)*SK+key);
                        sj[j]*=dropout_mul(p.philox_seed,idx,p.dropout_p,ik); } }
                // Ключи потока идут подряд (tx*KPT .. +KPT-1), поэтому пары пакуем сами в half2.
                // GEMM2 тогда читает готовый dword вместо двух half + v_perm (RGA: 32 v_perm на тайл).
                #pragma unroll
                for(int j=0;j<KPT;j+=2){
                    float a = SCALED? sj[j]*uv   : sj[j];
                    float b = SCALED? sj[j+1]*uv : sj[j+1];
                    PPA((j>>1)*16+tx, ty*QPT+i)=half2_t{(_Float16)a,(_Float16)b};
                }
            }
            __syncthreads();
            #pragma unroll 4
            for(int kh=0; kh<BCH; kh++){
                half2_t pv[QPT], vv[DPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) pv[i]=PPA(kh,ty*QPT+i);   // один b32, без распаковки
                // Измерения LDS: при d=tx*DPT+e шаг между линиями DPT=8 dword, и 16 линий
                // попадают всего в 4 банка — 4-кратный конфликт в самом горячем цикле (GEMM2
                // это 36% времени). Шаг 16 разводит линии по 16 разным банкам. Замер: +9%.
                // Бонусом эпилог пишет O подряд по линиям, а не с шагом 8.
                #pragma unroll
                for(int e=0;e<DPT;e++) vv[e]=VSA(kh,e*16+tx);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[g][i][e],false);
            }
            if(g+1<G) __syncthreads();   // барьер только МЕЖДУ головами; после последней — верхний sync тайла
        }
    }

    // ---- эпилог: O = acco/l, LSE ----
    #pragma unroll
    for(int g=0; g<G; g++){
        const int h=h0+g;
        T* Ob=(T*)p.O+orow0+(long long)h*p.sO.h;
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ) continue;
            // Строка, у которой ВСЕ ключи замаскированы (causal при SQ>SK, узкое окно, чанк,
            // seqused_k): у неё rm так и остался -1e30, и exp2(-1e30 -(-1e30)) даёт 1, а не 0 —
            // rl выходит ненулевым, и acco копит сумму V. Ловим по несдвинутому максимуму
            // и отдаём нули с LSE=-inf, как flash-attn. Проверка в эпилоге, вне горячего цикла.
            const bool empty = rm[g][i] <= -1e29f;
            float inv = (rl[g][i]>0.0f && !empty)? 1.0f/rl[g][i] : 0.0f;
            #pragma unroll
            for(int e=0;e<DPT;e++){ int d=e*16+tx; Ob[(long long)qq*p.sO.s+d]=(T)(acco[g][i][e]*inv); }
            if(p.LSE && tx==0){ float lse = (rl[g][i]>0.0f && !empty)? (rm[g][i]*0.693147181f + logf(rl[g][i])) : -INFINITY;
                p.LSE[(size_t)h*SQ+qq]=lse; }
        }
    }
    #undef QSA
    #undef KSA
    #undef VSA
    #undef PSA
}

// ======================= fp32 корректный путь (fdot2 нет) =======================
// Тайлинг тот же, MAC через fmaf по float LDS. Только для точности 1e-5; не перф-цель.
template<int HD,int BR,int BC,bool CAUSAL>
__global__ __launch_bounds__(256)
void fa_forward_fp32_k(FwdParams p)
{
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z, h=blockIdx.y, kvh=h/p.gqa;
    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR; if(qb>=SQ) return;
    const int delta=SK-SQ; const float scale2=p.scale*1.442695041f;
    __shared__ float Qs[BR*HD], Ks[BC*HD], Vs[BC*HD]; __shared__ float Ps[BC*BR];
    const float* Qg=(const float*)p.Q+qrow0+(long long)h*p.sQ.h;
    for(int e=t;e<BR*HD;e+=256){ int q=e/HD,d=e%HD; int sq=qb+q; Qs[q*HD+d]=(sq<SQ)?Qg[(long long)sq*p.sQ.s+d]:0.0f; }
    float acco[QPT][DPT], rm[QPT], rl[QPT];
    #pragma unroll
    for(int i=0;i<QPT;i++){ rm[i]=-1e30f; rl[i]=0.0f; for(int e=0;e<DPT;e++) acco[i][e]=0.0f; }
    __syncthreads();
    int kmax = CAUSAL? (( (qb+BR-1)+delta < SK-1)? (qb+BR-1)+delta : SK-1) : SK-1;
    for(int kb=0;kb<=kmax;kb+=BC){
        const bool diag=CAUSAL&&((kb+BC-1)>=qb+delta);
        __syncthreads();
        const float* Kg=(const float*)p.K+krow0+(long long)kvh*p.sK.h;
        const float* Vg=(const float*)p.V+krow0+(long long)kvh*p.sV.h;
        for(int e=t;e<BC*HD;e+=256){ int kc=e/HD,d=e%HD; int sk=kb+kc;
            Ks[kc*HD+d]=(sk<SK)?Kg[(long long)sk*p.sK.s+d]:0.0f; Vs[kc*HD+d]=(sk<SK)?Vg[(long long)sk*p.sV.s+d]:0.0f; }
        __syncthreads();
        float accs[QPT][KPT];
        #pragma unroll
        for(int i=0;i<QPT;i++) for(int j=0;j<KPT;j++) accs[i][j]=0.0f;
        for(int d=0;d<HD;d++){ float qv[QPT],kv[KPT];
            #pragma unroll
            for(int i=0;i<QPT;i++) qv[i]=Qs[(ty*QPT+i)*HD+d];
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=Ks[(tx*KPT+j)*HD+d];
            #pragma unroll
            for(int i=0;i<QPT;i++) for(int j=0;j<KPT;j++) accs[i][j]=fmaf(qv[i],kv[j],accs[i][j]); }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; float sj[KPT],lmax=-1e30f;
            float aslope=p.alibi_slopes?p.alibi_slopes[h]:0.0f;
            #pragma unroll
            for(int j=0;j<KPT;j++){ int key=kb+tx*KPT+j; float raw=accs[i][j]; float sv;
                if(p.softcap>0.0f){ float x=raw*p.scale; sv=p.softcap*tanhf(x/p.softcap)*1.442695041f; } else sv=raw*scale2;
                if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*1.442695041f;
                bool masked=(key>=SK); if(CAUSAL&&diag&&key>qq+delta) masked=true;
                if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
                if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
                if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0;
                    int cs=(pp/p.attn_chunk)*p.attn_chunk; if(key<cs||key>=cs+p.attn_chunk) masked=true; }
                if(masked) sv=-1e30f; sj[j]=sv; lmax=fmaxf(lmax,sv); }
            lmax=fmaxf(lmax,__shfl_xor(lmax,1,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,2,16));
            lmax=fmaxf(lmax,__shfl_xor(lmax,4,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,8,16));
            float mnew=fmaxf(rm[i],lmax); float corr=(mnew>rm[i])?__builtin_amdgcn_exp2f(rm[i]-mnew):1.0f; float lsum=0;
            #pragma unroll
            for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
            lsum+=__shfl_xor(lsum,1,16); lsum+=__shfl_xor(lsum,2,16); lsum+=__shfl_xor(lsum,4,16); lsum+=__shfl_xor(lsum,8,16);
            rl[i]=rl[i]*corr+lsum; rm[i]=mnew;
            if(corr!=1.0f){ for(int e=0;e<DPT;e++) acco[i][e]*=corr; }
            #pragma unroll
            for(int j=0;j<KPT;j++) Ps[(tx*KPT+j)*BR+ty*QPT+i]=sj[j]; }
        __syncthreads();
        for(int kc=0;kc<BC;kc++){ float pv[QPT];
            #pragma unroll
            for(int i=0;i<QPT;i++) pv[i]=Ps[kc*BR+ty*QPT+i];
            #pragma unroll
            for(int e=0;e<DPT;e++){ float v=Vs[kc*HD+tx*DPT+e];
                #pragma unroll
                for(int i=0;i<QPT;i++) acco[i][e]=fmaf(pv[i],v,acco[i][e]); } }
        __syncthreads();
    }
    float* Ob=(float*)p.O+orow0+(long long)h*p.sO.h;
    #pragma unroll
    for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ) continue;
        const bool empty = rm[i] <= -1e29f;             // все ключи строки замаскированы
        float inv=(rl[i]>0&&!empty)?1.0f/rl[i]:0.0f;
        #pragma unroll
        for(int e=0;e<DPT;e++) Ob[(long long)qq*p.sO.s+tx*DPT+e]=acco[i][e]*inv;
        if(p.LSE&&tx==0) p.LSE[(size_t)h*SQ+qq]= (rl[i]>0&&!empty)?(rm[i]*0.693147181f+logf(rl[i])):-INFINITY; }
}

// ======================= диспетчер запуска (host) =======================
enum FaDtype { FA_F16=0, FA_BF16=1, FA_F32=2 };

// Конфиг под gfx1030 (fp16 И bf16 — одинаковая LDS-раскладка, bf16 конвертится inline при загрузке):
//   HD<=128 -> BR=128, BC=64 (структура чемпиона: макс переиспользование K/V, потолок памяти).
//   HD=256  -> BR=64,  BC=32.
// CAUSAL и FEAT (softcap/alibi/window/dropout) — компайл-тайм ветви, чтобы плотный путь был без оверхеда.
template<class DT,int HD,int BR,int BC>
static void launch_cfg(FwdParams& p, hipStream_t st){
    dim3 grid((p.SQ+BR-1)/BR, p.NH, p.batch);
    bool feat = (p.softcap>0.0f)||(p.alibi_slopes!=nullptr)||(p.dropout_p>0.0f)
                ||(p.attn_chunk>0)||(p.win_left>=0)
                // правая граница 0 — это ровно causal, её несёт компайл-тайм ветка CAUSAL,
                // поэтому в «фичи» она не попадает и плотный путь не дешевеет зря
                ||(p.win_right>=0 && !(p.win_right==0 && p.causal));
    if(p.causal){ if(feat) fa_forward_k<DT,HD,BR,BC,1,true ,true ><<<grid,256,0,st>>>(p);
                  else      fa_forward_k<DT,HD,BR,BC,1,true ,false><<<grid,256,0,st>>>(p); }
    else        { if(feat) fa_forward_k<DT,HD,BR,BC,1,false,true ><<<grid,256,0,st>>>(p);
                  else      fa_forward_k<DT,HD,BR,BC,1,false,false><<<grid,256,0,st>>>(p); }
}

// публичная точка входа для torch-обёртки (второй агент)
extern "C" void fa_forward_launch(FwdParams p, int dtype, hipStream_t st){
    if(p.sc_q==0.0f) p.sc_q=1.0f; if(p.sc_k==0.0f) p.sc_k=1.0f; if(p.sc_v==0.0f) p.sc_v=1.0f;
    // Тайл выбирается по LDS: при BR=128,BC=64 пул занимает HD*512 байт (+16КБ под Ps, если
    // BR>HD) — до HD=128 это влезает в 64 КБ, дальше нет. Для HD>=112 берём BR=64,BC=32:
    // пул HD*256, Ps алиасит Ks (BR<=HD), влезает вплоть до HD=256.
    #define BYHD(DTt) do{ switch(p.HD){ \
        case 16: launch_cfg<DTt, 16,128,64>(p,st);break; case 32: launch_cfg<DTt, 32,128,64>(p,st);break; \
        case 48: launch_cfg<DTt, 48,128,64>(p,st);break; case 64: launch_cfg<DTt, 64,128,64>(p,st);break; \
        case 80: launch_cfg<DTt, 80,128,64>(p,st);break; case 96: launch_cfg<DTt, 96,128,64>(p,st);break; \
        case 128:launch_cfg<DTt,128,128,64>(p,st);break; \
        case 112:launch_cfg<DTt,112, 64,32>(p,st);break; case 160:launch_cfg<DTt,160, 64,32>(p,st);break; \
        case 192:launch_cfg<DTt,192, 64,32>(p,st);break; case 224:launch_cfg<DTt,224, 64,32>(p,st);break; \
        case 256:launch_cfg<DTt,256, 64,32>(p,st);break; \
        default: printf("fa_forward: head_dim=%d не инстанцирован\n",p.HD); abort(); } }while(0)
    if(dtype==FA_F16){ BYHD(DT_F16); }
    else if(dtype==FA_BF16){ BYHD(DT_BF16); }
    else { // fp32 (корректность 1e-5; тайлы малые чтобы влезть в LDS float)
        #define DISP32(HD) do{ if(p.causal) fa_forward_fp32_k<HD,16,16,true ><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); \
                               else          fa_forward_fp32_k<HD,16,16,false><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); }while(0)
        switch(p.HD){ case 16:DISP32(16);break; case 32:DISP32(32);break; case 48:DISP32(48);break;
            case 64:DISP32(64);break; case 80:DISP32(80);break; case 96:DISP32(96);break;
            case 112:DISP32(112);break; case 128:DISP32(128);break; case 160:DISP32(160);break;
            case 192:DISP32(192);break; case 224:DISP32(224);break; case 256:DISP32(256);break;
            default: printf("fa_forward fp32: head_dim=%d не инстанцирован\n",p.HD); abort(); }
        #undef DISP32
    }
    #undef BYHD
}

#ifdef FA_STANDALONE
#include "fa_forward_gate.inc"
#endif
