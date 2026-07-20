// fa_lab.hip — стенд вариантов ядра FA. Одна сборка, все варианты внутри как шаблонные параметры.
// Запуск печатает таблицу: корректность (fp64-оракул) + скорость, база всегда первой строкой.
// Откат варианта = убрать его из списка CFGS внизу, базовое ядро не трогается никогда.
//
// Варианты (выбраны по тому, что показал RGA на настоящем ISA, а не наугад):
//   PACKP — P пакуется парами ключей в half2 ДО записи в LDS. Поток владеет ключами 4tx..4tx+3,
//           то есть пары (4tx,4tx+1),(4tx+2,4tx+3) уже у него в регистрах. GEMM2 тогда читает
//           готовый dword вместо двух half + v_perm. Убирает 32 v_perm из горячего цикла.
//   DPP   — редукция softmax по 16 лейнам через DPP (чистый VALU) вместо __shfl_xor,
//           который компилируется в ds_bpermute_b32 (LDS-конвейер, длинная латентность).
//           Маски xor 1,2,7,15 образуют базис по GF(2) на 4 битах => полная редукция.
//   G     — GQA-переиспользование: воркгруп обслуживает G q-голов одной kv-головы,
//           глобальный трафик K/V делится на G. В боевом ядре сейчас захардкожено G=1.
//   BR/BC — размер тайлов. BR=64,BC=32 даёт LDS 32 КБ вместо 64 => вдвое больше волн на SIMD.
//
// Сборка: hipcc fa_lab.hip -o fa_lab.exe --offload-arch=gfx1030 -O3 -ffast-math -std=c++17 -Wno-unused-value
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>

#define HIP_CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(e),__LINE__);exit(1);}}while(0)
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

// ---- DPP-редукция по 16 лейнам ------------------------------------------------
// dpp_ctrl: quad_perm{1,0,3,2}=0xB1 (xor1), quad_perm{2,3,0,1}=0x4E (xor2),
//           row_half_mirror=0x141 (xor7), row_mirror=0x140 (xor15).
template<int CTRL>
__device__ __forceinline__ float dpp_move(float x){
    int i = __builtin_bit_cast(int, x);
    int r = __builtin_amdgcn_update_dpp(0, i, CTRL, 0xF, 0xF, false);
    return __builtin_bit_cast(float, r);
}
__device__ __forceinline__ float red_max_dpp(float x){
    x = fmaxf(x, dpp_move<0xB1>(x));
    x = fmaxf(x, dpp_move<0x4E>(x));
    x = fmaxf(x, dpp_move<0x141>(x));
    x = fmaxf(x, dpp_move<0x140>(x));
    return x;
}
__device__ __forceinline__ float red_sum_dpp(float x){
    x += dpp_move<0xB1>(x);
    x += dpp_move<0x4E>(x);
    x += dpp_move<0x141>(x);
    x += dpp_move<0x140>(x);
    return x;
}
__device__ __forceinline__ float red_max_shfl(float x){
    x=fmaxf(x,__shfl_xor(x,1,16)); x=fmaxf(x,__shfl_xor(x,2,16));
    x=fmaxf(x,__shfl_xor(x,4,16)); x=fmaxf(x,__shfl_xor(x,8,16)); return x;
}
__device__ __forceinline__ float red_sum_shfl(float x){
    x+=__shfl_xor(x,1,16); x+=__shfl_xor(x,2,16);
    x+=__shfl_xor(x,4,16); x+=__shfl_xor(x,8,16); return x;
}

struct LP { const _Float16 *Q,*K,*V; _Float16* O; int S,gqa,NH; float scale; };

// ================================ ядро ================================
template<int HD,int BR,int BC,int G,bool PACKP,bool DPP,bool VLATE=false,bool LDSPIPE=false,bool DINT=false,bool KINT=false,bool PT=false,bool PSW=false,bool KSW=false,bool VSW=false>
__global__ __launch_bounds__(256)
void fa_lab_k(LP p)
{
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
    constexpr float LOG2E=1.442695041f;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int S=p.S;
    const int hg=blockIdx.y, h0=hg*G, kvh=h0/p.gqa;
    const int qb=blockIdx.x*BR; if(qb>=S) return;
    const float scale2=p.scale*LOG2E;

    constexpr int QS_B=G*HH*BR*4, KS_B=HH*BC*4, VS_B=BCH*HD*4;
    __shared__ char pool[QS_B+KS_B+VS_B];
    half2_t* Qs=(half2_t*)pool;
    half2_t* Ks=(half2_t*)(pool+QS_B);
    half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
    _Float16* Ps=(_Float16*)Ks;          // P алиасит K (K мёртв к моменту записи P)
    half2_t*  Pp=(half2_t*)Ks;           // упакованный вид того же места
    #define QSA(g,dh,q) Qs[((g)*HH+(dh))*BR+(q)]
    #define KSA(dh,kc)  (KSW? Ks[(dh)*BC+((kc)^((dh)&31))] : Ks[(dh)*BC+(kc)])
    #define VSA(kh,d)   (VSW? Vs[(kh)*HD+((d)^(((d)>>4)&7))] : Vs[(kh)*HD+(d)])
    #define PSA(kc,q)   Ps[(kc)*BR+(q)]
    #define PPA(kh,q)   (PT? Pp[(q)*(BC/2)+(kh)] : (PSW? Pp[(kh)*BR+((q)^((kh)&7))] : Pp[(kh)*BR+(q)]))

    // ---- Q -> LDS (резидент, один раз) ----
    #pragma unroll
    for(int g=0; g<G; g++){
        const _Float16* Qb=p.Q+(long long)(h0+g)*S*HD;
        for(int e=t; e<BR*(HD/8); e+=256){
            int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q;
            half8_t a = sq<S ? *(const half8_t*)(Qb+(long long)sq*HD+dg*8) : half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<4;m++) QSA(g,dg*4+m,q)=half2_t{a[2*m],a[2*m+1]};
        }
    }

    float acco[G][QPT][DPT], rm[G][QPT], rl[G][QPT];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<QPT;i++){ rm[g][i]=-1e30f; rl[g][i]=0.f;
            #pragma unroll
            for(int e=0;e<DPT;e++) acco[g][i][e]=0.f; }
    __syncthreads();

    const int kmax = (qb+BR-1 < S-1)? qb+BR-1 : S-1;   // causal, SQ==SK

    for(int kb=0; kb<=kmax; kb+=BC){
        const bool diag=((kb+BC-1)>=qb);
        __syncthreads();
        const _Float16* Kb=p.K+(long long)kvh*S*HD;
        const _Float16* Vb=p.V+(long long)kvh*S*HD;
        constexpr int NK=(BC*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0;it<NK;it++){ int e=t+it*256; if(e<BC*(HD/8)){
            int dg=e%(HD/8), kc=e/(HD/8); int sk=kb+kc;
            half8_t a = sk<S ? *(const half8_t*)(Kb+(long long)sk*HD+dg*8) : half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<4;m++) KSA(dg*4+m,kc)=half2_t{a[2*m],a[2*m+1]}; } }
        constexpr int NV=(BCH*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0;it<NV;it++){ int e=t+it*256; if(e<BCH*(HD/8)){
            int dg=e%(HD/8), kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1,d0=dg*8;
            half8_t a = s0<S ? *(const half8_t*)(Vb+(long long)s0*HD+d0) : half8_t{0,0,0,0,0,0,0,0};
            half8_t b = s1<S ? *(const half8_t*)(Vb+(long long)s1*HD+d0) : half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<8;m++) VSA(kh,d0+m)=half2_t{a[m],b[m]}; } }
        __syncthreads();

        // ---------------- GEMM1: S = Q Kᵀ (K жив, все G голов) ----------------
        float accs[G][QPT][KPT];
        #pragma unroll
        for(int g=0;g<G;g++)
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) accs[g][i][j]=0.f;
        if constexpr(LDSPIPE){
            // Конвейер операндов: чтения шага dh+1 выпущены ДО счёта шага dh, поэтому их
            // латентность (а замер показал, что GEMM стоит вдвое дольше своего счёта именно
            // из-за неё, не из-за полосы) уходит под fdot2 текущего шага.
            half2_t kc[KPT], kn[KPT], qc[G][QPT], qn[G][QPT];
            #pragma unroll
            for(int j=0;j<KPT;j++) kc[j]=KSA(0, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
            #pragma unroll
            for(int g=0;g<G;g++)
                #pragma unroll
                for(int i=0;i<QPT;i++) qc[g][i]=QSA(g,0,ty*QPT+i);
            #pragma unroll 2
            for(int dh=0; dh<HH; dh++){
                if(dh+1<HH){
                    #pragma unroll
                    for(int j=0;j<KPT;j++) kn[j]=KSA(dh+1, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
                    #pragma unroll
                    for(int g=0;g<G;g++)
                        #pragma unroll
                        for(int i=0;i<QPT;i++) qn[g][i]=QSA(g,dh+1,ty*QPT+i);
                }
                #pragma unroll
                for(int g=0;g<G;g++)
                    #pragma unroll
                    for(int i=0;i<QPT;i++)
                        #pragma unroll
                        for(int j=0;j<KPT;j++)
                            accs[g][i][j]=__builtin_amdgcn_fdot2(qc[g][i],kc[j],accs[g][i][j],false);
                #pragma unroll
                for(int j=0;j<KPT;j++) kc[j]=kn[j];
                #pragma unroll
                for(int g=0;g<G;g++)
                    #pragma unroll
                    for(int i=0;i<QPT;i++) qc[g][i]=qn[g][i];
            }
        } else {
        #pragma unroll 4
        for(int dh=0; dh<HH; dh++){
            half2_t kv[KPT];
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=KSA(dh, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
            #pragma unroll
            for(int g=0;g<G;g++){
                half2_t qv[QPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) qv[i]=QSA(g,dh,ty*QPT+i);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int j=0;j<KPT;j++)
                        accs[g][i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[g][i][j],false);
            }
        } }
        __syncthreads();   // K мёртв -> место под P

        #pragma unroll
        for(int g=0; g<G; g++){
            // ---------------- softmax ----------------
            #pragma unroll
            for(int i=0;i<QPT;i++){
                const int qq=qb+ty*QPT+i;
                float sj[KPT], lmax=-1e30f;
                #pragma unroll
                for(int j=0;j<KPT;j++){
                    const int key=kb+(KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
                    float sv=accs[g][i][j]*scale2;
                    if(diag&&key>qq) sv=-1e30f;
                    sj[j]=sv; lmax=fmaxf(lmax,sv);
                }
                lmax = DPP? red_max_dpp(lmax) : red_max_shfl(lmax);
                float mnew=fmaxf(rm[g][i],lmax);
                float corr=(mnew>rm[g][i])?__builtin_amdgcn_exp2f(rm[g][i]-mnew):1.0f;
                float lsum=0.f;
                #pragma unroll
                for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
                lsum = DPP? red_sum_dpp(lsum) : red_sum_shfl(lsum);
                rl[g][i]=rl[g][i]*corr+lsum; rm[g][i]=mnew;
                if(corr!=1.0f){
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]*=corr;
                }
                // ---- запись P ----
                if constexpr(PACKP){
                    // ключи потока 4tx..4tx+3 идут подряд => пары пакуем сами, GEMM2 читает dword
                    #pragma unroll
                    for(int j=0;j<KPT;j+=2)
                        PPA(KINT? ((j>>1)*16+tx) : (tx*(KPT/2)+j/2), ty*QPT+i) = half2_t{(_Float16)sj[j],(_Float16)sj[j+1]};
                } else {
                    #pragma unroll
                    for(int j=0;j<KPT;j++) PSA(tx*KPT+j, ty*QPT+i)=(_Float16)sj[j];
                }
            }
            __syncthreads();

            // ---------------- GEMM2: O += P V ----------------
            if constexpr(LDSPIPE && PACKP){
                half2_t pc[QPT],pn[QPT],vc[DPT],vn[DPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) pc[i]=PPA(0,ty*QPT+i);
                #pragma unroll
                for(int e=0;e<DPT;e++) vc[e]=VSA(0, DINT? (e*16+tx) : (tx*DPT+e));
                #pragma unroll 2
                for(int kh=0; kh<BCH; kh++){
                    if(kh+1<BCH){
                        #pragma unroll
                        for(int i=0;i<QPT;i++) pn[i]=PPA(kh+1,ty*QPT+i);
                        #pragma unroll
                        for(int e=0;e<DPT;e++) vn[e]=VSA(kh+1, DINT? (e*16+tx) : (tx*DPT+e));
                    }
                    #pragma unroll
                    for(int i=0;i<QPT;i++)
                        #pragma unroll
                        for(int e=0;e<DPT;e++)
                            acco[g][i][e]=__builtin_amdgcn_fdot2(pc[i],vc[e],acco[g][i][e],false);
                    #pragma unroll
                    for(int i=0;i<QPT;i++) pc[i]=pn[i];
                    #pragma unroll
                    for(int e=0;e<DPT;e++) vc[e]=vn[e];
                }
            } else {
            #pragma unroll 4
            for(int kh=0; kh<BCH; kh++){
                half2_t pv[QPT], vv[DPT];
                if constexpr(PACKP){
                    #pragma unroll
                    for(int i=0;i<QPT;i++) pv[i]=PPA(kh,ty*QPT+i);            // один b32, без v_perm
                } else {
                    #pragma unroll
                    for(int i=0;i<QPT;i++) pv[i]=half2_t{ PSA(2*kh,ty*QPT+i), PSA(2*kh+1,ty*QPT+i) };
                }
                #pragma unroll
                for(int e=0;e<DPT;e++) vv[e]=VSA(kh, DINT? (e*16+tx) : (tx*DPT+e));
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int e=0;e<DPT;e++)
                        acco[g][i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[g][i][e],false);
            } }
            if(g+1<G) __syncthreads();
        }
    }

    // ---------------- эпилог ----------------
    #pragma unroll
    for(int g=0; g<G; g++){
        _Float16* Ob=p.O+(long long)(h0+g)*S*HD;
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=S) continue;
            float inv = rl[g][i]>0.f? 1.f/rl[g][i] : 0.f;
            #pragma unroll
            for(int e=0;e<DPT;e++) Ob[(long long)qq*HD+(DINT? (e*16+tx) : (tx*DPT+e))]=(_Float16)(acco[g][i][e]*inv);
        }
    }
    #undef QSA
    #undef KSA
    #undef VSA
    #undef PSA
    #undef PPA
}

// ======================= конвейеризованное ядро =======================
// Крупная переработка, а не правка: глобальные чтения K/V следующего тайла запускаются СРАЗУ
// после коммита текущего и летят все время GEMM1+softmax+GEMM2 (~26 из 32 мс). Пузырь
// «барьер-загрузка-барьер» (5.05 мс, 39% всего перерасхода над потолком MAC) исчезает.
// Цена — 32 VGPR под буфер. База 198 VGPR, лимит 256, поэтому шансы есть; если спиллит,
// вариант KONLY держит в регистрах только K (16 VGPR).
// packP и DPP-редукция встроены (обе уже доказаны замером).
template<int HD,int BR,int BC,int G,bool VPREF,bool PACKP=true,bool LDSPIPE=false,bool DINT=false,bool KINT=false,bool PT=false,bool PSW=false,bool KSW=false,bool VSW=false>
__global__ __launch_bounds__(256)
void fa_pipe_k(LP p)
{
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
    constexpr int NK=(BC*(HD/8)+255)/256, NV=(BCH*(HD/8)+255)/256;
    constexpr float LOG2E=1.442695041f;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int S=p.S;
    const int hg=blockIdx.y, h0=hg*G, kvh=h0/p.gqa;
    const int qb=blockIdx.x*BR; if(qb>=S) return;
    const float scale2=p.scale*LOG2E;

    constexpr int QS_B=G*HH*BR*4, KS_B=HH*BC*4, VS_B=BCH*HD*4;
    __shared__ char pool[QS_B+KS_B+VS_B];
    half2_t* Qs=(half2_t*)pool;
    half2_t* Ks=(half2_t*)(pool+QS_B);
    half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
    half2_t* Pp=(half2_t*)Ks;
    #define QSA(g,dh,q) Qs[((g)*HH+(dh))*BR+(q)]
    #define KSA(dh,kc)  (KSW? Ks[(dh)*BC+((kc)^((dh)&31))] : Ks[(dh)*BC+(kc)])
    #define VSA(kh,d)   (VSW? Vs[(kh)*HD+((d)^(((d)>>4)&7))] : Vs[(kh)*HD+(d)])
    #define PPA(kh,q)   Pp[(kh)*BR+(q)]

    #pragma unroll
    for(int g=0; g<G; g++){
        const _Float16* Qb=p.Q+(long long)(h0+g)*S*HD;
        for(int e=t; e<BR*(HD/8); e+=256){
            int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q;
            half8_t a = sq<S ? *(const half8_t*)(Qb+(long long)sq*HD+dg*8) : half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<4;m++) QSA(g,dg*4+m,q)=half2_t{a[2*m],a[2*m+1]};
        }
    }
    float acco[G][QPT][DPT], rm[G][QPT], rl[G][QPT];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<QPT;i++){ rm[g][i]=-1e30f; rl[g][i]=0.f;
            #pragma unroll
            for(int e=0;e<DPT;e++) acco[g][i][e]=0.f; }

    const _Float16* Kb=p.K+(long long)kvh*S*HD;
    const _Float16* Vb=p.V+(long long)kvh*S*HD;
    const int kmax = (qb+BR-1 < S-1)? qb+BR-1 : S-1;

    half8_t rgK[NK];
    half8_t rgVa[VPREF?NV:1], rgVb[VPREF?NV:1];

    auto issue=[&](int kb){                       // только запуск чтений, без записи в LDS
        #pragma unroll
        for(int it=0;it<NK;it++){ int e=t+it*256; if(e<BC*(HD/8)){
            int dg=e%(HD/8), kc=e/(HD/8); int sk=kb+kc;
            rgK[it] = sk<S ? *(const half8_t*)(Kb+(long long)sk*HD+dg*8) : half8_t{0,0,0,0,0,0,0,0}; } }
        if constexpr(VPREF){
            #pragma unroll
            for(int it=0;it<NV;it++){ int e=t+it*256; if(e<BCH*(HD/8)){
                int dg=e%(HD/8), kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1,d0=dg*8;
                rgVa[it] = s0<S ? *(const half8_t*)(Vb+(long long)s0*HD+d0) : half8_t{0,0,0,0,0,0,0,0};
                rgVb[it] = s1<S ? *(const half8_t*)(Vb+(long long)s1*HD+d0) : half8_t{0,0,0,0,0,0,0,0}; } }
        }
    };
    auto commit=[&](int kb){
        #pragma unroll
        for(int it=0;it<NK;it++){ int e=t+it*256; if(e<BC*(HD/8)){
            int dg=e%(HD/8), kc=e/(HD/8); half8_t a=rgK[it];
            #pragma unroll
            for(int m=0;m<4;m++) KSA(dg*4+m,kc)=half2_t{a[2*m],a[2*m+1]}; } }
        #pragma unroll
        for(int it=0;it<NV;it++){ int e=t+it*256; if(e<BCH*(HD/8)){
            int dg=e%(HD/8), kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1,d0=dg*8;
            half8_t a,b;
            if constexpr(VPREF){ a=rgVa[it]; b=rgVb[it]; }
            else {
                a = s0<S ? *(const half8_t*)(Vb+(long long)s0*HD+d0) : half8_t{0,0,0,0,0,0,0,0};
                b = s1<S ? *(const half8_t*)(Vb+(long long)s1*HD+d0) : half8_t{0,0,0,0,0,0,0,0};
            }
            #pragma unroll
            for(int m=0;m<8;m++) VSA(kh,d0+m)=half2_t{a[m],b[m]}; } }
    };

    issue(0);
    __syncthreads();

    for(int kb=0; kb<=kmax; kb+=BC){
        const bool diag=((kb+BC-1)>=qb);
        commit(kb);
        __syncthreads();
        if(kb+BC<=kmax) issue(kb+BC);      // летит весь остаток итерации

        float accs[G][QPT][KPT];
        #pragma unroll
        for(int g=0;g<G;g++)
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) accs[g][i][j]=0.f;
        if constexpr(LDSPIPE){
            // Конвейер операндов: чтения шага dh+1 выпущены ДО счёта шага dh, поэтому их
            // латентность (а замер показал, что GEMM стоит вдвое дольше своего счёта именно
            // из-за неё, не из-за полосы) уходит под fdot2 текущего шага.
            half2_t kc[KPT], kn[KPT], qc[G][QPT], qn[G][QPT];
            #pragma unroll
            for(int j=0;j<KPT;j++) kc[j]=KSA(0, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
            #pragma unroll
            for(int g=0;g<G;g++)
                #pragma unroll
                for(int i=0;i<QPT;i++) qc[g][i]=QSA(g,0,ty*QPT+i);
            #pragma unroll 2
            for(int dh=0; dh<HH; dh++){
                if(dh+1<HH){
                    #pragma unroll
                    for(int j=0;j<KPT;j++) kn[j]=KSA(dh+1, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
                    #pragma unroll
                    for(int g=0;g<G;g++)
                        #pragma unroll
                        for(int i=0;i<QPT;i++) qn[g][i]=QSA(g,dh+1,ty*QPT+i);
                }
                #pragma unroll
                for(int g=0;g<G;g++)
                    #pragma unroll
                    for(int i=0;i<QPT;i++)
                        #pragma unroll
                        for(int j=0;j<KPT;j++)
                            accs[g][i][j]=__builtin_amdgcn_fdot2(qc[g][i],kc[j],accs[g][i][j],false);
                #pragma unroll
                for(int j=0;j<KPT;j++) kc[j]=kn[j];
                #pragma unroll
                for(int g=0;g<G;g++)
                    #pragma unroll
                    for(int i=0;i<QPT;i++) qc[g][i]=qn[g][i];
            }
        } else {
        #pragma unroll 4
        for(int dh=0; dh<HH; dh++){
            half2_t kv[KPT];
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=KSA(dh, KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
            #pragma unroll
            for(int g=0;g<G;g++){
                half2_t qv[QPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) qv[i]=QSA(g,dh,ty*QPT+i);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int j=0;j<KPT;j++)
                        accs[g][i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[g][i][j],false);
            }
        } }
        __syncthreads();

        #pragma unroll
        for(int g=0; g<G; g++){
            #pragma unroll
            for(int i=0;i<QPT;i++){
                const int qq=qb+ty*QPT+i;
                float sj[KPT], lmax=-1e30f;
                #pragma unroll
                for(int j=0;j<KPT;j++){
                    const int key=kb+(KINT? ((j>>1)*32+2*tx+(j&1)) : (tx*KPT+j));
                    float sv=accs[g][i][j]*scale2;
                    if(diag&&key>qq) sv=-1e30f;
                    sj[j]=sv; lmax=fmaxf(lmax,sv);
                }
                lmax=red_max_dpp(lmax);
                float mnew=fmaxf(rm[g][i],lmax);
                float corr=(mnew>rm[g][i])?__builtin_amdgcn_exp2f(rm[g][i]-mnew):1.0f;
                float lsum=0.f;
                #pragma unroll
                for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
                lsum=red_sum_dpp(lsum);
                rl[g][i]=rl[g][i]*corr+lsum; rm[g][i]=mnew;
                if(corr!=1.0f){
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]*=corr;
                }
                #pragma unroll
                for(int j=0;j<KPT;j+=2)
                    PPA(tx*(KPT/2)+j/2, ty*QPT+i)=half2_t{(_Float16)sj[j],(_Float16)sj[j+1]};
            }
            __syncthreads();
            #pragma unroll 4
            for(int kh=0; kh<BCH; kh++){
                half2_t pv[QPT], vv[DPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) pv[i]=PPA(kh,ty*QPT+i);
                #pragma unroll
                for(int e=0;e<DPT;e++) vv[e]=VSA(kh, DINT? (e*16+tx) : (tx*DPT+e));
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int e=0;e<DPT;e++)
                        acco[g][i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[g][i][e],false);
            }
            __syncthreads();     // Vs/Ps свободны для коммита следующего тайла
        }
    }

    #pragma unroll
    for(int g=0; g<G; g++){
        _Float16* Ob=p.O+(long long)(h0+g)*S*HD;
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=S) continue;
            float inv = rl[g][i]>0.f? 1.f/rl[g][i] : 0.f;
            #pragma unroll
            for(int e=0;e<DPT;e++) Ob[(long long)qq*HD+tx*DPT+e]=(_Float16)(acco[g][i][e]*inv);
        }
    }
    #undef QSA
    #undef KSA
    #undef VSA
    #undef PPA
}

// ============================ обвязка стенда ============================
static const int HD=128;

struct Cfg {
    const char* name;
    int BR,BC,G; bool packp,dpp;
    void (*launch)(LP&,dim3&);
    double (*lds)();
};

template<int BR,int BC,int G,bool PACKP,bool DPP>
static void launch_impl(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,PACKP,DPP>),grid,dim3(256),0,0,p);
}
template<int BR,int BC,int G>
static void launch_all(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,true,false,false,true,true>),grid,dim3(256),0,0,p);
}
#define CFGALL(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_all<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_ksw(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,false,false,false,true>),grid,dim3(256),0,0,p);
}
#define CFGKS(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_ksw<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_kskv(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,true,false,false,true>),grid,dim3(256),0,0,p);
}
#define CFGKSK(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_kskv<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_psw(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,false,false,true>),grid,dim3(256),0,0,p);
}
#define CFGSW(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_psw<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_pswk(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,true,false,true>),grid,dim3(256),0,0,p);
}
#define CFGSWK(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_pswk<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_pt(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,false,true>),grid,dim3(256),0,0,p);
}
#define CFGPT(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_pt<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_ptk(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,true,true>),grid,dim3(256),0,0,p);
}
#define CFGPTK(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_ptk<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_dint(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,false>),grid,dim3(256),0,0,p);
}
#define CFGD(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_dint<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_dkint(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,true,true>),grid,dim3(256),0,0,p);
}
#define CFGDK(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_dkint<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static void launch_kint(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,true,true,false,false,false,true>),grid,dim3(256),0,0,p);
}
#define CFGK(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_kint<BR,BC,G>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G,bool PACKP,bool DPP>
static void launch_pipe_lds(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_lab_k<HD,BR,BC,G,PACKP,DPP,false,true>),grid,dim3(256),0,0,p);
}
#define CFGL(nm,BR,BC,G) { nm,BR,BC,G,true,true, &launch_pipe_lds<BR,BC,G,true,true>, &lds_impl<BR,BC,G> }
template<int BR,int BC,int G>
static double lds_impl(){ return (G*(HD/2)*BR*4 + (HD/2)*BC*4 + (BC/2)*HD*4)/1024.0; }

#define CFG(nm,BR,BC,G,PK,DP) { nm,BR,BC,G,PK,DP, &launch_impl<BR,BC,G,PK,DP>, &lds_impl<BR,BC,G> }

template<int BR,int BC,int G,bool VPREF>
static void launch_pipe(LP& p, dim3& grid){
    hipLaunchKernelGGL((fa_pipe_k<HD,BR,BC,G,VPREF>),grid,dim3(256),0,0,p);
}
#define CFGP(nm,BR,BC,G,VP) { nm,BR,BC,G,true,true, &launch_pipe<BR,BC,G,VP>, &lds_impl<BR,BC,G> }

// ---- fp64-оракул ----
static uint32_t xrng(uint32_t& s){ s=s*1664525u+1013904223u; return s; }
static float frand(uint32_t& s){ return ((xrng(s)>>16)&4095)/4096.0f-0.5f; }

static void ref_attn(const std::vector<float>& Q,const std::vector<float>& K,const std::vector<float>& V,
                     std::vector<float>& O,int n,int H,int gqa,float scale){
    std::vector<double> pr(n);
    for(int h=0;h<H;h++){ int kvh=h/gqa;
        for(int i=0;i<n;i++){
            double mx=-1e300;
            for(int j=0;j<n;j++){
                if(j>i){ pr[j]=-1e300; continue; }
                double s=0;
                for(int d=0;d<HD;d++) s+=(double)Q[((size_t)h*n+i)*HD+d]*(double)K[((size_t)kvh*n+j)*HD+d];
                s*=scale; pr[j]=s; if(s>mx)mx=s;
            }
            double sum=0;
            for(int j=0;j<n;j++){ if(pr[j]<=-1e299){pr[j]=0;continue;} pr[j]=exp(pr[j]-mx); sum+=pr[j]; }
            for(int d=0;d<HD;d++){ double acc=0;
                for(int j=0;j<n;j++) acc+=pr[j]*(double)V[((size_t)kvh*n+j)*HD+d];
                O[((size_t)h*n+i)*HD+d]=(float)(acc/sum);
            }
        }
    }
}
static void* to_dev(const std::vector<float>& v){
    std::vector<_Float16> h(v.size()); for(size_t i=0;i<v.size();i++) h[i]=(_Float16)v[i];
    void* d; HIP_CHECK(hipMalloc(&d,v.size()*2));
    HIP_CHECK(hipMemcpy(d,h.data(),v.size()*2,hipMemcpyHostToDevice)); return d;
}

static bool check(const Cfg& c,int n,int H,int gqa,double* err_out){
    float scale=1.f/sqrtf((float)HD); uint32_t s=4242u;
    int NKV=H/gqa;
    std::vector<float> Q((size_t)H*n*HD),K((size_t)NKV*n*HD),V(K.size()),O(Q.size(),0.f),Oref(Q.size());
    for(auto& x:Q)x=frand(s); for(auto& x:K)x=frand(s); for(auto& x:V)x=frand(s);
    for(auto& x:Q)x=(float)(_Float16)x; for(auto& x:K)x=(float)(_Float16)x; for(auto& x:V)x=(float)(_Float16)x;
    ref_attn(Q,K,V,Oref,n,H,gqa,scale);
    void *dQ=to_dev(Q),*dK=to_dev(K),*dV=to_dev(V),*dO=to_dev(O);
    LP p{(const _Float16*)dQ,(const _Float16*)dK,(const _Float16*)dV,(_Float16*)dO,n,gqa,H,scale};
    dim3 grid((n+c.BR-1)/c.BR, H/c.G, 1);
    c.launch(p,grid); HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16> oh(O.size());
    HIP_CHECK(hipMemcpy(oh.data(),dO,O.size()*2,hipMemcpyDeviceToHost));
    double maxe=0,se=0,sr=0;
    for(size_t i=0;i<O.size();i++){ double g=(double)(float)oh[i],r=Oref[i],d=fabs(g-r);
        if(d>maxe)maxe=d; se+=d*d; sr+=r*r; }
    *err_out=maxe;
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return maxe<2e-2 && sqrt(se/(sr>0?sr:1))<5e-3;
}

static float bench(const Cfg& c,int n,int H,int gqa){
    int NKV=H/gqa;
    size_t nQ=(size_t)H*n*HD, nK=(size_t)NKV*n*HD;
    std::vector<_Float16> hq(nQ,(_Float16)0.02f), hk(nK,(_Float16)0.02f);
    void *dQ,*dK,*dV,*dO;
    HIP_CHECK(hipMalloc(&dQ,nQ*2)); HIP_CHECK(hipMalloc(&dO,nQ*2));
    HIP_CHECK(hipMalloc(&dK,nK*2)); HIP_CHECK(hipMalloc(&dV,nK*2));
    HIP_CHECK(hipMemcpy(dQ,hq.data(),nQ*2,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK,hk.data(),nK*2,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV,hk.data(),nK*2,hipMemcpyHostToDevice));
    LP p{(const _Float16*)dQ,(const _Float16*)dK,(const _Float16*)dV,(_Float16*)dO,n,gqa,H,1.f/sqrtf((float)HD)};
    dim3 grid((n+c.BR-1)/c.BR, H/c.G, 1);
    c.launch(p,grid); HIP_CHECK(hipDeviceSynchronize());
    if(hipGetLastError()!=hipSuccess){ hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO); return -1.f; }
    for(int i=0;i<3;i++) c.launch(p,grid);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    int it = n>=8192?10:20;
    hipEventRecord(t0); for(int i=0;i<it;i++) c.launch(p,grid); hipEventRecord(t1); hipEventSynchronize(t1);
    float ms=0; hipEventElapsedTime(&ms,t0,t1); ms/=it;
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return ms;
}

int main(int argc,char** argv){
    int S   = argc>1? atoi(argv[1]) : 8192;
    int gqa = argc>2? atoi(argv[2]) : 4;
    const int NH=32;
    hipDeviceProp_t pr; HIP_CHECK(hipGetDeviceProperties(&pr,0));
    printf("GPU %s | S=%d NH=%d GQA=%d HD=%d causal fp16\n\n",pr.gcnArchName,S,NH,gqa,HD);

    // ---- список вариантов. Убрать строку = откатить вариант. База всегда первая. ----
    static const Cfg CFGS[] = {
        CFG("packP+dpp (в бою)",       128,64,1,true ,true ),
        CFGD("V без конфликта банков",  128,64,1),
        CFGK("K без конфликта",         128,64,1),
        CFGDK("K и V без конфликтов",   128,64,1),
        CFGKS("V + свизл записи K",     128,64,1),
        CFGKSK("K + V + свизл записи K",128,64,1),
        CFGALL("всё + свизл записи V",   128,64,1),
        // ≤32 КБ LDS => ДВА воркгрупа на CU. Барьеры одного прячутся за счётом другого;
        // сейчас воркгруп один и на каждом из 4 барьеров тайла машина стоит целиком.
        CFG("64x32 (32КБ, 2 WG/CU)",    64,32,1,true ,true ),
        CFG("32x32 (24КБ, 2+ WG/CU)",   32,32,1,true ,true ),
        CFG("32x64 (32КБ, 2 WG/CU)",    32,64,1,true ,true ),
        CFG("64x64 (48КБ, 1 WG/CU)",    64,64,1,true ,true ),
        CFG("96x32 (40КБ, 1 WG/CU)",    96,32,1,true ,true ),
        CFG("128x32 (48КБ, 1 WG/CU)",  128,32,1,true ,true ),
        // конвейеризация: чтения следующего тайла летят под счётом текущего
        CFGP("конвейер K+V",           128,64,1,true ),
        // Проверено и отвергнуто замером (строки оставлены как история, включать нечего):
        //   мелкий тайл 64x32 ради занятости      -35%
        //   GQA2/GQA4 (BR делится на G)             0%   трафик не падает: воркгрупов вдвое больше
        //   конвейеризация чтений K/V               0%   упор в полосу Infinity Cache, не в латентность
        //   BR=160/192 при BC=32                   -13%  и P не влезает в алиас Ks при BR>HD
    };
    const int NC=sizeof(CFGS)/sizeof(CFGS[0]);

    double fl=2.0*(double)NH*(double)S*(double)S*HD;   // causal: 4*H*S*S*D/2
    float base_ms=-1;
    printf("  %-22s %5s %4s %6s  %9s %9s %8s  %s\n","вариант","тайл","G","LDS КБ","время","TFLOP/s","к базе","корректность");
    { char bar[97]; memset(bar,'-',96); bar[96]=0; printf("  %s\n",bar); }
    for(int i=0;i<NC;i++){
        const Cfg& c=CFGS[i];
        if(NH % c.G){ printf("  %-22s пропуск: NH не делится на G\n",c.name); continue; }
        double err=0;
        bool ok = check(c,256,(c.G>1? (c.G*4):4), (gqa<c.G? c.G:gqa), &err);
        float ms=bench(c,S,NH,gqa);
        if(ms<0){ printf("  %-22s ОШИБКА ЗАПУСКА\n",c.name); continue; }
        if(base_ms<0) base_ms=ms;
        char tile[16]; snprintf(tile,sizeof(tile),"%dx%d",c.BR,c.BC);
        printf("  %-22s %5s %4d %6.1f  %7.3f ms %8.1f  %+6.1f%%  %s (max|err| %.1e)\n",
               c.name,tile,c.G,c.lds(),ms,fl/(ms*1e-3)/1e12,
               100.0*(base_ms-ms)/base_ms, ok?"OK":"ПРОВАЛ",err);
    }
    return 0;
}
