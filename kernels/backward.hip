// backward.hip — FlashAttention-2 BACKWARD для gfx1030 (RX 6800 XT), dtype-generic (fp16/bf16/fp32).
// Схема FA-2 БЕЗ АТОМИКОВ => детерминизм by construction (побитово воспроизводимо, deterministic=True/False
// дают ОДИН результат — строгое усиление контракта flash-attn):
//   1) preprocess:  D_i = Σ_d O_id · dO_id            (fp32, форма [B,Hq,SQ])
//   2) dK/dV-ядро:  параллелизм по K-тайлам; тайл ключей суммирует по ВСЕМ запросам и всем
//                   q-головам GQA-группы. dK,dV в регистрах. Атомиков нет.
//   3) dQ-ядро:     параллелизм по Q-тайлам; dQ в регистрах. Атомиков нет.
// S×S НЕ материализуется — только тайл BQ×BK в LDS; P пересчитывается из сохранённого LSE.
// Минимум VRAM: сверх dQ/dK/dV только воркспейс D_i=[B,Hq,SQ] fp32 (крошечный).
//
// dtype: fp16 и bf16 — на fp16-скорости через v_dot2_f32_f16. bf16 — степенной-двойки масштаб
// на тензор (сдвиг экспоненты, pow2_scale_for) => bf16->fp16 БЕЗ ПОТЕРЬ => fdot2 => делёж на 2^shift
// (точно). fp32 — отдельный путь на fmaf (точность ~1e-6; не перф-цель, fdot2 только fp16).
//
// СОГЛАСОВАНО С FORWARD (fa_forward.hip): LSE натур. ед. [NH][SQ] на батч; dropout idx=((h*SQ+qq)*SK+kk)
// БЕЗ батча + fa_common::dropout_mul; маска — единый window (allowed [qq+delta-window+1..qq+delta]),
// causal bottom-right delta=SK-SQ; softcap/alibi ровно как во forward. Пересчёт P идёт в натуральных
// единицах => совпадает с forward-LSE бит-в-бит по формуле.
//
// Сборка гейта: hipcc backward.hip -o backward.exe -DFA_STANDALONE --offload-arch=gfx1030 -O3 -ffast-math
#include "fa_common.hpp"

struct Str { long long b, h, s; };      // страйды в элементах, dim contiguous

enum FaBwdDT { BW_F16 = 0, BW_BF16 = 1, BW_F32 = 2 };

template<int MODE> struct ElemT        { using type = _Float16; };
template<>         struct ElemT<BW_F32>{ using type = float; };

// точное 2^e для разумных e
__host__ __device__ __forceinline__ float exp2i(int e) {
    union { uint32_t u; float f; } v; int be = e + 127;
    if (be <= 0)   return 0.0f;
    if (be >= 255) return __builtin_huge_valf();
    v.u = (uint32_t)be << 23; return v.f;
}
__device__ __forceinline__ float bf16bits_to_f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; __builtin_memcpy(&f,&u,4); return f; }
__device__ __forceinline__ uint16_t f_to_bf16_rn(float f){ union{uint32_t u;float f;}c; c.f=f; uint32_t lsb=(c.u>>16)&1u; c.u+=0x7fffu+lsb; return (uint16_t)(c.u>>16); }

// чтение элемента как fp16-тайла (с bf16-масштабом scl); для fp32-режима — как float.
template<int MODE> __device__ __forceinline__ typename ElemT<MODE>::type
load_tile(const void* p, long long idx, float scl){
    if (MODE == BW_F16)  return ((const _Float16*)p)[idx];
    if (MODE == BW_BF16) return (_Float16)(bf16bits_to_f(((const uint16_t*)p)[idx]) * scl);
    return (typename ElemT<MODE>::type)(((const float*)p)[idx]);
}
// точное чтение в float (для D_i — без потери bf16-точности через fp16)
template<int MODE> __device__ __forceinline__ float load_f32(const void* p, long long idx){
    if (MODE == BW_F16)  return (float)((const _Float16*)p)[idx];
    if (MODE == BW_BF16) return bf16bits_to_f(((const uint16_t*)p)[idx]);
    return ((const float*)p)[idx];
}
template<int MODE> __device__ __forceinline__ void store_out(void* p, long long idx, float v){
    if (MODE == BW_F16)  { ((_Float16*)p)[idx] = (_Float16)v; return; }
    if (MODE == BW_BF16) { ((uint16_t*)p)[idx] = f_to_bf16_rn(v); return; }
    ((float*)p)[idx] = v;
}

struct BwdParams {
    const void *Q,*K,*V,*O,*dO;  void *dQ,*dK,*dV;
    const float *lse; float *dsum;
    Str qs,ks,vs,os,dos,dqs,dks,dvs;
    int B,Hq,Hk,SQ,SK,D,gqa;
    float scale, softcap;
    const float *alibi;          // [Hq] или nullptr
    // Окно в семантике flash-attn: разрешены ключи [qq+delta-win_left .. qq+delta+win_right],
    // отрицательное = границы нет. causal эквивалентен win_right==0 (оставлен отдельным полем
    // ради быстрых веток по тайлам).
    int causal, win_left, win_right;
    int dropout; unsigned long long drop_seed; float dropout_p, keep_scale;
    int shQ,shK,shV,shdO;        // bf16 pow2-сдвиги (0 для fp16/fp32)
    int probe;                   // 1 = зонд: пропустить второй GEMM (результат неверен, только замер)
};

// маска forward: masked если key>=SK || (causal&&key>qq+delta) || (window: вне [lo..qq+delta])
__device__ __forceinline__ bool fwd_ok(const BwdParams& pr, int qq, int kk){
    if (kk >= pr.SK || qq >= pr.SQ) return false;
    const int delta = pr.SK - pr.SQ, diag = qq + delta;
    if (pr.causal && kk > diag) return false;
    if (pr.win_left  >= 0 && kk < diag - pr.win_left ) return false;
    if (pr.win_right >= 0 && kk > diag + pr.win_right) return false;
    return true;
}

// ============================ preprocess: D_i ============================
template<int D, int MODE>
__global__ __launch_bounds__(256)
void fa_bwd_preprocess(BwdParams pr){
    const int b=blockIdx.z, h=blockIdx.y, qq=blockIdx.x*256+threadIdx.x;
    if (qq >= pr.SQ) return;
    const long long ob = (long long)b*pr.os.b + (long long)h*pr.os.h + (long long)qq*pr.os.s;
    const long long db = (long long)b*pr.dos.b + (long long)h*pr.dos.h + (long long)qq*pr.dos.s;
    float acc=0.f;
    for (int d=0; d<D; d++) acc += load_f32<MODE>(pr.O, ob+d) * load_f32<MODE>(pr.dO, db+d);
    pr.dsum[((long long)b*pr.Hq + h)*pr.SQ + qq] = acc;
}

// ---- посчитать P̃(=z·P) и dS в LDS для тайла BQ×BK ----
// OUT — раскладка выхода. Второй GEMM читает LDS парами, и от того, КАК тут разложено,
// зависит, будет он читать один dword или два u16 + v_perm (это и стоило backward-у 2.7 TFLOP/s):
//   OUT_PLAIN — как было: Pt[i*BK+j], dSt[i*BK+j] (fp32-путь, точность, не перф).
//   OUT_PAIR_J — dQ-ядро: dS2[i*(BK/2)+jh] = half2{dS[i][2jh],dS[i][2jh+1]}. P не нужен вовсе.
//   OUT_PAIR_I — dK/dV-ядро: P2[j*(BQ/2)+ih] = half2{P[2ih][j],P[2ih+1][j]} (и dS так же).
// Упаковка здесь бесплатна: поток владеет подряд идущими i (QPT штук) и j (JPT штук).
enum { OUT_PLAIN=0, OUT_PAIR_J=1, OUT_PAIR_I=2 };

// LP — паддинг строки в LDS (в half). Строка длиной D даёт всем 16 линиям один банк
// (D/2 кратно 32 при D=64/128/256) => 16-кратный конфликт на чтениях K/V. LP это ломает.
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,typename E>
__device__ __forceinline__ void fill_pd(
    const E* Qf,const E* Kf,const E* Vf,const E* dOf,const float* lse,const float* di,
    E* Pt,E* dSt,const BwdParams& pr,int h,int qbase,int kbase,
    float invQK,float invdOV,float alibi_slope)
{
    constexpr int LD=D+LP;
    const int QPT=BQ/16, JPT=BK/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const float sc=pr.softcap;
    // регистровый fdot2-тайлинг S=Q·Kᵀ и dP=dO·Vᵀ: операнды грузятся раз на dh, переиспользуются
    // QPT/JPT раз (row-major тайлы дают непрерывные half2 по d).
    float Sa[QPT][JPT], Pa[QPT][JPT];
    #pragma unroll
    for (int a=0;a<QPT;a++)
        #pragma unroll
        for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=0.f; Pa[a][bb]=0.f; }
    if constexpr (MODE==BW_F32){
        #pragma unroll 2
        for (int d=0; d<D; d++){
            float qv[QPT], ov[QPT], kv[JPT], vv[JPT];
            #pragma unroll
            for (int a=0;a<QPT;a++){ qv[a]=(float)Qf[(ty*QPT+a)*LD+d]; ov[a]=(float)dOf[(ty*QPT+a)*LD+d]; }
            #pragma unroll
            for (int bb=0;bb<JPT;bb++){ kv[bb]=(float)Kf[(tx*JPT+bb)*LD+d]; vv[bb]=(float)Vf[(tx*JPT+bb)*LD+d]; }
            #pragma unroll
            for (int a=0;a<QPT;a++)
                #pragma unroll
                for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=fmaf(qv[a],kv[bb],Sa[a][bb]); Pa[a][bb]=fmaf(ov[a],vv[bb],Pa[a][bb]); }
        }
    } else {
        // Развёртка на UN шагов вручную: `#pragma unroll UN` не берёт шаблонный параметр.
        // UN задаёт, сколько чтений LDS летит одновременно до первого fdot2 — это и есть ILP.
        #pragma unroll 1
        for (int dh0=0; dh0<D/2; dh0+=UN){
            #pragma unroll
            for (int uu=0; uu<UN; uu++){
                const int dh=dh0+uu;
                half2_t qv[QPT], ov[QPT], kv[JPT], vv[JPT];
                #pragma unroll
                for (int a=0;a<QPT;a++){ qv[a]=((const half2_t*)(Qf+(ty*QPT+a)*LD))[dh]; ov[a]=((const half2_t*)(dOf+(ty*QPT+a)*LD))[dh]; }
                #pragma unroll
                for (int bb=0;bb<JPT;bb++){ kv[bb]=((const half2_t*)(Kf+(tx*JPT+bb)*LD))[dh]; vv[bb]=((const half2_t*)(Vf+(tx*JPT+bb)*LD))[dh]; }
                #pragma unroll
                for (int a=0;a<QPT;a++)
                    #pragma unroll
                    for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=__builtin_amdgcn_fdot2(qv[a],kv[bb],Sa[a][bb],false); Pa[a][bb]=__builtin_amdgcn_fdot2(ov[a],vv[bb],Pa[a][bb],false); }
            }
        }
    }
    E Pv[QPT][JPT], Dv[QPT][JPT];
    #pragma unroll
    for (int a=0; a<QPT; a++){
        const int i=ty*QPT+a, qq=qbase+i; const float LSEi=lse[i], Di=di[i];
        #pragma unroll
        for (int bb=0; bb<JPT; bb++){
            const int j=tx*JPT+bb, kk=kbase+j;
            float s=Sa[a][bb]*invQK, dp=Pa[a][bb]*invdOV;
            float g0=pr.scale*s, deriv=1.f, logit;
            if (sc>0.f){ float tnh=tanhf(g0/sc); logit=sc*tnh; deriv=1.f-tnh*tnh; } else logit=g0;
            logit += alibi_slope*(float)(kk-qq);
            float p = fwd_ok(pr,qq,kk) ? __expf(logit-LSEi) : 0.f;
            float z = 1.f;
            if (pr.dropout){ unsigned long long idx=(((unsigned long long)h*pr.SQ+qq)*pr.SK+kk);
                z = dropout_mul(pr.drop_seed, idx, pr.dropout_p, pr.keep_scale); }
            Pv[a][bb] = (E)(z*p);
            Dv[a][bb] = (E)(p*(z*dp - Di)*deriv);
        }
    }
    if constexpr (OUT==OUT_PLAIN){
        #pragma unroll
        for (int a=0;a<QPT;a++){ const int i=ty*QPT+a;
            #pragma unroll
            for (int bb=0;bb<JPT;bb++){ const int j=tx*JPT+bb;
                Pt[i*BK+j]=Pv[a][bb]; dSt[i*BK+j]=Dv[a][bb]; } }
    } else if constexpr (OUT==OUT_PAIR_J){
        // dS2[i][jh]; P не пишем — dQ-ядру он не нужен
        half2_t* dS2=(half2_t*)dSt;
        #pragma unroll
        for (int a=0;a<QPT;a++){ const int i=ty*QPT+a;
            #pragma unroll
            for (int bb=0;bb<JPT;bb+=2){ const int jh=(tx*JPT+bb)>>1;
                dS2[i*(BK/2)+jh]=half2_t{Dv[a][bb],Dv[a][bb+1]}; } }
    } else {
        // P2/dS2[j][ih] — k-major, пара по i: второй GEMM берёт готовый dword
        half2_t* P2=(half2_t*)Pt; half2_t* D2=(half2_t*)dSt;
        #pragma unroll
        for (int bb=0;bb<JPT;bb++){ const int j=tx*JPT+bb;
            #pragma unroll
            for (int a=0;a<QPT;a+=2){ const int ih=(ty*QPT+a)>>1;
                P2[j*(BQ/2)+ih]=half2_t{Pv[a][bb],Pv[a+1][bb]};
                D2[j*(BQ/2)+ih]=half2_t{Dv[a][bb],Dv[a+1][bb]}; } }
    }
}

// ============================ dK/dV-ядро ============================
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
__global__ __launch_bounds__(256)
void fa_bwd_dkdv(BwdParams pr){
    using E=typename ElemT<MODE>::type;
    constexpr int LD=D+LP;
    const int KPT=BK/16, EPT=D/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const int kt=blockIdx.x, kvh=blockIdx.y, b=blockIdx.z, kbase=kt*BK;
    const float invQ=exp2i(-pr.shQ), invdO=exp2i(-pr.shdO), invdOV=exp2i(-(pr.shdO+pr.shV));
    const float invQK=exp2i(-(pr.shQ+pr.shK));

    __shared__ __align__(16) E Kf[BK*LD]; __shared__ __align__(16) E Vf[BK*LD];
    __shared__ __align__(16) E Qf[BQ*LD]; __shared__ __align__(16) E dOf[BQ*LD];
    __shared__ __align__(16) E Pt[BQ*BK]; __shared__ __align__(16) E dSt[BQ*BK];
    __shared__ float lse[BQ]; __shared__ float di[BQ];

    float accK[BK/16][D/16], accV[BK/16][D/16];   // D/16, а не 8: при D>128 EPT больше
    #pragma unroll
    for (int r=0;r<KPT;r++)
        #pragma unroll
        for (int e=0;e<EPT;e++){ accK[r][e]=0.f; accV[r][e]=0.f; }

    for (int hg=0; hg<pr.gqa; hg++){
        const int h=kvh*pr.gqa+hg; const float aslope = pr.alibi? pr.alibi[h] : 0.f;
        const long long kb0=(long long)b*pr.ks.b+(long long)kvh*pr.ks.h;
        const long long vb0=(long long)b*pr.vs.b+(long long)kvh*pr.vs.h;
        for (int e=t; e<BK*D; e+=256){ int j=e/D, d=e%D, sk=kbase+j;
            Kf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.K,kb0+(long long)sk*pr.ks.s+d,exp2i(pr.shK)):(E)0;
            Vf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.V,vb0+(long long)sk*pr.vs.s+d,exp2i(pr.shV)):(E)0; }

        // Для ключа kk запросы разрешены в [kk-delta-win_right .. kk-delta+win_left].
        // Берём объединение по всем ключам тайла: начало по первому ключу, конец по последнему.
        const int delta=pr.SK-pr.SQ; int qstart=0, qend=pr.SQ;
        if (pr.causal)          qstart=max(0, kbase-delta);
        if (pr.win_right>=0)    qstart=max(qstart, max(0, kbase-delta-pr.win_right));
        if (pr.win_left>=0)     qend  =min(pr.SQ, kbase+BK-1-delta+pr.win_left+1);
        qstart=(qstart/BQ)*BQ;

        for (int qb=qstart; qb<qend; qb+=BQ){
            const long long qb0=(long long)b*pr.qs.b+(long long)h*pr.qs.h;
            const long long ob0=(long long)b*pr.dos.b+(long long)h*pr.dos.h;
            __syncthreads();
            for (int e=t; e<BQ*D; e+=256){ int i=e/D, d=e%D, sq=qb+i;
                Qf[i*LD+d] =(sq<pr.SQ)?load_tile<MODE>(pr.Q, qb0+(long long)sq*pr.qs.s+d, exp2i(pr.shQ)):(E)0;
                dOf[i*LD+d]=(sq<pr.SQ)?load_tile<MODE>(pr.dO,ob0+(long long)sq*pr.dos.s+d,exp2i(pr.shdO)):(E)0; }
            for (int i=t; i<BQ; i+=256){ int sq=qb+i; long long li=(((long long)b*pr.Hq+h)*pr.SQ+(sq<pr.SQ?sq:0));
                lse[i]=(sq<pr.SQ)?pr.lse[li]:1e30f; di[i]=(sq<pr.SQ)?pr.dsum[li]:0.f; }
            __syncthreads();
            fill_pd<D,MODE,BQ,BK,OUT,LP,UN,E>(
                Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qb,kbase,invQK,invdOV,aslope);
            __syncthreads();
            // dV[k][d]+=Σ_i P̃[i][k]·dO[i][d]; dK[k][d]+=Σ_i dS[i][k]·Q[i][d] (контракция по i)
            if constexpr (MODE==BW_F32 || OUT==OUT_PLAIN){
                #pragma unroll
                for (int r=0;r<KPT;r++){ const int k=ty*KPT+r;
                    #pragma unroll
                    for (int e=0;e<EPT;e++){ const int d=tx*EPT+e; float av=0.f, ak=0.f;
                        for (int i=0;i<BQ;i++){ av+=(float)Pt[i*BK+k]*(float)dOf[i*LD+d]; ak+=(float)dSt[i*BK+k]*(float)Qf[i*LD+d]; }
                        accV[r][e]+=av; accK[r][e]+=ak; } }
            } else if (!pr.probe) {
                const half2_t* P2=(const half2_t*)Pt; const half2_t* D2=(const half2_t*)dSt;
                #pragma unroll 4
                for (int ih=0; ih<BQ/2; ih++){
                    half2_t pv[KPT], dv[KPT], ov[D/16], qv[D/16];
                    #pragma unroll
                    for (int r=0;r<KPT;r++){ const int k=ty*KPT+r;
                        pv[r]=P2[k*(BQ/2)+ih];      // готовый dword вместо двух u16 + v_perm
                        dv[r]=D2[k*(BQ/2)+ih]; }
                    #pragma unroll
                    for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
                        ov[e]=half2_t{dOf[(2*ih)*LD+d], dOf[(2*ih+1)*LD+d]};
                        qv[e]=half2_t{Qf[(2*ih)*LD+d],  Qf[(2*ih+1)*LD+d]}; }
                    #pragma unroll
                    for (int r=0;r<KPT;r++)
                        #pragma unroll
                        for (int e=0;e<EPT;e++){ accV[r][e]=__builtin_amdgcn_fdot2(pv[r],ov[e],accV[r][e],false);
                                                 accK[r][e]=__builtin_amdgcn_fdot2(dv[r],qv[e],accK[r][e],false); }
                }
            }
        }
    }
    const long long dk0=(long long)b*pr.dks.b+(long long)kvh*pr.dks.h;
    const long long dv0=(long long)b*pr.dvs.b+(long long)kvh*pr.dvs.h;
    #pragma unroll
    for (int r=0;r<KPT;r++){ const int k=ty*KPT+r, sk=kbase+k; if(sk>=pr.SK) continue;
        #pragma unroll
        for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
            store_out<MODE>(pr.dV, dv0+(long long)sk*pr.dvs.s+d, accV[r][e]*invdO);
            store_out<MODE>(pr.dK, dk0+(long long)sk*pr.dks.s+d, accK[r][e]*invQ*pr.scale); }
    }
}

// ============================ dQ-ядро ============================
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
__global__ __launch_bounds__(256)
void fa_bwd_dq(BwdParams pr){
    using E=typename ElemT<MODE>::type;
    constexpr int LD=D+LP;
    const int QPT=BQ/16, EPT=D/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const int qt=blockIdx.x, h=blockIdx.y, b=blockIdx.z, kvh=h/pr.gqa, qbase=qt*BQ;
    const float invK=exp2i(-pr.shK), invdOV=exp2i(-(pr.shdO+pr.shV)), invQK=exp2i(-(pr.shQ+pr.shK));
    const float aslope = pr.alibi? pr.alibi[h] : 0.f;

    // __align__(16): буферы переиспользуются как half2 (упакованные пары) — нужна выравненность dword
    __shared__ __align__(16) E Qf[BQ*LD]; __shared__ __align__(16) E dOf[BQ*LD];
    __shared__ __align__(16) E Kf[BK*LD]; __shared__ __align__(16) E Vf[BK*LD];
    __shared__ __align__(16) E Pt[BQ*BK]; __shared__ __align__(16) E dSt[BQ*BK];
    __shared__ float lse[BQ]; __shared__ float di[BQ];

    float accQ[BQ/16][D/16];                      // D/16, а не 8: при D>128 EPT больше
    #pragma unroll
    for (int a=0;a<QPT;a++)
        #pragma unroll
        for (int e=0;e<EPT;e++) accQ[a][e]=0.f;

    const long long qb0=(long long)b*pr.qs.b+(long long)h*pr.qs.h;
    const long long ob0=(long long)b*pr.dos.b+(long long)h*pr.dos.h;
    for (int e=t; e<BQ*D; e+=256){ int i=e/D, d=e%D, sq=qbase+i;
        Qf[i*LD+d] =(sq<pr.SQ)?load_tile<MODE>(pr.Q, qb0+(long long)sq*pr.qs.s+d, exp2i(pr.shQ)):(E)0;
        dOf[i*LD+d]=(sq<pr.SQ)?load_tile<MODE>(pr.dO,ob0+(long long)sq*pr.dos.s+d,exp2i(pr.shdO)):(E)0; }
    for (int i=t; i<BQ; i+=256){ int sq=qbase+i; long long li=(((long long)b*pr.Hq+h)*pr.SQ+(sq<pr.SQ?sq:0));
        lse[i]=(sq<pr.SQ)?pr.lse[li]:1e30f; di[i]=(sq<pr.SQ)?pr.dsum[li]:0.f; }

    // Для запроса qq ключи разрешены в [qq+delta-win_left .. qq+delta+win_right].
    const int delta=pr.SK-pr.SQ; int kstart=0, kend=pr.SK;
    if (pr.causal)       kend  =min(pr.SK, qbase+BQ-1+delta+1);
    if (pr.win_right>=0) kend  =min(kend,  min(pr.SK, qbase+BQ-1+delta+pr.win_right+1));
    if (pr.win_left>=0)  kstart=max(0, qbase+delta-pr.win_left);
    kstart=(kstart/BK)*BK;

    const long long kb0=(long long)b*pr.ks.b+(long long)kvh*pr.ks.h;
    const long long vb0=(long long)b*pr.vs.b+(long long)kvh*pr.vs.h;
    for (int kb=kstart; kb<kend; kb+=BK){
        __syncthreads();
        for (int e=t; e<BK*D; e+=256){ int j=e/D, d=e%D, sk=kb+j;
            Kf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.K,kb0+(long long)sk*pr.ks.s+d,exp2i(pr.shK)):(E)0;
            Vf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.V,vb0+(long long)sk*pr.vs.s+d,exp2i(pr.shV)):(E)0; }
        __syncthreads();
        if constexpr (MODE==BW_F32 || OUT==OUT_PLAIN){
            fill_pd<D,MODE,BQ,BK,OUT_PLAIN,LP,UN,E>(Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qbase,kb,invQK,invdOV,aslope);
            __syncthreads();
            #pragma unroll
            for (int a=0;a<QPT;a++){ const int q=ty*QPT+a;
                #pragma unroll
                for (int e=0;e<EPT;e++){ const int d=tx*EPT+e; float aq=0.f;
                    for(int j=0;j<BK;j++) aq+=(float)dSt[q*BK+j]*(float)Kf[j*LD+d]; accQ[a][e]+=aq; } }
        } else {
            fill_pd<D,MODE,BQ,BK,OUT_PAIR_J,LP,UN,E>(Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qbase,kb,invQK,invdOV,aslope);
            // K, спаренный по j: KT[jh][d] = half2{K[2jh][d], K[2jh+1][d]}. Кладём в память Vf —
            // после fill_pd она мертва, а на следующей итерации всё равно перезагружается из глобалки.
            // Транспон стоит (BK/2)*D записей на тайл, а в GEMM та же упаковка делалась 16 раз
            // подряд (по одному разу на каждое ty) — то есть это ровно в 16 раз дешевле.
            half2_t* KT=(half2_t*)Vf;
            __syncthreads();
            for (int e=t; e<(BK/2)*D; e+=256){ int jh=e/D, d=e%D;
                KT[jh*D+d]=half2_t{Kf[(2*jh)*LD+d], Kf[(2*jh+1)*LD+d]}; }
            const half2_t* dS2=(const half2_t*)dSt;
            __syncthreads();
            // dQ[q][d]+=Σ_j dS[q][j]·K[j][d] — оба операнда уже готовые dword'ы
            if (pr.probe) continue;
            #pragma unroll 4
            for (int jh=0; jh<BK/2; jh++){
                half2_t sv[QPT], kv[D/16];
                #pragma unroll
                for (int a=0;a<QPT;a++) sv[a]=dS2[(ty*QPT+a)*(BK/2)+jh];
                #pragma unroll
                for (int e=0;e<EPT;e++) kv[e]=KT[jh*D+(DINT? (e*16+tx) : (tx*EPT+e))];
                #pragma unroll
                for (int a=0;a<QPT;a++)
                    #pragma unroll
                    for (int e=0;e<EPT;e++) accQ[a][e]=__builtin_amdgcn_fdot2(sv[a],kv[e],accQ[a][e],false);
            }
        }
    }
    const long long dq0=(long long)b*pr.dqs.b+(long long)h*pr.dqs.h;
    #pragma unroll
    for (int a=0;a<QPT;a++){ const int q=ty*QPT+a, sq=qbase+q; if(sq>=pr.SQ) continue;
        #pragma unroll
        for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
            store_out<MODE>(pr.dQ, dq0+(long long)sq*pr.dqs.s+d, accQ[a][e]*invK*pr.scale); }
    }
}

// ============================ лаунчер ============================
// stage: 0 = все три (боевой путь), 1 = только preprocess, 2 = только dK/dV, 3 = только dQ.
// Нужен, чтобы мерить стадии по отдельности: суммарное время ничего не говорит о том, кто ест.
static int g_bwd_stage = 0;
// Ablation ВНУТРИ ядра: pr.probe=1 пропускает второй GEMM (остаётся только fill_pd).
// Результат при этом неверен — это зонд для замера доли, не режим работы.
static int g_bwd_probe = 0;

// Вариант раскладки/тайла выбирается в рантайме — стенд перебирает весь куб и мерит каждый.
// 0 = боевая конфигурация (BQ64 BK32, пары, паддинг 0). Меняется только через замер.
static int g_bwd_variant = 0;

template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
static void launch_mode(BwdParams pr, hipStream_t st){
    dim3 pblk(256), pgrid((pr.SQ+255)/256, pr.Hq, pr.B);
    dim3 gK((pr.SK+BK-1)/BK, pr.Hk, pr.B), gQ((pr.SQ+BQ-1)/BQ, pr.Hq, pr.B), blk(256);
    const int s = g_bwd_stage;
    if (s==0 || s==1) hipLaunchKernelGGL((fa_bwd_preprocess<D,MODE>), pgrid, pblk, 0, st, pr);
    if (s==0 || s==2) hipLaunchKernelGGL((fa_bwd_dkdv<D,MODE,BQ,BK,OUT,LP,UN,DINT>), gK, blk, 0, st, pr);
    if (s==0 || s==3) hipLaunchKernelGGL((fa_bwd_dq<D,MODE,BQ,BK,OUT,LP,UN,DINT>),   gQ, blk, 0, st, pr);
}

// LDS одного варианта (байт): K,V,Q,dO по (D+LP) + P,dS + lse,di
static constexpr size_t bwd_lds(int D,int BQ,int BK,int LP,int el){
    return (size_t)el*( (size_t)2*BK*(D+LP) + (size_t)2*BQ*(D+LP) + (size_t)2*BQ*BK ) + 8*(size_t)BQ;
}

// Вариант выполним, если влезает в LDS и раскладка пар совместима с тайлом.
template<int D,int MODE,int BQ,int BK,int OUT,int LP>
static constexpr bool cfg_ok(){
    constexpr int el = (MODE==BW_F32)?4:2;
    return bwd_lds(D,BQ,BK,LP,el) <= 65536
        && (OUT!=OUT_PAIR_I || (BQ/16)%2==0)   // пара по i внутри потока
        && (OUT!=OUT_PAIR_J || (BK/16)%2==0);  // пара по j внутри потока
}

// probe=true -> только сказать, выполним ли; probe=false -> запустить.
template<int D,int MODE,int BQ,int BK,int LP,bool PAIR,int UN,int DINT>
static bool run_cfg(BwdParams& pr, hipStream_t st, bool probe){
    constexpr int OUT = PAIR ? OUT_PAIR_I : OUT_PLAIN;
    if constexpr (cfg_ok<D,MODE,BQ,BK,OUT,LP>() && (!PAIR || (BK/16)%2==0)){
        if (!probe) launch_mode<D,MODE,BQ,BK,OUT,LP,UN,DINT>(pr,st);
        return true;
    } else { (void)pr; (void)st; return false; }
}

// Список вариантов стенда. Убрать строку = откатить вариант; боевой всегда №0.
#define BWD_VARIANT_LIST(X) \
    X(0, 64,32, 6, true , 2, 0, "LP6 UN2 (v boyu)") \
    X(1, 64,32, 6, true , 2, 1, "LP6 UN2 + d bez konflikta") \
    X(2, 64,32, 6, true , 4, 1, "LP6 UN4 + d bez konflikta") \
    X(3, 64,32, 6, true , 1, 1, "LP6 UN1 + d bez konflikta") \
    X(4, 64,32, 2, true , 2, 1, "LP2 UN2 + d bez konflikta") \
    X(5, 64,32, 0, true , 2, 1, "LP0 UN2 + d bez konflikta") \
    X(6, 32,32, 6, true , 2, 1, "BQ32 LP6 + d") \
    X(7, 32,64, 6, true , 2, 1, "BQ32 BK64 LP6 + d") \
    X(8, 64,32, 0, true , 4, 0, "LP0 UN4 (do paddinga)") \
    X(9, 64,32, 0, false, 4, 0, "LP0 plain UN4 (ishodnoe)")

template<int D,int MODE>
static bool launch_variant(BwdParams& pr, hipStream_t st, int v, bool probe){
    switch(v){
        #define X(id,bq,bk,lp,pair,un,di,nm) case id: return run_cfg<D,MODE,bq,bk,lp,pair,un,di>(pr,st,probe);
        BWD_VARIANT_LIST(X)
        #undef X
        default: return false;
    }
}
static const char* bwd_variant_name(int v){
    switch(v){
        #define X(id,bq,bk,lp,pair,un,di,nm) case id: return nm;
        BWD_VARIANT_LIST(X)
        #undef X
        default: return "?";
    }
}
static constexpr int BWD_NVARIANT = 10;

// Запасные конфиги с уменьшающимся тайлом — для больших D боевой (BQ=64,BK=32,LP=6) не влезает
// в 64 КБ LDS. Раньше в этом случае НЕ ЗАПУСКАЛОСЬ НИ ОДНО ЯДРО (launch_variant возвращал false
// и на боевом варианте тоже), а dQ/dK/dV — это empty_like, то есть наружу уходил неинициализи-
// рованный мусор вместо градиентов. Молча. Теперь берём первый конфиг, который влезает,
// а если не влез ни один — падаем громко.
// LDS считается так: (Kf+Vf)=2*BK*(D+LP) + (Qf+dOf)=2*BQ*(D+LP) + (Pt+dSt)=2*BQ*BK элементов.
// PAIR требует (BK/16)%2==0, поэтому при BK=16 он выключается.
template<int D,int MODE>
static void launch_fallback(BwdParams& pr, hipStream_t st){
    // BQ и BK не меньше 16: тайлинг раскладывается на 16 линий (QPT=BQ/16, JPT=BK/16),
    // при меньшем размере это даёт массивы нулевой длины и ядро не компилируется.
    if (run_cfg<D,MODE,64,32,6,true ,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,32,32,6,true ,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,32,16,6,false,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,16,16,6,false,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,16,16,0,false,1,0>(pr,st,false)) return;
    printf("fable_attn backward: head_dim=%d не влезает в LDS ни в одной раскладке\n", D);
    abort();
}
template<int D>
static void launch_D(BwdParams pr, int MODE, hipStream_t st){
    const int v = g_bwd_variant;
    if (MODE==BW_F16)       { if(!launch_variant<D,BW_F16 >(pr,st,v,false)) launch_fallback<D,BW_F16 >(pr,st); }
    else if (MODE==BW_BF16) { if(!launch_variant<D,BW_BF16>(pr,st,v,false)) launch_fallback<D,BW_BF16>(pr,st); }
    else {
        // fp32 держит вчетверо больше байт на элемент, поэтому при D=256 не влезает даже
        // минимальный тайл 16x16. Это не дыра паритета: у flash-attn fp32 нет вовсе, он наш
        // бонус для gradcheck. Обёртка ловит этот случай заранее и говорит внятно.
        if (!run_cfg<D,BW_F32,32,16,0,false,4,0>(pr,st,false))
            if (!run_cfg<D,BW_F32,16,16,0,false,2,0>(pr,st,false)) {
                printf("fable_attn backward fp32: head_dim=%d не влезает в LDS (потолок 224)\n", D);
                abort(); }
    }
}
// выполним ли вариант для боевой формы гейта (D=128, fp16/bf16)
static bool bwd_variant_buildable(int v,int MODE){
    BwdParams dummy{};
    if (MODE==BW_F16)  return launch_variant<128,BW_F16 >(dummy,0,v,true);
    if (MODE==BW_BF16) return launch_variant<128,BW_BF16>(dummy,0,v,true);
    return v==0;
}
extern "C" void fable_attn_backward_launch(BwdParams pr, int MODE, hipStream_t st){
    switch (pr.D){
        case 16: launch_D<16>(pr,MODE,st); break;   case 32: launch_D<32>(pr,MODE,st); break;
        case 48: launch_D<48>(pr,MODE,st); break;   case 64: launch_D<64>(pr,MODE,st); break;
        case 80: launch_D<80>(pr,MODE,st); break;   case 96: launch_D<96>(pr,MODE,st); break;
        case 112:launch_D<112>(pr,MODE,st);break;   case 128:launch_D<128>(pr,MODE,st);break;
        case 160:launch_D<160>(pr,MODE,st);break;   case 192:launch_D<192>(pr,MODE,st);break;
        case 224:launch_D<224>(pr,MODE,st);break;   case 256:launch_D<256>(pr,MODE,st);break;
        // Раньше здесь стоял пустой default: при head_dim=256 backward не запускал НИЧЕГО,
        // и наружу уходили неинициализированные dQ/dK/dV. Молчать нельзя.
        default:
            printf("fable_attn backward: head_dim=%d не поддержан\n", pr.D); abort();
    }
}

// ==========================================================================
//                          STANDALONE-ГЕЙТ (fp64-оракул)
// ==========================================================================
#ifdef FA_STANDALONE
#include <cstring>
static uint32_t GRS=12345u;
static float rndf(){ GRS=GRS*1664525u+1013904223u; return ((int)((GRS>>9)&0xFFFF)-32768)/32768.0f; }
// хост-реплика fa_common::dropout_mul (та же philox)
static float hdrop(unsigned long long seed, unsigned long long idx, float p, float invk){
    uint32_t k0=(uint32_t)seed,k1=(uint32_t)(seed>>32),c0=(uint32_t)idx,c1=(uint32_t)(idx>>32),c2=0,c3=0;
    for(int i=0;i<10;i++){ uint32_t hi0=(uint32_t)(((uint64_t)0xD2511F53u*c0)>>32), lo0=0xD2511F53u*c0;
        uint32_t hi1=(uint32_t)(((uint64_t)0xCD9E8D57u*c2)>>32), lo1=0xCD9E8D57u*c2;
        uint32_t n0=hi1^c1^k0,n1=lo1,n2=hi0^c3^k1,n3=lo0; c0=n0;c1=n1;c2=n2;c3=n3; k0+=0x9E3779B9u;k1+=0xBB67AE85u; }
    float u=(c0>>8)*(1.0f/16777216.0f); return (u>=p)?invk:0.0f;
}
static void pack(std::vector<uint8_t>& o,const std::vector<float>& v,int m){
    if(m==BW_F32){ o.resize(v.size()*4); memcpy(o.data(),v.data(),o.size()); return; }
    o.resize(v.size()*2);
    for(size_t i=0;i<v.size();i++){ if(m==BW_F16){ _Float16 h=(_Float16)v[i]; memcpy(&o[i*2],&h,2);}
        else { union{uint32_t u;float f;}c;c.f=v[i];uint32_t lsb=(c.u>>16)&1;c.u+=0x7fff+lsb;uint16_t bf=(uint16_t)(c.u>>16);memcpy(&o[i*2],&bf,2);} } }
static float unpk(const uint8_t* p,int i,int m){
    if(m==BW_F32){ float f; memcpy(&f,p+i*4,4); return f; }
    if(m==BW_F16){ _Float16 h; memcpy(&h,p+i*2,2); return (float)h; }
    uint16_t bf; memcpy(&bf,p+i*2,2); uint32_t u=(uint32_t)bf<<16; float f; memcpy(&f,&u,4); return f; }
static int host_shift(const std::vector<float>& v,int m){ if(m!=BW_BF16) return 0;
    float mx=0; for(float x:v) mx=fmaxf(mx,fabsf(x)); if(mx<=0) return 0; int e; frexpf(mx,&e); return 13-e; }

struct Case{ int B,Hq,Hk,SQ,SK,D; bool causal; int wl,wr; float softcap; bool alibi; float pdrop; const char* name; };

static int run_case(const Case& c,int mode,double tol,bool print){
    // fp32 держит вчетверо больше байт на элемент: при D>224 не влезает даже минимальный
    // тайл 16x16 (16896 элементов * 4 = 67.6 КБ против лимита 64). У flash-attn fp32 нет
    // вовсе, так что это не дыра паритета — просто честно пропускаем клетку.
    if (mode==BW_F32 && c.D>224){
        if(print) printf("  [fp32/%-22s] пропуск: не влезает в LDS (потолок fp32 — D=224)\n", c.name);
        return 0;
    }
    const int B=c.B,Hq=c.Hq,Hk=c.Hk,SQ=c.SQ,SK=c.SK,D=c.D,gqa=Hq/Hk;
    const float scale=1.0f/sqrtf((float)D);
    std::vector<float> Q(B*Hq*SQ*D),K(B*Hk*SK*D),V(B*Hk*SK*D),dO(B*Hq*SQ*D);
    for(auto&x:Q)x=rndf(); for(auto&x:K)x=rndf(); for(auto&x:V)x=rndf(); for(auto&x:dO)x=rndf();
    std::vector<float> al(Hq); for(int h=0;h<Hq;h++) al[h]= c.alibi? -0.1f*(h+1):0.f;
    unsigned long long seed=777; float keep=1.0f-c.pdrop, invk=1.0f/keep;
    auto ok=[&](int qq,int kk){ if(kk>=SK||qq>=SQ)return false; int delta=SK-SQ,diag=qq+delta;
        if(c.causal&&kk>diag)return false;
        if(c.wl>=0&&kk<diag-c.wl)return false; if(c.wr>=0&&kk>diag+c.wr)return false; return true; };
    // forward-оракул (fp64): O, LSE(fp32) с dropout как в ядре (idx без батча => h,qq,kk)
    std::vector<float> O(B*Hq*SQ*D,0.f), LSE(B*Hq*SQ);
    for(int b=0;b<B;b++)for(int h=0;h<Hq;h++){int kvh=h/gqa;
        for(int qq=0;qq<SQ;qq++){ std::vector<double> lg(SK,-1e30); double mx=-1e30;
            for(int kk=0;kk<SK;kk++){ if(!ok(qq,kk))continue; double s=0;
                for(int d=0;d<D;d++) s+=(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double g0=scale*s,logit; if(c.softcap>0){double t=tanh(g0/c.softcap);logit=c.softcap*t;}else logit=g0;
                logit+=al[h]*(double)(kk-qq); lg[kk]=logit; if(logit>mx)mx=logit; }
            double sum=0; for(int kk=0;kk<SK;kk++){ if(lg[kk]<=-1e29){lg[kk]=0;continue;} double e=exp(lg[kk]-mx); lg[kk]=e; sum+=e; }
            LSE[((long long)b*Hq+h)*SQ+qq]=(float)(mx+log(sum>0?sum:1));
            for(int d=0;d<D;d++){ double acc=0; for(int kk=0;kk<SK;kk++){ double p=lg[kk]/(sum>0?sum:1);
                double z=(c.pdrop>0)? (double)hdrop(seed,(((unsigned long long)h*SQ+qq)*SK+kk),c.pdrop,invk):1.0;
                acc+=z*p*(double)V[(((long long)b*Hk+kvh)*SK+kk)*D+d]; }
                O[(((long long)b*Hq+h)*SQ+qq)*D+d]=(float)acc; } } }
    // backward-оракул (fp64)
    std::vector<double> gQ(Q.size(),0),gK(K.size(),0),gV(V.size(),0);
    for(int b=0;b<B;b++)for(int h=0;h<Hq;h++){int kvh=h/gqa;
        for(int qq=0;qq<SQ;qq++){ double LSEi=LSE[((long long)b*Hq+h)*SQ+qq], Di=0;
            for(int d=0;d<D;d++) Di+=(double)O[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d];
            for(int kk=0;kk<SK;kk++){ if(!ok(qq,kk))continue; double s=0;
                for(int d=0;d<D;d++) s+=(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double g0=scale*s,deriv=1,logit; if(c.softcap>0){double t=tanh(g0/c.softcap);logit=c.softcap*t;deriv=1-t*t;}else logit=g0;
                logit+=al[h]*(double)(kk-qq); double p=exp(logit-LSEi);
                double z=(c.pdrop>0)? (double)hdrop(seed,(((unsigned long long)h*SQ+qq)*SK+kk),c.pdrop,invk):1.0;
                double dp=0; for(int d=0;d<D;d++) dp+=(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)V[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double dS=p*(z*dp-Di)*deriv;
                for(int d=0;d<D;d++){ gV[(((long long)b*Hk+kvh)*SK+kk)*D+d]+=z*p*(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d];
                    gK[(((long long)b*Hk+kvh)*SK+kk)*D+d]+=scale*dS*(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d];
                    gQ[(((long long)b*Hq+h)*SQ+qq)*D+d]+=scale*dS*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d]; } } } }
    // GPU
    std::vector<uint8_t> qb,kb,vb,ob,dob; pack(qb,Q,mode);pack(kb,K,mode);pack(vb,V,mode);pack(ob,O,mode);pack(dob,dO,mode);
    void *dQp,*dKp,*dVp,*dOp,*ddOp,*ddQ,*ddK,*ddV; float *dl,*ds,*dal=nullptr;
    HIP_CHECK(hipMalloc(&dQp,qb.size()));HIP_CHECK(hipMalloc(&dKp,kb.size()));HIP_CHECK(hipMalloc(&dVp,vb.size()));
    HIP_CHECK(hipMalloc(&dOp,ob.size()));HIP_CHECK(hipMalloc(&ddOp,dob.size()));
    HIP_CHECK(hipMalloc(&ddQ,qb.size()));HIP_CHECK(hipMalloc(&ddK,kb.size()));HIP_CHECK(hipMalloc(&ddV,vb.size()));
    HIP_CHECK(hipMalloc(&dl,LSE.size()*4));HIP_CHECK(hipMalloc(&ds,LSE.size()*4));
    if(c.alibi){HIP_CHECK(hipMalloc(&dal,Hq*4));HIP_CHECK(hipMemcpy(dal,al.data(),Hq*4,hipMemcpyHostToDevice));}
    HIP_CHECK(hipMemcpy(dQp,qb.data(),qb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dKp,kb.data(),kb.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dVp,vb.data(),vb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dOp,ob.data(),ob.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddOp,dob.data(),dob.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dl,LSE.data(),LSE.size()*4,hipMemcpyHostToDevice));
    auto sD=[&](int H,int S){ return Str{(long long)H*S*D,(long long)S*D,(long long)D}; };
    BwdParams pr{}; pr.Q=dQp;pr.K=dKp;pr.V=dVp;pr.O=dOp;pr.dO=ddOp;pr.dQ=ddQ;pr.dK=ddK;pr.dV=ddV;pr.lse=dl;pr.dsum=ds;
    pr.qs=sD(Hq,SQ);pr.ks=sD(Hk,SK);pr.vs=sD(Hk,SK);pr.os=sD(Hq,SQ);pr.dos=sD(Hq,SQ);pr.dqs=sD(Hq,SQ);pr.dks=sD(Hk,SK);pr.dvs=sD(Hk,SK);
    pr.B=B;pr.Hq=Hq;pr.Hk=Hk;pr.SQ=SQ;pr.SK=SK;pr.D=D;pr.gqa=gqa;pr.scale=scale;pr.softcap=c.softcap;pr.alibi=dal;
    pr.causal=c.causal;pr.win_left=c.wl;pr.win_right=c.wr;pr.dropout=(c.pdrop>0);pr.drop_seed=seed;pr.dropout_p=c.pdrop;pr.keep_scale=invk;
    pr.shQ=host_shift(Q,mode);pr.shK=host_shift(K,mode);pr.shV=host_shift(V,mode);pr.shdO=host_shift(dO,mode);
    fable_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> hq(qb.size()),hk(kb.size()),hv(vb.size());
    HIP_CHECK(hipMemcpy(hq.data(),ddQ,qb.size(),hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hk.data(),ddK,kb.size(),hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hv.data(),ddV,vb.size(),hipMemcpyDeviceToHost));
    auto rr=[&](const std::vector<uint8_t>& g,const std::vector<double>& r)->double{ double se=0,sr=0;
        for(size_t i=0;i<r.size();i++){ double gg=unpk(g.data(),(int)i,mode),d=gg-r[i]; se+=d*d; sr+=r[i]*r[i]; } return sqrt(se/(sr>0?sr:1)); };
    double eQ=rr(hq,gQ),eK=rr(hk,gK),eV=rr(hv,gV);
    // NaN не проходит НИ ОДНО сравнение, поэтому eQ<tol на NaN давал false... но при -ffast-math
    // компилятор волен считать NaN невозможным и свернуть сравнение в true. Так гейт печатал
    // «dQ -nan ... PASS» на заведомо сломанном ядре. Проверяем конечность явно, через диапазон.
    auto finite=[](double x){ return x>-1e300 && x<1e300; };
    bool pass = finite(eQ)&&finite(eK)&&finite(eV) && eQ<tol&&eK<tol&&eV<tol;
    // детерминизм
    fable_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> hq2(qb.size()); HIP_CHECK(hipMemcpy(hq2.data(),ddQ,qb.size(),hipMemcpyDeviceToHost));
    bool det=(memcmp(hq.data(),hq2.data(),qb.size())==0);
    if(print) printf("  [%-13s %-4s] dQ %.2e dK %.2e dV %.2e  %s%s\n", c.name,
        mode==BW_F16?"f16":mode==BW_BF16?"bf16":"f32", eQ,eK,eV, pass?"PASS":"FAIL", det?"":" [НЕ ДЕТЕРМ]");
    hipFree(dQp);hipFree(dKp);hipFree(dVp);hipFree(dOp);hipFree(ddOp);hipFree(ddQ);hipFree(ddK);hipFree(ddV);hipFree(dl);hipFree(ds);if(dal)hipFree(dal);
    return (pass&&det)?0:1;
}

// print=true — печатает строку бенча; всегда возвращает полное время и раскладывает по стадиям.
static float bench(int mode, bool print=true, float* out_kv=nullptr, float* out_q=nullptr){
    const int B=2,H=16,S=2048,D=128; const float scale=1.0f/sqrtf((float)D);
    std::vector<float> Q(B*H*S*D); for(auto&x:Q)x=rndf(); auto K=Q,V=Q,dO=Q,O=Q; std::vector<float> LSE(B*H*S,0.f);
    std::vector<uint8_t> qb,kb,vb,ob,dob; pack(qb,Q,mode);pack(kb,K,mode);pack(vb,V,mode);pack(ob,O,mode);pack(dob,dO,mode);
    void *dQp,*dKp,*dVp,*dOp,*ddOp,*ddQ,*ddK,*ddV; float*dl,*ds;
    HIP_CHECK(hipMalloc(&dQp,qb.size()));HIP_CHECK(hipMalloc(&dKp,kb.size()));HIP_CHECK(hipMalloc(&dVp,vb.size()));HIP_CHECK(hipMalloc(&dOp,ob.size()));HIP_CHECK(hipMalloc(&ddOp,dob.size()));
    HIP_CHECK(hipMalloc(&ddQ,qb.size()));HIP_CHECK(hipMalloc(&ddK,kb.size()));HIP_CHECK(hipMalloc(&ddV,vb.size()));HIP_CHECK(hipMalloc(&dl,LSE.size()*4));HIP_CHECK(hipMalloc(&ds,LSE.size()*4));
    HIP_CHECK(hipMemcpy(dQp,qb.data(),qb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dKp,kb.data(),kb.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dVp,vb.data(),vb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dOp,ob.data(),ob.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddOp,dob.data(),dob.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemset(dl,0,LSE.size()*4));
    auto sD=[&](int S2){ return Str{(long long)H*S2*D,(long long)S2*D,(long long)D}; };
    BwdParams pr{}; pr.Q=dQp;pr.K=dKp;pr.V=dVp;pr.O=dOp;pr.dO=ddOp;pr.dQ=ddQ;pr.dK=ddK;pr.dV=ddV;pr.lse=dl;pr.dsum=ds;
    pr.qs=sD(S);pr.ks=sD(S);pr.vs=sD(S);pr.os=sD(S);pr.dos=sD(S);pr.dqs=sD(S);pr.dks=sD(S);pr.dvs=sD(S);
    pr.B=B;pr.Hq=H;pr.Hk=H;pr.SQ=S;pr.SK=S;pr.D=D;pr.gqa=1;pr.scale=scale;pr.softcap=0;pr.alibi=nullptr;pr.causal=1;pr.win_left=-1;pr.win_right=-1;
    pr.dropout=0;pr.shQ=host_shift(Q,mode);pr.shK=host_shift(K,mode);pr.shV=host_shift(V,mode);pr.shdO=host_shift(dO,mode);
    auto timeit=[&](int stage,int probe=0){
        g_bwd_stage=stage; pr.probe=probe;
        for(int i=0;i<3;i++) fable_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0,t1; HIP_CHECK(hipEventCreate(&t0));HIP_CHECK(hipEventCreate(&t1));
        const int IT=20; HIP_CHECK(hipEventRecord(t0));
        for(int i=0;i<IT;i++) fable_attn_backward_launch(pr,mode,0);
        HIP_CHECK(hipEventRecord(t1));HIP_CHECK(hipEventSynchronize(t1));
        float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,t0,t1));
        hipEventDestroy(t0);hipEventDestroy(t1); g_bwd_stage=0; pr.probe=0; return ms/IT; };
    float ms=timeit(0), mp=timeit(1), mkv=timeit(2), mq=timeit(3);
    if(out_kv)*out_kv=mkv; if(out_q)*out_q=mq;
    double fwd=2.0*2.0*B*H*(double)S*S*D*0.5; double flops=2.5*fwd;
    if(print){
    float pkv=timeit(2,1), pq=timeit(3,1);   // зонд: те же ядра без второго GEMM
    printf("  bench %-4s B%dH%dS%dD%d causal: %.3f ms -> %.2f TFLOP/s (эфф. useful)\n",
        mode==BW_F16?"f16":mode==BW_BF16?"bf16":"f32",B,H,S,D,ms,flops/(ms*1e-3)/1e12);
    printf("        по стадиям: preprocess %.3f мс (%.1f%%) | dK/dV %.3f мс (%.1f%%) | dQ %.3f мс (%.1f%%)\n",
        mp,100*mp/ms, mkv,100*mkv/ms, mq,100*mq/ms);
    printf("        внутри: dK/dV = fill_pd %.3f + GEMM2 %.3f | dQ = fill_pd %.3f + GEMM2 %.3f\n",
        pkv, mkv-pkv, pq, mq-pq); }
    hipFree(dQp);hipFree(dKp);hipFree(dVp);hipFree(dOp);hipFree(ddOp);hipFree(ddQ);hipFree(ddK);hipFree(ddV);hipFree(dl);hipFree(ds);
    return ms;
}

// Стенд: перебрать ВСЕ варианты, каждый проверить оракулом и померить. Ничего не выбираем
// рассуждением — выбирает таблица.
static void sweep(int mode){
    printf("\n=== СТЕНД backward (%s), B2 H16 S2048 D128 causal ===\n", mode==BW_F16?"fp16":"bf16");
    printf("  %-28s %9s %9s %8s %9s %9s  %s\n","вариант","всего","dK/dV","dQ","TFLOP/s","к базе","корректность");
    { char bar[104]; memset(bar,'-',103); bar[103]=0; printf("  %s\n",bar); }
    float base=-1;
    for (int v=0; v<BWD_NVARIANT; v++){
        if (!bwd_variant_buildable(v,mode)){
            printf("  %-28s пропуск: не влезает в LDS или тайл несовместим с упаковкой\n", bwd_variant_name(v));
            continue;
        }
        g_bwd_variant=v;
        // корректность на тех же случаях, что и гейт
        Case cs[]={{1,4,4,128,128,128,true, -1,-1,0.f,false,0.f,"sweep-causal"},
                   {1,4,4,128,128,128,false, -1,-1,0.f,false,0.f,"sweep-full"},
                   {2,8,2,128,128,128,true, -1,-1,0.f,false,0.f,"sweep-gqa"}};
        int bad=0; for(auto&c:cs) bad+=run_case(c,mode,mode==BW_F16?2.5e-3:6.0e-3,false);
        float mkv=0, mq=0;
        float ms=bench(mode,false,&mkv,&mq);
        if (base<0) base=ms;
        double fwd=2.0*2.0*2*16*(double)2048*2048*128*0.5, flops=2.5*fwd;
        printf("  %-28s %7.3f мс %7.3f %7.3f мс %8.2f  %+6.1f%%  %s\n",
               bwd_variant_name(v), ms, mkv, mq, flops/(ms*1e-3)/1e12,
               base>0?100.0*(base-ms)/base:0.0, bad?"ПРОВАЛ":"OK");
    }
    g_bwd_variant=0;
}

int main(int argc,char** argv){
    bool do_bench=(argc>1&&strcmp(argv[1],"--bench")==0);
    bool do_sweep=(argc>1&&strcmp(argv[1],"--sweep")==0);
    Case cs[]={
        {1,4,4, 64, 64,64,false, -1,-1,  0.f,false,0.f,"base"},
        {1,4,4, 64, 64,64,true, -1,-1,  0.f,false,0.f,"causal"},
        {2,8,2,128,128,64,true, -1,-1,  0.f,false,0.f,"causal+gqa"},
        {1,4,4,100,100,64,true, -1,-1,  0.f,false,0.f,"causal-tail"},
        {1,4,4, 96,128,64,true, -1,-1,  0.f,false,0.f,"cross SQ<SK"},
        {1,4,4,128, 96,64,true, -1,-1,  0.f,false,0.f,"cross SQ>SK"},
        {1,4,4,128,128,64,false,47,0,  0.f,false,0.f,"window (47,0)"},
        {1,4,4,128,128,64,false,24,24, 0.f,false,0.f,"window (24,24)"},
        {1,4,4,128,128,64,false,-1, 0,  0.f,false,0.f,"window (-1,0)=causal"},
        {1,4,4, 64, 64,192,false,-1,-1, 0.f,false,0.f,"D192"},
        {1,4,4, 64, 64,256,true, -1,-1, 0.f,false,0.f,"D256"},
        {1,4,4, 64, 64,160,true, -1,-1, 0.f,false,0.f,"D160"},
        {1,4,4, 64, 64,224,false,-1,-1, 0.f,false,0.f,"D224"},
        {1,4,4,128,128,64,false, -1,-1, 30.f,false,0.f,"softcap"},
        {1,4,4,128,128,64,true, -1,-1,  0.f,true, 0.f,"alibi"},
        {1,4,4,128,128,64,false, -1,-1,  0.f,false,0.2f,"dropout"},
        {1,4,4, 64, 64,16,false, -1,-1,  0.f,false,0.f,"D16"},
        {1,4,4, 64, 64,128,false, -1,-1, 0.f,false,0.f,"D128"},
        {1,4,4, 80, 80,96,true, -1,-1,  0.f,false,0.f,"D96"},
    };
    int fails=0;
    if(!do_sweep){   // в режиме стенда корректность проверяется внутри него, на каждом варианте
        printf("=== BACKWARD fp64-оракул (rel_rms dQ/dK/dV) ===\n");
        for(auto&c:cs){ fails+=run_case(c,BW_F16,2.5e-3,true); fails+=run_case(c,BW_BF16,6.0e-3,true); fails+=run_case(c,BW_F32,1.0e-4,true); }
        printf(fails?"\n*** ПАДЕНИЙ: %d ***\n":"\nВСЁ ЗЕЛЁНОЕ\n",fails);
    }
    if(do_bench){ printf("\n=== БЕНЧ (один короткий прогон) ===\n"); bench(BW_F16); bench(BW_BF16); }
    if(do_sweep){ sweep(BW_F16); sweep(BW_BF16); }
    return fails;
}
#endif
