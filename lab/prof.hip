// prof.hip — послойный профиль текущего ядра fa_forward.hip (BR=128, BC=64, HD=128, causal, fp16).
// Метод: собираем одно и то же ядро N раз, каждый билд включает на одну стадию больше.
// Дельта времени между соседними стадиями = цена этой стадии в µs. Плюс два отдельных зонда:
// MAC-only (операнды подняты в регистры — чистый темп VALU на нашей смеси инструкций) и
// LDS-only (те же чтения LDS без fdot2 — чистая цена трафика LDS).
//
// Сборка: hipcc prof.hip -o prof.exe --offload-arch=gfx1030 -O3 -ffast-math -std=c++17 -Wno-unused-value
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

#define HIP_CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(e),__LINE__);exit(1);}}while(0)
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

template<int CTRL>
__device__ __forceinline__ float dpp_mov_f32(float x){
  union { float f; int i; } a,b; a.f=x;
  b.i=__builtin_amdgcn_update_dpp(0,a.i,CTRL,0xF,0xF,false); return b.f;
}
__device__ __forceinline__ float red16_max(float x){
  x=fmaxf(x,dpp_mov_f32<0xB1>(x)); x=fmaxf(x,dpp_mov_f32<0x4E>(x));
  x=fmaxf(x,dpp_mov_f32<0x141>(x)); x=fmaxf(x,dpp_mov_f32<0x140>(x)); return x;
}
__device__ __forceinline__ float red16_sum(float x){
  x+=dpp_mov_f32<0xB1>(x); x+=dpp_mov_f32<0x4E>(x);
  x+=dpp_mov_f32<0x141>(x); x+=dpp_mov_f32<0x140>(x); return x;
}

// стадии (кумулятивные)
#define ST_LOAD   1   // глобалка -> LDS (K,V) + барьеры
#define ST_GEMM1  2   // + QK^T
#define ST_SOFTM  3   // + маска/max/exp2/sum/rescale
#define ST_PSTORE 4   // + запись P в LDS + барьер
#define ST_FULL   5   // + P·V  (== полное ядро)
// зонды
#define PR_MACONLY 6  // GEMM1+GEMM2 с операндами из регистров (LDS почти не трогаем)
#define PR_LDSONLY 7  // чтения LDS как в GEMM1, но без fdot2
#define PR_NOBAR   8  // ПОЛНОЕ ядро, но без барьеров внутри тайла (результат неверен —
                      // это зонд: показывает, сколько стоит сама сериализация стадий)
#define PR_NOLOAD  9  // полное ядро без загрузки K/V из глобалки (тайл считается по мусору в LDS)

struct P { const _Float16 *Q,*K,*V; _Float16* O; int S,gqa; float scale; };

template<int HD,int BR,int BC,int STAGE>
__global__ __launch_bounds__(256)
void prof_k(P p)
{
  constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
  constexpr float LOG2E=1.442695041f;
  const int t=threadIdx.x, tx=t&15, ty=t>>4;
  const int h=blockIdx.y, kvh=h/p.gqa, S=p.S;
  const int qb=blockIdx.x*BR; if(qb>=S) return;
  const float scale2=p.scale*LOG2E;

  constexpr int QS_B=HH*BR*4, KS_B=HH*BC*4, VS_B=BCH*HD*4;
  __shared__ char pool[QS_B+KS_B+VS_B];
  half2_t* Qs=(half2_t*)pool; half2_t* Ks=(half2_t*)(pool+QS_B); half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
  _Float16* Ps=(_Float16*)Ks;                 // Ps алиасит Ks (K мёртв к этому моменту)
  half2_t*  Pp=(half2_t*)Ks;
  #define QSA(dh,q)  Qs[(dh)*BR+(q)]
  #define KSA(dh,kc) Ks[(dh)*BC+(kc)]
  #define VSA(kh,d)  Vs[(kh)*HD+(d)]
  #define PSA(kc,q)  Ps[(kc)*BR+(q)]
  #define PPA(kh,q)  Pp[(kh)*BR+(q)]

  const _Float16* Qb=p.Q+(long long)h*S*HD;
  const _Float16* Kb=p.K+(long long)kvh*S*HD;
  const _Float16* Vb=p.V+(long long)kvh*S*HD;

  for(int e=t;e<BR*(HD/8);e+=256){ int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q;
    half8_t a = sq<S ? *(const half8_t*)(Qb+(long long)sq*HD+dg*8) : half8_t{0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int m=0;m<4;m++) QSA(dg*4+m,q)=half2_t{a[2*m],a[2*m+1]}; }

  float acco[QPT][DPT], rm[QPT], rl[QPT];
  #pragma unroll
  for(int i=0;i<QPT;i++){ rm[i]=-1e30f; rl[i]=0.f;
    #pragma unroll
    for(int e=0;e<DPT;e++) acco[i][e]=0.f; }
  __syncthreads();

  const int kmax = (qb+BR-1<S-1)? qb+BR-1 : S-1;     // causal, SQ==SK

  for(int kb=0; kb<=kmax; kb+=BC){
    const bool diag=((kb+BC-1)>=qb);
    if constexpr(STAGE!=PR_NOBAR) __syncthreads();
    // ---------- стадия 1: глобалка -> LDS ----------
    if constexpr(STAGE!=PR_NOLOAD){
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
    }
    if constexpr(STAGE!=PR_NOBAR) __syncthreads();

    if constexpr(STAGE==ST_LOAD){          // потребляем LDS, чтобы запись не выкосило
      acco[0][0]+=(float)KSA(tx,ty)[0]+(float)VSA(tx,ty)[0];
      continue;
    }

    // ---------- зонд: только чтения LDS, без MAC ----------
    if constexpr(STAGE==PR_LDSONLY){
      float sink=0.f;
      #pragma unroll 4
      for(int dh=0;dh<HH;dh++){
        half2_t kv[KPT], qv[QPT];
        #pragma unroll
        for(int j=0;j<KPT;j++) kv[j]=KSA(dh,(j>>1)*32+2*tx+(j&1));
        #pragma unroll
        for(int i=0;i<QPT;i++) qv[i]=QSA(dh,ty*QPT+i);
        sink+=(float)kv[0][0]+(float)qv[0][0];
      }
      #pragma unroll 4
      for(int kh=0;kh<BCH;kh++){
        half2_t vv[DPT];
        #pragma unroll
        for(int e=0;e<DPT;e++) vv[e]=VSA(kh,e*16+tx);
        sink+=(float)vv[0][0];
      }
      acco[0][0]+=sink;
      continue;
    }

    // ---------- зонд: только MAC, операнды из регистров ----------
    if constexpr(STAGE==PR_MACONLY){
      half2_t kv[KPT], qv[QPT], vv[DPT], pv[QPT];
      #pragma unroll
      for(int j=0;j<KPT;j++) kv[j]=KSA(0,tx*KPT+j);
      #pragma unroll
      for(int i=0;i<QPT;i++){ qv[i]=QSA(0,ty*QPT+i); pv[i]=qv[i]; }
      #pragma unroll
      for(int e=0;e<DPT;e++) vv[e]=VSA(0,tx*DPT+e);
      float accs[QPT][KPT];
      #pragma unroll
      for(int i=0;i<QPT;i++)
        #pragma unroll
        for(int j=0;j<KPT;j++) accs[i][j]=0.f;
      #pragma unroll 4
      for(int dh=0;dh<HH;dh++)                    // столько же fdot2, сколько в GEMM1
        #pragma unroll
        for(int i=0;i<QPT;i++)
          #pragma unroll
          for(int j=0;j<KPT;j++) accs[i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[i][j],false);
      #pragma unroll 4
      for(int kh=0;kh<BCH;kh++)                   // столько же fdot2, сколько в GEMM2
        #pragma unroll
        for(int i=0;i<QPT;i++)
          #pragma unroll
          for(int e=0;e<DPT;e++) acco[i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[i][e],false);
      #pragma unroll
      for(int i=0;i<QPT;i++)
        #pragma unroll
        for(int j=0;j<KPT;j++) acco[i][j]+=accs[i][j];
      continue;
    }

    // ---------- стадия 2: GEMM1 ----------
    float accs[QPT][KPT];
    #pragma unroll
    for(int i=0;i<QPT;i++)
      #pragma unroll
      for(int j=0;j<KPT;j++) accs[i][j]=0.f;
    #pragma unroll 4
    for(int dh=0;dh<HH;dh++){
      half2_t kv[KPT], qv[QPT];
      #pragma unroll
      for(int j=0;j<KPT;j++) kv[j]=KSA(dh,(j>>1)*32+2*tx+(j&1));
      #pragma unroll
      for(int i=0;i<QPT;i++) qv[i]=QSA(dh,ty*QPT+i);
      #pragma unroll
      for(int i=0;i<QPT;i++)
        #pragma unroll
        for(int j=0;j<KPT;j++) accs[i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[i][j],false);
    }

    if constexpr(STAGE==ST_GEMM1){
      #pragma unroll
      for(int i=0;i<QPT;i++)
        #pragma unroll
        for(int e=0;e<DPT;e++) acco[i][e]+=accs[i][e&(KPT-1)];
      continue;
    }
    if constexpr(STAGE!=PR_NOBAR) __syncthreads();   // K мёртв -> Ps алиас

    // ---------- стадия 3: softmax ----------
    #pragma unroll
    for(int i=0;i<QPT;i++){
      const int qq=qb+ty*QPT+i;
      float sj[KPT], lmax=-1e30f;
      #pragma unroll
      for(int j=0;j<KPT;j++){ int key=kb+tx*KPT+j; float sv=accs[i][j]*scale2;
        if(diag&&key>qq) sv=-1e30f; sj[j]=sv; lmax=fmaxf(lmax,sv); }
      lmax=red16_max(lmax);
      float mnew=fmaxf(rm[i],lmax);
      float corr=(mnew>rm[i])?__builtin_amdgcn_exp2f(rm[i]-mnew):1.0f;
      float lsum=0.f;
      #pragma unroll
      for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
      lsum=red16_sum(lsum);
      rl[i]=rl[i]*corr+lsum; rm[i]=mnew;
      if(corr!=1.0f){
        #pragma unroll
        for(int e=0;e<DPT;e++) acco[i][e]*=corr;
      }
      if constexpr(STAGE==ST_SOFTM){
        #pragma unroll
        for(int j=0;j<KPT;j++) acco[i][j]+=sj[j];      // потребляем P без записи в LDS
      } else {
        #pragma unroll
        for(int j=0;j<KPT;j+=2) PPA((j>>1)*16+tx,ty*QPT+i)=half2_t{(_Float16)sj[j],(_Float16)sj[j+1]};
      }
    }
    if constexpr(STAGE==ST_SOFTM) continue;
    if constexpr(STAGE!=PR_NOBAR) __syncthreads();

    if constexpr(STAGE==ST_PSTORE){
      acco[0][0]+=(float)PSA(tx,ty);
      continue;
    }

    // ---------- стадия 5: GEMM2 ----------
    #pragma unroll 4
    for(int kh=0;kh<BCH;kh++){
      half2_t pv[QPT], vv[DPT];
      #pragma unroll
      for(int i=0;i<QPT;i++) pv[i]=PPA(kh,ty*QPT+i);
      #pragma unroll
      for(int e=0;e<DPT;e++) vv[e]=VSA(kh,e*16+tx);
      #pragma unroll
      for(int i=0;i<QPT;i++)
        #pragma unroll
        for(int e=0;e<DPT;e++) acco[i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[i][e],false);
    }
  }

  _Float16* Ob=p.O+(long long)h*S*HD;
  #pragma unroll
  for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=S) continue;
    float inv=((STAGE==ST_FULL||STAGE==PR_NOBAR||STAGE==PR_NOLOAD)&&rl[i]>0.f)?1.f/rl[i]:1.f;
    #pragma unroll
    for(int e=0;e<DPT;e++) Ob[(long long)qq*HD+tx*DPT+e]=(_Float16)(acco[i][e]*inv); }
  #undef QSA
  #undef KSA
  #undef VSA
  #undef PSA
  #undef PPA
}

__global__ void empty_k(P p){ if(threadIdx.x==1024) p.O[0]=(_Float16)0; }

// ---------------------------------------------------------------------------
static const int HD=128, BR=128, BC=64;
static P g_p; static dim3 g_grid;

template<int STAGE>
static float time_stage(int iters){
  hipLaunchKernelGGL((prof_k<HD,BR,BC,STAGE>),g_grid,dim3(256),0,0,g_p);
  HIP_CHECK(hipDeviceSynchronize());
  hipError_t e=hipGetLastError(); if(e!=hipSuccess){ printf("ERR stage %d: %s\n",STAGE,hipGetErrorString(e)); return 0; }
  for(int i=0;i<3;i++) hipLaunchKernelGGL((prof_k<HD,BR,BC,STAGE>),g_grid,dim3(256),0,0,g_p);
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
  hipEventRecord(t0);
  for(int i=0;i<iters;i++) hipLaunchKernelGGL((prof_k<HD,BR,BC,STAGE>),g_grid,dim3(256),0,0,g_p);
  hipEventRecord(t1); hipEventSynchronize(t1);
  float ms=0; hipEventElapsedTime(&ms,t0,t1);
  return ms/iters;
}

int main(int argc,char** argv){
  int S   = argc>1? atoi(argv[1]) : 4096;
  int gqa = argc>2? atoi(argv[2]) : 4;
  const int NH=32, iters=20;
  hipDeviceProp_t pr; HIP_CHECK(hipGetDeviceProperties(&pr,0));
  printf("GPU %s  CU=%d  %d MHz | S=%d NH=%d GQA=%d HD=%d BR=%d BC=%d causal fp16\n\n",
         pr.gcnArchName,pr.multiProcessorCount,pr.clockRate/1000,S,NH,gqa,HD,BR,BC);

  size_t nQ=(size_t)NH*S*HD, nK=(size_t)(NH/gqa)*S*HD;
  std::vector<_Float16> hq(nQ,(_Float16)0.02f), hk(nK,(_Float16)0.02f);
  void *dQ,*dK,*dV,*dO;
  HIP_CHECK(hipMalloc(&dQ,nQ*2)); HIP_CHECK(hipMalloc(&dO,nQ*2));
  HIP_CHECK(hipMalloc(&dK,nK*2)); HIP_CHECK(hipMalloc(&dV,nK*2));
  HIP_CHECK(hipMemcpy(dQ,hq.data(),nQ*2,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK,hk.data(),nK*2,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV,hk.data(),nK*2,hipMemcpyHostToDevice));
  g_p=P{(const _Float16*)dQ,(const _Float16*)dK,(const _Float16*)dV,(_Float16*)dO,S,gqa,1.f/sqrtf((float)HD)};
  g_grid=dim3((S+BR-1)/BR,NH,1);

  // режим цели для RGP: только полное ядро, немного диспатчей, чистый выход
  if(argc>3 && argv[3][0]=='r'){
    printf("RGP-цель: 8 диспатчей полного ядра S=%d gqa=%d\n",S,gqa);
    for(int i=0;i<8;i++){ hipLaunchKernelGGL((prof_k<HD,BR,BC,ST_FULL>),g_grid,dim3(256),0,0,g_p);
      HIP_CHECK(hipDeviceSynchronize()); }
    printf("готово\n");
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return 0;
  }

  // launch overhead
  hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
  hipLaunchKernelGGL(empty_k,g_grid,dim3(256),0,0,g_p); HIP_CHECK(hipDeviceSynchronize());
  hipEventRecord(t0); for(int i=0;i<iters;i++) hipLaunchKernelGGL(empty_k,g_grid,dim3(256),0,0,g_p);
  hipEventRecord(t1); hipEventSynchronize(t1); float ms_empty=0; hipEventElapsedTime(&ms_empty,t0,t1); ms_empty/=iters;

  float s1=time_stage<ST_LOAD>(iters);
  float s2=time_stage<ST_GEMM1>(iters);
  float s3=time_stage<ST_SOFTM>(iters);
  float s4=time_stage<ST_PSTORE>(iters);
  float s5=time_stage<ST_FULL>(iters);
  float pm=time_stage<PR_MACONLY>(iters);
  float pl=time_stage<PR_LDSONLY>(iters);
  float pb=time_stage<PR_NOBAR>(iters);
  float pn=time_stage<PR_NOLOAD>(iters);

  double fl=2.0*(double)NH*(double)S*(double)S*HD;   // causal: 4*H*S*S*D/2
  double tf_full=fl/(s5*1e-3)/1e12;

  // сколько всего тайлов обработано (для µs на тайл)
  double tiles=0; for(int qb=0;qb<S;qb+=BR){ int kmax=(qb+BR-1<S-1)?qb+BR-1:S-1; tiles+=(kmax/BC)+1; }
  tiles*=NH;

  auto row=[&](const char* nm,float cum,float prev){
    double d=cum-prev;
    printf("  %-26s %8.3f ms  %+8.3f ms  %5.1f%%   %7.2f ns/тайл\n",
           nm,cum,d,100.0*d/s5,d*1e6/tiles);
  };
  printf("КУМУЛЯТИВНЫЙ ПРОФИЛЬ (дельта = цена стадии)\n");
  printf("  %-26s %8s  %8s  %5s   %s\n","стадия","всего","дельта","доля","на KV-тайл");
  row("0 пустой запуск",ms_empty,0.f);
  row("1 + K/V глобалка->LDS",s1,ms_empty);
  row("2 + GEMM1 (QK^T)",s2,s1);
  row("3 + softmax",s3,s2);
  row("4 + запись P в LDS",s4,s3);
  row("5 + GEMM2 (P*V) = ПОЛНОЕ",s5,s4);
  printf("\n  ПОЛНОЕ ЯДРО: %.3f ms = %.1f TFLOP/s   (тайлов %.0f)\n",s5,tf_full,tiles);

  printf("\nЗОНДЫ (не кумулятивные, тот же счёт fdot2)\n");
  printf("  %-26s %8.3f ms   %5.1f%% от полного   -> %.1f TFLOP/s экв.\n",
         "MAC-only (из регистров)",pm,100.0*pm/s5,fl/(pm*1e-3)/1e12);
  printf("  %-28s %7.3f ms  %5.1f%% от полного   -> %5.1f TFLOP/s экв.\n",
         "полное БЕЗ БАРЬЕРОВ",pb,100.0*pb/s5,fl/(pb*1e-3)/1e12);
  printf("  %-28s %7.3f ms  %5.1f%% от полного   -> %5.1f TFLOP/s экв.\n",
         "полное БЕЗ загрузки K/V",pn,100.0*pn/s5,fl/(pn*1e-3)/1e12);
  printf("  %-26s %8.3f ms   %5.1f%% от полного\n","LDS-only (чтения, без MAC)",pl,100.0*pl/s5);
  printf("  %-26s %8.3f ms   %5.1f%% от полного\n","загрузка K/V (стадия 1)",s1-ms_empty,100.0*(s1-ms_empty)/s5);

  hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
  return 0;
}
