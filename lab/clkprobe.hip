// clkprobe.hip — РЕАЛЬНАЯ частота шейдеров под нагрузкой и настоящий потолок v_dot2.
// Зачем: весь план опирается на «сырой темп v_dot2 ~36.9 TFLOP/s», посчитанный из паспортных
// 2.0 ГГц. Если под плотной fp16-нагрузкой карта сбрасывает клок, то этот потолок вымышленный,
// и все проценты утилизации считаются не от того числа.
//
// Как мерим: s_memtime считает ТАКТЫ ШЕЙДЕРА, s_memrealtime — константный таймер (его частота
// известна из hipDeviceAttributeWallClockRate). Отношение даёт фактическую частоту.
// Нагрузка — длинная цепочка независимых v_dot2_f32_f16 из РЕГИСТРОВ: ни LDS, ни глобалки,
// то есть чистый счётный темп, тот самый MAC-пол.
//
// Сборка: hipcc clkprobe.hip -o clkprobe.exe --offload-arch=gfx1030 -O3 -ffast-math -std=c++17
#include "../kernels/fa_common.hpp"

#define ACC 32          // независимых аккумуляторов (ILP, чтобы не упереться в латентность)
#define INNER 512       // итераций внутреннего цикла

__global__ __launch_bounds__(256)
void mac_probe(float* out, unsigned long long* clk, unsigned long long* wall, int iters)
{
    half2_t a[ACC], b[ACC];
    #pragma unroll
    for (int i=0;i<ACC;i++){
        a[i]=half2_t{(_Float16)(0.5f+i*0.001f),(_Float16)(0.25f+i*0.001f)};
        b[i]=half2_t{(_Float16)(0.125f+i*0.001f),(_Float16)(0.0625f+i*0.001f)};
    }
    float acc[ACC];
    #pragma unroll
    for (int i=0;i<ACC;i++) acc[i]=0.0f;

    __syncthreads();
    unsigned long long c0=__builtin_amdgcn_s_memtime();
    unsigned long long w0=__builtin_amdgcn_s_memrealtime();

    for (int it=0; it<iters; it++){
        #pragma unroll
        for (int k=0;k<INNER;k++){
            #pragma unroll
            for (int i=0;i<ACC;i++) acc[i]=__builtin_amdgcn_fdot2(a[i],b[i],acc[i],false);
        }
    }

    unsigned long long c1=__builtin_amdgcn_s_memtime();
    unsigned long long w1=__builtin_amdgcn_s_memrealtime();
    __syncthreads();

    float s=0;
    #pragma unroll
    for (int i=0;i<ACC;i++) s+=acc[i];
    if (threadIdx.x==0){
        out [blockIdx.x]=s;
        clk [blockIdx.x]=c1-c0;
        wall[blockIdx.x]=w1-w0;
    }
}

// КОНТРОЛЬ: тот же цикл, но обычный fp32-FMA (2 FLOP/лейн/такт, ровно половина от dot2).
// Если контроль выходит на свои 100%, значит методика замера верна и 76% — это темп самого
// v_dot2. Если контроль тоже даёт 76% — врёт замер, а не железо.
__global__ __launch_bounds__(256)
void fma_probe(float* out, unsigned long long* clk, unsigned long long* wall, int iters)
{
    float a[ACC], b[ACC], acc[ACC];
    #pragma unroll
    for (int i=0;i<ACC;i++){ a[i]=0.5f+i*0.001f; b[i]=0.125f+i*0.001f; acc[i]=0.0f; }
    __syncthreads();
    unsigned long long c0=__builtin_amdgcn_s_memtime(), w0=__builtin_amdgcn_s_memrealtime();
    for (int it=0; it<iters; it++){
        #pragma unroll
        for (int k=0;k<INNER;k++){
            #pragma unroll
            for (int i=0;i<ACC;i++) acc[i]=__builtin_fmaf(a[i],b[i],acc[i]);
        }
    }
    unsigned long long c1=__builtin_amdgcn_s_memtime(), w1=__builtin_amdgcn_s_memrealtime();
    __syncthreads();
    float s=0;
    #pragma unroll
    for (int i=0;i<ACC;i++) s+=acc[i];
    if (threadIdx.x==0){ out[blockIdx.x]=s; clk[blockIdx.x]=c1-c0; wall[blockIdx.x]=w1-w0; }
}

// ГЛАВНЫЙ ЗОНД: тот же счётный цикл, но с 40 КБ LDS на воркгруп — ровно та занятость,
// на которой работает боевой forward (LDS 64 КБ => один воркгруп на CU => 4 волны на SIMD).
// Он отвечает на вопрос: 17.1 TFLOP/s у forward — это его память или это стена занятости?
__global__ __launch_bounds__(256)
void mac_probe_lowocc(float* out, unsigned long long* clk, unsigned long long* wall, int iters)
{
    __shared__ float hog[10240];            // 40 КБ => 1 воркгруп на CU, 4 волны на SIMD
    hog[threadIdx.x]=(float)threadIdx.x;
    __syncthreads();
    half2_t a[ACC], b[ACC];
    #pragma unroll
    for (int i=0;i<ACC;i++){
        a[i]=half2_t{(_Float16)(0.5f+i*0.001f),(_Float16)(0.25f+i*0.001f)};
        b[i]=half2_t{(_Float16)(0.125f+i*0.001f),(_Float16)(0.0625f+i*0.001f)};
    }
    float acc[ACC];
    #pragma unroll
    for (int i=0;i<ACC;i++) acc[i]=0.0f;
    __syncthreads();
    unsigned long long c0=__builtin_amdgcn_s_memtime(), w0=__builtin_amdgcn_s_memrealtime();
    for (int it=0; it<iters; it++){
        #pragma unroll
        for (int k=0;k<INNER;k++){
            #pragma unroll
            for (int i=0;i<ACC;i++) acc[i]=__builtin_amdgcn_fdot2(a[i],b[i],acc[i],false);
        }
    }
    unsigned long long c1=__builtin_amdgcn_s_memtime(), w1=__builtin_amdgcn_s_memrealtime();
    __syncthreads();
    float s=hog[threadIdx.x]*0.0f;           // чтобы LDS не выкинули
    #pragma unroll
    for (int i=0;i<ACC;i++) s+=acc[i];
    if (threadIdx.x==0){ out[blockIdx.x]=s; clk[blockIdx.x]=c1-c0; wall[blockIdx.x]=w1-w0; }
}

int main(int argc,char** argv){
    int iters = argc>1? atoi(argv[1]) : 40;
    hipDeviceProp_t pr; HIP_CHECK(hipGetDeviceProperties(&pr,0));
    int wallKHz=0;
    HIP_CHECK(hipDeviceGetAttribute(&wallKHz,hipDeviceAttributeWallClockRate,0));
    const int CU = pr.multiProcessorCount;            // на gfx1030 это WGP (36) => 72 CU
    // Воркгрупов надо МНОГО: они не влезают на карту разом и идут кругами, и если кругов
    // всего два, недогруз последнего круга съедает проценты и врёт про потолок.
    const int WG = (argc>2? atoi(argv[2]) : 40)*CU;
    printf("GPU %s | multiProcessorCount=%d | паспортный clockRate=%d МГц | wall clock=%d кГц\n",
           pr.gcnArchName, CU, pr.clockRate/1000, wallKHz);

    float* dout; unsigned long long *dclk,*dwall;
    HIP_CHECK(hipMalloc(&dout,WG*4)); HIP_CHECK(hipMalloc(&dclk,WG*8)); HIP_CHECK(hipMalloc(&dwall,WG*8));

    std::vector<unsigned long long> hclk(WG),hwall(WG);
    double lanes = (double)CU*2.0*2.0*32.0;   // 36 WGP * 2 CU * 2 SIMD32 * 32 лейна = 4608

    auto run=[&](const char* name, void(*kern)(float*,unsigned long long*,unsigned long long*,int),
                 double flop_per_instr){
        // прогрев: нужен установившийся режим, а не первый запуск на холодной карте
        hipLaunchKernelGGL(kern,dim3(WG),dim3(256),0,0,dout,dclk,dwall,4);
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0,t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
        hipLaunchKernelGGL(kern,dim3(WG),dim3(256),0,0,dout,dclk,dwall,iters);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,t0,t1));
        HIP_CHECK(hipMemcpy(hclk.data(),dclk,WG*8,hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hwall.data(),dwall,WG*8,hipMemcpyDeviceToHost));
        double sumc=0,sumw=0;
        for(int i=0;i<WG;i++){ sumc+=(double)hclk[i]; sumw+=(double)hwall[i]; }
        double sclk = (sumc/sumw)*(wallKHz/1000.0);
        double flop = (double)WG*256.0*(double)iters*INNER*ACC*flop_per_instr;
        double tf = flop/(ms*1e-3)/1e12;
        double peak = lanes*flop_per_instr*sclk*1e6/1e12;
        printf("  %-22s %7.2f мс | частота %.0f МГц | %5.1f TFLOP/s из %5.1f возможных = %.0f%%\n",
               name, ms, sclk, tf, peak, 100.0*tf/peak);
        hipEventDestroy(t0); hipEventDestroy(t1);
    };

    printf("\nчистые счётные циклы из регистров (ни LDS, ни глобалки, %d независимых аккумуляторов):\n",ACC);
    run("v_dot2 (полная зан.)", mac_probe, 4.0);          // 2 MAC на инструкцию
    run("v_fma_f32 (контроль)", fma_probe, 2.0);          // 1 MAC на инструкцию
    run("v_dot2 при 40КБ LDS", mac_probe_lowocc, 4.0);    // занятость боевого forward
    hipFree(dout);hipFree(dclk);hipFree(dwall);
    return 0;
}
