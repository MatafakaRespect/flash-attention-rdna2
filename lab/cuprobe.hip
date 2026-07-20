// cuprobe.hip — сколько параллельных слотов реально даёт карта нашему ядру.
// Метод: ядро с той же LDS-нагрузкой, что у нас (64 КБ => 1 воркгруп на слот) и фиксированной
// зависимой цепочкой VALU. Гоняем W воркгрупов и смотрим время: пока W <= числа слотов, время
// постоянно; как только W превысило — ступенька (вторая волна планирования).
// Ступеньки прямо показывают, сколько CU/WGP реально исполняют наш воркгруп одновременно.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define HIP_CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(e),__LINE__);exit(1);}}while(0)

template<int LDSKB>
__global__ __launch_bounds__(256) void spin_k(float* out,int iters){
  __shared__ char pool[LDSKB*1024];
  pool[threadIdx.x]=(char)threadIdx.x;                 // чтобы LDS не выкосили
  __syncthreads();
  float a=(float)threadIdx.x*1e-6f+1.0f, b=1.0000001f;
  for(int i=0;i<iters;i++) a=__builtin_fmaf(a,b,1e-7f); // зависимая цепочка: время ~ iters, не ILP
  if(a==12345.0f) out[0]=a+(float)pool[threadIdx.x];
}

template<int LDSKB>
static void sweep(const char* nm,int maxW){
  printf("\n%s (LDS %d КБ на воркгруп):\n",nm,LDSKB);
  float* d; HIP_CHECK(hipMalloc(&d,4));
  const int iters=200000;
  hipLaunchKernelGGL((spin_k<LDSKB>),dim3(1),dim3(256),0,0,d,iters); HIP_CHECK(hipDeviceSynchronize());
  float base=0;
  for(int W=1;W<=maxW;W++){
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for(int r=0;r<3;r++) hipLaunchKernelGGL((spin_k<LDSKB>),dim3(W),dim3(256),0,0,d,iters);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms=0; hipEventElapsedTime(&ms,t0,t1); ms/=3;
    if(W==1) base=ms;
    double ratio=ms/base;
    // печатаем только точки роста (ступеньки) и края
    static double prev; if(W==1) prev=1.0;
    if(W==1||ratio>prev*1.15||W==maxW){
      printf("  W=%-4d %7.3f ms   ×%.2f от W=1%s\n",W,ms,ratio, (ratio>prev*1.15&&W>1)?"   <-- СТУПЕНЬКА":"");
      prev=ratio;
    }
    hipEventDestroy(t0); hipEventDestroy(t1);
  }
  HIP_CHECK(hipFree(d));
}

int main(int argc,char** argv){
  hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p,0));
  printf("hipGetDeviceProperties говорит:\n");
  printf("  gcnArchName            = %s\n",p.gcnArchName);
  printf("  multiProcessorCount    = %d\n",p.multiProcessorCount);
  printf("  maxThreadsPerMultiProcessor = %d\n",p.maxThreadsPerMultiProcessor);
  printf("  sharedMemPerBlock      = %zu КБ\n",p.sharedMemPerBlock/1024);
  printf("  warpSize               = %d\n",p.warpSize);
  printf("  clockRate              = %d МГц\n",p.clockRate/1000);
  int maxW = argc>1? atoi(argv[1]) : 160;
  sweep<64>("ЯДРО FA (64 КБ LDS)",maxW);   // как наше ядро: 1 воркгруп на слот
  sweep<16>("лёгкий по LDS (16 КБ)",maxW); // 4 воркгрупа влезают по LDS
  return 0;
}
