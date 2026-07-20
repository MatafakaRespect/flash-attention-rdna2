// stream.cu — забор текущего torch-стрима в ХИПИФАЕМОЙ TU (hipcc понимает ext_vector_type,
// MSVC — нет, поэтому dispatch.cpp (host) сюда не лезет). Пишем CUDA-написание — hipify торча
// заменит at::cuda::getCurrentCUDAStream -> HIP-масккарад (документированный путь, как в примере).
#include <hip/hip_runtime.h>   // ПЕРВЫМ: определяет float2/vector-types до hip_complex.h (иначе RTC-сборка падает)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void* fable_current_stream(long long dev){
    return (void*)at::cuda::getCurrentCUDAStream((c10::DeviceIndex)dev).stream();
}
