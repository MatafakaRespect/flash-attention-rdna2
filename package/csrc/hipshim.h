// Shim: clang's cuda_wrappers/complex forces MSVC's <complex> to __host__ __device__.
// HIP's __clang_hip_cmath.h only provides device signbit(float)/signbit(double),
// so signbit(long double) has no exact device match -> ambiguous. Supply the gap.
#pragma once
__device__ bool signbit(long double __x) { return ::signbit((double)__x); }
