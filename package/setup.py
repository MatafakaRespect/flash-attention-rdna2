import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Лекарство №1 сборки (PORT_NOTES_nvdiffrast.md): clang cuda_wrappers/complex форсит MSVC-шный
# <complex> в __host__ __device__ => signbit(long double) неоднозначен. C++17 обязателен.
# Шим лежит рядом (csrc/hipshim.h). Переопределяется переменной окружения HIPSHIM.
SHIM = os.environ.get("HIPSHIM") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "csrc", "hipshim.h")

setup(
    name="fable_attn",                  # имя НАШЕ
    version="0.2.0",   # держать в паре с pyproject.toml (версию берут оттуда)
    description="Точный FlashAttention на HIP-ядрах для gfx1030 (API один-в-один с flash-attn)",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "fable_attn._C",            # ВАЖНО: имя _C, а не fable_attn — иначе расширение конфликтует с одноимённым модулем
            ["csrc/dispatch.cpp", "csrc/stream.cu", "csrc/_fwd_shim.cu",
             "csrc/_bwd_shim.cu", "csrc/_dec_shim.cu"],
            extra_compile_args={
                "cxx": ["/std:c++17"],
                # --offload-arch торч подставляет из PYTORCH_ROCM_ARCH=gfx1030
                "nvcc": ["-include", SHIM, "-std=c++17", "-O3", "-ffast-math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
