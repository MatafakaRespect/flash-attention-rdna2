import os,sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
SHIM=os.path.abspath(os.environ.get("HIPSHIM") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc", "hipshim.h"))
which=os.environ.get("PROBE_KERN","none")
src=["probe.cpp"]
if which in ("fwd","both"): src.append("../csrc/_fwd_shim.cu")
if which in ("bwd","both"): src.append("../csrc/_bwd_shim.cu")
setup(name="fable_probe", ext_modules=[CUDAExtension("fable_probe._P",src,
    extra_compile_args={"cxx":["/std:c++17"],"nvcc":["-include",SHIM,"-std=c++17","-O3","-ffast-math"]})],
    cmdclass={"build_ext":BuildExtension})
