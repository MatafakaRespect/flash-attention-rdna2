#include <torch/extension.h>
#pragma comment(linker, "/alternatename:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z=__imp_??0Error@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z")
static torch::Tensor idf(torch::Tensor x){ return x; }
TORCH_LIBRARY(fable_probe, m){ m.def("idf(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(fable_probe, CUDA, m){ m.impl("idf", TORCH_FN(idf)); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.doc()="probe"; }
