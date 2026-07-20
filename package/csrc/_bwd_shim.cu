// Тонкий .cu-шим для backward-ядра. torch/extension.h ПЕРВЫМ (см. _fwd_shim.cu). Без FA_STANDALONE => без main().
#include <torch/extension.h>
#include "../../kernels/backward.hip"
