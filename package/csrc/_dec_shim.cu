// Тонкий .cu-шим для ядра декода. torch/extension.h ПЕРВЫМ (см. _fwd_shim.cu). Без FA_STANDALONE => без main().
#include <torch/extension.h>
#include "../../kernels/fa_decode.hip"
