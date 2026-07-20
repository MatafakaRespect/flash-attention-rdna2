// Тонкий .cu-шим для forward-ядра. torch/extension.h ПЕРВЫМ — как в рабочем образце
// D:\extensions\fable_attn\src\fable_attn_kernel.cu: иначе fatbin device-кода не регистрируется
// в torch-расширении и .pyd падает с ошибкой инициализации DLL (1114). Без FA_STANDALONE => без main().
#include <torch/extension.h>
#include "../../kernels/fa_forward.hip"
