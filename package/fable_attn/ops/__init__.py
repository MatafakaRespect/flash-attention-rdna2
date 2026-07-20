"""flash_attn.ops — паритет по именам с flash-attn (чистый torch, без Triton).

Тритона под ROCm на gfx1030 у нас нет, поэтому здесь те же функции обычным torch: числа те же,
слияния ядер нет. Модули существуют ради импорта — без них чужой код падает на `import`,
а не на числах.
"""
from . import triton
from .rms_norm import RMSNorm, rms_norm, dropout_add_rms_norm
from .layer_norm import LayerNorm, dropout_add_layer_norm

__all__ = ["triton", "RMSNorm", "rms_norm", "dropout_add_rms_norm",
           "LayerNorm", "dropout_add_layer_norm"]
