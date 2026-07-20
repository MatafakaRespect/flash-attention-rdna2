"""flash_attn_interface — верхнее имя пакета FA-3 (hopper).

У FA-3 другой порядок позиционных аргументов, чем у FA-2 (четвёртый — softmax_scale, а не
dropout_p), поэтому имена разведены: `import flash_attn` = FA-2, `import flash_attn_interface`
= FA-3. Оба зовут одни ядра.
"""
from fable_attn.fa3_interface import *                 # noqa: F401,F403
from fable_attn.fa3_interface import (                 # noqa: F401
    flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func,
    flash_attn_with_kvcache, flash_attn_combine, get_scheduler_metadata,
)
