"""flash_attn.flash_attn_interface — порядок аргументов FA-2, как у оригинала."""
from fable_attn.flash_attn_interface import *          # noqa: F401,F403
from fable_attn.flash_attn_interface import (          # noqa: F401
    flash_attn_func, flash_attn_qkvpacked_func, flash_attn_kvpacked_func,
    flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func, flash_attn_combine, flash_attn_with_kvcache,
)
from fable_attn import FLASH_ATTN_API_VERSION as __version__   # noqa: F401
