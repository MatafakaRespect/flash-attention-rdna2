"""flash_attn — точный FlashAttention для gfx1030 (RX 6800 XT).

Модуль называется так же, как у Dao-AILab, поэтому чужой код работает без единой правки:
`import flash_attn` берёт эти ядра. Сам движок — пакет fable_attn (там ядра, гейты, PLAN.md),
здесь только публичные имена в раскладке оригинала.
"""
from fable_attn import FLASH_ATTN_API_VERSION as __version__   # noqa: F401
from fable_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_combine,
    flash_attn_with_kvcache,
)
from . import bert_padding, flash_attn_interface, layers, modules, ops

__all__ = [
    "flash_attn_func", "flash_attn_qkvpacked_func", "flash_attn_kvpacked_func",
    "flash_attn_varlen_func", "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func", "flash_attn_combine", "flash_attn_with_kvcache",
    "bert_padding", "flash_attn_interface", "layers", "modules", "ops",
]
