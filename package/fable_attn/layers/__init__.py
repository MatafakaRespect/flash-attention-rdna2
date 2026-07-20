"""flash_attn.layers — слои, которые ставит flash-attn рядом с ядром.

Их импортирует чужой код (`from flash_attn.layers.rotary import apply_rotary_emb`), и без них
drop-in ломается на импорте, а не на числах. Реализация — чистый torch.
"""
from . import rotary

__all__ = ["rotary"]
