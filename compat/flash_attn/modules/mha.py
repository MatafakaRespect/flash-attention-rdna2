"""flash_attn.modules.mha — MHA и Flash*Attention поверх ядер."""
from fable_attn.modules.mha import *                   # noqa: F401,F403
from fable_attn.modules.mha import (                   # noqa: F401
    FlashSelfAttention, FlashCrossAttention, SelfAttention, CrossAttention,
    MHA, LinearResidual,
)
