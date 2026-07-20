"""fable_attn — drop-in flash-attn для gfx1030 (RX 6800 XT / ROCm).

Точный FlashAttention (forward+backward+decode) на HIP-ядрах v_dot2_f32_f16; API 1:1 с flash-attn.
CUDA-проект на ROCm подключается без правок:

    import fable_attn; fable_attn.install_as_flash_attn()   # теперь `import flash_attn` = это
или напрямую:
    from fable_attn import flash_attn_func
"""
import sys

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_combine,
    flash_attn_with_kvcache,
)
from . import bert_padding
from . import torch_op

# Оператор torch.ops.fable_attn.forward — его ищет fable_integrate из старой однофайловой
# сборки, чтобы вписать ядро в реестры diffusers/TRELLIS. Стоит нашему пакету занять имя
# fable_attn, как старая сборка перестаёт грузиться и оператор исчезает вместе со всей
# привязкой: вызовы тихо уходят на torch sdpa. Поэтому объявляем оператор сами.
torch_op.register()

__all__ = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_combine",
    "flash_attn_with_kvcache",
    "bert_padding",
    "install_as_flash_attn",
]

# Своя версия пакета.
__version__ = "0.2.0"

# Версия, которую пакет объявляет, когда его ставят вместо flash_attn.
# Это НЕ приписывание себе чужих заслуг, а часть контракта drop-in: библиотеки включают фичи
# по номеру версии (`parse(flash_attn.__version__) >= parse("2.4.2")` — типичная проверка перед
# тем, как передать window_size или softcap). Скажешь «0.2.0» — потребитель молча отключит
# скользящее окно и softcap, хотя они здесь есть. Число означает ровно одно: «поверхность API
# FA-2 версии 2.8.3 закрыта целиком», и после закрытия дыр из PLAN.md §3 это правда.
# Локальный суффикс +fable оставляет видимым, что реализация не их.
FLASH_ATTN_API_VERSION = "2.8.3+fable." + __version__


def install_as_flash_attn():
    """Зарегистрировать пакет под именами, по которым его ищет чужой код.

      flash_attn                        — FA-2 (порядок аргументов FA-2)
      flash_attn.flash_attn_interface   — то же, подмодулем
      flash_attn.bert_padding           — unpad_input/pad_input, их тянет половина проектов
      flash_attn_interface              — FA-3 (СВОЙ порядок аргументов, см. fa3_interface)

    FA-2 и FA-3 разведены по разным именам намеренно: у них разный порядок позиционных
    аргументов (4-й у FA-2 — dropout_p, у FA-3 — softmax_scale), и склеивать их в одну
    функцию значит однажды посчитать не то.
    """
    from . import flash_attn_interface as _fi
    from . import fa3_interface as _fi3
    from . import bert_padding as _bp
    from . import layers as _layers
    from .layers import rotary as _rotary
    from . import modules as _modules
    from .modules import mha as _mha
    from . import ops as _ops
    from .ops import triton as _triton
    from .ops.triton import layer_norm as _tln
    from .ops import rms_norm as _rms, layer_norm as _ln

    me = sys.modules[__name__]
    me.__dict__.setdefault("_fable_own_version", __version__)
    me.__version__ = FLASH_ATTN_API_VERSION      # см. комментарий выше
    _fi.__version__ = FLASH_ATTN_API_VERSION

    sys.modules["flash_attn"] = me
    sys.modules["flash_attn.flash_attn_interface"] = _fi
    sys.modules["flash_attn.bert_padding"] = _bp
    # Подмодули, которые чужой код тянет напрямую: без них drop-in падает на import,
    # а не на числах. Это НЕ заглушки — внутри рабочая реализация на torch.
    sys.modules["flash_attn.layers"] = _layers
    sys.modules["flash_attn.layers.rotary"] = _rotary
    sys.modules["flash_attn.modules"] = _modules
    sys.modules["flash_attn.modules.mha"] = _mha
    sys.modules["flash_attn.ops"] = _ops
    sys.modules["flash_attn.ops.triton"] = _triton
    sys.modules["flash_attn.ops.triton.layer_norm"] = _tln
    sys.modules["flash_attn.ops.rms_norm"] = _rms
    sys.modules["flash_attn.ops.layer_norm"] = _ln
    sys.modules["flash_attn_interface"] = _fi3
    return me
