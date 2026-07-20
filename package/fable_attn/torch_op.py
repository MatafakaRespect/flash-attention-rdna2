"""Регистрация `torch.ops.fable_attn.forward` — контракт старой сборки из D:\\extensions.

ЗАЧЕМ. В окружении стоит `sitecustomize`, который зовёт `fable_integrate.install()`, а тот
вписывает ядро в родные точки расширения фреймворков (реестр бэкендов diffusers, ветка
ATTN_BACKEND у TRELLIS.2, перехват F.scaled_dot_product_attention как последняя мера). Он ищет
именно ОПЕРАТОР `torch.ops.fable_attn.forward`, который заводила прежняя однофайловая сборка.

Как только наш пакет занимает имя `fable_attn`, старая сборка перестаёт грузиться, оператор
исчезает — и вся эта привязка тихо превращается в тыкву: вызовы уходят на torch sdpa. Поэтому
оператор объявляем сами, с той же сигнатурой и тем же контрактом.

КОНТРАКТ (сверен по fable_integrate.py):
    forward(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor
    q, k, v — раскладка [B, H, S, D] (та же, что у F.scaled_dot_product_attention),
    возврат — [B, H, S, D]. На форме, которую обслужить нельзя, возвращается ПУСТОЙ тензор:
    вызывающий проверяет `o.numel() == 0` и уходит на свой обычный путь. Бросать исключение
    здесь нельзя — это сломало бы чужой фолбэк.
"""
import torch

LAST_ERROR = None      # причина, если зарегистрировать не вышло (для разбора руками)
REGISTERED = False

_LIB = None


def _forward(q, k, v, scale, causal):
    from . import _C
    empty = q.new_empty(0)
    # Ограничения ядра: fp16/bf16, dim contiguous, GQA кратна. Всё, что не подходит, —
    # пустой тензор, а не исключение (контракт фолбэка).
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return empty
    if q.dtype not in (torch.float16, torch.bfloat16) or not q.is_cuda:
        return empty
    if q.shape[-1] != k.shape[-1] or q.shape[-1] > 256:
        return empty
    if q.shape[1] % k.shape[1]:
        return empty
    qc = q if q.stride(-1) == 1 else q.contiguous()
    kc = k if k.stride(-1) == 1 else k.contiguous()
    vc = v if v.stride(-1) == 1 else v.contiguous()
    try:
        out, _ = _C.fwd(qc, kc, vc, float(scale), bool(causal), -1, 0 if causal else -1,
                        0.0, None, 0.0, 0, 0, None)
    except Exception:
        return empty
    return out


def register():
    """Идемпотентно. Не бросает: провал регистрации не должен ронять `import fable_attn`."""
    global _LIB, REGISTERED, LAST_ERROR
    if REGISTERED:
        return True
    try:
        ns = getattr(torch.ops, "fable_attn", None)
        if ns is not None and "forward" in dir(ns):
            REGISTERED = True          # кто-то уже объявил (например старая сборка) — не спорим
            return True
    except Exception:
        pass
    try:
        _LIB = torch.library.Library("fable_attn", "FRAGMENT")
        _LIB.define("forward(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor")
        _LIB.impl("forward", _forward, "CUDA")
        REGISTERED = True
        return True
    except Exception as e:     # pragma: no cover
        LAST_ERROR = f"{type(e).__name__}: {e}"
        return False
