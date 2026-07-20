"""flash_attn.ops.triton.layer_norm — паритет на чистом torch.

У flash-attn это Triton-ядра (слитые LayerNorm/RMSNorm с residual). Триптона под ROCm на
gfx1030 у нас нет, поэтому здесь те же функции обычным torch: результат тот же, слияния нет.
Модуль существует ради импорта — `from flash_attn.ops.triton.layer_norm import rms_norm_fn`
стоит в моделях сплошь и рядом, и без него drop-in падает на импорте.

Честно: это НЕ ускоритель, а совместимость. Скорость здесь на уровне обычного torch.
"""
import torch
from torch import nn

__all__ = ["layer_norm_fn", "rms_norm_fn", "layer_norm_ref", "rms_norm_ref",
           "RMSNorm", "LayerNorm"]


def _norm(x, weight, bias, eps, is_rms, upcast=True):
    dt = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float() if weight is not None else None
        bias = bias.float() if bias is not None else None
    if is_rms:
        out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    else:
        out = (x - x.mean(dim=-1, keepdim=True))
        out = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out.to(dt)


def layer_norm_fn(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None,
                  eps=1e-6, dropout_p=0.0, rowscale=None, prenorm=False,
                  residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False,
                  out=None, residual_out=None):
    """Слитый (у нас — обычный) LayerNorm/RMSNorm с residual, как в flash-attn.

    Возвращает out, либо (out, residual_out) при prenorm=True. При заданных weight1/bias1
    считается вторая нормировка того же входа — как в оригинале.
    """
    if dropout_p and dropout_p > 0:
        raise NotImplementedError(
            "fable_attn: dropout внутри layer_norm_fn не реализован — примените torch.dropout "
            "снаружи, результат тот же")
    if rowscale is not None:
        x = x * rowscale.unsqueeze(-1)
    res = x if residual is None else x + residual
    if x1 is not None:
        res = res + x1
    res_out = res.float() if residual_in_fp32 else res
    y = _norm(res, weight, bias, eps, is_rms_norm)
    if out is not None:
        out.copy_(y); y = out
    if weight1 is not None:
        y1 = _norm(res, weight1, bias1, eps, is_rms_norm)
        return ((y, y1, res_out) if prenorm else (y, y1))
    if prenorm:
        if residual_out is not None:
            residual_out.copy_(res_out); res_out = residual_out
        return y, res_out
    return y


def rms_norm_fn(x, weight, bias=None, residual=None, x1=None, weight1=None, bias1=None,
                eps=1e-6, dropout_p=0.0, rowscale=None, prenorm=False,
                residual_in_fp32=False, return_dropout_mask=False, out=None,
                residual_out=None):
    return layer_norm_fn(x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p,
                         rowscale, prenorm, residual_in_fp32, True, return_dropout_mask,
                         out, residual_out)


def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False,
                   upcast=False, **kw):
    res = x if residual is None else x + residual
    y = _norm(res, weight, bias, eps, False, upcast)
    return (y, res) if prenorm else y


def rms_norm_ref(x, weight, bias=None, residual=None, eps=1e-6, prenorm=False,
                 upcast=False, **kw):
    res = x if residual is None else x + residual
    y = _norm(res, weight, bias, eps, True, upcast)
    return (y, res) if prenorm else y


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(x, self.weight, self.bias, residual=residual, eps=self.eps,
                           dropout_p=self.drop.p if self.drop is not None else 0.0,
                           prenorm=prenorm, residual_in_fp32=residual_in_fp32)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_fn(x, self.weight, self.bias, residual=residual, eps=self.eps,
                             dropout_p=self.drop.p if self.drop is not None else 0.0,
                             prenorm=prenorm, residual_in_fp32=residual_in_fp32)
