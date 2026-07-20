"""Drop-in flash-attn API поверх HIP-ядер fable_attn._C (gfx1030 / RX 6800 XT).

Совместимость 1:1 с flash-attn: те же имена и сигнатуры функций и та же раскладка входов
(batch, seqlen, nheads, headdim). Внутри тензоры переставляются в [B,H,S,D] (раскладка ядер),
считаются точным FlashAttention (forward — ядро forward-агента, backward — своё ядро), выход
переставляется обратно.

Порядок аргументов здесь — FA-2 (пакет ставится как `flash_attn`). Аргументы, которых в FA-2
нет, а в FA-3 есть (qv, attention_chunk, num_splits, pack_gqa, sm_margin, q/k/v_descale),
принимаются как keyword-only, чтобы не сдвинуть позиционные. Кто зовёт по-FA-3-шному
(`import flash_attn_interface`), получает модуль `fable_attn.fa3_interface` с ИХ порядком —
см. install_as_flash_attn().

Тихого отката НЕТ: неподдержанная конфигурация — внятное исключение. Раньше пакет молча
подменял ядро эталоном на питоне, который материализует полную матрицу S×S (при S=8192 и
32 головах это 8.6 ГБ VRAM), то есть притворялся работающим, будучи в сотни раз медленнее.
"""
import itertools
import math
import random
import torch
import torch.nn.functional as F

# Windows: _C.pyd зависит от amdhip64_7.dll (HIP-рантайм), который лежит в rocm-sdk bin,
# а не в torch/lib. `import torch` кладёт в поиск DLL только torch/lib → HIP-рантайм не находится
# и _C падает с "DLL load failed". Явно добавляем rocm bin в поиск DLL до импорта расширения.
import os as _os
_sp = _os.path.dirname(_os.path.dirname(torch.__file__))  # .../site-packages
for _sub in (_os.path.join("_rocm_sdk_core", "bin"),
             _os.path.join("_rocm_sdk_devel", "bin"),
             _os.path.join("torch", "lib")):
    _d = _os.path.join(_sp, _sub)
    if _os.path.isdir(_d):
        try:
            _os.add_dll_directory(_d)
        except Exception:
            pass

try:
    from fable_attn import _C  # даёт _C.fwd / _C.bwd (привязки pybind, без TORCH_LIBRARY)
except Exception as _e:  # pragma: no cover
    # НИКАКОГО тихого отката. Раньше здесь выставлялся _HAS_C=False, и каждый вызов молча
    # уходил в эталон на питоне, который материализует полную матрицу S×S: при S=8192 и
    # 32 головах это 8.6 ГБ VRAM, при S=16384 — 34 ГБ, с выселением в системную память.
    # Пакет притворялся работающим, будучи в сотни раз медленнее и с квадратичной памятью.
    raise ImportError(
        "fable_attn._C не загрузился — расширение собрано под другой torch/ROCm.\n"
        f"Причина: {type(_e).__name__}: {_e}\n"
        "Пересоберите под текущий интерпретатор:\n"
        "  $env:FA_PY='<путь>\\python.exe'; .\\package\\build.ps1"
    ) from _e

# head_dim, под которые ядро инстанцировано (см. BYHD в fa_forward.hip и switch в backward.hip).
# Всё, чего здесь нет, добивается нулями до ближайшего большего — ровно так же поступает
# flash-attn, у которого поддержаны только степени сетки, а произвольный head_dim паддится.
_VALID_D = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)
_MAX_D = _VALID_D[-1]


def _pad_dim(d):
    """Ближайший поддержанный head_dim >= d."""
    for x in _VALID_D:
        if x >= d:
            return x
    return None


def _pad_to(t, d):
    """Добить последнюю ось нулями до d. Для Q/K это не меняет скалярное произведение,
    для V — даёт нулевые лишние столбцы выхода, которые потом отрезаются."""
    if t.shape[-1] == d:
        return t
    return F.pad(t, (0, d - t.shape[-1]))


def _map_window(causal, window_size):
    """(causal_flag, win_left, win_right) в семантике flash-attn.

    window_size=(left, right): ключ j виден запросу i, если
        i+delta-left <= j <= i+delta+right,
    а отрицательное значение с любой стороны означает «границы нет».
    causal=True принудительно ставит правую границу на диагональ (так же делает flash-attn).

    Раньше поддерживались только (-1,-1) и (l, 0), а всё остальное — включая (-1, 0), то есть
    распространённую запись обычного causal, — отклонялось как неподдержанное.

    causal_flag выставляется при win_right==0: у ядра это компайл-тайм ветка, которая режет
    тайлы выше диагонали, и терять её на эквивалентной записи через окно незачем.
    """
    l, r = window_size
    wl = int(l) if l is not None and l >= 0 else -1
    wr = int(r) if r is not None and r >= 0 else -1
    if causal:
        wr = 0
    return (1 if wr == 0 else 0), wl, wr


# ------------------------- аргументы FA-3, которые мы не считаем сами -------------------------
def _reject_fp8(q_descale, k_descale, v_descale):
    """FP8-масштабы. Отказ ЯВНЫЙ, а не молчаливый: на gfx1030 нет тензорных ядер, FP8 там не
    ускорение, а потеря точности — а весь смысл FlashAttention в том, что он бит-в-бит равен
    обычному вниманию. Принять и проигнорировать эти аргументы значило бы вернуть неверные числа."""
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError(
            "fable_attn: q_descale/k_descale/v_descale (FP8) не поддержаны и не будут. "
            "На gfx1030 нет тензорных ядер: FP8 не даёт ускорения, а точность теряет, "
            "и ядро перестаёт быть точным FlashAttention. Передавайте fp16/bf16.")


def _accept_hints(num_splits=None, pack_gqa=None, sm_margin=None):
    """num_splits / pack_gqa / sm_margin у FA-3 — подсказки планировщику: они меняют РАСПИСАНИЕ,
    а не результат («can be tuned for speed» в их же доке). У нас расписание своё:

      * num_splits в prefill-forward нам нечего резать — воркгрупов и так хватает на все 72 CU
        (SQ/128 × nheads). В ДЕКОДЕ num_splits настоящий и передаётся в ядро.
      * pack_gqa — у нас замерено (см. PLAN.md §7): GQA-переиспользование во forward даёт ровно 0,
        потому что BR делится на G и воркгрупов становится вдвое больше при том же трафике.
        В декоде оно включено ВСЕГДА (один воркгруп на kv-голову), там это и дало 12%→99% полосы.
      * sm_margin — резервирование CU под коммуникацию; у нас однокарточный путь без NCCL.

    Поэтому аргументы принимаются и проверяются на тип, но на выход не влияют — он бит-в-бит тот же.
    Молча игнорировать неизвестный тип нельзя, поэтому мусор ловим сразу."""
    if num_splits is not None and not isinstance(num_splits, int):
        raise TypeError(f"fable_attn: num_splits должен быть int, получено {type(num_splits)}")
    if pack_gqa is not None and not isinstance(pack_gqa, bool):
        raise TypeError(f"fable_attn: pack_gqa должен быть bool или None, получено {type(pack_gqa)}")
    if sm_margin is not None and not isinstance(sm_margin, int):
        raise TypeError(f"fable_attn: sm_margin должен быть int, получено {type(sm_margin)}")


def _check_chunk(attention_chunk):
    if attention_chunk is None:
        return 0
    if not isinstance(attention_chunk, int) or attention_chunk < 0:
        raise ValueError(f"fable_attn: attention_chunk должен быть неотрицательным int, "
                         f"получено {attention_chunk!r}")
    return attention_chunk


def _check_supported(q, k, window_size, softcap, dim_ok=True):
    """Ничего не поддерживаем молча: неподдержанная конфигурация — внятное исключение,
    а не подмена медленным эталоном за спиной у вызывающего."""
    if not q.is_cuda:
        raise ValueError("fable_attn: тензоры должны быть на GPU (q.is_cuda == False)")
    D = q.shape[-1]
    if dim_ok and D > _MAX_D:
        raise ValueError(f"fable_attn: head_dim={D} больше максимума {_MAX_D} "
                         "(у flash-attn тот же потолок)")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"fable_attn: dtype {q.dtype} не поддержан; нужен fp16/bf16/fp32")
    # Раскладка публичного API — (batch, seqlen, nheads, headdim), число голов это ось 2.
    # Раньше здесь стояло shape[1] — сравнивались длины последовательностей вместо голов,
    # и настоящая проверка GQA не работала.
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(
            f"fable_attn: nheads_q={q.shape[2]} не кратно nheads_k={k.shape[2]} (GQA)")


# ------------------------- qv и headdim_v != headdim -------------------------
def _fuse_qv(q, k, v, qv):
    """FA-3 при заданном qv считает скор как  qk += einsum('bshd,bthd->bhst', qv, v)
    (проверено по их же tests/test_flash_attn.py), то есть

        S = Q·Kᵀ + QV·Vᵀ = [Q | QV] · [K | V]ᵀ,   O = softmax(S)·V.

    Значит фича сводится к конкатенации и НЕ требует отдельного ядра:
        q' = [q | qv]        ширина d+dv
        k' = [k | v]         ширина d+dv   (даёт ровно оба слагаемых скора)
        v' = [v | нули]      ширина d+dv   (первые dv столбцов выхода — искомый P·V)
    и потом срез out[..., :dv]. Результат точный, горячий путь ядра не тронут вообще.

    Тем же приёмом закрывается headdim_v != headdim без qv: тогда qv = нули, и лишнее
    слагаемое скора равно нулю тождественно.

    Плата — три временных тензора и требование, чтобы d+dv было в списке head_dim ядра.
    Это компат-путь, а не горячий: гнать через него обычное внимание незачем."""
    d, dv = q.shape[-1], v.shape[-1]
    hds = d + dv
    if hds > _MAX_D:
        raise NotImplementedError(
            f"fable_attn: qv/headdim_v требует head_dim скора = headdim+headdim_v = {d}+{dv} = {hds}, "
            f"а потолок ядра {_MAX_D} (у flash-attn тот же). Сумма должна умещаться в {_MAX_D}.")
    if qv is None:
        qv = torch.zeros(q.shape[:-1] + (dv,), dtype=q.dtype, device=q.device)
    elif qv.shape[-1] != dv:
        raise ValueError(f"fable_attn: qv.shape[-1]={qv.shape[-1]} должен совпадать с "
                         f"v.shape[-1]={dv}")
    qc = torch.cat([q, qv], dim=-1)
    kc = torch.cat([k, v], dim=-1)
    vc = torch.cat([v, torch.zeros_like(v[..., :d])], dim=-1) if d else v
    return qc, kc, vc, dv


def _needs_qv_path(q, v, qv):
    return qv is not None or v.shape[-1] != q.shape[-1]


# ------------------------- раскладка входов -------------------------
# Публичный API принимает (B,S,H,D), ядро считает в [B,H,S,D]. Раньше обёртка на КАЖДЫЙ вызов
# делала .transpose(1,2).contiguous() на q, k и v — три полные копии тензора. Ядро при этом
# умеет произвольные страйды (нужен только stride_d==1), то есть копии не обязательны.
#
# Замерено (lab/sweep_strides.py, lab/sweep_rule_check.py), выход при обоих путях БИТ-В-БИТ
# одинаковый — вопрос чисто про скорость:
#   * копии стоят 9–30% времени публичного вызова;
#   * чтение K/V по страйдам стоит ядру от 0 до 10%, и штраф растёт не с объёмом K/V,
#     а с КРАТНОСТЬЮ их перечитывания: он платится на каждом проходе, а копия — один раз.
#     (Проверено прямо: при фиксированной кратности объём K+V от 16 до 256 МБ штраф не меняет.)
#
# Кратность перечитывания каждого байта K/V = (число q-тайлов) * (сколько q-голов делят
# kv-голову) = ceil(SQ/BR) * gqa, для causal вдвое меньше — воркгруп читает в среднем половину.
#
# Правило намеренно КОНСЕРВАТИВНОЕ: страйды включаются только там, где выигрыш замерен
# устойчиво, во всех прочих случаях остаётся сегодняшнее поведение (копия). Одной формулой
# «кратность >= порог» обойтись не вышло — попытка проверена на 13 свежих точках и промахнулась
# на 4 из них (до 7% цены), потому что на весах две разные величины: штраф зависит от кратности,
# а цена копии — от объёма. Поэтому вместо подгонки одной кривой берём только те области,
# где замер однозначен.
_BYTES_128MB = 128 * 2 ** 20


def _kv_can_be_strided(SQ, D, Hq, Hkv, causal, kv_bytes):
    BR = 64 if D == 256 else 128            # q-тайл ядра (см. launch_cfg в fa_forward.hip)
    reread = math.ceil(SQ / BR) * (Hq / Hkv) / (2 if causal else 1)
    if reread <= 32:
        return True     # замерено: страйды выигрывают на всех клетках, иногда на 20–40%
    if reread <= 64 and kv_bytes >= _BYTES_128MB:
        return True     # при кратности 64 решает объём: копия большого K/V дороже штрафа
    return False        # дальше не рискуем — оставляем как было


# ------------------------- autograd поверх ядер -------------------------
class _FableAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal, wl, wr, softcap, alibi, dropout_p, deterministic,
                attn_chunk, out_buf):
        # q,k,v: [B,H,S,D] (обязательна только непрерывность по D). deterministic — наша схема
        # ВСЕГДА детерминирована (FA-2 без атомиков), флаг принят для совместимости, no-op.
        seed = random.getrandbits(63) if dropout_p > 0 else 0
        out, lse = _C.fwd(q, k, v, float(scale), bool(causal), int(wl), int(wr),
                          float(softcap), alibi, float(dropout_p), int(seed), int(attn_chunk),
                          out_buf)
        ctx.save_for_backward(q, k, v, out, lse, alibi if alibi is not None else torch.empty(0))
        ctx.scale = scale; ctx.causal = causal; ctx.wl = wl; ctx.wr = wr; ctx.softcap = softcap
        ctx.dropout_p = dropout_p; ctx.seed = seed; ctx.has_alibi = alibi is not None
        ctx.attn_chunk = attn_chunk
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        # FA-3 тоже не умеет backward с attention_chunk ("FA3 backward does not support
        # attention_chunk" у них в коде) — падаем так же явно, а не считаем молча не то.
        if ctx.attn_chunk:
            raise NotImplementedError(
                "fable_attn: backward с attention_chunk не поддержан (как и в FA-3). "
                "Для обучения с чанкованным вниманием считайте forward под torch.no_grad().")
        q, k, v, out, lse, alibi = ctx.saved_tensors
        al = alibi if ctx.has_alibi else None
        dq, dk, dv = _C.bwd(dout.contiguous(), q, k, v, out, lse,
                            float(ctx.scale), bool(ctx.causal), int(ctx.wl), int(ctx.wr),
                            float(ctx.softcap), al, float(ctx.dropout_p), int(ctx.seed))
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


# ------------------------- attn_probs (отладочный путь) -------------------------
def _attn_probs(q, k, lse, scale, causal, wl, wr, softcap, alibi, attn_chunk):
    """S_dmask для return_attn_probs: полная матрица вероятностей [B,H,SQ,SK].

    Это ОТЛАДОЧНЫЙ путь, ровно как у flash-attn (там про эти числа честно написано, что они
    «not guaranteed to be correct»). Он материализует матрицу S×S — то самое, чего ядро не
    делает никогда: при B=1,H=32,S=8192 это 8.6 ГБ. Поэтому включается только явным
    return_attn_probs=True и заранее проверяет, что тензор влезет в свободную VRAM.

    Считаем через LSE, который вернуло ядро: p = exp(s - lse). Значит числа согласованы
    с настоящим выходом, а не с независимым пересчётом softmax.
    Dropout к возвращаемым вероятностям НЕ применяется (это вероятности ДО него).
    q,k: [B,H,S,D]. Возвращает [B,Hq,SQ,SK]."""
    B, Hq, SQ, D = q.shape
    Hk, SK = k.shape[1], k.shape[2]
    need = B * Hq * SQ * SK * 4
    free = torch.cuda.mem_get_info(q.device)[0]
    if need > free * 0.8:
        raise torch.cuda.OutOfMemoryError(
            f"fable_attn: return_attn_probs требует матрицу S×S на {need/2**30:.1f} ГБ "
            f"(свободно {free/2**30:.1f} ГБ). Это отладочный путь; само ядро S×S не материализует.")
    kk = k if Hk == Hq else k.repeat_interleave(Hq // Hk, dim=1)
    s = torch.matmul(q.float(), kk.float().transpose(-1, -2))
    if softcap and softcap > 0:
        s = softcap * torch.tanh(s * scale / softcap)
    else:
        s = s * scale
    delta = SK - SQ
    qi = torch.arange(SQ, device=q.device).view(-1, 1)
    ki = torch.arange(SK, device=q.device).view(1, -1)
    if alibi is not None:
        s = s + alibi.view(1, Hq, 1, 1).float() * (ki - qi - delta).float()
    mask = torch.zeros(SQ, SK, device=q.device, dtype=torch.bool)
    if causal:
        mask |= ki > qi + delta
    if wl >= 0:
        mask |= ki < qi + delta - wl
    if wr >= 0:
        mask |= ki > qi + delta + wr
    if attn_chunk:
        cs = ((qi + delta).clamp_min(0) // attn_chunk) * attn_chunk
        mask |= (ki < cs) | (ki >= cs + attn_chunk)
    p = torch.exp(s - lse.unsqueeze(-1))
    return p.masked_fill(mask, 0.0)


# ------------------------- общий forward -------------------------
def _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
              deterministic, return_attn_probs, qv, attention_chunk,
              q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin):
    """Общее тело для flash_attn_func и packed-обёрток. Вход (B,S,H,D)."""
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    chunk = _check_chunk(attention_chunk)

    dv_out = None
    if _needs_qv_path(q, v, qv):
        if softmax_scale is None:   # FA-3: 1/sqrt(headdim + headdim_qv)
            softmax_scale = (q.shape[-1] + v.shape[-1]) ** (-0.5)
        q, k, v, dv_out = _fuse_qv(q, k, v, qv)
    elif softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    _check_supported(q, k, window_size, softcap)
    cflag, wl, wr = _map_window(causal, window_size)

    # head_dim, под который ядро не инстанцировано, добиваем нулями до ближайшего большего.
    # Скалярные произведения от нулевых столбцов не меняются, лишние столбцы выхода — нули.
    d_real = q.shape[-1]
    d_pad = _pad_dim(d_real)
    if d_pad != d_real:
        q = _pad_to(q, d_pad); k = _pad_to(k, d_pad); v = _pad_to(v, d_pad)
        if dv_out is None:
            dv_out = d_real          # отрежем добивку на выходе

    B, SQ, Hq, D = q.shape
    Hkv = k.shape[2]
    need_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)

    qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
    if need_grad:
        # Путь обучения оставлен ровно таким, каким был: замер показал, что у backward выигрыш
        # от страйдов около процента и знак плавает от формы — рисковать не за что.
        qh = qh.contiguous(); kh = kh.contiguous(); vh = vh.contiguous()
    else:
        # Q читается один раз (кладётся в LDS) — копировать его смысла нет никогда.
        # K/V решаем по замеренному правилу, см. _kv_can_be_strided.
        kv_bytes = 2 * B * k.shape[1] * Hkv * D * q.element_size()
        if not _kv_can_be_strided(SQ, D, Hq, Hkv, bool(cflag), kv_bytes):
            kh = kh.contiguous(); vh = vh.contiguous()

    # Выход выделяем в раскладке (B,S,H,D) и отдаём ядру его транспонированный вид. Так
    # потребитель получает ПЛОТНЫЙ тензор — как настоящий flash-attn. Раньше возвращался
    # транспонированный вид плотного [B,H,S,D], на котором падает `.view(B, S, H*D)` —
    # ровно та строка, что стоит в attention-блоках HuggingFace.
    out_buf = torch.empty((B, SQ, Hq, D), dtype=q.dtype, device=q.device).transpose(1, 2)

    al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
    out, lse = _FableAttnFn.apply(qh, kh, vh, softmax_scale, cflag, wl, wr, softcap, al,
                                  dropout_p, deterministic, chunk, out_buf)
    res = out.transpose(1, 2)
    if dv_out is not None:
        # срез по последней оси плотным не бывает, а контракт «выход плотный» должен держаться
        # и здесь (компат-путь qv и так не горячий)
        res = res[..., :dv_out].contiguous()
    if return_attn_probs:
        # flash-attn отдаёт тройку (out, softmax_lse, S_dmask) — отдаём её же.
        probs = _attn_probs(qh, kh, lse, softmax_scale, bool(cflag), wl, wr, softcap, al, chunk)
        return res, lse, probs
    return res


# ------------------------- публичный API -------------------------
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, *,
                    qv=None, attention_chunk=0,
                    q_descale=None, k_descale=None, v_descale=None,
                    num_splits=1, pack_gqa=None, sm_margin=0):
    """q,k,v: (batch, seqlen, nheads, headdim). Возвращает out той же раскладки,
    а при return_attn_probs=True — тройку (out, softmax_lse, S_dmask), как FA-2."""
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                              deterministic=False, return_attn_probs=False, *,
                              qv=None, attention_chunk=0,
                              q_descale=None, k_descale=None, v_descale=None,
                              num_splits=1, pack_gqa=None, sm_margin=0):
    """qkv: (batch, seqlen, 3, nheads, headdim)."""
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                             deterministic=False, return_attn_probs=False, *,
                             qv=None, attention_chunk=0,
                             q_descale=None, k_descale=None, v_descale=None,
                             num_splits=1, pack_gqa=None, sm_margin=0):
    """q: (batch, sq, nheads, d); kv: (batch, sk, 2, nheads_k, d)."""
    k, v = kv[:, :, 0], kv[:, :, 1]
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                           deterministic=False, return_attn_probs=False, *,
                           seqused_q=None, seqused_k=None,
                           qv=None, attention_chunk=0,
                           q_descale=None, k_descale=None, v_descale=None,
                           num_splits=1, pack_gqa=None, sm_margin=0,
                           block_table=None):
    """Varlen (packed, cu_seqlens): q,k,v: (total_tokens, nheads, headdim).

    Без градиента идёт нативным путём ядра (cu_seqlens прямо в параметрах — один запуск на всё).
    С градиентом разбирается по последовательностям: backward-ядро плотное.

    seqused_q/seqused_k — фактически используемая длина каждой последовательности (может быть
    короче, чем даёт cu_seqlens): передаётся прямо в ядро, лишних копий нет.
    """
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    chunk = _check_chunk(attention_chunk)

    if block_table is not None:
        # Страничный KV в varlen-prefill: k/v лежат как (num_blocks, page, nheads_k, d), а
        # block_table (batch, max_blocks) говорит, какие страницы принадлежат последовательности.
        # Разворачиваем страницы в плотную последовательность и идём обычным varlen-путём —
        # ядро prefill читает K/V подряд, ему страничная косвенность не нужна.
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                "fable_attn: при block_table ключи и значения должны быть в страничной "
                f"раскладке (num_blocks, page_block_size, nheads_k, headdim), получено "
                f"k.shape={tuple(k.shape)}. Без страниц block_table не нужен.")
        bt = block_table.to(torch.int32)
        nseq = bt.shape[0]
        page = k.shape[1]
        if bt.dim() != 2 or bt.shape[0] != cu_seqlens_k.numel() - 1:
            raise ValueError(
                f"fable_attn: block_table ждётся формы (batch, max_blocks), получено "
                f"{tuple(bt.shape)} при batch={cu_seqlens_k.numel() - 1}")
        lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist() if seqused_k is None \
            else seqused_k.tolist()
        kd, vd = [], []
        for i in range(nseq):
            L = int(lens_k[i])
            nb = (L + page - 1) // page
            idx = bt[i, :nb].long()
            kd.append(k[idx].reshape(-1, k.shape[2], k.shape[3])[:L])
            vd.append(v[idx].reshape(-1, v.shape[2], v.shape[3])[:L])
        k = torch.cat(kd, 0).contiguous()
        v = torch.cat(vd, 0).contiguous()
        cu_seqlens_k = torch.tensor([0] + list(itertools.accumulate(int(x) for x in lens_k)),
                                    device=k.device, dtype=torch.int32)
        seqused_k = None
        block_table = None

    dv_out = None
    if _needs_qv_path(q, v, qv):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + v.shape[-1]) ** (-0.5)
        q, k, v, dv_out = _fuse_qv(q, k, v, qv)
    elif softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    need_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if not need_grad:
        if q.dim() != 3:
            raise ValueError(f"fable_attn: varlen ждёт (total, nheads, headdim), получено {tuple(q.shape)}")
        if q.shape[-1] > _MAX_D:
            raise ValueError(f"fable_attn: head_dim={q.shape[-1]} больше максимума {_MAX_D}")
        d_real = q.shape[-1]
        d_pad = _pad_dim(d_real)
        if d_pad != d_real:
            q = _pad_to(q, d_pad); k = _pad_to(k, d_pad); v = _pad_to(v, d_pad)
            if dv_out is None:
                dv_out = d_real
        cflag, wl, wr = _map_window(causal, window_size)
        al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
        seed = random.getrandbits(63) if dropout_p > 0 else 0
        su_q = seqused_q.to(torch.int32).contiguous() if seqused_q is not None else None
        su_k = seqused_k.to(torch.int32).contiguous() if seqused_k is not None else None
        out = _C.fwd_varlen(q.contiguous(), k.contiguous(), v.contiguous(),
                            cu_seqlens_q.to(torch.int32).contiguous(),
                            cu_seqlens_k.to(torch.int32).contiguous(),
                            int(max_seqlen_q), int(max_seqlen_k),
                            float(softmax_scale), bool(cflag), int(wl), int(wr), float(softcap),
                            al, float(dropout_p), int(seed), int(chunk), su_q, su_k)
        if dv_out is not None:
            out = out[..., :dv_out]
        if return_attn_probs:
            # У рваных длин нет одной формы S×S, поэтому вероятности отдаём СПИСКОМ — по одной
            # матрице (nheads, Lq, Lk) на последовательность. Это отладочный путь, как и в
            # flash-attn: он материализует то, чего ядро не материализует никогда.
            lse_l, probs_l = [], []
            for i in range(cu_seqlens_q.numel() - 1):
                a, b_ = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                c_, d_ = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
                if seqused_q is not None:
                    b_ = a + int(seqused_q[i])
                if seqused_k is not None:
                    d_ = c_ + int(seqused_k[i])
                _o, _l, _p = flash_attn_func(
                    q[a:b_].unsqueeze(0), k[c_:d_].unsqueeze(0), v[c_:d_].unsqueeze(0),
                    dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
                    deterministic, True, attention_chunk=chunk)
                lse_l.append(_l.squeeze(0)); probs_l.append(_p.squeeze(0))
            return out, lse_l, probs_l
        return out
    outs = []
    nseq = cu_seqlens_q.numel() - 1
    for i in range(nseq):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        if seqused_q is not None:
            qe = qs + int(seqused_q[i])
        if seqused_k is not None:
            ke = ks + int(seqused_k[i])
        oi = flash_attn_func(q[qs:qe].unsqueeze(0), k[ks:ke].unsqueeze(0), v[ks:ke].unsqueeze(0),
                             dropout_p, softmax_scale, causal, window_size,
                             softcap, alibi_slopes, deterministic, False,
                             attention_chunk=chunk)
        outs.append(oi.squeeze(0))
    res = torch.cat(outs, dim=0)
    if dv_out is not None:
        res = res[..., :dv_out]
    return res


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None, causal=False, window_size=(-1, -1),
                                     softcap=0.0, alibi_slopes=None, deterministic=False,
                                     return_attn_probs=False, **kw):
    """qkv: (total_tokens, 3, nheads, headdim)."""
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                  dropout_p, softmax_scale, causal, window_size, softcap,
                                  alibi_slopes, deterministic, return_attn_probs, **kw)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                    max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                                    causal=False, window_size=(-1, -1), softcap=0.0,
                                    alibi_slopes=None, deterministic=False,
                                    return_attn_probs=False, **kw):
    """q: (total_q, nheads, d); kv: (total_k, 2, nheads_k, d)."""
    k, v = kv[:, 0], kv[:, 1]
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                  max_seqlen_k, dropout_p, softmax_scale, causal,
                                  window_size, softcap, alibi_slopes, deterministic,
                                  return_attn_probs, **kw)


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    """Слить частичные выходы split-KV: out_partial (nsplit, B, S, H, D),
    lse_partial (nsplit, B, H, S) в натуральных единицах. Совместимо с FA-3."""
    lse = lse_partial.float()
    m = lse.amax(dim=0, keepdim=True)
    w = torch.exp(lse - m)                      # (nsplit,B,H,S)
    wsum = w.sum(dim=0)                         # (B,H,S)
    w = w.permute(0, 1, 3, 2).unsqueeze(-1)     # (nsplit,B,S,H,1)
    acc = (out_partial.float() * w).sum(dim=0)  # (B,S,H,D)
    res = acc / wsum.permute(0, 2, 1).unsqueeze(-1).clamp_min(1e-30)
    res = res.to(out_dtype or out_partial.dtype)
    if out is not None:
        out.copy_(res); return out
    return res


@torch.no_grad()
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, qv=None,
                            rotary_cos=None, rotary_sin=None,
                            cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None,
                            block_table=None, softmax_scale=None, causal=False,
                            window_size=(-1, -1), softcap=0.0, rotary_interleaved=True,
                            alibi_slopes=None, num_splits=0, return_softmax_lse=False, *,
                            page_table=None, attention_chunk=0,
                            q_descale=None, k_descale=None, v_descale=None,
                            pack_gqa=None, sm_margin=0, scheduler_metadata=None,
                            cu_seqlens_q=None, cu_seqlens_k_new=None, max_seqlen_q=None,
                            rotary_seqlens=None):
    """Инференс-декод через ядро fa_decode (split-KV, страничный кэш, rotary на лету).

    q: (batch, seqlen_q, nheads, headdim). seqlen_q > 1 — это спекулятивное декодирование и
    chunked prefill: новые токены встают на позиции cache_seqlens..cache_seqlens+seqlen_q-1,
    причинность между ними держится (токен i видит ключи по свою позицию включительно).

    Страничный кэш — раскладка flash-attn: k_cache/v_cache
    (num_blocks, page_block_size, nheads_k, headdim) + block_table (batch, max_blocks).
    Плотный кэш (batch, seqlen, nheads_k, headdim) тоже принимается — он трактуется как
    один блок на элемент батча.

    cache_seqlens — длина кэша ДО дозаписи, ровно как у flash-attn: если переданы k и v,
    они пишутся на позиции cache_seqlens.. и внимание идёт по ним тоже.
    rotary_cos/rotary_sin — ТАБЛИЦА (seqlen_ro, rotary_dim/2), строка берётся по позиции токена;
    rotary_dim может быть меньше headdim (частичный rotary).
    page_table — имя того же аргумента в FA-3, принимается как синоним block_table.
    cache_batch_idx — какой строкой кэша пользуется каждый элемент батча.
    cache_leftpad — с какой позиции начинается контекст (ключи левее не наши).
    return_softmax_lse — вернуть (out, softmax_lse) формы (batch, nheads, seqlen_q).

    Раньше эта функция молча игнорировала block_table и считала обычным prefill-ядром по
    срезу кэша: со страничным кэшем это давало ТИХО НЕВЕРНЫЙ результат (внимание по одной
    странице вместо всего контекста). Теперь зовётся настоящее ядро декода.
    """
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    if q.dim() != 4:
        raise ValueError(f"fable_attn: декод ждёт q формы (batch, seqlen_q, nheads, headdim), "
                         f"получено {tuple(q.shape)}")
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** -0.5
    # scheduler_metadata у FA-3 — предпосчитанное расписание. У нас лаунчер считает его на
    # каждом запуске за единицы микросекунд, кэшировать нечего: принимаем и игнорируем,
    # на результат это не влияет (get_scheduler_metadata и возвращает None).
    _ = scheduler_metadata
    if cu_seqlens_k_new is not None:
        raise NotImplementedError(
            "fable_attn: cu_seqlens_k_new (рваная дозапись) не поддержан; дозаписывайте "
            "одинаковое число токенов на элемент батча")

    # varlen-раскладка запроса: разбираем по последовательностям и склеиваем обратно.
    if cu_seqlens_q is not None:
        cu = cu_seqlens_q.to(torch.int32).cpu()
        outs, lses = [], []
        for i in range(cu.numel() - 1):
            a, b_ = int(cu[i]), int(cu[i + 1])
            qi = q[a:b_].unsqueeze(0) if q.dim() == 3 else q[:, a:b_]
            csi = cache_seqlens[i:i + 1] if torch.is_tensor(cache_seqlens) else cache_seqlens
            r = flash_attn_with_kvcache(
                qi, k_cache, v_cache, k=k, v=v, qv=qv, rotary_cos=rotary_cos,
                rotary_sin=rotary_sin, cache_seqlens=csi,
                cache_batch_idx=cache_batch_idx, cache_leftpad=cache_leftpad,
                block_table=block_table, softmax_scale=softmax_scale, causal=causal,
                window_size=window_size, softcap=softcap,
                rotary_interleaved=rotary_interleaved, alibi_slopes=alibi_slopes,
                num_splits=num_splits, return_softmax_lse=return_softmax_lse,
                page_table=page_table, attention_chunk=attention_chunk,
                pack_gqa=pack_gqa, sm_margin=sm_margin, rotary_seqlens=rotary_seqlens)
            if return_softmax_lse:
                outs.append(r[0].squeeze(0)); lses.append(r[1])
            else:
                outs.append(r.squeeze(0))
        out = torch.cat(outs, dim=0)
        return (out, torch.cat(lses, dim=0)) if return_softmax_lse else out

    if page_table is not None:
        if block_table is not None:
            raise ValueError("fable_attn: block_table и page_table — одно и то же, задайте одно")
        block_table = page_table

    B, NQ = q.shape[0], q.shape[1]
    dev = q.device
    d_real = q.shape[-1]

    # qv в декоде — тот же приём, что в prefill (S = q·kᵀ + qv·vᵀ = [q|qv]·[k|v]ᵀ), только
    # конкатенировать приходится ВЕСЬ кэш, а он большой. Поэтому путь честно помечен компат-путём:
    # он верен, но удваивает трафик кэша на шаг. Тем, кому нужна скорость, — не сюда.
    if qv is not None:
        if qv.shape[-1] != v_cache.shape[-1]:
            raise ValueError("fable_attn: qv.shape[-1] должен совпадать с headdim кэша значений")
        if d_real + qv.shape[-1] > _MAX_D:
            raise NotImplementedError(
                f"fable_attn: qv в декоде требует headdim+headdim_v <= {_MAX_D}")
        q = torch.cat([q, qv], dim=-1)
        k_cache = torch.cat([k_cache, v_cache], dim=-1)
        v_cache = torch.cat([v_cache, torch.zeros_like(v_cache[..., :d_real])], dim=-1)
        if k is not None:
            k = torch.cat([k, v], dim=-1)
            v = torch.cat([v, torch.zeros_like(v[..., :d_real])], dim=-1)

    if cache_seqlens is None:
        seqlens = torch.full((B,), k_cache.shape[1], dtype=torch.int32, device=dev)
    elif isinstance(cache_seqlens, int):
        seqlens = torch.full((B,), cache_seqlens, dtype=torch.int32, device=dev)
    else:
        seqlens = cache_seqlens.to(device=dev, dtype=torch.int32).contiguous()

    if block_table is None:
        # плотный кэш: один блок на элемент батча, страница = вся длина.
        # cache_batch_idx именно здесь и работает: строка кэша выбирается индексом, а не b.
        page = k_cache.shape[1]
        if page & (page - 1):
            raise NotImplementedError(
                f"fable_attn: длина плотного кэша {page} должна быть степенью двойки "
                "(или передайте block_table)")
        if cache_batch_idx is None:
            bt = torch.arange(B, dtype=torch.int32, device=dev).view(B, 1)
        else:
            bt = cache_batch_idx.to(device=dev, dtype=torch.int32).contiguous().view(B, 1)
        kc, vc = k_cache, v_cache
    else:
        if cache_batch_idx is not None:
            raise ValueError(
                "fable_attn: cache_batch_idx имеет смысл только для плотного кэша; при "
                "страничном кэше строки выбирает block_table")
        bt = block_table.to(device=dev, dtype=torch.int32).contiguous()
        kc, vc = k_cache, v_cache

    # Ядро декода берёт окно как ЧИСЛО ключей назад от позиции запроса; правая граница
    # для запроса, стоящего в конце последовательности, ничего не режет.
    _, wl, _wr = _map_window(True, window_size)
    win = (wl + 1) if wl >= 0 else -1

    lp = cache_leftpad.to(device=dev, dtype=torch.int32).contiguous() if cache_leftpad is not None else None

    # attention_chunk: запрос видит только свой чанк. Для запроса на позиции p это ровно
    # левая граница floor(p/chunk)*chunk, то есть выражается через тот же cache_leftpad.
    chunk = _check_chunk(attention_chunk)
    if chunk:
        if NQ > 1:
            raise NotImplementedError(
                "fable_attn: attention_chunk в декоде поддержан для одного нового токена: "
                "у нескольких токенов границы чанка разные, а кэш задаёт её на элемент батча")
        qpos = seqlens.to(torch.int64)
        cl = (qpos // chunk) * chunk
        cl = cl.to(torch.int32)
        lp = cl if lp is None else torch.maximum(lp, cl)

    al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
    cos = rotary_cos.float().contiguous() if rotary_cos is not None else None
    sin = rotary_sin.float().contiguous() if rotary_sin is not None else None
    # rotary_dim берём из таблицы: у flash-attn она (seqlen_ro, rotary_dim/2)
    rdim = int(cos.shape[-1] * 2) if cos is not None else 0
    rpos = (rotary_seqlens.to(device=dev, dtype=torch.int32).contiguous()
            if rotary_seqlens is not None else None)
    nk = k.contiguous() if k is not None else None
    nv = v.contiguous() if v is not None else None
    if (nk is None) != (nv is None):
        raise ValueError("fable_attn: k и v дописываются в кэш только вместе")

    # head_dim, под который ядро не инстанцировано, добиваем нулями (как в prefill)
    dv_cut = None
    dq = q.shape[-1]
    d_pad = _pad_dim(dq)
    if d_pad is None:
        raise ValueError(f"fable_attn: head_dim={dq} больше максимума {_MAX_D}")
    if d_pad != dq:
        q = _pad_to(q, d_pad); kc = _pad_to(kc, d_pad); vc = _pad_to(vc, d_pad)
        if nk is not None:
            nk = _pad_to(nk, d_pad); nv = _pad_to(nv, d_pad)
        dv_cut = dq
    if qv is not None:
        dv_cut = v_cache.shape[-1] - d_real if dv_cut is None else dv_cut

    out, lse = _C.dec(q.contiguous(), kc.contiguous(), vc.contiguous(), bt, seqlens, nk, nv,
                      cos, sin, bool(rotary_interleaved), float(softmax_scale), int(win),
                      int(num_splits or 0), lp, bool(return_softmax_lse),
                      float(softcap or 0.0), al, int(rdim), rpos)
    if qv is not None:
        # v_cache стал [v | нули], значит искомый P·V лежит в первых headdim_v столбцах
        out = out[..., :qv.shape[-1]].contiguous()
    elif dv_cut is not None:
        out = out[..., :dv_cut].contiguous()
    if return_softmax_lse:
        # ядро пишет LSE как (batch, seqlen_q, nheads); flash-attn отдаёт (batch, nheads, seqlen_q)
        return out, lse.transpose(1, 2).contiguous()
    return out
