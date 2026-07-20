"""parity_new_api.py — гейт на возможности, закрытые по PLAN.md §3 (дыры паритета с flash-attn).

Проверяется не «функция вызывается», а «функция считает то же, что точный эталон на torch».
Отдельный файл от parity_test.py: тот стережёт то, что работало раньше, этот — то, что добавлено.

Запуск: python lab/parity_new_api.py
"""
import os, sys, math, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package")))

import torch
import fable_attn as fa

DEV = "cuda"
FAILS = []


def check(name, got, ref, tol=3e-3):
    e = (got.float() - ref.float()).abs().max().item()
    ok = e < tol and torch.isfinite(got.float()).all()
    print(f"  {name:56s} max|err|={e:.2e}  {'ОК' if ok else 'ПРОВАЛ'}")
    if not ok:
        FAILS.append(name)


def ok(name, cond, info=""):
    print(f"  {name:56s} {'ОК' if cond else 'ПРОВАЛ'} {info}")
    if not cond:
        FAILS.append(name)


def rnd(*shape, dt=torch.float16):
    return torch.randn(*shape, device=DEV, dtype=dt)


def ref_attn(q, k, v, scale, causal=False, window=None, chunk=0, qv=None,
             wl=None, wr=None, softcap=0.0, alibi=None, qpos=None):
    """Точный эталон в fp32. q,k,v: (B,S,H,D). qv: (B,Sq,H,Dv) — FA-3 добавляет qv·vᵀ к скору.
    wl/wr — окно в семантике flash-attn; qpos — позиция запроса (для декода, где Sq=1)."""
    q_, k_, v_ = (x.transpose(1, 2).float() for x in (q, k, v))     # (B,H,S,D)
    B, H, Sq, _ = q_.shape
    Hk = k_.shape[1]
    if H != Hk:
        k_ = k_.repeat_interleave(H // Hk, dim=1)
        v_ = v_.repeat_interleave(H // Hk, dim=1)
    s = q_ @ k_.transpose(-1, -2)
    if qv is not None:
        s = s + (qv.transpose(1, 2).float() @ v_.transpose(-1, -2))
    if softcap and softcap > 0:
        s = softcap * torch.tanh(s * scale / softcap)
    else:
        s = s * scale
    Sk = k_.shape[2]
    qi = torch.arange(Sq, device=q.device).view(-1, 1)
    ki = torch.arange(Sk, device=q.device).view(1, -1)
    delta = Sk - Sq
    if alibi is not None:
        base = qi + delta if qpos is None else torch.full_like(qi, qpos)
        s = s + alibi.view(1, H, 1, 1).float() * (ki - base).float()
    mask = torch.zeros(Sq, Sk, device=q.device, dtype=torch.bool)
    if causal:
        mask |= ki > qi + delta
    if window is not None and window > 0:
        mask |= (ki < qi + delta - window + 1) | (ki > qi + delta)
    if wl is not None and wl >= 0:
        mask |= ki < qi + delta - wl
    if wr is not None and wr >= 0:
        mask |= ki > qi + delta + wr
    if chunk:
        cs = ((qi + delta).clamp_min(0) // chunk) * chunk
        mask |= (ki < cs) | (ki >= cs + chunk)
    s = s.masked_fill(mask, float("-inf"))
    p = s.softmax(-1)
    p = torch.nan_to_num(p, nan=0.0)          # полностью замаскированная строка -> нули
    return (p @ v_).transpose(1, 2)


def main():
    torch.manual_seed(0)
    B, S, H, D = 1, 256, 8, 128
    sc = 1.0 / math.sqrt(D)
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)

    print("=== attention_chunk (FA-3) ===")
    for chunk, causal in ((64, True), (100, True), (128, False), (37, True)):
        check(f"chunk={chunk} causal={causal}",
              fa.flash_attn_func(q, k, v, causal=causal, attention_chunk=chunk),
              ref_attn(q, k, v, sc, causal=causal, chunk=chunk))
    q2 = rnd(B, 64, H, D)
    check("chunk=64 SQ=64 SK=256 causal",
          fa.flash_attn_func(q2, k, v, causal=True, attention_chunk=64),
          ref_attn(q2, k, v, sc, causal=True, chunk=64))
    check("chunk + sliding-window вместе",
          fa.flash_attn_func(q, k, v, causal=True, attention_chunk=128, window_size=(31, 0)),
          ref_attn(q, k, v, sc, causal=True, chunk=128, window=32))

    print("\n=== раскладка выхода (drop-in: у настоящего flash-attn выход ПЛОТНЫЙ) ===")
    o = fa.flash_attn_func(q, k, v, causal=True)
    ok("выход contiguous", o.is_contiguous(), str(o.stride()))
    try:
        o.view(B, S, H * D)
        ok("out.view(B,S,H*D) работает (строка из attention-блоков HF)", True)
    except RuntimeError as e:
        ok("out.view(B,S,H*D) работает (строка из attention-блоков HF)", False, str(e)[:60])
    # выход не должен зависеть от того, копировали мы K/V или читали по страйдам
    ok("выход не зависит от выбранной раскладки K/V",
       torch.equal(o, fa.flash_attn_func(q, k.contiguous(), v.contiguous(), causal=True)))
    qg2 = q.clone().requires_grad_()
    og = fa.flash_attn_func(qg2, k, v, causal=True)
    ok("выход contiguous и на пути с градиентом", og.is_contiguous())
    ok("путь с градиентом даёт тот же результат", torch.equal(og.detach(), o))

    print("\n=== пустые строки (все ключи замаскированы) ===")
    qbig = rnd(B, 256, H, D)
    ksm, vsm = rnd(B, 64, H, D), rnd(B, 64, H, D)
    o = fa.flash_attn_func(qbig, ksm, vsm, causal=True)
    r = ref_attn(qbig, ksm, vsm, sc, causal=True)
    check("SQ>SK causal: первые строки = нули", o, r)
    ok("первые 192 строки ровно нули", bool((o[:, :192] == 0).all()))

    print("\n=== qv (FA-3: скор += qv·vᵀ) ===")
    dqk, dv = 128, 128
    qq, kk = rnd(B, S, H, dqk), rnd(B, S, H, dqk)
    vv, qvt = rnd(B, S, H, dv), rnd(B, S, H, dv)
    scq = (dqk + dv) ** -0.5
    for causal in (False, True):
        check(f"qv d={dqk}+{dv} causal={causal}",
              fa.flash_attn_func(qq, kk, vv, causal=causal, qv=qvt),
              ref_attn(qq, kk, vv, scq, causal=causal, qv=qvt), tol=6e-3)
    # headdim_v != headdim без qv
    q64, k64 = rnd(B, S, H, 64), rnd(B, S, H, 64)
    v192 = rnd(B, S, H, 192)
    ok("выход qv-пути тоже плотный",
       fa.flash_attn_func(qq, kk, vv, causal=True, qv=qvt).is_contiguous())
    check("headdim_v=192 != headdim=64 (без qv)",
          fa.flash_attn_func(q64, k64, v192, causal=True, softmax_scale=1 / math.sqrt(64)),
          ref_attn(q64, k64, v192, 1 / math.sqrt(64), causal=True), tol=6e-3)
    # сумма headdim+headdim_v, не совпадающая с инстанцированной, теперь добивается нулями
    v80, qv80 = rnd(B, S, H, 80), rnd(B, S, H, 80)
    check("qv d=128+80 (сумма 208 -> добивка до 224)",
          fa.flash_attn_func(qq, kk, v80, causal=True, qv=qv80),
          ref_attn(qq, kk, v80, (128 + 80) ** -0.5, causal=True, qv=qv80), tol=6e-3)
    # а вот сумма выше потолка head_dim должна падать внятно
    try:
        fa.flash_attn_func(qq, kk, rnd(B, S, H, 160, dt=torch.float16),
                           qv=rnd(B, S, H, 160, dt=torch.float16))
        ok("qv с суммой выше потолка — внятная ошибка", False, "исключения не было")
    except NotImplementedError as e:
        ok("qv с суммой выше потолка — внятная ошибка", "256" in str(e))

    print("\n=== return_attn_probs (FA-2: тройка out, lse, probs) ===")
    res = fa.flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    ok("возвращается тройка", isinstance(res, tuple) and len(res) == 3)
    o3, lse3, p3 = res
    ok("формы out/lse/probs", o3.shape == q.shape and tuple(lse3.shape) == (B, H, S)
       and tuple(p3.shape) == (B, H, S, S))
    ok("вероятности суммируются в 1", bool((p3.sum(-1) - 1).abs().max() < 3e-3))
    check("probs·V совпадает с out", (p3 @ v.transpose(1, 2).float()).transpose(1, 2), o3.float())
    ref_lse = ref_attn(q, k, v, sc, causal=True)   # только чтобы форма совпала
    del ref_lse

    print("\n=== varlen: seqused_q / seqused_k ===")
    lens = [128, 64, 200]
    used_q = [70, 64, 150]
    used_k = [90, 30, 180]
    tot = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=DEV, dtype=torch.int32)
    uq = torch.tensor(used_q, device=DEV, dtype=torch.int32)
    uk = torch.tensor(used_k, device=DEV, dtype=torch.int32)
    qv_, kv_, vv_ = rnd(tot, H, D), rnd(tot, H, D), rnd(tot, H, D)
    out = fa.flash_attn_varlen_func(qv_, kv_, vv_, cu, cu, max(lens), max(lens),
                                    causal=True, seqused_q=uq, seqused_k=uk)
    exp = torch.zeros_like(out)
    for i in range(len(lens)):
        o0 = int(cu[i])
        r = ref_attn(qv_[o0:o0 + used_q[i]].unsqueeze(0), kv_[o0:o0 + used_k[i]].unsqueeze(0),
                     vv_[o0:o0 + used_k[i]].unsqueeze(0), sc, causal=True).squeeze(0)
        exp[o0:o0 + used_q[i]] = r
    check("varlen seqused_q+seqused_k", out, exp)

    print("\n=== декод: cache_leftpad / cache_batch_idx / return_softmax_lse ===")
    ctx, page, Hk = 400, 32, 2
    nblk = (ctx + page - 1) // page
    qd = rnd(1, 1, H, D)
    kc, vc = rnd(nblk, page, Hk, D), rnd(nblk, page, Hk, D)
    bt = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, nblk)
    cs_ = torch.tensor([ctx], device=DEV, dtype=torch.int32)
    kd = kc.reshape(nblk * page, Hk, D)[:ctx].unsqueeze(0)
    vd = vc.reshape(nblk * page, Hk, D)[:ctx].unsqueeze(0)
    for lp in (0, 37, 100):
        lpt = torch.tensor([lp], device=DEV, dtype=torch.int32)
        o = fa.flash_attn_with_kvcache(qd, kc, vc, cache_seqlens=cs_, block_table=bt,
                                       cache_leftpad=lpt)
        check(f"cache_leftpad={lp}", o, ref_attn(qd, kd[:, lp:], vd[:, lp:], sc))
    # page_table как синоним block_table
    check("page_table == block_table",
          fa.flash_attn_with_kvcache(qd, kc, vc, cache_seqlens=cs_, page_table=bt),
          fa.flash_attn_with_kvcache(qd, kc, vc, cache_seqlens=cs_, block_table=bt), tol=1e-6)
    # return_softmax_lse
    o, lse = fa.flash_attn_with_kvcache(qd, kc, vc, cache_seqlens=cs_, block_table=bt,
                                        return_softmax_lse=True)
    s_ref = (qd.transpose(1, 2).float() @ kd.transpose(1, 2).float()
             .repeat_interleave(H // Hk, dim=1).transpose(-1, -2)) * sc
    lse_ref = torch.logsumexp(s_ref, dim=-1)          # (B,H,1)
    check("return_softmax_lse из ядра", lse, lse_ref, tol=1e-2)
    # cache_batch_idx (плотный кэш)
    Bd, Sd = 4, 256
    qdd = rnd(Bd, 1, H, D)
    kcd, vcd = rnd(Bd, Sd, Hk, D), rnd(Bd, Sd, Hk, D)
    idx = torch.tensor([3, 1, 0, 2], device=DEV, dtype=torch.int32)
    L = 100
    o = fa.flash_attn_with_kvcache(qdd, kcd, vcd, cache_seqlens=L, cache_batch_idx=idx)
    exp = torch.cat([ref_attn(qdd[b:b + 1], kcd[idx[b]:idx[b] + 1, :L],
                              vcd[idx[b]:idx[b] + 1, :L], sc) for b in range(Bd)], dim=0)
    check("cache_batch_idx", o, exp)

    print("\n=== flash_attn_combine (слияние split-KV) ===")
    half = S // 2
    o1, l1, _ = fa.flash_attn_func(q, k[:, :half], v[:, :half], return_attn_probs=True)
    o2, l2, _ = fa.flash_attn_func(q, k[:, half:], v[:, half:], return_attn_probs=True)
    op = torch.stack([o1, o2], 0)
    lp_ = torch.stack([l1, l2], 0)
    check("combine двух половин == полное внимание",
          fa.flash_attn_combine(op, lp_), ref_attn(q, k, v, sc), tol=6e-3)

    print("\n=== подсказки планировщику и явные отказы ===")
    base = fa.flash_attn_func(q, k, v, causal=True)
    for nm, kw in (("num_splits=4", dict(num_splits=4)), ("pack_gqa=True", dict(pack_gqa=True)),
                   ("sm_margin=8", dict(sm_margin=8))):
        got = fa.flash_attn_func(q, k, v, causal=True, **kw)
        ok(f"{nm}: принят, выход бит-в-бит тот же", torch.equal(got, base))
    try:
        fa.flash_attn_func(q, k, v, q_descale=torch.ones(1, device=DEV))
        ok("q_descale (FP8) — явный отказ", False, "исключения не было")
    except NotImplementedError as e:
        ok("q_descale (FP8) — явный отказ", "FP8" in str(e))
    try:
        fa.flash_attn_func(q, k, v, num_splits="много")
        ok("мусор в num_splits — TypeError", False, "исключения не было")
    except TypeError:
        ok("мусор в num_splits — TypeError", True)
    qg = rnd(1, 128, 4, 64).requires_grad_()
    try:
        fa.flash_attn_func(qg, qg.detach(), qg.detach(), causal=True,
                           attention_chunk=64).sum().backward()
        ok("backward с attention_chunk — явный отказ (как в FA-3)", False, "исключения не было")
    except NotImplementedError:
        ok("backward с attention_chunk — явный отказ (как в FA-3)", True)

    print("\n=== bert_padding ===")
    from fable_attn import bert_padding as bp
    hs = rnd(3, 16, 8)
    am = torch.zeros(3, 16, dtype=torch.long, device=DEV)
    am[0, :10] = 1; am[1, :16] = 1; am[2, :4] = 1
    unp, idxs, cus, mx, used = bp.unpad_input(hs, am)
    ok("unpad_input отдаёт пятёрку", unp.shape[0] == 30 and mx == 16 and int(cus[-1]) == 30)
    back = bp.pad_input(unp, idxs, 3, 16)
    check("pad_input(unpad_input(x)) == x на валидных токенах",
          back * am.unsqueeze(-1), hs * am.unsqueeze(-1), tol=1e-9)

    print("\n=== регистрация имён (drop-in) ===")
    fa.install_as_flash_attn()
    import flash_attn
    import flash_attn_interface as fa3
    from flash_attn.bert_padding import pad_input, unpad_input      # noqa: F401
    ok("flash_attn.__version__ читается как FA-2",
       flash_attn.__version__.startswith("2."), flash_attn.__version__)
    # FA-3 порядок: 4-й позиционный — softmax_scale, а НЕ dropout_p
    check("flash_attn_interface (порядок FA-3)", fa3.flash_attn_func(q, k, v, sc, True),
          ref_attn(q, k, v, sc, causal=True))
    o3b, lse3b = fa3.flash_attn_func(q, k, v, sc, True, return_attn_probs=True)
    ok("FA-3 return_attn_probs -> пара", o3b.shape == q.shape and tuple(lse3b.shape) == (B, H, S))
    ok("get_scheduler_metadata не падает", fa3.get_scheduler_metadata() is None)

    # ================= второй заход: то, что закрывалось после «1 в 1» =================
    print("\n=== окно (left, right) целиком ===")
    for wl_, wr_, nm in ((32, 0, "(32,0) левое"), (32, 32, "(32,32) двустороннее"),
                         (-1, 0, "(-1,0) = causal"), (0, 0, "(0,0) только диагональ"),
                         (0, -1, "(0,-1) только вперёд"), (16, 48, "(16,48) несимметричное")):
        got = fa.flash_attn_func(q, k, v, window_size=(wl_, wr_))
        ref = ref_attn(q, k, v, sc, wl=wl_, wr=wr_)
        check(f"window_size={nm}", got, ref)

    print("\n=== head_dim: все инстанцированные + добивка произвольных ===")
    for D_ in (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256):
        qd, kd_, vd_ = rnd(1, 128, 4, D_), rnd(1, 128, 4, D_), rnd(1, 128, 4, D_)
        check(f"head_dim={D_} (нативный)", fa.flash_attn_func(qd, kd_, vd_, causal=True),
              ref_attn(qd, kd_, vd_, D_ ** -0.5, causal=True), tol=6e-3)
    for D_ in (40, 100, 144, 200, 255):
        qd, kd_, vd_ = rnd(1, 128, 4, D_), rnd(1, 128, 4, D_), rnd(1, 128, 4, D_)
        check(f"head_dim={D_} (добивка)", fa.flash_attn_func(qd, kd_, vd_, causal=True),
              ref_attn(qd, kd_, vd_, D_ ** -0.5, causal=True), tol=6e-3)
    qg2 = rnd(1, 96, 4, 192).requires_grad_()
    kg2 = rnd(1, 96, 4, 192).requires_grad_()
    vg2 = rnd(1, 96, 4, 192).requires_grad_()
    fa.flash_attn_func(qg2, kg2, vg2, causal=True).sum().backward()
    ok("backward при head_dim=192 даёт конечные градиенты",
       all(torch.isfinite(x.grad).all() for x in (qg2, kg2, vg2)))
    qg3 = rnd(1, 64, 2, 256).requires_grad_()
    fa.flash_attn_func(qg3, qg3.detach(), qg3.detach(), causal=True).sum().backward()
    ok("backward при head_dim=256 даёт конечные градиенты (раньше ядро не запускалось)",
       bool(torch.isfinite(qg3.grad).all()))

    print("\n=== декод: softcap, alibi, несколько токенов, семантика cache_seqlens ===")
    Hk2, D2, L2 = 2, 64, 100
    sc2 = D2 ** -0.5
    qd = rnd(2, 1, 8, D2)
    kcd, vcd = rnd(2, 256, Hk2, D2), rnd(2, 256, Hk2, D2)
    check("декод softcap=30",
          fa.flash_attn_with_kvcache(qd, kcd, vcd, cache_seqlens=L2, softcap=30.0),
          ref_attn(qd, kcd[:, :L2], vcd[:, :L2], sc2, softcap=30.0), tol=6e-3)
    al2 = (-0.1 * torch.arange(1, 9, device=DEV)).float()
    check("декод alibi",
          fa.flash_attn_with_kvcache(qd, kcd, vcd, cache_seqlens=L2, alibi_slopes=al2),
          ref_attn(qd, kcd[:, :L2], vcd[:, :L2], sc2, alibi=al2, qpos=L2), tol=6e-3)
    # несколько новых токенов без дозаписи: каждый видит один и тот же кэш
    qm = rnd(2, 4, 8, D2)
    om = fa.flash_attn_with_kvcache(qm, kcd, vcd, cache_seqlens=L2)
    rm = ref_attn(qm, kcd[:, :L2], vcd[:, :L2], sc2)
    check("декод 4 новых токена (спекулятивка)", om, rm, tol=6e-3)
    # с дозаписью: токен i дополнительно видит ключи новых токенов 0..i
    kcd2, vcd2 = kcd.clone(), vcd.clone()
    kn, vn = rnd(2, 4, Hk2, D2), rnd(2, 4, Hk2, D2)
    oa = fa.flash_attn_with_kvcache(qm, kcd2, vcd2, k=kn, v=vn, cache_seqlens=L2)
    exp = torch.stack([ref_attn(qm[:, i:i + 1],
                                torch.cat([kcd[:, :L2], kn[:, :i + 1]], 1),
                                torch.cat([vcd[:, :L2], vn[:, :i + 1]], 1), sc2).squeeze(1)
                       for i in range(4)], dim=1)
    check("декод 4 токена с дозаписью (причинность между новыми)", oa, exp, tol=6e-3)
    ok("дозапись положила ключи на позиции cache_seqlens..",
       torch.allclose(kcd2[:, L2:L2 + 4].float(), kn.float(), atol=1e-3))

    print("\n=== varlen: страничный KV и вероятности ===")
    lens2 = [96, 64]
    page2 = 32
    tot2 = sum(lens2)
    cu2 = torch.tensor([0] + list(itertools.accumulate(lens2)), device=DEV, dtype=torch.int32)
    qv2 = rnd(tot2, H, D)
    nblk_per = [(L + page2 - 1) // page2 for L in lens2]
    kpg = rnd(sum(nblk_per), page2, H, D)
    vpg = rnd(sum(nblk_per), page2, H, D)
    bt2 = torch.zeros(2, max(nblk_per), dtype=torch.int32, device=DEV)
    off = 0
    for i, nb in enumerate(nblk_per):
        bt2[i, :nb] = torch.arange(off, off + nb, device=DEV, dtype=torch.int32); off += nb
    o_pg = fa.flash_attn_varlen_func(qv2, kpg, vpg, cu2, cu2, max(lens2), max(lens2),
                                     causal=True, block_table=bt2)
    exp_pg = []
    for i, L in enumerate(lens2):
        a = int(cu2[i])
        kd2 = kpg[bt2[i, :nblk_per[i]].long()].reshape(-1, H, D)[:L]
        vd2 = vpg[bt2[i, :nblk_per[i]].long()].reshape(-1, H, D)[:L]
        exp_pg.append(ref_attn(qv2[a:a + L].unsqueeze(0), kd2.unsqueeze(0), vd2.unsqueeze(0),
                               sc, causal=True).squeeze(0))
    check("varlen со страничным KV (block_table)", o_pg, torch.cat(exp_pg, 0))
    r3 = fa.flash_attn_varlen_func(qv2, qv2, qv2, cu2, cu2, max(lens2), max(lens2),
                                   return_attn_probs=True)
    ok("varlen return_attn_probs -> (out, список lse, список probs)",
       isinstance(r3, tuple) and len(r3) == 3 and len(r3[2]) == 2
       and tuple(r3[2][0].shape) == (H, lens2[0], lens2[0]))

    print()
    if FAILS:
        print(f"*** ПРОВАЛЕНО: {len(FAILS)} — {', '.join(FAILS)}")
        return 1
    print("ВСЕ НОВЫЕ ВОЗМОЖНОСТИ РАБОТАЮТ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
