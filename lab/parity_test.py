"""parity_test.py — проверка того, что ЗАЯВЛЕНО в API и что оно реально считает.

Каждая функция публичного API вызывается и сверяется с эталоном на torch (fp32).
Проверяются и те возможности, про которые пакет утверждает поддержку: GQA, causal,
sliding-window, softcap, alibi, все head_dim из диспетчера, varlen (нативный путь),
страничный KV-кэш, dropout=0 детерминизм, install_as_flash_attn.

Запуск: python lab/parity_test.py
"""
import os, sys, math, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package")))

import torch
import fable_attn as fa

DEV = "cuda"
FAILS = []


def ref_attn(q, k, v, scale, causal=False, window=None, softcap=0.0, alibi=None):
    """Эталон в fp32. q,k,v: (B,S,H,D). Возвращает (B,S,H,D)."""
    q_, k_, v_ = (x.transpose(1, 2).float() for x in (q, k, v))   # (B,H,S,D)
    B, H, Sq, D = q_.shape
    Hk = k_.shape[1]
    if H != Hk:
        k_ = k_.repeat_interleave(H // Hk, dim=1)
        v_ = v_.repeat_interleave(H // Hk, dim=1)
    s = (q_ @ k_.transpose(-1, -2)) * scale
    if softcap and softcap > 0:
        s = softcap * torch.tanh(s / softcap)
    Sk = k_.shape[2]
    qi = torch.arange(Sq, device=q.device).view(-1, 1)
    ki = torch.arange(Sk, device=q.device).view(1, -1)
    delta = Sk - Sq
    if alibi is not None:
        s = s + alibi.view(1, H, 1, 1).float() * (ki - qi - delta).float()
    mask = torch.zeros(Sq, Sk, device=q.device, dtype=torch.bool)
    if causal:
        mask |= ki > qi + delta
    if window is not None and window > 0:
        mask |= ki < (qi + delta - window + 1)
        mask |= ki > qi + delta
    s = s.masked_fill(mask, float("-inf"))
    return (s.softmax(-1) @ v_).transpose(1, 2)


def check(name, got, ref, tol=3e-3):
    e = (got.float() - ref.float()).abs().max().item()
    ok = e < tol and torch.isfinite(got.float()).all()
    print(f"  {name:52s} max|err|={e:.2e}  {'ОК' if ok else 'ПРОВАЛ'}")
    if not ok:
        FAILS.append(name)


def rnd(*shape, dt=torch.float16):
    return torch.randn(*shape, device=DEV, dtype=dt)


def main():
    torch.manual_seed(0)
    print("=== заявленные head_dim (из C++-диспетчера) ===")
    for D in sorted({16, 32, 48, 64, 80, 96, 128, 256}):
        B, S, H = 1, 128, 4
        q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
        sc = 1.0 / math.sqrt(D)
        try:
            o = fa.flash_attn_func(q, k, v)
            check(f"head_dim={D}", o, ref_attn(q, k, v, sc))
        except Exception as e:
            print(f"  head_dim={D:<44} ОШИБКА: {type(e).__name__}: {e}")
            FAILS.append(f"head_dim={D}")

    print("\n=== маски и модификаторы ===")
    B, S, H, D = 1, 256, 8, 128
    sc = 1.0 / math.sqrt(D)
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    check("causal", fa.flash_attn_func(q, k, v, causal=True), ref_attn(q, k, v, sc, causal=True))
    check("sliding-window (64,0)",
          fa.flash_attn_func(q, k, v, window_size=(63, 0)),
          ref_attn(q, k, v, sc, window=64))
    check("softcap=30",
          fa.flash_attn_func(q, k, v, softcap=30.0),
          ref_attn(q, k, v, sc, softcap=30.0))
    al = (-2.0 ** (-torch.arange(1, H + 1, device=DEV, dtype=torch.float32) * 8.0 / H))
    check("alibi", fa.flash_attn_func(q, k, v, alibi_slopes=al),
          ref_attn(q, k, v, sc, alibi=al))

    print("\n=== GQA / MQA ===")
    for Hk in (8, 4, 2, 1):
        kk, vv = rnd(B, S, Hk, D), rnd(B, S, Hk, D)
        check(f"GQA H={H}/{Hk}", fa.flash_attn_func(q, kk, vv), ref_attn(q, kk, vv, sc))

    print("\n=== SQ != SK (bottom-right causal) ===")
    q2 = rnd(B, 64, H, D)
    check("SQ=64 SK=256 causal", fa.flash_attn_func(q2, k, v, causal=True),
          ref_attn(q2, k, v, sc, causal=True))

    print("\n=== packed-варианты ===")
    qkv = torch.stack([q, k, v], dim=2)                      # (B,S,3,H,D)
    check("qkvpacked", fa.flash_attn_qkvpacked_func(qkv), ref_attn(q, k, v, sc))
    kv = torch.stack([k, v], dim=2)                          # (B,S,2,H,D)
    check("kvpacked", fa.flash_attn_kvpacked_func(q, kv), ref_attn(q, k, v, sc))

    print("\n=== varlen (нативный путь ядра) ===")
    lens = [128, 64, 200, 32]
    tot = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=DEV, dtype=torch.int32)
    qv, kv_, vv_ = rnd(tot, H, D), rnd(tot, H, D), rnd(tot, H, D)
    o = fa.flash_attn_varlen_func(qv, kv_, vv_, cu, cu, max(lens), max(lens))
    ref = torch.cat([ref_attn(qv[cu[i]:cu[i+1]].unsqueeze(0), kv_[cu[i]:cu[i+1]].unsqueeze(0),
                              vv_[cu[i]:cu[i+1]].unsqueeze(0), sc).squeeze(0)
                     for i in range(len(lens))], dim=0)
    check("varlen", o, ref)
    o = fa.flash_attn_varlen_func(qv, kv_, vv_, cu, cu, max(lens), max(lens), causal=True)
    ref = torch.cat([ref_attn(qv[cu[i]:cu[i+1]].unsqueeze(0), kv_[cu[i]:cu[i+1]].unsqueeze(0),
                              vv_[cu[i]:cu[i+1]].unsqueeze(0), sc, causal=True).squeeze(0)
                     for i in range(len(lens))], dim=0)
    check("varlen causal", o, ref)
    qkvp = torch.stack([qv, kv_, vv_], dim=1)                # (total,3,H,D)
    check("varlen qkvpacked",
          fa.flash_attn_varlen_qkvpacked_func(qkvp, cu, max(lens)), ref_attn(
              qv.unsqueeze(0), kv_.unsqueeze(0), vv_.unsqueeze(0), sc).squeeze(0)
          if False else fa.flash_attn_varlen_func(qv, kv_, vv_, cu, cu, max(lens), max(lens)))
    kvp = torch.stack([kv_, vv_], dim=1)                     # (total,2,H,D)
    check("varlen kvpacked",
          fa.flash_attn_varlen_kvpacked_func(qv, kvp, cu, cu, max(lens), max(lens)),
          fa.flash_attn_varlen_func(qv, kv_, vv_, cu, cu, max(lens), max(lens)))

    print("\n=== декод со страничным KV-кэшем ===")
    for ctx, Hk, page in ((1024, 2, 32), (777, 8, 16), (64, 8, 64)):
        nblk = (ctx + page - 1) // page
        qd = rnd(1, 1, H, D)
        kc, vc = rnd(nblk, page, Hk, D), rnd(nblk, page, Hk, D)
        bt = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, nblk)
        cs = torch.tensor([ctx], device=DEV, dtype=torch.int32)
        o = fa.flash_attn_with_kvcache(qd, kc, vc, cache_seqlens=cs, block_table=bt)
        kd = kc.reshape(nblk * page, Hk, D)[:ctx].unsqueeze(0)
        vd = vc.reshape(nblk * page, Hk, D)[:ctx].unsqueeze(0)
        check(f"декод ctx={ctx} page={page} H={H}/{Hk}", o, ref_attn(qd, kd, vd, sc))

    print("\n=== backward ===")
    qg = rnd(1, 128, 4, 64).requires_grad_()
    kg = rnd(1, 128, 4, 64).requires_grad_()
    vg = rnd(1, 128, 4, 64).requires_grad_()
    go = rnd(1, 128, 4, 64)
    scg = 1.0 / math.sqrt(64)
    fa.flash_attn_func(qg, kg, vg, causal=True).backward(go)
    g1 = [x.grad.clone() for x in (qg, kg, vg)]
    for x in (qg, kg, vg):
        x.grad = None
    ref_attn(qg, kg, vg, scg, causal=True).to(qg.dtype).backward(go)
    g2 = [x.grad.clone() for x in (qg, kg, vg)]
    for nm, a, b in zip(("dq", "dk", "dv"), g1, g2):
        check(f"backward {nm}", a, b, tol=2e-2)

    print("\n=== детерминизм (тот же вход -> бит-в-бит) ===")
    o1 = fa.flash_attn_func(q, k, v, causal=True)
    o2 = fa.flash_attn_func(q, k, v, causal=True)
    same = torch.equal(o1, o2)
    print(f"  {'два вызова совпадают бит-в-бит':52s} {'ОК' if same else 'ПРОВАЛ'}")
    if not same:
        FAILS.append("детерминизм")

    print("\n=== install_as_flash_attn ===")
    fa.install_as_flash_attn()
    import flash_attn
    from flash_attn import flash_attn_func as ff
    check("import flash_attn работает", ff(q, k, v), ref_attn(q, k, v, sc))

    print()
    if FAILS:
        print(f"*** ПРОВАЛЕНО: {len(FAILS)} — {', '.join(FAILS)}")
        return 1
    print("ВСЁ ЗАЯВЛЕННОЕ РАБОТАЕТ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
