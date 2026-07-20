"""sweep_strides.py — где страйдовое чтение K/V дороже предварительной копии в плотный вид.

Обёртка сейчас копирует q,k,v в плотный [B,H,S,D]. Копии стоят 9–30% времени вызова, но на
длинных последовательностях ядро на страйдах теряет больше, чем копии экономят. Надо понять,
ОТ ЧЕГО это зависит, а не подобрать порог по двум точкам.

Две гипотезы:
  A) объём K+V (не влезает в Infinity Cache 128 МБ -> разбросанные чтения бьют по DRAM);
  B) кратность перечитывания K/V = число q-тайлов = ceil(SQ/BR) (для causal вдвое меньше):
     штраф платится на КАЖДОМ проходе, а копия — один раз.

Меряем только ядро (без обёртки), плотное против страйдового, разводя объём и кратность.

Запуск: python lab/sweep_strides.py
"""
import os, sys, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package")))

import torch
from fable_attn import _C

DEV = "cuda"
BR = 128          # q-тайл ядра при HD<=128
IC_MB = 128.0     # Infinity Cache на RX 6800 XT


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def one(B, H, S, D, causal, Hkv=None, iters=3):
    Hkv = Hkv or H
    dt = torch.float16
    q = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt)
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt)
    sc = 1.0 / math.sqrt(D)

    qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    qh = qs.contiguous(); kh = ks.contiguous(); vh = vs.contiguous()

    f_dense = lambda: _C.fwd(qh, kh, vh, sc, causal, -1, 0.0, None, 0.0, 0, 0)
    f_strid = lambda: _C.fwd(qs, ks, vs, sc, causal, -1, 0.0, None, 0.0, 0, 0)
    # копия только K/V (Q считаем бесплатным — он читается один раз)
    f_copykv = lambda: (ks.contiguous(), vs.contiguous())

    # чередуем, чтобы тепловой дрейф не оседал в одной из веток
    td = ts = tc = 0.0
    for _ in range(3):
        td += timeit(f_dense, iters)
        ts += timeit(f_strid, iters)
        tc += timeit(f_copykv, iters)
    td /= 3; ts /= 3; tc /= 3

    kv_mb = 2 * B * S * Hkv * D * 2 / 2**20
    reread = math.ceil(S / BR) / (2 if causal else 1)
    pen = (ts - td) / td * 100
    win = (td + tc) - ts       # >0 => страйды выгоднее (копия дороже штрафа)
    print(f"  B{B} H{H}/{Hkv} S{S:<6d} D{D} {'caus' if causal else 'full'} | "
          f"K+V {kv_mb:7.1f} МБ | перечит. {reread:6.1f} | "
          f"плотно {td:8.3f}  страйд {ts:8.3f}  штраф {pen:+6.1f}% | "
          f"копия K/V {tc:7.3f} | итог {'СТРАЙД' if win > 0 else 'КОПИЯ '} {win:+8.3f} мс")
    del q, k, v, qs, ks, vs, qh, kh, vh
    torch.cuda.empty_cache()
    return dict(B=B, H=H, S=S, D=D, causal=causal, kv_mb=kv_mb, reread=reread,
                dense=td, strided=ts, copy=tc, pen=pen, win=win)


def main():
    print(f"Infinity Cache {IC_MB:.0f} МБ, q-тайл BR={BR}\n")
    rows = []
    print("--- меняем S при H=32 (растут И объём, И кратность перечитывания) ---")
    for S in (2048, 4096, 8192, 12288, 16384):
        rows.append(one(1, 32, S, 128, False, iters=3 if S >= 8192 else 8))

    print("\n--- фиксируем кратность (S=8192), меняем объём K+V числом kv-голов ---")
    for Hkv in (4, 8, 16, 32):
        rows.append(one(1, 32, 8192, 128, False, Hkv=Hkv, iters=3))

    print("\n--- фиксируем объём (K+V ~268 МБ), меняем кратность через B и S ---")
    rows.append(one(8, 32, 2048, 128, False, iters=4))    # перечит. 16
    rows.append(one(4, 32, 4096, 128, False, iters=3))    # перечит. 32
    rows.append(one(2, 32, 8192, 128, False, iters=3))    # перечит. 64
    rows.append(one(1, 32, 16384, 128, False, iters=3))   # перечит. 128

    print("\n--- causal (кратность вдвое меньше при том же объёме) ---")
    for S in (4096, 8192, 16384):
        rows.append(one(1, 32, S, 128, True, iters=3 if S >= 8192 else 6))

    print("\n=== где страйды проигрывают ===")
    bad = [r for r in rows if r["win"] <= 0]
    if not bad:
        print("  нигде")
    for r in bad:
        print(f"  B{r['B']} H{r['H']} S{r['S']} {'caus' if r['causal'] else 'full'}: "
              f"K+V {r['kv_mb']:.0f} МБ, перечит. {r['reread']:.0f}, штраф {r['pen']:+.1f}%, "
              f"итог {r['win']:+.2f} мс")
    print("\n=== штраф страйдов против кратности перечитывания ===")
    for r in sorted(rows, key=lambda x: x["reread"]):
        print(f"  перечит. {r['reread']:6.1f}  K+V {r['kv_mb']:7.1f} МБ  штраф {r['pen']:+6.1f}%")


if __name__ == "__main__":
    main()
