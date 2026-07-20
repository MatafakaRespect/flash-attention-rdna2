"""sweep_rule_check.py — проверка правила выбора раскладки K/V на НОВЫХ точках.

Правило выведено по свипу sweep_strides.py и читается физически:

    кратность перечитывания K/V = gqa * ceil(SQ/BR),  gqa = Hq/Hkv, BR = 128 (q-тайл),
    для causal вдвое меньше (workgroup читает в среднем половину ключей)

    кратность >= 128  ->  выгодно один раз скопировать K/V в плотный вид
    кратность <  128  ->  выгоднее читать по страйдам, копия не окупается

Смысл: штраф за разбросанное чтение платится на КАЖДОМ перечитывании, копия — один раз.

Здесь проверяются точки, по которым правило НЕ выводилось: другие gqa, другой head_dim,
SQ != SK, backward. Плюс отдельно меряется backward, потому что он читает q/k/v своим ядром.

Запуск: python lab/sweep_rule_check.py
"""
import os, sys, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package")))

import torch
from fable_attn import _C

DEV = "cuda"
BR = 128


def rule_copy(SQ, Hq, Hkv, causal):
    """True => копировать K/V в плотный вид."""
    r = math.ceil(SQ / BR) / (2 if causal else 1)
    return r * (Hq / Hkv) >= 128


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def case(B, Hq, Hkv, SQ, SK, D, causal, iters=3):
    dt = torch.float16
    q = torch.randn(B, SQ, Hq, D, device=DEV, dtype=dt)
    k = torch.randn(B, SK, Hkv, D, device=DEV, dtype=dt)
    v = torch.randn(B, SK, Hkv, D, device=DEV, dtype=dt)
    sc = 1.0 / math.sqrt(D)
    qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    kh, vh = ks.contiguous(), vs.contiguous()

    td = ts = tc = 0.0
    for _ in range(3):   # чередуем ветки: тепловой дрейф не должен осесть в одной
        td += timeit(lambda: _C.fwd(qs, kh, vh, sc, causal, -1, 0.0, None, 0.0, 0, 0), iters)
        ts += timeit(lambda: _C.fwd(qs, ks, vs, sc, causal, -1, 0.0, None, 0.0, 0, 0), iters)
        tc += timeit(lambda: (ks.contiguous(), vs.contiguous()), iters)
    td /= 3; ts /= 3; tc /= 3

    reread = math.ceil(SQ / BR) / (2 if causal else 1) * (Hq / Hkv)
    want_copy = rule_copy(SQ, Hq, Hkv, causal)
    real_copy = (td + tc) < ts            # что на самом деле быстрее
    agree = want_copy == real_copy
    diff = abs((td + tc) - ts)
    print(f"  B{B} H{Hq}/{Hkv} SQ{SQ} SK{SK} D{D} {'caus' if causal else 'full'} | "
          f"кратн. {reread:6.0f} | копия+плотно {td+tc:8.3f}  страйд {ts:8.3f} | "
          f"правило: {'КОПИЯ' if want_copy else 'СТРАЙД'}  факт: {'КОПИЯ' if real_copy else 'СТРАЙД'} "
          f"({diff:.2f} мс) {'ОК' if agree else '*** РАСХОЖДЕНИЕ ***'}")
    del q, k, v, qs, ks, vs, kh, vh
    torch.cuda.empty_cache()
    return agree, diff, want_copy, real_copy


def main():
    print("Правило: копировать K/V, если gqa*ceil(SQ/BR) (÷2 для causal) >= 128\n")
    print("--- проверочные точки: другие gqa (правило выводилось почти только на gqa=1) ---")
    res = []
    res.append(case(1, 32, 16, 4096, 4096, 128, False))    # кратн. 64  -> страйд
    res.append(case(1, 32,  8, 4096, 4096, 128, False))    # кратн. 128 -> копия
    res.append(case(1, 32,  4, 2048, 2048, 128, False))    # кратн. 128 -> копия
    res.append(case(1, 32, 16, 12288, 12288, 128, False))  # кратн. 192 -> копия
    res.append(case(1, 32,  8, 8192, 8192, 128, True))     # кратн. 256 -> копия
    res.append(case(1, 32,  4, 8192, 8192, 128, True))     # кратн. 512 -> копия

    print("\n--- другой head_dim ---")
    res.append(case(1, 32, 32, 8192, 8192, 64, False))     # кратн. 64  -> страйд
    res.append(case(1, 32,  8, 8192, 8192, 64, False))     # кратн. 256 -> копия
    res.append(case(1, 16, 16, 8192, 8192, 256, False))    # HD=256, BR=64 -> кратн. 128

    print("\n--- SQ != SK ---")
    res.append(case(1, 32, 32, 1024, 8192, 128, False))    # кратн. 8   -> страйд
    res.append(case(1, 32,  4, 16384, 1024, 128, False))   # кратн. 1024-> копия

    print("\n--- батч ---")
    res.append(case(4, 32, 32, 2048, 2048, 128, False))    # кратн. 16  -> страйд
    res.append(case(2, 32,  8, 8192, 8192, 128, False))    # кратн. 256 -> копия

    bad = [r for r in res if not r[0]]
    print(f"\n=== расхождений правила с фактом: {len(bad)} из {len(res)} ===")
    if bad:
        for a, d, w, rl in bad:
            print(f"  правило говорило {'КОПИЯ' if w else 'СТРАЙД'}, а быстрее "
                  f"{'КОПИЯ' if rl else 'СТРАЙД'}, цена ошибки {d:.2f} мс")
        worst = max(b[1] for b in bad)
        print(f"  худшая цена ошибки: {worst:.2f} мс")
    else:
        print("  правило угадало везде")


if __name__ == "__main__":
    main()
