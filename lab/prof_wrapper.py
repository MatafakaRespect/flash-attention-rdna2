"""prof_wrapper.py — из чего складывается время публичного вызова flash_attn_func.

Вопрос: обёртка делает .transpose(1,2).contiguous() на q, k и v — три полные копии тензора
на каждый вызов. Ядро при этом умеет произвольные страйды (нужен только stride_d==1).
Меряем, сколько стоят копии и сколько — само ядро, на страйдах обоих видов.

Запуск: python lab/prof_wrapper.py
"""
import os, sys, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package")))

import torch
from fable_attn import _C

DEV = "cuda"


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(B, H, S, D, causal, iters):
    dt = torch.float16
    q = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    k = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    v = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    sc = 1.0 / math.sqrt(D)
    cf = 1 if causal else 0

    # 1) только три копии, без ядра
    def copies():
        return (q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous())
    t_cp = timeit(copies, iters)

    # 2) ядро на ПЛОТНЫХ [B,H,S,D] (как сейчас: после копий)
    qh, kh, vh = copies()
    t_dense = timeit(lambda: _C.fwd(qh, kh, vh, sc, bool(cf), -1, 0.0, None, 0.0, 0, 0), iters)

    # 3) ядро прямо на СТРАЙДАХ транспонированного вида (без копий)
    qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    t_strided = timeit(lambda: _C.fwd(qs, ks, vs, sc, bool(cf), -1, 0.0, None, 0.0, 0, 0), iters)

    # 4) смешанный: Q на страйдах (читается один раз), K/V плотные (читаются в цикле)
    t_mixed = timeit(lambda: _C.fwd(qs, kh, vh, sc, bool(cf), -1, 0.0, None, 0.0, 0, 0), iters)

    fl = 4.0 * B * H * S * S * D / (2 if causal else 1)
    name = f"B{B} H{H} S{S} D{D} {'causal' if causal else 'full  '}"
    print(f"  {name}")
    print(f"      копии q,k,v                   {t_cp:8.3f} мс")
    print(f"      ядро на плотных               {t_dense:8.3f} мс   {fl/(t_dense*1e-3)/1e12:5.2f} TFLOP/s")
    print(f"      ядро на страйдах (без копий)  {t_strided:8.3f} мс   {fl/(t_strided*1e-3)/1e12:5.2f} TFLOP/s")
    print(f"      ядро: Q страйд, K/V плотные   {t_mixed:8.3f} мс   {fl/(t_mixed*1e-3)/1e12:5.2f} TFLOP/s")
    print(f"      ИТОГО сейчас (копии+плотное)  {t_cp+t_dense:8.3f} мс   {fl/((t_cp+t_dense)*1e-3)/1e12:5.2f} TFLOP/s")
    print(f"      ИТОГО без копий               {t_strided:8.3f} мс   "
          f"{'ЛУЧШЕ' if t_strided < t_cp+t_dense else 'ХУЖЕ'} на "
          f"{abs(t_strided-(t_cp+t_dense))/(t_cp+t_dense)*100:.1f}%")
    print()


def main():
    print("Разбор публичного вызова: копии обёртки против страйдов ядра\n")
    for B, H, S, D, causal, it in [
        (1, 32, 2048, 128, False, 20),
        (1, 32, 4096, 128, True, 10),
        (1, 32, 8192, 128, True, 5),
        (1, 32, 8192, 128, False, 5),
        (1, 32, 16384, 128, False, 3),
        (8, 32, 2048, 128, False, 5),
        (1, 32, 8192, 64, False, 5),
    ]:
        run(B, H, S, D, causal, it)


if __name__ == "__main__":
    main()
