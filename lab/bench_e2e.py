"""bench_e2e.py — замер fable_attn через публичный Python-API, как его получит потребитель.

Меряем не синтетику на HIP, а вызов flash_attn_func / flash_attn_varlen_func /
flash_attn_with_kvcache на torch-тензорах: со всеми накладными обёртки и конверсий.

VRAM: каждая клетка гоняется ОТДЕЛЬНЫМ процессом, поэтому память освобождается полностью
и кэширующий аллокатор torch ничего не копит. Перед запуском клетки её потребность считается
по формуле, и если она выше потолка — клетка пропускается, а не уводит машину в спилл.
Фактический пик печатается по факту (torch.cuda.max_memory_allocated), а не предсказывается.

Запуск всей матрицы:   python lab/bench_e2e.py [потолок_ГБ]
Запуск одной клетки:   ... lab/bench_e2e.py --case <json>     (внутренний режим)
"""
import os, sys, json, time, subprocess, itertools, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "package")))

PEAK = 36.9      # сырой темп v_dot2 на RX 6800 XT, TFLOP/s
HBM  = 512.0     # ГБ/с
VRAM_CAP_GB = 6.0   # грубый предфильтр; настоящий сторож — счётчики Windows ниже
SPILL_LIMIT_MB = 128.0   # Non Local выше этого = выселение в системную память, клетку глушим


# ------------------------------------------------- сторож VRAM (счётчики Windows)
# torch про спилл ВРЁТ: max_memory_allocated не видит, как драйвер выселяет память
# в системную. Реальная картина — только в счётчиках Windows по pid:
#   Local Usage     — сколько реально в VRAM карты
#   Non Local Usage — сколько выселено в системную память (это и есть спилл)
PS_QUERY = (
    "$s=(Get-Counter '\\GPU Process Memory(pid_{pid}*)\\Local Usage',"
    "'\\GPU Process Memory(pid_{pid}*)\\Non Local Usage' -EA SilentlyContinue)"
    ".CounterSamples;"
    "if($s){{$l=($s|?{{$_.Path -like '*local usage'}}|measure -Property CookedValue -Sum).Sum;"
    "$n=($s|?{{$_.Path -like '*non local usage'}}|measure -Property CookedValue -Sum).Sum;"
    "Write-Output ((($l-$n)/1MB).ToString('0.0')+' '+(($n)/1MB).ToString('0.0'))}}"
)


def gpu_mem_mb(pid):
    """(в VRAM, выселено) в МБ по счётчикам Windows. (None,None) если счётчика ещё нет."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            PS_QUERY.format(pid=pid)],
                           capture_output=True, text=True, timeout=20)
        parts = r.stdout.strip().split()
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None, None


class SpillGuard(threading.Thread):
    """Следит за дочерним процессом. Как только выселение превысило порог — глушит его."""

    def __init__(self, proc, limit_mb=SPILL_LIMIT_MB):
        super().__init__(daemon=True)
        self.proc, self.limit = proc, limit_mb
        self.peak_vram = 0.0
        self.peak_spill = 0.0
        self.killed = False
        self._ev = threading.Event()

    def run(self):
        while not self._ev.is_set() and self.proc.poll() is None:
            loc, non = gpu_mem_mb(self.proc.pid)
            if loc is not None:
                self.peak_vram = max(self.peak_vram, loc)
                self.peak_spill = max(self.peak_spill, non)
                if non > self.limit:
                    self.killed = True
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    return
            self._ev.wait(1.0)

    def stop(self):
        self._ev.set()


# ---------------------------------------------------------------- оценка VRAM
def est_gb(c):
    """Грубая оценка ПОТРЕБНОСТИ клетки в ГБ. Заведомо с запасом — она только для отсева,
    настоящий пик печатается после прогона."""
    el = 2  # fp16/bf16
    t = c["type"]
    if t in ("forward", "varlen"):
        B, H, S, D, Hkv = c["B"], c["H"], c["S"], c["D"], c["Hkv"]
        q = B * S * H * D * el
        kv = 2 * B * S * Hkv * D * el
        return (q * 2 + kv + B * H * S * 4) / 1e9        # q, out, k, v, lse
    if t == "backward":
        B, H, S, D = c["B"], c["H"], c["S"], c["D"]
        one = B * S * H * D * el
        return (one * 8 + B * H * S * 4 * 2) / 1e9       # q,k,v,+grads,out,dout,воркспейс
    if t == "decode":
        ctx, Hkv, D = c["ctx"], c["Hkv"], c["D"]
        return (2 * ctx * Hkv * D * el) / 1e9 * 1.3
    return 0.0


# ---------------------------------------------------------------- одна клетка
def run_case(c):
    import torch
    import fable_attn
    from fable_attn import (flash_attn_func, flash_attn_varlen_func,
                            flash_attn_with_kvcache)
    dev = "cuda"
    dt = torch.float16 if c["dt"] == "fp16" else torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    def timeit(fn, iters, warmup=2):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t = c["type"]
    out = {"ms": None, "flops": None, "bytes": None}

    if t == "forward":
        B, H, S, D, Hkv = c["B"], c["H"], c["S"], c["D"], c["Hkv"]
        q = torch.randn(B, S, H,   D, device=dev, dtype=dt, generator=g)
        k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt, generator=g)
        v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt, generator=g)
        win = (c["win"], 0) if c["win"] > 0 else (-1, -1)
        f = lambda: flash_attn_func(q, k, v, causal=c["causal"], window_size=win)
        out["ms"] = timeit(f, iters=3 if S >= 8192 else 5)
        fl = 4.0 * B * H * S * S * D
        out["flops"] = fl / 2 if (c["causal"] or c["win"] > 0) else fl

    elif t == "varlen":
        H, D, lens = c["H"], c["D"], c["lens"]
        tot = sum(lens)
        cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=dev, dtype=torch.int32)
        q = torch.randn(tot, H, D, device=dev, dtype=dt, generator=g)
        k = torch.randn(tot, H, D, device=dev, dtype=dt, generator=g)
        v = torch.randn(tot, H, D, device=dev, dtype=dt, generator=g)
        mx = max(lens)
        f = lambda: flash_attn_varlen_func(q, k, v, cu, cu, mx, mx, causal=False)
        out["ms"] = timeit(f, iters=5)
        out["flops"] = sum(4.0 * H * L * L * D for L in lens)

    elif t == "backward":
        B, H, S, D = c["B"], c["H"], c["S"], c["D"]
        q = torch.randn(B, S, H, D, device=dev, dtype=dt, generator=g, requires_grad=True)
        k = torch.randn(B, S, H, D, device=dev, dtype=dt, generator=g, requires_grad=True)
        v = torch.randn(B, S, H, D, device=dev, dtype=dt, generator=g, requires_grad=True)
        go = torch.randn(B, S, H, D, device=dev, dtype=dt, generator=g)

        def f():
            o = flash_attn_func(q, k, v, causal=c["causal"])
            o.backward(go, retain_graph=True)
            q.grad = k.grad = v.grad = None
        out["ms"] = timeit(f, iters=3)
        fl = 4.0 * B * H * S * S * D
        out["flops"] = 2.5 * (fl / 2 if c["causal"] else fl)

    elif t == "decode":
        ctx, H, Hkv, D, page = c["ctx"], c["H"], c["Hkv"], c["D"], c["page"]
        nblk = (ctx + page - 1) // page
        q  = torch.randn(1, 1, H, D, device=dev, dtype=dt, generator=g)
        kc = torch.randn(nblk, page, Hkv, D, device=dev, dtype=dt, generator=g)
        vc = torch.randn(nblk, page, Hkv, D, device=dev, dtype=dt, generator=g)
        bt = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, nblk)
        cs = torch.tensor([ctx], device=dev, dtype=torch.int32)
        f = lambda: flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs,
                                            block_table=bt, causal=False)
        out["ms"] = timeit(f, iters=10)
        out["bytes"] = 2 * Hkv * ctx * D * 2

    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


# ---------------------------------------------------------------- матрица
def matrix():
    cs = []
    for dt in ("fp16", "bf16"):
        for B, H, S, D, Hkv, causal, win in [
            (1, 32,   512, 128, 32, False, -1),
            (1, 32,  2048, 128, 32, False, -1),
            (1, 32,  8192, 128, 32, False, -1),
            (1, 32, 16384, 128, 32, False, -1),
            (1, 32,  8192, 128, 32, True,  -1),
            (1, 32,  8192, 128,  8, False, -1),   # GQA4
            (1, 32,  8192, 128,  4, False, -1),   # GQA8
            (1, 32,  8192,  64, 32, False, -1),
            (1, 16,  8192, 256, 16, False, -1),
            (8, 32,  2048, 128, 32, False, -1),
            (1, 32,  8192, 128, 32, True, 512),   # sliding-window
        ]:
            cs.append(dict(type="forward", dt=dt, B=B, H=H, S=S, D=D, Hkv=Hkv,
                           causal=causal, win=win))
        for lens in ([512, 1024, 2048, 256], [4096, 2048, 1024, 1024]):
            cs.append(dict(type="varlen", dt=dt, H=32, D=128, lens=lens,
                           B=1, S=sum(lens), Hkv=32))
        for S, causal in ((2048, False), (4096, False), (4096, True)):
            cs.append(dict(type="backward", dt=dt, B=1, H=32, S=S, D=128, causal=causal))
        for ctx, Hkv in ((2048, 32), (8192, 8), (32768, 8), (131072, 8)):
            cs.append(dict(type="decode", dt=dt, ctx=ctx, H=32, Hkv=Hkv, D=128, page=32))
    return cs


def label(c):
    t = c["type"]
    if t == "forward":
        n = "forward" + ("-causal" if c["causal"] else "") + ("-window" if c["win"] > 0 else "")
        return n, f"B{c['B']} H{c['H']}/{c['Hkv']} S{c['S']} D{c['D']}"
    if t == "varlen":
        return "varlen", f"{len(c['lens'])} посл., итого {sum(c['lens'])}"
    if t == "backward":
        return "backward" + ("-causal" if c["causal"] else ""), f"H{c['H']} S{c['S']} D{c['D']}"
    return "декод", f"ctx{c['ctx']} H{c['H']}/{c['Hkv']}"


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else VRAM_CAP_GB
    print(f"потолок VRAM на клетку: {cap:.1f} ГБ (клетка выше — пропуск)")
    print(f"сырой потолок v_dot2 ~{PEAK} TFLOP/s, HBM {HBM} ГБ/с\n")
    print(f"  {'путь':22s} {'форма':26s} {'dt':5s} {'время':>10s}  {'скорость':>16s} {'исп.':>6s} {'VRAM':>7s}")
    rows = []
    for c in matrix():
        name, shape = label(c)
        need = est_gb(c)
        if need > cap:
            print(f"  {name:22s} {shape:26s} {c['dt']:5s} пропуск: нужно ~{need:.1f} ГБ")
            continue
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                 "--case", json.dumps(c)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        guard = SpillGuard(proc)
        guard.start()
        try:
            so, se = proc.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill(); so, se = proc.communicate()
        guard.stop(); guard.join(timeout=3)

        if guard.killed:
            print(f"  {name:22s} {shape:26s} {c['dt']:5s} ОСТАНОВЛЕНО: спилл "
                  f"{guard.peak_spill:.0f} МБ в системную память (в VRAM было {guard.peak_vram:.0f} МБ)")
            continue
        line = [l for l in so.splitlines() if l.startswith("RESULT ")]
        if not line:
            err = (se.strip().splitlines() or ["нет вывода"])[-1]
            print(f"  {name:22s} {shape:26s} {c['dt']:5s} ОШИБКА: {err[:70]}")
            continue
        o = json.loads(line[0][7:])
        vram = f"{guard.peak_vram:5.0f}МБ" if guard.peak_vram else f"{o['peak_gb']*1000:5.0f}МБ*"
        if o.get("flops"):
            tf = o["flops"] / (o["ms"] * 1e-3) / 1e12
            print(f"  {name:22s} {shape:26s} {c['dt']:5s} {o['ms']:8.3f} мс  "
                  f"{tf:8.1f} TFLOP/s {100*tf/PEAK:5.1f}% {vram}")
            rows.append((name, shape, c["dt"], tf))
        else:
            gb = o["bytes"] / (o["ms"] * 1e-3) / 1e9
            print(f"  {name:22s} {shape:26s} {c['dt']:5s} {o['ms']:8.3f} мс  "
                  f"{gb:8.1f} ГБ/с    {100*gb/HBM:5.1f}% {vram}")
    if rows:
        print("\n--- худшие клетки по утилизации ---")
        for r in sorted(rows, key=lambda x: x[3])[:5]:
            print(f"  {r[0]:22s} {r[1]:26s} {r[2]:5s} {r[3]:6.1f} TFLOP/s  {100*r[3]/PEAK:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(run_case(json.loads(sys.argv[2]))))
    else:
        main()
