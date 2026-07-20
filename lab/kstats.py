"""kstats.py — регистры/LDS/спиллы конкретного инстанцирования ядра из ISA.

ЗАЧЕМ ОТДЕЛЬНЫЙ ФАЙЛ. Метаданные в .s — это YAML-список, поля внутри записи идут ПО АЛФАВИТУ:

    .group_segment_fixed_size: ...
    .name:            <имя ядра>
    ...
    .vgpr_count:      ...
    .vgpr_spill_count: ...

то есть `.vgpr_count` стоит ПОСЛЕ `.name`. Я читала его через `grep -B30 '.name: <ядро>'`,
то есть брала числа СОСЕДНЕГО ядра, и полдня «ловила регрессию», которой не было: стоило
добавить новые head_dim — сменился порядок записей, сменился и сосед. Прибор врал, не код.

Запуск:
    hipcc -S --cuda-device-only --offload-arch=gfx1030 -O3 -ffast-math -std=c++17 \\
          kernels/fa_forward.hip -o /tmp/t.s
    python lab/kstats.py /tmp/t.s fa_forward_k          # все инстанцирования по подстроке
    python lab/kstats.py /tmp/t.s <полное_искажённое_имя>

Горячее ядро forward (fp16, HD=128, BR=128, BC=64, G=1, CAUSAL, FEAT=off) —
    _Z12fa_forward_kI6DT_F16Li128ELi128ELi64ELi1ELb1ELb0EEv9FwdParams
эталон на 2026-07-20: VGPR 227, спиллов 0, SGPR 48, LDS 65536.
"""
import re
import sys

FIELDS = ("vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
          "group_segment_fixed_size", "private_segment_fixed_size")


def stats(text, name):
    """Числа ИЗ ЗАПИСИ ядра name (ищем вперёд от .name, а не назад)."""
    i = text.find(".name:")
    out = None
    while i >= 0:
        line_end = text.find("\n", i)
        if name in text[i:line_end]:
            seg = text[i:i + 4000]
            out = {}
            for f in FIELDS:
                m = re.search(r"\." + f + r":\s*(\d+)", seg)
                out[f] = int(m.group(1)) if m else None
            return out
        i = text.find(".name:", line_end)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    pat = sys.argv[2]
    names = sorted({m.group(1) for m in re.finditer(r"\.name:\s+(\S+)", text) if pat in m.group(1)})
    if not names:
        print(f"ядер по подстроке {pat!r} не найдено")
        return 1
    for n in names:
        s = stats(text, n)
        sp = s["vgpr_spill_count"]
        flag = "  <-- СПИЛЛЫ" if sp else ""
        print(f"  VGPR {s['vgpr_count']:>4}  спиллы {sp:>3}  SGPR {s['sgpr_count']:>4}  "
              f"LDS {s['group_segment_fixed_size']:>6}  scratch {s['private_segment_fixed_size']:>5}"
              f"  {n}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
