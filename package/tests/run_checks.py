"""run_checks.py — один вход для всех проверок пакета, без pytest (его нет в окружении).

Гоняет три гейта подряд и возвращает ненулевой код, если хоть один красный:
  1. lab/parity_test.py    — то, что пакет умел раньше (все head_dim, маски, GQA, varlen,
                             страничный кэш, backward, детерминизм, install_as_flash_attn);
  2. lab/parity_new_api.py — то, что закрыто по PLAN.md §3 (attention_chunk, qv, seqused_*,
                             cache_leftpad, cache_batch_idx, return_attn_probs,
                             return_softmax_lse, combine, bert_padding, отказ по FP8);
  3. package/tests/test_fable_attn.py под pytest — если pytest вдруг доступен.

Раньше этот файл импортировал `_ref_attention` и `_HAS_C` из flash_attn_interface, которых
там нет уже давно: он падал на импорте, то есть не проверял ничего. Дублировать эталон здесь
незачем — он есть в обоих гейтах lab/.

Запуск: python package/tests/run_checks.py
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATES = [
    ("паритет: то, что работало раньше", os.path.join(ROOT, "lab", "parity_test.py")),
    ("паритет: закрытые дыры API",       os.path.join(ROOT, "lab", "parity_new_api.py")),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for name, path in GATES:
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
        r = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
        if r.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 70}\n== pytest (если установлен)\n{'=' * 70}")
    have_pytest = subprocess.run([sys.executable, "-c", "import pytest"],
                                 capture_output=True).returncode == 0
    if not have_pytest:
        # Отсутствие pytest — не красный гейт: обе проверки выше уже прошли своим ходом.
        print("pytest не установлен в этом окружении — пропуск "
              "(покрытие даёт lab/parity_test.py + lab/parity_new_api.py)")
    else:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            os.path.join(ROOT, "package", "tests", "test_fable_attn.py")],
                           cwd=ROOT, env=env)
        if r.returncode not in (0, 5):   # 5 = тестов не нашлось
            failed.append("pytest test_fable_attn.py")

    print()
    if failed:
        print(f"*** КРАСНЫЕ ГЕЙТЫ: {', '.join(failed)}")
        return 1
    print("ВСЕ ГЕЙТЫ ПАКЕТА ЗЕЛЁНЫЕ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
