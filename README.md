# flash-attention-rdna2

Exact FlashAttention for **AMD RDNA2** (gfx1030 / RX 6800 XT), written in HIP.
Drop-in for `flash_attn`: install it and unmodified CUDA-oriented code just runs.

The upstream [flash-attention](https://github.com/Dao-AILab/flash-attention) does not support
RDNA2 — no matrix cores, no `cp.async`, a different LDS budget. These kernels are written for
that hardware from scratch, on `v_dot2_f32_f16`.

**Exact, not approximate.** No int8/FP8 quantisation anywhere on the default path. The output
matches an fp64 CPU oracle to the fp16 rounding floor (`rel_rms ≈ 2.4e-4`), which is the whole
point of FlashAttention: you can drop it into any model without re-validating quality.

> Source comments and design notes are in Russian; the API and this file are in English.

## What it gives you

```python
import torch
from flash_attn import flash_attn_func          # ← unmodified consumer code

q = torch.randn(1, 8192, 32, 128, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, q, q, causal=True)     # (batch, seqlen, nheads, headdim)
```

Stock `transformers` works with no changes at all:

```python
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
```

## Measured on RX 6800 XT

End-to-end through the public Python API (not a synthetic kernel loop), fp16.
The card's raw `v_dot2` issue rate is **36 TFLOP/s** — that is the real ceiling, measured with
a shader-clock counter, not the 41.5 from the spec sheet (that number assumes a boost clock the
card does not hold under fp16 load).

| path | shape | measured |
|---|---|---|
| forward | S=16384, H=32, D=128 | **19.8 TFLOP/s** (55% of the ceiling) |
| forward, causal | S=8192 | 18.6 TFLOP/s |
| forward, causal | S=4096 | 17.2 TFLOP/s |
| decode, paged KV | ctx=32768, GQA 4 | **505–528 GB/s ≈ 99% of HBM bandwidth** |
| backward, causal | S=4096 | 3.85 TFLOP/s |
| sliding window | S=8192, window=512 | 5.6 ms (tiles outside the window are skipped) |

Peak VRAM for a forward at S=8192, H=32, D=128 is **0.27 GB** — the S×S score matrix is never
materialised.

## API coverage

Everything from FA-2 and FA-3, same names, same arguments, same results:

`flash_attn_func`, `flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func`,
`flash_attn_varlen_func`, `flash_attn_varlen_qkvpacked_func`, `flash_attn_varlen_kvpacked_func`,
`flash_attn_with_kvcache`, `flash_attn_combine`, plus `flash_attn.bert_padding`,
`flash_attn.layers.rotary`, `flash_attn.modules.mha`, `flash_attn.ops.triton.layer_norm`.

FA-3 has a different positional argument order (4th is `softmax_scale`, not `dropout_p`), so the
two conventions are kept apart: `import flash_attn` is FA-2, `import flash_attn_interface` is FA-3.

Supported: causal, sliding window `(left, right)` including two-sided, GQA/MQA, `softcap`,
`alibi`, dropout with a deterministic Philox mask, varlen with `cu_seqlens` and `seqused_*`,
paged KV cache, rotary (NeoX and interleaved, partial `rotary_dim`), `attention_chunk`,
`qv`, `cache_leftpad`, `cache_batch_idx`, `return_softmax_lse`, `return_attn_probs`,
multi-token decode (speculative decoding / chunked prefill), head_dim up to 256 (any value —
non-instantiated sizes are zero-padded, as upstream does), fp16 / bf16, and fp32 as a bonus
that upstream does not have.

**Deliberately not supported:** FP8 descales (`q_descale` / `k_descale` / `v_descale`). RDNA2 has
no tensor cores, so FP8 buys no speed and only costs accuracy — and an inexact kernel is not
FlashAttention. The arguments are accepted and rejected with a clear message rather than
silently ignored.

`deterministic=True` costs nothing here: the backward has no atomics by construction, so every
run is bit-for-bit reproducible either way.

## Requirements

- AMD GPU with **gfx1030** (RX 6800 / 6800 XT / 6900 XT). Other architectures need a rebuild
  with the right `--offload-arch`; the kernels are not portable to NVIDIA at all.
- ROCm with a PyTorch build for it.
- Build scripts are PowerShell/Windows (MSVC + hipcc). The kernels themselves are plain HIP.

## Install

```powershell
package\install.ps1 <path to your python.exe>
```

This installs two distributions:

| | |
|---|---|
| `fable_attn` | the engine: HIP kernels, dispatcher, tests |
| `flash_attn` | zero lines of logic — only upstream's names, pointing at `fable_attn` |

The second one exists because real consumers ask for the **distribution**, not the module:
`transformers` calls `importlib.metadata.version("flash_attn")` and a module-level `__version__`
does not satisfy it. Use `-NoCompat` to install the engine alone.

## Tests

Correctness is checked against an **fp64 CPU oracle**, not against another GPU implementation:

```powershell
kernels\build.ps1                  # builds and runs the kernel gates
python package\tests\run_checks.py # API parity gates
```

- `kernels/fa_forward.hip`, `kernels/backward.hip`, `kernels/fa_decode.hip` each carry a
  standalone gate covering every dtype, head_dim, mask, window, varlen and paged-cache case.
- `lab/parity_test.py` and `lab/parity_new_api.py` check the public Python API against an exact
  torch reference, cell by cell.
- `lab/` also holds the measurement harnesses: `bench_e2e.py`, `sweep_strides.py`, `kstats.py`
  (register/LDS/spill counts straight from the ISA).

## Design notes

`PLAN.md` is the working log: what was tried, what was measured, and what was rejected **by
measurement** rather than by argument. Some findings that cost real time to learn:

- On RDNA2 a register double-buffer for the KV tile — the standard NVIDIA trick — is poison:
  32 extra VGPRs blow up the register file and cost **3×**. LDS barriers are cheaper.
- LDS bank conflicts were worth **58%**, and not one of them was visible by reading the code.
  They only show up if you compute the bank for every access explicitly, or sweep configurations.
- The largest remaining win was not in the kernel at all: the Python wrapper copied q, k and v
  on every call (9–30% of wall time). Removing those copies gave **+35%** on some shapes and cut
  peak VRAM by 43%. Measure the whole path the caller takes, not just the kernel.

## Licence

Apache 2.0.
