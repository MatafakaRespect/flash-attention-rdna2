# flash-attention-rdna2

Notes:
- The Linux installation scripts aren't available yet.
- I needed for this to work on Linux specifically, and I used AI to do the work (only writing the equivalents of the .ps1 scripts in the repo). Rest of the work remains untouched and just as it is in the original repo.

Exact FlashAttention for **AMD RDNA2** (gfx1030 / RX 6800 XT), written in HIP.
Provides the `flash_attn` module: code written against the CUDA build runs unmodified.

The upstream [flash-attention](https://github.com/Dao-AILab/flash-attention) does not support
RDNA2 — no matrix cores, no `cp.async`, a different LDS budget. These kernels are written for
that hardware from scratch, on `v_dot2_f32_f16`.

The kernels are exact. There is no int8/FP8 quantisation on the default path; output matches an
fp64 CPU oracle to the fp16 rounding floor (`rel_rms ≈ 2.4e-4`), so results are equivalent to a
standard attention implementation in the same dtype.

Source comments and design notes are in Russian; the API and this file are in English.

## Usage

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 8192, 32, 128, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, q, q, causal=True)     # (batch, seqlen, nheads, headdim)
```

Integration with `transformers`:

```python
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
```

## Measured on RX 6800 XT

End-to-end through the public Python API (not a synthetic kernel loop), fp16.
The card's raw `v_dot2` issue rate is **36 TFLOP/s**, measured with a shader-clock counter. The
41.5 TFLOP/s figure from the spec sheet assumes a boost clock the card does not sustain under
fp16 load and is not used as a reference here.

| path | shape | measured |
|---|---|---|
| forward | S=16384, H=32, D=128 | **19.8 TFLOP/s** (55% of the ceiling) |
| forward, causal | S=8192 | 18.6 TFLOP/s |
| forward, causal | S=4096 | 17.2 TFLOP/s |
| decode, paged KV | ctx=32768, GQA 4 | **505–528 GB/s ≈ 99% of HBM bandwidth** |
| backward, causal | S=4096 | 3.85 TFLOP/s |
| sliding window | S=8192, window=512 | 5.6 ms (tiles outside the window are skipped) |

Peak VRAM for a forward at S=8192, H=32, D=128 is **0.27 GB**; the S×S score matrix is never
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
multi-token decode (speculative decoding / chunked prefill), head_dim up to 256 (any value;
sizes without a dedicated instantiation are zero-padded, as upstream does), fp16 and bf16.
fp32 is also supported, which upstream does not provide.

Not supported: FP8 descales (`q_descale` / `k_descale` / `v_descale`). RDNA2 has no tensor cores,
so FP8 yields no throughput gain while reducing accuracy. The arguments are accepted and
rejected with an explicit error rather than ignored.

`deterministic=True` has no cost: the backward pass uses no atomics, so results are bit-for-bit
reproducible regardless of the flag.

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
| `flash_attn` | no logic; upstream's names resolving to `fable_attn` |

The second distribution is required because consumers query package metadata rather than the
module: `transformers` calls `importlib.metadata.version("flash_attn")`, which a module-level
`__version__` does not satisfy. `-NoCompat` installs the engine alone.

## Tests

Correctness is checked against an fp64 CPU oracle rather than another GPU implementation:

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

## Implementation notes

`PLAN.md` is the working log: approaches tried, measurements taken, and options rejected on the
basis of measurement. Selected results:

- A register double-buffer for the KV tile, standard practice on NVIDIA hardware, is
  counterproductive on RDNA2: the 32 additional VGPRs exhaust the register file and cost **3×**.
  LDS barriers are cheaper.
- LDS bank conflicts accounted for **58%** of the achievable gain. None were apparent from
  reading the code; they were found by computing the bank index for each access and by sweeping
  configurations.
- The largest remaining gain was outside the kernel: the Python wrapper copied q, k and v on
  every call, accounting for 9–30% of wall time. Removing those copies gave **+35%** on some
  shapes and reduced peak VRAM by 43%.

## Licence

Apache 2.0.
