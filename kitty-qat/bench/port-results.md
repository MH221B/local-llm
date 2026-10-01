# Kitty port results

Spec: `../../docs/superpowers/specs/2026-09-30-kitty-qat-port-design.md`
Plan: `../../docs/superpowers/plans/2026-09-30-kitty-qat-port.md`

## Environment

```
torch 2.14.0+cpu
transformers 5.17.0
DynamicLayer ok
['chunked_attention', 'deepseek_sparse_attention', 'full_attention', 'linear_attention', 'qwen_sparse_attention', 'sliding_attention']
```

Local machine, CPU-only build of torch (Radeon 890M has no ROCm/WSL GPU stack wired up).
All offline checks (1-3) ran on this machine; check 4 ran with `--max-new-tokens 32` on
CPU (the plan's default 64 discarded to keep CPU wall-clock sane).

Third-party hashes (Kitty `main` at fetch time, 2026-09-30):
- `utils_quant.py` `6bb6df3bbfb702793cff79d712f908f7dee89ada66d3bffc399718b218bfc27a`
- `kitty_simulate.py` `c63d8bff512594a7bd5412b8c201e8a706b2757abbfabd8a95436b9f343ee133`

## Model facts

From the live HF config, all agreeing with spec §5:

```
head_dim: 256
num_key_value_heads: 4
mtp_num_hidden_layers: 1
full attention layers: [3, 7, 11, 15, 19, 23, 27, 31]
```

HF checkpoint `model.safetensors.index.json`: 738 keys total, 15 `mtp.*` (Spec 2 must
carry these across; `blk.32.nextn.*` is the same head in GGUF naming).

`layer_types` in full (32 entries, 3 linear : 1 full repeating):
`['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 8`

## Cache construction seam (transformers 5.17.0)

`DynamicCache(config=model.config)` works: `__init__(self, ddp_cache_data=None,
config: PreTrainedConfig | None = None, offloading=False, offload_only_non_sliding=False)`,
and `layer_types` dispatch through `DYNAMIC_LAYER_TYPE_MAPPING`:
indices 0-2 are `LinearAttentionLayer`, index 3 is `DynamicLayer`, alternating
linear/full through layer 7. `build_kitty_cache`'s `isinstance` assert fires never.

## Checks

| Check | Result | Notes |
|---|---|---|
| 1. Golden trace | PASS | prefill=200 decode=300, tensor equality of stored + returned K/V at every step; pages=3, value_blocks=301 |
| 2. Structural | PASS | layers [3,7,11,15,19,23,27,31], head_dim 256, kv heads 4 |
| 3. Quantization exercised | PASS | quantized region differs from fp16 route; 160-token boundary quantizes nothing |
| 4. Baseline | PASS (weakly) | 3/3 = 3/3 = 3/3 across fp16 / int2 (0.1) / int2 (0.2); prompts are past the ceiling, outputs byte-identical across configs |

### What the golden trace caught

One real port bug before any model weights were involved. First failing produced
`step 1: stored V diverged`. Cause: the port's decode branch fake-quantized a whole
128-token V block per K-quantization event; Kitty's reference quantizes **exactly one
token per decode step** (`current_value_cache[:, :, -buffer-1:-buffer, :]`, the comment
reads "quantizing a Token each Decoding Step"). Full arithmetic after the fix, verified
against the trace: prefill=200 yields 1 prefill V block `[32:72)` + 300 single-token
decodes = 301 blocks, and K pages at 1 prefill (`[32:160)`) + 2 decode K events =
3 pages. Both counters appear in the passing run's output.

One adapter shim, recorded per spec §8: transformers 5.17's `DynamicCache` no longer
carries the legacy `key_cache` / `value_cache` PyObject lists Kitty's `update()` mutates
(store moved to `cache.layers[i].keys/.values`). Not-evaluated shim
(`reference.key_cache = []`, `reference.value_cache = []`) is set in
`scripts/equiv_check.py` only; no third-party line is edited.

## Baseline

Machine: local CPU (torch 2.14.0+cpu), `--max-new-tokens 32`.

| Config | promote_ratio | kbits/vbits | Passed |
|---|---:|---|---:|
| fp16 | 0.0 | 16/16 | 3/3 |
| kitty-int2 | 0.1 | 2/2 | 3/3 |
| kitty-pro-int2 | 0.2 | 2/2 | 3/3 |

Raw per-prompt detail is in `bench/baseline-raw.json`. `quantized_pages = 8` for every
quantizing config (1 prefill page + 7 decode K-quant events; `kitty_simulate.py`-equivalent
V decay also active each decode step). Prompt tokenization and the assert confirm every
run exceeded the 160-token dead zone.

**Key observation:** thinking disabled via the chat template after the first run (the
32-token budget was otherwise eaten whole by `` Thinking Process:``). With it off, the
three configs produce **byte-identical completions** (`43/Paris/72`): at this depth
(~253 cached tokens) and with only 8 of 32 layers quantized, 2-bit KV leaves visible
output unchanged when the answer is one token deep. The weak gate
(`mean(0.2) >= mean(0.0)`) is satisfied, but as nothing discriminated the configs, this
run proves the harness works, not that quantization is free — both facts belong in the
expected Spec 2 picture (fine-tunes are trained on the losses, not on pass rates).

## What the results do and do not show

- Only 8 of 32 layers form a KV cache, so the effect is diluted relative to the
  paper, which quantized every layer of dense Qwen3/LLaMA3 models. A small curve
  is expected and is not a port defect.
- n=3 prompts is a smoke signal, not a measurement.
- The gate is deliberately weak (`mean(0.2) >= mean(0.0) - noise`, quantization
  provably exercised). A strict monotonicity gate on this scale would be a coin flip.
- The post-quant return semantics reproduced here mean every eval config attends on
  one-step-lagged quantized pages — matching the reference, on the same numbers Spec 2
  builds on.

## Spec 2 pre-Colab gate (2026-10-01)

`scripts/smoke_trainstep.py` (CPU): trainable 458,752 params / 0.0109%, fresh-cache
training step asserts all green (3 K pages + 1 prefill V block per swapped layer,
512-token windows), adapter grads finite and nonzero, KL finite across 2 steps
(3.44 → 1.34, decreasing). Green before any Colab spend, per spec §8.
