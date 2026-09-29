# Qwen3.5-4B-MTP-Heretic i1-Q4_K_S — local validation

## Artifacts
- Model: `models\qwen35-mtp\Qwen3.5-4B-MTP-Heretic.i1-Q4_K_S.gguf` (2.45 GiB)
- Projector: `models\qwen35-mtp\Qwen3.5-4B-MTP-Heretic.mmproj-Q8_0.gguf` (0.34 GiB)
- Source: `mradermacher/Qwen3.5-4B-MTP-Heretic-i1-GGUF` / `-GGUF` (static, mmproj)

## Hardware / build
- ASUS Vivobook S16 (M5606WA), Ryzen AI 9 HX 370, Radeon 890M (gfx1150)
- 32 GB LPDDR5X-7500, 128-bit (~120 GB/s peak); VGM ~19.8 GiB
- llama.cpp **0.5.0-dev, build 11228 (4364bf723)**, Vulkan, `-fa on` (new syntax: on|off|auto)

## Step 1 — build + MTP head: PASS
- `--spec-type` includes `draft-mtp`; `--spec-draft-n-max` present (default 3)
- GGUF: version 3, 441 tensors, block range 0..32 (33 blocks)
- MTP head present: 15 `blk.32.*` tensors incl. `blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`
- 441 = 426 language + 15 MTP (vision lives in mmproj)

## Step 2 — MTP speculation: PASS (modest)

256-token greedy request, thinking off, `-c 8192 -np 1`, same prompt. Server reused across
3 reps per config; decode t/s parsed from the server log.

| Config | Rep1 | Rep2 | Rep3 | Mean | Min | Max | Acc | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (no spec) | 29.13 | 29.87 | 29.83 | **29.61** | 29.13 | 29.87 | — | — |
| draft-mtp n-max 1 | 38.99 | 37.92 | 38.32 | 38.41 | 37.92 | 38.99 | 0.78 | 1.30x |
| **draft-mtp n-max 2** | 40.74 | 39.26 | 39.52 | **39.84** | 39.26 | 40.74 | 0.65 | **1.35x** |
| draft-mtp n-max 3 | 38.82 | 38.73 | 40.37 | 39.31 | 38.73 | 40.37 | 0.57 | 1.33x |
| draft-mtp n-max 4 | 33.82 | 32.91 | 32.74 | 33.16 | 32.74 | 33.82 | 0.48 | 1.12x |
| draft-mtp n-max 6 | 8.59 | 8.70 | 8.70 | 8.66 | 8.59 | 8.70 | 0.32 | 0.29x |

Run-to-run variance is low (baseline spread 0.74 t/s, ~2.5%). No thermal drift visible across
back-to-back reps.

Observations:
- **Pick n-max 2** (~39.8 t/s, 1.35x). n-max 1/2/3 are statistically tied at 1.30-1.35x.
- Acceptance falls monotonically with n-max (0.78 -> 0.32) while the mean accepted run
  saturates near 3, so depth beyond ~2 buys nothing and costs draft+verify work.
- n-max 4 still gains (1.12x) — a single early run of 25.1 t/s was a low outlier, not a cliff.
- n-max 6 is a genuine cliff (0.29x, i.e. ~3.4x slower than baseline) and is reproducible.
- Short-context ceiling with MTP is ~40 t/s, not the 70-90 t/s the pipeline spec assumes.
- NOTE: baseline here (29.6 t/s) is above the first single-run baseline (25.8 t/s); use the
  repeated numbers.

## Step 3 — quantized KV on Vulkan: PASS

`-ctk q4_0 -ctv q4_0` loads and runs. The log labels the type explicitly, so the setting
is applied and not silently ignored:

```
llama_kv_cache: size = 4096.00 MiB (131072 cells, 8 layers, 1/1 seqs), K (f16): 2048.00 MiB, V (f16): 2048.00 MiB
llama_kv_cache: size = 1152.00 MiB (131072 cells, 8 layers, 1/1 seqs), K (q4_0):  576.00 MiB, V (q4_0):  576.00 MiB
```

- KV @131072: f16 4096 MiB -> q4_0 1152 MiB (**3.56x smaller**)
- "8 layers" confirms the hybrid layout (8 full-attn of 32); GDN recurrent state is a flat 50 MiB
- MTP + q4_0 composes: loads and decodes (~34 t/s short ctx)
- With MTP enabled all 15 blk.32 tensors are used; with MTP off all 15 are reported unused (expected)

Memory @128k / @256k (MiB):

| Component | f16 @128k | q4_0 @128k | q4_0 @256k |
|---|---:|---:|---:|
| Weights | 2932 | 2932 | 2932 |
| KV cache | 4096 | 1152 | 2304 |
| Compute buffers | 335 | 335 | 335 |
| GDN recurrent | 50 | 50 | 50 |
| Output | 1 | 1 | 1 |
| **Total** | **7.24 GiB** | **4.37 GiB** | **5.49 GiB** |

## Step 4 — needle: PASS for q4_0 at 64k and 128k (f16 control pending)

Single needle ("The access code for the Zephyr vault is 7741-KESTREL.") at 50% depth,
temperature 0, thinking off.

| Run | KV | Actual prompt | Wall | Prefill | Decode @depth | Result |
|---|---|---:|---:|---:|---:|---|
| 64k q4_0 | q4_0 | 57,318 | 219 s | 262.8 t/s | 19.43 t/s | **FOUND** |
| 128k q4_0 | q4_0 | 114,565 | 672 s | ~170 t/s | 14.18 t/s | **FOUND** |

- q4_0 KV retains recall at 128k (50% depth) — core premise holds at this dose.
- Prefill falls 262.8 -> ~170 t/s from 57k -> 114k (attention O(n^2) term), matching the
  earlier MiMo 9B curve (309 t/s @16k -> 224 t/s @64k).
- Decode at depth: 29.6 (short) -> 19.4 (57k) -> 14.2 (114k).
- CAVEAT: one needle, one depth, one sample. "FOUND" is not parity with f16. To claim
  "no degradation" needs the f16 control and ideally multiple depths/positions.

## Step 5 — vision + MTP coexistence: WORKS on build 11228

A single `llama-server` with **both** `--mmproj ...mmproj-Q8_0.gguf` **and**
`--spec-type draft-mtp --spec-draft-n-max 2` loads and runs. Log shows both subsystems active:

```
srv load_model: loaded multimodal model, '...Qwen3.5-4B-MTP-Heretic.mmproj-Q8_0.gguf'
common_speculative_init_result: creating MTP draft context against the target model ...
slot print_timing: draft acceptance = 0.71429 (10 accepted / 14 generated), mean len = 2.43
```

Vision verified with a generated control image (white bg, black "VISION OK", red ellipse):

| Request | Answer |
|---|---|
| with image | `The text is "VISION OK" and there is a red oval shape.` (correct) |
| without image | `The text is "The quick brown fox jumps over the lazy dog," and ... a red circle.` (hallucinated) |

- The with/without split proves the projector is genuinely bound, not a lucky guess.
- MTP acceptance while answering image requests: 0.71-0.89.
- **The upstream "--mmproj not supported with MTP" limitation does not apply to build 11228 / this model.**
  VLM + long-context + MTP are all simultaneously available.
- Caveats: tested at 8k ctx only; `-np 1` required; multi-image/large-image paths untested.
- The control image and logs were generated during the test and have since been deleted.
  The image is trivial to regenerate (white 640x320, black "VISION OK", red ellipse).

## Prefill tuning sweep (-ub / -b): hypothesis refuted

`llama-bench -p 4096,16384 -r 3`, Vulkan, fa 1, f16 KV, build 11228.

| Config | `-b` | `-ub` | pp4096 | pp16384 | tg128 |
|---|---:|---:|---:|---:|---:|
| baseline | 2048 | 512 | **658.87 ± 1.82** | **532.81 ± 17.02** | 28.44 ± 0.03 |
| ub1024 | 2048 | 1024 | 646.41 ± 2.24 | 522.45 ± 20.48 | 28.24 ± 0.11 |
| ub2048 | 2048 | 2048 | 584.91 ± 0.57 | 503.90 ± 1.57 | 28.16 ± 0.13 |
| ub1024_b4096 | 4096 | 1024 | 595.38 ± 1.37 | 517.42 ± 15.63 | 30.86 ± 0.25 |
| ub2048_b8192 | 8192 | 2048 | 630.29 ± 6.01 | 526.12 ± 5.79 | 30.46 ± 0.19 |
| ub512_b4096 | 4096 | 512 | 595.95 ± 46.07 | 542.60 ± 11.66 | 31.56 ± 0.04 |
| ub512_b8192 | 8192 | 512 | 649.99 ± 7.35 | 544.60 ± 15.58 | 31.11 ± 0.03 |

- Hypothesis (larger `-ub` -> better CU utilisation -> faster prefill) is **refuted**.
  Prefill degrades monotonically with `-ub`: -2% at 1024, -11% at 2048.
- Counter-hypothesis confirmed: on Vulkan/coopmat the shader already saturates at `-ub 512`;
  larger ubatches cost occupancy/cache headroom.
- Holding `-ub 512` and raising `-b` leaves prefill within noise (pp16384 +2%, ~1 sigma;
  pp4096 baseline still highest, and the b4096 pp4096 run was noisy at ±46).
- **Keep defaults for prefill: `-b 2048 -ub 512`.**
- **Anomaly worth confirming:** every run with `-b > 2048` reports tg128 ~30.5-31.6 vs
  ~28.2-28.4 at `-b 2048` (4/4 runs, clean separation, tight SDs). ~+10% decode from a
  setting that should not affect decode. Mechanism unknown; confirm before adopting.
- Bonus: this 4B prefills far faster than the 9B MiMo (658 t/s @4k vs 327), and tg128
  ~28-31 t/s is consistent with the earlier repeated decode (~29.6).
- Note: prefill here is f16 KV. Raw log: `prefill-ub-sweep.txt`.

