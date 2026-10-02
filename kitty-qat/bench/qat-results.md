# Spec 2 — KV-QAT on Kitty INT2 (Qwen3.5-4B-MTP-Heretic)

**Status: partial.** PPL: run0 for held-out + WikiText-2 (eval interrupted before the
final save; run1 would duplicate run0). GSM8K: run0 complete. NIAH: **run0 returned
0/10 for every config, including fp16 — a harness bug (no chat template + a
needle-truncation bug), now fixed; the result is discarded.** Task 9 (GGUF + MTP
smoke) not done.

Date: 2026-10-02. Adapter: `qat-lora` (LoRA r=8, `k_proj`+`v_proj` on the 8
full-attention layers; 458,752 trainable params / 0.0109%). Merged artifact
`qat-merged` verified: 441 index keys, 15 `mtp.*` byte-identical to the original
checkpoint.

## Protocol

Each window is 512 tokens: 192-token prefill, then ~320 teacher-forced decode steps
against the quantized cache (`post_quant=True`, so the current step attends to the
pre-quantization snapshot and each later step lags one page). The gate counts
**decode-position** NLL; prefill NLL is the sanity check. Configs:

- `fp16` — base weights, 16-bit cache (quantizer no-op)
- `kitty-int2` — base weights, Kitty 2-bit K+V, `promote_ratio=0.1`
- `kitty-int2+qat` — base + merged `qat-lora`, same 2-bit cache

Held-out = last 64 smoltalk windows (held out from training). WikiText-2 = last 64
raw-text windows (out-of-domain, never trained on).

## Results

### Held-out (smoltalk, in-domain)

| config | prefill_ppl | decode_ppl | vs `kitty-int2` |
|---|---|---|---|
| fp16 | 4.4693 | 3.2159 | −0.0680 |
| kitty-int2 | 4.4693 | 3.2839 | — |
| **kitty-int2+qat** | 4.4530 | **3.2114** | **−0.0725** |

### WikiText-2 (out-of-domain)

| config | prefill_ppl | decode_ppl | vs `kitty-int2` |
|---|---|---|---|
| fp16 | 16.5982 | 9.8303 | −0.2671 |
| kitty-int2 | 16.5982 | 10.0974 | — |
| **kitty-int2+qat** | 16.7092 | **9.9965** | **−0.1009** |

## Gate

`decode_ppl(INT2+QAT) < decode_ppl(INT2)` — **passes in both columns**:

- held: 3.2114 < 3.2839 (Δ −0.0725; also edges fp16 by −0.0045)
- wt2: 9.9965 < 10.0974 (Δ −0.1009; recovers ~38% of the 0.2671 INT2 penalty)

**Harness sanity:** `fp16` and `kitty-int2` have *identical* prefill PPL in both
columns (4.4693 = 4.4693; 16.5982 = 16.5982). Quantization is not leaking into
prefill — the port behaves as designed.

## GSM8K (sanity, run0)

50 test questions, 8-shot CoT, greedy, max-new 512. Single-pass direction check, not a
gate.

| config | correct | total_pages |
|---|---|---|
| fp16 | 44/50 | 4544 |
| kitty-int2 | 46/50 | 4552 |
| kitty-int2+qat | 44/50 | 4632 |

Read: INT2 does **not** damage GSM8K here (46 vs 44), and QAT does not change it
(44 = 44). The INT2 ≥ fp16 gap is within noise. At n=50 single-pass this is
underpowered — it rules out a catastrophic regression, not a small delta.

Source: `bench/session-b-gsm8k.log`; summary in `bench/gsm8k-raw.run0.json`
(per-question `details` were not captured from the console).

## Interpretation (and the confound)

The QAT prefill deltas expose that the adapter also changes full-precision behaviour,
because prefill never consumes quantization:

- **In-domain:** QAT prefill is *better* than base (4.4530 < 4.4693). So part of the
  in-domain decode win is plain fine-tuning / domain adaptation on smoltalk, not QAT.
- **OOD:** QAT prefill is *worse* (16.7092 > 16.5982) — the adapter degraded general
  text modelling — **yet its INT2 decode still improved** (10.0974 → 9.9965). A
  general fine-tune cannot produce that; it points to a quantization-specific effect.

So the honest headline: **QAT reduces the INT2 penalty, and the OOD column shows it is
not merely domain adaptation.** The in-domain delta alone cannot separate the two.

## Limitations / not yet done

- **GSM8K ran (sanity); NIAH is broken.** GSM8K shows no INT2/QAT regression (see
  above) but is underpowered. NIAH returned 0/10 for *all* configs including fp16 —
  a harness bug, now fixed (raw prompt with no chat template; needle could be
  truncated off the end). Its result is discarded pending a re-run. So the
  out-of-objective check is only partial.
- **No `fp16+qat` row**, so the QAT-vs-fine-tuning split is inferred from prefill
  deltas, not measured directly.
- **No matched fp16-cache fine-tune** (the clean control for the confound).
- **Single run (run0 only).** The eval is deterministic — fixed windows,
  teacher-forced decoding, same weights — so there is no run-to-run spread to report.
  Real uncertainty would need a bootstrap over the 64 windows; `--runs 1` suffices.
- **Task 9 not done:** GGUF convert + Assert 2 (33 blocks / 15 nextn) + llama-server
  MTP smoke. MTP is not exercised by any eval above (the HF graph has no MTP
  submodule); its integrity is only checked at the artifact/smoke stage.
- Speed/memory not measured — the HF port simulates quantization (fake-quant over
  fp16 storage), so there is no real memory or latency benefit to report here.

## Reproduction

```python
# Session A (train + merge + carry mtp)
!python scripts/train_qat_colab.py --out /content/qat-lora --batch 1 --accum 2
!python scripts/export_artifact.py --adapter /content/qat-lora --out /content/qat-merged
# Session B (PPL; --runs 1 is sufficient)
!python scripts/eval_ppl.py --adapter /content/drive/MyDrive/qat-lora --wt2 --runs 1 --out /content/ppl-raw.json
```

Source of the numbers above: `bench/session-b-run0.log` (verbatim PPL run0 console
output) and `bench/session-b-gsm8k.log` (verbatim GSM8K console output).
`bench/ppl-raw.run0.json` and `bench/gsm8k-raw.run0.json` reconstruct those into the
shape the scripts write — run0 only, `spread` null (the script only fills `spread`
when `--runs >= 2`). The real `ppl-raw.json` was never written (the run was stopped
before the final save); `gsm8k-raw.json` was written but not transferred, so its
per-question `details` are lost.
