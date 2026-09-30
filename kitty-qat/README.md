# kitty-qat

Port of Kitty's 2-bit KV cache accuracy simulation onto `Qwen3.5-4B-MTP-Heretic`.
Spec 1 of two; Spec 2 adds KV-QAT on top of this harness.

Spec: `../docs/superpowers/specs/2026-09-30-kitty-qat-port-design.md`

## Run

    python scripts/equiv_check.py --reference third_party/kitty_sim
    python scripts/eval_baseline.py --model RBergBauer/Qwen3.5-4B-MTP-Heretic

## Status

| Check | Status |
|---|---|
| 1. Cache-level golden trace | PASS (prefill=200, decode=300, every-step tensor equality with the reference cache) |
| 2. Structural (layer map, head dims) | PASS ([3,7,11,15,19,23,27,31], head_dim 256, 4 KV heads) |
| 3. Quantization exercised | PASS (quantized region differs from fp16; 160-token boundary quantizes nothing) |
| 4. Baseline FP16 / INT2 / Kitty-Pro | 3/3 = 3/3 = 3/3 (local CPU, thinking off, cached ~253 tokens; byte-identical outputs — see `bench/port-results.md`) |
