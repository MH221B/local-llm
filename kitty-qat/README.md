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
| 1. Cache-level golden trace | not started |
| 2. Structural (layer map, head dims) | not started |
| 3. Quantization exercised | not started |
| 4. Baseline FP16 / INT2 / Kitty-Pro | not started |
