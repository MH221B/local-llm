"""Pre-Colab gate: two trainable steps through the training-mode cache, on CPU.

Gate claims (spec 8.1):
- per-LAYER page count == 3 (512-token window: pages [32,160),[160,288),[288,416);
  summed over the 8 swapped layers the naive totals would be 24 and 8)
- prefill V block == 1 per layer ([32:384))
- fresh cache per step (construct per call)
- adapter grads finite and nonzero; KL finite
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.cache import build_kitty_cache, load_model  # noqa: E402
from src.data import HOLDOUT, batched, build_windows, split_windows  # noqa: E402
from scripts.train_qat_colab import CFG, kl_loss  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"


def one_step(model, batch) -> float:
    device = next(model.parameters()).device
    batch = batch.to(device)

    # teacher: same weights, adapters disabled, no cache, no grads
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        t_logits = model(input_ids=batch, use_cache=False).logits.float()

    # student: fresh quantizing cache, adapters active, graph on
    model.train()
    cache, swapped = build_kitty_cache(
        DynamicCache(config=model.config), model.config, **CFG
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        s_logits = model(input_ids=batch, use_cache=True, past_key_values=cache).logits.float()

    per_layer_pages = {cache.layers[i].quantized_pages for i in swapped}
    per_layer_vblocks = {cache.layers[i].quantized_value_blocks for i in swapped}
    assert per_layer_pages == {3}, per_layer_pages
    assert per_layer_vblocks == {1}, per_layer_vblocks
    assert cache.layers[swapped[0]].keys.shape[-2] == batch.shape[-1], "cache length mismatch"
    del cache

    # imported from the trainer module: the smoke cannot drift from the L4 run
    kl = kl_loss(t_logits, s_logits)
    kl.backward()
    assert torch.isfinite(kl), kl

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no adapter parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite adapter gradient"
    assert any(torch.count_nonzero(g) > 0 for g in grads), "all adapter gradients zero"
    model.zero_grad(set_to_none=True)
    return kl.item()


def main() -> None:
    model, tokenizer = load_model(MODEL_ID, dtype=torch.bfloat16)

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=8, target_modules=["k_proj", "v_proj"], lora_alpha=16,
                      lora_dropout=0.05)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Decision 4: adapters must train in fp32 — bf16 trainables underflow at lr 2e-4.
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    windows = build_windows(tokenizer, n_conversations=32)
    train, held = split_windows(windows, holdout=4)   # explicit: 32 convs yield ~61
    # windows, fewer than the shared HOLDOUT of 64 (the trainer/eval split); the smoke
    # is not testing the holdout, only the training-step mechanics.
    print(f"train windows {len(train)}, held-out {len(held)}", flush=True)

    losses = [one_step(model, b) for b in list(batched(train, batch_size=1))[:2]]
    print("kl:", losses)
    print("smoke_trainstep.py ok")


if __name__ == "__main__":
    main()
