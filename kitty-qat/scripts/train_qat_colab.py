"""Session A trainer: LoRA r=8 on k/v of the 8 full-attention layers.

No gradient checkpointing anywhere (spec 4.3: the replay re-enters update() on
an initialized cache). Teacher = same weights with adapters disabled, no second
model copy. Fresh quantizing cache per step.

Memory: a 512-token window through a 4B hybrid model at micro-batch 2 exceeds the
L4's 22 GiB (the reference chunk_gated_delta_rule fallback is the peak). We keep
the effective batch at 2 via --accum with --batch 1 micro-steps, which is exactly
how the pre-Colab CPU gate already ran.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Set before torch initializes the CUDA allocator. Reduces fragmentation, which
# is what turns a "just barely fits" step into an OOM at the last allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Allow running as a plain script (`python scripts/train_qat_colab.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.cache import build_kitty_cache, load_model  # noqa: E402
from src.data import HOLDOUT, batched, build_windows, split_windows  # noqa: E402
from src.train_quant import ste_fake_quant  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"
CFG = dict(post_quant=False, detach_states=False, quant_fn=ste_fake_quant)
DATA_CONV = 2600


def kl_loss(t_logits, s_logits):
    """KL(teacher ‖ student), temp 1, last 64 positions (spec 4.3).

    Slices to the last 64 positions BEFORE the fp32 upcast, so the full-vocab
    logits are never materialized in fp32 (~0.6 GB saved per micro-step at
    vocab ~150k).
    """
    t = torch.log_softmax(t_logits[:, -64:, :].float(), dim=-1)
    s = torch.log_softmax(s_logits[:, -64:, :].float(), dim=-1)
    return torch.nn.functional.kl_div(s, t, log_target=True, reduction="batchmean")


def train_step(model, batch, accum: int) -> float:
    """One micro-step: forward both paths, backward the scaled KL. No optim step."""
    device = next(model.parameters()).device

    model.eval()
    with model.disable_adapter(), torch.no_grad():
        t_logits = model(input_ids=batch.to(device), use_cache=False).logits

    model.train()
    cache, _ = build_kitty_cache(
        DynamicCache(config=model.config), model.config, **CFG
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_logits = model(input_ids=batch.to(device), use_cache=True,
                         past_key_values=cache).logits
    del cache

    loss = kl_loss(t_logits, s_logits)
    (loss / accum).backward()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("qat-lora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1, help="micro-batch (VRAM-bound)")
    parser.add_argument("--accum", type=int, default=2,
                        help="gradient accumulation; effective batch = batch*accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    model, tokenizer = load_model(MODEL_ID, dtype=torch.bfloat16)
    model.to("cuda")
    model.config.use_cache = True

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=8, target_modules=["k_proj", "v_proj"], lora_alpha=16,
                      lora_dropout=0.05)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Decision 4: adapters must train in fp32 — bf16 trainables underflow at
    # lr 2e-4. peft initializes adapters in the base dtype, so cast after wrap.
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    windows = build_windows(tokenizer, n_conversations=DATA_CONV)
    train, held = split_windows(windows, holdout=HOLDOUT)
    print(f"train windows {len(train)}, held-out {len(held)}", flush=True)

    effective = args.batch * args.accum
    n_steps = math.ceil(len(train) / effective) * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=n_steps, pct_start=0.03)

    trainables = [p for p in model.parameters() if p.requires_grad]
    model.train()
    micro = step = 0
    for epoch in range(args.epochs):
        for batch in batched(train, batch_size=args.batch):
            loss = train_step(model, batch, args.accum)
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainables, 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == 1:
                    print(f"step {step}/{n_steps} kl {loss:.4f} "
                          f"lr {sched.get_last_lr()[0]:.3e}", flush=True)
    # flush a trailing partial accumulation (len(train) may not divide batch*accum)
    if micro % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    print(f"done at step {step}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapters to {args.out}")


if __name__ == "__main__":
    main()
