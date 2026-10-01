"""Windows for QAT training and PPL eval.

Chat windows stream smoltalk conversations through the chat template (thinking
off) into one token thread, chunked to seq_len. Raw windows (WikiText-2) chunk
plain text with no chat template. The cache asserts below 161 cached tokens
and quantizes nothing at or below 160, so windows shorter than MIN_TOKENS
never exist here by construction (all windows are exactly seq_len long).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running as a plain script (`python src/data.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset

SEED = 0
SEQ_LEN = 512
HOLDOUT = 64   # single source of truth: trainer and eval scripts import this


def _chunk(stream: list[int], seq_len: int) -> list[list[int]]:
    n = len(stream) // seq_len
    windows = [stream[i * seq_len:(i + 1) * seq_len] for i in range(n)]
    assert all(len(w) == seq_len for w in windows)
    return windows


def build_windows(tokenizer, n_conversations: int = 2600, seq_len: int = SEQ_LEN):
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split=f"train[:{n_conversations}]")
    stream: list[int] = []
    for ex in ds:
        msgs = [m for m in ex["messages"] if m["role"] in ("user", "assistant")]
        if not msgs:
            continue
        stream.extend(tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=True,
            enable_thinking=False)["input_ids"])
    return _chunk(stream, seq_len)


def build_windows_raw(tokenizer, split: str = "test", n_chars: int = 2_500_000,
                      seq_len: int = SEQ_LEN):
    """Raw-text windows (no chat template) for the out-of-domain PPL column."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "".join(ds["text"])[:n_chars]   # NOT "".join(ds["text"][:n_chars]) — that
    # slices rows, not characters; here it would just use the whole split
    return _chunk(tokenizer(text, add_special_tokens=False).input_ids, seq_len)


def split_windows(windows, holdout: int = HOLDOUT):
    """Final `holdout` windows are held out; train never sees them."""
    assert len(windows) > holdout
    return windows[:-holdout], windows[-holdout:]


def batched(windows, batch_size: int):
    """Deterministic shuffle (SEED), then yield (B, seq_len) tensors."""
    order = list(range(len(windows)))
    random.Random(SEED).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield torch.tensor([windows[j] for j in order[i:i + batch_size]], dtype=torch.long)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("RBergBauer/Qwen3.5-4B-MTP-Heretic")
    windows = build_windows(tok, n_conversations=32)
    train, held = split_windows(windows, holdout=4)
    print(f"windows={len(windows)} train={len(train)} held-out={len(held)}")
    b = next(batched(train, batch_size=2))
    assert b.shape == (2, SEQ_LEN), b.shape
    assert torch.isin(b, torch.tensor(list(tok.get_vocab().values()))).all()
    print("data.py ok")
