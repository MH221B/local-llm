"""Eval-set disjointness guard (spec section 11.1). Non-negotiable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from .canonical import iter_jsonl
from . import textutil

# (dataset, config, split, field) for the remote eval holdouts in spec section 8.
# The refusal slice is local: seeds.py writes eval/refusal.jsonl, joined via --extra-eval.
EVAL_SETS = [
    ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    ("bigcode/humanevalpack", "python", "test", "prompt"),
    ("openai/gsm8k", "main", "test", "question"),
    # The section 8 vision probe. Excluded from training by construction, so this is a
    # belt-and-braces check that no future cap pulls them back into the pool.
    ("HuggingFaceM4/the_cauldron", "chartqa", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "ai2d", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "tqa", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "scienceqa", "train", "texts"),
]


def _field_text(row: dict, field: str) -> str:
    """Cauldron's `texts` is a list of {user, assistant, source} structs; others are strings."""
    value = row.get(field)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
        if isinstance(value, dict):
            value = value.get("user") or value.get("assistant") or ""
    return str(value or "").strip()


def eval_prompts(limit_per_set: int = 2000, skip: tuple[str, ...] = ()) -> list[str]:
    out: list[str] = []
    for name, config, split, field in EVAL_SETS:
        if name in skip or f"{name}:{config}" in skip:
            print(f"skip {name}:{config}")
            continue
        stream = load_dataset(name, config, split=split, streaming=True)
        taken = 0
        for i, row in enumerate(stream):
            if i >= limit_per_set:
                break
            text = _field_text(row, field)
            if text:
                out.append(text)
                taken += 1
        print(f"{name}:{config} -> {taken} eval prompts")
    return out


def _local_prompts(path: Path) -> list[str]:
    """Prompt text from a local JSONL: seeds use `text`, canonical rows use messages."""
    out: list[str] = []
    for row in iter_jsonl(path):
        if isinstance(row.get("text"), str):
            out.append(row["text"])
            continue
        text = "".join(p.get("text", "") for m in row.get("messages", [])
                       if m.get("role") == "user"
                       for p in (m["content"] if isinstance(m["content"], list)
                                 else [{"type": "text", "text": m["content"]}]))
        if text:
            out.append(text)
    return out


def check(*, dataset: Path, threshold: float = 0.85,
          skip: tuple[str, ...] = (), extra_eval: tuple[Path, ...] = ()) -> int:
    prompts = eval_prompts(skip=skip)  # fetched once; both derived lists reuse it
    for path in extra_eval:
        prompts.extend(_local_prompts(path))
    eval_hashes = {textutil.prompt_hash(p) for p in prompts}
    eval_sigs = [textutil.minhash(p) for p in prompts]
    collisions = 0
    for row in iter_jsonl(dataset):
        prompt = "".join(
            p.get("text", "") for m in row["messages"] if m["role"] == "user"
            for p in (m["content"] if isinstance(m["content"], list)
                      else [{"type": "text", "text": m["content"]}]))
        if not prompt:
            continue
        if textutil.prompt_hash(prompt) in eval_hashes:
            collisions += 1
            print(f"EXACT COLLISION {row['id']}")
            continue
        sig = textutil.minhash(prompt)
        for other in eval_sigs:
            if textutil.jaccard_est(sig, other) >= threshold:
                collisions += 1
                print(f"NEAR COLLISION {row['id']}")
                break
    print(f"collisions: {collisions}")
    return 1 if collisions else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="eval contamination guard")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--skip", action="append", default=[],
                    help="dataset or dataset:config to leave out of the eval side")
    ap.add_argument("--extra-eval", action="append", type=Path, default=[],
                    help="local JSONL whose prompts join the eval side")
    args = ap.parse_args(argv)
    return check(dataset=args.dataset, threshold=args.threshold,
                 skip=tuple(args.skip), extra_eval=tuple(args.extra_eval))


if __name__ == "__main__":
    sys.exit(main())
