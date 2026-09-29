"""Fetch the uncensored prompt seeds and quarantine the refusal eval slice.

Prompts only; M2 supplies every response. The in-the-wild harvest also carves the
500-row held-out slice from spec section 8, so it can never train.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from . import textutil
from .canonical import Prompt, iter_jsonl

# (dataset, config, split, field, cap) - spec section 7.4.
# Caps respect what each config actually ships: in-the-wild 2023_12_25 has 1,405 rows,
# 2023_05_07 has 666, JBB-Behaviors ships 100 harmful + 100 benign.
SEED_SOURCES = [
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train", "prompt", 1400),
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_05_07", "train", "prompt", 600),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", "Goal", 100),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "benign", "Goal", 100),
    ("mlabonne/harmful_behaviors", None, "train", "text", 200),
    ("mlabonne/harmless_alpaca", None, "train", "text", 600),
]

IN_THE_WILD = "TrustAIRLab/in-the-wild-jailbreak-prompts"
REFUSAL_HOLDOUT = 500          # spec section 8: 1,500 train / 500 quarantined


def harvest(*, dry_run: bool = False) -> list[dict]:
    seeds: list[dict] = []
    for name, config, split, field, cap in SEED_SOURCES:
        if dry_run:
            print(f"[dry] {name}:{config}/{split} cap={cap} field={field}")
            continue
        stream = load_dataset(name, config, split=split, streaming=True)
        taken = 0
        for row in stream:
            text = (row.get(field) or "").strip()
            if not text:
                continue
            seeds.append({"id": f"{name.split('/')[-1]}-{config or 'default'}-{split}-{taken}",
                          "source": name, "config": config, "text": text})
            taken += 1
            if taken >= cap:
                break
        print(f"{name}:{config}/{split} -> {taken}")
    return seeds


def split_refusal_holdout(seeds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hash-select 500 in-the-wild prompts for the section 8 refusal eval."""
    inwild = [s for s in seeds if s["source"] == IN_THE_WILD]
    if len(inwild) <= REFUSAL_HOLDOUT:
        return seeds, []
    order = sorted(inwild, key=lambda s: textutil.prompt_hash(s["text"]))
    heldout = {id(s) for s in order[:REFUSAL_HOLDOUT]}
    return ([s for s in seeds if id(s) not in heldout],
            [s for s in seeds if id(s) in heldout])


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def load_prompts(root: Path) -> list[Prompt]:
    """The trainable seeds as canonical prompts; the pipeline merges them as uncensored."""
    path = Path(root) / "seeds" / "uncensored.jsonl"
    if not path.exists():
        print(f"WARNING: {path} missing; run tools.dataset.seeds before pipeline.py")
        return []
    return [
        Prompt(id=row["id"], domain="uncensored", origin="prebuilt",
               source={"name": row["source"], "config": row["config"]},
               messages=[{"role": "user",
                          "content": [{"type": "text", "text": row["text"]}]}])
        for row in iter_jsonl(path)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="uncensored prompt seeds")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    print("seed sources:", len(SEED_SOURCES),
          "total cap:", sum(s[4] for s in SEED_SOURCES))
    seeds = harvest(dry_run=args.dry_run)
    if args.dry_run:
        return 0

    trainable, holdout = split_refusal_holdout(seeds)
    n_train = write_jsonl(args.root / "seeds" / "uncensored.jsonl", trainable)
    n_eval = write_jsonl(args.root / "eval" / "refusal.jsonl", holdout)
    print(f"seeds trainable: {n_train}, refusal holdout: {n_eval}")
    return 0 if n_train and n_eval else 1


if __name__ == "__main__":
    sys.exit(main())
