"""Emit the section 8 vision holdout: held-out Cauldron VQA, never trained on."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from . import canonical
from .adapters.base import Ctx
from .adapters.cauldron import VQA_CONFIGS, build_vqa
from .imgstore import ImageStore, materialise_images

DATASET = "HuggingFaceM4/the_cauldron"
DEFAULT_PER_CONFIG = 125          # 4 configs x 125 = 500, per spec section 8


def harvest(root: Path, per_config: int, dry_run: bool = False) -> list[canonical.Example]:
    store = ImageStore(root / "images")
    out: list[canonical.Example] = []
    for config in VQA_CONFIGS:
        if dry_run:
            print(f"[dry] {config}: {per_config} rows")
            continue
        ctx = Ctx(DATASET, config, "reasoning", "apache-2.0")
        stream = load_dataset(DATASET, config, split="train", streaming=True)
        kept = 0
        for index, row in enumerate(stream):
            ex = build_vqa(row, index, ctx)
            if ex is None:
                continue
            materialise_images(ex, store)
            if canonical.validate(ex):
                continue
            out.append(ex)
            kept += 1
            if kept >= per_config:
                break
        print(f"{config} -> {kept}")
    return out


def run(*, root: Path, per_config: int, dry_run: bool) -> int:
    examples = harvest(root, per_config, dry_run)
    if dry_run:
        return 0
    path = root / "eval" / "vision.jsonl"
    written = canonical.write_jsonl(path, examples)
    print(f"vision holdout: {written} rows -> {path}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vision holdout exporter")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--per-config", type=int, default=DEFAULT_PER_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(root=args.root, per_config=args.per_config, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
