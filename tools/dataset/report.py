"""Manifest: counts, caps, filter drops, token share, image totals (spec section 9.7)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .canonical import Prompt


def summarise(records: list[Prompt], label: str) -> dict:
    by_domain = Counter(ex.domain for ex in records)
    tokens_by_domain: Counter = Counter()
    images = 0
    image_examples = 0
    for ex in records:
        tokens_by_domain[ex.domain] += ex.tokens
        if ex.images:
            image_examples += 1
            images += len(ex.images)
    total_tokens = sum(tokens_by_domain.values())
    return {
        "split": label,
        "examples": len(records),
        "by_domain": dict(sorted(by_domain.items())),
        "example_share": {d: round(n / len(records), 4) for d, n in sorted(by_domain.items())}
        if records else {},
        "tokens": total_tokens,
        "token_share": {d: round(n / total_tokens, 4) for d, n in sorted(tokens_by_domain.items())}
        if total_tokens else {},
        "image_bearing_examples": image_examples,
        "image_bearing_share": round(image_examples / len(records), 4) if records else 0.0,
        "images": images,
    }


def build_manifest(*, run: dict, train: list[Prompt], val: list[Prompt],
                   sources: list[dict], drops: Counter) -> dict:
    return {
        "run": run,
        "sources": sources,
        "drops": dict(drops.most_common()),
        "train": summarise(train, "train"),
        "val": summarise(val, "val"),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_manifest(path: Path, *, train: list, val: list, m2: dict) -> dict:
    """Read the M1 manifest, add the M2 pass-rate block, and refresh the split summaries."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest["m2"] = m2
    manifest["train_final"] = summarise(train, "train")
    manifest["val_final"] = summarise(val, "val")
    return manifest


if __name__ == "__main__":
    from .canonical import Prompt

    rows = [Prompt(id=f"r{i}", domain="reasoning", origin="prebuilt", source={"name": "s"},
                   messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                   tokens=100)
            for i in range(3)]
    rows += [Prompt(id="c", domain="coding", origin="prebuilt", source={"name": "s"},
                    messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                    tokens=300)]
    s = summarise(rows, "train")
    print("examples:", s["examples"], "token_share:", s["token_share"])
    print("image_bearing_share:", s["image_bearing_share"])
