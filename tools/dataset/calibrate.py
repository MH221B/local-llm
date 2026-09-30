"""Carve calibration chunks for Phase 3's llama-imatrix (spec section 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical import iter_jsonl


def assistant_text(row: dict) -> str:
    for m in reversed(row["messages"]):
        if m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


# Spec section 3: chunks of 2-4k tokens. At the planning ratio (~4 chars/token) that is
# roughly 8k-16k characters; exact token counts are not material for imatrix calibration.
MIN_CHARS = 8000
MAX_CHARS = 16000


def run(*, dataset: Path, out: Path, chunks: int,
        min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> int:
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in iter_jsonl(dataset):
            text = assistant_text(row)
            if len(text) < min_chars:
                continue
            fh.write(text[:max_chars].replace("\n", " ") + "\n")
            written += 1
            if written >= chunks:
                break
    print(f"calibration chunks written: {written} -> {out}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibration set exporter")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("calibration.txt"))
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args(argv)
    return run(dataset=args.dataset, out=args.out, chunks=args.chunks,
               min_chars=args.min_chars, max_chars=args.max_chars)


if __name__ == "__main__":
    sys.exit(main())
