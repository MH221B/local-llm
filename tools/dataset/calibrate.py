"""Carve calibration chunks for Phase 3's llama-imatrix (spec section 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import canonical, render
from .canonical import iter_jsonl

DEFAULT_TEMPLATE = Path(__file__).parent / "student.jinja"


def message_text(m: dict) -> str:
    content = m.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def assistant_text(row: dict) -> str:
    """Every assistant turn in the row, joined and whitespace-collapsed.

    Fallback for when no chat template is available; a chat or multi-turn turn is short
    by nature, so reading only the last one starves the chunk packer in `run`.
    """
    return " ".join(" ".join(message_text(m).split())
                    for m in row.get("messages", []) if m.get("role") == "assistant")


def row_text(row: dict, template: str | None) -> str:
    """The row as the student sees it, rendered through its chat template.

    The teacher's own parent model builds its imatrix corpus this way and parses it with
    `--parse-special`, so chat-format special tokens contribute to the importance matrix
    (bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF). Bare assistant text never exercises
    `<|im_start|>` / `<|im_end|>` or the think markers, which is what an imatrix exists to
    weight.
    """
    messages = render.strip_image_data(row.get("messages", []))
    if template is None:
        return assistant_text(row)
    return " ".join(render.render(template, canonical.for_template(messages)).split())


# Spec section 3: chunks of 2-4k tokens. At the planning ratio (~4 chars/token) that is
# roughly 8k-16k characters; exact token counts are not material for imatrix calibration.
MIN_CHARS = 8000
MAX_CHARS = 16000


def run(*, dataset: Path, out: Path, chunks: int, template_path: Path | None = DEFAULT_TEMPLATE,
        min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> int:
    """Write up to `chunks` chunks of 2-4k tokens of templated training text.

    Text accumulates across rows until it reaches `min_chars`, because no single message
    is expected to be a whole chunk. Requiring one long message emitted nothing in
    practice: measured on the 9-row M2 run, the longest assistant message was 6,181
    chars against the 8,000 floor, so `calibration.txt` came out empty and the command
    exited 1 - which the Task 15 acceptance reads as a failure.
    """
    template = None
    if template_path is not None:
        template = Path(template_path).read_text(encoding="utf-8")
    written = 0
    buffer: list[str] = []
    size = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in iter_jsonl(dataset):
            text = row_text(row, template)
            if not text:
                continue
            buffer.append(text)
            size += len(text)
            if size < min_chars:
                continue
            fh.write(" ".join(buffer)[:max_chars] + "\n")
            written += 1
            buffer, size = [], 0
            if written >= chunks:
                break
    print(f"calibration chunks written: {written} -> {out}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibration set exporter")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("calibration.txt"))
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="chat template to render rows through; pass /dev/null-style "
                         "empty to fall back to bare assistant text")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args(argv)
    if args.template is not None and not args.template.exists():
        print(f"template not found: {args.template}")
        return 2
    return run(dataset=args.dataset, out=args.out, chunks=args.chunks,
               template_path=args.template, min_chars=args.min_chars,
               max_chars=args.max_chars)


def smoke() -> int:
    """Offline: eight ~6k-char rows, each below the floor, so only accumulation can pack.

    Also pins the templated path - the Task 15 regression was a bare-text exporter that
    required one assistant message over `MIN_CHARS` and emitted nothing, and omitted the
    chat special tokens the imatrix is meant to weight.
    """
    import json
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "word " * 1200}]}
            for _ in range(8)]
    src = tmp / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp / "calibration.txt"
    rc = run(dataset=src, out=out, chunks=3)
    text = out.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    marked = "<|im_start|>" in text and "<|im_end|>" in text
    print("smoke: rc", rc, "| chunks", len(lines), "| sizes", [len(l) for l in lines],
          "| chat markers:", marked)
    return 0 if (rc == 0 and len(lines) == 3 and marked) else 1


if __name__ == "__main__":
    sys.exit(smoke() if len(sys.argv) == 1 else main())
