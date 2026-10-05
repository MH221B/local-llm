"""One-off repair for built corpora whose tool schemas use non-standard JSON types.

ToolACE's embedded schemas spell types Python-ish (``dict``/``int``/``float``), which
llama.cpp's schema->grammar converter rejects: ``HTTP 500 {"message":"JSON schema error at
#: unrecognized type dict"}``. ``trajparse._toolace_schema`` now normalizes at parse time;
this rewrites corpora that were built before that fix, in place. Idempotent -- re-running
reports zero hits.

    python -m tools.dataset.repair_tool_schemas datasets/qwen35-4b-sft-full/prompts/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .canonical import _SCHEMA_TYPE_ALIASES, normalize_tools


def _nonstd_types(value) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        t = value.get("type")
        if isinstance(t, str) and t in _SCHEMA_TYPE_ALIASES:
            out.append(t)
        for v in value.values():
            out += _nonstd_types(v)
    elif isinstance(value, list):
        for v in value:
            out += _nonstd_types(v)
    return out


def repair_file(path: Path, dry_run: bool = False) -> tuple[int, int, dict[str, int]]:
    """Return (rows, rows_fixed, type_counts). Rewrites atomically unless dry_run."""
    path = Path(path)
    rows = 0
    fixed = 0
    counts: dict[str, int] = {}
    out_lines: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            row = json.loads(stripped)
            bad = _nonstd_types(row.get("tools")) if row.get("tools") else []
            if bad:
                fixed += 1
                for t in bad:
                    counts[t] = counts.get(t, 0) + 1
                row["tools"] = normalize_tools(row["tools"])
            out_lines.append(json.dumps(row, ensure_ascii=False))
    if not dry_run and fixed:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + "\n")
        os.replace(tmp, path)
    return rows, fixed, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="normalize non-standard tool schema types in a corpus")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for p in args.paths:
        rows, fixed, counts = repair_file(p, dry_run=args.dry_run)
        tag = "would fix" if args.dry_run else "fixed"
        print(f"{p}: rows={rows} {tag}={fixed} types={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
