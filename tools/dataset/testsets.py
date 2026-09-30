"""Unit-test verification seeds: prompts whose tests filter teacher attempts.

MBPP ships assert-style tests; APPS ships stdin/stdout pairs. Each seed becomes a
canonical Prompt with a `verify` spec attached, so M2 can execute the teacher's
completion and drop failures before they can reach the student.

Two sourcing notes, both measured against the live datasets:

- MBPP's `sanitized` config holds only 120 rows, far under the plan's 400 cap, so the
  `test` split is read too (257 more). The `full` config is larger but renames its fields
  (`text`, `test_setup_code`), so it is left alone rather than given a second code path.
- `codeparrot/apps` cannot be loaded by `datasets` 5.x: the repo ships a loading script
  plus raw JSONL, and scripts are no longer supported. Its `train.jsonl` is streamed
  directly over HTTP instead, which returns the same shape the mapper expects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset

from .canonical import Prompt, write_jsonl

MBPP = {"dataset": "google-research-datasets/mbpp", "config": "sanitized",
        "splits": ("train", "test"), "cap": 400}
APPS = {"dataset": "codeparrot/apps", "config": None, "cap": 1000}
APPS_URL = ("https://huggingface.co/datasets/codeparrot/apps/"
            "resolve/main/train.jsonl")


def mbpp_prompt(row: dict, index: int) -> Prompt | None:
    prompt, tests = row.get("prompt"), row.get("test_list")
    if not prompt or not isinstance(tests, list) or not tests:
        return None
    return Prompt(
        id=f"mbpp-{row.get('task_id', index)}",
        domain="coding",
        origin="prebuilt",
        source={"name": MBPP["dataset"], "config": MBPP["config"], "row": index},
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        verify={"type": "python_tests",
                "setup": "\n".join(row.get("test_imports") or []),
                "tests": tests},
        meta={"difficulty": row.get("difficulty")},
    )


def apps_prompt(row: dict, index: int) -> Prompt | None:
    question, raw = row.get("question"), row.get("input_output")
    if not question or not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    except ValueError:
        # Some rows carry integers with thousands of digits, which Python refuses to
        # parse under its int-conversion guard (a DoS limit, not a data error). The
        # row cannot be verified against stdin/stdout, so it is skipped like any other
        # unusable one rather than aborting the harvest.
        return None
    inputs, outputs = data.get("inputs") or [], data.get("outputs") or []
    pairs = [[i, o] for i, o in zip(inputs, outputs)
             if isinstance(i, str) and isinstance(o, str)]
    if not pairs:
        return None
    return Prompt(
        # The raw JSONL keys this `id`; the loading script called it `problem_id`.
        id=f"apps-{row.get('id', row.get('problem_id', index))}",
        domain="coding",
        origin="prebuilt",
        source={"name": APPS["dataset"], "row": index},
        messages=[{"role": "user", "content": [{"type": "text", "text": question}]}],
        verify={"type": "python_io", "pairs": pairs[:20]},
        meta={"difficulty": row.get("difficulty")},
    )


def _mbpp_rows():
    """Stream MBPP across its splits; `sanitized` is small, so both are needed."""
    for split in MBPP["splits"]:
        stream = load_dataset(MBPP["dataset"], MBPP["config"], split=split,
                              streaming=True)
        yield from stream


def _apps_rows():
    """Stream APPS' raw JSONL directly: `load_dataset` rejects its loading script."""
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(APPS_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            line = line.strip()
            if line:
                yield json.loads(line)


def harvest(rows, mapper, cap: int, label: str) -> list[Prompt]:
    out: list[Prompt] = []
    seen = 0
    for row in rows:
        seen += 1
        rec = mapper(row, seen)
        if rec is None:
            continue
        out.append(rec)
        if len(out) >= cap:
            break
    print(f"{label} -> {len(out)} verification seeds (from {seen} rows)")
    return out


def smoke() -> int:
    mb = mbpp_prompt({"task_id": 2, "prompt": "Write add(a, b).", "test_imports": [],
                      "test_list": ["assert add(1, 2) == 3"]}, 0)
    apps = apps_prompt({"problem_id": 1, "question": "Read two ints and print the sum.",
                        "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]})}, 0)
    print("mbpp verify:", mb.verify["type"], len(mb.verify["tests"]))
    print("apps verify:", apps.verify["type"], len(apps.verify["pairs"]))
    print("no-test row dropped:", mbpp_prompt({"prompt": "x", "test_list": []}, 0))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="unit-test verification seeds")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--smoke", action="store_true", help="fixture checks, no network")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke()
    seeds = (harvest(_mbpp_rows(), mbpp_prompt, MBPP["cap"], MBPP["dataset"])
             + harvest(_apps_rows(), apps_prompt, APPS["cap"], APPS["dataset"]))
    if not seeds:
        print("no verification seeds; nothing written")
        return 1
    written = write_jsonl(args.root / "verification" / "seeds.jsonl", seeds)
    print(f"verification seeds: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

