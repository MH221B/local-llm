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
import re
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


_SIGNATURE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M)


def reference_signature(code: str) -> tuple[str, str] | None:
    """`("sub_list", "sub_list(list1, list2)")` from MBPP's reference solution, or None.

    MBPP's `prompt` is a vague one-liner ("Write a function to subtract two lists
    element-wise") while its tests call the reference function *by name and arity*. Without
    stating the interface the oracle is unsatisfiable: measured on the first M2 run, all
    three sampled MBPP seeds failed `verify_failed` and two of them had implemented the
    right behaviour under a self-chosen name (`subtract_lists`, `top_n`).
    """
    if not isinstance(code, str):
        return None
    match = _SIGNATURE.search(code)
    if not match:
        return None
    name = match.group(1)
    params = re.sub(r"\s+", " ", match.group(2)).strip().rstrip(",")
    return name, f"{name}({params})"


def mbpp_prompt(row: dict, index: int) -> Prompt | None:
    prompt, tests = row.get("prompt"), row.get("test_list")
    if not prompt or not isinstance(tests, list) or not tests:
        return None
    verify = {"type": "python_tests",
              "setup": "\n".join(row.get("test_imports") or []),
              "tests": tests}
    text = prompt
    signature = reference_signature(row.get("code"))
    if signature:
        name, rendered = signature
        text = (f"{prompt}\n\nName the function `{name}` and use exactly this signature: "
                f"`def {rendered}:`")
        # The verifier uses it to pick the block that defines the function rather than the
        # usage example a coding answer usually ends with.
        verify["name"] = name
    return Prompt(
        id=f"mbpp-{row.get('task_id', index)}",
        domain="coding",
        origin="prebuilt",
        source={"name": MBPP["dataset"], "config": MBPP["config"], "row": index},
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        verify=verify,
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
    mb = mbpp_prompt({"task_id": 2, "prompt": "Write a function that adds two numbers.",
                      "code": "def add(a, b):\n    return a + b", "test_imports": [],
                      "test_list": ["assert add(1, 2) == 3"]}, 0)
    bare = mbpp_prompt({"task_id": 3, "prompt": "Write something.", "test_list": ["assert f()"]}, 0)
    apps = apps_prompt({"problem_id": 1, "question": "Read two ints and print the sum.",
                        "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]})}, 0)
    stated = "def add(a, b):" in mb.messages[0]["content"][0]["text"]
    print("mbpp verify:", mb.verify["type"], len(mb.verify["tests"]))
    print("mbpp states the signature:", stated, "| verify name:", mb.verify.get("name"))
    print("row without a reference signature untouched:", "signature" not in bare.messages[0]["content"][0]["text"])
    print("apps verify:", apps.verify["type"], len(apps.verify["pairs"]))
    print("no-test row dropped:", mbpp_prompt({"prompt": "x", "test_list": []}, 0))
    return 0 if (stated and mb.verify.get("name") == "add") else 1


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

