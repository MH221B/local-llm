"""Execute a teacher completion against a seed's verification spec.

Used by M2 to drop failed distillation attempts before they become student data. This
runs generated code with the local interpreter; it is a correctness check, not a security
sandbox — run generation and verification on a disposable environment/profile.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

_FENCE = "`" * 3
_CODE_FENCE = re.compile(_FENCE + r"(?:python)?\s*(.*?)" + _FENCE, re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_FINAL = re.compile(r"####\s*([^\n]+)")


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_code(completion: str, name: str | None = None) -> str:
    """The block to execute.

    Prefer the block that defines `name` (the seed states it), then the first block with a
    module-level `def`, then the last block. Joining every block looked tempting and is
    wrong: a trailing usage example can be a fragment with a bare `return`, so the joined
    program dies with `SyntaxError: 'return' outside function` and hides the real reason
    (measured on an MBPP row whose true failure was `NameError`).
    """
    blocks = [b.strip() for b in _CODE_FENCE.findall(completion)]
    if not blocks:
        return strip_think(completion).strip()
    if name:
        wanted = re.compile(rf"^\s*def\s+{re.escape(name)}\s*\(", re.M)
        for block in blocks:
            if wanted.search(block):
                return block
    for block in blocks:
        if re.search(r"^\s*def\s+\w+\s*\(", block, re.M):
            return block
    return blocks[-1]


def run_python(source: str, stdin: str = "", timeout: float = 10.0) -> tuple[bool, str]:
    # The program goes in a temp file, not `python -c <source>`: Windows caps a command line
    # at ~32k chars, and a long generated solution (13-15k tokens) blew that limit and killed
    # a whole M3 run with `FileNotFoundError: [WinError 206]`. `stdin` must stay free for the
    # program's own input (check_python_io feeds test cases there), so a file is the fix.
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".py", prefix="m3verify-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
        proc = subprocess.run([sys.executable, path], input=stdin,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"exit {proc.returncode}: {tail[-1] if tail else 'no output'}"
    return True, proc.stdout


def check_python_tests(completion: str, tests: list[str], setup: str = "",
                       timeout: float = 10.0, name: str | None = None) -> tuple[bool, str]:
    body = "\n".join([setup, extract_code(completion, name), *tests])
    ok, info = run_python(body, timeout=timeout)
    return (True, "") if ok else (False, info)


def check_python_io(completion: str, pairs: list[list[str]],
                    timeout: float = 10.0) -> tuple[bool, str]:
    # Stdout is compared exactly, so a trailing usage example would pollute it: the last
    # block only here, unlike `extract_code`, which joins them all.
    blocks = [b.strip() for b in _CODE_FENCE.findall(completion)]
    code = blocks[-1] if blocks else strip_think(completion).strip()
    for stdin, expected in pairs[:20]:
        ok, info = run_python(code, stdin=stdin, timeout=timeout)
        if not ok:
            return False, info
        if info.strip() != expected.strip():
            return False, f"output mismatch: {info.strip()[:40]!r} != {expected.strip()[:40]!r}"
    return True, ""


def extract_answer(text: str) -> str:
    clean = strip_think(text)
    boxed = _BOXED.findall(clean)
    if boxed:
        return boxed[-1].strip()
    finals = _HASH_FINAL.findall(clean)
    if finals:
        return finals[-1].strip()
    return clean.strip().splitlines()[-1].strip() if clean.strip() else ""


def normalise_answer(text: str) -> str:
    text = text.strip().lower().replace(",", "").replace("$", "").replace("\\%", "%")
    return re.sub(r"\s+", " ", text).rstrip(".")


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def check_answer(completion: str, gold: str) -> bool:
    """Exact match after normalisation, with a numeric fallback for prose answers.

    The pool prompts carry no "end with just the answer" instruction, so a correct
    reasoning answer usually ends with prose. Measured: a GSM8K row with the right value
    ended `James made **$126** from selling all the water.` and the last-line rule
    rejected it ("answer mismatch"). When the gold normalises to a number, also accept
    the answer's last numeric token - which is where the model puts its final value.
    """
    gold_norm = normalise_answer(gold)
    if normalise_answer(extract_answer(completion)) == gold_norm:
        return True
    if _NUMBER.fullmatch(gold_norm):
        numbers = _NUMBER.findall(strip_think(completion))
        if numbers and normalise_answer(numbers[-1]) == gold_norm:
            return True
    return False


def check_record(verify: dict | None, completion: str) -> tuple[bool, str]:
    """Dispatch on the seed's verify spec. Returns (passed, reason)."""
    if verify is None:
        return True, ""
    vtype = verify.get("type")
    if vtype == "python_tests":
        return check_python_tests(completion, verify.get("tests") or [],
                                  setup=verify.get("setup") or "",
                                  name=verify.get("name"))
    if vtype == "python_io":
        return check_python_io(completion, verify.get("pairs") or [])
    if vtype == "answer_match":
        ok = check_answer(completion, str(verify.get("gold") or ""))
        return (True, "") if ok else (False, "answer mismatch")
    return False, f"unknown verify type {vtype!r}"


_JSON_ERROR = re.compile(r'"error"\s*:')
# Known limitation: a legitimate tool result may carry an `"error"` key (a well-formed
# API error response is still a valid observation). Measured on ToolACE, ~1.6% of tool
# results match and are dropped as "error-laden". Left as specified; the rate shows up in
# the trajectory drops, and M3 can narrow the pattern to a top-level error field.


def check_turns(trajectory) -> list[str]:
    """Turn-level structural checks (spec section 10.2). Empty list means valid."""
    from .canonical import declared_tool_names
    problems: list[str] = []
    names = declared_tool_names(trajectory.tools) if trajectory.tools else None
    calls: dict[str, int] = {}
    answered: set[str] = set()
    for i, m in enumerate(trajectory.messages):
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if not (strip_think(str(m.get("content") or "")).strip() or tcs):
                problems.append(f"empty assistant turn at {i}")
            for tc in tcs:
                cid = tc.get("id")
                fn = tc.get("function") or {}
                if not cid:
                    problems.append(f"tool_call without id at {i}")
                    continue
                if names is not None and fn.get("name") not in names:
                    problems.append(f"undeclared tool {fn.get('name')!r} at {i}")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        problems.append(f"tool_call arguments not JSON at {i}")
                calls[cid] = i
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in calls:
                problems.append(f"tool result without a preceding call at {i}")
            else:
                answered.add(cid)
            text = str(m.get("content") or "")
            if text.strip().lower().startswith("error:") or _JSON_ERROR.search(text):
                problems.append(f"error-laden tool result at {i}")
    for cid, idx in calls.items():
        if cid not in answered:
            problems.append(f"call {cid!r} at {idx} has no tool result")
    return problems


def check_trajectory(verify_spec: dict | None, trajectory) -> tuple[bool, str]:
    """Trajectory-level check: structure first, then the final visible answer (spec 10.1).

    `think` blocks are stripped before any oracle runs, so reasoning cannot satisfy a test
    by accident (spec section 10.1).
    """
    problems = check_turns(trajectory)
    if problems:
        return False, problems[0]
    if verify_spec is not None:
        from .canonical import final_assistant_text
        return check_record(verify_spec, final_assistant_text(trajectory.messages))
    return True, ""


if __name__ == "__main__":
    from .canonical import Trajectory

    ok, _ = check_python_tests("def add(a, b):\n    return a + b",
                               ["assert add(1, 2) == 3"])
    bad, why_bad = check_python_tests("def add(a, b):\n    return a - b",
                                      ["assert add(1, 2) == 3"])
    io_ok, _ = check_python_io("print(sum(map(int, input().split())))", [["1 2", "3"]])
    print("tests positive:", ok, "| negative:", bad, "|", why_bad)
    print("io positive:", io_ok)

    # Regression: a program longer than Windows' ~32k-char command line must not crash the
    # run with WinError 206 (it goes in a temp file now, not a `-c` argv).
    long_src = "x = 1\n" * 8000 + "print(x)"
    long_ok, long_info = run_python(long_src)
    print("long source (>32k chars):", long_ok, "|", long_info.strip(), "| chars:", len(long_src))
    print("answer match:", check_answer("<think>\n4\n</think>\n\n\\boxed{4}", "4"))
    print("answer mismatch:", check_answer("The answer is 5.", "4"))
    print("prose numeric answer:", check_answer("James made **$126** from selling all the water.", "126"))
    print("prose wrong number:", check_answer("James made **$105** from selling all the water.", "126"))

    # Regression: a coding answer that ends with a fragment. Joining every fence makes the
    # program invalid (`return` outside a function) and hides the real failure.
    fragmented = ("```python\ndef top_n(items, n):\n    return sorted(items)[-n:]\n```\n\n"
                  "```python\nprint(top_n([3, 1], 1))\n```\n\n"
                  "```python\nfor x in items:\n    return x\n```")
    blocks = _CODE_FENCE.findall(fragmented)
    joined_invalid = False
    try:
        compile("\n\n".join(blocks), "<joined>", "exec")
    except SyntaxError:
        joined_invalid = True
    picked = extract_code(fragmented, "larg_nnum")
    print("joining every fence is invalid:", joined_invalid)
    print("picks the defining block:", picked.startswith("def top_n"))
    print("picks the named block when present:",
          extract_code("```python\ndef a():\n    return 1\n```\n\n```python\ndef sub_list(x):\n    return x\n```",
                       "sub_list").startswith("def sub_list"))

    good = Trajectory(
        id="t1", domain="coding", origin="teacher", source={"name": "s"},
        messages=[
            {"role": "user", "content": "Book a table."},
            {"role": "assistant", "content": "<think>need the tool</think>",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "book", "arguments": "{\"n\": 2}"}}]},
            {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Booked."},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}])
    orphan = Trajectory(
        id="t2", domain="coding", origin="teacher", source={"name": "s"},
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "nope", "arguments": "{bad"}}]},
            {"role": "assistant", "content": "z"},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}])
    print("turns good:", check_turns(good))
    print("turns bad:", len(check_turns(orphan)), "problems")
    print("traj verify pass:", check_trajectory({"type": "answer_match", "gold": "Booked."}, good))
