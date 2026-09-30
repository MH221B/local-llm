"""Execute a teacher completion against a seed's verification spec.

Used by M2 to drop failed distillation attempts before they become student data. This
runs generated code with the local interpreter; it is a correctness check, not a security
sandbox — run generation and verification on a disposable environment/profile.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

_FENCE = "`" * 3
_CODE_FENCE = re.compile(_FENCE + r"(?:python)?\s*(.*?)" + _FENCE, re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_FINAL = re.compile(r"####\s*([^\n]+)")


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_code(completion: str) -> str:
    """Every fenced block, joined.

    A coding answer usually ends with a usage example in its own fence, so taking the
    last block alone executes the demo without the definition. Measured: three MBPP seeds
    failed `NameError: name 'tuple_intersection' is not defined` - names that appear only
    in the final block.
    """
    blocks = [b.strip() for b in _CODE_FENCE.findall(completion)]
    return "\n\n".join(blocks) if blocks else strip_think(completion).strip()


def run_python(source: str, stdin: str = "", timeout: float = 10.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run([sys.executable, "-c", source], input=stdin,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"exit {proc.returncode}: {tail[-1] if tail else 'no output'}"
    return True, proc.stdout


def check_python_tests(completion: str, tests: list[str], setup: str = "",
                       timeout: float = 10.0) -> tuple[bool, str]:
    body = "\n".join([setup, extract_code(completion), *tests])
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
                                  setup=verify.get("setup") or "")
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
    print("answer match:", check_answer("<think>\n4\n</think>\n\n\\boxed{4}", "4"))
    print("answer mismatch:", check_answer("The answer is 5.", "4"))
    print("prose numeric answer:", check_answer("James made **$126** from selling all the water.", "126"))
    print("prose wrong number:", check_answer("James made **$105** from selling all the water.", "126"))

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
