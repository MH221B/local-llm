"""Execute a teacher completion against a seed's verification spec.

Used by M2 to drop failed distillation attempts before they become student data. This
runs generated code with the local interpreter; it is a correctness check, not a security
sandbox — run generation and verification on a disposable environment/profile.
"""
from __future__ import annotations

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
    blocks = _CODE_FENCE.findall(completion)
    return (blocks[-1] if blocks else strip_think(completion)).strip()


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
    code = extract_code(completion)
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


def check_answer(completion: str, gold: str) -> bool:
    return normalise_answer(extract_answer(completion)) == normalise_answer(gold)


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


if __name__ == "__main__":
    ok, _ = check_python_tests("def add(a, b):\n    return a + b",
                               ["assert add(1, 2) == 3"])
    bad, why_bad = check_python_tests("def add(a, b):\n    return a - b",
                                      ["assert add(1, 2) == 3"])
    io_ok, _ = check_python_io("print(sum(map(int, input().split())))", [["1 2", "3"]])
    print("tests positive:", ok, "| negative:", bad, "|", why_bad)
    print("io positive:", io_ok)
    print("answer match:", check_answer("<think>\n4\n</think>\n\n\\boxed{4}", "4"))
    print("answer mismatch:", check_answer("The answer is 5.", "4"))
