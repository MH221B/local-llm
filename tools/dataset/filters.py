"""Reject predicates.

Prompt-side predicates run in M1; response-side predicates run in M2 on teacher
completions, before they can become student data. Each returns a reason string when the
record is dropped.
"""
from __future__ import annotations

import json
import re

from .canonical import Example
from . import textutil

MIN_PROMPT_TOKENS = 4
MAX_PROMPT_TOKENS = 8192
MIN_ANSWER_TOKENS = 16
MAX_ANSWER_TOKENS = 16384

# The student template stores the CoT as angle-bracket delimited tags, read from the
# GGUF's own chat template in Task 15 rather than assumed: the template writes
# '<think>' / '</think>', each on its own line. The earlier open constant
# here was missing its '<', which is what made substring counting look unreliable.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Spec section 9.4's language filter: reject rows dominated by non-Latin scripts.
# Symbols common in maths (Greek, arrows) are deliberately not in the class.
_NON_ENGLISH = re.compile(
    r"[\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
NON_ENGLISH_RATIO_MAX = 0.02

def assistant_text(ex: Example) -> str:
    for m in reversed(ex.messages):
        if m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def prompt_reject_reason(text: str, tokens: int) -> str | None:
    """Prompt-side predicates, used by the M1 pipeline."""
    if not text.strip():
        return "empty_prompt"
    if tokens < MIN_PROMPT_TOKENS:
        return "prompt_too_short"
    if tokens > MAX_PROMPT_TOKENS:
        return "prompt_too_long"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
    return None


MIN_TURN_TOKENS = 4


def response_reject_reason(text: str, tokens: int,
                           min_tokens: int = MIN_ANSWER_TOKENS) -> str | None:
    """Structural response-side predicates on one assistant span (spec 10.1, 11.3).

    The response-quality predicates were removed during M2 Task 15. Once the teacher's
    reasoning trace is stored inside `content`, each one false-fires on legitimate
    reasoning: refusal markers are quoted in the trace, the trace echoes its own test
    fixtures (so the n-gram check trips), and a trace or answer may legitimately carry
    non-Latin script. Measured: 2 of the first 3 seeded rows were dropped as false
    positives. Decided: accept those rows outright; M3's reward filter owns response
    quality. What remains is structure only, which cannot false-fire.

    ponytail: M2 ships no response-quality gate at all. M3's reward filter is the gate.
    """
    if not text.strip():
        return "empty_assistant"
    if text.count(THINK_OPEN) != text.count(THINK_CLOSE):
        return "unbalanced_think_tags"
    if tokens < min_tokens:
        return "too_short"
    if tokens > MAX_ANSWER_TOKENS:
        return "too_long"
    return None


def _response_text(message: dict) -> str:
    """Assistant text for filtering. A tool-call-only turn contributes its calls, so a
    legitimate function call is not mistaken for an empty turn."""
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    else:
        text = ""
    if not text.strip() and message.get("tool_calls"):
        return json.dumps(message["tool_calls"], ensure_ascii=False)
    return text


def example_reject_reason(ex: Example) -> str | None:
    """Single-turn response predicates. M2 runs these on teacher completions."""
    for m in reversed(ex.messages):
        if m.get("role") == "assistant":
            return response_reject_reason(_response_text(m), ex.tokens)
    return "empty_assistant"


def trajectory_reject_reason(traj) -> str | None:
    """Run the response predicates on every assistant turn; reject if any turn fails.

    Chat turns are shorter than standalone answers, so the per-turn minimum is
    `MIN_TURN_TOKENS`.
    """
    for i, m in enumerate(traj.messages):
        if m.get("role") != "assistant":
            continue
        text = _response_text(m)
        reason = response_reject_reason(text, textutil.count_tokens(text),
                                        min_tokens=MIN_TURN_TOKENS)
        if reason:
            return f"turn {i}: {reason}"
    return None


if __name__ == "__main__":
    from .canonical import Example, Trajectory

    def mk(text, tokens=200):
        return Example(id="f", domain="reasoning", origin="prebuilt", source={"name": "s"},
                       messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]},
                                 {"role": "assistant", "content": text}], tokens=tokens)

    print("prompt clean:", prompt_reject_reason("Solve x^2 = 4.", 200))
    print("prompt empty:", prompt_reject_reason("", 0))
    print("clean:", example_reject_reason(mk("A clear worked answer.")))
    print("unbalanced:", example_reject_reason(mk("<think>reasoning without a close tag")))
    print("short:", example_reject_reason(mk("ok", tokens=3)))
    print("long:", example_reject_reason(mk("a b c", tokens=99999)))
    # The removed quality predicates must no longer drop anything. Every measured false
    # positive from Task 15 is pinned here: a quoted refusal, an echoed test fixture, a
    # genuinely repeating answer, and a non-Latin answer.
    print("refusal text accepted:", example_reject_reason(mk("I cannot help with that request.")))
    fixture = " ".join(["zebra"] * 40)
    print("repetitive trace accepted:",
          example_reject_reason(mk(f"<think>test with {fixture}</think>\n\nThe counter works.")))
    print("repeating answer accepted:",
          example_reject_reason(mk(" ".join(["a b c d e f g h"] * 30))))
    print("non-english accepted:", example_reject_reason(mk("这是一段中文回答，用于测试语言过滤。")))

    good = Trajectory(id="t1", domain="roleplay", origin="teacher", source={"name": "s"},
                      messages=[{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "Hello there, how are you?"}])
    print("traj clean:", trajectory_reject_reason(good))

    tool_only = Trajectory(
        id="t3", domain="coding", origin="teacher", source={"name": "s"},
        messages=[{"role": "user", "content": "book it"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "c", "type": "function",
                                   "function": {"name": "book", "arguments": "{\"n\": 2}"}}]}])
    print("traj tool-only kept:", trajectory_reject_reason(tool_only))
