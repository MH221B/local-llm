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
LOOP_RATIO_MAX = 0.05

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


def _loop_ratio(text: str) -> float:
    """Degenerate-repeat ratio, scored on the answer span only.

    The reasoning trace is excluded on purpose: it repeats itself legitimately. Measured
    on a coding row (correct answer, `All tests passed!`), the trace scored 0.058 - driven
    by a run of the word `zebra` the model used as its own test fixture - while the answer
    scored 0.045. Only the answer reaches the student's response slot, so only the answer
    is scored. The trace is a different distribution: re-printed code while iterating.

    ponytail: the trace is left unchecked, so a severely looping trace can still ship.
    Legitimate traces measured ~0.06; if M3's reward filter does not catch degenerate
    traces, add a trace guard at a much higher bar (e.g. 0.25), not at LOOP_RATIO_MAX.
    """
    if THINK_CLOSE in text:
        _, answer = text.split(THINK_CLOSE, 1)
    else:
        answer = text
    return textutil.repeating_ngram_ratio(answer)


def response_reject_reason(text: str, tokens: int,
                           min_tokens: int = MIN_ANSWER_TOKENS) -> str | None:
    """Response-side predicates on one assistant span (M2, spec sections 10.1, 11.3)."""
    if not text.strip():
        return "empty_assistant"
    if text.count(THINK_OPEN) != text.count(THINK_CLOSE):
        return "unbalanced_think_tags"
    if tokens < min_tokens:
        return "too_short"
    if tokens > MAX_ANSWER_TOKENS:
        return "too_long"
    if _loop_ratio(text) > LOOP_RATIO_MAX:
        return "looping"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
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
    print("loop:", example_reject_reason(mk(" ".join(["a b c d e f g h"] * 30))))
    # Regression: the trace drafts the answer, so the joined text repeats. Scored per
    # span this is clean; scored joined it would trip `looping`.
    drafted = " ".join(f"word{i}" for i in range(20))
    print("draft-echo not looping:",
          example_reject_reason(mk(f"<think>Draft: {drafted}</think>\n\n{drafted}")))
    # Regression: a trace legitimately repeats itself (echoed test fixture, re-printed
    # code). Only the answer span is scored, so this row is kept.
    fixture = " ".join(["zebra"] * 40)
    print("repetitive trace, clean answer kept:",
          example_reject_reason(mk(f"<think>test with {fixture}</think>\n\nThe counter works.")))
    print("repeating answer still looping:",
          example_reject_reason(mk(f"<think>short plan</think>\n\n{' '.join(['a b c d e f g h'] * 30)}")))
    print("non-english:", example_reject_reason(mk("这是一段中文回答，用于测试语言过滤。")))

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
