"""Reject predicates.

Prompt-side predicates run in M1; response-side predicates run in M2 on teacher
completions, before they can become student data. Each returns a reason string when the
record is dropped.
"""
from __future__ import annotations

import re

from .canonical import Example
from . import textutil

MIN_PROMPT_TOKENS = 4
MAX_PROMPT_TOKENS = 8192
MIN_ANSWER_TOKENS = 16
MAX_ANSWER_TOKENS = 16384
LOOP_RATIO_MAX = 0.05

# The student template stores the CoT as literal ASCII tags; confirmed against the
# GGUF template in Task 16 Step 2 (spec sections 5 and 11.2). The open tag is followed by
# content on the same line (spec section 5), so the marker alone is what to count.
#
# Known limitation: bare-substring counting cannot tell a real tag from the word
# "thinking" in prose ("I was thinking about it"). That only affects M2 review of teacher
# text; M1 prompts carry no tags at all.
THINK_OPEN = " thinking"
THINK_CLOSE = "</think>"

# Spec section 9.4's language filter: reject rows dominated by non-Latin scripts.
# Symbols common in maths (Greek, arrows) are deliberately not in the class.
_NON_ENGLISH = re.compile(
    r"[\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
NON_ENGLISH_RATIO_MAX = 0.02

# The spec (section 9.4) calls this an inverted refusal filter. It is aimed at the
# prebuilt uncensored sets, whose answers ship refusals by design (section 7.4). M1
# strips those answers, and the teacher is abliterated so it does not refuse, so this
# check runs against text that should rarely contain a refusal.
#
# Known limitation: bare substring matching also fires on legitimate answers that open
# with a limitation ("I cannot emphasize enough how important units are", "I must decline
# to guess and will compute it instead"). Left as specified; the rate is visible in the
# M2 pass-rate report.
REFUSAL_MARKERS = (
    "i cannot", "i can't", "i can not", "i'm unable", "i am unable", "i won't",
    "i will not", "i must decline", "as an ai language model", "i'm sorry, but",
)


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


def example_reject_reason(ex: Example) -> str | None:
    """Response-side predicates. M2 runs these on teacher completions before writing."""
    text = assistant_text(ex)
    if not text.strip():
        return "empty_assistant"
    if text.count(THINK_OPEN) != text.count(THINK_CLOSE):
        return "unbalanced_think_tags"
    if ex.tokens < MIN_ANSWER_TOKENS:
        return "too_short"
    if ex.tokens > MAX_ANSWER_TOKENS:
        return "too_long"
    if textutil.repeating_ngram_ratio(text) > LOOP_RATIO_MAX:
        return "looping"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
    low = text.lower()
    if any(marker in low for marker in REFUSAL_MARKERS):
        return "refusal"
    return None


if __name__ == "__main__":
    from .canonical import Example

    def mk(text, tokens=200):
        return Example(id="f", domain="reasoning", origin="prebuilt", source={"name": "s"},
                       messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]},
                                 {"role": "assistant", "content": text}], tokens=tokens)

    print("prompt clean:", prompt_reject_reason("Solve x^2 = 4.", 200))
    print("prompt empty:", prompt_reject_reason("", 0))
    print("clean:", example_reject_reason(mk("A clear worked answer.")))
    print("refusal:", example_reject_reason(mk("I cannot help with that request.")))
    print("unbalanced:", example_reject_reason(mk(" thinkingreasoning without a close tag")))
    print("short:", example_reject_reason(mk("ok", tokens=3)))
    print("loop:", example_reject_reason(mk(" ".join(["a b c d e f g h"] * 30))))
    print("non-english:", example_reject_reason(mk("这是一段中文回答，用于测试语言过滤。")))
