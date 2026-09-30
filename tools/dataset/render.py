"""Render gate: apply the student's real chat template to a sample (spec section 11.2)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from jinja2 import Environment
from jinja2.exceptions import TemplateError

from . import canonical
from .canonical import iter_jsonl

# The student template's literal delimiters (verified against the GGUF in Task 16).
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def render(template: str, messages: list[dict], add_generation_prompt: bool = False) -> str:
    env = Environment(trim_blocks=False, lstrip_blocks=False)
    tpl = env.from_string(template)
    return tpl.render(messages=messages, add_generation_prompt=add_generation_prompt,
                      bos_token="", eos_token="")


def strip_image_data(messages: list[dict]) -> list[dict]:
    """Templates that cannot handle image parts get a text-only view for this check."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(m)
            continue
        parts = [p for p in content if p.get("type") == "text"]
        out.append({"role": m["role"], "content": parts or [{"type": "text", "text": ""}]})
    return out


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def tool_calls_in(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            out.extend(m.get("tool_calls") or [])
    return out


def check_multiturn(template: str, rows: list[dict], sample: int = 50) -> int:
    """Multi-turn render gate: think survival, mask markers, tool round-trip."""
    has_markers = "{% generation %}" in template and "{% endgeneration %}" in template
    problems: list[str] = []
    think_stored = think_kept = 0
    for row in rows[:sample]:
        messages = strip_image_data(row["messages"])
        stored = sum(m["content"].count(THINK_OPEN) for m in messages
                     if m["role"] == "assistant" and isinstance(m.get("content"), str))
        try:
            # The template iterates `tool_call.arguments|items`, which needs a mapping;
            # stored arguments are JSON strings. `for_template` is that one conversion.
            text = render(template, canonical.for_template(messages))
        except TemplateError as exc:
            problems.append(f"{row['id']}: {exc}")
            continue
        kept = len(_THINK_BLOCK.findall(text))
        think_stored += stored
        think_kept += kept
        # The template fabricates a `<think>` block for every assistant turn after the
        # last user turn (student.jinja splits a stored block out of `content` and re-emits
        # it in its own delimiters). So the expected count is those turns, NOT the number
        # of stored tag pairs: a content-less scaffold legitimately renders more than it
        # stores. More than one block per qualifying turn is the real failure.
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"),
                        default=-1)
        expected = sum(1 for i, m in enumerate(messages)
                       if m.get("role") == "assistant" and i > last_user)
        if kept > expected:
            problems.append(
                f"{row['id']}: rendered {kept} think blocks, expected {expected}")
        for call in tool_calls_in(messages):
            args = canonical.tool_call_arguments(call)
            name = (call.get("function") or {}).get("name")
            if name and f"<function={name}>" not in text:
                problems.append(f"{row['id']}: tool name {name!r} did not render")
            for key, value in args.items():
                # The template writes `<parameter=key>\nvalue\n</parameter>`; check the
                # parameter survived rather than the raw JSON string, which it cannot.
                if f"<parameter={key}>" not in text:
                    problems.append(f"{row['id']}: parameter {key!r} did not render")
    print(f"generation mask markers present: {has_markers}")
    print(f"think blocks stored={think_stored} kept={think_kept} (Qwen3 keeps the latest segment)")
    print(f"multi-turn problems: {len(problems)}")
    for p in problems[:5]:
        print("  ", p)
    return 1 if problems else 0


def run(*, dataset: Path, template_path: Path, sample: int,
        multiturn: bool = False) -> int:
    template_text = template_path.read_text(encoding="utf-8")
    rows = list(iter_jsonl(dataset))[:sample]
    if multiturn:
        return check_multiturn(template_text, rows, sample=sample)
    failures = 0
    tags = {"open": 0, "close": 0}
    for row in rows:
        try:
            text = render(template_text, strip_image_data(row["messages"]))
        except TemplateError as exc:
            failures += 1
            print(f"FAIL {row['id']}: {exc}")
            continue
        tags["open"] += text.count(THINK_OPEN)
        tags["close"] += text.count(THINK_CLOSE)

    # M1 prompts carry no think tags at all, so the gate also proves the template
    # preserves a stored think block (spec section 11.2).
    probe = [{"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
             {"role": "assistant",
              "content": f"{THINK_OPEN}\nOne plus one is two.\n{THINK_CLOSE}\n\nTwo."}]
    try:
        probe_text = render(template_text, strip_image_data(probe))
        probe_ok = (THINK_OPEN in probe_text and THINK_CLOSE in probe_text
                    and "One plus one is two." in probe_text)
    except TemplateError as exc:
        probe_ok = False
        print(f"FAIL probe: {exc}")

    # The student must mimic the teacher's reasoning process, so the generation prompt
    # itself has to open a think block (spec sections 5 and 11.2).
    try:
        gen_text = render(template_text, strip_image_data(probe[:1]), add_generation_prompt=True)
        thinking_ok = THINK_OPEN in gen_text
    except TemplateError as exc:
        thinking_ok = False
        print(f"FAIL generation prompt: {exc}")

    print(f"rendered {len(rows)} examples, {failures} template failures")
    print(f"think tags in rendered output: open={tags['open']} close={tags['close']}")
    print(f"probe think block survived: {probe_ok}")
    print(f"generation prompt opens thinking: {thinking_ok}")
    if failures or not probe_ok or not thinking_ok:
        print("VERDICT: template cannot render this data as written")
        return 1
    balanced = tags["open"] == tags["close"]
    print("VERDICT:", "think tags balanced" if balanced else "think tags UNBALANCED")
    return 0 if balanced else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chat-template render gate")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--multiturn", action="store_true",
                    help="trajectory gates: think survival, mask markers, tool round-trip")
    args = ap.parse_args(argv)
    return run(dataset=args.dataset, template_path=args.template, sample=args.sample,
               multiturn=args.multiturn)


if __name__ == "__main__":
    sys.exit(main())
