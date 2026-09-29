"""HuggingFaceTB/smoltalk configs. Already in the `messages` shape (spec section 6)."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, text_part

# Spec section 7.3 asks for the smol-magpie-ultra slice to be filtered on
# `quality` / `reward_model_score`. Observed values run excellent/good/average/poor,
# with scores in the 0.15-0.20 band.
MAGPIE_MIN_SCORE = 0.15
MAGPIE_QUALITIES = ("excellent", "good")


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    if ctx.config == "smol-magpie-ultra":
        score = row.get("reward_model_score")
        if row.get("quality") not in MAGPIE_QUALITIES:
            return None
        if not isinstance(score, (int, float)) or score < MAGPIE_MIN_SCORE:
            return None
    messages: list[dict] = []
    for m in row.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": [text_part(content)]})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    label = ctx.config or "default"
    return Example(
        id=f"smoltalk-{label}-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


if __name__ == "__main__":
    from .base import Ctx

    ctx = Ctx("HuggingFaceTB/smoltalk", "metamathqa-50k", "reasoning", "apache-2.0")
    row = {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
    ]}
    ex = build(row, 7, ctx)
    print(ex.id, ex.domain, [m["role"] for m in ex.messages])
    print("dropped:", build({"messages": [{"role": "user", "content": "x"}]}, 0, ctx))
