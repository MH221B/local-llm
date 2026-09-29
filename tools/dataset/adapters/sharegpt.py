"""`conversations` shaped sources: OpenHermes-2.5 (roleplay slice) and ToolACE."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, sharegpt_turns


def build_openhermes(row: dict, index: int, ctx: Ctx) -> Example | None:
    if row.get("category") != "roleplay":
        return None
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system_prompt"))
    if messages is None:
        return None
    return Example(
        id=f"openhermes-roleplay-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


def build_toolace(row: dict, index: int, ctx: Ctx) -> Example | None:
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system"))
    if messages is None:
        return None
    return Example(
        id=f"toolace-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


if __name__ == "__main__":
    ctx = Ctx("teknium/OpenHermes-2.5", None, "roleplay", "apache-2.0")
    rp = {"category": "roleplay", "system_prompt": "You are Ada.",
          "conversations": [{"from": "human", "value": "Hello<|im_end|>"},
                            {"from": "gpt", "value": "Hi there."}]}
    ex = build_openhermes(rp, 3, ctx)
    sys_content = ex.messages[0]["content"]
    sys_text = sys_content if isinstance(sys_content, str) else sys_content[0]["text"]
    print(ex.id, [m["role"] for m in ex.messages], repr(sys_text))
    print("non-roleplay dropped:", build_openhermes({"category": "coding", "conversations": []}, 1, ctx))
    print("unterminated dropped:", build_openhermes(
        {"category": "roleplay", "conversations": [{"from": "human", "value": "hi"}]}, 2, ctx))
