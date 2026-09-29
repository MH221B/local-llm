"""NousResearch/hermes-function-calling-v1. Carries a `tools` schema per example."""
from __future__ import annotations

import json

from ..canonical import Example
from .base import Ctx, sharegpt_turns


def _valid_tool_schema(tools: list) -> bool:
    """A tool list must hold OpenAI-style function objects; anything else breaks rendering."""
    for entry in tools:
        if not isinstance(entry, dict):
            return False
        fn = entry.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            return False
    return True


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system"))
    if messages is None:
        return None
    # The column ships JSON-encoded tool schemas as a string.
    tools = row.get("tools")
    if isinstance(tools, str):
        tools = tools.strip() or None
        if tools is not None:
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                return None
    if tools is not None:
        if not isinstance(tools, list) or not _valid_tool_schema(tools):
            return None
    return Example(
        id=f"hermesfc-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
        tools=tools or None,
        meta={"category": row.get("category"), "task": row.get("task")},
    )


if __name__ == "__main__":
    ctx = Ctx("NousResearch/hermes-function-calling-v1", "func_calling", "coding", "apache-2.0")
    row = {
        "conversations": [{"from": "human", "value": "What is the weather in Paris?"},
                          {"from": "gpt", "value": "<tool_call>{\"name\": \"get_weather\"}</tool_call>"}],
        "tools": json.dumps([{"type": "function", "function": {"name": "get_weather"}}]),
        "category": "weather", "task": "tool_call",
    }
    ex = build(row, 11, ctx)
    print(ex.id, "tools:", len(ex.tools), "meta:", ex.meta)
    print("bad tools dropped:", build({**row, "tools": "not-a-list"}, 12, ctx))
