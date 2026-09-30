"""Prebuilt multi-turn sources -> prompt trajectories for M2.

Assistant turns are teacher-written in M2, so this module keeps only the context the
teacher needs: system and user turns, the declared `tools` schema, and — for tool-calling
sources — the observations (`tool` results) that follow each call. Assistant turns are
kept as inert scaffolds (their content is ignored, their `tool_calls` only exist so a
following observation has a valid `tool_call_id`) and are marked `"_scaffold": true`.
`trajectory.generate_prebuilt` regenerates them and drops the flag before writing.
"""
from __future__ import annotations

import json

from .canonical import Trajectory
from .adapters.base import Ctx, system_msg, user_msg, assistant_msg, text_part

HERMES_DATASET = "NousResearch/hermes-function-calling-v1"
_TAG = "<tool_call>"


def _blocks(text: str, tag: str) -> list[str]:
    out, chunks = [], text.split(f"<{tag}>")
    for chunk in chunks[1:]:
        out.append(chunk.split(f"</{tag}>")[0].strip())
    return out


def _hermes_tool_schema(raw) -> list | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    return raw


def _hermes_messages(conversations: list) -> list[dict] | None:
    messages: list[dict] = []
    call_seq = 0
    for turn in conversations:
        frm, value = turn.get("from"), turn.get("value")
        if not isinstance(value, str):
            return None
        if frm == "system":
            text = value
            if "<tools>" in text:
                text = text.split("<tools>")[0].strip()
            messages.append(system_msg(text))
        elif frm == "human":
            messages.append(user_msg([text_part(value)]))
        elif frm == "gpt":
            blocks = _blocks(value, "tool_call")
            prose = value
            for b in blocks:
                prose = prose.replace(f"{_TAG}\n{b}\n</tool_call>", "")
                prose = prose.replace(f"{_TAG}{b}</tool_call>", "")
            prose = prose.strip()
            if blocks:
                calls = []
                for b in blocks:
                    try:
                        obj = json.loads(b)
                    except json.JSONDecodeError:
                        return None
                    name = obj.get("name")
                    args = obj.get("arguments", {})
                    if not isinstance(name, str) or not name:
                        return None
                    calls.append({"id": f"call_{call_seq}", "type": "function",
                                  "function": {"name": name,
                                               "arguments": json.dumps(args, ensure_ascii=False)}})
                    call_seq += 1
                msg = assistant_msg(prose)
                msg["tool_calls"] = calls
                msg["_scaffold"] = True
                messages.append(msg)
            else:
                messages.append(assistant_msg(value))
        elif frm == "tool":
            responses = _blocks(value, "tool_response")
            content = responses[0] if responses else value
            messages.append({"role": "tool", "content": content, "tool_call_id": ""})
        else:
            return None
    if not messages or messages[-1]["role"] != "assistant":
        return None
    # Pair each scaffold call with the tool message that follows it, in order.
    pending: list[str] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("_scaffold"):
            pending = [c["id"] for c in m["tool_calls"]]
        elif m["role"] == "tool":
            if not pending:
                return None
            m["tool_call_id"] = pending.pop(0)
    if any(m["role"] == "tool" and not m["tool_call_id"] for m in messages):
        return None
    return messages


def build_hermes(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    tools = _hermes_tool_schema(row.get("tools"))
    messages = _hermes_messages(row.get("conversations") or [])
    if messages is None:
        return None
    if tools is not None and any(
            m.get("tool_calls") and c["function"]["name"] not in
            {e["function"]["name"] for e in tools if isinstance(e.get("function"), dict)}
            for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []):
        return None
    return Trajectory(
        id=f"hermesfc-{index}", domain=ctx.domain, origin="prebuilt",
        source=ctx.source(index), messages=messages, tools=tools,
        meta={"category": row.get("category"), "task": row.get("task"), "simulated": False},
    )


import re

TOOLACE_DATASET = "Team-ACE/ToolACE"
# Argument values are quoted strings or scalars and contain no parentheses, so a call can
# be split at its outermost parens. Function names may contain spaces and balanced parens
# ("User Feed (Video Posts) V2"), so we split the body on top-level commas (tracking paren
# depth and quote state) and take the last "(" as the call's opening paren.
_ARG = re.compile(r"(\w+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\d+|True|False|None)")


def _split_calls(body: str) -> list[str]:
    """Split `A(x), B(y)` into ['A(x)', 'B(y)'] at top-level commas only."""
    parts: list[str] = []
    depth, quote, start = 0, "", 0
    for i, ch in enumerate(body):
        if quote:
            if ch == quote and body[i - 1] != "\\":
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p.strip() for p in parts if p.strip()]


def _toolace_calls(text: str) -> list[dict] | None:
    """Parse ToolACE's `[Name(arg=val), ...]` DSL into structured calls."""
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return None
    calls: list[dict] = []
    for chunk in _split_calls(body[1:-1]):
        if not chunk.endswith(")"):
            return None
        name, args_blob = chunk[:-1].rsplit("(", 1)
        name = name.strip()
        if not name:
            return None
        args: dict = {}
        for key, raw in _ARG.findall(args_blob):
            try:
                args[key] = json.loads(raw)
            except json.JSONDecodeError:
                args[key] = raw.strip("'\"")
        calls.append({"id": f"call_{len(calls)}", "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls or None


def _toolace_schema(system: str) -> list | None:
    """Extract the embedded JSON function list and wrap it in the OpenAI shape."""
    start = system.find("[")
    end = system.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(system[start:end + 1])
    except json.JSONDecodeError:
        return None
    tools = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        tools.append({"type": "function", "function": {
            "name": entry["name"], "description": entry.get("description", ""),
            "parameters": entry.get("parameters", {})}})
    return tools or None


def build_toolace(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    system = row.get("system") or ""
    tools = _toolace_schema(system)
    messages: list[dict] = []
    # Keep the prose preamble before the embedded JSON function list as the system turn;
    # the function list itself becomes the structured `tools` field.
    if "[" in system:
        preamble = system[:system.find("[")].strip()
        if preamble:
            messages.append(system_msg(preamble))
    pending: list[str] = []
    for turn in row.get("conversations") or []:
        frm, value = turn.get("from"), turn.get("value")
        if not isinstance(value, str):
            return None
        if frm == "user":
            messages.append(user_msg([text_part(value)]))
        elif frm == "assistant":
            calls = _toolace_calls(value)
            if calls:
                msg = assistant_msg("")
                msg["tool_calls"] = calls
                msg["_scaffold"] = True
                pending = [c["id"] for c in calls]
                messages.append(msg)
            else:
                messages.append(assistant_msg(value))
        elif frm == "tool":
            if not pending:
                return None
            messages.append({"role": "tool", "content": value, "tool_call_id": pending.pop(0)})
        else:
            return None
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return Trajectory(
        id=f"toolace-{index}", domain=ctx.domain, origin="prebuilt",
        source=ctx.source(index), messages=messages, tools=tools,
        meta={"simulated": False},
    )


if __name__ == "__main__":
    ctx = Ctx(HERMES_DATASET, "func_calling", "coding", "apache-2.0")
    row = {
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model.\n<tools>\n[{}]\n</tools>"},
            {"from": "human", "value": "Check the front door camera."},
            {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera", "arguments": {"id": "front"}}\n</tool_call>'},
            {"from": "tool", "value": '<tool_response>\n{"status": "ok"}\n</tool_response>'},
            {"from": "gpt", "value": "The front door camera is live."},
        ],
        "tools": json.dumps([{"type": "function", "function": {"name": "get_camera"}}]),
        "category": "security", "task": "tool_call",
    }
    traj = build_hermes(row, 3, ctx)
    print("id:", traj.id, "roles:", [m["role"] for m in traj.messages],
          "calls:", len(traj.messages[2].get("tool_calls") or []))
    print("tool_call_id paired:", traj.messages[3]["tool_call_id"])
    print("system stripped:", "<tools>" not in traj.messages[0]["content"])
    print("no-last-assistant dropped:", build_hermes(
        {"conversations": [{"from": "human", "value": "hi"}]}, 4, ctx))

    actx = Ctx(TOOLACE_DATASET, None, "coding", "apache-2.0")
    trow = {
        "system": 'You are an expert in composing functions. Here is a list of functions:\n'
                  '[{"name": "Timezones", "description": "Get times.", "parameters": {}}]',
        "conversations": [
            {"from": "user", "value": "What time is it in New York and Tokyo?"},
            {"from": "assistant", "value": '[Timezones(timezone="New York"), Timezones(timezone="Tokyo")]'},
            {"from": "tool", "value": '[{"name": "Timezones", "results": {}}]'},
            {"from": "assistant", "value": "New York is 3pm; Tokyo is 4am."},
        ],
    }
    tt = build_toolace(trow, 9, actx)
    print("toolace roles:", [m["role"] for m in tt.messages],
          "calls:", len(tt.messages[2]["tool_calls"]),
          "tool:", tt.messages[2]["tool_calls"][1]["function"]["arguments"])
    print("toolace tools:", len(tt.tools), "no-call row dropped:", build_toolace(
        {"system": "x", "conversations": [{"from": "user", "value": "hi"}]}, 10, actx))
