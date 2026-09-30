"""Multi-turn teacher generation: prebuilt-backed and simulated (spec sections 5.1, 10.2).

`_scaffold` assistant turns coming from trajparse are never sent to the teacher; they only
tell the walk how many assistant turns the source had. Every generated turn is teacher text.
"""
from __future__ import annotations

import json

from . import gencache, textutil
from .canonical import Trajectory, final_assistant_text, message_text
from .teacher import TeacherClient

MAX_TOOL_TURNS = 8
MAX_USER_TURNS = 8
SIM_USER_SYSTEM = (
    "You are the user in a tool-use conversation. Write the user's next message: a natural, "
    "specific request that one of the available tools can satisfy. Reply with only the message.")
SIM_TOOL_SYSTEM = (
    "You are the tool environment. Given a tool call, reply with only a short JSON object a "
    "plausible API would return. No prose and no code fences.")


def _gen(client, cache, key: str, messages: list[dict], **kw) -> dict:
    """One cached teacher turn. Cached value: {'completion': {...}}."""
    cached = cache.get(key)
    if cached is None:
        msg = client.complete(messages, **kw)
        cached = {"completion": {"content": msg.get("content") or "",
                                 "tool_calls": msg.get("tool_calls") or []}}
        cache.put(key, cached)
    return cached["completion"]


def _remap(calls: list[dict], traj_id: str, start: int) -> list[dict]:
    out = []
    for k, tc in enumerate(calls):
        fn = tc.get("function") or {}
        out.append({"id": f"{traj_id}-call-{start + k}", "type": "function",
                    "function": {"name": fn.get("name", ""),
                                 "arguments": fn.get("arguments", "{}")}})
    return out


def _truncate(messages: list[dict], max_user: int, max_tool: int) -> list[dict]:
    """Cut at the first turn that breaks a bound, then back up to an assistant turn."""
    users = tools = 0
    cut = len(messages)
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            users += 1
        elif m.get("role") == "tool":
            tools += 1
        if users > max_user or tools > max_tool:
            cut = i
            break
    trimmed = [dict(m) for m in messages[:cut]]
    while trimmed and trimmed[-1].get("role") != "assistant":
        trimmed.pop()
    return trimmed


def _source_run(messages: list[dict], i: int) -> list[dict]:
    """The source `tool` messages immediately following index `i`."""
    out, j = [], i + 1
    while j < len(messages) and messages[j].get("role") == "tool":
        out.append(messages[j])
        j += 1
    return out


def generate_prebuilt(prompt_traj: Trajectory, client: TeacherClient, cache,
                      *, store=None, thinking: bool = True) -> Trajectory | None:
    """Regenerate every assistant turn; splice the source's observations after teacher calls."""
    source = _truncate(prompt_traj.messages, MAX_USER_TURNS, MAX_TOOL_TURNS)
    new: list[dict] = []
    calls_made = 0
    i = 0
    while i < len(source):
        m = source[i]
        if m.get("role") != "assistant":
            new.append({"role": m["role"], "content": m.get("content", ""),
                        **({"tool_call_id": m["tool_call_id"]} if m.get("tool_call_id") else {})})
            i += 1
            continue

        key = gencache.prefix_key(new, kind=f"prebuilt:{prompt_traj.id}")
        comp = _gen(client, cache, key, new, store=store, tools=prompt_traj.tools,
                    thinking=thinking)
        calls = comp.get("tool_calls") or []
        run = _source_run(source, i)
        assistant = {"role": "assistant", "content": comp["content"]}
        if calls:
            if not run:
                return None  # a call with no observation to splice in
            assistant["tool_calls"] = _remap(calls, prompt_traj.id, calls_made)
            calls_made += len(assistant["tool_calls"])
            new.append(assistant)
            for call, src_tool in zip(assistant["tool_calls"], run):
                new.append({"role": "tool", "content": src_tool.get("content", ""),
                            "tool_call_id": call["id"]})
        else:
            new.append(assistant)
        i += 1 + len(run)

    while new and new[-1].get("role") == "tool":
        new.pop()
    if not new or new[-1].get("role") != "assistant":
        return None
    text = "\n".join(message_text(m) for m in new if m.get("role") == "assistant")
    return Trajectory(
        id=prompt_traj.id, domain=prompt_traj.domain, origin="teacher",
        source=dict(prompt_traj.source), messages=new, tools=prompt_traj.tools,
        images=prompt_traj.images, tokens=textutil.count_tokens(text),
        verify=prompt_traj.verify, meta={**prompt_traj.meta, "simulated": False})


if __name__ == "__main__":
    from .gencache import GenCache
    from .canonical import validate_trajectory
    import tempfile
    from pathlib import Path

    class FakeTeacher:
        def __init__(self, script):
            self.script = list(script)
            self.sent = []

        def complete(self, messages, **kw):
            self.sent.append(messages)
            return self.script.pop(0)

    src = Trajectory(
        id="toolace-1", domain="coding", origin="prebuilt", source={"name": "smoke"},
        messages=[
            {"role": "user", "content": "What time is it in Tokyo?"},
            {"role": "assistant", "content": "", "_scaffold": True,
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": "clock", "arguments": "{\"city\": \"Tokyo\"}"}}]},
            {"role": "tool", "content": '{"time": "04:00"}', "tool_call_id": "call_0"},
            {"role": "assistant", "content": "", "_scaffold": True},
        ],
        tools=[{"type": "function", "function": {"name": "clock"}}])

    fake = FakeTeacher([
        {"content": "<think>look up the time</think>",
         "tool_calls": [{"id": "x", "type": "function",
                         "function": {"name": "clock", "arguments": "{\"city\": \"Tokyo\"}"}}]},
        {"content": "It is 04:00 in Tokyo.", "tool_calls": []},
    ])
    cache = GenCache(Path(tempfile.mkdtemp()) / "c.jsonl")
    out = generate_prebuilt(src, fake, cache)
    print("roles:", [m["role"] for m in out.messages])
    print("tool id:", out.messages[2]["tool_call_id"], "| problems:", validate_trajectory(out))
    print("teacher never saw the scaffold:", all(
        not any(m.get("_scaffold") for m in msgs) for msgs in fake.sent))

    # Render the generated trajectory through the student template. This is the check that
    # catches the string-vs-mapping `arguments` mismatch: the template iterates
    # `tool_call.arguments|items`, so a stored JSON string raises a TemplateError here.
    # Skips cleanly when the template has not been pinned yet.
    template_path = Path(__file__).parent / "student.jinja"
    if template_path.exists():
        from . import canonical
        from .render import render
        try:
            text = render(template_path.read_text(encoding="utf-8"),
                          canonical.for_template(out.messages))
            print("renders through student.jinja: True |",
                  "<function=clock>" in text)
        except Exception as exc:
            print("renders through student.jinja: False |", type(exc).__name__, exc)
    else:
        print("renders through student.jinja: skipped (no student.jinja)")
