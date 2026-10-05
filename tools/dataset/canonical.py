"""Canonical records. Adapters emit Example; the M1 pipeline persists Prompt (spec section 5)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

DOMAINS = ("reasoning", "coding", "roleplay", "uncensored")
ORIGINS = ("prebuilt", "teacher")
PART_TYPES = ("text", "image")
ROLES = ("system", "user", "assistant", "tool")

# Opening pleasantries with no request in them. Compared lowercased with trailing
# "!" / "." stripped. See _is_bare_greeting for why this exists.
_BARE_GREETINGS = frozenset({
    "hi", "hi there", "hello", "hello there", "hey", "hey there",
    "good morning", "good afternoon", "good evening", "how are you",
    "how are you doing", "how's it going", "greetings", "yo",
})


@dataclass
class ImageRef:
    sha256: str
    path: str
    w: int
    h: int
    origin_url: str | None = None

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "path": self.path, "w": self.w,
                "h": self.h, "origin_url": self.origin_url}


@dataclass
class Example:
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    images: list[ImageRef] = field(default_factory=list)
    tools: Any = None
    tokens: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "origin": self.origin,
            "source": self.source,
            "messages": self.messages,
            "tools": self.tools,
            "images": [i.to_dict() for i in self.images],
            "tokens": self.tokens,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Prompt:
    """A user-side record: leading system turn + first user turn, images and tools attached.

    M2 stems the teacher completion onto `messages` (same content-part shape), runs the
    attached `verify` spec, and only then may the row become training data.
    """
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    images: list[ImageRef] = field(default_factory=list)
    tools: Any = None
    tokens: int = 0
    verify: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "origin": self.origin,
            "source": self.source,
            "messages": self.messages,
            "tools": self.tools,
            "images": [i.to_dict() for i in self.images],
            "tokens": self.tokens,
            "verify": self.verify,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Trajectory:
    """A multi-turn record: system + user/assistant/tool turns, ending on an assistant turn.

    M2 stores the whole conversation. Prebuilt tool results are observations and are kept;
    every assistant turn is teacher-written before the row can be trained (spec section 5.1).
    """
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    tools: Any = None
    images: list[ImageRef] = field(default_factory=list)
    tokens: int = 0
    verify: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "origin": self.origin,
            "source": self.source,
            "messages": self.messages,
            "tools": self.tools,
            "images": [i.to_dict() for i in self.images],
            "tokens": self.tokens,
            "verify": self.verify,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _is_bare_greeting(m: dict) -> bool:
    """True for an opener that carries no request, e.g. smoltalk's "Hi there".

    everyday-conversations opens with a greeting handshake before the topic; keeping
    that turn as the prompt yields a 1-2 token record that prompt_too_short rejects,
    so the source would contribute nothing. Checked against the dataset: every one of
    270 sampled rows opened with one of the five strings below and all had a later
    user turn. Text-only and short by construction, so a real request never matches.
    """
    content = m.get("content")
    if isinstance(content, list):
        if any(p.get("type") != "text" for p in content):
            return False
        text = "".join(p.get("text", "") for p in content)
    elif isinstance(content, str):
        text = content
    else:
        return False
    return text.strip().lower().rstrip("!.") in _BARE_GREETINGS


def strip_to_prompt(ex: Example) -> Prompt | None:
    """Reduce an adapter Example to its prompt: leading system + first user turn.

    Every prebuilt assistant turn is discarded here, before anything is written: the
    source datasets supply prompts only, and M2's teacher supplies every response.
    A leading greeting turn is dropped when a later user turn exists, so the prompt
    carries an actual request. Returns None when the row has no usable user turn.
    """
    messages = list(ex.messages)
    # Drop a leading greeting handshake (greeting, assistant reply, greeting, ...) so
    # the scan starts at the first turn that carries a request. A leading system turn
    # is preserved, so the greeting is found at index 1 in that case.
    while True:
        i = 1 if messages and messages[0].get("role") == "system" else 0
        if len(messages) > i + 1 and messages[i].get("role") == "user" \
                and _is_bare_greeting(messages[i]) \
                and any(n.get("role") == "user" for n in messages[i + 1:]):
            del messages[i]
            if i < len(messages) and messages[i].get("role") == "assistant":
                del messages[i]
        else:
            break

    kept: list[dict] = []
    for m in messages:
        if m.get("role") == "system" and not kept:
            kept.append(m)
            continue
        if m.get("role") == "user":
            kept.append(m)
            used = {p["sha256"] for msg in kept
                    for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                    if p.get("type") == "image" and "sha256" in p}
            return Prompt(
                id=ex.id, domain=ex.domain, origin=ex.origin, source=ex.source,
                messages=kept, images=[i for i in ex.images if i.sha256 in used],
                tools=ex.tools, tokens=ex.tokens, verify=ex.meta.get("verify"), meta=ex.meta)
        return None
    return None


def image_shas(ex: Example | Prompt) -> list[str]:
    """sha256 of every image part in the example or prompt, in order."""
    out: list[str] = []
    for m in ex.messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image" and "sha256" in part:
                    out.append(part["sha256"])
    return out


def validate(ex: Example) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    if not ex.id:
        problems.append("empty id")
    if ex.domain not in DOMAINS:
        problems.append(f"bad domain: {ex.domain!r}")
    if ex.origin not in ORIGINS:
        problems.append(f"bad origin: {ex.origin!r}")
    if not isinstance(ex.source, dict) or "name" not in ex.source:
        problems.append("source missing name")
    if not ex.messages:
        problems.append("no messages")
        return problems

    roles = [m.get("role") for m in ex.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "assistant":
        problems.append("last message is not assistant")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")

    refs = {i.sha256 for i in ex.images}
    for m in ex.messages:
        content = m.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            problems.append("content is neither str nor list")
            continue
        for part in content:
            ptype = part.get("type")
            if ptype not in PART_TYPES:
                problems.append(f"bad part type: {ptype!r}")
            if ptype == "image" and part.get("sha256") not in refs:
                problems.append(f"image part {part.get('sha256')!r} has no ImageRef")

    if ex.tools is not None and not isinstance(ex.tools, list):
        problems.append("tools is neither list nor null")
    if ex.tokens < 0:
        problems.append("negative token count")
    return problems


def validate_prompt(p: Prompt) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    if not p.id:
        problems.append("empty id")
    if p.domain not in DOMAINS:
        problems.append(f"bad domain: {p.domain!r}")
    if p.origin not in ORIGINS:
        problems.append(f"bad origin: {p.origin!r}")
    if not isinstance(p.source, dict) or "name" not in p.source:
        problems.append("source missing name")
    if not p.messages:
        problems.append("no messages")
        return problems

    roles = [m.get("role") for m in p.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "user":
        problems.append("last message is not user")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")
    if "assistant" in roles:
        problems.append("prebuilt assistant turn was not stripped")

    refs = {i.sha256 for i in p.images}
    for m in p.messages:
        content = m.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            problems.append("content is neither str nor list")
            continue
        for part in content:
            ptype = part.get("type")
            if ptype not in PART_TYPES:
                problems.append(f"bad part type: {ptype!r}")
            if ptype == "image" and part.get("sha256") not in refs:
                problems.append(f"image part {part.get('sha256')!r} has no ImageRef")

    if p.tools is not None and not isinstance(p.tools, list):
        problems.append("tools is neither list nor null")
    if p.tokens < 0:
        problems.append("negative token count")
    if p.verify is not None and not isinstance(p.verify, dict):
        problems.append("verify is neither dict nor null")
    return problems


def write_jsonl(path: Path, records: Iterable[Example | Prompt]) -> int:
    count = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# Sources carrying an embedded JSON tool schema (ToolACE via ShareGPT) spell the schema
# with Python-ish type names -- `dict`, `int`, `float` -- which are not valid JSON Schema.
# llama.cpp converts `tools` to a grammar and rejects the whole request otherwise; measured
# `HTTP 500 {"message":"JSON schema error at #: unrecognized type dict"}` on every ToolACE
# trajectory row. Coerce at the parse boundary so the canonical record holds the OpenAI form
# the plan's example uses (`"type": "object"`).
_SCHEMA_TYPE_ALIASES = {
    "dict": "object", "int": "integer", "float": "number", "str": "string",
    "text": "string", "bool": "boolean", "list": "array", "tuple": "array", "set": "array",
}


def normalize_tool_schema(schema: Any) -> Any:
    """Recursively map non-standard JSON Schema `type` names to canonical ones.

    Returns a new structure (the input is not mutated). An `array` that declares no `items`
    gets a permissive `{}` so the grammar still builds.
    """
    if isinstance(schema, list):
        return [normalize_tool_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: normalize_tool_schema(v) for k, v in schema.items()}
    t = out.get("type")
    if isinstance(t, str) and t in _SCHEMA_TYPE_ALIASES:
        out["type"] = _SCHEMA_TYPE_ALIASES[t]
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {}
    return out


def normalize_tools(tools: Any) -> Any:
    """Normalize every OpenAI tool entry's `function.parameters` schema (spec section 5.1)."""
    if not tools:
        return tools
    out = []
    for entry in tools:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and "parameters" in fn:
            entry = {**entry, "function": {**fn,
                     "parameters": normalize_tool_schema(fn["parameters"])}}
        out.append(entry)
    return out


def tool_calls_of(message: dict) -> list[dict]:
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def tool_call_arguments(call: dict) -> dict:
    """A tool call's arguments as a mapping, for template rendering (spec sections 5.1, 11.2).

    Canonical records and the OpenAI wire format both store `arguments` as a JSON string,
    but the student template iterates it as a mapping (`tool_call.arguments|items`). This
    is the single place that converts between the two, so the schema stays a string.
    Returns `{}` when the arguments are absent or unparseable rather than raising; a
    malformed call is caught by `validate_trajectory`, not silently mis-rendered.
    """
    args = (call.get("function") or {}).get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def for_template(messages: list[dict]) -> list[dict]:
    """A render-time view: tool-call `arguments` become mappings, everything else is copied.

    The student template reads `tool_call.arguments|items`, which requires a mapping; the
    stored form is a JSON string. Convert here rather than mutating the record so the
    schema and the written JSONL keep the OpenAI string form (spec section 5.1).
    """
    out: list[dict] = []
    for m in messages:
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        rendered = []
        for call in calls:
            fn = dict(call.get("function") or {})
            fn["arguments"] = tool_call_arguments(call)
            rendered.append({**call, "function": fn})
        out.append({**m, "tool_calls": rendered})
    return out


def declared_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    for entry in tools or []:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def final_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return message_text(m)
    return ""


def validate_trajectory(t: Trajectory) -> list[str]:
    """Return a list of problems. Empty list means valid (spec sections 5.1, 11.3)."""
    problems: list[str] = []
    if not t.id:
        problems.append("empty id")
    if t.domain not in DOMAINS:
        problems.append(f"bad domain: {t.domain!r}")
    if t.origin not in ORIGINS:
        problems.append(f"bad origin: {t.origin!r}")
    if not isinstance(t.source, dict) or "name" not in t.source:
        problems.append("source missing name")
    if not t.messages:
        return problems + ["no messages"]

    roles = [m.get("role") for m in t.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "assistant":
        problems.append("last message is not assistant")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")

    calls: dict[str, int] = {}
    answered: set[str] = set()
    for i, m in enumerate(t.messages):
        if m.get("role") == "assistant":
            for call in tool_calls_of(m):
                cid = call.get("id")
                if not cid or call.get("type") != "function":
                    problems.append(f"bad tool_call at message {i}")
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    problems.append(f"tool_call without function name at message {i}")
                calls[cid] = i
                args = (fn or {}).get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        problems.append(f"tool_call arguments are not JSON at message {i}")
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if not cid:
                problems.append(f"tool message without tool_call_id at message {i}")
            elif cid not in calls:
                problems.append(f"tool result {cid!r} has no preceding call")
            else:
                answered.add(cid)
    for cid, idx in calls.items():
        if cid not in answered:
            problems.append(f"call {cid!r} at message {idx} has no tool result")

    refs = {i.sha256 for i in t.images}
    for m in t.messages:
        content = m.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            problems.append("content is neither str nor list")
            continue
        for part in content:
            ptype = part.get("type")
            if ptype not in PART_TYPES:
                problems.append(f"bad part type: {ptype!r}")
            if ptype == "image" and part.get("sha256") not in refs:
                problems.append(f"image part {part.get('sha256')!r} has no ImageRef")

    if t.tools is not None and not isinstance(t.tools, list):
        problems.append("tools is neither list nor null")
    if t.tokens < 0:
        problems.append("negative token count")
    if t.verify is not None and not isinstance(t.verify, dict):
        problems.append("verify is neither dict nor null")
    return problems


def _images_from(raw: list) -> list[ImageRef]:
    return [ImageRef(**{k: d.get(k) for k in ("sha256", "path", "w", "h", "origin_url")})
            for d in raw or []]


def prompt_from_dict(row: dict) -> Prompt:
    return Prompt(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        verify=row.get("verify"), meta=row.get("meta") or {})


def trajectory_from_dict(row: dict) -> Trajectory:
    return Trajectory(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        verify=row.get("verify"), meta=row.get("meta") or {})


def example_from_dict(row: dict) -> Example:
    return Example(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        meta=row.get("meta") or {})


if __name__ == "__main__":
    good = Example(
        id="smoke-1", domain="reasoning", origin="prebuilt",
        source={"name": "smoke", "row": 0, "license": None},
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                  {"role": "assistant", "content": "hello"}],
    )
    bad = Example(
        id="smoke-2", domain="nope", origin="prebuilt",
        source={"name": "smoke"},
        messages=[{"role": "user", "content": "hi"}],
    )
    prompt = strip_to_prompt(good)
    print("good:", validate(good))
    print("bad:", len(validate(bad)), "problems")
    print("prompt:", validate_prompt(prompt), [m["role"] for m in prompt.messages])

    # ToolACE spells schema types Python-ish (`dict`/`int`/`float`); llama.cpp 500s on them.
    _schema = {"type": "dict",
               "properties": {"n": {"type": "int"}, "x": {"type": "float"}},
               "required": ["n"]}
    _norm = normalize_tool_schema(_schema)
    print("normalize types:", _norm["type"], _norm["properties"]["n"]["type"],
          _norm["properties"]["x"]["type"])
    print("normalize leaves input untouched:", _schema["type"] == "dict")
    print("normalize array gets items:", normalize_tool_schema({"type": "list"}))
    print("normalize tools entry:",
          normalize_tools([{"type": "function", "function": {
              "name": "f", "parameters": {"type": "dict"}}}]))

    ok_traj = Trajectory(
        id="traj-1", domain="coding", origin="teacher", source={"name": "smoke"},
        messages=[
            {"role": "user", "content": "Book a table for two."},
            {"role": "assistant", "content": "Checking.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "book", "arguments": "{\"n\": 2}"}}]},
            {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Booked — T-991."},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}],
        tokens=42,
    )
    bad_traj = Trajectory(
        id="traj-2", domain="coding", origin="teacher", source={"name": "smoke"},
        messages=[
            {"role": "assistant", "content": "x",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "book", "arguments": "{bad"}}]},
            {"role": "tool", "content": "r", "tool_call_id": "call_9"},
        ],
    )
    print("traj ok:", validate_trajectory(ok_traj),
          [m["role"] for m in ok_traj.messages])
    print("traj bad:", len(validate_trajectory(bad_traj)), "problems")

    call = ok_traj.messages[1]["tool_calls"][0]
    print("args as mapping:", tool_call_arguments(call))
    print("args unparseable:", tool_call_arguments(
        {"function": {"name": "book", "arguments": "{bad"}}))
    rendered = for_template(ok_traj.messages)
    print("for_template converts:", isinstance(
        rendered[1]["tool_calls"][0]["function"]["arguments"], dict),
        "| record unchanged:", isinstance(call["function"]["arguments"], str))
