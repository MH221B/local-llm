"""Canonical records. Adapters emit Example; the M1 pipeline persists Prompt (spec section 5)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

DOMAINS = ("reasoning", "coding", "roleplay", "uncensored")
ORIGINS = ("prebuilt", "teacher")
PART_TYPES = ("text", "image")
ROLES = ("system", "user", "assistant")

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
