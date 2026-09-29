"""Adapter protocol and helpers shared by every source module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..canonical import Example


@dataclass
class Ctx:
    """Per-source run context handed to an adapter's build()."""
    name: str
    config: str | None
    domain: str
    license: str | None = None

    def source(self, row_index: int | None = None) -> dict:
        s = {"name": self.name, "license": self.license}
        if self.config is not None:
            s["config"] = self.config
        if row_index is not None:
            s["row"] = row_index
        return s


@dataclass(frozen=True)
class Adapter:
    """One entry per (dataset, config). `build` returns None to drop the row."""
    key: str
    build: Callable[[dict, int, Ctx], Example | None]


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def user_msg(parts: list[dict]) -> dict:
    return {"role": "user", "content": parts}


def assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def system_msg(text: str) -> dict:
    return {"role": "system", "content": text}


_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
             "user": "user", "assistant": "assistant"}

_MARKUP = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>")


def clean_markup(text: str) -> str:
    for token in _MARKUP:
        text = text.replace(token, "")
    return text.strip()


def sharegpt_turns(conversations: list, *, system: str | None = None) -> list[dict] | None:
    """Convert ShareGPT turns to canonical messages. Accepts from/value or role/content."""
    messages: list[dict] = []
    if system and system.strip():
        messages.append(system_msg(system.strip()))
    for turn in conversations:
        raw_role = turn.get("from") or turn.get("role")
        text = turn.get("value")
        if text is None:
            text = turn.get("content")
        role = _ROLE_MAP.get(raw_role)
        if role is None or not isinstance(text, str):
            return None
        messages.append({"role": role, "content": [text_part(clean_markup(text))]})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages


def plain_pair(question: str, answer: str) -> list[dict]:
    return [user_msg([text_part(question.strip())]), assistant_msg(answer.strip())]
