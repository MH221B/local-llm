"""HuggingFaceM4/the_cauldron configs. Images ride on prompts; the VQA builder keeps reference answers for the holdout only."""
from __future__ import annotations

import io
from typing import Any

from ..canonical import Example
from ..imgstore import image_part
from .base import Ctx, assistant_msg, user_msg

# Prompt configs: long-form description tasks whose prompts carry images.
GENERATION_CONFIGS = ("chart2text", "diagram_image_to_text", "tabmwp", "screen2words")

# Holdout configs: short reference answers are kept for the section 8 vision probe only.
VQA_CONFIGS = ("chartqa", "ai2d", "tqa", "scienceqa")


def _as_bytes(image: Any) -> bytes | None:
    if isinstance(image, dict):
        if image.get("bytes"):
            return image["bytes"]
        path = image.get("path")
        if path and isinstance(path, (bytes, bytearray)):
            return bytes(path)
        return None
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "save"):  # a PIL image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    return None


def _text_pair(texts: list) -> tuple[str, str] | None:
    """Cauldron `texts` are single-element lists of {user, assistant, source} structs.

    Plain [prompt, response] lists are accepted too, so hand-built fixtures stay usable.
    Values must be strings: str()-coercing a dict or a number would write a Python repr
    into the prompt, which no filter can tell apart from real content.
    """
    if not texts:
        return None
    first = texts[0]
    if isinstance(first, dict):
        user, assistant = first.get("user"), first.get("assistant")
    else:
        user = texts[0]
        assistant = texts[1] if len(texts) > 1 else None
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    prompt, response = user.strip(), assistant.strip()
    if not prompt or not response:
        return None
    return prompt, response


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    if ctx.config not in GENERATION_CONFIGS:
        return None
    pair = _text_pair(row.get("texts") or [])
    if pair is None:
        return None
    prompt, response = pair

    parts: list[dict] = [{"type": "text", "text": prompt}]
    for image in row.get("images") or []:
        data = _as_bytes(image)
        if data is None:
            return None
        parts.append(image_part(data))

    return Example(
        id=f"cauldron-{ctx.config}-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=[user_msg(parts), assistant_msg(response)],
    )


def build_vqa(row: dict, index: int, ctx: Ctx) -> Example | None:
    """Holdout builder for the section 8 vision probe. Used by Task 17, never by the pipeline."""
    if ctx.config not in VQA_CONFIGS:
        return None
    pair = _text_pair(row.get("texts") or [])
    if pair is None:
        return None
    question, answer = pair

    parts: list[dict] = [{"type": "text", "text": question}]
    for image in row.get("images") or []:
        data = _as_bytes(image)
        if data is None:
            return None
        parts.append(image_part(data))

    return Example(
        id=f"visionholdout-{ctx.config}-{index}",
        domain="reasoning",
        origin="prebuilt",
        source=ctx.source(index),
        messages=[user_msg(parts), assistant_msg(answer)],
    )


if __name__ == "__main__":
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (200, 30, 30)).save(buf, format="PNG")
    payload = buf.getvalue()

    ctx = Ctx("HuggingFaceM4/the_cauldron", "chart2text", "reasoning", "apache-2.0")
    row = {"images": [{"bytes": payload}],
           "texts": [{"user": "Describe the chart.", "assistant": "It rises steadily.",
                      "source": "chart2text"}]}
    ex = build(row, 4, ctx)
    print(ex.id, "parts:", [p["type"] for p in ex.messages[0]["content"]])
    print("vqa config skipped:", build(row, 5, Ctx("HuggingFaceM4/the_cauldron", "chartqa", "reasoning")))
    print("single-text skipped:", build({"images": [{"bytes": payload}], "texts": ["only"]}, 6, ctx))
