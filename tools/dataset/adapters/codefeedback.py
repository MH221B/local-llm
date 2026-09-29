"""m-a-p/CodeFeedback-Filtered-Instruction: query/answer/resource/lang columns."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, plain_pair


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    query, answer = row.get("query"), row.get("answer")
    if not query or not answer:
        return None
    return Example(
        id=f"codefeedback-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(query, answer),
        meta={"lang": row.get("lang"), "resource": row.get("resource")},
    )


if __name__ == "__main__":
    ctx = Ctx("m-a-p/CodeFeedback-Filtered-Instruction", None, "coding", "apache-2.0")
    row = {"query": "Write a reverse function.", "answer": "def rev(s): return s[::-1]",
           "lang": "python", "resource": "codefeedback"}
    ex = build(row, 9, ctx)
    print(ex.id, ex.meta, "|", ex.messages[-1]["content"][:20])
    print("missing answer dropped:", build({"query": "x"}, 0, ctx))
