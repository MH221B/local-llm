"""NuminaMath-1.5 and gsm8k. Plain columns plus the validity flags (spec section 7.1)."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, plain_pair


def build_numina(row: dict, index: int, ctx: Ctx) -> Example | None:
    # NuminaMath-1.5 ships these as strings ("Yes" / "Incomplete" / "No"), not bools.
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return None
    problem, solution = row.get("problem"), row.get("solution")
    if not problem or not solution:
        return None
    return Example(
        id=f"numina-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(problem, solution),
        meta={"problem_type": row.get("problem_type"), "synthetic": row.get("synthetic")},
    )


def build_gsm8k(row: dict, index: int, ctx: Ctx) -> Example | None:
    question, answer = row.get("question"), row.get("answer")
    if not question or not answer:
        return None
    return Example(
        id=f"gsm8k-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(question, answer),
    )


if __name__ == "__main__":
    ctx = Ctx("AI-MO/NuminaMath-1.5", None, "reasoning", "apache-2.0")
    good = {"problem": "Solve x.", "solution": "x = 1", "problem_is_valid": "Yes",
            "solution_is_valid": "Yes", "problem_type": "algebra", "synthetic": False}
    bad = {"problem": "Solve y.", "solution": "y = 2", "problem_is_valid": "Incomplete",
           "solution_is_valid": "Yes"}
    print(build_numina(good, 5, ctx).id, build_numina(good, 5, ctx).meta)
    print("invalid dropped:", build_numina(bad, 6, ctx))

    gctx = Ctx("openai/gsm8k", "main", "reasoning", "mit")
    print(build_gsm8k({"question": "1+1?", "answer": "#### 2"}, 1, gctx).messages[-1]["content"])
