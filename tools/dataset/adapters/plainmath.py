"""NuminaMath-1.5 and gsm8k. Plain columns plus the validity flags (spec section 7.1)."""
from __future__ import annotations

import re

from ..canonical import Example
from .base import Ctx, plain_pair

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_FINAL = re.compile(r"####\s*([^\n]+)")


def _gsm8k_gold(answer: str) -> str | None:
    """gsm8k solutions end with the final answer after `####`."""
    finals = _HASH_FINAL.findall(answer)
    return finals[-1].strip() if finals else None


def _numina_gold(solution: str) -> str | None:
    """NuminaMath solutions put the final answer in the last \\boxed{}.

    Measured against the live dataset: NuminaMath-1.5's `solution` is competition-maths
    prose that states its answer inline ("Answer: 0 or 3", "the answer is no") and does
    **not** use `\\boxed{}` — 0 of 2,830 valid rows sampled carried one. So this extractor
    attaches almost no golds in practice; gsm8k below is what actually supplies them.
    Kept because the cost is nil and a future Numina config may ship boxed answers.
    """
    boxed = _BOXED.findall(solution)
    return boxed[-1].strip() if boxed else None


def build_numina(row: dict, index: int, ctx: Ctx) -> Example | None:
    # NuminaMath-1.5 ships these as strings ("Yes" / "Incomplete" / "No"), not bools.
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return None
    problem, solution = row.get("problem"), row.get("solution")
    if not problem or not solution:
        return None
    meta = {"problem_type": row.get("problem_type"), "synthetic": row.get("synthetic")}
    gold = _numina_gold(solution)
    if gold:
        meta["verify"] = {"type": "answer_match", "gold": gold}
    return Example(
        id=f"numina-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(problem, solution),
        meta=meta,
    )


def build_gsm8k(row: dict, index: int, ctx: Ctx) -> Example | None:
    question, answer = row.get("question"), row.get("answer")
    if not question or not answer:
        return None
    meta: dict = {}
    gold = _gsm8k_gold(answer)
    if gold:
        meta["verify"] = {"type": "answer_match", "gold": gold}
    return Example(
        id=f"gsm8k-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(question, answer),
        meta=meta,
    )


if __name__ == "__main__":
    ctx = Ctx("AI-MO/NuminaMath-1.5", None, "reasoning", "apache-2.0")
    good = {"problem": "Solve x.", "solution": "Thus \\boxed{1}.", "problem_is_valid": "Yes",
            "solution_is_valid": "Yes", "problem_type": "algebra", "synthetic": False}
    bad = {"problem": "Solve y.", "solution": "y = 2", "problem_is_valid": "Incomplete",
           "solution_is_valid": "Yes"}
    ex = build_numina(good, 5, ctx)
    print(ex.id, ex.meta["verify"])
    print("invalid dropped:", build_numina(bad, 6, ctx))

    gctx = Ctx("openai/gsm8k", "main", "reasoning", "mit")
    g = build_gsm8k({"question": "1+1?", "answer": "Sum is 1+1.\n#### 2"}, 1, gctx)
    print(g.id, g.meta["verify"], "| tail:", g.messages[-1]["content"][-6:])
