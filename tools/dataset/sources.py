"""The frozen source table (spec section 7) and the M1 prompt quota allocator."""
from __future__ import annotations

from dataclasses import dataclass

# Domain shares of the final mix (spec section 4).
DOMAIN_SHARE = {"reasoning": 0.30, "coding": 0.30, "roleplay": 0.20, "uncensored": 0.20}
TOTAL_CANDIDATES = 39_000


@dataclass(frozen=True)
class Source:
    dataset: str
    config: str | None
    split: str
    domain: str
    cap: int
    adapter: str
    teacher: bool = False          # True -> M2 territory, excluded from M1
    anchored_on: float | None = None   # M2 pass rate this cap was scaled by (spec 10.1)


# M1 prompt sources: every row supplies a user prompt only. Prebuilt answers are
# discarded centrally (canonical.strip_to_prompt); spec sections 7.1-7.3, 7.6.
#
# Caps are still spec section 7's values, not anchored ones. Task 6's re-anchor (spec 10.1)
# refused on M2's manifest: 12 single-turn attempts, every domain below the 200-attempt
# floor, so `--target-train 39000 --val-size 1067` stands until Task 8 phase 1 measures a
# few hundred rows per domain. `python -m tools.dataset.anchor` prints the table it would
# propose and never edits this file.
SOURCES: list[Source] = [
    Source("AI-MO/NuminaMath-1.5", None, "train", "reasoning", 8000, "numina"),
    Source("openai/gsm8k", "main", "train", "reasoning", 1000, "gsm8k"),
    Source("HuggingFaceTB/smoltalk", "metamathqa-50k", "train", "reasoning", 500, "smoltalk"),
    Source("HuggingFaceM4/the_cauldron", "chart2text", "train", "reasoning", 1500, "cauldron"),
    # diagram_image_to_text ships only 300 rows; the cap is a maximum, not a promise.
    Source("HuggingFaceM4/the_cauldron", "diagram_image_to_text", "train", "reasoning", 1000, "cauldron"),
    Source("HuggingFaceM4/the_cauldron", "tabmwp", "train", "reasoning", 500, "cauldron"),
    Source("HuggingFaceM4/the_cauldron", "screen2words", "train", "coding", 1000, "cauldron"),
    Source("m-a-p/CodeFeedback-Filtered-Instruction", None, "train", "coding", 5000, "codefeedback"),
    Source("NousResearch/hermes-function-calling-v1", "func_calling", "train", "coding", 1500, "toolcalls"),
    Source("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "coding", 1500, "toolcalls"),
    Source("Team-ACE/ToolACE", None, "train", "coding", 3000, "sharegpt_toolace"),
    Source("teknium/OpenHermes-2.5", None, "train", "roleplay", 2500, "sharegpt_openhermes"),
    Source("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "systemchats-30k", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "longalign", "train", "roleplay", 500, "smoltalk"),
]

# M2 teacher rows, counted so the pool arithmetic stays honest and visible.
TEACHER_SOURCES: list[Source] = [
    Source("teacher:tools", None, "-", "coding", 1000, "teacher", teacher=True),
    Source("teacher:uncensored", None, "-", "uncensored", 6000, "teacher", teacher=True),
]


def per_domain_caps(sources: list[Source]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for s in sources:
        caps[s.domain] = caps.get(s.domain, 0) + s.cap
    return caps


def allocate(target_train: int, targets: list[Source]) -> dict[tuple[str, str | None], int]:
    """Split `target_train` across sources in proportion to their caps within each domain."""
    domain_budget = {d: round(target_train * share) for d, share in DOMAIN_SHARE.items()}
    by_domain: dict[str, list[Source]] = {}
    for s in targets:
        by_domain.setdefault(s.domain, []).append(s)

    quotas: dict[tuple[str, str | None], int] = {}
    for domain, sources in by_domain.items():
        budget = domain_budget.get(domain, 0)
        total_cap = sum(s.cap for s in sources)
        for s in sources:
            quotas[(s.dataset, s.config)] = int(budget * s.cap / total_cap) if total_cap else 0
    return quotas


if __name__ == "__main__":
    prebuilt_caps = per_domain_caps(SOURCES)
    teacher_caps = per_domain_caps(TEACHER_SOURCES)
    m1_total = sum(prebuilt_caps.values())
    pool = sum(prebuilt_caps.get(d, 0) + teacher_caps.get(d, 0) for d in DOMAIN_SHARE)
    print("M1 prebuilt caps:", prebuilt_caps, "=", m1_total)
    print("teacher caps:", teacher_caps)
    print("full pool:", pool, "(spec target 39000, M3 minimum)")
    anchored = {s.dataset: s.anchored_on for s in SOURCES + TEACHER_SOURCES
                if s.anchored_on}
    by_domain: dict[str, int] = {}
    for s in SOURCES + TEACHER_SOURCES:
        by_domain[s.domain] = by_domain.get(s.domain, 0) + s.cap
    print("per-domain caps:", dict(sorted(by_domain.items())))
    print("anchored sources:", anchored)
    # The sum can never fall below the original caps — every row is `max(s.cap, scaled)` —
    # so this assert alone proves nothing. The check that bites is per-domain: `allocate`
    # splits the budget by the fixed DOMAIN_SHARE, so `--target-train` must be at least
    # `max_d(need_d / share_d)` (anchor.py prints it) or a domain whose cap grew is still
    # under-served.
    assert pool >= 39_000, f"re-anchored pool {pool} is below the 39,000 candidate budget"
    q = allocate(3000, SOURCES)
    print("M1 quota total:", sum(q.values()),
          "(uncensored prompts come from seeds.py; downsample renormalises)")
    print("first three:", list(q.items())[:3])
