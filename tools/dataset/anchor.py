"""Re-anchor M1's prompt caps on the pass rates M2 measured (spec section 10.1).

A source that passes at rate `p` needs about `1/p` times as many candidates to deliver the
same number of accepted examples. This module reads the per-source pass rates out of
`manifest.json`, aggregates them per domain, and prints the cap table to paste into
`sources.py`. It never edits a file: the table is a proposal and the edit is a review step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from . import sources
from .sources import DOMAIN_SHARE, SOURCES, TEACHER_SOURCES, Source


def per_domain_pass_rate(stats: dict) -> tuple[dict[str, float], dict[str, int]]:
    """Attempt-weighted pass rate per domain, plus the attempts it was computed from.

    Weighting by attempts is what makes the estimate usable: an unweighted mean over sources
    would let a 20-attempt source outvote a 3,000-attempt one. Returning the attempt counts
    alongside the rates is what lets `main` refuse to anchor on a sample too small to mean
    anything — M2's manifest holds 12 single-turn attempts in total.

    Single-turn shards only. Spec section 10.1: "Trajectory pass rates are reported
    separately from single-turn pass rates, since a trajectory passes only if every one of
    its turns passes" — folding them together would feed a multi-turn number into a
    single-turn pool's size.
    """
    accepted: Counter = Counter()
    attempted: Counter = Counter()
    for shard in ("seeded",):
        block = stats.get(shard) or {}
        for key, n in (block.get("attempted_by_source") or {}).items():
            domain = key.split("|")[-1]
            attempted[domain] += n
        for key, n in (block.get("accepted_by_source") or {}).items():
            domain = key.split("|")[-1]
            accepted[domain] += n
    rates = {d: (accepted[d] / attempted[d]) for d in attempted if attempted[d]}
    return rates, dict(attempted)


MIN_ANCHOR_ATTEMPTS = 200


def per_domain_caps_by_domain(rows: list[tuple[Source, int]]) -> dict[str, int]:
    """Anchored cap total per domain, which is the number `allocate` cannot see."""
    out: dict[str, int] = {}
    for source, cap in rows:
        out[source.domain] = out.get(source.domain, 0) + cap
    return out


def candidates_for(accepted_target: int, pass_rate: float) -> int:
    """Spec section 10.1: ceil(accepted / p). A zero pass rate cannot be anchored."""
    if pass_rate <= 0:
        raise ValueError("cannot anchor a source whose pass rate is 0")
    return math.ceil(accepted_target / pass_rate)


def anchored_caps(target_by_domain: dict[str, int], rates: dict[str, float],
                  table: list[Source]) -> list[tuple[Source, int]]:
    """`(source, anchored_cap)` for every source, so callers keep the pairing.

    Returned as `(Source, int)` pairs rather than parallel lists: the rows come back grouped
    by domain, so any caller that zips them against a differently-ordered sequence is
    silently wrong.
    """
    by_domain: dict[str, list[Source]] = {}
    for s in table:
        by_domain.setdefault(s.domain, []).append(s)
    out: list[tuple[Source, int]] = []
    for domain, group in by_domain.items():
        rate = rates.get(domain)
        target = target_by_domain.get(domain, 0)
        if rate is None or not target:
            out.extend((s, s.cap) for s in group)
            continue
        need = candidates_for(target, rate)
        total_cap = sum(s.cap for s in group)
        for s in group:
            # Integer ceiling: `(a + b - 1) // b`, so a cap is never rounded down by float
            # division. A cap must never shrink, hence the max() against the spec's value.
            scaled = (need * s.cap + total_cap - 1) // total_cap
            out.append((s, max(s.cap, scaled)))
    return out


def driver_flags(rows: list[tuple[Source, int]], val_target: int,
                 val_rate: float) -> dict[str, int]:
    """The driver flags the anchored table implies, so the steps stay tied together.

    `--target-train` cannot simply be `sum(anchored caps)`. `sources.allocate` splits the
    budget by the *fixed* `DOMAIN_SHARE` (30/30/20/20) and uses each cap only as that
    source's share *within* its domain, so scaling four domains' caps by four different
    factors cannot be expressed as one total — a domain whose cap grew would be under-served
    by the fixed share. Verified: anchoring then re-running `allocate` moves every quota by
    at most one row.

    The operative value is therefore the smallest total whose 30/30/20/20 split gives every
    domain at least what the anchor says it needs: `max_d(ceil(need_d / share_d))`. That is
    also the only form a rebuild can act on, since the pipeline takes one `--target-train`.

    `--val-size` and `--val-limit` are the same value on purpose: `--val-size` is how many
    val *candidates* the pipeline must build, `--val-limit` is how many of them the driver
    answers. Both are sized against the pass rate, because the val set is filtered too.
    """
    candidates = candidates_for(val_target, val_rate)
    needs = per_domain_caps_by_domain(rows)
    target = max((math.ceil(need / share) for domain, need in needs.items()
                  if (share := DOMAIN_SHARE.get(domain, 0)) > 0), default=sum(needs.values()))
    return {
        "target_train": target,
        "limit": target,
        "val_size": candidates,
        "val_limit": candidates,
        "difficulty_limit": target,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="re-anchor caps on M2's measured pass rates")
    ap.add_argument("--manifest", type=Path,
                    default=Path("datasets/qwen35-4b-sft/manifest.json"))
    ap.add_argument("--target-total", type=int, default=25_000,
                    help="accepted examples across all domains (spec section 4)")
    args = ap.parse_args(argv)

    stats = json.loads(args.manifest.read_text(encoding="utf-8")).get("m2") or {}
    rates, attempts = per_domain_pass_rate(stats)
    targets = {d: round(args.target_total * share)
               for d, share in sources.DOMAIN_SHARE.items()}
    print("measured pass rates:", {d: round(r, 4) for d, r in sorted(rates.items())})
    print("attempts behind them:", dict(sorted(attempts.items())))
    print("accepted targets:  ", targets)

    thin = {d: n for d, n in attempts.items() if n < MIN_ANCHOR_ATTEMPTS}
    if thin:
        raise SystemExit(
            f"refusing to anchor: {thin} have fewer than {MIN_ANCHOR_ATTEMPTS} attempts. "
            "Keep the spec section 7 caps (Task 6 Step 3 is a recorded no-op) and re-anchor "
            "after Task 8 phase 1 has measured a few hundred rows per domain. Measured on "
            "M2's manifest: 12 single-turn attempts, individual sources at 1-3 rows, and the "
            "pessimistic corner of those intervals implies a 128,000-candidate pool.")

    # One call over both tables. `coding` is fed by prebuilt sources *and* by
    # `teacher:tools`, so anchoring each table separately would apply that domain's
    # shortfall twice and roughly double coding's candidate budget. Combining them lets a
    # domain's requirement be split across every source that serves it.
    rows = anchored_caps(targets, rates, list(SOURCES) + list(TEACHER_SOURCES))
    print(f"{'dataset':<52} {'domain':<11} {'cap':>6} {'anchored':>9}")
    before = 0
    for source, anchored in rows:
        before += source.cap
        mark = "  <- change" if anchored != source.cap else ""
        print(f"{source.dataset:<52} {source.domain:<11} {source.cap:>6} "
              f"{anchored:>9}{mark}")
    after = sum(cap for _, cap in rows)
    print(f"\nnew candidate pool: {after} (was {before})")
    print("per-domain anchored caps:", dict(sorted(per_domain_caps_by_domain(rows).items())))
    print("  spec section 7's caps imply 32.1 / 33.3 / 19.2 / 15.4 percent, not 30/30/20/20,")
    print("  and `allocate` splits by the fixed DOMAIN_SHARE — so a domain's anchored total is")
    print("  only realised when `--target-train` >= need_d / share_d, which is what `flags`")
    print("  below computes.")
    print("spec section 7 target: 39,000 candidates for 25,000 accepted")

    # The val split is not anchored by the cap table — it is the same pass rate applied to a
    # 1,000-example target — and an unanchored `--val-limit` is the easiest way to end up
    # with a val set far below spec section 12. Weighted by attempts, not by the mean of the
    # four domain rates: the unweighted mean is the exact error `per_domain_pass_rate`'s
    # docstring warns about, and it sizes the val set off a 2-attempt domain.
    total_attempts = sum(attempts.values()) or 1
    weighted = sum(rates[d] * attempts[d] for d in rates) / total_attempts
    flags = driver_flags(rows, val_target=1000, val_rate=weighted)
    print("\nflags for Task 7 and Task 8 (from the table above):")
    print(f"  --target-train {flags['target_train']}")
    print(f"  --limit        {flags['limit']}")
    print(f"  --val-size     {flags['val_size']}   (the candidate count the pipeline must build)")
    print(f"  --val-limit    {flags['val_limit']}   (1,000 accepted at a "
          f"{weighted:.2f} blended rate)")
    print(f"  uncensored supply ceiling: 4,370 candidates "
          f"(1,939 seeds the pipeline keeps, plus at most one invention each)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())

    smoke_rates = {"reasoning": 0.70, "coding": 0.55, "roleplay": 0.80, "uncensored": 0.62}
    smoke_targets = {"reasoning": 7500, "coding": 7500, "roleplay": 5000, "uncensored": 5000}
    print("candidates_for(7500, 0.70):", candidates_for(7500, 0.70), "(expect 10715)")
    rows = anchored_caps(smoke_targets, smoke_rates,
                         list(SOURCES) + list(TEACHER_SOURCES))
    teacher_names = {s.dataset for s in TEACHER_SOURCES}
    by_domain: dict[str, int] = {}
    for source, cap in rows:
        by_domain[source.domain] = by_domain.get(source.domain, 0) + cap
    print("TEACHER_SOURCES anchored caps:",
          [cap for source, cap in rows if source.dataset in teacher_names],
          "(expect [1049, 8065])")
    print("no source shrinks:", all(cap >= source.cap for source, cap in rows))
    print("pool:", sum(cap for _, cap in rows), "(expect 41703)")
    print("meets every domain target:", all(by_domain.get(d, 0) >= t
                                            for d, t in smoke_targets.items()))
    print("per-domain anchored:", dict(sorted(by_domain.items())))
    print("driver flags:", driver_flags(rows, val_target=1000, val_rate=0.65))
