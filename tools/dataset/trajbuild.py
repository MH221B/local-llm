"""Fetch the prebuilt multi-turn sources and emit prompts/trajectories.jsonl (spec 5.1).

Also emits *seed-only* rows: conversations that end on the assistant's tool call. Spec
section 5.1 wants call -> result -> continue dialogs, so an unresolved final call is not
shippable as a trajectory -- `generate_prebuilt` returns None for it, having no source
observation to splice in. It is still a good seed for the simulated loop, which has the
teacher play agent *and* tool environment and so produces the half the row is missing.
`run_trajectory` skips them; `run_simulated` seeds from them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from . import canonical, trajparse
from .adapters.base import Ctx

# (dataset, config, split, domain, parser, cap). Caps are prompt-candidate maxima, and are
# now spec sections 7.2/7.3's multi-turn values: hermes 3,000 (split across its two configs),
# ToolACE 3,000, and smoltalk's two roleplay configs 1,500 each. M2 sampled a 2,800-row
# subset; `--limit-per-source` caps a run below the table, never above it.
SOURCES = [
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "coding", "hermes", 1500),
    ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "coding", "hermes", 1500),
    ("Team-ACE/ToolACE", None, "train", "coding", "toolace", 3000),
    ("HuggingFaceTB/smoltalk", "systemchats-30k", "train", "roleplay", "chat", 1500),
    ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "roleplay", "chat", 1500),
]

PARSERS = {
    "hermes": trajparse.build_hermes,
    "toolace": trajparse.build_toolace,
    "chat": trajparse.build_chat,
}


def _seed_only(traj: canonical.Trajectory, problems: list[str]) -> bool:
    """True when a row is structurally sound except that a tool call has no result.

    Spec section 5.1 defines the multi-turn tool column as "call -> result -> continue", so
    an unresolved call is not shippable as a trajectory: the student would learn to emit a
    call but not to read its result, which is the half the spec says flattening already
    costs. It is a good *seed* for the simulated loop, which needs only the first user turn
    and a `tools` schema and generates the missing half itself.

    Scoped on purpose: an unresolved call must be the *only* problem, the row must carry a
    schema to call against, and it must have a user turn to seed from. Measured on the two
    hermes configs: 1,671 builds are rejected this way and *nothing* is rejected for any
    other reason, so the tight scope costs nothing. Both shapes occur -- 1,090 end on the
    call, 581 carry it mid-conversation -- and both are equally usable as seeds, because
    `run_simulated` reads only the first user turn and `tools`.
    """
    return (bool(problems)
            and all("has no tool result" in p for p in problems)
            and bool(traj.tools)
            and any(m.get("role") == "user" for m in traj.messages))


def harvest(limit_per_source: int, dry_run: bool = False) -> list[canonical.Trajectory]:
    out: list[canonical.Trajectory] = []
    for dataset, config, split, domain, parser, cap in SOURCES:
        want = min(cap, limit_per_source)
        if dry_run:
            print(f"[dry] {dataset}:{config} domain={domain} parser={parser} take={want}")
            continue
        ctx = Ctx(dataset, config, domain, None)
        build = PARSERS[parser]
        stream = load_dataset(dataset, config, split=split, streaming=True)
        kept = 0
        seed_only = 0
        for index, row in enumerate(stream):
            traj = build(row, index, ctx)
            if traj is None:
                continue
            problems = canonical.validate_trajectory(traj)
            if problems:
                if not _seed_only(traj, problems):
                    continue
                # A seed does not count against `want`. The cap is a promise about
                # *shippable* trajectories, and letting seeds share it silently displaced
                # valid rows -- ToolACE fell from 2,763 shipped to 1,471 when they did.
                traj.meta["seed_only"] = True
                out.append(traj)
                seed_only += 1
                continue
            out.append(traj)
            kept += 1
            if kept >= want:
                break
        print(f"{dataset}:{config} -> {kept} trajectories ({seed_only} seed-only)")
    return out


def run(*, root: Path, limit_per_source: int, dry_run: bool) -> int:
    trajectories = harvest(limit_per_source, dry_run)
    if dry_run:
        return 0
    path = root / "prompts" / "trajectories.jsonl"
    written = canonical.write_jsonl(path, trajectories)
    sim = sum(1 for t in trajectories if t.meta.get("simulated"))
    seed_only = sum(1 for t in trajectories if t.meta.get("seed_only"))
    tools = sum(1 for t in trajectories if t.tools)
    print(f"trajectory prompts: {written} -> {path} (simulated: {sim}, "
          f"tool-carrying: {tools}, seed-only: {seed_only})")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prebuilt trajectory prompt corpus")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--limit-per-source", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(root=args.root, limit_per_source=args.limit_per_source, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
