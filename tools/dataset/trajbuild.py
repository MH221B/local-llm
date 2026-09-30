"""Fetch the prebuilt multi-turn sources and emit prompts/trajectories.jsonl (spec 5.1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from . import canonical, trajparse
from .adapters.base import Ctx

# (dataset, config, split, domain, parser, cap). Caps are prompt-candidate maxima; the
# M2 run samples a subset. spec sections 7.2, 7.3.
SOURCES = [
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "coding", "hermes", 800),
    ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "coding", "hermes", 400),
    ("Team-ACE/ToolACE", None, "train", "coding", "toolace", 800),
    ("HuggingFaceTB/smoltalk", "systemchats-30k", "train", "roleplay", "chat", 400),
    ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "roleplay", "chat", 400),
    ("teknium/OpenHermes-2.5", None, "train", "roleplay", "openhermes", 400),
]

PARSERS = {
    "hermes": trajparse.build_hermes,
    "toolace": trajparse.build_toolace,
    "chat": trajparse.build_chat,
    "openhermes": trajparse.build_openhermes,
}


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
        for index, row in enumerate(stream):
            traj = build(row, index, ctx)
            if traj is None:
                continue
            problems = canonical.validate_trajectory(traj)
            if problems:
                continue
            out.append(traj)
            kept += 1
            if kept >= want:
                break
        print(f"{dataset}:{config} -> {kept} trajectories")
    return out


def run(*, root: Path, limit_per_source: int, dry_run: bool) -> int:
    trajectories = harvest(limit_per_source, dry_run)
    if dry_run:
        return 0
    path = root / "prompts" / "trajectories.jsonl"
    written = canonical.write_jsonl(path, trajectories)
    sim = sum(1 for t in trajectories if t.meta.get("simulated"))
    print(f"trajectory prompts: {written} -> {path} (simulated: {sim})")
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
