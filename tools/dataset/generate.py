"""M2 driver: teacher generation and verification over every M1 prompt column.

`all` runs magpie -> seeded (single-turn) -> trajectory (prebuilt) -> simulated -> merge.
Only verified, filtered rows are written; drops and pass rates are counted per source and
per domain so M3 can re-anchor its caps on measured numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import canonical, filters, gencache, split, textutil, verify
from .canonical import Example, iter_jsonl, prompt_from_dict, trajectory_from_dict
from .gencache import GenCache
from .imgstore import ImageStore
from .report import merge_manifest, write_manifest
from .teacher import TeacherClient, TeacherError
from . import magpie, trajectory

TRAIN_POOLS = ["prompts/train.jsonl", "verification/seeds.jsonl", "prompts/magpie.jsonl"]
VAL_POOLS = ["prompts/val.jsonl"]


def _seed(text: str) -> int:
    return int(text[:8], 16)


def _source_key(record) -> str:
    return f"{record.source.get('name', '?')}|{record.domain}"


def generate_one(prompt, client, cache, *, store, thinking: bool) -> dict:
    """One cached single-turn completion, verified and filtered. Cached value is the verdict."""
    key = gencache.prefix_key(prompt.messages, kind=f"single:{prompt.id}")
    cached = cache.get(key)
    if cached is not None:
        return cached
    msg = client.complete(prompt.messages, store=store, tools=prompt.tools,
                          thinking=thinking, seed=_seed(key))
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    if prompt.verify:
        ok, why = verify.check_record(prompt.verify, content)
        if not ok:
            cached = {"accepted": False, "reason": "verify_failed", "detail": why}
            cache.put(key, cached)
            return cached
    assistant = {"role": "assistant", "content": content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    tokens = textutil.count_tokens(content) + (
        textutil.count_tokens(json.dumps(tool_calls, ensure_ascii=False)) if tool_calls else 0)
    ex = Example(id=prompt.id, domain=prompt.domain, origin="teacher",
                 source=prompt.source, messages=prompt.messages + [assistant],
                 images=prompt.images, tools=prompt.tools,
                 tokens=tokens, meta=dict(prompt.meta))
    reason = filters.example_reject_reason(ex)
    cached = {"accepted": reason is None, "reason": reason, "example": ex.to_dict()}
    cache.put(key, cached)
    return cached


def _run_pool(pool: Path, limit: int | None, client, cache, *, store, thinking, stats):
    records = [prompt_from_dict(r) for r in iter_jsonl(pool)]
    if limit:
        records = split.downsample(records, limit)
    out = []
    for prompt in records:
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            verdict = generate_one(prompt, client, cache, store=store, thinking=thinking)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}")
            continue
        if verdict["accepted"]:
            out.append(canonical.example_from_dict(verdict["example"]))
            stats["by_domain"][prompt.domain] += 1
            stats["accepted"] += 1
            stats["accepted_by_source"][key] += 1
        else:
            stats["drops"][verdict["reason"]] += 1
    return out


def run_seeded(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
               thinking: bool = True):
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    train: list[Example] = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            train.extend(_run_pool(path, limit, client, cache, store=store,
                                   thinking=thinking, stats=stats))
    val: list[Example] = []
    for rel in VAL_POOLS:
        path = root / rel
        if path.exists():
            val.extend(_run_pool(path, val_limit, client, cache, store=store,
                                 thinking=thinking, stats=stats))
    canonical.write_jsonl(root / "m2" / "single.jsonl", train)
    canonical.write_jsonl(root / "m2" / "val.jsonl", val)
    _dump_stats(root / "m2" / "seeded.stats.json", stats)
    print(f"seeded: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def _accept_trajectory(traj, stats) -> bool:
    """Validate, verify, and filter one trajectory; record the drop reason."""
    problems = canonical.validate_trajectory(traj)
    if problems:
        stats["drops"]["invalid"] += 1
        return False
    ok, _ = verify.check_trajectory(traj.verify, traj)
    if not ok:
        stats["drops"]["verify_failed"] += 1
        return False
    reason = filters.trajectory_reject_reason(traj)
    if reason:
        stats["drops"][reason.split(":")[-1].strip()] += 1
        return False
    return True


def _traj_stats() -> dict:
    return {"attempted": 0, "accepted": 0, "simulated": 0, "by_domain": Counter(),
            "attempted_by_source": Counter(), "accepted_by_source": Counter(),
            "drops": Counter()}


def run_trajectory(*, root: Path, client, cache, limit: int | None, thinking: bool = True):
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    records = [trajectory_from_dict(r) for r in iter_jsonl(path)] if path.exists() else []
    if limit:
        records = split.downsample(records, limit)
    out = []
    for prompt in records:
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            traj = trajectory.generate_prebuilt(prompt, client, cache, store=store,
                                                thinking=thinking)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}")
            continue
        if traj is None:
            stats["drops"]["turn_structure"] += 1
            continue
        if not _accept_trajectory(traj, stats):
            continue
        out.append(traj)
        stats["accepted"] += 1
        stats["by_domain"][prompt.domain] += 1
        stats["accepted_by_source"][key] += 1
    canonical.write_jsonl(root / "m2" / "trajectory.jsonl", out)
    _dump_stats(root / "m2" / "trajectory.stats.json", stats)
    print(f"trajectories: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def run_simulated(*, root: Path, client, cache, limit: int | None, thinking: bool = True):
    """Simulated trajectories (spec section 5.1): the teacher plays user, agent, and tool
    environment, seeded from the real first user turns and tool schemas of the prebuilt
    trajectory prompts. Rows carry `meta.simulated = true`."""
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    seeds = [t for t in (trajectory_from_dict(r) for r in iter_jsonl(path))
             if t.tools] if path.exists() else []
    if limit:
        seeds = split.downsample(seeds, limit)
    out = []
    for seed in seeds:
        first_user = next((canonical.message_text(m) for m in seed.messages
                           if m.get("role") == "user"), "")
        if not first_user:
            continue
        key = f"teacher:simulated|{seed.domain}"
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            traj = trajectory.generate_simulated(
                client, cache, id=f"sim-{seed.id}", domain=seed.domain,
                source={"name": "teacher:simulated"}, tools=seed.tools,
                first_user=first_user, thinking=thinking, store=store)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {seed.id}: {exc}")
            continue
        if traj is None:
            stats["drops"]["turn_structure"] += 1
            continue
        if not _accept_trajectory(traj, stats):
            continue
        out.append(traj)
        stats["accepted"] += 1
        stats["by_domain"][seed.domain] += 1
        stats["accepted_by_source"][key] += 1
        stats["simulated"] += 1
    canonical.write_jsonl(root / "m2" / "simulated.jsonl", out)
    _dump_stats(root / "m2" / "simulated.stats.json", stats)
    print(f"simulated: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def _dump_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    att = serial.get("attempted_by_source", {})
    acc = serial.get("accepted_by_source", {})
    serial["pass_rate_by_source"] = {
        k: round(acc.get(k, 0) / n, 4) for k, n in att.items() if n}
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def run_merge(*, root: Path) -> dict:
    # Training rows are Examples: `example_from_dict` is the right loader for all three
    # shards, including the trajectory ones. A `Trajectory`'s `verify` spec is a
    # generation-time filter (`check_trajectory` already ran in `_accept_trajectory`) and
    # is deliberately not carried into the written row — `Example.to_dict` has no
    # `verify` field. If a downstream stage ever needs to re-run an oracle, it must read
    # the originating prompt, not train.jsonl.
    def load(rel):
        p = root / "m2" / rel
        return [canonical.example_from_dict(r) for r in iter_jsonl(p)] if p.exists() else []

    train = load("single.jsonl") + load("trajectory.jsonl") + load("simulated.jsonl")
    val = load("val.jsonl")
    canonical.write_jsonl(root / "train.jsonl", train)
    canonical.write_jsonl(root / "val.jsonl", val)

    def stats(rel):
        p = root / "m2" / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    m2 = {"seeded": stats("seeded.stats.json"), "trajectory": stats("trajectory.stats.json"),
          "simulated": stats("simulated.stats.json"),
          "train_written": len(train), "val_written": len(val)}
    shards = ("seeded", "trajectory", "simulated")
    passed = sum(m2[k].get("accepted", 0) for k in shards)
    attempted = sum(m2[k].get("attempted", 0) for k in shards)
    m2["pass_rate"] = round(passed / attempted, 4) if attempted else 0.0
    manifest = merge_manifest(root / "manifest.json", train=train, val=val, m2=m2)
    write_manifest(root / "manifest.json", manifest)
    print(f"merged: train {len(train)}, val {len(val)}, pass_rate {m2['pass_rate']}")
    return m2


def run_all(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
            multi_limit: int | None, sim_limit: int | None, magpie_limit: int,
            thinking: bool, dry_run: bool):
    if dry_run:
        print("[dry] magpie:", magpie_limit)
        print("[dry] trajectories:", multi_limit)
        print("[dry] simulated:", sim_limit)
        print("[dry] single-turn:", limit, "val:", val_limit)
        return
    magpie.run(root=root, client=client, cache=cache, limit=magpie_limit, dry_run=False)
    run_seeded(root=root, client=client, cache=cache, limit=limit, val_limit=val_limit,
               thinking=thinking)
    run_trajectory(root=root, client=client, cache=cache, limit=multi_limit, thinking=thinking)
    run_simulated(root=root, client=client, cache=cache, limit=sim_limit, thinking=thinking)
    run_merge(root=root)


def smoke() -> int:
    """Offline end-to-end: one prompt, a fake teacher, a temp root. No network."""
    import tempfile

    from .canonical import Prompt, validate

    tmp = Path(tempfile.mkdtemp())
    (tmp / "prompts").mkdir(parents=True, exist_ok=True)
    canonical.write_jsonl(tmp / "prompts" / "train.jsonl", [Prompt(
        id="p1", domain="reasoning", origin="prebuilt", source={"name": "smoke"},
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}])])

    class FakeTeacher:
        def complete(self, messages, **kw):
            return {"content": "Four is the sum of two and two. Working it through: two plus "
                               "two equals four, so the answer is four."}

    cache = GenCache(tmp / "m2" / "cache.jsonl")
    stats = run_seeded(root=tmp, client=FakeTeacher(), cache=cache, limit=10,
                       val_limit=0, thinking=True)
    kept = canonical.iter_jsonl(tmp / "m2" / "single.jsonl")
    first = next(kept)
    print("smoke accepted:", stats["accepted"],
          "| valid:", validate(canonical.example_from_dict(first)) == [])
    return 0 if stats["accepted"] == 1 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 teacher generation")
    ap.add_argument("--mode", choices=["all", "merge"], default="all")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8086")
    ap.add_argument("--model", default="local-teacher")
    ap.add_argument("--limit", type=int, default=300,
                    help="single-turn attempts PER POOL (three pools: "
                         "prompts/train.jsonl, verification/seeds.jsonl, prompts/magpie.jsonl)")
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--multi-limit", type=int, default=100)
    ap.add_argument("--sim-limit", type=int, default=40)
    ap.add_argument("--magpie-limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # Spec section 10 fixes enable_thinking: true; the non-thinking roles (Magpie, the
    # simulated user, the simulated tool environment) hardcode thinking=False internally.
    cache = GenCache(args.root / "m2" / "cache.jsonl")
    client = TeacherClient(args.base_url, model=args.model)
    if args.dry_run:
        run_all(root=args.root, client=client, cache=cache, limit=args.limit,
                val_limit=args.val_limit, multi_limit=args.multi_limit,
                sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
                thinking=True, dry_run=True)
        return 0
    if args.mode == "merge":
        run_merge(root=args.root)
        return 0
    run_all(root=args.root, client=client, cache=cache, limit=args.limit,
            val_limit=args.val_limit, multi_limit=args.multi_limit,
            sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
            thinking=True, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(smoke() if len(sys.argv) == 1 else main())
