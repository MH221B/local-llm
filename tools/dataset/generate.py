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


def _map(fn, items, concurrency: int):
    """Order-preserving map. `concurrency <= 1` runs inline, so the M2 path is unchanged."""
    if concurrency <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


def _progress(rel: str, done: int, total: int, stats: dict, every: int = 100) -> None:
    """A heartbeat, so a multi-hour run is observable rather than a silent process.

    M2's full run was unobservable for hours at a time; that is the defect this fixes.
    """
    if done % every and done != total:
        return
    rate = (stats["accepted"] / done) if done else 0.0
    print(f"  {rel}: {done}/{total} attempted, {stats['accepted']} accepted "
          f"(pass {rate:.2f})", flush=True)


def _source_key(record) -> str:
    return f"{record.source.get('name', '?')}|{record.domain}"


def _record_reject(root: Path, *, stage: str, reason: str, detail=None,
                   pool: str | None = None, id: str | None = None,
                   record: dict | None = None) -> None:
    """Persist a rejected item, with its completion, to `m2/rejected.jsonl`.

    Drops used to be counted only, and `verify_failed` kept its reason but not the
    completion, so a false-positive oracle verdict was indistinguishable from a bad
    generation (measured: three MBPP seeds and one GSM8K row were dropped wrongly and
    could not be inspected afterwards). Appended per row so a crash keeps what ran.
    """
    path = root / "m2" / "rejected.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"stage": stage, "reason": reason, "detail": detail, "pool": pool,
            "id": id, "record": record}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _completion_example(prompt, content: str, tool_calls: list) -> Example:
    assistant = {"role": "assistant", "content": content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    tokens = textutil.count_tokens(content) + (
        textutil.count_tokens(json.dumps(tool_calls, ensure_ascii=False)) if tool_calls else 0)
    return Example(id=prompt.id, domain=prompt.domain, origin="teacher",
                   source=prompt.source, messages=prompt.messages + [assistant],
                   images=prompt.images, tools=prompt.tools,
                   tokens=tokens, meta=dict(prompt.meta))


def generate_one(prompt, client, cache, *, store, thinking: bool) -> dict:
    """One cached single-turn completion, verified and filtered. Cached value is the verdict.

    Every verdict carries its `example`, `verify_failed` included: a drop whose completion
    is not stored cannot be audited.
    """
    key = gencache.prefix_key(prompt.messages, kind=f"single:{prompt.id}")
    cached = cache.get(key)
    if cached is not None:
        return cached
    msg = client.complete(prompt.messages, store=store, tools=prompt.tools,
                          thinking=thinking, seed=_seed(key))
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    ex = _completion_example(prompt, content, tool_calls)
    if prompt.verify:
        ok, why = verify.check_record(prompt.verify, content)
        if not ok:
            cached = {"accepted": False, "reason": "verify_failed", "detail": why,
                      "example": ex.to_dict()}
            cache.put(key, cached)
            return cached
    reason = filters.example_reject_reason(ex)
    cached = {"accepted": reason is None, "reason": reason, "example": ex.to_dict()}
    cache.put(key, cached)
    return cached


def _run_pool(pool: Path, limit: int | None, client, cache, *, root, store, thinking, stats,
              concurrency: int = 1):
    records = [prompt_from_dict(r) for r in iter_jsonl(pool)]
    if limit is not None:
        records = split.downsample(records, limit)

    def work(prompt):
        try:
            return generate_one(prompt, client, cache, store=store, thinking=thinking), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (prompt, (verdict, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
            _record_reject(root, stage="seeded", reason="teacher_error", detail=str(exc),
                           pool=pool.name, id=prompt.id)
        elif verdict["accepted"]:
            out.append(canonical.example_from_dict(verdict["example"]))
            stats["by_domain"][prompt.domain] += 1
            stats["accepted"] += 1
            stats["accepted_by_source"][key] += 1
        else:
            stats["drops"][verdict["reason"]] += 1
            _record_reject(root, stage="seeded", reason=verdict["reason"],
                           detail=verdict.get("detail"), pool=pool.name, id=prompt.id,
                           record=verdict.get("example"))
        _progress(pool.name, done, len(records), stats)
    return out


def run_seeded(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
               thinking: bool = True, concurrency: int = 1):
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    train: list[Example] = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            train.extend(_run_pool(path, limit, client, cache, root=root, store=store,
                                   thinking=thinking, stats=stats, concurrency=concurrency))
    val: list[Example] = []
    for rel in VAL_POOLS:
        path = root / rel
        if path.exists():
            val.extend(_run_pool(path, val_limit, client, cache, root=root, store=store,
                                 thinking=thinking, stats=stats, concurrency=concurrency))
    canonical.write_jsonl(root / "m2" / "single.jsonl", train)
    canonical.write_jsonl(root / "m2" / "val.jsonl", val)
    _dump_stats(root / "m2" / "seeded.stats.json", stats)
    print(f"seeded: accepted {stats['accepted']} of {stats['attempted']}", flush=True)
    return stats


def _trajectory_reject(traj) -> tuple[str | None, str | None]:
    """Validate, verify, and filter one trajectory. Returns `(reason, detail)` or `(None, None)`.

    Separate from the accept path so a dropped trajectory can be written to
    `m2/rejected.jsonl` with its reason instead of only being counted.
    """
    problems = canonical.validate_trajectory(traj)
    if problems:
        return "invalid", problems[0]
    ok, why = verify.check_trajectory(traj.verify, traj)
    if not ok:
        return "verify_failed", why
    reason = filters.trajectory_reject_reason(traj)
    if reason:
        return reason.split(":")[-1].strip(), reason
    return None, None


def _traj_stats() -> dict:
    return {"attempted": 0, "accepted": 0, "simulated": 0, "by_domain": Counter(),
            "attempted_by_source": Counter(), "accepted_by_source": Counter(),
            "drops": Counter()}


def run_trajectory(*, root: Path, client, cache, limit: int | None, thinking: bool = True,
                   concurrency: int = 1):
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    records = [trajectory_from_dict(r) for r in iter_jsonl(path)] if path.exists() else []
    if limit is not None:
        records = split.downsample(records, limit)

    def work(prompt):
        try:
            return trajectory.generate_prebuilt(prompt, client, cache, store=store,
                                                thinking=thinking), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (prompt, (traj, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
            _record_reject(root, stage="trajectory", reason="teacher_error", detail=str(exc),
                           pool="prompts/trajectories.jsonl", id=prompt.id)
        elif traj is None:
            stats["drops"]["turn_structure"] += 1
            _record_reject(root, stage="trajectory", reason="turn_structure",
                           pool="prompts/trajectories.jsonl", id=prompt.id)
        else:
            # Unchanged from M2: the reject reason and its detail come from
            # `_trajectory_reject`, not from a boolean helper.
            reason, detail = _trajectory_reject(traj)
            if reason:
                stats["drops"][reason] += 1
                _record_reject(root, stage="trajectory", reason=reason, detail=detail,
                               pool="prompts/trajectories.jsonl", id=prompt.id,
                               record=traj.to_dict())
            else:
                out.append(traj)
                stats["accepted"] += 1
                stats["by_domain"][prompt.domain] += 1
                stats["accepted_by_source"][key] += 1
        _progress("trajectories", done, len(records), stats, every=25)
    canonical.write_jsonl(root / "m2" / "trajectory.jsonl", out)
    _dump_stats(root / "m2" / "trajectory.stats.json", stats)
    print(f"trajectories: accepted {stats['accepted']} of {stats['attempted']}", flush=True)
    return stats


def run_simulated(*, root: Path, client, cache, limit: int | None, thinking: bool = True,
                  concurrency: int = 1):
    """Simulated trajectories (spec section 5.1): the teacher plays user, agent, and tool
    environment, seeded from the real first user turns and tool schemas of the prebuilt
    trajectory prompts. Rows carry `meta.simulated = true`."""
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    seeds = [t for t in (trajectory_from_dict(r) for r in iter_jsonl(path))
             if t.tools] if path.exists() else []
    if limit is not None:
        seeds = split.downsample(seeds, limit)
    # A seed with no first user turn is skipped here, in the main thread, so it never
    # reaches `work`: a seed's shape is a prompt property, not a teacher result, and this
    # keeps the skip out of the concurrent path.
    prepared = []
    for seed in seeds:
        first_user = next((canonical.message_text(m) for m in seed.messages
                           if m.get("role") == "user"), "")
        if first_user:
            prepared.append((seed, first_user))

    def work(item):
        seed, first_user = item
        try:
            return trajectory.generate_simulated(
                client, cache, id=f"sim-{seed.id}", domain=seed.domain,
                source={"name": "teacher:simulated"}, tools=seed.tools,
                first_user=first_user, thinking=thinking, store=store), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, ((seed, _first_user), (traj, exc)) in enumerate(zip(
            prepared, _map(work, prepared, concurrency)), start=1):
        key = f"teacher:simulated|{seed.domain}"
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {seed.id}: {exc}", flush=True)
            _record_reject(root, stage="simulated", reason="teacher_error", detail=str(exc),
                           pool="prompts/trajectories.jsonl", id=f"sim-{seed.id}")
        elif traj is None:
            stats["drops"]["turn_structure"] += 1
            _record_reject(root, stage="simulated", reason="turn_structure",
                           pool="prompts/trajectories.jsonl", id=f"sim-{seed.id}")
        else:
            reason, detail = _trajectory_reject(traj)
            if reason:
                stats["drops"][reason] += 1
                _record_reject(root, stage="simulated", reason=reason, detail=detail,
                               pool="prompts/trajectories.jsonl", id=f"sim-{seed.id}",
                               record=traj.to_dict())
            else:
                out.append(traj)
                stats["accepted"] += 1
                stats["by_domain"][seed.domain] += 1
                stats["accepted_by_source"][key] += 1
                stats["simulated"] += 1
        _progress("simulated", done, len(prepared), stats, every=25)
    canonical.write_jsonl(root / "m2" / "simulated.jsonl", out)
    _dump_stats(root / "m2" / "simulated.stats.json", stats)
    print(f"simulated: accepted {stats['accepted']} of {stats['attempted']}", flush=True)
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
    # generation-time filter (`check_trajectory` already ran in `_trajectory_reject`) and
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
            thinking: bool, dry_run: bool, concurrency: int = 1, schema_limit: int = 0):
    if dry_run:
        print("[dry] magpie:", magpie_limit)
        print("[dry] trajectories:", multi_limit)
        print("[dry] simulated:", sim_limit)
        print("[dry] single-turn:", limit, "val:", val_limit)
        return
    magpie.run(root=root, client=client, cache=cache, limit=magpie_limit, dry_run=False,
               schema_limit=schema_limit)
    run_seeded(root=root, client=client, cache=cache, limit=limit, val_limit=val_limit,
               thinking=thinking, concurrency=concurrency)
    run_trajectory(root=root, client=client, cache=cache, limit=multi_limit, thinking=thinking,
                   concurrency=concurrency)
    run_simulated(root=root, client=client, cache=cache, limit=sim_limit, thinking=thinking,
                  concurrency=concurrency)
    run_merge(root=root)


def smoke() -> int:
    """Offline end-to-end: one prompt, a fake teacher, a temp root. No network."""
    import tempfile

    from .canonical import Prompt, validate

    tmp = Path(tempfile.mkdtemp())
    (tmp / "prompts").mkdir(parents=True, exist_ok=True)
    canonical.write_jsonl(tmp / "prompts" / "train.jsonl", [
        Prompt(id="p1", domain="reasoning", origin="prebuilt", source={"name": "smoke"},
               messages=[{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]),
        # The fake teacher never emits digits, so this one fails `answer_match` and must
        # still land in rejected.jsonl with its completion.
        Prompt(id="p2", domain="reasoning", origin="prebuilt", source={"name": "smoke"},
               messages=[{"role": "user", "content": [{"type": "text", "text": "What is 2+3?"}]}],
               verify={"type": "answer_match", "gold": "5"}),
    ])

    class FakeTeacher:
        def complete(self, messages, **kw):
            return {"content": "Four is the sum of two and two. Working it through: two plus "
                               "two equals four, so the answer is four."}

    cache = GenCache(tmp / "m2" / "cache.jsonl")
    stats = run_seeded(root=tmp, client=FakeTeacher(), cache=cache, limit=10,
                       val_limit=0, thinking=True)
    kept = canonical.iter_jsonl(tmp / "m2" / "single.jsonl")
    first = next(kept)
    rejected = list(canonical.iter_jsonl(tmp / "m2" / "rejected.jsonl"))
    has_completion = bool(rejected and rejected[0].get("record"))
    print("smoke accepted:", stats["accepted"],
          "| valid:", validate(canonical.example_from_dict(first)) == [])
    print("smoke rejected:", len(rejected),
          "| reason:", rejected[0]["reason"] if rejected else None,
          "| completion kept:", has_completion)
    ok = (stats["accepted"] == 1 and len(rejected) == 1
          and rejected[0]["reason"] == "verify_failed" and has_completion)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    # Line-buffered, so a `Tee-Object` or a redirected log shows progress as it happens
    # rather than in 8 KiB blocks at the end.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):        # not a TextIOWrapper (e.g. under pytest)
        pass
    ap = argparse.ArgumentParser(description="M2 teacher generation")
    ap.add_argument("--mode", choices=["all", "merge"], default="all")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8086")
    ap.add_argument("--model", default="local-teacher")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel teacher requests; >1 is what makes batching pay")
    ap.add_argument("--api-key", default=None, help="Bearer token for the teacher endpoint")
    ap.add_argument("--cache", type=Path, default=None,
                    help="resume cache path; defaults to <root>/m2/cache.jsonl")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds per teacher call; a timeout is terminal, not retried")
    ap.add_argument("--retries", type=int, default=3)
    # These four used to default to a number and were guarded by truthiness, so `0` meant
    # *everything*. `None` now means "every candidate" and `0` means "none", which is what
    # the whole-pool statements elsewhere in this plan depend on.
    ap.add_argument("--limit", type=int, default=None,
                    help="per-pool candidate cap; 0 answers none in the pool, omitted "
                         "answers every candidate")
    ap.add_argument("--val-limit", type=int, default=None, help="same, for the val pool")
    ap.add_argument("--multi-limit", type=int, default=None,
                    help="same, for the prebuilt trajectory pool")
    ap.add_argument("--sim-limit", type=int, default=None,
                    help="same, for the simulated pool (prebuilt seeds + schema seeds)")
    ap.add_argument("--magpie-limit", type=int, default=0,
                    help="Magpie inventions; 0 leaves the existing pool alone")
    ap.add_argument("--schema-limit", type=int, default=0,
                    help="Magpie tool-schema inventions; 0 leaves any existing pool alone")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # Spec section 10 fixes enable_thinking: true; the non-thinking roles (Magpie, the
    # simulated user, the simulated tool environment) hardcode thinking=False internally.
    cache = GenCache(args.cache or (args.root / "m2" / "cache.jsonl"))
    client = TeacherClient(args.base_url, model=args.model, timeout=args.timeout,
                           retries=args.retries, api_key=args.api_key)
    if args.dry_run:
        run_all(root=args.root, client=client, cache=cache, limit=args.limit,
                val_limit=args.val_limit, multi_limit=args.multi_limit,
                sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
                thinking=True, dry_run=True, concurrency=args.concurrency,
                schema_limit=args.schema_limit)
        return 0
    # Probe the endpoint before any mode runs. A dead or rotated tunnel URL turns every row
    # into `teacher_error`, and the passes still complete: `run_seeded` writes
    # `single.jsonl`, `run_difficulty` writes an all-error verdict file, and `run_merge`
    # overwrites `train.jsonl`. The corpus is destroyed at the shipped-artefact level while
    # the process exits 0.
    try:
        client._get("/health")
    except Exception as exc:
        raise SystemExit(f"teacher endpoint {args.base_url} is unhealthy: {exc}")
    if args.mode == "merge":
        run_merge(root=args.root)
        return 0
    run_all(root=args.root, client=client, cache=cache, limit=args.limit,
            val_limit=args.val_limit, multi_limit=args.multi_limit,
            sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
            thinking=True, dry_run=False, concurrency=args.concurrency,
            schema_limit=args.schema_limit)
    return 0


if __name__ == "__main__":
    sys.exit(smoke() if len(sys.argv) == 1 else main())
