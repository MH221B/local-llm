"""M2 driver: teacher generation and verification over every M1 prompt column.

`all` runs magpie -> seeded (single-turn) -> trajectory (prebuilt) -> simulated -> merge.
Only verified, filtered rows are written; drops and pass rates are counted per source and
per domain so M3 can re-anchor its caps on measured numbers.
"""
from __future__ import annotations

import argparse
import json
import os
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
    """Order-preserving, streaming map. `concurrency <= 1` runs inline, so the M2 path is unchanged.

    Yields in input order as results complete, so the caller's `_progress` heartbeat ticks
    while a pool is being processed instead of bursting at the end. This must be a generator:
    `list(pool.map(...))` would compute the whole pool before yielding a single row, which is
    exactly the silence `_progress` exists to remove.
    """
    if concurrency <= 1 or len(items) <= 1:
        for item in items:
            yield fn(item)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        yield from pool.map(fn, items)


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


def generate_one(prompt, client, cache, *, store, thinking: bool, rollout: int = 0) -> dict:
    """One cached completion, verified and filtered. The verdict is recomputed every run.

    The cache holds the *completion*, not the verdict. Recomputing costs a local `verify`
    call (milliseconds) and is the difference between a filter or verifier edit applying to
    the whole cache and applying only to rows generated after it — this plan defers one such
    edit (`error_laden`), and mixing old-code rollout 0 with new-code rollout 1 would
    otherwise decide the difficulty verdict with two different rules.

    `rollout` 0 keeps M2's key, so an existing cache replays and the difficulty filter only
    pays for rollouts 1..K-1 (spec section 10.2).
    """
    kind = f"single:{prompt.id}" if rollout == 0 else f"single:{prompt.id}:r{rollout}"
    key = gencache.prefix_key(prompt.messages, kind=kind)
    cached = cache.get(key)
    stale = cached is not None and "completion" not in cached
    if cached is None or stale:
        # The `stale` half is what retires M2's verdict-shaped entries: they hold
        # `accepted`/`reason`/`example` with no completion to re-verify, so they are
        # regenerated once (and `put` is told to upgrade them, or the entry would be
        # regenerated again on every run and never persisted).
        msg = client.complete(prompt.messages, store=store, tools=prompt.tools,
                              thinking=thinking, seed=_seed(key))
        cached = {"completion": {"content": msg.get("content") or "",
                                 "tool_calls": msg.get("tool_calls") or []}}
        cache.put(key, cached, upgrade=stale)
    content = cached["completion"]["content"]
    ex = _completion_example(prompt, content, cached["completion"]["tool_calls"])
    if prompt.verify:
        ok, why = verify.check_record(prompt.verify, content)
        if not ok:
            return {"accepted": False, "reason": "verify_failed", "detail": why,
                    "example": ex.to_dict()}
    reason = filters.example_reject_reason(ex)
    return {"accepted": reason is None, "reason": reason, "example": ex.to_dict()}


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


DIFFICULTY_K = 2
ACCEPTED_TRAIN_TARGET = 25_000   # spec section 4; merge trims to it when the filters do not


def run_difficulty(*, root: Path, client, cache, k: int = DIFFICULTY_K,
                   limit: int | None = None, thinking: bool = True, concurrency: int = 1):
    """Spec section 10.2 rejection sampling over oracle-bearing prompts.

    Keeps a prompt only when the teacher passes some rollouts and fails others. Rollout 0
    reuses M2's cache key, so a warm cache costs only the extra rollouts.

    Writes the survivors *and* a verdict line for every judged prompt. Merge filters against
    the verdicts, not the survivors: an all-fail prompt is absent from the survivors but
    present in the corpus from the seeded pass, and only the verdicts name it.

    `verify` is the oracle, so a prompt without one is out of scope: "all-pass" is
    meaningless for a column with no mechanical test.
    """
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "k": k, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    records = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            records.extend(p for p in (prompt_from_dict(r) for r in iter_jsonl(path))
                           if p.verify)
    if limit is not None:
        records = split.downsample(records, limit)

    def work(prompt):
        rollouts = []
        for i in range(k):
            try:
                rollouts.append(generate_one(prompt, client, cache, store=store,
                                             thinking=thinking, rollout=i))
            except TeacherError as exc:
                return rollouts, exc
        return rollouts, None

    out: list[Example] = []
    verdicts: list[dict] = []
    for done, (prompt, (rollouts, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None or len(rollouts) < k:
            stats["drops"]["teacher_error"] += 1
            verdicts.append({"id": prompt.id, "verdict": "teacher_error",
                             "source": key, "passes": None})
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
        else:
            passes = [r for r in rollouts if r["accepted"]]
            if len(passes) == k:
                stats["drops"]["all_pass"] += 1
                verdicts.append({"id": prompt.id, "verdict": "all_pass", "source": key,
                                 "passes": k})
            elif not passes:
                stats["drops"]["all_fail"] += 1
                verdicts.append({"id": prompt.id, "verdict": "all_fail", "source": key,
                                 "passes": 0})
            else:
                out.append(canonical.example_from_dict(passes[0]["example"]))
                stats["accepted"] += 1
                stats["by_domain"][prompt.domain] += 1
                stats["accepted_by_source"][key] += 1
                verdicts.append({"id": prompt.id, "verdict": "keep", "source": key,
                                 "passes": len(passes)})
        _progress("difficulty", done, len(records), stats, every=50)
    canonical.write_jsonl(root / "m2" / "difficulty.jsonl", out)
    # The first line is a fingerprint, not a verdict. `run_merge` refuses to filter against a
    # file written for a different set of oracle prompts, which is what makes a *partial*
    # difficulty pass (or a rebuilt corpus) fail loudly instead of silently un-filtering: a
    # `--limit 12` verdict file would otherwise re-admit every row the full pass had dropped.
    meta = {"kind": "meta", "k": k, "judged": len(verdicts), "oracle_in_pools": len(records)}
    (root / "m2" / "difficulty.verdicts.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in [meta, *verdicts]),
        encoding="utf-8")
    _dump_stats(root / "m2" / "difficulty.stats.json", stats)
    print(f"difficulty: accepted {stats['accepted']} of {stats['attempted']} judged "
          f"(k={k}, drops {dict(stats['drops'])})", flush=True)
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
    # Seed-only rows (a conversation that ends on the assistant's tool call) can never pass
    # `generate_prebuilt`: it returns None for a call with no source observation to splice
    # in. They exist for `run_simulated`, which reads the same file and needs only the first
    # user turn and `tools`. Skipping them here is what keeps that split honest -- attempting
    # them would spend a teacher call to record a drop, every time.
    seed_only = [p for p in records if p.meta.get("seed_only")]
    records = [p for p in records if not p.meta.get("seed_only")]
    if seed_only:
        print(f"trajectory: skipping {len(seed_only)} seed-only rows "
              f"(the simulated loop seeds from them)", flush=True)
    stats["seed_only_skipped"] = len(seed_only)
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
    environment. Seeded two ways: from a prebuilt trajectory's own first user turn, and from
    a tool schema with no conversation at all, where the request comes from the schema
    (spec section 10.2). Rows carry `meta.simulated = true`."""
    from types import SimpleNamespace

    store = ImageStore(root / "images")
    stats = _traj_stats()
    stats["schema_seeded"] = 0

    prepared: list[tuple] = []
    traj_path = root / "prompts" / "trajectories.jsonl"
    traj_seeds = [t for t in (trajectory_from_dict(r) for r in iter_jsonl(traj_path))
                  if t.tools] if traj_path.exists() else []
    for t in traj_seeds:
        first_user = next((canonical.message_text(m) for m in t.messages
                           if m.get("role") == "user"), "")
        if first_user:
            # `sim-` matters: the raw id is the trajectory prompt's id, which
            # `m2/trajectory.jsonl` already uses, and `run_merge` concatenates both shards.
            prepared.append((f"sim-{t.id}", t.domain, dict(t.source), t.tools, first_user,
                             False))

    # Schema-seeded seeds. The Magpie row's own user turn *is* the schema-conditioned
    # request Task 4 invented, so it is passed straight through as `first_user` rather than
    # re-invented: `generate_simulated`'s own invention path uses a canned prompt and never
    # sees the row's `tools`, so re-inventing there would drop the schema from the loop
    # (spec section 10.2 asks for a user turn invented *from* a tool schema).
    mag_path = root / "prompts" / "magpie.jsonl"
    for row in (iter_jsonl(mag_path) if mag_path.exists() else []):
        if not row.get("tools") or not row.get("meta", {}).get("schema_seeded"):
            continue
        first_user = next((canonical.message_text(m) for m in row.get("messages", [])
                           if m.get("role") == "user"), "")
        if not first_user:
            continue
        prepared.append((f"sim-schema-{row['id']}", row["domain"],
                         {"name": "teacher:simulated:schema"}, row["tools"], first_user,
                         True))

    # `--sim-limit` bounds the whole simulated pool, so it is applied once, after both
    # sources are pooled. It is deliberately *not* applied to `traj_seeds` above: doing both
    # would downsample the prebuilt rows twice while the schema rows are capped once, and
    # the prebuilt:schema mix would skew. `downsample` reads only `.domain`, hence the wrapper.
    wrapped = [SimpleNamespace(domain=item[1], item=item, id=item[0]) for item in prepared]
    if limit is not None:
        wrapped = split.downsample(wrapped, limit)
    schema_count = sum(1 for w in wrapped if w.item[5])
    stats["schema_seeded"] = schema_count
    print(f"simulated seeds: {len(wrapped)} ({schema_count} schema-seeded)", flush=True)

    def work(w):
        sid, domain, source, tools, first_user, _schema = w.item
        try:
            return trajectory.generate_simulated(
                client, cache, id=sid, domain=domain, source=source, tools=tools,
                first_user=first_user, thinking=thinking, store=store), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (w, (traj, exc)) in enumerate(zip(wrapped, _map(
            work, wrapped, concurrency)), start=1):
        sid, domain, source, tools, first_user, schema_seeded = w.item
        key = f"teacher:simulated|{domain}"
        # The reject ledger names the pool the row came from, so a schema-seeded drop is
        # distinguishable from a prebuilt-seeded one.
        pool = ("prompts/magpie.jsonl" if schema_seeded else "prompts/trajectories.jsonl")
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {sid}: {exc}", flush=True)
            _record_reject(root, stage="simulated", reason="teacher_error", detail=str(exc),
                           pool=pool, id=sid)
        elif traj is None:
            stats["drops"]["turn_structure"] += 1
            _record_reject(root, stage="simulated", reason="turn_structure",
                           pool=pool, id=sid)
        else:
            reason, detail = _trajectory_reject(traj)
            if reason:
                stats["drops"][reason] += 1
                _record_reject(root, stage="simulated", reason=reason, detail=detail,
                               pool=pool, id=sid, record=traj.to_dict())
            else:
                out.append(traj)
                stats["accepted"] += 1
                stats["by_domain"][domain] += 1
                stats["accepted_by_source"][key] += 1
                stats["simulated"] += 1
        _progress("simulated", done, len(wrapped), stats, every=25)
    canonical.write_jsonl(root / "m2" / "simulated.jsonl", out)
    _dump_stats(root / "m2" / "simulated.stats.json", stats)
    print(f"simulated: accepted {stats['accepted']} of {stats['attempted']} "
          f"({schema_count} schema-seeded)", flush=True)
    return stats


def _dump_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    att = serial.get("attempted_by_source", {})
    acc = serial.get("accepted_by_source", {})
    serial["pass_rate_by_source"] = {
        k: round(acc.get(k, 0) / n, 4) for k, n in att.items() if n}
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


OUTAGE_TEACHER_ERROR_SHARE = 0.5


def _guard_teacher_outage(stats_by_stage: dict) -> None:
    """Refuse to merge when a stage mostly failed on the endpoint.

    A tunnel or Colab session that dies mid-run turns every remaining call into
    `teacher_error`. The passes still "complete": `run_seeded` writes `single.jsonl` with the
    few rows that succeeded, and `run_merge` would then overwrite `train.jsonl` with that
    gutted set while exiting 0. Measured 2026-10-03 — 2,742/3,428 seeded attempts and 100% of
    difficulty/trajectory/simulated were `teacher_error`, and the merge shipped 679 rows and
    an empty val. This guard makes that merge fail loudly instead. Set `M3_FORCE_MERGE=1` to
    override when a high error share is deliberate.
    """
    if os.environ.get("M3_FORCE_MERGE") == "1":
        return
    for stage, s in stats_by_stage.items():
        attempted = (s or {}).get("attempted", 0)
        err = ((s or {}).get("drops") or {}).get("teacher_error", 0)
        if attempted and err / attempted > OUTAGE_TEACHER_ERROR_SHARE:
            raise SystemExit(
                f"run_merge: {stage} lost {err}/{attempted} attempts to teacher_error "
                f"({err / attempted:.0%}); the endpoint was down. Refusing to overwrite "
                f"train.jsonl with a gutted corpus. Bring the endpoint back and re-run, or "
                f"set M3_FORCE_MERGE=1 to merge the rows that did succeed.")


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
    survivors = load("difficulty.jsonl")
    verdicts_path = root / "m2" / "difficulty.verdicts.jsonl"
    judged: list[dict] = []
    meta: dict = {}
    if verdicts_path.exists():
        rows = [json.loads(line) for line in
                verdicts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = next((r for r in rows if r.get("kind") == "meta"), {})
        judged = [r for r in rows if r.get("kind") != "meta"]
    if judged:
        # Guard first, filter second. A verdict file covering a subset of the oracle prompts
        # re-admits every row the full pass had dropped and drops the survivors it never saw,
        # and nothing else in the pipeline would notice.
        current_oracle = sum(1 for rel in TRAIN_POOLS if (root / rel).exists()
                             for r in iter_jsonl(root / rel) if prompt_from_dict(r).verify)
        if meta.get("oracle_in_pools") != current_oracle:
            raise SystemExit(
                f"{verdicts_path} was written for {meta.get('oracle_in_pools')} oracle "
                f"prompts but the pools hold {current_oracle}; re-run `--mode difficulty` "
                f"(idempotent and cache-backed) before merging")
        # The filter's decision replaces the seeded row, whatever that decision was:
        # `keep` contributes the survivor, `all_pass` and `all_fail` contribute nothing.
        # A `teacher_error` left no verdict, so its seeded row stands; that keeps an
        # interrupted endpoint from silently shrinking the corpus.
        decided = {v["id"] for v in judged if v["verdict"] != "teacher_error"}
        train = [ex for ex in train if ex.id not in decided] + survivors
    # Before any write: a mid-run endpoint outage must not silently gut the shipped corpus.
    # Placed after the verdict guard above so a partial difficulty pass is still reported as
    # such, and before the writes so `train.jsonl` is untouched when this raises.
    _guard_teacher_outage({
        stage: (json.loads((root / "m2" / f"{stage}.stats.json").read_text(encoding="utf-8"))
                if (root / "m2" / f"{stage}.stats.json").exists() else {})
        for stage in ("seeded", "trajectory", "simulated", "difficulty")
    })
    if not train:
        raise SystemExit("merge produced no train rows; refusing to overwrite train.jsonl")
    accepted_before = len(train)
    if len(train) > ACCEPTED_TRAIN_TARGET:
        # Spec section 13: "filter down to 25,000 rather than capping at 25,000". When pass
        # rates come in high the filters do not reduce the count, and section 4's mix is what
        # the trim holds -- `downsample` weights by DOMAIN_SHARE, which is 30/30/20/20.
        train = split.downsample(train, ACCEPTED_TRAIN_TARGET)
    val = load("val.jsonl")
    canonical.write_jsonl(root / "train.jsonl", train)
    canonical.write_jsonl(root / "val.jsonl", val)

    def stats(rel):
        p = root / "m2" / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    m2 = {"seeded": stats("seeded.stats.json"), "trajectory": stats("trajectory.stats.json"),
          "simulated": stats("simulated.stats.json"),
          "difficulty": stats("difficulty.stats.json"),
          "accepted_before_trim": accepted_before,
          "difficulty_judged": len(judged),
          "difficulty_shipped": len(survivors),
          "difficulty_removed": len([v for v in judged
                                     if v["verdict"] not in ("keep", "teacher_error")]),
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
    run_difficulty(root=root, client=client, cache=cache, k=DIFFICULTY_K,
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

    class FlakyTeacher:
        """Passes the odd-numbered call, fails the even-numbered one.

        With `k=2` and one prompt that is one pass and one fail: the mixed verdict the
        filter is supposed to keep.
        """

        def __init__(self, always_fail: bool = False):
            self.calls = 0
            self.always_fail = always_fail

        def complete(self, messages, **kw):
            self.calls += 1
            # Code inside a fence, prose outside it. `verify.extract_code` with no fence
            # returns the whole completion, so prose in the same string reaches `exec` and
            # raises SyntaxError: every rollout fails the oracle and the "mixed" case
            # silently becomes an all-fail case instead of exercising the filter.
            good = ("```python\ndef add(a, b):\n    return a + b\n```\n\n"
                    "The function sums its two arguments and returns the result, so "
                    "add(1, 2) is 3.")
            bad = ("```python\ndef add(a, b):\n    return a - b\n```\n\n"
                   "This version subtracts the second argument from the first and returns "
                   "that value instead.")
            return {"content": bad if (self.always_fail or self.calls % 2 == 0) else good}

    def diff_case(teacher, name):
        dtmp = Path(tmp) / f"diff-{name}"
        canonical.write_jsonl(dtmp / "prompts" / "train.jsonl", [Prompt(
            id="d1", domain="coding", origin="prebuilt", source={"name": "smoke"},
            messages=[{"role": "user", "content": [{"type": "text", "text": "Write add."}]}],
            verify={"type": "python_tests", "setup": "",
                    "tests": ["assert add(1, 2) == 3"]})])
        st = run_difficulty(root=dtmp, client=teacher,
                            cache=GenCache(dtmp / "m2" / "cache.jsonl"), k=2,
                            concurrency=1)
        print(f"difficulty {name}: accepted {st['accepted']} "
              f"drops {dict(st['drops'])}", flush=True)
        return st

    mixed = diff_case(FlakyTeacher(), "mixed")
    failed = diff_case(FlakyTeacher(always_fail=True), "all-fail")

    # Outage guard: a stage that mostly failed on the endpoint must block the merge; a normal
    # shortfall and an empty stage must not.
    def _aborts(stats):
        try:
            _guard_teacher_outage(stats)
            return False
        except SystemExit:
            return True

    outage_ok = (_aborts({"seeded": {"attempted": 100, "drops": {"teacher_error": 90}}})
                 and _aborts({"trajectory": {"attempted": 10, "drops": {"teacher_error": 10}}})
                 and not _aborts({"seeded": {"attempted": 100, "drops": {"teacher_error": 10}}})
                 and not _aborts({"seeded": {"attempted": 0, "drops": {}}}))
    print("outage guard blocks gutted merge:", outage_ok)

    # `_map` must stream. A list-returning `_map` computed the whole pool before yielding, so
    # `_progress` only burst at the end and a live run looked hung. Guard the laziness.
    import inspect

    lazy_ok = inspect.isgenerator(_map(lambda x: x, [1, 2, 3], concurrency=2))
    print("map streams lazily:", lazy_ok)

    ok = (stats["accepted"] == 1
          and len(rejected) == 1
          and rejected[0]["reason"] == "verify_failed" and has_completion
          and mixed["accepted"] == 1 and not dict(mixed["drops"])
          and failed["accepted"] == 0 and failed["drops"]["all_fail"] == 1
          and outage_ok
          and lazy_ok
          and validate(canonical.example_from_dict(first)) == [])
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    # Line-buffered, so a `Tee-Object` or a redirected log shows progress as it happens
    # rather than in 8 KiB blocks at the end.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):        # not a TextIOWrapper (e.g. under pytest)
        pass
    ap = argparse.ArgumentParser(description="M2 teacher generation")
    ap.add_argument("--mode", choices=["all", "merge", "difficulty"], default="all")
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
    if args.mode == "difficulty":
        run_difficulty(root=args.root, client=client, cache=cache, k=DIFFICULTY_K,
                       limit=args.limit, concurrency=args.concurrency)
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
