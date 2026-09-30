"""M1 pipeline: fetch -> normalise -> strip to prompt -> dedup -> filter -> split -> report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from . import canonical, domains, filters, sources, textutil
from .adapters import get as get_adapter
from .adapters.base import Ctx
from .dedup import Deduper
from .imgstore import ImageStore, materialise_images
from .report import build_manifest, write_manifest
from .seeds import load_prompts as load_seed_prompts
from .split import downsample, stratified_split


def fetch_meta(dataset: str) -> tuple[str | None, str | None]:
    """License and revision from the HF API, for the manifest (best effort)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(dataset)
        return (info.cardData or {}).get("license"), info.sha
    except Exception:
        return None, None


def stream_source(src: sources.Source, quota: int, store: ImageStore, ctx: Ctx):
    adapter = get_adapter(src.adapter)
    stream = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
    produced = 0
    seen = 0
    for row in stream:
        if produced >= quota:
            return
        seen += 1
        ex = adapter.build(row, seen, ctx)
        if ex is None:
            continue
        materialise_images(ex, store)
        yield ex, adapter
        produced += 1


def run(*, root: Path, target_train: int, val_size: int, dry_run: bool,
        token_server_url: str | None = None) -> dict:
    store = ImageStore(root / "images")
    if token_server_url:
        from .tokenserver import LlamaServer, ServerTokenizer
        textutil.set_tokenizer(ServerTokenizer(LlamaServer(token_server_url)))
    quotas = sources.allocate(target_train + val_size, sources.SOURCES)
    deduper = Deduper()
    drops: Counter = Counter()
    accepted: list[canonical.Prompt] = []
    source_stats: list[dict] = []

    for src in sources.SOURCES:
        quota = quotas.get((src.dataset, src.config), 0) * 2  # candidate overshoot
        kept = 0
        if dry_run:
            print(f"[dry] {src.dataset}:{src.config} domain={src.domain} quota={quota}")
            continue
        license_, revision = fetch_meta(src.dataset)
        ctx = Ctx(src.dataset, src.config, src.domain, license_)
        for ex, _adapter in stream_source(src, quota, store, ctx):
            prompt_rec = canonical.strip_to_prompt(ex)
            if prompt_rec is None:
                drops["no_user_turn"] += 1
                continue
            prompt = domains.prompt_text(prompt_rec.messages)
            images = canonical.image_shas(prompt_rec)
            reason = deduper.check_prompt(prompt, images) if prompt else "empty_prompt"
            if reason:
                drops[reason] += 1
                continue
            if not deduper.has_new_images(prompt_rec):
                drops["duplicate_images"] += 1
                continue
            prompt_rec.domain = domains.assign_domain(prompt, prompt_rec.domain)
            prompt_rec.tokens = textutil.count_tokens(prompt)
            reason = filters.prompt_reject_reason(prompt, prompt_rec.tokens)
            if reason:
                drops[reason] += 1
                continue
            problems = canonical.validate_prompt(prompt_rec)
            if problems:
                drops["invalid"] += 1
                if drops["invalid"] <= 3:
                    print(f"  invalid {prompt_rec.id}: {problems[:2]}")
                continue
            # Accepted: only now does it enter the dedup index, so rejected rows
            # cannot suppress a later viable row.
            deduper.commit_prompt(prompt, images)
            deduper.commit_images(prompt_rec)
            accepted.append(prompt_rec)
            kept += 1
        source_stats.append({"dataset": src.dataset, "config": src.config,
                             "domain": src.domain, "cap": src.cap, "kept": kept,
                             "license": license_, "revision": revision})
        print(f"{src.dataset}:{src.config} kept {kept}")

    if dry_run:
        print("[dry] seeds:uncensored (local file from Task 10)")
        return {}

    # Uncensored prompts come from the seed harvest (Task 10), not from SOURCES.
    seed_records = load_seed_prompts(root)
    seeds_kept = 0
    for prompt_rec in seed_records:
        prompt = domains.prompt_text(prompt_rec.messages)
        images = canonical.image_shas(prompt_rec)   # seeds are text-only; empty here
        reason = deduper.check_prompt(prompt, images) if prompt else "empty_prompt"
        if reason:
            drops[reason] += 1
            continue
        prompt_rec.tokens = textutil.count_tokens(prompt)
        reason = filters.prompt_reject_reason(prompt, prompt_rec.tokens)
        if reason:
            drops[reason] += 1
            continue
        if canonical.validate_prompt(prompt_rec):
            drops["invalid"] += 1
            continue
        deduper.commit_prompt(prompt, images)
        accepted.append(prompt_rec)
        seeds_kept += 1
    if seed_records:
        source_stats.append({"dataset": "seeds:uncensored", "config": None,
                             "domain": "uncensored", "cap": len(seed_records),
                             "kept": seeds_kept, "license": None, "revision": None})
    print(f"seeds:uncensored kept {seeds_kept}")

    # Spec section 11.3: every image sha must resolve to a file on disk.
    missing = [(ex.id, ref.sha256) for ex in accepted for ref in ex.images
               if store.resolve(ref.sha256) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} image refs do not resolve to files; "
                           f"first: {missing[:3]}")

    trimmed = downsample(accepted, target_train + val_size)
    train, val = stratified_split(trimmed, val_size=val_size)

    write_count = canonical.write_jsonl(root / "prompts" / "train.jsonl", train)
    val_count = canonical.write_jsonl(root / "prompts" / "val.jsonl", val)
    if write_count < target_train or val_count < val_size:
        print(f"WARNING: short of target — train {write_count}/{target_train}, "
              f"val {val_count}/{val_size}, pool {len(accepted)}")
    manifest = build_manifest(
        run={"root": str(root), "artifact": "prompts", "target_train": target_train,
             "val_size": val_size, "train_written": write_count, "val_written": val_count,
             "token_counter": "exact" if textutil._TOKENIZER else "chars/4 estimate",
             "caveats": ["NuminaMath-1.5 may contain GSM8K-derived problems; "
                         "gsm8k-test is not fully clean (spec section 8)."]},
        train=train, val=val, sources=source_stats, drops=drops)
    write_manifest(root / "manifest.json", manifest)
    print(f"prompts train {write_count}, val {val_count}, manifest written")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 prompt pipeline")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--target-train", type=int, default=3000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-server", default="http://127.0.0.1:8085")
    args = ap.parse_args(argv)
    manifest = run(root=args.root, target_train=args.target_train,
                   val_size=args.val_size, dry_run=args.dry_run,
                   token_server_url=args.token_server)
    if manifest:
        print(json.dumps(manifest["train"]["by_domain"], indent=2))
        if (manifest["run"]["train_written"] < args.target_train
                or manifest["run"]["val_written"] < args.val_size):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
