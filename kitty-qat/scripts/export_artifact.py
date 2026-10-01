"""Merge the QAT adapters, restore the 15 mtp.* tensors, assert the artifact.

Runs AFTER the last save touching the output dir: any later save_pretrained
silently drops mtp.* again (the HF graph has no MTP submodule — Spec 1 §3).

Handles both output layouts save_pretrained can produce:
  * multi-shard : model-0000X-of-0000N.safetensors + model.safetensors.index.json
  * single-shard: model.safetensors with NO index. A 4B fp16 model lands under
    the default 5 GB shard threshold once the 297 visual.* tensors are gone, so
    this is the common case for this checkpoint. We synthesize the index in that
    branch so the mtp shard is reachable by both HF and the GGUF converter.

safetensors files are append-immutable, so the mtp restore always writes a new
shard; the failure mode this assert exists to catch is tensors on disk but not
referenced by the index.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/export_artifact.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.cache import load_model  # noqa: E402

MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"
MTP_SHARD = "model-mtp.safetensors"
INDEX_NAME = "model.safetensors.index.json"


def _tensor_bytes(path: Path) -> int:
    """Sum of tensor payload bytes in a safetensors file (no header overhead)."""
    from safetensors import safe_open

    total = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            numel = 1
            for d in sl.get_shape():
                numel *= d
            total += numel * torch.empty(0, dtype=sl.get_dtype()).element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=MODEL_ID)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base, dtype=torch.float16)   # fp16 to match Spec 1's
    # baseline route (spec 6 parity); the merge casts adapter fp32 -> weight fp16 itself
    if torch.cuda.is_available():
        model = model.cuda()

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter, cast_adapter_dtype=False)
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / MTP_SHARD).unlink(missing_ok=True)   # idempotent re-runs
    model.save_pretrained(args.out)          # 426 keys: no mtp submodule in the graph
    tokenizer.save_pretrained(args.out)

    # ---- carry the 15 mtp.* tensors verbatim from the original checkpoint
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    snap = Path(snapshot_download(args.base))
    orig_index = json.loads((snap / "model.safetensors.index.json").read_text())
    mtp_names = sorted(k for k in orig_index["weight_map"] if k.startswith("mtp."))
    assert len(mtp_names) == 15, mtp_names
    mtp_map = {k: orig_index["weight_map"][k] for k in mtp_names}

    tensors = {}
    for shard in sorted(set(mtp_map.values())):
        with safe_open(snap / shard, framework="pt", device="cpu") as f:
            for k, s in mtp_map.items():
                if s == shard:
                    tensors[k] = f.get_tensor(k)
    assert len(tensors) == 15, len(tensors)
    save_file(tensors, args.out / MTP_SHARD)

    # ---- locate the text shard(s) and (re)write the index
    text_shards = sorted(p for p in args.out.glob("*.safetensors") if p.name != MTP_SHARD)
    index_path = args.out / INDEX_NAME

    if index_path.exists():
        idx = json.loads(index_path.read_text())          # multi-shard: keep its map
    else:
        # single-shard: synthesize an index over the text file + the mtp shard
        assert len(text_shards) == 1, (
            f"no index and not a single text shard: {[p.name for p in text_shards]}")
        with safe_open(text_shards[0], framework="pt", device="cpu") as f:
            text_names = list(f.keys())
        idx = {"metadata": {"total_size": 0},
               "weight_map": {k: text_shards[0].name for k in text_names}}

    idx["weight_map"].update({k: MTP_SHARD for k in mtp_names})
    all_shards = sorted(p for p in args.out.glob("*.safetensors"))
    idx["metadata"]["total_size"] = sum(_tensor_bytes(p) for p in all_shards)
    index_path.write_text(json.dumps(idx, indent=2))

    # ---- ASSERT 1 (spec 7): index ⊇ 15 mtp names, files exist, tensors in headers
    idx = json.loads(index_path.read_text())
    assert set(mtp_names).issubset(idx["weight_map"]), "index missing mtp names"
    for name in mtp_names:
        shard_path = args.out / idx["weight_map"][name]
        assert shard_path.exists(), shard_path
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            assert name in list(f.keys()), f"{name} absent from {shard_path} header"

    total = len(idx["weight_map"])
    print(f"layout: {'multi-shard' if len(text_shards) > 1 else 'single-shard'} "
          f"({[p.name for p in text_shards]}) + {MTP_SHARD}")
    print("assert 1 ok: 15 mtp tensors carried, index written, files on disk")
    print(f"total index keys: {total}")
    assert total == 441, (
        f"expected 441 index keys (426 text + 15 mtp; the 297 visual.* keys live only "
        f"in the original checkpoint and never enter the saved artifact), got {total}")


if __name__ == "__main__":
    main()
