"""Merge the QAT adapters, restore the 15 mtp.* tensors, assert the index.

Runs AFTER the last save touching the output dir: any later save_pretrained
silently drops mtp.* again (the HF graph has no MTP submodule — Spec 1 §3).
safetensors files are append-immutable, so the restore writes a new shard and
rewrites model.safetensors.index.json; tensors-on-disk-but-not-in-index is the
failure mode this assert exists to catch.
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

    out_index_path = args.out / "model.safetensors.index.json"
    out_index = json.loads(out_index_path.read_text())
    mtp_shard = "model-mtp.safetensors"
    save_file(tensors, args.out / mtp_shard)
    out_index["weight_map"].update({k: mtp_shard for k in mtp_names})
    out_index["metadata"]["total_size"] += sum(
        t.numel() * t.element_size() for t in tensors.values())
    out_index_path.write_text(json.dumps(out_index, indent=2))

    # ---- ASSERT 1 (spec 7): index ⊇ 15 mtp names, files exist, tensors in headers
    idx = json.loads((args.out / "model.safetensors.index.json").read_text())
    assert set(mtp_names).issubset(idx["weight_map"]), "index missing mtp names"
    for name in mtp_names:
        shard_path = args.out / idx["weight_map"][name]
        assert shard_path.exists(), shard_path
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            assert name in list(f.keys()), f"{name} absent from {shard_path} header"

    total = len(idx["weight_map"])
    print("assert 1 ok: 15 mtp tensors carried, index rewritten, files on disk")
    print(f"total index keys: {total}")
    assert total == 441, (
        f"expected 441 index keys (426 text + 15 mtp; the 297 visual.* keys live only "
        f"in the original checkpoint and never enter the saved artifact), got {total}")


if __name__ == "__main__":
    main()
