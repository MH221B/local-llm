"""PPL rows: fp16 / INT2 / INT2+QAT over held-out smoltalk (+ optional WikiText-2).

Protocol (spec amendment): each window is prefilled (192 tokens) then decoded
teacher-forced one token at a time. The decode branch's lagged quantized K/V
is the thing INT2 does to the model; the gate counts decode-position NLL.
Prefill NLL is recorded per row as a cross-config sanity check (prefill
positions never consume quantization; fp16 vs INT2 prefill PPL should match).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/eval_ppl.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.cache import build_kitty_cache, load_model  # noqa: E402
from src.data import HOLDOUT, build_windows, build_windows_raw, split_windows  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"
PREFILL = 192
CONFIGS = {
    "fp16": {"kbits": 16, "vbits": 16, "promote_ratio": 0.0},
    "kitty-int2": {"kbits": 2, "vbits": 2, "promote_ratio": 0.1},
    "kitty-int2+qat": {"kbits": 2, "vbits": 2, "promote_ratio": 0.1},
}


def load(config_name: str, base: str, adapter: str | None, dtype):
    model, tok = load_model(base, dtype=dtype)
    if config_name == "kitty-int2+qat":
        assert adapter, "--adapter required for the qat row"
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, cast_adapter_dtype=False)
        model = model.merge_and_unload()
        model.eval()
    return model, tok


def nll_window(model, window: list[int], cfg: dict) -> tuple[float, float, int, int]:
    """(prefill NLL sum, decode NLL sum, prefill tokens, decode tokens)."""
    device = next(model.parameters()).device
    ids = torch.tensor([window], dtype=torch.long, device=device)

    cache, swapped = build_kitty_cache(
        DynamicCache(config=model.config), model.config, **cfg
    )
    with torch.no_grad():
        out = model(input_ids=ids[:, :PREFILL], use_cache=True, past_key_values=cache)
        pre_logits = out.logits.float()
        pre_nll = torch.nn.functional.cross_entropy(
            pre_logits[:, :-1].reshape(-1, pre_logits.shape[-1]),
            ids[:, 1:PREFILL].reshape(-1), reduction="sum").item()

        dec_nll = 0.0
        n_dec = 0
        # the prefill branch's LAST logit predicts token PREFILL — score it as the
        # first decode-position term so the boundary token isn't silently dropped
        dec_nll += torch.nn.functional.cross_entropy(
            pre_logits[:, -1], ids[:, PREFILL], reduction="sum").item()
        n_dec += 1
        for t in range(PREFILL, ids.shape[-1] - 1):
            # feed token t; its logits predict token t+1. Target the NEXT token,
            # never the one just fed (an off-by-one here mis-scores the gate).
            out = model(input_ids=ids[..., t:t + 1], use_cache=True,
                        past_key_values=cache)
            dec_nll += torch.nn.functional.cross_entropy(
                out.logits.float()[:, -1], ids[..., t + 1], reduction="sum").item()
            n_dec += 1

    pages = sum(cache.layers[i].quantized_pages for i in swapped)
    if cfg["kbits"] < 16:
        assert pages > 0, f"nothing quantized (pages={pages})"
    return pre_nll, dec_nll, PREFILL - 1, n_dec


def eval_config(model, windows, cfg: dict) -> dict:
    pre_nll = dec_nll = 0.0
    n_pre = n_dec = 0
    for w in windows:
        p, d, np_, nd = nll_window(model, w, cfg)
        pre_nll += p; dec_nll += d; n_pre += np_; n_dec += nd
    decode_ppl = float(torch.exp(torch.tensor(dec_nll / n_dec)))
    prefill_ppl = float(torch.exp(torch.tensor(pre_nll / n_pre)))
    return {"prefill_ppl": round(prefill_ppl, 4), "decode_ppl": round(decode_ppl, 4),
            "windows": len(windows), "decode_tokens": n_dec}


def run_rows(args, held, skip_fp16: bool = False) -> dict:
    rows = {}
    for name, cfg in CONFIGS.items():
        if skip_fp16 and name == "fp16":
            continue   # spec 6: fp16 rows are not double-run (not in the gate)
        model, _ = load(name, args.model, args.adapter, torch.float16)

        rows[name] = eval_config(model, held, cfg)
        print(f"{name:16s} {rows[name]}", flush=True)
        del model
        torch.cuda.empty_cache()   # safe no-op on CPU-only torch
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--runs", type=int, default=2,
                        help="2 -> records the run-to-run spread on int2/int2+qat")
    parser.add_argument("--wt2", action="store_true", help="also eval the WikiText-2 column")
    parser.add_argument("--out", type=Path, default=REPO / "bench" / "ppl-raw.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)   # tokenizer only; models load per row
    all_windows = build_windows(tok, n_conversations=2600)
    _, held = split_windows(all_windows, holdout=HOLDOUT)  # same split as the trainer

    results = {"held": {}}
    if args.wt2:
        results["wt2"] = {}
    for run in range(args.runs):
        results["held"][f"run{run}"] = run_rows(args, held, skip_fp16=(run > 0))
        if args.wt2:
            wt2_windows = build_windows_raw(tok)[-HOLDOUT:]
            results["wt2"][f"run{run}"] = run_rows(args, wt2_windows, skip_fp16=(run > 0))

    def spread(name: str):
        if args.runs < 2 or name == "fp16":
            return None
        a = results["held"]["run0"][name]["decode_ppl"]
        b = results["held"]["run1"][name]["decode_ppl"]
        return round(abs(a - b), 4)

    results["spread"] = {n: spread(n) for n in CONFIGS}
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
