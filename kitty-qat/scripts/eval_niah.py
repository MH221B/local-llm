"""NIAH guardrail: 10 needles over 2-4K-token haystacks, 3 configs.

2609.04263's caveat: PPL recovery can mask retrieval damage that greedy NLL
never sees. Reported, not gated.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/eval_niah.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.eval_ppl import CONFIGS, load  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from src.cache import build_kitty_cache  # noqa: E402

SEED = 0
NEEDLES = 10
HAY_MIN, HAY_MAX = 2000, 4000
MAX_NEW = 64
FILLER = "The grass is green. The sky is blue. The sun is bright.\n\n"


def haystack_with_needle(rng: random.Random, tok) -> tuple[str, str]:
    """Length in TOKENS (the spec's numbers), not characters."""
    target = rng.randrange(HAY_MIN, HAY_MAX)
    code = f"HORSE-{rng.randrange(10 ** 6):06d}"
    fill_ids = tok.encode(FILLER, add_special_tokens=False)
    needle_ids = tok.encode(
        f"One unremarkable line hides the fact that the passcode is {code}.\n",
        add_special_tokens=False)
    body_ids = fill_ids * ((target - len(needle_ids)) // len(fill_ids) + 1)
    pos = rng.randrange(0, len(body_ids) - len(needle_ids))
    full_ids = (body_ids[:pos] + needle_ids + body_ids[pos + len(needle_ids):])[:target]
    return tok.decode(full_ids), code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RBergBauer/Qwen3.5-4B-MTP-Heretic")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "bench" / "niah-raw.json")
    args = parser.parse_args()

    rng = random.Random(SEED)
    stub_tok = load("fp16", args.model, None, torch.float16)[1]
    examples = [haystack_with_needle(rng, stub_tok) for _ in range(NEEDLES)]

    results = {}
    for name in CONFIGS:
        model, tok = load(name, args.model, args.adapter, torch.float16)
        cache, swapped = build_kitty_cache(
            DynamicCache(config=model.config), model.config, **CONFIGS[name]
        )
        hits = 0
        details = []
        for haystack, code in examples:
            ids = tok(
                "Below is a long document. Find the passcode in it and answer "
                "with just the passcode.\n\n" + haystack,
                return_tensors="pt").input_ids.to(model.device)
            assert ids.shape[-1] > 160, ids.shape  # dead-zone parity assert (spec 6)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                                     do_sample=False, past_key_values=cache)
            text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            found = code in text
            hits += found
            details.append({"haystack_tokens": int(ids.shape[-1]), "pass": found,
                            "completion": text[:120]})
        pages = sum(cache.layers[i].quantized_pages for i in swapped)
        if name != "fp16":
            assert pages > 0, f"{name}: nothing quantized - INT2 row is fp16"
        results[name] = {"hits": hits, "of": NEEDLES, "details": details}
        print(f"{name:16s} {hits}/{NEEDLES} pages={pages}")
        del model, cache
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
