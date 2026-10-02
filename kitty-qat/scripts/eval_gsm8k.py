"""GSM8K sanity row: 50 test prompts, 8-shot CoT, greedy, max-new 512.

Not expected to resolve a delta (single-pass n=50). Direction check vs the
paper's band; the cache-side asserts (pages>0, window>160) stay honest here.

Single-pass sanity check, so results carry only "run0" (no run1). Each prompt
gets its own fresh quantizing cache.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/eval_gsm8k.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset

from scripts.eval_ppl import CONFIGS, load  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from src.cache import build_kitty_cache  # noqa: E402

FEWSHOT = 8
MAX_NEW = 512


def build_prompt(train) -> str:
    shots = []
    for ex in train:
        reasoning, _, final = ex["answer"].rpartition("####")
        shots.append(
            f"Q: {ex['question']}\n"
            f"A: {reasoning.strip()}\n"
            f"{final.strip()}"
        )
        if len(shots) == FEWSHOT:
            break
    return "\n".join(shots) + "\n\n"


def final_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RBergBauer/Qwen3.5-4B-MTP-Heretic")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "bench" / "gsm8k-raw.json",
    )
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    prompt = build_prompt(ds["train"])
    test = list(ds["test"])[: args.n]

    # GSM8K is intentionally a single-pass sanity check. There is no run1.
    results = {"run0": {}}

    for name in CONFIGS:
        print(f"\n=== gsm8k {name} ===", flush=True)
        print(f"  loading {name} ...", flush=True)

        model, tok = load(name, args.model, args.adapter, torch.float16)

        correct = 0
        details = []
        total_pages = 0

        for i, ex in enumerate(test, 1):
            # Every prompt must have its own empty cache. Reusing a cache across
            # independent prompts makes generate() treat the next prompt as a
            # continuation and can produce a zero-length model input.
            cache, swapped = build_kitty_cache(
                DynamicCache(config=model.config),
                model.config,
                **CONFIGS[name],
            )

            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt + ex["question"]}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            ids = tok(chat, return_tensors="pt").input_ids.to(model.device)
            assert ids.shape[-1] > 160, ids.shape  # dead-zone parity assert

            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=MAX_NEW,
                    do_sample=False,
                    past_key_values=cache,
                )

            completion = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            pred = final_number(completion)
            expected = final_number(ex["answer"].rpartition("####")[2])
            ok = (
                pred is not None
                and expected is not None
                and float(pred) == float(expected)
            )
            correct += ok
            details.append(
                {
                    "question": ex["question"],
                    "pred": pred,
                    "expected": expected,
                    "pass": ok,
                    "completion_head": completion[:200],
                }
            )

            pages = sum(
                cache.layers[layer_index].quantized_pages
                for layer_index in swapped
            )
            total_pages += pages
            if name != "fp16":
                assert pages > 0, f"{name}: nothing quantized - INT2 row is fp16"
            del cache

            if i % 5 == 0 or i == len(test):
                print(
                    f"    gsm8k {name:14s} "
                    f"{i:>3}/{len(test)} correct={correct}",
                    flush=True,
                )

        results["run0"][name] = {
            "correct": correct,
            "of": len(test),
            "details": details,
            "total_pages": total_pages,
        }
        print(f"{name:16s} {correct}/{len(test)} total_pages={total_pages}")

        del model
        torch.cuda.empty_cache()

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
