"""Check 4: FP16 vs Kitty INT2 vs Kitty-Pro pass rate on a small prompt set."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.cache import build_kitty_cache, load_model  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

# name -> config. Ratios are stated explicitly in every result row because the repo
# carries three different "defaults" (0.1 in the config, 0.0 in the CLI, 0.2 in
# accuracy_eval.sh). fp16 is expressed as "16-bit" so it routes through the identical
# code path and differs only in the quantization.
CONFIGS = {
    "fp16": {"kbits": 16, "vbits": 16, "promote_ratio": 0.0},
    "kitty-int2": {"kbits": 2, "vbits": 2, "promote_ratio": 0.1},
    "kitty-pro-int2": {"kbits": 2, "vbits": 2, "promote_ratio": 0.2},
}

# Kitty quantizes nothing until the cached length exceeds sink + buffer = 160.
# Short prompts would make every config measure an unquantized run.
DEAD_ZONE = 160


def pad_to(tokenizer, text: str, min_tokens: int) -> str:
    """Pad a prompt with neutral filler until it tokenizes to at least min_tokens."""
    filler = " The following context is provided for length only and is not relevant:"
    while len(tokenizer(text).input_ids) < min_tokens:
        text = text + filler + " filler token padding text."
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RBergBauer/Qwen3.5-4B-MTP-Heretic")
    parser.add_argument("--prompts", type=Path, default=REPO / "bench" / "prompts.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-prompt-tokens", type=int, default=DEAD_ZONE + 32,
                        help="must exceed the 160-token dead zone so prefill quantizes")
    parser.add_argument("--out", type=Path, default=REPO / "bench" / "baseline-raw.json")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]

    results = {}
    for name in CONFIGS:
        passed = 0
        details = []
        for item in prompts:
            question = pad_to(tokenizer, item["prompt"], args.min_prompt_tokens)
            chat = [{"role": "user", "content": question}]
            inputs = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                enable_thinking=False,  # reasoning would eat the token budget before the answer
            )
            assert inputs["input_ids"].shape[-1] > DEAD_ZONE, (
                f"prompt is {inputs['input_ids'].shape[-1]} tokens, at or below the "
                f"{DEAD_ZONE}-token dead zone; nothing would be quantized"
            )

            cache, swapped = build_kitty_cache(
                DynamicCache(config=model.config), model.config, **CONFIGS[name]
            )
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, past_key_values=cache)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            pages = sum(cache.layers[i].quantized_pages for i in swapped)
            if name != "fp16":
                assert pages > 0, f"{name}: nothing was quantized - the swap did not take effect"

            hit = bool(re.search(re.escape(item["answer"]), text, re.IGNORECASE))
            passed += hit
            details.append({"prompt": item["prompt"], "output": text, "pass": hit,
                            "prompt_tokens": inputs["input_ids"].shape[-1],
                            "quantized_pages": pages})
        results[name] = {"config": CONFIGS[name], "passed": passed, "of": len(prompts),
                         "details": details}
        print(f"{name:16s} {passed}/{len(prompts)}  pages={details[0]['quantized_pages']}  {CONFIGS[name]}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
