# Session B — evals (~2-2.5 h, Colab L4)

Fresh runtime. **Mount Drive first** (cell 1) — a new session has no
`/content/drive` until you do, and `unzip` will report "cannot find" even though
the zip is right there in My Drive.

**IMPORTANT:** pass `--adapter /content/drive/MyDrive/qat-lora` (the adapter dir,
not the merged dir) — `PeftModel.from_pretrained` expects an adapter checkpoint;
the merged dir is for the local Task-9 GGUF convert.

```python
# Cell 1
from google.colab import drive
drive.mount("/content/drive")
# Remove Colab's stale torchao (0.10.0): peft >= 0.21 raises on it during LoRA
# injection (is_torchao_available) even though we never use torchao.
!pip uninstall -y torchao
!pip install -q peft datasets gguf
!unzip -q -o /content/drive/MyDrive/kitty-qat.zip -d /content
%cd /content/kitty-qat
!ls
```

```python
# Cell 2 — PPL rows x2 runs + WikiText-2 column. ~1.5-2 h (10 row-evals; decode
# is memory-bound). Prints a running decode_ppl every 4 windows, a finished row
# per config, then a summary block. Drop --wt2 to halve it (gate unaffected).
!python scripts/eval_ppl.py --adapter /content/drive/MyDrive/qat-lora --wt2 --out /content/ppl-raw.json
```

```python
# Cell 3 — GSM8K sanity + NIAH guardrail. ~20-30 min. Progress per 5 examples.
!python scripts/eval_gsm8k.py --adapter /content/drive/MyDrive/qat-lora --out /content/gsm8k-raw.json
!python scripts/eval_niah.py  --adapter /content/drive/MyDrive/qat-lora --out /content/niah-raw.json
```

```python
# Cell 4 — copy the three JSONs home
!cp /content/ppl-raw.json /content/gsm8k-raw.json /content/niah-raw.json /content/drive/MyDrive/
```

## What to watch

- **The gate:** `decode_ppl(kitty-int2+qat) < decode_ppl(kitty-int2)` at the same
  2-bit budget. That is the success criterion.
- **Sanity:** `fp16` and `kitty-int2` should have near-identical `prefill_ppl`
  (prefill never consumes quantization) while their `decode_ppl` diverge. If
  prefill PPL differs wildly between them, stop and investigate before trusting
  the numbers.
- The scripts print a `loading <config> ...` line, so if it looks idle, check that
  line is followed by running-ppl lines — that confirms the GPU is actually in use.
