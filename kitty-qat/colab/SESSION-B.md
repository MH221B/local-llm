# Session B — evals (~1 h, Colab L4)

Fresh runtime, Drive mounted. Requires Session A's artifacts on Drive.

**IMPORTANT:** pass `--adapter /content/drive/MyDrive/qat-lora` (the adapter dir,
not the merged dir) — `PeftModel.from_pretrained` expects an adapter checkpoint;
the merged dir is for the local Task-9 GGUF convert.

```python
# Cell 1
!pip install -q peft datasets gguf
!unzip -q /content/drive/MyDrive/kitty-qat.zip -d /content
%cd /content/kitty-qat
```

```python
# Cell 2 — PPL rows x2 runs + WikiText-2 column (~40 min)
!python scripts/eval_ppl.py --adapter /content/drive/MyDrive/qat-lora --wt2 --out /content/ppl-raw.json
```

```python
# Cell 3 — sanity + guardrail (~20 min)
!python scripts/eval_gsm8k.py --adapter /content/drive/MyDrive/qat-lora --out /content/gsm8k-raw.json
!python scripts/eval_niah.py --adapter /content/drive/MyDrive/qat-lora --out /content/niah-raw.json
```

```python
# Cell 4 — copy the three JSONs home
!cp /content/ppl-raw.json /content/gsm8k-raw.json /content/niah-raw.json /content/drive/MyDrive/
```
