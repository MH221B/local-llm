# Session A — train, merge, carry mtp, save (~2.5-3 h, Colab L4)

Fresh Colab L4 runtime, Drive mounted at /content/drive.

**Cell 0 (before anything) — get the code onto the box.** From the local machine
(PowerShell), build a zip whose entries are nested under `kitty-qat/`:

```powershell
cd C:\Users\tnmh\projects\local-llm
# NOTE: `Compress-Archive -Path kitty-qat\src, kitty-qat\scripts` strips the parent
# folder and produces a zip with src/ at the ROOT — then `unzip -d /content` yields
# /content/src and /content/kitty-qat never exists. Zip the folder itself:
Compress-Archive -Path kitty-qat -DestinationPath kitty-qat.zip -Force
# then drag kitty-qat.zip into Drive MyDrive in the browser
```

Alternative (cleaner, if the repo is ever pushed to a git remote): skip the zip and
`!git clone <url> /content/kitty-qat` in cell 1.

```python
# Cell 1
# Colab ships torchao 0.10.0; peft >= 0.21 hard-raises from is_torchao_available()
# during LoRA injection when torchao is present but < 0.16, even though we never
# use it. Removing it makes peft's probe return False cleanly. (Do NOT upgrade
# torchao: 0.16+ can drag a torch/CUDA bump that breaks the Colab stack.)
!pip uninstall -y torchao
!pip install -q peft datasets gguf
# -d creates the destination; the zip already nests entries under kitty-qat/
!unzip -q -o /content/drive/MyDrive/kitty-qat.zip -d /content
%cd /content/kitty-qat
!ls
```

```python
# Cell 2 — training. NOTE: 2,600 conversations yield ~4,225 windows → ~2,081
# optimizer steps at effective batch 2 (micro-batch 1 x accum 2), ~2-2.5 h.
# --batch 2 OOMs the L4 (22 GiB) at the first step: the reference
# chunk_gated_delta_rule fallback is the peak, and spec 4.3 forbids gradient
# checkpointing. Accumulation keeps the effective batch at 2 at half the peak.
# If the session cap is tight, lower the module constant DATA_CONV in
# scripts/train_qat_colab.py (there is no CLI flag for it).
# Adapter checkpoint exists only at the end.
!python scripts/train_qat_colab.py --out /content/qat-lora --batch 1 --accum 2
```

```python
# Cell 3 — merge + mtp carry + assert 1 (must be the LAST save: any later
# save_pretrained silently drops mtp.* again)
!python scripts/export_artifact.py --adapter /content/qat-lora --out /content/qat-merged
# expect: "total index keys: 441"
```

```python
# Cell 4 — copy both artifacts home: the adapter dir (Session B re-merges) and
# the merged artifact (local Task-9 convert)
!cp -r /content/qat-lora /content/qat-merged /content/drive/MyDrive/
```
