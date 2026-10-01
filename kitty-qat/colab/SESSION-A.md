# Session A — train, merge, carry mtp, save (~2.5-3 h, Colab L4)

Fresh Colab L4 runtime, Drive mounted at /content/drive.

**Cell 0 (before anything) — get the code onto the box. From the local machine
(PowerShell), upload `kitty-qat/` zipped into `MyDrive/kitty-qat.zip`:**

```powershell
cd C:\Users\tnmh\projects\local-llm
Compress-Archive -Path kitty-qat\src, kitty-qat\scripts -DestinationPath kitty-qat.zip -Force
# then drag kitty-qat.zip into Drive MyDrive in the browser
```

```python
# Cell 1
!pip install -q peft datasets gguf
!unzip -q /content/drive/MyDrive/kitty-qat.zip -d /content
%cd /content/kitty-qat
```

```python
# Cell 2 — training. NOTE: 2,600 conversations yield ~4,225 windows → ~2,050
# steps at batch 2 (~2-2.5 h). If the session cap is tight, pass a smaller
# DATA_CONV via -e edits. Adapter checkpoint exists only at the end.
!python scripts/train_qat_colab.py --out /content/qat-lora
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
