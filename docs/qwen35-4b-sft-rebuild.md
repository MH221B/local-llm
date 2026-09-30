# qwen35-4b-sft — M1 prompt pipeline output

Rebuild instructions for the Phase 1 / M1 dataset: **prompts only**. Every response is
written by the teacher in M2; nothing in `prompts/` contains a prebuilt assistant answer.

All commands run from the repo root, using the `dataset` conda env:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.<module> [args...]
```

`$env:HF_TOKEN` should be set: the source list streams from the Hub and anonymous
requests are rate-limited. New shells do not inherit an existing value, so read it from
the registry when running in a fresh session:

```powershell
$env:HF_TOKEN = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'   # cosmetic; the cache works without symlinks
```

## Layout

```
prompts/train.jsonl          3,000 prompt records  (M1 output; M2 input)
prompts/val.jsonl              200 prompt records
manifest.json                counts, caps, drop reasons, token/image share, source revisions
seeds/uncensored.jsonl      2,500 trainable uncensored prompts (spec §7.4)
eval/refusal.jsonl            500 quarantined prompts, refusal-rate eval (spec §8)
images/<ab>/<sha>.png        content-addressed store; path is <first two hex>/<sha>.png
```

`raw/` is **empty by design**. Ingestion streams from the Hub and materialises images
straight into `images/`; no intermediate raw copy is kept.

`prompts/` plus `verification/seeds.jsonl` are **M2's inputs**. M2 writes the
teacher-completed `train.jsonl` / `val.jsonl` back to `prompts/`.

## Rebuild, in order

### 1. Uncensored seed harvest (Task 10)

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.seeds --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.seeds --root datasets/qwen35-4b-sft
```

Writes `seeds/uncensored.jsonl` (2,500) and `eval/refusal.jsonl` (500). The split is
hash-selected, so it is stable across runs. The two files share one corpus, which is
intended: the refusal eval is in-distribution measurement (spec §8).

### 2. Student template (Task 15)

The render gate and the exact token counter both read the template the student model
ships. Start the server and extract it:

```powershell
Start-Process -FilePath "C:\Users\tnmh\projects\local-llm\llama-cpp\llama-server.exe" `
  -ArgumentList @("-m","C:\Users\tnmh\projects\local-llm\models\qwen35-mtp\Qwen3.5-4B-MTP-Heretic.i1-Q4_K_S.gguf",
                  "-ngl","99","-fa","on","-c","8192","-np","1","--port","8085") `
  -WindowStyle Hidden
Invoke-RestMethod http://127.0.0.1:8085/health
Invoke-RestMethod http://127.0.0.1:8085/props | Select-Object -ExpandProperty chat_template | Set-Content tools/dataset/student.jinja -Encoding utf8
```

`tools/dataset/student.jinja` is committed so the gate is reproducible without a server;
re-extract it if the student model changes.

### 3. Render gate (Task 15)

**Must pass before the full run.** It proves the template can render this data, that a
stored `Thinking` block survives a round trip, and that the generation prompt opens the
block so the student can learn to reason.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft-smoke/prompts/train.jsonl --template tools/dataset/student.jinja
```

Expected: `0 template failures`, `probe think block survived: True`,
`generation prompt opens thinking: True`, `VERDICT: think tags balanced`.

### 4. Prompt pipeline (Task 14, spec §9)

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --target-train 3000 --val-size 200 --root datasets/qwen35-4b-sft
```

`--dry-run` resolves the source table and prints each quota without any network access.
The live run needs the llama-server from step 2 for exact token counts; without it the
manifest records `token_counter: chars/4 estimate` instead of `exact`.

Roughly 25 minutes: the source list streams ~39,000 candidate rows, and each accepted
row makes one tokenise call per prompt.

### 5. Contamination guard (Task 16, spec §11.1)

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.contaminate --dataset datasets/qwen35-4b-sft/prompts/train.jsonl --extra-eval datasets/qwen35-4b-sft/eval/refusal.jsonl
```

Checks the training prompts against MATH-500, humanevalpack, gsm8k test and the four
Cauldron vision configs (spec §8).

**`collisions` is expected to be non-zero and is not a failure.** The refusal eval is a
held-out slice of a corpus that also trains, so near-duplicates against it are the
documented in-distribution limitation. What matters is that there are **no exact
collisions with the quarantined 500** — verify that separately if the count looks high:

```powershell
# no quarantined row may also appear in training
# (exact hash join between prompts/train.jsonl and eval/refusal.jsonl)
```

Add `--skip HuggingFaceM4/the_cauldron` when checking the vision holdout, since those
configs are the eval side for that pass.

### 6. Vision holdout (Task 17)

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.visionholdout --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.visionholdout --root datasets/qwen35-4b-sft
```

### 7. Unit-test seeds and verifier (Task 18)

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --smoke
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --root datasets/qwen35-4b-sft
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.verify
```

### 8. After M2 — calibration (Task 16, spec §3)

`calibrate.py` needs completed answers, so it runs on the teacher-written
`prompts/train.jsonl`, not the M1 prompt file.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.calibrate --dataset datasets/qwen35-4b-sft/prompts/train.jsonl --out datasets/qwen35-4b-sft/calibration.txt
```

## What the manifest records

- `run` — target sizes, what was written, the token-counter mode, and the caveats below.
- `sources` — per source: config, domain, cap, kept, license, **and the Hub revision**
  the data was read at, so a rebuild can be pinned to the same commit.
- `drops` — reject reasons, descending. Legitimate values: `duplicate_prompt`,
  `near_duplicate`, `empty_prompt`, `prompt_too_short`, `prompt_too_long`, `non_english`,
  `no_user_turn`, `duplicate_images`, `invalid`.
- `train` / `val` — examples, per-domain counts and shares, token totals and shares,
  image-bearing counts.

## Caveats

- **`NuminaMath-1.5` may contain GSM8K-derived problems**, and `gsm8k-test` is not fully
  clean as an eval set (spec §8). Recorded in `run.caveats`.
- **Token share is prompt-only.** The spec's §4 targets (~50% reasoning, ~25% coding) are
  measured over prompts *plus teacher responses*; M2's long reasoning traces are what
  produce that shape. Do not compare this manifest's `token_share` against §4 directly.
- **The image column is thinner than §12's ~12%.** M1 realises ~9%: Cauldron supplies
  4,000 of 32,000 cap, and at the training target the image configs are limited by quota
  rather than by cap. Image-aware dedup (see below) removed the larger loss.
- **Dedup is image-aware.** A row's identity is its normalised text *plus its image shas*,
  in both the exact key and the MinHash index. Image configs reuse one templated
  instruction across many rows with different images; deduping on text alone held
  Cauldron to ~3% yield.
- **Spec §8 dropped the RL/DPO gap as residual risk**, not closed by any M4 stage.

## Verifying a rebuild

Each module has a `__main__` smoke check with known-good output:

```powershell
foreach ($m in 'canonical','imgstore','textutil','seeds','sources','domains','dedup','filters','split','report','render','calibrate') {
  & "$HOME\miniconda3\envs\dataset\python.exe" -m "tools.dataset.$m"
}
```

Every one prints a benign `<frozen runpy>: RuntimeWarning` under `python -m`; it is not
a failure.
