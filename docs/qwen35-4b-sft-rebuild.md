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

MBPP seeds state the required function name and signature in the prompt — derived from the
reference solution — because MBPP's own `prompt` field never names the function its tests
call, which made every MBPP seed unsatisfiable (`mbpp|coding` measured **0.0**). This is why
`verification/seeds.jsonl` must be regenerated whenever `testsets.py` changes: the prompt text
is part of each row's cache key downstream.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --smoke
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --root datasets/qwen35-4b-sft
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.verify
```

### 8. After M2 — calibration (Task 16, spec §3)

`calibrate.py` needs completed answers, so it runs on M2's merged `train.jsonl`, not the M1
prompt file. It packs rows into 2-4k-token chunks and renders each row through
`tools/dataset/student.jinja`, so the chat special tokens reach the importance matrix — pass
`--parse-special` to `llama-imatrix` in Phase 3. Full detail in **M2 — teacher generation**
below.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.calibrate --dataset datasets/qwen35-4b-sft/train.jsonl --out datasets/qwen35-4b-sft/calibration.txt --chunks 200
```

## M2 — teacher generation (spec §10, §12)

M2 turns the M1 prompt columns into training rows: the teacher writes each assistant turn,
`verify.py` runs the seed oracles, `filters.py` applies the response predicates, and
`generate.py` merges the shards.

**Inputs → outputs.** `prompts/train.jsonl`, `prompts/val.jsonl`, `verification/seeds.jsonl`
and `prompts/trajectories.jsonl` are M2's **inputs**; `train.jsonl`, `val.jsonl`,
`manifest.json` and `calibration.txt` are its **outputs**. `m2/` holds the intermediates:
`single.jsonl`, `val.jsonl`, `trajectory.jsonl`, `simulated.jsonl`, `cache.jsonl` (one line
per completed request, so a re-run replays), `rejected.jsonl` (every drop **with its
completion**), and the per-phase `*.stats.json` pass rates.

The reasoning trace lives inside `messages[…].content` as a leading
`<think>…</think>` block, not in a separate `reasoning_content` field: the server splits it
out and `teacher._fold_reasoning` folds it back, because the spec, `student.jinja`, the
filters and the verifier all expect it in `content`. `student.jinja` splits it out again at
render time.

### Teacher server

The spec names `mradermacher/Ornith-1.5-9B-Abliterated-i1-GGUF` (spec §10); the local run used
**`MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf` + `.mmproj-BF16.gguf`** instead, so its
numbers are not directly comparable to a spec-model run.

```powershell
& "$PWD\llama-cpp\llama-server.exe" `
  -m models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf `
  --mmproj models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.mmproj-BF16.gguf `
  -ngl 99 -fa on -np 1 --port 8086 `
  --jinja --reasoning-format deepseek --reasoning-preserve `
  -c 131072 --reasoning-budget 6144
```

- `-c 131072` is cheap for this architecture: only the 8 full-attention layers carry a KV
  cache (~32 KiB/token at f16); the 24 linear-attention layers hold a constant-size state.
- `--reasoning-budget 6144` is a backstop that forces `</think>` and an answer instead of
  letting a trace run to the context limit. It is not from the model card.
- **Sampling** follows the teacher's parent model card (`ornith-ai/Ornith-1.5-9B`, which
  publishes `presence_penalty=1.5, min_p=0.0` for general tasks) rather than the spec alone:
  `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5`. Without the
  penalty the model loops on open-ended prompts — measured A/B at the same seed
  (`magpie-tools-2`, `865351961`): **21,038 tokens and still going at the 30-minute client
  timeout before, 174 tokens in 10.7 s after.**

### Generation

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --mode all `
  --root datasets/qwen35-4b-sft --limit 3 --val-limit 3 --multi-limit 3 --sim-limit 3 --magpie-limit 3
```

The local run is a **pipeline smoke, not the corpus** — the point is that every shape, oracle,
filter, merge and manifest step fires end to end. `--limit` is *per pool*, so the M3 handoff's
larger flags are the real target. Every call is cached, so a re-run only regenerates the rows
that are absent from `m2/cache.jsonl`.

### Measured numbers (local smoke, 2026-09-30)

| metric | value |
|---|---|
| seeded | 10 accepted of 12 |
| trajectories | 2 accepted of 3 |
| simulated | 3 accepted of 3 |
| merged | train 12, val 3, blended `pass_rate` **0.8333** |
| `train_final.by_domain` | coding 8, roleplay 2, uncensored 1, reasoning 1 |
| `train_final.token_share` | coding 0.864, roleplay 0.080, uncensored 0.043, reasoning 0.013 |
| `pass_rate_by_source` | gsm8k 1.0 · smoltalk 1.0 · in-the-wild 1.0 · the_cauldron 1.0 · teacher:uncensored 1.0 · teacher:tools 1.0 · **mbpp 0.6667** · CodeFeedback 0.5 |
| trace coverage | 9 of 9 single-turn train rows carry a `<think>` block |

`coding` dominates the token share because `verification/seeds.jsonl` is 100% coding and
oracle-bearing, so read `pass_rate_by_source` rather than the blended `pass_rate` when
setting M3 caps.

`mbpp|coding` first measured **0.0** and that was a false positive by construction: MBPP's
`prompt` field is a vague one-liner while its tests call the reference function by *name and
arity*, which the seed never stated — two of three sampled rows had implemented the correct
behaviour under a self-chosen name. `testsets.py` now derives the signature from the reference
solution and states it in the prompt; MBPP sits at **0.6667**, and the one remaining drop is
genuine (`AssertionError`: the model returned the right elements in the wrong container type).
Regenerating `verification/seeds.jsonl` is required for that fix, because the prompt text is
part of each row's cache key.

Drop reasons this run (every drop is written to `m2/rejected.jsonl` with its completion):

| reason | stage | rows |
|---|---|---|
| `verify_failed` | seeded | 1 (`mbpp-473`, genuine) |
| `teacher_error` | seeded | 1 (a cached skip: `codefeedback-239`) |
| `turn_structure` | trajectory | 1 |

### Calibration

Runs on M2's output, not M1's prompts:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.calibrate --dataset datasets/qwen35-4b-sft/train.jsonl --out datasets/qwen35-4b-sft/calibration.txt --chunks 200
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
- `m2` (added by `generate.py --mode all`) — `seeded` / `trajectory` / `simulated` stats with
  their `drops` and `pass_rate_by_source`, plus `pass_rate` and `train_written` /
  `val_written`. Legitimate M2 drop reasons: `verify_failed`, `too_short`, `too_long`,
  `unbalanced_think_tags`, `empty_assistant`, `turn_structure`, `invalid`, `teacher_error`.
  Response-quality reasons (`refusal`, `looping`, `non_english`) were removed during M2 Task 15
  because they false-fired on the reasoning trace; `rejected.jsonl` carries the completions.

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

Most modules have a `__main__` smoke check with known-good output. `render` is the exception —
it requires `--dataset` and `--template`, so it is exercised by step 3's gate command and by M2
Task 15 step 6 instead.

```powershell
foreach ($m in 'canonical','imgstore','textutil','seeds','sources','domains','dedup','filters','split','report','calibrate') {
  & "$HOME\miniconda3\envs\dataset\python.exe" -m "tools.dataset.$m"
}
# render needs arguments:
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft/train.jsonl --template tools/dataset/student.jinja --sample 200
```

Every one prints a benign `<frozen runpy>: RuntimeWarning` under `python -m`; it is not
a failure.
