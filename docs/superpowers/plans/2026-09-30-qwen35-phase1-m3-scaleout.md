# Phase 1 / M3 — Scale-out: Colab teacher, batch driver, difficulty filter Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Phase 1's full teacher-distillation pass at the spec's target size: 25,000 train and 1,000 val examples across four domains, with the teacher served on a Colab Pro L4 and the local driver driving it concurrently.

**Architecture:** Colab is a *stateless teacher*. An L4 runs `vllm serve` over the same teacher GGUF M2 used, exposed to your machine by a `cloudflared` quick tunnel. Your machine keeps everything durable: the prompt corpus, the content-addressed image store, the unit-test verifier, the sample audit, and the append-only resume cache. It reaches the teacher through `--base-url`. A bounded worker pool in `generate.py` turns the sequential M2 driver into one that uses vLLM's continuous batching. Every teacher call is cached by request hash, so a Colab session dying costs at most the calls in flight.

**Engine caveat.** Whether vLLM can serve this model at all is unverified, and PIPELINE.md names vLLM's Qwen3.5 support as Phase 1's main risk. Task 1 Step 6 tests both engines before any bulk work and requires the *image* smoke to pass, because spec §7.6 needs a vision teacher: a text-only server is a failure, not a shortcut. If the accepted engine turns out to be `llama.cpp`, re-measure throughput before trusting Task 8's budget.

**Tech Stack:** Python 3.12 (`miniconda` env `dataset`), stdlib only inside `tools/dataset/` (`urllib`, `threading`, `concurrent.futures`); vLLM + `vllm-gguf-plugin` on Colab; `cloudflared`; `llama.cpp` `llama-server` as the fallback server; `datasets`, `pillow`, `jinja2` and `huggingface_hub` on the ingest side (the last one is already imported by `pipeline.py`).

**Testing:** Per-module `__main__` smokes with known-good output, matching the existing convention — `pytest` is not installed and is not being added. New modules get a smoke block driven by a `FakeTeacher`, so every code task is verifiable offline. No test-first steps: the user has not opted into TDD and the repo's established pattern is smoke-and-measure. Server-dependent checks are confined to Task 1 Step 6 (the live Colab server and its four canaries), Task 3 Step 7 (batching), Task 5 Step 6 (the filter against live data) and Task 8 (the full run).

---

## Prerequisites and dependencies

**M2 Task 15 must be accepted before Task 6.** Task 6 re-anchors the source caps on M2's *measured* `pass_rate_by_source`; without an accepted `manifest.json` there is no measurement to anchor on.

**Tasks 1–5 can proceed while M2 runs, but Task 4's signature change is load-bearing.** Task 4 adds a `schema_limit` parameter to `magpie.run`. Its default of `0` is what keeps the existing `generate.run_all` call site working, so an M2 restart part-way still runs. If you change that default, land Tasks 1–5 only after M2 is accepted, or M2 can no longer be resumed.

**Before starting Task 8, the teacher GGUF pair must live somewhere Colab can read quickly.** The two files are already on disk:

```
C:\Users\tnmh\projects\local-llm\models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf
C:\Users\tnmh\projects\local-llm\models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.mmproj-BF16.gguf
```

Task 1 uploads them to Google Drive once; every later session stages them from Drive.

**Non-goals, stated so they are not quietly added:**

- **No roleplay/trajectory judge — a deliberate, recorded deviation.** Be honest about the citation: the spec does *not* mark it optional. §10.1 says "Roleplay, chat-style coding, and uncensored completions have no mechanical check; they pass filters and review only", and §10.2 says roleplay trajectories are "judge-scored plus sample-audited". The softening came from the M2 handoff ("only if a roleplay/multi-turn quality signal is wanted"), and this plan knowingly takes the softer reading. A judge that is the same 9B teacher scores text it wrote and is correlated with the very bugs it is meant to catch; a credible judge needs a *second*, independent served model plus threshold calibration against human labels. Task 8 substitutes the spec's *other* mechanism — a measured sample audit (§10.1, §11.7) of the un-oracled columns — and the deviation is stated in "Deferred to a later milestone", in the Open risks table and in the Task 8 guide section, not hidden.
- **No RL/DPO stage.** The spec accepts the multi-turn robustness gap as residual risk (§13, §14).
- **No change to the teacher weights.** M3 serves the same GGUF M2 did, so M2's measured pass rates remain the correct anchor.
- **No new third-party dependency** in `tools/dataset/`. Concurrency uses `concurrent.futures` from the standard library.

---

## File structure

**Create**

| Path | Responsibility |
|---|---|
| `tools/colab/serve_teacher.py` | The whole Colab side as one runnable script: check GPU, mount Drive, install vLLM, stage weights, serve, run the four canaries (text, image, reasoning, tools), expose the tunnel, keep alive. |
| `tools/colab/README.md` | Colab primer for a first-time user: runtimes, GPU selection, sessions and their limits, Drive, the tunnel, stopping billing, troubleshooting. |
| `tools/dataset/prompts/magpie_toolschema.md` | Invention prompt conditioned on a tool schema (spec §10.2's "invented from a `tools` schema"). |
| `tools/dataset/anchor.py` | Read M2's measured pass rates out of `manifest.json`, scale the source caps by `1/pass_rate`, print the re-anchored table. Reporting only — never edits a file. |
| `tools/dataset/audit.py` | Draw a stratified sample across all four domains — flagging the un-oracled ones — and render it to Markdown for the human spot-check (spec §10.1 and §11.7; §11.3 is the validator, not this). |

**Modify**

| Path | Change |
|---|---|
| `tools/dataset/gencache.py` | Thread-safe `put`/`get`; first-write-wins so a re-run never grows the file. |
| `tools/dataset/imgstore.py` | Atomic image writes, so a worker killed mid-write cannot leave a torn file at a content-addressed path. |
| `tools/dataset/generate.py` | `--concurrency`, `--api-key`, `--cache`, `--timeout`, `--retries`; `rollout` parameter on `generate_one`; the four pool limits default to `None` so `0` means "attempt none" and omitted means "every candidate"; new `run_difficulty` which writes both the survivors and the list of judged ids; `--mode difficulty`; merge filters against the judged ids; observable progress. `root` and every `_record_reject` call survive the concurrency rewrite. |
| `tools/dataset/magpie.py` | Tool-schema invention; content-derived ids; dedup at scale; a `schema_limit` defaulting to 0 so existing callers keep working; `--api-key`. |
| `tools/dataset/teacher.py` | Optional `api_key` (Bearer header) so the public tunnel is not trivially abusable. |
| `tools/dataset/sources.py` | Re-anchored caps; `SOURCE` rows carry the M2 pass rate they were anchored on. |
| `docs/qwen35-4b-sft-rebuild.md` | An `## M3 — scale-out` section: the Colab launch and which engine served it, the driver command actually used, measured pass rates, the difficulty-filter drops, the audit result, the uncensored shortfall, and the deferred judge. This is the tracked M1/M2 guide, and it is the only place corpus prose can live: `.gitignore` line 8 is `datasets/`, so a `datasets/qwen35-4b-sft/README.md` is untrackable and `git add` would refuse it. |

Each helper answers one question: `anchor.py` sizes the pools, `audit.py` renders a review sample, `magpie.py` invents prompts. `generate.py` remains the only module that orchestrates a run, as it was in M2.

---

### Task 1: Serve the teacher on Colab Pro

**Files:**
- Create: `tools/colab/serve_teacher.py`
- Create: `tools/colab/README.md`

This is the only task whose verification needs the internet and a Colab account. Everything after it is local.

- [ ] **Step 1: Write the Colab primer**

`tools/colab/README.md` exists so the first Colab session is not guesswork. Write it with this content:

````markdown
# Running the M3 teacher on Google Colab Pro

Colab gives you a temporarily-rented Linux machine with a GPU, driven from a notebook in
your browser. The notebook is just a list of cells; each cell is Python (or a `!`-prefixed
shell command) you run by clicking the play button or pressing Shift+Enter. `Runtime` in
the menu bar is where you pick hardware and stop the machine.

## One-time setup

1. Sign in at <https://colab.research.google.com> and pick **New notebook**.
2. Menu: **Runtime → Change runtime type**. Set **Hardware accelerator** to **L4 GPU**.
   Under Colab Pro, L4 is billed in compute units; an A100 costs roughly 3x as much and is
   not needed for a 9B model. Click **Save**.
3. Confirm you are on Pro: **Runtime** shows a "Resources" / compute-units panel. If **L4
   GPU** is not offered at all, stop here — no engine choice fixes a missing hardware tier,
   and the whole plan assumes an L4. (The `llama.cpp` fallback is about GGUF compatibility,
   not about GPU availability.)
4. Put the teacher weights in Drive, once. In a normal browser tab, open
   <https://drive.google.com>, create a folder called `models`, and upload both files from
   `C:\Users\tnmh\projects\local-llm\models\`:
   - `MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf`
   - `MiMo-Ornith-9B-AGSI-Abliterated-HQ.mmproj-BF16.gguf`

   The upload is about 6 GB (a ~5.1 GB Q4_K_S model plus a ~0.9 GB BF16 projector) and
   happens once. Every session after this reads from Drive.

## Every session

1. **Runtime → Change runtime type** and re-select **L4 GPU** (Colab resets this between
   sessions).
2. **Mount Drive in its own cell before running the launcher.** `google.colab.drive.mount`
   talks to the browser through the notebook kernel, so it only works in a `python` cell —
   in the `!python` subprocess below it fails with `AttributeError: 'NoneType' object has no
   attribute 'kernel'`, which reads like a Colab bug. Approve the prompt.

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
3. Run the launcher cell. The engine is `llama-cpp` and the tokenizer repo is fixed for this
   teacher — Task 1 Step 5 read it out of the GGUF. vLLM cannot serve this checkpoint; see
   Troubleshooting.

   ```
   !python /content/serve_teacher.py --engine llama-cpp --tokenizer XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
   ```

   Copy the `trycloudflare.com` URL it prints. That URL is the teacher's address for this
   session and **changes every session**.
4. On your own machine, set the URL and the key, then drive the run:

   ```powershell
   $env:TEACHER_URL = "PASTE_THE_URL_HERE"
   $env:TEACHER_API_KEY = "PASTE_THE_KEY_HERE"
   ```

5. When you are done, **Runtime → Disconnect and delete runtime**, or the machine keeps
   burning compute units.

## What you need to know about sessions

- A notebook can run for at most **12 hours**; an **idle** runtime is deleted after about
  **90 minutes**. "Idle" means no code executing, so a long-running cell is what keeps it
  alive.
- Runtime deletion is routine, not a failure. The driver caches every completed teacher
  call, so a lost session costs at most the handful of calls in flight. Re-connect, get a
  new URL, run the same driver command again — it replays the cache and continues.
- **Closing the browser tab may kill the session.** If you want to leave it unattended, keep
  the tab open, on a machine that will not sleep.
- The tunnel URL is **public**. Anyone who learns it can spend your GPU. This is why the
  launcher serves with an API key and why the driver sends it.
- Colab's disk is wiped when the runtime is deleted. Nothing that matters lives on it: the
  weights come from Drive and the *outputs* are written on your machine.

## Troubleshooting

- The launcher dies with `AttributeError: 'NoneType' object has no attribute 'kernel'` —
  Drive is not mounted. Mount it in a `python` cell first (step 2 above); the mount call
  cannot run inside the `!python` subprocess.
- `CUDA out of memory` on load — another runtime is still alive. **Runtime → Manage
  sessions**, terminate the others, retry.
- The launcher exits with `Unknown gguf model_type: qwen3_5` — vLLM's GGUF plugin cannot map
  this checkpoint. `vllm-gguf-plugin` 0.0.5 is the newest release and upstream issue
  vllm-project/vllm#38122 is still open; supplying the missing name mapping only reaches a
  second wall, because this checkpoint's vision config carries `depth` where the loader reads
  `num_hidden_layers`. Vision is not optional (spec section 7.6 is 12% of the corpus), so that
  path is out. Use `--engine llama-cpp`: same weights, same projector, same chat template, and
  it is the engine M2 actually ran.
- `-c`/`-np` are one trade-off and llama-server splits the context evenly across slots, so
  `n_ctx_slot = ctx_size / slots`. Throughput is roughly 45 t/s per slot, so slots is the
  speed lever and per-row context is its price. Do not let a slot fall below ~8192: the
  reasoning backstop alone is 6144 tokens, and a row that overruns its slot is truncated
  rather than rejected, which is a silent quality loss. Defaults are 262144 / 16 (16384 per
  row, ~700 t/s); `--slots 32` doubles that and halves the context to 8192 per row.
- The launcher exits with `image canary failed` — the engine cannot see images. On vLLM that
  means the projector was not picked up; re-run with `--engine llama-cpp`. Do not continue
  with a text-only server.
- The launcher exits with `tools canary: FAIL` — the request's `tools` schema is not reaching
  the chat template. On vLLM, add `--enable-auto-tool-choice --tool-call-parser hermes` to
  the serve command; if that does not fix it, re-run with `--engine llama-cpp`. Do not
  continue without it: every trajectory and simulated row is tool-carrying, so those columns
  would come back empty rather than wrong.
- The launcher exits with `reasoning canary: FAIL` — a thinking-enabled call returned no
  trace. The corpus stores the CoT verbatim inside `content` (spec section 5.2), so the
  template's thinking branch has to be reachable before anything is generated.
- The tunnel cell prints no URL after ~30 s — re-run it. If it still fails, restart the
  runtime; Cloudflare's quick tunnel occasionally refuses a session.
- The driver reports timeouts to the teacher — the session was pruned. Get a new URL and
  re-run; the cache replays.

## Cost

The spec (§12) budgets **5–8 hours and roughly 20–30 compute units** for M3 re-anchored on
M2's measured throughput. That figure was written for an FP8 teacher on an L4. This plan
serves a Q4_K_S GGUF, so the token rate differs and the number has to be re-measured rather
than trusted. If the accepted engine is `llama.cpp`, expect materially less than vLLM's
batching throughput. Watch the compute-units panel and stop the runtime when you are not
generating.
````

- [ ] **Step 2: Write the launcher script**

`tools/colab/serve_teacher.py` is one file so a Colab beginner has one thing to run. Write it with this content:

```python
"""Serve the M3 teacher on a Colab GPU and expose it to the local driver.

Run this file in a Colab notebook (see `README.md` next to it). It mounts Drive, stages the
two teacher GGUFs, serves them with vLLM, opens a cloudflared quick tunnel, and then keeps
the session alive. The tunnel URL it prints is the value of `--base-url` on your machine.

Spec section 10: the teacher is the same Ornith-family GGUF M2 used, so M2's measured pass
rates remain the correct anchor for M3. Nothing here regenerates training data.

Deliberate fallback: if vLLM cannot map this GGUF's architecture, or cannot see images
through it, `--engine llama-cpp` serves the identical weights and projector through
llama.cpp. It is slower because it has no continuous batching, but it is the same teacher,
so rows generated under it are comparable. Only throughput differs, and Task 8's cost
estimate must be re-measured if the fallback is what runs.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

DRIVE_MODELS = Path("/content/drive/MyDrive/models")
LOCAL_MODELS = Path("/content/models")
TEXT_GGUF = "MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf"
MMPROJ_GGUF = "MiMo-Ornith-9B-AGSI-Abliterated-HQ.mmproj-BF16.gguf"
PORT = 8000
SERVED_NAME = "ornith-teacher"


def sh(cmd: str, **kw) -> int:
    print(f"+ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True, **kw)


def check_gpu() -> None:
    if sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") != 0:
        raise SystemExit("no NVIDIA GPU: set Runtime -> Change runtime type -> L4 GPU")
    try:
        import torch
        print("torch:", torch.__version__, "| cuda:", torch.cuda.get_device_name(0), flush=True)
    except Exception as exc:                                  # torch not installed yet
        print("torch not importable yet:", exc, flush=True)


def mount_drive() -> None:
    """Fail clearly when Drive is not mounted yet.

    `google.colab.drive.mount` talks to the browser through the *kernel's* message channel,
    so it cannot run in this `!python` subprocess: Colab raises
    `AttributeError: 'NoneType' object has no attribute 'kernel'`, which reads like a Colab
    bug rather than a usage error. The launcher cell mounts Drive first; see README.md.
    """
    if Path("/content/drive/MyDrive").exists():
        print("Drive already mounted.", flush=True)
        return
    raise SystemExit(
        "Drive is not mounted. Mount it in a notebook cell first, then re-run this file:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')")


def stage_weights() -> None:
    LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    for name in (TEXT_GGUF, MMPROJ_GGUF):
        src, dst = DRIVE_MODELS / name, LOCAL_MODELS / name
        if not src.exists():
            raise SystemExit(f"missing {src} — upload it to Drive once; see README.md")
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"already staged: {name}", flush=True)
            continue
        print(f"staging {name} ({src.stat().st_size / 1e9:.2f} GB)", flush=True)
        sh(f"cp -f {shlex.quote(str(src))} {shlex.quote(str(dst))}")


def serve_vllm(tokenizer: str) -> subprocess.Popen:
    """Serve the GGUF with vLLM.

    The projector is **not** a flag here: vLLM's GGUF loader looks for `mmproj*.gguf` in the
    same directory as the model, which is why `stage_weights` puts both files in one place.
    That auto-detection is unverified for this model, and the image canary in `smoke()` is
    what decides whether it worked.

    `--max-num-seqs` is the ceiling on concurrently running sequences, and on this model that
    ceiling is throughput: decode is weight-bound, so aggregate tokens/second is roughly
    `steps_per_second x batch`. The driver's `--concurrency` must not exceed it, or the server
    queues what the driver sends and phase 1 measures the lower number. Raise both together —
    32 to start, 48 if the KV budget allows.
    """
    sh("pip install -q --upgrade vllm vllm-gguf-plugin")
    cmd = (f"vllm serve {shlex.quote(str(LOCAL_MODELS / TEXT_GGUF))} "
           f"--served-model-name {SERVED_NAME} --tokenizer {shlex.quote(tokenizer)} "
           f"--host 127.0.0.1 --port {PORT} --max-model-len 32768 "
           f"--gpu-memory-utilization 0.90 --max-num-seqs 32 "
           f"--api-key {'$TEACHER_API_KEY'}")
    print("launching vLLM (first load takes several minutes)", flush=True)
    return subprocess.Popen(cmd, shell=True, env=os.environ)


def serve_llama_cpp(ctx_size: int, slots: int) -> subprocess.Popen:
    """Fallback: same weights and projector, no continuous batching. Spec section 10.

    The flags after `-np` are not optional. `--jinja` is what makes llama-server apply the
    GGUF's own chat template, which is where `enable_thinking` and a request's `tools` schema
    are injected; without it the reasoning and tools canaries cannot pass, and the rows would
    come back shaped differently from M2's. The three `--reasoning-*` flags are M2's measured
    serving line, kept identical so the pass rates stay comparable across engines.

    `-c` and `-np` are one trade-off, because the server splits the context evenly across
    slots: `n_ctx_slot = ctx_size / slots`. Decode is weight-bound, so aggregate throughput is
    roughly `tokens_per_second_per_slot x slots`, which makes slots the throughput lever and
    the per-row context its price. Measured on the L4 with `-c 32768 -np 8`: 44.7 t/s per slot
    (~350 t/s aggregate) and only **4096 tokens per row** — less than the 6144-token reasoning
    backstop below, so long rows would have truncated silently. The defaults give 16384 per
    row; `--slots 32` roughly doubles throughput and halves that to 8192.
    """
    sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp "
       "|| true")
    sh("cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=ON "
       "-DLLAMA_CURL=OFF > /dev/null && "
       "cmake --build /content/llama.cpp/build --target llama-server -j 8 > /dev/null")
    cmd = (f"/content/llama.cpp/build/bin/llama-server "
           f"-m {shlex.quote(str(LOCAL_MODELS / TEXT_GGUF))} "
           f"--mmproj {shlex.quote(str(LOCAL_MODELS / MMPROJ_GGUF))} "
           f"--alias {SERVED_NAME} --host 127.0.0.1 --port {PORT} "
           f"--api-key $TEACHER_API_KEY -ngl 99 -fa on "
           f"-c {ctx_size} -np {slots} "
           f"--jinja --reasoning-format deepseek --reasoning-preserve "
           f"--reasoning-budget 6144")
    print(f"launching llama.cpp: {slots} slots x {ctx_size // slots} ctx tokens "
          f"(GPU build takes ~15 minutes the first time)", flush=True)
    return subprocess.Popen(cmd, shell=True, env=os.environ)


def wait_for_health(proc: subprocess.Popen, timeout: int = 1800) -> None:
    """Poll `/health` until it returns 200.

    The two engines differ here. llama-server answers `{"status": "ok"}`; vLLM answers an
    empty 200. The status code is the only thing they share, so that is all this checks —
    parsing the body makes the probe succeed on one engine and hang until timeout on the
    other. With `--api-key` set, both engines require the Bearer header even on `/health`.
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/health",
        headers={"Authorization": f"Bearer {os.environ['TEACHER_API_KEY']}"})
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("the server exited during startup; scroll up for the error")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print("server is up", flush=True)
                    return
        except urllib.error.HTTPError as exc:
            # Not a retryable condition: a rejected key will never start working.
            if exc.code in (401, 403):
                raise SystemExit(f"the server rejected TEACHER_API_KEY ({exc.code})")
            time.sleep(5)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    raise SystemExit(f"server did not become healthy within {timeout}s")


def smoke() -> bool:
    """Text, image, reasoning and tools canary, exactly as the driver makes them.

    Four checks, and all four are hard requirements rather than warnings:

    - **text** proves the OpenAI surface answers at all;
    - **image** proves the projector was picked up (spec section 7.6: a teacher that cannot
      see silently deletes the image column);
    - **reasoning** proves a thinking-enabled call comes back with the CoT, which the corpus
      stores verbatim inside `content` (spec section 5.2);
    - **tools** proves a request carrying a `tools` schema reaches the chat template, which
      spec section 14 lists as unverified and which every trajectory and simulated row
      depends on.

    M2 served the reasoning and tools paths through llama.cpp's `--jinja`, so both are
    re-tested here rather than assumed to survive the engine change.
    """
    import base64
    import io
    import json
    import urllib.request

    from PIL import Image

    def post(messages: list[dict], thinking: bool, max_tokens: int = 32,
             tools: list[dict] | None = None) -> dict:
        payload = {"model": SERVED_NAME, "messages": messages, "temperature": 0.6,
                   "max_tokens": max_tokens, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": thinking}}
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['TEACHER_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]

    text = post([{"role": "user", "content": "Reply with the single word: ok"}], False)
    print("text reply:", repr((text.get("content") or "").strip())[:60], flush=True)

    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (200, 30, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = post([{"role": "user", "content": [
        {"type": "text", "text": "What single colour fills this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}], False)
    reply = (img.get("content") or "").strip()
    image_ok = "red" in reply.lower()
    print("image reply:", repr(reply)[:60], flush=True)
    print("image canary:", "PASS" if image_ok else "FAIL", flush=True)

    think = post([{"role": "user", "content": "What is 17 times 23? Think it through."}],
                 True, max_tokens=512)
    blob = f"{think.get('content') or ''}\n{think.get('reasoning_content') or ''}"
    trace_ok = "<think>" in blob or "</think>" in blob or bool(think.get("reasoning_content"))
    print("thinking reply:", repr(blob.strip())[:60], flush=True)
    print("reasoning canary:", "PASS" if trace_ok else "FAIL", flush=True)

    schema = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    tools_reply = post([{"role": "user", "content": "What is the weather in Lima right now?"}],
                       False, max_tokens=512, tools=schema)
    tool_blob = (f"{tools_reply.get('content') or ''}"
                 f"{tools_reply.get('tool_calls') or ''}")
    tools_ok = "get_weather" in tool_blob
    print("tools reply:", repr(tool_blob.strip())[:60], flush=True)
    print("tools canary:", "PASS" if tools_ok else "FAIL", flush=True)
    return image_ok and trace_ok and tools_ok


def open_tunnel() -> subprocess.Popen:
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && "
       "chmod +x /usr/local/bin/cloudflared")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        match = pattern.search(line)
        if match:
            url = match.group(0)
            print("=" * 72, flush=True)
            print(f"TEACHER_URL={url}", flush=True)
            print(f"TEACHER_API_KEY={os.environ['TEACHER_API_KEY']}", flush=True)
            print("On your machine, in the repo root:", flush=True)
            print(f'  $env:TEACHER_URL = "{url}"', flush=True)
            print(f'  $env:TEACHER_API_KEY = "{os.environ["TEACHER_API_KEY"]}"', flush=True)
            print("=" * 72, flush=True)
            return proc
    raise SystemExit("cloudflared printed no tunnel URL; re-run this cell")


def main() -> int:
    ap = argparse.ArgumentParser(description="serve the M3 teacher on Colab")
    ap.add_argument("--engine", choices=["vllm", "llama-cpp"], default="vllm")
    ap.add_argument("--ctx-size", type=int, default=262144,
                    help="llama.cpp only: total context, divided evenly across --slots. "
                         "262144 is ~8 GiB of KV at 32 KiB/token, which is what the L4 holds "
                         "alongside the 5.34 GB of weights. vLLM ignores this and uses its "
                         "own --max-model-len.")
    ap.add_argument("--slots", type=int, default=16,
                    help="llama.cpp only: parallel sequences. Throughput is ~45 t/s x slots; "
                         "each slot gets ctx-size/slots tokens. 16 -> 16384 per row, "
                         "32 -> 8192 per row at about twice the speed.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo whose tokenizer matches the GGUF; see Task 1 Step 5. vLLM "
                         "needs it because converting a tokenizer out of a GGUF is lossy. "
                         "llama.cpp reads the vocab out of the GGUF and ignores this.")
    ap.add_argument("--api-key", default=None,
                    help="defaults to a random per-session key printed with the URL")
    args = ap.parse_args()

    if args.engine == "vllm" and not args.tokenizer:
        raise SystemExit("--tokenizer is required for --engine vllm; see Task 1 Step 5")

    if not args.api_key:
        import secrets
        args.api_key = secrets.token_urlsafe(24)
    os.environ["TEACHER_API_KEY"] = args.api_key

    check_gpu()
    mount_drive()
    stage_weights()

    proc = (serve_vllm(args.tokenizer) if args.engine == "vllm"
            else serve_llama_cpp(args.ctx_size, args.slots))
    try:
        wait_for_health(proc)
        if not smoke():
            raise SystemExit(
                f"a canary failed on {args.engine}; the line above names which. Text, image, "
                "reasoning and tools are all required (spec sections 5.2, 7.6, 10.2). Re-run "
                "with --engine llama-cpp; if the image canary still fails there, stop and "
                "investigate the projector rather than generating anything.")
        open_tunnel()
        print("running. This cell must stay alive; Runtime -> Disconnect to stop billing.",
              flush=True)
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the script is syntactically valid and imports cleanly**

`/content` and `google.colab` do not exist locally, so this checks syntax and the module-level imports the script does at import time (everything else is imported inside functions, deliberately, so this check works):

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import ast, pathlib; ast.parse(pathlib.Path('tools/colab/serve_teacher.py').read_text(encoding='utf-8')); print('parses ok')"
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import sys; sys.path.insert(0, 'tools/colab'); import serve_teacher as s; print('imports ok:', s.SERVED_NAME, s.PORT, s.TEXT_GGUF)"
```

Expected: `parses ok` then `imports ok: ornith-teacher 8000 MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf`.

- [ ] **Step 4: Check the file for angle-bracket corruption**

The write tool has corrupted `<`/`>` in this project before, so confirm the two load-bearing pairs are intact. Do **not** count them with `t.count(chr(60))`: the file legitimately contains 14 `->` return annotations and four bare `<` comparisons, so an ordinal count tells you nothing. Check the tags themselves:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import pathlib; t=pathlib.Path('tools/colab/serve_teacher.py').read_text(encoding='utf-8'); assert '<think>' in t and '</think>' in t; assert 'Bearer ' in t and 'trycloudflare' in t; print('markers intact')"
```

Expected: `markers intact`, and nothing else. If the assert fires, one of the four literals was substituted — re-read the file and fix it before committing.

- [ ] **Step 5: Discover the tokenizer repo**

vLLM needs a Hugging Face tokenizer for this GGUF, and guessing wrong shows up as garbled output rather than a clean error. Read it out of the GGUF's own metadata with `gguf`, which is a small pure-Python package. The metadata lives inside the file, so run this against the local copy — it needs no Colab session and can be done before Step 6:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m pip install -q gguf
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import gguf; r=gguf.GGUFReader('models/MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf'); [print(k, '=', bytes(f.parts[f.data[0]]).decode() if (f:=r.get_field(k)) else None) for k in ('general.base_model.0.repo_url','general.base_model.0.name','general.base_model.0.version','general.architecture')]"
```

The same two lines work on Colab with `/content/models/...` once Step 6 has staged the file, which is the fallback if the local copy is unavailable.

Expected: a `repo_url` naming the teacher's base repository, and `general.architecture` reading `qwen35`. **Record the `repo_url`** — that string is the `--tokenizer` value for every later launch.

Measured on the local GGUF when Task 1 was first executed:

```
general.architecture               = qwen35
general.base_model.0.repo_url      = https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
general.base_model.0.name          = MiMo V2.6 Distill Qwen 9B
tokenizer.ggml.tokens (vocab size) = 248320
```

so the value is the *repo id* `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, not the full URL. That repository is public (`gated: false`) and its `config.json` reports `vocab_size: 248320`, which is the match the smoke depends on — no `HF_TOKEN` is needed to fetch it.

Note the base is **not** the `ornith-ai/Ornith-1.5-9B` an earlier draft guessed: that is a different family with a different chat template and vocabulary, and vLLM would accept it and then produce garbled output rather than a clean error. If `repo_url` is ever absent, fall back to `Qwen/Qwen3.5-9B` (the GGUF's `base_model` tag) and confirm the vocabulary is 248,320 before trusting the run.

- [ ] **Step 6: Run it on Colab and prove the endpoint**

This is the task's real acceptance. Do it once, now, so the risk is found here rather than in Task 8.

The cell below fetches the script from `main` over HTTPS, and the repo is public, so that works — but only once the file has been pushed. Step 7 is where the commit lives, so run Step 7 first. The alternative is to skip the `curl` line and upload the file to `/content/serve_teacher.py` through Colab's file browser (or paste it into a cell); everything after that is identical. Until the push lands, the raw URL is a 404, which is the only failure mode in this step.

1. Open Colab and set the runtime to an L4 GPU. **Mount Drive in its own `python` cell first.** `google.colab.drive.mount` talks to the browser through the notebook kernel, so it cannot run inside the `!python` subprocess below — there it dies with `AttributeError: 'NoneType' object has no attribute 'kernel'`, which reads like a Colab bug rather than a usage error. Approve the prompt it raises.

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

2. Then run the launcher cell, which fetches the script from `main` and serves the teacher:

   ```
   !pip install -q pillow
   !curl -fsSL https://raw.githubusercontent.com/MH221B/local-llm/main/tools/colab/serve_teacher.py -o /content/serve_teacher.py
   !python /content/serve_teacher.py --tokenizer XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
   ```

Expected: the GPU line names an L4; `staging` lines for both GGUFs; a vLLM startup log; `server is up`; a short text reply; an image reply containing `red` and `image canary: PASS`; `reasoning canary: PASS` from a thinking call; `tools canary: PASS` from a call carrying `get_weather`; then the boxed `TEACHER_URL=` / `TEACHER_API_KEY=` banner. The tunnel only opens if all three PASS lines appear — `smoke()` returns False otherwise, so a failure here is loud rather than silent.

The three PASS lines are what matter, and each one covers a column:

- `image canary: FAIL` — the engine cannot see. spec §7.6's image column is roughly 12% of the corpus, and a teacher that cannot see would silently produce zero image rows, so **a text-only endpoint is not an acceptable substitute**.
- `reasoning canary: FAIL` — thinking-enabled calls return no trace. Every reasoning row and every `<think>` block in the corpus comes from here (spec §5.2), so this is not a cosmetic difference.
- `tools canary: FAIL` — the `tools` schema is not reaching the template. The trajectory and simulated passes are entirely tool-carrying (spec §10.2), so those two columns would come back empty rather than noisy.

Each failure prints which engine it came from and stops before the tunnel, so the run cannot start on a crippled endpoint.

3. From your machine, check the endpoint directly. Do **not** use `tools.dataset.teacher`'s `__main__` smoke here: it calls `/props`, which is a llama-server endpoint that vLLM does not implement.

```powershell
$env:TEACHER_URL = "PASTE_THE_URL_HERE"
$env:TEACHER_API_KEY = "PASTE_THE_KEY_HERE"
$body = @{ model = "ornith-teacher"; max_tokens = 32; temperature = 0.6
           messages = @(@{ role = "user"; content = "Reply with the single word: ok" }) } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri "$env:TEACHER_URL/v1/chat/completions" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $env:TEACHER_API_KEY" } -Body $body | Select-Object -ExpandProperty choices | Select-Object -ExpandProperty message | Select-Object -ExpandProperty content
```

Expected: a short reply such as `ok`. That proves the tunnel, the API key and the OpenAI-compatible surface work from your machine — but **not** that a real generation survives them, because a 32-token reply returns in a second. Cloudflare's edge abandons a proxied request that sends no bytes for about 100 seconds, so the second check is the one that matters:

```powershell
$body = @{ model = "ornith-teacher"; max_tokens = 3000; temperature = 0.6
           messages = @(@{ role = "user"; content = "Write a detailed 600-word essay about the history of the bicycle." }) } | ConvertTo-Json -Depth 6
$t0 = Get-Date
$reply = Invoke-RestMethod -Uri "$env:TEACHER_URL/v1/chat/completions" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $env:TEACHER_API_KEY" } -Body $body
"seconds {0:N0} | chars {1:N0}" -f ((Get-Date) - $t0).TotalSeconds, $reply.choices[0].message.content.Length
```

Expected: a reply of a few thousand characters that takes **more than 100 seconds** and still returns. If it fails with a 5xx around the 100-second mark, the request is not streaming: the driver must send `"stream": true` and read SSE (Task 3 Step 1), because no timeout or retry setting can hold a non-streamed request open past Cloudflare's edge limit. Verify this here, not in Task 8 — it is the difference between a corpus and a corpus-shaped file of `teacher_error` rows.

4. **The accepted engine is `llama-cpp`.** Measured on the first real run (Colab L4, 2026-10-01): vLLM 0.30.0 with `vllm-gguf-plugin` 0.0.5 dies at `RuntimeError: Unknown gguf model_type: qwen3_5` in `weights_adapter/default.py`, and upstream issue vllm-project/vllm#38122 is still open. The name map is only the first wall: this checkpoint's `vision_config` carries `depth` where the loader reads `num_hidden_layers`, and spec §7.6's image column makes vision mandatory. So the fallback is *the* path, not a plan B.

   ```
   !python /content/serve_teacher.py --engine llama-cpp --tokenizer XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
   ```

   It builds llama.cpp with CUDA (~10 minutes, cached in the runtime) and serves the same weights and projector through M2's own `--jinja --reasoning-format deepseek --reasoning-preserve --reasoning-budget 6144`. All four canaries passed first try: `text reply: 'ok'`, `image canary: PASS`, `reasoning canary: PASS`, `tools canary: PASS`.

   **Record two numbers from the server log before continuing — they set Task 8's budget.**
   - `print_timing ... tg = 44.7 t/s`: the per-slot decode rate. Aggregate is that times `--slots`, so 8 slots is ~350 t/s.
   - `n_ctx_slot = 4096`: the per-row context, which is `--ctx-size / --slots`. The first run used `-c 32768 -np 8`, and 4096 per row is *below* the 6144-token reasoning backstop, so long reasoning rows would truncate silently. The launcher now takes `--ctx-size` (default 262144) and `--slots` (default 16), giving 16384 per row; `--slots 32` roughly doubles throughput at 8192 per row.

   Task 8 phase 1 re-measures the rate at the launch settings rather than trusting this line.

- [ ] **Step 7: Commit and push**

This is what makes Step 6's `curl` work, so it runs *before* the Colab cell when you are fetching the script over HTTPS:

```bash
git add tools/colab/serve_teacher.py tools/colab/README.md
git commit -m "feat(colab): vLLM teacher server and first-run primer"
git push origin main
```

---

### Task 2: Make the shared state safe for workers

**Files:**
- Modify: `tools/dataset/gencache.py`
- Modify: `tools/dataset/imgstore.py`

Every worker in Task 3 appends to one cache file, and every worker can also write an image into the content-addressed store. Without a lock, two threads interleave a partial cache line and the cache becomes unreadable — which would silently discard hours of teacher work. And `imgstore.put` writes bytes straight to the final path, so a worker killed mid-write leaves a torn image where a later run's `Image.open(existing)` raises. This task lands first so Task 3 has something safe to write to.

- [ ] **Step 1: Add the lock and first-write-wins semantics**

In `tools/dataset/gencache.py`, replace the module docstring's last paragraph and the `GenCache` class with this. The docstring correction matters: the class is now called from several threads, so the "keys cover `{kind, messages}` only" note gains a concurrency clause.

```python
"""Append-only resume cache. One JSONL line per completed request; re-runs replay.

Spec section 10: re-running is idempotent. Single-turn rows key on the request payload;
a trajectory keys on the whole conversation prefix sent so far, so an interrupted
trajectory resumes at its next missing turn rather than restarting.

Concurrency-safe: `get` and `put` take a lock, so the M3 worker pool can share one cache.

Reload-safe: the file is split on `\n` and an unparseable line is skipped with a count, so a
line-separator character in a completion or a torn tail does not make hours of teacher work
unloadable.

Stores the **completion**, not the verdict: `generate_one` recomputes the verdict on every
replay, so changing a filter or a verifier applies to the whole cache rather than only to rows
generated after the edit.

ponytail: keys cover `{kind, messages}` only. Changing a sampling parameter, the `tools`
field, or the model between runs replays the cached completion rather than regenerating.
That is fine while the spec fixes sampling; if a run needs a different parameter set,
delete the cache. The lock makes writes safe, not keys unique: two workers must never be
handed the same key, which is why every caller derives its key from the prompt's content.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def request_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def prefix_key(messages: list[dict], *, kind: str) -> str:
    """Key for one teacher call: the whole prefix sent so far, tagged by its role in the run."""
    return request_key({"kind": kind, "messages": messages})


class GenCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            # Split on "\n" rather than `str.splitlines()`. `splitlines` also breaks on
            # U+2028, U+2029 and \x85, which `json.dumps(ensure_ascii=False)` writes raw, so
            # one such character inside a completion splits its record in two and makes the
            # whole cache unloadable. Measured: `prompts/train.jsonl` in the live corpus
            # already contains a raw U+2028, so this is a live trigger, not a hypothetical.
            unparsed = 0
            for line in self.path.read_text(encoding="utf-8").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A torn tail from a killed writer, or a line from an older format.
                    # Skip it and say so rather than refusing to start at all.
                    unparsed += 1
                    continue
                # First write wins, matching `put`: a re-run must never grow the file
                # or change a value the run already used.
                self.records.setdefault(rec["key"], rec)
            if unparsed:
                print(f"gencache: skipped {unparsed} unparseable line(s) of "
                      f"{unparsed + len(self.records)} in {self.path}", flush=True)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self.records.get(key)

    def put(self, key: str, value: dict) -> None:
        rec = {"key": key, **value}
        with self._lock:
            if key in self.records:
                return
            self.records[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.records)
```

- [ ] **Step 2: Make the image store's writes atomic**

Content addressing makes two workers writing the *same* image harmless, but `ImageStore.put` writes bytes straight to the final path. A write interrupted mid-flight (a killed worker, a full disk, Ctrl-C) leaves a truncated file at `images/<aa>/<sha>.<ext>`; the next run's `resolve` finds it and hands it to `Image.open`, which raises. A temp name plus `os.replace` fixes it: the final path only ever appears complete.

In `tools/dataset/imgstore.py`, add `import os` to the imports (it is not there today), then replace the tail of `put`:

```python
        ext = EXT_BY_FORMAT.get(fmt, ".png")
        target = self.dir_for(sha) / f"{sha}{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # A temp name the `resolve` glob (`{sha}.*`) cannot match, then an atomic
            # rename: no reader ever opens a half-written image.
            tmp = target.parent / f".tmp-{target.name}"
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return sha, w, h, str(target.relative_to(self.root))
```

The leading dot is load-bearing: `resolve` globs `{sha}.*`, so a temp file named `{sha}{ext}.part` would still be picked up by a concurrent reader. `.tmp-` is not matched by that pattern.

- [ ] **Step 3: Extend the gencache smoke to prove the concurrency claim**

Append to the existing `if __name__ == "__main__":` block in the same file, after the `print("same key twice:", ...)` line:

```python
    # Concurrency: many threads appending at once must leave the file parseable, one line
    # per distinct key, and a re-`put` of an existing key must not add a second line.
    import threading

    many = Path(tempfile.mkdtemp()) / "many.jsonl"
    shared = GenCache(many)
    threads = [threading.Thread(target=lambda i=i: shared.put(f"k{i}", {"i": i}))
               for i in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reloaded = GenCache(many)
    print("threaded put reload:", len(reloaded), "of 200",
          "| malformed lines:", sum(
              1 for line in many.read_text(encoding="utf-8").splitlines()
              if line.strip() and "key" not in json.loads(line)))
    before = many.stat().st_size
    reloaded.put("k0", {"i": 999})
    print("re-put of an existing key appends nothing:", many.stat().st_size == before)

    # A line-separator character inside a value must not split the record. The corpus
    # already contains a raw U+2028, so this is the case that matters most.
    sep = Path(tempfile.mkdtemp()) / "sep.jsonl"
    GenCache(sep).put("k", {"completion": {"content": f"one{chr(0x2028)}two"}})
    print("U+2028 record reloads whole:", len(GenCache(sep)) == 1)
```

- [ ] **Step 4: Smoke both changes**

Run:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.gencache
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import io, pathlib, tempfile; from PIL import Image; from tools.dataset.imgstore import ImageStore; s=ImageStore(pathlib.Path(tempfile.mkdtemp())/'images'); b=io.BytesIO(); Image.new('RGB',(4,4),(1,2,3)).save(b,format='PNG'); sha,w,h,rel=s.put(b.getvalue()); print('stored', sha[:8], w, h, rel); print('no temp file left behind:', not list((s.root/sha[:2]).glob('.tmp-*'))); print('second put converges:', s.put(b.getvalue())[0]==sha)"
```

Expected, in order:

```
records after reload: 2
replayed value: hello
same key twice: True
threaded put reload: 200 of 200 | malformed lines: 0
re-put of an existing key appends nothing: True
U+2028 record reloads whole: True
stored <8 hex> 4 4 <the sha>.<ext>
no temp file left behind: True
second put converges: True
```

The `stored` line is the sha prefix, the image's width and height, and its path relative to the store root; the last two lines are the atomicity claim. `tools.dataset.imgstore`'s own smoke is unchanged by this task, so run it too (`-m tools.dataset.imgstore`) and check it does not regress.

The benign `<frozen runpy>: RuntimeWarning` is expected under `python -m` and is not a failure.

- [ ] **Step 5: Commit**

```bash
git add tools/dataset/gencache.py tools/dataset/imgstore.py
git commit -m "fix(dataset): make the cache and image store safe to share across workers"
```

---

### Task 3: Drive the teacher concurrently

**Files:**
- Modify: `tools/dataset/generate.py`
- Modify: `tools/dataset/teacher.py`

vLLM's whole advantage is continuous batching, and a sequential driver leaves it idle. This task adds a bounded worker pool. Two invariants keep the change safe: workers touch only the cache and the network, and results are folded into statistics by the main thread in submission order, so `--concurrency 1` is identical to M2's behaviour in structure and in every statistic. Identical *acceptance* is a structural guarantee; identical *text* is not, because batching changes floating-point numerics even with a fixed seed.

- [ ] **Step 1: Let the teacher client authenticate**

In `tools/dataset/teacher.py`, replace the `TeacherClient.__init__` and `_post` so the tunnel can require a key. The Colab URL is public; without this, anyone who sees it can spend the GPU.

```python
    def __init__(self, base_url: str = "http://127.0.0.1:8086",
                 model: str = "local-teacher", timeout: int = 1800,
                 retries: int = 3, backoff: float = 5.0, api_key: str | None = None):
        # 1800, not 900. `_post_retry` treats a timeout as terminal (it breaks before
        # sleeping), and M2 measured rows being killed at 900s against this same teacher.
        # The default stays at the value that survived that measurement.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.api_key = api_key

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _stream(self, payload: dict) -> dict:
        """One completion, read as SSE and reassembled into a message dict.

        Streaming is not an optimisation here, it is the only way a long completion
        survives the tunnel. Cloudflare abandons a proxied request that has sent no bytes
        for about 100 seconds (error 524), and a non-streaming completion sends nothing at
        all until it is finished. M2 never hit this because the teacher was on
        127.0.0.1; every M3 row goes through a quick tunnel, and the long-CoT rows are
        exactly the ones that take longer than that. Retries and a larger timeout cannot
        fix it, because the edge gives up regardless.
        """
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers=self._headers())
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue                        # keep-alive or comment frame
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta") or {}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for call in delta.get("tool_calls") or []:
                    # Tool calls arrive as fragments addressed by `index`: the name and
                    # the arguments are spread across several frames.
                    slot = calls.setdefault(call.get("index", 0),
                                            {"id": None, "type": "function",
                                             "function": {"name": "", "arguments": ""}})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        out: dict = {"content": "".join(content)}
        if reasoning:
            out["reasoning_content"] = "".join(reasoning)
        if calls:
            out["tool_calls"] = [calls[i] for i in sorted(calls)]
        return out

    def _post_retry(self, path: str, payload: dict, *, stream: bool = False) -> dict:
        last: Exception | None = None
        attempts = 0
        for attempt in range(1, self.retries + 1):
            attempts = attempt
            try:
                return self._stream(payload) if stream else self._post(path, payload)
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as exc:
                last = exc
                # A timeout means the teacher is still generating. Retrying replays the
                # same deterministic request and burns the same wall-clock again, so a
                # timeout is terminal. Retries stay for connection-level failures.
                if isinstance(exc, TimeoutError):
                    break
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise TeacherError(f"{path} failed after {attempts} attempt(s): {last}")
```

Then make `complete` use the streamed path. Three things change: the payload's `stream` flag, the retry call, and the return, because `_stream` already returns the assistant *message* rather than the full envelope.

```python
        payload = {
            "model": self.model,
            "messages": to_wire_messages(messages, store),
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "min_p": MIN_P, "presence_penalty": PRESENCE_PENALTY,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        reply = self._post_retry("/v1/chat/completions", payload, stream=True)
        return _fold_reasoning(reply)
```

`_post` stays for non-streamed endpoints (`/health`, `/props`), and `_post_retry`'s terminal-timeout rule is unchanged. Do not "fix" a tunnel timeout by raising `--timeout`: streaming removes the 100-second gap between bytes, which is the actual cause.

Also add `api_key` to the `__main__` smoke's client construction so a protected server can be checked from the shell:

```python
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8086"
    client = TeacherClient(url, api_key=os.environ.get("TEACHER_API_KEY") or None)
```

and add `import os` to that block's imports (it currently imports `io`, `sys`, `tempfile`, `Path`).

- [ ] **Step 2: Add the order-preserving map and the progress line**

In `tools/dataset/generate.py`, add these after the `_seed` helper. `_map` is the single place concurrency is introduced; everything stateful stays sequential.

```python
def _map(fn, items, concurrency: int):
    """Order-preserving map. `concurrency <= 1` runs inline, so the M2 path is unchanged."""
    if concurrency <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


def _progress(rel: str, done: int, total: int, stats: dict, every: int = 100) -> None:
    """A heartbeat, so a multi-hour run is observable rather than a silent process.

    M2's full run was unobservable for hours at a time; that is the defect this fixes.
    """
    if done % every and done != total:
        return
    rate = (stats["accepted"] / done) if done else 0.0
    print(f"  {rel}: {done}/{total} attempted, {stats['accepted']} accepted "
          f"(pass {rate:.2f})", flush=True)
```

- [ ] **Step 3: Thread `concurrency` through `_run_pool`**

Replace `_run_pool` in `tools/dataset/generate.py` with this version. Two things stay exactly as they are: `root`, because it is where a rejected row is written, and the two `_record_reject` calls, because a drop whose completion is not stored cannot be audited. Only the request ordering in time changes.

```python
def _run_pool(pool: Path, limit: int | None, client, cache, *, root, store, thinking, stats,
              concurrency: int = 1):
    records = [prompt_from_dict(r) for r in iter_jsonl(pool)]
    if limit is not None:
        records = split.downsample(records, limit)

    def work(prompt):
        try:
            return generate_one(prompt, client, cache, store=store, thinking=thinking), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (prompt, (verdict, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
            _record_reject(root, stage="seeded", reason="teacher_error", detail=str(exc),
                           pool=pool.name, id=prompt.id)
        elif verdict["accepted"]:
            out.append(canonical.example_from_dict(verdict["example"]))
            stats["by_domain"][prompt.domain] += 1
            stats["accepted"] += 1
            stats["accepted_by_source"][key] += 1
        else:
            stats["drops"][verdict["reason"]] += 1
            _record_reject(root, stage="seeded", reason=verdict["reason"],
                           detail=verdict.get("detail"), pool=pool.name, id=prompt.id,
                           record=verdict.get("example"))
        _progress(pool.name, done, len(records), stats)
    return out
```

`run_seeded` gains a `concurrency` keyword and passes it to each `_run_pool` call, in both the train and the val loop:

```python
def run_seeded(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
               thinking: bool = True, concurrency: int = 1):
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    train: list[Example] = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            train.extend(_run_pool(path, limit, client, cache, root=root, store=store,
                                   thinking=thinking, stats=stats, concurrency=concurrency))
    val: list[Example] = []
    for rel in VAL_POOLS:
        path = root / rel
        if path.exists():
            val.extend(_run_pool(path, val_limit, client, cache, root=root, store=store,
                                 thinking=thinking, stats=stats, concurrency=concurrency))
    canonical.write_jsonl(root / "m2" / "single.jsonl", train)
    canonical.write_jsonl(root / "m2" / "val.jsonl", val)
    _dump_stats(root / "m2" / "seeded.stats.json", stats)
    print(f"seeded: accepted {stats['accepted']} of {stats['attempted']}", flush=True)
    return stats
```

- [ ] **Step 4: Thread `concurrency` through the trajectory and simulated passes**

A multi-turn loop is a *sequence* of dependent calls, so trajectories cannot be parallelised within one row — but different rows can. Apply the same shape to `run_trajectory` and `run_simulated`: build the work function over the record list, `_map` it, then fold results sequentially — and keep every `_record_reject` call inside that sequential fold, so the reject ledger still names each dropped row.

In `run_trajectory`, change the seed guard and replace the `for prompt in records:` loop body with a mapped one. The guard change is a separate one-liner because the loop body starts below it: `if limit:` must become `if limit is not None:`, or `--multi-limit 0` would answer the **entire** trajectory pool — Task 3 Step 7 passes `--multi-limit 0` expecting zero attempts.

```python
    def work(prompt):
        try:
            return trajectory.generate_prebuilt(prompt, client, cache, store=store,
                                                thinking=thinking), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (prompt, (traj, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
            _record_reject(root, stage="trajectory", reason="teacher_error", detail=str(exc),
                           pool="prompts/trajectories.jsonl", id=prompt.id)
        elif traj is None:
            stats["drops"]["turn_structure"] += 1
            _record_reject(root, stage="trajectory", reason="turn_structure",
                           pool="prompts/trajectories.jsonl", id=prompt.id)
        else:
            # Unchanged from M2: the reject reason and its detail come from
            # `_trajectory_reject`, not from a boolean helper.
            reason, detail = _trajectory_reject(traj)
            if reason:
                stats["drops"][reason] += 1
                _record_reject(root, stage="trajectory", reason=reason, detail=detail,
                               pool="prompts/trajectories.jsonl", id=prompt.id,
                               record=traj.to_dict())
            else:
                out.append(traj)
                stats["accepted"] += 1
                stats["by_domain"][prompt.domain] += 1
                stats["accepted_by_source"][key] += 1
        _progress("trajectories", done, len(records), stats, every=25)
```

and give the signature `concurrency: int = 1`. `_trajectory_reject` and the `_record_reject` calls stay in the main thread, so the `stats` writes and the `m2/rejected.jsonl` appends are still sequential.

`run_simulated` is the same edit, mapping a `work` that calls `trajectory.generate_simulated(...)` and returns `(traj, exc)`, with `_progress("simulated", ...)`. Note the `continue` guards: a seed with no first user turn is still skipped in the main thread, before the mapped call, so add `first_user` to the record list rather than computing it inside `work`:

```python
    seeds = [...]
    if limit is not None:
        seeds = split.downsample(seeds, limit)
    prepared = []
    for seed in seeds:
        first_user = next((canonical.message_text(m) for m in seed.messages
                           if m.get("role") == "user"), "")
        if first_user:
            prepared.append((seed, first_user))

    def work(item):
        seed, first_user = item
        try:
            return trajectory.generate_simulated(
                client, cache, id=f"sim-{seed.id}", domain=seed.domain,
                source={"name": "teacher:simulated"}, tools=seed.tools,
                first_user=first_user, thinking=thinking, store=store), None
        except TeacherError as exc:
            return None, exc
```

- [ ] **Step 5: Add the CLI flags and route them**

In `tools/dataset/generate.py`'s `main`, add:

```python
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel teacher requests; >1 is what makes vLLM's batching pay")
    ap.add_argument("--api-key", default=None, help="Bearer token for the teacher endpoint")
    ap.add_argument("--cache", type=Path, default=None,
                    help="resume cache path; defaults to <root>/m2/cache.jsonl")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds per teacher call; a timeout is terminal, not retried")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--schema-limit", type=int, default=0,
                    help="Magpie tool-schema inventions; 0 leaves any existing pool alone")
```

then **replace the four pool-limit defaults**, which are numbers today (`--limit 300`, `--val-limit 100`, `--multi-limit 100`, `--sim-limit 40`) and are guarded by truthiness in the M2 runners:

```python
    ap.add_argument("--limit", type=int, default=None,
                    help="per-pool candidate cap; 0 answers none in the pool, omitted "
                         "answers every candidate")
    ap.add_argument("--val-limit", type=int, default=None,
                    help="same, for the val pool")
    ap.add_argument("--multi-limit", type=int, default=None,
                    help="same, for the prebuilt trajectory pool")
    ap.add_argument("--sim-limit", type=int, default=None,
                    help="same, for the simulated pool (prebuilt seeds + schema seeds)")
```

construct the client and cache with them. The cache path becomes overridable because Task 3
Step 7 needs two runs over one prompt corpus that do *not* share a cache:

```python
    cache = GenCache(args.cache or (args.root / "m2" / "cache.jsonl"))
    client = TeacherClient(args.base_url, model=args.model, timeout=args.timeout,
                           retries=args.retries, api_key=args.api_key)
```

and before routing to any mode, probe the endpoint and exit on failure. Without this a dead or
rotated tunnel URL turns every row into `teacher_error`, and the passes *still complete* —
`run_seeded` writes `single.jsonl`, `run_difficulty` writes an all-error verdict file, and
`run_merge` overwrites `train.jsonl` — so the previous corpus is destroyed at the shipped
artefact level while the process exits 0:

```python
    try:
        client._get("/health")
    except Exception as exc:
        raise SystemExit(f"teacher endpoint {args.base_url} is unhealthy: {exc}")
```

and pass `concurrency=args.concurrency` and `schema_limit=args.schema_limit` to `run_all`, which gains both keywords and forwards `schema_limit` to its `magpie.run` call. Add `sys.stdout.reconfigure(line_buffering=True)` as the first statement of `main()` so progress appears in a redirected log immediately.

A caveat on the limits, because the two words "none" and "everything" are easy to swap and the swap is silent:

- `--limit 0` now means "attempt none in the pool". It works because `downsample(records, 0)` returns `[]` and the new guards are `if limit is not None:` rather than `if limit:`.
- `--limit` omitted now means "attempt every candidate in the pool", because the default is `None`.
- That is a change from M2, where all four flags defaulted to a number and the truthiness guard made `0` mean *everything*. Change the defaults, or the sentence above stops being true: with the old `default=300`, an omitted `--limit` silently answers 300 prompts — which is why the whole-pool statements elsewhere in this plan (Task 5 Step 3's `--mode difficulty`, Task 6 Step 3's `flags` block) only hold once this change has landed.
- `--magpie-limit` changes its default to **`0`**, meaning "leave the existing pool alone", and `magpie.run` refuses to write a *smaller* pool than the file already holds. The current default of `20` is a data-loss trap: `magpie.run` rebuilds the pool with `write_jsonl`, which opens the file in `"w"`, so a `--mode all` run that forgets the flag replaces Task 7's ~6,000 invented rows with ~40, and the coding and uncensored Magpie columns collapse without any error. `magpie.build` needs a number (`len(out) >= limit`), so `0` is the safe value rather than `None`.

The consequence to know before running anything: a bare `--mode all` now answers the **whole** corpus rather than M2's 300-prompt sample. That is the intent (spec §7: "The teacher answers every candidate"), but it is not a command to run casually against a Colab session.

- [ ] **Step 6: Verify the offline smoke still passes and concurrency does nothing at 1**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --dry-run --concurrency 8
```

Expected: the first prints

```
smoke accepted: 1 | valid: True
smoke rejected: 1 | reason: verify_failed | completion kept: True
```

and exit code 0. The second prints the four `[dry]` lines and exits 0.

Check **both** lines and the exit code. `smoke()` asserts `len(rejected) == 1` with its completion stored, so if the Step 3 rewrite dropped `root` or the `_record_reject` calls, this step fails instead of shipping an empty `m2/rejected.jsonl`. The `smoke accepted:` line alone looks identical either way — the rejected line is the one that catches it.

- [ ] **Step 7: Prove concurrency actually batches, against the live teacher**

The test has to satisfy three things at once: the two runs must see the *same* prompts, they must *not* share a warm cache, and they must not write into the real corpus. The `--cache` flag is what makes that possible — one fixture directory, two cache paths.

```powershell
$env:TEACHER_URL = "PASTE_THE_URL_HERE"
$env:TEACHER_API_KEY = "PASTE_THE_KEY_HERE"
$fix = "datasets/qwen35-4b-sft-smoke-conc"
if (-not (Test-Path $fix)) { New-Item -ItemType Directory -Path $fix | Out-Null; Copy-Item "datasets/qwen35-4b-sft-smoke/*" $fix -Recurse -Force }
Remove-Item "$env:TEMP\opencode\conc-1.jsonl","$env:TEMP\opencode\conc-8.jsonl" -ErrorAction SilentlyContinue

$args1 = @("--mode","all","--root",$fix,"--base-url",$env:TEACHER_URL,"--api-key",$env:TEACHER_API_KEY,
           "--limit","24","--val-limit","0","--multi-limit","0","--sim-limit","0","--magpie-limit","0")
$t1 = Measure-Command { & "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate @args1 --cache "$env:TEMP\opencode\conc-1.jsonl" --concurrency 1 | Tee-Object "$env:TEMP\opencode\conc-serial.txt" }
$t8 = Measure-Command { & "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate @args1 --cache "$env:TEMP\opencode\conc-8.jsonl" --concurrency 8 | Tee-Object "$env:TEMP\opencode\conc-parallel.txt" }
"serial {0:N1}s  parallel {1:N1}s  speedup {2:N1}x" -f $t1.TotalSeconds, $t8.TotalSeconds, ($t1.TotalSeconds / $t8.TotalSeconds)
Select-String -Path "$env:TEMP\opencode\conc-serial.txt","$env:TEMP\opencode\conc-parallel.txt" -Pattern 'seeded:|merged:'
```

Expected: the `seeded:` and `merged:` lines are **identical in `N`, `M` and `T`** across the two runs, and the speedup line reports at least `2.0x`.

`--magpie-limit 0` is what keeps both runs from rewriting `prompts/magpie.jsonl` in the fixture.

Identical accepted counts are the point: concurrency must not change what ships. If they differ, a worker is mutating shared state — stop and fix before Task 8. The speedup is a secondary signal: below `2.0x` at `--concurrency 8`, batching is not happening, and the usual cause is a tunnel or a server configured for one sequence rather than a bug in the driver. Level `pass_rate` can legitimately differ in the last decimal between the two runs even with matched counts, because batching changes floating-point numerics; compare counts, not the rate.

- [ ] **Step 8: Commit**

```bash
git add tools/dataset/generate.py tools/dataset/teacher.py
git commit -m "feat(dataset): concurrent driver and authenticated teacher client"
```

---

### Task 4: Magpie tool-schema invention at scale

**Files:**
- Create: `tools/dataset/prompts/magpie_toolschema.md`
- Modify: `tools/dataset/magpie.py`

Spec §10.2 asks for simulated trajectories whose user turn is "invented from a `tools` schema when the seed is a tool set rather than a conversation". M2's Magpie invents from a *seed request* only, so the simulated path had nothing schema-conditioned to seed from. This task adds it, and fixes two things that only break at scale: ids derived from a list length (nondeterministic once generation is concurrent) and no dedup on invented prompts (irrelevant at 40 rows, not at 4,000).

- [ ] **Step 1: Write the schema-conditioned template**

Create `tools/dataset/prompts/magpie_toolschema.md`:

```markdown
You invent user requests for a tool-calling dataset. You are given the JSON schema of one
available tool, and an unrelated example request for flavour. Write ONE new, concrete user
request that this tool is genuinely needed to answer, and that names realistic argument
values for it. The request must be answerable by calling this tool, not by describing it.
Reply with only the request — no preamble, no list, no markdown.

Tool schema: {schema}

Example request (do not copy it): {seed}
```

- [ ] **Step 2: Add schema-conditioned invention**

In `tools/dataset/magpie.py`, add `import hashlib` and `import json` to the imports, then add after `invent`:

```python
def _extract_schemas(path: Path) -> list[tuple[dict, str]]:
    """`(tool schema, example first-user-turn)` for every prompt row that declares tools.

    The schema is the seed here, not the prompt: spec section 10.2 invents a user turn from
    a tool set so the simulated loop has a reason to call a tool. Taking the example request
    from the row as well is what keeps invented rows from converging on one wording.
    """
    out: list[tuple[dict, str]] = []
    if not Path(path).exists():
        return out
    for row in iter_jsonl(path):
        tools = row.get("tools") or []
        if not tools:
            continue
        example = ""
        for m in row.get("messages", []):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            example = content if isinstance(content, str) else "".join(
                p.get("text", "") for p in content if p.get("type") == "text")
            break
        for tool in tools:
            out.append((tool, example.strip()))
    return out


def invent_from_schema(client: TeacherClient, cache, *, schema: dict, example: str = "",
                       thinking: bool = False, max_tokens: int | None = None) -> str | None:
    """One invented user request conditioned on a tool schema (spec section 10.2)."""
    template = (PROMPT_DIR / "magpie_toolschema.md").read_text(encoding="utf-8")
    blob = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    key = gencache.request_key({"magpie_toolschema": blob, "example": example, "v": 1})
    cached = cache.get(key)
    if cached is None:
        instruction = template.replace("{schema}", blob).replace("{seed}", example or "none")
        msg = client.complete(
            [{"role": "system", "content": "You are a dataset generator."},
             {"role": "user", "content": instruction}],
            thinking=thinking, max_tokens=max_tokens)
        cached = {"completion": {"content": (msg.get("content") or "").strip()}}
        cache.put(key, cached)
    text = cached["completion"]["content"].splitlines()[0].strip() if cached[
        "completion"]["content"] else ""
    return text or None
```

- [ ] **Step 3: Make ids content-derived and dedup the invented pool**

A length-derived id is already fragile across re-runs; with concurrent invention it becomes actively wrong, because two workers append to `out` in arbitrary order and two different requests can receive the same id. Replace the id line in `build` and add the schema-building pass:

```python
def _prompt_id(domain: str, text: str) -> str:
    """Content-derived id, so the same shipped text always yields the same id.

    Hashing the *invented text* rather than the seed makes an id identify the row that
    actually ships, and keeps it stable across a `--concurrency` change and across re-runs.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"magpie-{domain}-{digest}"
```

then give `build` a `deduper`, content-derived ids, and a dedup guard. Replace `build`'s signature and its `out.append(...)` block with this:

```python
def build(domain: str, seeds: list[str], client: TeacherClient, cache, *,
          limit: int, source_name: str, deduper=None) -> list[Prompt]:
    from .dedup import Deduper

    deduper = deduper or Deduper()
    out: list[Prompt] = []
    for seed in seeds:
        if len(out) >= limit:
            break
        text = invent(client, cache, domain=domain, seed=seed)
        if not text:
            continue
        if deduper.check_prompt(text, ()):
            continue                      # near-duplicate of an already-kept invention
        deduper.commit_prompt(text, ())
        out.append(Prompt(
            id=_prompt_id(domain, text), domain=DOMAIN_FOR[domain], origin="teacher",
            source={"name": source_name, "license": None},
            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
            meta={"magpie": True, "seed": seed[:200]}))
    return out
```

Then add the schema pass, reusing the same `Deduper` rather than growing a second dedup implementation:

```python
def build_schema_prompts(items, client, cache, *, limit, source_name, deduper=None):
    """Invent user turns from tool schemas; each row carries its `tools` schema.

    The schema is the point: spec section 10.2 seeds the simulated loop from a tool set so
    the teacher has a reason to call a tool. `deduper` is shared with the other Magpie pools
    so a request invented twice across pools is rejected rather than shipped twice.
    """
    from .dedup import Deduper

    deduper = deduper or Deduper()
    out: list[Prompt] = []
    for schema, example in items:
        if len(out) >= limit:
            break
        text = invent_from_schema(client, cache, schema=schema, example=example)
        if not text:
            continue
        if text.strip().lower() == (example or "").strip().lower():
            continue                      # the teacher echoed its example: nothing invented
        if deduper.check_prompt(text, ()):
            continue
        deduper.commit_prompt(text, ())
        out.append(Prompt(
            id=_prompt_id("tools", text), domain=DOMAIN_FOR["tools"], origin="teacher",
            source={"name": source_name, "license": None},
            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
            tools=[schema],
            meta={"magpie": True, "schema_seeded": True, "example": (example or "")[:200]}))
    return out
```

- [ ] **Step 4: Extend `run` to build the schema pool and use content ids**

Replace `run` with a version that builds all three pools and writes them together. `schema_limit` defaults to `0` and `0` means "leave the existing pool alone", which is what keeps `generate.run_all`'s existing call working and keeps M2 resumable (see the note in Prerequisites).

```python
def run(*, root: Path, client: TeacherClient, cache, limit: int, dry_run: bool,
        schema_limit: int = 0) -> int:
    if dry_run:
        for domain in TEMPLATES:
            print(f"[dry] magpie {domain}: limit={limit}")
        print(f"[dry] magpie toolschema: limit={schema_limit}")
        return 0
    if limit <= 0 and schema_limit <= 0:
        print("magpie: skipped, both limits are 0; the existing pool is untouched",
              flush=True)
        return 0
    seed_rows = [r for r in iter_jsonl(root / "seeds" / "uncensored.jsonl")] \
        if (root / "seeds" / "uncensored.jsonl").exists() else []
    uncensored_seeds = [r["text"] for r in seed_rows[:limit]]
    tool_seeds = _tool_prompt_seeds(root / "prompts" / "train.jsonl")[:limit]
    # One Deduper across all three pools: `teacher:tools` is fed by both the request-seeded
    # and the schema-seeded pass, so without sharing, the same request can ship twice under
    # two ids. This is the cross-pool duplicate that only appears at M3 scale.
    from .dedup import Deduper
    deduper = Deduper()
    prompts = build("uncensored", uncensored_seeds, client, cache, limit=limit,
                    source_name="teacher:uncensored", deduper=deduper)
    prompts += build("tools", tool_seeds, client, cache, limit=limit,
                     source_name="teacher:tools", deduper=deduper)
    schema_items = _extract_schemas(root / "prompts" / "train.jsonl")[:schema_limit]
    prompts += build_schema_prompts(schema_items, client, cache, limit=schema_limit,
                                    source_name="teacher:tools", deduper=deduper)
    path = root / "prompts" / "magpie.jsonl"
    existing = sum(1 for _ in iter_jsonl(path)) if path.exists() else 0
    if len(prompts) < existing:
        # `write_jsonl` opens in "w", so without this guard one forgotten `--magpie-limit`
        # replaces a 6,000-row pool with 40 and the corpus loses both Magpie columns
        # silently. Shrinking is only ever deliberate.
        print(f"magpie: refusing to shrink {path} from {existing} to {len(prompts)} rows; "
              f"raise the limit, or delete the file deliberately", flush=True)
        return 1
    written = canonical.write_jsonl(path, prompts)
    by_source: Counter = Counter(p.source["name"] for p in prompts)
    attempted = {"teacher:uncensored": len(uncensored_seeds),
                 "teacher:tools": len(tool_seeds) + len(schema_items)}
    dropped = {name: n - by_source[name] for name, n in attempted.items()}
    print(f"magpie prompts: {written} -> {path} | {dict(by_source)} | "
          f"dropped as duplicate, echoed or empty: {dropped}", flush=True)
    return 0 if written else 1
```

The `dropped` line is the arithmetic that makes a shortfall legible. Without it, a pool that
returned 900 of 3,000 rows looks the same as one that returned 3,000 duplicate rows.

Add `from collections import Counter` to the imports, and to `main` add `--api-key` (the
launcher's endpoint rejects unauthenticated requests) and `--schema-limit`:

```python
    ap.add_argument("--api-key", default=None, help="Bearer token for the teacher endpoint")
    ap.add_argument("--schema-limit", type=int, default=0)
```

construct the client with the key, and pass `schema_limit=args.schema_limit` to `run`:

```python
    return run(root=args.root, client=TeacherClient(args.base_url, api_key=args.api_key),
               cache=GenCache(cache_path), limit=args.limit,
               schema_limit=args.schema_limit, dry_run=args.dry_run)
```

- [ ] **Step 5: Smoke it**

The smoke must not hit the network, so it uses a `FakeTeacher` and asserts the two things that were broken: ids are content-derived, and a duplicate invention is rejected.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.magpie
```

Expected (extend the `__main__` block so it prints these lines):

```
template: You invent user requests for a tool-calling dataset. ...
invented: Plan a three-day trip to Kyoto using the train tool.
domain: coding
schema id stable: True | magpie-tools-aede0235a25c
duplicate rejected: True | carries tools: True
```

The digest is twelve hex characters of the **invented request's** SHA-256, so it is stable
for a given teacher response and changes when the response does. `aede0235a25c` is
`sha256("Book a table for two at eight o'clock.")`, which is what `RepeatTeacher` returns —
if the digest differs, the id is being hashed from the seed instead of the text.

One consequence worth knowing: changing the id scheme orphans any existing cache entries for
magpie rows, because `build` keys its cache on the same values. M2's full run is the only
thing that has such entries, and magpie invention is the first step of `--mode all`, so a
re-run simply re-invents them.

Add to the `__main__` block, after the existing prints:

```python
        # Content-derived ids and duplicate rejection, the two scale defects this task fixes.
        from .dedup import Deduper

        class RepeatTeacher:
            def complete(self, messages, **kw):
                return {"content": "Book a table for two at eight o'clock."}

        def tmp_cache():
            # `gencache.GenCache`, not a bare `GenCache`: the module imports the package, and
            # `GenCache` is only bound inside `main`.
            return gencache.GenCache(Path(tempfile.mkdtemp()) / "c.jsonl")

        one = build("tools", ["Book a train"], RepeatTeacher(), tmp_cache(), limit=1,
                    source_name="teacher:tools")[0]
        two = build("tools", ["Book a train"], RepeatTeacher(), tmp_cache(), limit=1,
                    source_name="teacher:tools")[0]
        print("schema id stable:", one.id == two.id, "|", one.id)

        schema = {"type": "function", "function": {"name": "book", "parameters": {}}}
        first = build_schema_prompts([(schema, "Book a table")], RepeatTeacher(), tmp_cache(),
                                     limit=1, source_name="teacher:tools")
        dupe = build_schema_prompts([(schema, "Book a table"), (schema, "Book a table")],
                                    RepeatTeacher(), tmp_cache(), limit=2,
                                    source_name="teacher:tools", deduper=Deduper())
        print("duplicate rejected:", len(dupe) == 1,
              "| carries tools:", bool(first and first[0].tools))
```

`GenCache` over a fresh temp path is used rather than a bare dict because it is what the real callers pass; two separate calls must use two separate caches, or the second is a cache hit and proves nothing about id stability.

- [ ] **Step 6: Commit**

```bash
git add tools/dataset/magpie.py tools/dataset/prompts/magpie_toolschema.md
git commit -m "feat(dataset): schema-conditioned Magpie invention at scale"
```

---

### Task 5: The §10.2 difficulty filter

**Files:**
- Modify: `tools/dataset/generate.py`

Spec §10.2: "Where a prompt is rolled out more than once, all-pass and all-fail trajectories are dropped, keeping only tasks where the teacher sometimes succeeds." This applies to prompts with a mechanical oracle — the MBPP/APPS seed corpus and the maths golds — which is where all-pass and all-fail are *measurements* rather than guesses.

Two reading notes, so the deviation is visible where the code is rather than only in the appendix:

- The spec's word is *trajectories*; this task applies the rule to **single-turn** oracle-bearing prompts. That is deliberate: a multi-turn rollout costs N calls rather than 1, and `K` rollouts of the same prebuilt trajectory would mostly re-measure the prefix the source already shipped. The reasoning and the cost estimate it would need are under "Deferred to a later milestone".
- The filter is also the one place where a drop is *expected by design* rather than a defect, which is why Task 8's drop table labels `all_pass` and `all_fail` separately from `teacher_error`.

The design keeps the filter nearly free on a warm cache: rollout 0 is keyed exactly as M2's single-turn call (`single:{id}`), so for prompts M2 already answered, only rollouts 1..K−1 are new calls.

- [ ] **Step 1: Parameterise a call by rollout**

In `generate.py`, replace the key derivation in `generate_one` and add the parameter:

```python
def generate_one(prompt, client, cache, *, store, thinking: bool, rollout: int = 0) -> dict:
    """One cached completion, verified and filtered. The verdict is recomputed every run.

    The cache holds the *completion*, not the verdict. Recomputing costs a local `verify`
    call (milliseconds) and is the difference between a filter or verifier edit applying to
    the whole cache and applying only to rows generated after it — the plan itself defers
    one such edit (`error_laden`), and mixing old-code rollout 0 with new-code rollout 1
    would otherwise decide the difficulty verdict with two different rules.

    `rollout` 0 keeps M2's key, so an existing cache replays and the difficulty filter only
    pays for rollouts 1..K-1 (spec section 10.2).
    """
    kind = f"single:{prompt.id}" if rollout == 0 else f"single:{prompt.id}:r{rollout}"
    key = gencache.prefix_key(prompt.messages, kind=kind)
    cached = cache.get(key)
    if cached is None or "completion" not in cached:
        # The `"completion" not in cached` half is what retires M2's verdict-shaped entries:
        # they hold `accepted`/`reason`/`example` with no completion to re-verify, so they
        # are regenerated once instead of being replayed under the old rules.
        msg = client.complete(prompt.messages, store=store, tools=prompt.tools,
                              thinking=thinking, seed=_seed(key))
        cached = {"completion": {"content": msg.get("content") or "",
                                 "tool_calls": msg.get("tool_calls") or []}}
        cache.put(key, cached)
    content = cached["completion"]["content"]
    ex = _completion_example(prompt, content, cached["completion"]["tool_calls"])
    if prompt.verify:
        ok, why = verify.check_record(prompt.verify, content)
        if not ok:
            return {"accepted": False, "reason": "verify_failed", "detail": why,
                    "example": ex.to_dict()}
    reason = filters.example_reject_reason(ex)
    return {"accepted": reason is None, "reason": reason, "example": ex.to_dict()}
```

The cost of retiring the old entries is the 44 M2 rows, which regenerate in about a minute; the benefit is that every later filter change applies everywhere.

- [ ] **Step 2: Add `run_difficulty`**

Add this after `run_seeded`. Two artefacts come out of it, and the distinction is what makes
Step 3 work:

- `m2/difficulty.jsonl` — the **survivors**: prompts the teacher sometimes passed and
  sometimes failed.
- `m2/difficulty.verdicts.jsonl` — one line per **judged** prompt, kept or dropped.

Merge needs the second file. Filtering only against the survivors would leave every all-pass
and all-fail prompt in the corpus untouched, because those rows were already accepted by the
seeded pass and are absent from the survivor set.

`verify` is the oracle: a prompt without one is out of scope here, because "all-pass" is
meaningless for a column with no test.

```python
DIFFICULTY_K = 2
ACCEPTED_TRAIN_TARGET = 25_000   # spec section 4; merge trims to it when the filters do not


def run_difficulty(*, root: Path, client, cache, k: int = DIFFICULTY_K,
                   limit: int | None = None, thinking: bool = True, concurrency: int = 1):
    """Spec section 10.2 rejection sampling over oracle-bearing prompts.

    Keeps a prompt only when the teacher passes some rollouts and fails others. Rollout 0
    reuses M2's cache key, so a warm cache costs only the extra rollouts.

    Writes the survivors *and* a verdict line for every judged prompt. Merge filters against
    the verdicts, not the survivors: an all-fail prompt is absent from the survivors but
    present in the corpus from the seeded pass, and only the verdicts name it.
    """
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "k": k, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    records = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            records.extend(p for p in (prompt_from_dict(r) for r in iter_jsonl(path))
                           if p.verify)
    if limit is not None:
        records = split.downsample(records, limit)

    def work(prompt):
        rollouts = []
        for i in range(k):
            try:
                rollouts.append(generate_one(prompt, client, cache, store=store,
                                             thinking=thinking, rollout=i))
            except TeacherError as exc:
                return rollouts, exc
        return rollouts, None

    out: list[Example] = []
    verdicts: list[dict] = []
    for done, (prompt, (rollouts, exc)) in enumerate(zip(records, _map(
            work, records, concurrency)), start=1):
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None or len(rollouts) < k:
            stats["drops"]["teacher_error"] += 1
            verdicts.append({"id": prompt.id, "verdict": "teacher_error",
                             "source": key, "passes": None})
            print(f"  teacher error on {prompt.id}: {exc}", flush=True)
        else:
            passes = [r for r in rollouts if r["accepted"]]
            if len(passes) == k:
                stats["drops"]["all_pass"] += 1
                verdicts.append({"id": prompt.id, "verdict": "all_pass", "source": key,
                                 "passes": k})
            elif not passes:
                stats["drops"]["all_fail"] += 1
                verdicts.append({"id": prompt.id, "verdict": "all_fail", "source": key,
                                 "passes": 0})
            else:
                out.append(canonical.example_from_dict(passes[0]["example"]))
                stats["accepted"] += 1
                stats["by_domain"][prompt.domain] += 1
                stats["accepted_by_source"][key] += 1
                verdicts.append({"id": prompt.id, "verdict": "keep", "source": key,
                                 "passes": len(passes)})
        _progress("difficulty", done, len(records), stats, every=50)
    canonical.write_jsonl(root / "m2" / "difficulty.jsonl", out)
    # The first line is a fingerprint, not a verdict. `run_merge` refuses to filter against a
    # file written for a different set of oracle prompts, which is what makes a *partial*
    # difficulty pass (or a rebuilt corpus) fail loudly instead of silently un-filtering: a
    # `--limit 12` verdict file would otherwise re-admit every row the full pass had dropped.
    meta = {"kind": "meta", "k": k, "judged": len(verdicts), "oracle_in_pools": len(records)}
    (root / "m2" / "difficulty.verdicts.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in [meta, *verdicts]),
        encoding="utf-8")
    _dump_stats(root / "m2" / "difficulty.stats.json", stats)
    print(f"difficulty: accepted {stats['accepted']} of {stats['attempted']} judged "
          f"(k={k}, drops {dict(stats['drops'])})", flush=True)
    return stats
```

The verdict file is written with `write_text` rather than `canonical.write_jsonl`, because
`write_jsonl` takes `Example | Prompt` objects and a verdict is neither.

- [ ] **Step 3: Make merge filter against the judged ids**

The seeded pass already generated every oracle-bearing prompt once, unfiltered, so the
all-pass and all-fail rows are sitting in `m2/single.jsonl`. Merge has to remove them. It
cannot do that by consulting the survivors, because an all-fail prompt is *absent* from the
survivors and *present* in the corpus. Only the verdict file names it.

```python
    train = load("single.jsonl") + load("trajectory.jsonl") + load("simulated.jsonl")
    survivors = load("difficulty.jsonl")
    verdicts_path = root / "m2" / "difficulty.verdicts.jsonl"
    judged: list[dict] = []
    meta: dict = {}
    if verdicts_path.exists():
        rows = [json.loads(line) for line in
                verdicts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = next((r for r in rows if r.get("kind") == "meta"), {})
        judged = [r for r in rows if r.get("kind") != "meta"]
    if judged:
        # Guard first, filter second. A verdict file covering a subset of the oracle prompts
        # re-admits every row the full pass had dropped and drops the survivors it never saw,
        # and nothing else in the pipeline would notice.
        current_oracle = sum(1 for rel in TRAIN_POOLS if (root / rel).exists()
                             for r in iter_jsonl(root / rel) if prompt_from_dict(r).verify)
        if meta.get("oracle_in_pools") != current_oracle:
            raise SystemExit(
                f"{verdicts_path} was written for {meta.get('oracle_in_pools')} oracle "
                f"prompts but the pools hold {current_oracle}; re-run `--mode difficulty` "
                f"(idempotent and cache-backed) before merging")
        # The filter's decision replaces the seeded row, whatever that decision was:
        # `keep` contributes the survivor, `all_pass` and `all_fail` contribute nothing.
        # A `teacher_error` left no verdict, so its seeded row stands; that keeps an
        # interrupted endpoint from silently shrinking the corpus.
        decided = {v["id"] for v in judged if v["verdict"] != "teacher_error"}
        train = [ex for ex in train if ex.id not in decided] + survivors
    if not train:
        raise SystemExit("merge produced no train rows; refusing to overwrite train.jsonl")
    accepted_before = len(train)
    if len(train) > ACCEPTED_TRAIN_TARGET:
        # Spec section 13: "filter down to 25,000 rather than capping at 25,000". When pass
        # rates come in high the filters do not reduce the count, and §4's mix is what the
        # trim holds — `downsample` weights by DOMAIN_SHARE, which is 30/30/20/20.
        train = split.downsample(train, ACCEPTED_TRAIN_TARGET)
    val = load("val.jsonl")
```

and extend the manifest block so the difficulty pass is reported on its own terms rather
than folded into the seeded pass rate:

```python
    m2 = {"seeded": stats("seeded.stats.json"), "trajectory": stats("trajectory.stats.json"),
          "simulated": stats("simulated.stats.json"),
          "difficulty": stats("difficulty.stats.json"),
          "accepted_before_trim": accepted_before,
          "difficulty_judged": len(judged),
          "difficulty_shipped": len(survivors),
          "difficulty_removed": len([v for v in judged
                                     if v["verdict"] not in ("keep", "teacher_error")]),
          "train_written": len(train), "val_written": len(val)}
    shards = ("seeded", "trajectory", "simulated")
    passed = sum(m2[k].get("accepted", 0) for k in shards)
    attempted = sum(m2[k].get("attempted", 0) for k in shards)
    m2["pass_rate"] = round(passed / attempted, 4) if attempted else 0.0
```

and add `difficulty` as a mode:

```python
    ap.add_argument("--mode", choices=["all", "merge", "difficulty"], default="all")
```

with, in `main` before the merge branch. `--limit` is forwarded, so a bare `--mode difficulty` judges every oracle-bearing prompt (its default is `None` after Task 3 Step 5), and the small-sample checks below pass `--limit 12` explicitly.

```python
    if args.mode == "difficulty":
        run_difficulty(root=args.root, client=client, cache=cache, k=DIFFICULTY_K,
                       limit=args.limit, concurrency=args.concurrency)
        run_merge(root=args.root)
        return 0
```

and in `run_all`, between the seeded and trajectory passes:

```python
    run_difficulty(root=root, client=client, cache=cache, k=DIFFICULTY_K,
                   thinking=thinking, concurrency=concurrency)
```

`DIFFICULTY_K` is a constant rather than a flag. Add `--difficulty-k` when a second value is
actually wanted; right now a knob with one setting is just a way to get the run wrong.

- [ ] **Step 4: Verify the filter's three verdicts offline**

Extend the `smoke()` function so it exercises a mixed verdict and an all-fail verdict without a server. Insert this block immediately **after** the existing `print("smoke rejected:", ...)` statement, so the order on stdout is accepted, rejected, mixed, all-fail — which is the order the expected output below lists.

Three details matter, and all three are easy to get wrong. The fake answer must keep its code inside a fence (prose in the same string is a `SyntaxError` and every rollout fails the oracle — measured: the unfenced form gives `accepted 0 drops {'all_fail': 1}`, the exact opposite of what this step asserts). The prose must stay, though: the fenced code alone is 31 characters and would be rejected as `too_short`, so the fixture would test the length filter instead of the difficulty filter. And each case needs its own root, because a shared cache would replay the first case's rollouts.

````python
    class FlakyTeacher:
        """Passes the odd-numbered call, fails the even-numbered one.

        With `k=2` and one prompt that is one pass and one fail: the mixed verdict the
        filter is supposed to keep.
        """

        def __init__(self, always_fail: bool = False):
            self.calls = 0
            self.always_fail = always_fail

        def complete(self, messages, **kw):
            self.calls += 1
            # Code inside a fence, prose outside it. `verify.extract_code` with no fence
            # returns the whole completion, so prose in the same string reaches `exec` and
            # raises SyntaxError: every rollout fails the oracle and the "mixed" case
            # silently becomes an all-fail case instead of exercising the filter.
            good = ("```python\ndef add(a, b):\n    return a + b\n```\n\n"
                    "The function sums its two arguments and returns the result, so "
                    "add(1, 2) is 3.")
            bad = ("```python\ndef add(a, b):\n    return a - b\n```\n\n"
                   "This version subtracts the second argument from the first and returns "
                   "that value instead.")
            return {"content": bad if (self.always_fail or self.calls % 2 == 0) else good}

    def diff_case(teacher, name):
        dtmp = Path(tmp) / f"diff-{name}"
        canonical.write_jsonl(dtmp / "prompts" / "train.jsonl", [Prompt(
            id="d1", domain="coding", origin="prebuilt", source={"name": "smoke"},
            messages=[{"role": "user", "content": [{"type": "text", "text": "Write add."}]}],
            verify={"type": "python_tests", "setup": "",
                    "tests": ["assert add(1, 2) == 3"]})])
        st = run_difficulty(root=dtmp, client=teacher,
                            cache=GenCache(dtmp / "m2" / "cache.jsonl"), k=2,
                            concurrency=1)
        print(f"difficulty {name}: accepted {st['accepted']} "
              f"drops {dict(st['drops'])}", flush=True)
        return st

    mixed = diff_case(FlakyTeacher(), "mixed")
    failed = diff_case(FlakyTeacher(always_fail=True), "all-fail")
````

`write_jsonl` creates parent directories itself, so the `dtmp` tree needs no `mkdir`.

Run it. Expected:

```
difficulty mixed: accepted 1 drops {}
difficulty all-fail: accepted 0 drops {'all_fail': 1}
```

Both lines matter. `mixed` proves the branch that keeps a prompt, and `all-fail` proves the
branch that drops one — a filter that keeps everything would pass the first check alone.

Note that `dict(st["drops"])` prints `{}` rather than `{'all_pass': 0, 'all_fail': 0}`,
because a `Counter` does not retain zero-count keys. Assert on the keys you expect, not on
the absence of keys you do not.

To make the assertion exact rather than eyeballed, *replace* the existing `ok = ...` / `return 0 if ok else 1` with a version that keeps the seeded checks and adds the difficulty ones:

```python
    ok = (stats["accepted"] == 1
          and len(rejected) == 1
          and rejected[0]["reason"] == "verify_failed" and has_completion
          and mixed["accepted"] == 1 and not dict(mixed["drops"])
          and failed["accepted"] == 0 and failed["drops"]["all_fail"] == 1
          and validate(canonical.example_from_dict(first)) == [])
    return 0 if ok else 1
```

The ledger half is the part that must not be dropped: it is the only thing standing between a concurrency rewrite and a silently empty `m2/rejected.jsonl`.

- [ ] **Step 5: Smoke it**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate
```

Expected:

```
smoke accepted: 1 | valid: True
smoke rejected: 1 | reason: verify_failed | completion kept: True
difficulty mixed: accepted 1 drops {}
difficulty all-fail: accepted 0 drops {'all_fail': 1}
```

All four lines, in that order, and an exit code of 0 (`echo $LASTEXITCODE` prints `0`).

- [ ] **Step 6: Corroborate against the live teacher**

With the Colab session up, run the filter over a small oracle-bearing slice and confirm two things: the accepted count is strictly below the judged count, and the *shipped* corpus is smaller than before the filter ran. The second is the check that would have caught the Step 3 bug, so do it explicitly.

```powershell
$env:TEACHER_URL = "PASTE_THE_URL_HERE"
$env:TEACHER_API_KEY = "PASTE_THE_KEY_HERE"
$fix = "datasets/qwen35-4b-sft-smoke-diff"
if (-not (Test-Path $fix)) { New-Item -ItemType Directory -Path $fix | Out-Null; Copy-Item "datasets/qwen35-4b-sft-smoke/*" $fix -Recurse -Force }
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --mode difficulty --root $fix --base-url $env:TEACHER_URL --api-key $env:TEACHER_API_KEY --limit 12 --concurrency 8

$m = Get-Content "$fix/manifest.json" | ConvertFrom-Json
"judged  {0}" -f $m.m2.difficulty_judged
"removed {0}" -f $m.m2.difficulty_removed
"shipped {0}" -f $m.m2.difficulty_shipped
"train   {0}" -f $m.m2.train_written
(($m.m2.difficulty_removed -gt 0) -and ($m.m2.train_written -gt 0)) | ForEach-Object {
  "filter is live: $_" }
```

Expected: a `difficulty: accepted X of Y judged (k=2, drops {...})` line with `X <= Y`, then
`judged`/`removed`/`shipped`/`train` printed, then `filter is live: True`.

`removed 0` on a slice where some prompt passed every rollout means the merge filter is not
wired, which is the defect this task exists to prevent. A slice of 12 is small, so `removed
0` with `judged 12` is possible by chance — if it comes out zero, raise `--limit` to 40 and
re-check before concluding anything. `--mode difficulty` also re-runs `run_merge`, so
`train.jsonl` and the manifest are refreshed in one command.

- [ ] **Step 7: Seed the simulated loop from the invented tool schemas**

Spec §10.2 seeds a simulated trajectory from a `tools` schema "when the seed is a tool set
rather than a conversation". Today `run_simulated` reads only `prompts/trajectories.jsonl`,
so the schema rows Task 4 builds are invisible to it and that path does not exist. Widen the
seed selection and pass `first_user=None` for a seed that has no conversation yet —
`trajectory.generate_simulated` already has that invention path.

Replace `run_simulated`'s seed-collection and `work` blocks with this. As in Task 3 Step 3, the `root` parameter and every `_record_reject` call survive the rewrite — the ledger is what makes a schema-seeded drop distinguishable from a bad one.

```python
def run_simulated(*, root: Path, client, cache, limit: int | None, thinking: bool = True,
                  concurrency: int = 1):
    """Simulated trajectories (spec section 5.1): the teacher plays user, agent, and tool
    environment. Seeded two ways: from a prebuilt trajectory's own first user turn, and from
    a tool schema with no conversation at all, where the teacher invents the request
    (spec section 10.2). Rows carry `meta.simulated = true`."""
    from types import SimpleNamespace

    store = ImageStore(root / "images")
    stats = _traj_stats()
    stats["schema_seeded"] = 0

    prepared: list[tuple] = []
    traj_path = root / "prompts" / "trajectories.jsonl"
    traj_seeds = [t for t in (trajectory_from_dict(r) for r in iter_jsonl(traj_path))
                  if t.tools] if traj_path.exists() else []
    for t in traj_seeds:
        first_user = next((canonical.message_text(m) for m in t.messages
                           if m.get("role") == "user"), "")
        if first_user:
            # `sim-` matters: the raw id is the trajectory prompt's id, which
            # `m2/trajectory.jsonl` already uses, and `run_merge` concatenates both shards.
            prepared.append((f"sim-{t.id}", t.domain, dict(t.source), t.tools, first_user,
                             False))

    # Schema-seeded seeds. The Magpie row's own user turn *is* the schema-conditioned
    # request Task 4 invented, so it is passed straight through as `first_user` rather than
    # re-invented: `generate_simulated`'s own invention path uses a canned prompt and never
    # sees the row's `tools`, so re-inventing there would drop the schema from the loop
    # (spec section 10.2 asks for a user turn invented *from* a tool schema).
    mag_path = root / "prompts" / "magpie.jsonl"
    for row in (iter_jsonl(mag_path) if mag_path.exists() else []):
        if not row.get("tools") or not row.get("meta", {}).get("schema_seeded"):
            continue
        first_user = next((canonical.message_text(m) for m in row.get("messages", [])
                           if m.get("role") == "user"), "")
        if not first_user:
            continue
        prepared.append((f"sim-schema-{row['id']}", row["domain"],
                         {"name": "teacher:simulated:schema"}, row["tools"], first_user,
                         True))

    # `--sim-limit` bounds the whole simulated pool, so it is applied once, after both
    # sources are pooled. It is deliberately *not* applied to `traj_seeds` above: doing both
    # would downsample the prebuilt rows twice while the schema rows are capped once, and
    # the prebuilt:schema mix would skew. `downsample` reads only `.domain`, hence the wrapper.
    wrapped = [SimpleNamespace(domain=item[1], item=item, id=item[0]) for item in prepared]
    if limit is not None:
        wrapped = split.downsample(wrapped, limit)
    schema_count = sum(1 for w in wrapped if w.item[5])
    stats["schema_seeded"] = schema_count
    print(f"simulated seeds: {len(wrapped)} ({schema_count} schema-seeded)", flush=True)

    def work(w):
        sid, domain, source, tools, first_user, _schema = w.item
        try:
            return trajectory.generate_simulated(
                client, cache, id=sid, domain=domain, source=source, tools=tools,
                first_user=first_user, thinking=thinking, store=store), None
        except TeacherError as exc:
            return None, exc

    out = []
    for done, (w, (traj, exc)) in enumerate(zip(wrapped, _map(
            work, wrapped, concurrency)), start=1):
        sid, domain, source, tools, first_user, schema_seeded = w.item
        key = f"teacher:simulated|{domain}"
        # The reject ledger names the pool the row came from, so a schema-seeded drop is
        # distinguishable from a prebuilt-seeded one.
        pool = ("prompts/magpie.jsonl" if schema_seeded else "prompts/trajectories.jsonl")
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        if exc is not None:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {sid}: {exc}", flush=True)
            _record_reject(root, stage="simulated", reason="teacher_error", detail=str(exc),
                           pool=pool, id=sid)
        elif traj is None:
            stats["drops"]["turn_structure"] += 1
            _record_reject(root, stage="simulated", reason="turn_structure",
                           pool=pool, id=sid)
        else:
            reason, detail = _trajectory_reject(traj)
            if reason:
                stats["drops"][reason] += 1
                _record_reject(root, stage="simulated", reason=reason, detail=detail,
                               pool=pool, id=sid, record=traj.to_dict())
            else:
                out.append(traj)
                stats["accepted"] += 1
                stats["by_domain"][domain] += 1
                stats["accepted_by_source"][key] += 1
                stats["simulated"] += 1
        _progress("simulated", done, len(wrapped), stats, every=25)
    canonical.write_jsonl(root / "m2" / "simulated.jsonl", out)
    _dump_stats(root / "m2" / "simulated.stats.json", stats)
    print(f"simulated: accepted {stats['accepted']} of {stats['attempted']} "
          f"({schema_count} schema-seeded)", flush=True)
    return stats
```

Expected on the next live run: the `simulated seeds: N (M schema-seeded)` line with `M > 0`
after Task 7 Step 4 has built the Magpie pool, and `simulated: accepted ... (M schema-seeded)`.
`M == 0` means the schema rows are missing or lack `meta.schema_seeded`, and §10.2's
schema-seeded path is still unbuilt.

- [ ] **Step 8: Commit**

```bash
git add tools/dataset/generate.py
git commit -m "feat(dataset): spec 10.2 difficulty filter and schema-seeded simulated trajectories"
```

---

### Task 6: Re-anchor the caps on M2's measured pass rates

**Files:**
- Create: `tools/dataset/anchor.py`
- Modify: `tools/dataset/sources.py`

**Do not start until M2 Task 15 is accepted.** The anchor is M2's measurement; without it this task has no input.

Spec §10.1: "a domain at 0.7 pass rate needs 1.43× candidates for the same accepted count".

Be precise about which lever does what, because the intuitive reading is wrong and cost hours if you act on it:

- `sources.allocate()` splits a **fixed** per-domain budget (`DOMAIN_SHARE` × target) across the sources in that domain, in proportion to their caps. A cap therefore sets a source's *share of its domain*; it is not an absolute candidate bound, and raising one domain's caps does not add candidates to that domain by itself.
- `TEACHER_SOURCES` is read by nothing at runtime. It is a record of the teacher-only rows so the pool arithmetic stays visible.
- The domain budget itself comes from `--target-train`, and how many of those prompts actually get answered comes from the per-pool `--limit` flags.

The operative sequence is therefore: anchor the caps (this task), rebuild the prompt corpus at the anchored total (Task 7), then answer every candidate by setting `--limit` to the pool size (Task 8). Anchoring without the rebuild changes nothing, and the rebuild without the `--limit` change discards the new candidates. `anchor.py` prints the driver flags alongside the cap table so those three steps stay tied together.

**One cap cannot be reached at all.** The uncensored column is limited by supply, not by choice: `seeds/uncensored.jsonl` holds 2,500 rows, but the pipeline's dedup and filters keep only **1,939** of them, and Magpie invents at most one request per seed, so the candidate pool tops out at about **4,370** against a 5,000 *accepted* target. Spec §7.5 already calls this column "the tight one". Anchor for it anyway — the number records the intent — but expect a measured shortfall of roughly 12% even at a pass rate of 1.0, and record it as a shortfall rather than treating it as a bug.

- [ ] **Step 1: Write the anchoring module**

Create `tools/dataset/anchor.py`:

```python
"""Re-anchor M1's prompt caps on the pass rates M2 measured (spec section 10.1).

A source that passes at rate `p` needs about `1/p` times as many candidates to deliver the
same number of accepted examples. This module reads the per-source pass rates out of
`manifest.json`, aggregates them per domain, and prints the cap table to paste into
`sources.py`. It never edits a file: the table is a proposal and the edit is a review step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from . import sources
from .sources import DOMAIN_SHARE, SOURCES, TEACHER_SOURCES, Source


def per_domain_pass_rate(stats: dict) -> tuple[dict[str, float], dict[str, int]]:
    """Attempt-weighted pass rate per domain, plus the attempts it was computed from.

    Weighting by attempts is what makes the estimate usable: an unweighted mean over sources
    would let a 20-attempt source outvote a 3,000-attempt one. Returning the attempt counts
    alongside the rates is what lets `main` refuse to anchor on a sample too small to mean
    anything — M2's manifest holds 12 single-turn attempts in total.

    Single-turn shards only. Spec section 10.1: "Trajectory pass rates are reported
    separately from single-turn pass rates, since a trajectory passes only if every one of
    its turns passes" — folding them together would feed a multi-turn number into a
    single-turn pool's size.
    """
    accepted: Counter = Counter()
    attempted: Counter = Counter()
    for shard in ("seeded",):
        block = stats.get(shard) or {}
        for key, n in (block.get("attempted_by_source") or {}).items():
            domain = key.split("|")[-1]
            attempted[domain] += n
        for key, n in (block.get("accepted_by_source") or {}).items():
            domain = key.split("|")[-1]
            accepted[domain] += n
    rates = {d: (accepted[d] / attempted[d]) for d in attempted if attempted[d]}
    return rates, dict(attempted)


MIN_ANCHOR_ATTEMPTS = 200


def per_domain_caps_by_domain(rows: list[tuple[Source, int]]) -> dict[str, int]:
    """Anchored cap total per domain, which is the number `allocate` cannot see."""
    out: dict[str, int] = {}
    for source, cap in rows:
        out[source.domain] = out.get(source.domain, 0) + cap
    return out


def candidates_for(accepted_target: int, pass_rate: float) -> int:
    """Spec section 10.1: ceil(accepted / p). A zero pass rate cannot be anchored."""
    if pass_rate <= 0:
        raise ValueError("cannot anchor a source whose pass rate is 0")
    return math.ceil(accepted_target / pass_rate)


def anchored_caps(target_by_domain: dict[str, int], rates: dict[str, float],
                  table: list[Source]) -> list[tuple[Source, int]]:
    """`(source, anchored_cap)` for every source, so callers keep the pairing.

    Returned as `(Source, int)` pairs rather than parallel lists: the rows come back grouped
    by domain, so any caller that zips them against a differently-ordered sequence is
    silently wrong.
    """
    by_domain: dict[str, list[Source]] = {}
    for s in table:
        by_domain.setdefault(s.domain, []).append(s)
    out: list[tuple[Source, int]] = []
    for domain, group in by_domain.items():
        rate = rates.get(domain)
        target = target_by_domain.get(domain, 0)
        if rate is None or not target:
            out.extend((s, s.cap) for s in group)
            continue
        need = candidates_for(target, rate)
        total_cap = sum(s.cap for s in group)
        for s in group:
            # Integer ceiling: `(a + b - 1) // b`, so a cap is never rounded down by float
            # division. A cap must never shrink, hence the max() against the spec's value.
            scaled = (need * s.cap + total_cap - 1) // total_cap
            out.append((s, max(s.cap, scaled)))
    return out


def driver_flags(rows: list[tuple[Source, int]], val_target: int,
                 val_rate: float) -> dict[str, int]:
    """The driver flags the anchored table implies, so the steps stay tied together.

    `--target-train` cannot simply be `sum(anchored caps)`. `sources.allocate` splits the
    budget by the *fixed* `DOMAIN_SHARE` (30/30/20/20) and uses each cap only as that
    source's share *within* its domain, so scaling four domains' caps by four different
    factors cannot be expressed as one total — a domain whose cap grew would be under-served
    by the fixed share. Verified: anchoring then re-running `allocate` moves every quota by
    at most one row.

    The operative value is therefore the smallest total whose 30/30/20/20 split gives every
    domain at least what the anchor says it needs: `max_d(ceil(need_d / share_d))`. That is
    also the only form a rebuild can act on, since the pipeline takes one `--target-train`.

    `--val-size` and `--val-limit` are the same value on purpose: `--val-size` is how many
    val *candidates* the pipeline must build, `--val-limit` is how many of them the driver
    answers. Both are sized against the pass rate, because the val set is filtered too.
    """
    candidates = candidates_for(val_target, val_rate)
    needs = per_domain_caps_by_domain(rows)
    target = max((math.ceil(need / share) for domain, need in needs.items()
                  if (share := DOMAIN_SHARE.get(domain, 0)) > 0), default=sum(needs.values()))
    return {
        "target_train": target,
        "limit": target,
        "val_size": candidates,
        "val_limit": candidates,
        "difficulty_limit": target,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="re-anchor caps on M2's measured pass rates")
    ap.add_argument("--manifest", type=Path,
                    default=Path("datasets/qwen35-4b-sft/manifest.json"))
    ap.add_argument("--target-total", type=int, default=25_000,
                    help="accepted examples across all domains (spec section 4)")
    args = ap.parse_args(argv)

    stats = json.loads(args.manifest.read_text(encoding="utf-8")).get("m2") or {}
    rates, attempts = per_domain_pass_rate(stats)
    targets = {d: round(args.target_total * share)
               for d, share in sources.DOMAIN_SHARE.items()}
    print("measured pass rates:", {d: round(r, 4) for d, r in sorted(rates.items())})
    print("attempts behind them:", dict(sorted(attempts.items())))
    print("accepted targets:  ", targets)

    thin = {d: n for d, n in attempts.items() if n < MIN_ANCHOR_ATTEMPTS}
    if thin:
        raise SystemExit(
            f"refusing to anchor: {thin} have fewer than {MIN_ANCHOR_ATTEMPTS} attempts. "
            "Keep the spec section 7 caps (Task 6 Step 3 is a recorded no-op) and re-anchor "
            "after Task 8 phase 1 has measured a few hundred rows per domain. Measured on "
            "M2's manifest: 12 single-turn attempts, individual sources at 1-3 rows, and the "
            "pessimistic corner of those intervals implies a 128,000-candidate pool.")

    # One call over both tables. `coding` is fed by prebuilt sources *and* by
    # `teacher:tools`, so anchoring each table separately would apply that domain's
    # shortfall twice and roughly double coding's candidate budget. Combining them lets a
    # domain's requirement be split across every source that serves it.
    rows = anchored_caps(targets, rates, list(SOURCES) + list(TEACHER_SOURCES))
    print(f"{'dataset':<52} {'domain':<11} {'cap':>6} {'anchored':>9}")
    before = 0
    for source, anchored in rows:
        before += source.cap
        mark = "  <- change" if anchored != source.cap else ""
        print(f"{source.dataset:<52} {source.domain:<11} {source.cap:>6} "
              f"{anchored:>9}{mark}")
    after = sum(cap for _, cap in rows)
    print(f"\nnew candidate pool: {after} (was {before})")
    print("per-domain anchored caps:", dict(sorted(per_domain_caps_by_domain(rows).items())))
    print("  spec section 7's caps imply 32.1 / 33.3 / 19.2 / 15.4 percent, not 30/30/20/20,")
    print("  and `allocate` splits by the fixed DOMAIN_SHARE — so a domain's anchored total is")
    print("  only realised when `--target-train` >= need_d / share_d, which is what `flags`")
    print("  below computes.")
    print("spec section 7 target: 39,000 candidates for 25,000 accepted")

    # The val split is not anchored by the cap table — it is the same pass rate applied to a
    # 1,000-example target — and an unanchored `--val-limit` is the easiest way to end up
    # with a val set far below spec section 12. Weighted by attempts, not by the mean of the
    # four domain rates: the unweighted mean is the exact error `per_domain_pass_rate`'s
    # docstring warns about, and it sizes the val set off a 2-attempt domain.
    total_attempts = sum(attempts.values()) or 1
    weighted = sum(rates[d] * attempts[d] for d in rates) / total_attempts
    flags = driver_flags(rows, val_target=1000, val_rate=weighted)
    print("\nflags for Task 7 and Task 8 (from the table above):")
    print(f"  --target-train {flags['target_train']}")
    print(f"  --limit        {flags['limit']}")
    print(f"  --val-size     {flags['val_size']}   (the candidate count the pipeline must build)")
    print(f"  --val-limit    {flags['val_limit']}   (1,000 accepted at a "
          f"{weighted:.2f} blended rate)")
    print(f"  uncensored supply ceiling: 4,370 candidates "
          f"(1,939 seeds the pipeline keeps, plus at most one invention each)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    # No arguments: the synthetic smoke from Step 2 runs instead, so the arithmetic is
    # checkable without waiting on M2.
```

- [ ] **Step 2: Add a smoke with a synthetic measurement**

So the arithmetic is verified without waiting on M2, replace the `__main__` block you just wrote with this complete version. The `len(sys.argv) > 1` guard follows the convention the other modules use (`magpie.py`, `audit.py`): no arguments means smoke, arguments mean the real run.

```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())

    smoke_rates = {"reasoning": 0.70, "coding": 0.55, "roleplay": 0.80, "uncensored": 0.62}
    smoke_targets = {"reasoning": 7500, "coding": 7500, "roleplay": 5000, "uncensored": 5000}
    print("candidates_for(7500, 0.70):", candidates_for(7500, 0.70), "(expect 10715)")
    rows = anchored_caps(smoke_targets, smoke_rates,
                         list(SOURCES) + list(TEACHER_SOURCES))
    teacher_names = {s.dataset for s in TEACHER_SOURCES}
    by_domain: dict[str, int] = {}
    for source, cap in rows:
        by_domain[source.domain] = by_domain.get(source.domain, 0) + cap
    print("TEACHER_SOURCES anchored caps:",
          [cap for source, cap in rows if source.dataset in teacher_names],
          "(expect [1049, 8065])")
    print("no source shrinks:", all(cap >= source.cap for source, cap in rows))
    print("pool:", sum(cap for _, cap in rows), "(expect 41703)")
    print("meets every domain target:", all(by_domain.get(d, 0) >= t
                                            for d, t in smoke_targets.items()))
    print("per-domain anchored:", dict(sorted(by_domain.items())))
    print("driver flags:", driver_flags(rows, val_target=1000, val_rate=0.65))
```

Iterating `rows` as `(source, cap)` pairs is the point of the return type: an earlier version
returned parallel tuples and a caller zipped them against a differently-ordered sequence,
which is exactly the kind of check that passes for the wrong reason.

Run with no arguments so the synthetic path executes:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.anchor
```

Expected — every value below was checked against the real source table before this plan was written:

```
candidates_for(7500, 0.70): 10715 (expect 10715)
TEACHER_SOURCES anchored caps: [1049, 8065] (expect [1049, 8065])
no source shrinks: True
pool: 41703 (expect 41703)
meets every domain target: True
per-domain anchored: {'coding': 13638, 'reasoning': 12500, 'roleplay': 7500, 'uncensored': 8065}
driver flags: {'target_train': 45460, 'limit': 45460, 'val_size': 1539, 'val_limit': 1539, 'difficulty_limit': 45460}
```

The checks worth doing by hand, because the whole task turns on the arithmetic: `ceil(7500 / 0.70) = 10715`, `ceil(7500 / 0.55) = 13637`, `ceil(5000 / 0.62) = 8065`, `ceil(1000 / 0.65) = 1539`.

The two things this smoke is really guarding:

- **`1049`, not `1613`.** Anchoring `coding` in isolation would give `teacher:tools` a cap of 1,613 (the whole domain requirement squeezed into a 1,000-cap source). Combined with the five prebuilt coding sources, its share is its 1,000 out of the domain's 13,000, so 1,049. The paired-table version roughly doubles coding's budget for no reason, which is the bug this smoke pins down.
- **`41703`, not `39000`.** A pool larger than the spec's 39,000 is the intended outcome *when rates are below 1*: the spec's budget assumed prebuilt answers, so every response being teacher-written with a sub-1 pass rate grows the candidate pool by roughly `1/pass_rate`. With no rate below its domain threshold, nothing grows and the pool stays `39000` — which is what M2's data gives. The `45460` in the `driver flags` line is neither: it is `max_d(need_d / share_d)`, the smallest `--target-train` whose 30/30/20/20 split still funds every domain's anchored cap.

- [ ] **Step 3: Run it against the real M2 measurement**

The no-argument form runs the smoke, so pass `--manifest` explicitly to reach `main`:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.anchor --manifest datasets/qwen35-4b-sft/manifest.json --target-total 25000
```

Expected, on M2's manifest as it stands: the measured rates and the attempts behind them, then **`refusing to anchor:`** and exit 1, because every domain has fewer than 200 attempts (12 single-turn rows in total, sources at 1-3 rows each). That is this step's correct outcome, and it is why the task is a **recorded no-op**: keep the spec §7 caps unchanged, keep Task 7 and Task 8 at the default `--target-train 39000 --val-size 1067`, and re-anchor in Task 8 phase 2 once a few hundred rows per domain have been measured. Do not paper over the refusal with invented rates — Tasks 7 and 8 used to carry `41703`/`1539` derived from exactly that, which is the "numbers with no derivation" defect this plan criticises elsewhere.

If it does print a table (phase 2, after the measurement): a `<- change` mark means that cap grew; `per-domain anchored caps` is the number `allocate` cannot see; and the flags block's `--target-train` is `max_d(need_d / share_d)`, not the sum of the caps. Copy all five flag values into Task 7 Step 2 and Task 8 Step 3 — they go stale the moment one is changed without the others. If a domain's rate is 0, stop: `candidates_for` raises, and a genuine 0 means that source produced nothing usable — a measurement to discuss, not a number to paper over.

- [ ] **Step 4: Edit the caps, recording the anchor**

In `tools/dataset/sources.py`, update both tables with the anchored numbers. Extend `Source` so the anchoring is auditable rather than a bare integer — the M2 pass rate a cap was derived from belongs next to the cap:

```python
@dataclass(frozen=True)
class Source:
    dataset: str
    config: str | None
    split: str
    domain: str
    cap: int
    adapter: str
    teacher: bool = False          # True -> M2 territory, excluded from M1
    anchored_on: float | None = None   # M2 pass rate this cap was scaled by (spec 10.1)
```

Then set each changed `cap` from step 3's table and fill `anchored_on` with the domain rate used. Using Step 2's worked rates, the two teacher rows come out as:

```python
TEACHER_SOURCES: list[Source] = [
    Source("teacher:tools", None, "-", "coding", 1049, "teacher", teacher=True,
           anchored_on=0.55),          # its 1000 of coding's 13000 cap, scaled to 13637
    Source("teacher:uncensored", None, "-", "uncensored", 8065, "teacher", teacher=True,
           anchored_on=0.62),          # 5000 accepted / 0.62; supply-limited, see Task 7
]
```

(Those are Step 2's worked-example rates, not the values to ship. Ship the values step 3 printed, and the same for the prebuilt table. `anchored_on` is documentation: nothing reads it at runtime, so it is worth keeping only for the next person wondering where `8065` came from.)

- [ ] **Step 5: Assert the pool still covers the target**

Extend `sources.py`'s `__main__` so the smoke fails loudly if the re-anchored pool cannot deliver the spec's mix:

```python
    print("full pool:", pool, "(spec target 39000, M3 minimum)")
    anchored = {s.dataset: s.anchored_on for s in SOURCES + TEACHER_SOURCES
                if s.anchored_on}
    by_domain: dict[str, int] = {}
    for s in SOURCES + TEACHER_SOURCES:
        by_domain[s.domain] = by_domain.get(s.domain, 0) + s.cap
    print("per-domain caps:", dict(sorted(by_domain.items())))
    print("anchored sources:", anchored)
    # The sum can never fall below the original caps — every row is `max(s.cap, scaled)` —
    # so this assert alone proves nothing. The check that bites is per-domain: `allocate`
    # splits the budget by the fixed DOMAIN_SHARE, so `--target-train` must be at least
    # `max_d(need_d / share_d)` (anchor.py prints it) or a domain whose cap grew is still
    # under-served.
    assert pool >= 39_000, f"re-anchored pool {pool} is below the 39,000 candidate budget"
```

Run:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.sources
```

Expected: `M1 prebuilt caps: {...} = N`, `teacher caps: {...}`, `full pool: P (spec target 39000, M3 minimum)`, `per-domain caps: {...}`, `anchored sources: {...}`, then `M1 quota total: ...` and `first three: ...`, and no `AssertionError`.

Two things to read carefully, because both are easy to misread as failures:

- **On M2's data, expect `full pool: 39000` and `anchored sources: {}`.** That is Step 3's refusal honoured: no cap changed, so there is nothing to record. It is *not* evidence that Step 4 was skipped — the thing that would show that is a pool below 39,000, which the assert catches.
- **After phase 2 has re-anchored**, `anchored sources` must list exactly the rows Step 4 edited, and that is the check that the edit matched the printed table.

Note what the assert does and does not prove. It proves the pool still covers the spec's budget; it does *not* prove the pool can be filled: `SOURCES` caps are maxima per source, and the pipeline's `WARNING: short of target` line in Task 7 Step 2 is where a source that cannot reach its cap shows up — for `uncensored`, whose ceiling is 4,370 candidates rather than the 5,000 the earlier draft assumed.

- [ ] **Step 6: Commit**

```bash
git add tools/dataset/anchor.py tools/dataset/sources.py
git commit -m "feat(dataset): re-anchor source caps on M2's measured pass rates"
```

---

### Task 7: Rebuild the prompt corpus at M3 scale

**Files:**
- Modify: `datasets/qwen35-4b-sft/` (generated artefacts; not committed)

M3's accepted target is 25,000 examples, so the *candidate* pool has to be the re-anchored one from Task 6, not M1's 3,000-prompt milestone corpus. This task regenerates it and re-invents the Magpie pools at scale.

- [ ] **Step 1: Bring up the local student server for exact token counts**

The pipeline counts tokens exactly only if `llama-server` on 8085 is up; without it the manifest records a `chars/4` estimate. Start it as in the M1 rebuild guide:

```powershell
Start-Process -FilePath "C:\Users\tnmh\projects\local-llm\llama-cpp\llama-server.exe" `
  -ArgumentList @("-m","C:\Users\tnmh\projects\local-llm\models\qwen35-mtp\Qwen3.5-4B-MTP-Heretic.i1-Q4_K_S.gguf",
                  "-ngl","99","-fa","on","-c","8192","-np","1","--port","8085") `
  -WindowStyle Hidden
Start-Sleep -Seconds 20
Invoke-RestMethod http://127.0.0.1:8085/health
```

Expected: `status ok`.

- [ ] **Step 2: Rebuild the prompt corpus at the re-anchored caps**

`--target-train` comes from Task 6 Step 3's `flags` block, not from a round number. If Step 3
refused (M2's case), that is the default: `39000` with val flags of `1067` / `1067`. After
phase 2 re-anchors, use whatever the flags block printed.

`--val-size` is a **candidate** count, not an accepted one. `stratified_split` writes that
many val prompts and M3 accepts them at roughly the blended pass rate, so `--val-size 1000`
would deliver about 600-830 val rows against spec §12's 1,000. That is why the flags block
prints `--val-size` and `--val-limit` as the same number.

```powershell
$env:HF_TOKEN = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --target-train 39000 --val-size 1067 --root datasets/qwen35-4b-sft
```

`--target-train` is the *candidate* budget, not the accepted target: `allocate()` splits it
across the caps. The run streams tens of thousands of candidate rows with one tokenise call
per accepted prompt, so expect hours rather than minutes. A re-run repeats the same
deterministic work, so do not interrupt it for a progress check.

Expected: a `... kept N` line per source, then `prompts train T, val V, manifest written`,
and a `token_counter: exact` entry in the manifest's `run` block. Two things to read
carefully:

- `WARNING: short of target` is information, not failure. `uncensored` will be short for the
  supply reason in Task 6, and `diagram_image_to_text` ships only ~300 rows against a 1,000
  cap, so it is short too. Record both.
- `V` will be near the `--val-size` you passed only if the val pool is large enough. The pipeline derives val by `stratified_split`, so a small `V` means the candidate pool was thin, not that the flag is wrong. And `V` is a *candidate* count: the accepted val rows are what `run_seeded`'s val loop writes, which is `V` times the blended pass rate.

- [ ] **Step 3: Rebuild the multi-turn pools, then verify the corpus is prompt-only**

`prompts/trajectories.jsonl` is not produced by `pipeline` — it comes from `trajbuild.py`, and nothing else in this plan regenerates it. M2 left **170** rows, 50 of them tool-carrying, against spec §7.2/§7.3's 3,000 coding and 3,000 roleplay multi-turn candidates. Without this step M3 ships a 25,000-row corpus whose entire prebuilt multi-turn content is those 170 rows, and spec §5.1 is explicit that flattening multi-turn sources to a first user turn discards the half that matters.

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajbuild --root datasets/qwen35-4b-sft --limit-per-source 3000
```

Expected: one line per multi-turn source, then the row count. `trajbuild`'s own caps sum to 2,800 today, so its `SOURCES` table needs raising to §7.2/§7.3's 3,000/3,000/1,500/1,500 in the same commit — otherwise `--limit-per-source 3000` is silently capped below what you asked for and the multi-turn columns stay thin, which is the failure this step exists to prevent.

Then confirm the corpus is prompt-only and internally consistent:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset import canonical; from pathlib import Path; root=Path('datasets/qwen35-4b-sft'); rows=[canonical.prompt_from_dict(r) for r in canonical.iter_jsonl(root/'prompts'/'train.jsonl')]; bad=[p.id for p in rows if canonical.validate_prompt(p)]; print('prompts', len(rows), 'bad', len(bad), bad[:3]); print('with tools', sum(1 for p in rows if p.tools), '| with verify', sum(1 for p in rows if p.verify))"
```

Expected: `prompts N bad 0 []`, a non-zero `with tools` count (ToolACE and Hermes supply the schemas Task 4 needs), and a non-zero `with verify` count (the MBPP/APPS seeds and maths golds the difficulty filter needs).

Also print the two multi-turn pool sizes, because Task 8's command no longer carries a
hand-written cap for them and the expected `train` total depends on them:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset import canonical; from pathlib import Path; root=Path('datasets/qwen35-4b-sft'); print('trajectory prompts:', sum(1 for _ in canonical.iter_jsonl(root/'prompts'/'trajectories.jsonl'))); print('tool-carrying:', sum(1 for r in canonical.iter_jsonl(root/'prompts'/'trajectories.jsonl') if r.get('tools')))"
```

Expected: `trajectory prompts: M` and `tool-carrying: K` with `K <= M`. `K` is the number of
prebuilt rows the trajectory pass can attempt, and both are the reason Task 8 omits
`--multi-limit` (omitted means "every candidate") instead of naming a number that would have
to be maintained by hand.

- [ ] **Step 4: Re-invent the Magpie pools at scale**

With the Colab teacher up. `--limit` is per Magpie pool, and `--schema-limit` is the
tool-schema pass:

```powershell
$env:TEACHER_URL = "PASTE_THE_URL_HERE"
$env:TEACHER_API_KEY = "PASTE_THE_KEY_HERE"
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.magpie --root datasets/qwen35-4b-sft --base-url $env:TEACHER_URL --api-key $env:TEACHER_API_KEY --limit 2500 --schema-limit 1000
```

`--limit 2500` rather than 3,000 because that is what the uncensored pool can actually feed:
`seeds/uncensored.jsonl` holds 2,500 rows and `build` invents at most one request per seed.
Asking for 3,000 buys nothing and hides the constraint.

Expected: `magpie prompts: N -> ...\prompts\magpie.jsonl | {'teacher:uncensored': ~2500, 'teacher:tools': ~3500}` — the tools source is the request-seeded pool plus the schema-seeded pool — then a `dropped as duplicate, echoed or empty:` map. `N` materially below 6,000 means inventions were dropped or came back empty; that map is what tells you which pool.

- [ ] **Step 5: Confirm the schema-seeded rows carry their schemas**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset import canonical; from pathlib import Path; rows=[r for r in canonical.iter_jsonl(Path('datasets/qwen35-4b-sft/prompts/magpie.jsonl'))]; seeded=[r for r in rows if r.get('meta',{}).get('schema_seeded')]; print('magpie', len(rows), '| schema_seeded', len(seeded), '| all carry tools', all(r.get('tools') for r in seeded))"
```

Expected: `magpie N | schema_seeded ~1000 | all carry tools True`.

What that proves, and what it does not: it proves the schema-seeded rows are usable as
**single-turn** tool-use prompts, which is spec §7.2's "teacher (Magpie) invented tool-use
prompts" cap. It does **not** prove they feed the simulated loop — `run_simulated` reads its
seeds from `prompts/trajectories.jsonl`, so those rows are invisible to it by construction.
Task 5 extends `run_simulated` to also draw on the schema-seeded rows; if it was skipped,
say so in the guide rather than implying §10.2's schema-seeded simulated path was built.

- [ ] **Step 6: Commit**

Nothing here is committed: `datasets/` artefacts are gitignored, and the source tables were committed in Task 6. Confirm nothing *tracked* was modified:

```powershell
git status --short
```

Expected: no `M` lines. `datasets/` is gitignored, so the regenerated corpus, the manifest and
the prompts are invisible to git. The M3 plan document itself (and the unrelated `kitty-qat/`
work) are untracked and show as `??` — that is expected in this milestone, so this step is
checking for `M`, not for an empty output.

---

### Task 8: The full M3 run, the audit, and acceptance

**Files:**
- Create: `tools/dataset/audit.py`
- Modify: `docs/qwen35-4b-sft-rebuild.md` (the tracked guide; see Step 10 for why it is not a `datasets/` README)

- [ ] **Step 1: Write the sample-audit module**

The spec's alternative to an oracle for the un-oracled columns is a sample audit (§10.1) and a 20-example human spot-check (§11.7). This makes that repeatable and gives the README something concrete to point at.

Create `tools/dataset/audit.py`:

```python
"""Render a stratified sample of the finished corpus for human review.

Spec sections 10.1 and 11.7: roleplay and uncensored have no mechanical oracle at all, so the
substitute is a measured sample audit; coding is mixed, with the seed corpus carrying a
`verify` spec that already filtered it and the chat-style rows readable only by a human. This
module is that artefact: it writes one Markdown file with N examples per domain, and reports
how many of them came from the two fully un-oracled columns so the audit's coverage is
explicit rather than implied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import canonical

ORACLE_LESS = ("roleplay", "uncensored")


def sample(records: list, per_domain: int, seed: int = 17) -> dict[str, list]:
    import random

    rnd = random.Random(seed)
    by_domain: dict[str, list] = {}
    for ex in records:
        by_domain.setdefault(ex.domain, []).append(ex)
    out = {}
    for domain, items in by_domain.items():
        items = items[:]
        rnd.shuffle(items)
        out[domain] = items[:per_domain]
    return out


def render_markdown(records: dict[str, list], total: int) -> str:
    lines = ["# M3 sample audit", "",
             f"Corpus: {total} examples. Sampled uniformly at random per domain.",
             "Read each response and record a verdict in the table at the end.", ""]
    for domain in sorted(records):
        lines += [f"## {domain}", ""]
        for i, ex in enumerate(records[domain], start=1):
            lines.append(f"### {domain} {i} — `{ex.id}`")
            lines.append("")
            for m in ex.messages:
                role = m.get("role")
                content = m.get("content")
                text = content if isinstance(content, str) else "".join(
                    p.get("text", "") for p in content if p.get("type") == "text")
                if m.get("images") or (isinstance(content, list)
                                       and any(p.get("type") == "image" for p in content)):
                    text = f"[image] {text}"
                lines.append(f"**{role}:** {text}")
                lines.append("")
                if m.get("tool_calls"):
                    lines.append(f"**tool_calls:** `{m['tool_calls']}`")
                    lines.append("")
            lines += [f"- verdict: <!-- ok | fix | drop -->", ""]
    ours = sum(len(v) for k, v in records.items() if k in ORACLE_LESS)
    lines += ["## Coverage", "",
              f"- examples drawn from un-oracled columns: {ours}",
              f"- `roleplay` and `uncensored` have no oracle at all, so every row here is a",
              f"  judgement call rather than a re-check",
              f"- `coding` is mixed: the seed corpus carries a `verify` spec and was filtered by",
              f"  it, while the chat-style rows are read here like the other two",
              f"- nothing in this file overrides `verify.py`; it is a human signal, not a gate", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3 sample audit")
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/qwen35-4b-sft/train.jsonl"))
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path("datasets/qwen35-4b-sft/audit.md"))
    args = ap.parse_args(argv)
    records = [canonical.example_from_dict(r)
               for r in canonical.iter_jsonl(args.dataset)]
    picked = sample(records, args.per_domain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(picked, len(records)), encoding="utf-8")
    print("audit written:", args.out, "|",
          {d: len(v) for d, v in sorted(picked.items())})
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from .canonical import Example
        rows = [Example(id=f"r{i}", domain="roleplay", origin="teacher",
                        source={"name": "s"},
                        messages=[{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": f"answer {i}"}])
                for i in range(4)]
        rows[0].messages[1]["tool_calls"] = [{"id": "c", "type": "function",
                                              "function": {"name": "t",
                                                           "arguments": "{}"}}]
        picked = sample(rows, 2)
        md = render_markdown(picked, len(rows))
        print("sampled:", {d: len(v) for d, v in picked.items()})
        print("has tool_calls line:", "**tool_calls:**" in md)
        print("has verdict slot:", "- verdict:" in md)
        print("coverage line:", [line for line in md.splitlines()
                                 if "un-oracled" in line])
    else:
        sys.exit(main())
```

- [ ] **Step 2: Smoke the audit module offline**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.audit
```

Expected:

```
sampled: {'roleplay': 2}
has tool_calls line: True
has verdict slot: True
coverage line: ['- examples drawn from un-oracled columns: 2']
```

- [ ] **Step 3: Run M3 in two phases: measure, then spend**

The run is split because Task 6 could not re-anchor: M2 left 12 single-turn attempts, so the
caps are still the spec's §7 values and the pass rates are unknown. Phase 1 buys that
knowledge at a small share of the budget, and its calls are cached, so anything it generates
is free for phase 2.

**Phase 1.** Answer a few hundred candidates per domain and read the real rates. The numbers
below are illustrative of the shape, not a budget — size the slice so the smallest domain
gets at least ~200 attempts, which is the bar `anchor.py` now enforces:

```powershell
Start-Process -FilePath "$HOME\miniconda3\envs\dataset\python.exe" `
  -ArgumentList @("-m","tools.dataset.generate","--mode","all","--root","datasets/qwen35-4b-sft",
                  "--base-url",$env:TEACHER_URL,"--api-key",$env:TEACHER_API_KEY,
                  "--model","ornith-teacher","--concurrency","32","--timeout","1800",
                  "--cache","datasets/qwen35-4b-sft/m2/cache.jsonl",
                  "--limit","800","--val-limit","0","--magpie-limit","0") `
  -RedirectStandardOutput "$env:TEMP\opencode\m3-phase1.log" `
  -RedirectStandardError "$env:TEMP\opencode\m3-phase1.err" -WindowStyle Hidden
```

Then read `m2/seeded.stats.json` and the manifest: `pass_rate_by_source`, and the observed
**tokens/second** (total output tokens over wall-clock, from the log's first and last
timestamps). To turn a projection into a measurement, run two small slices through **two
separate caches** at two concurrency settings — the same slice replayed from a warm cache would
measure nothing:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --mode all --root datasets/qwen35-4b-sft --base-url $env:TEACHER_URL --api-key $env:TEACHER_API_KEY --limit 200 --val-limit 0 --magpie-limit 0 --cache "$env:TEMP\opencode\perf-32.jsonl" --concurrency 32
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --mode all --root datasets/qwen35-4b-sft --base-url $env:TEACHER_URL --api-key $env:TEACHER_API_KEY --limit 200 --val-limit 0 --magpie-limit 0 --cache "$env:TEMP\opencode\perf-48.jsonl" --concurrency 48
```

Take the better setting as phase 2's `--concurrency`. Decode on this model is weight-bound —
each step reads the 5.34 GB of weights, so aggregate throughput is roughly
`steps_per_second × batch` — which is why the plan's earlier `--concurrency 12` implied ~43 h
for the full pass while 32-48 implies 11-16 h. The server's `--max-num-seqs` must be at least
as large as the driver's concurrency, or it queues what the driver sends and the measurement
reports the lower number.

Two decisions come out of phase 1 and nothing else may overrule them:

- if a domain's rate is materially below its §7 threshold (reasoning < 0.60, coding < 0.577,
  roleplay < 0.667, uncensored < 0.833), re-anchor with `anchor.py` (Task 6 Step 3) and
  rebuild the corpus (Task 7) before continuing;
- if the measured throughput implies the full pass cannot finish inside the session budget,
  cut the candidate total now, while the corpus is cheap to rebuild, rather than discovering
  it in session three.

**Phase 2** is the full run. It is the same command with `--limit` set to the anchor's
`target_train` (or the §7 default `39000`) and the val flags from Task 6's `flags` block...

Three of them differ from the M2 handoff's sketch, deliberately:

- `--limit 39000` (or the re-anchored `target_train`). `--limit` is a per-pool *cap*, so a value below a pool's size silently answers only part of it. Note that it is per pool, not in total: with `TRAIN_POOLS` at three entries the candidates actually answered are the sum of the pool sizes, so record `attempted` from the manifest rather than assuming it equals this flag.
- `--magpie-limit 0`, because Task 7 Step 4 already built the Magpie pool. `--mode all`
  re-runs Magpie, and `write_jsonl` opens the file in `"w"`, so a non-zero limit here
  rewrites `prompts/magpie.jsonl` and can discard the schema-seeded rows.
- **No `--multi-limit` and no `--sim-limit`.** Earlier drafts carried `6000` and `3000` with
  no derivation behind them, which is the kind of number that goes stale silently. Since the
  flags now default to `None`, omitting them means "attempt every candidate in the pool" —
  which is exactly what spec §7 asks for ("The teacher answers every candidate") and needs no
  maintenance when Task 7's corpus changes size. Task 7 Step 3 prints both pool sizes so the
  expected `train` total is still checkable.

Run it detached with output redirected, so an idle session cannot kill it. This is the only
form of the command in this plan — do not also run a foreground copy:

```powershell
Start-Process -FilePath "$HOME\miniconda3\envs\dataset\python.exe" `
  -ArgumentList @("-m","tools.dataset.generate","--mode","all","--root","datasets/qwen35-4b-sft",
                  "--base-url",$env:TEACHER_URL,"--api-key",$env:TEACHER_API_KEY,
                  "--model","ornith-teacher","--concurrency","32","--timeout","1800",
                  "--cache","datasets/qwen35-4b-sft/m2/cache.jsonl",
                  "--limit","39000","--val-limit","1067","--magpie-limit","0") `
  -RedirectStandardOutput "$env:TEMP\opencode\m3-run.log" `
  -RedirectStandardError "$env:TEMP\opencode\m3-run.err" -WindowStyle Hidden
```

Expected, in order: `magpie: skipped, both limits are 0; the existing pool is untouched`, the
seeded pools' `... kept`/`seeded: accepted N of M`, `difficulty: accepted X of Y judged
(k=2, drops {...})`, `trajectories: accepted ...`, `simulated seeds: N (M schema-seeded)` and
`simulated: accepted ... (M schema-seeded)`, then `merged: train T, val V, pass_rate P`.

On the arithmetic, with the numbers the plan actually configures: `--limit` is per pool, so the
seeded pools answer roughly `39,000 + 1,001 + 6,000 ≈ 46,000` train candidates plus the val
pool, and 25,000 accepted out of that needs a blended rate near **0.54**, not 0.60. Two
consequences worth holding in mind while reading Step 4:

- The difficulty filter removes `p² + (1-p)²` of the oracle-bearing subset — `2p(1-p)` is the
  fraction it *keeps*, so at `p = 0.5` it removes half but at `p = 0.67` it removes 0.56, and
  at `p ≈ 1.0` it removes essentially all of it. The oracle subset is small (the 1,001 seeds
  plus ~74 maths golds), so this is a few hundred rows, not the reason the pool is sized where
  it is — but it means a very high reasoning rate shows up as a *smaller* mechanically
  verified corpus, which Step 4's `drops` line is there to reveal.
- If `train` lands over 25,000, that is expected rather than a bug: the filters do not reduce a
  high-rate corpus, so `run_merge` trims to §4's target by `DOMAIN_SHARE` weight and reports
  both the pre-trim and post-trim counts. If it lands *under* 25,000, read
  `pass_rate_by_source` before reacting: the fix is either a phase-2 re-anchor (a domain's rate
  is genuinely low) or a recorded shortfall (it is supply).

**The Colab session will die mid-run.** That is planned for: get a new URL, set it, and re-run
the identical command. The cache replays every completed call, so a killed session costs at
most the calls in flight. Two rules: do not delete the cache file (back it up in Step 11), and
do not re-run against a *stale* URL — the `/health` preflight refuses it rather than letting
every row become `teacher_error` and the merge overwrite the corpus with a gutted one.

- [ ] **Step 4: Read the pass rates and the drop reasons**

```powershell
$m = Get-Content datasets/qwen35-4b-sft/manifest.json | ConvertFrom-Json
$m.m2.pass_rate
$m.m2.seeded.pass_rate_by_source
$m.m2.difficulty.drops; $m.m2.difficulty.accepted
$m.m2.difficulty_judged; $m.m2.difficulty_removed; $m.m2.difficulty_shipped
$m.m2.trajectory.pass_rate_by_source; $m.m2.simulated.pass_rate_by_source
$m.train_final.by_domain; $m.train_final.example_share
$m.train_final.token_share; $m.train_final.image_bearing_share
```

Expected: a `pass_rate` in (0, 1); per-source maps keyed `dataset|domain`; difficulty `drops`
containing only `all_pass`, `all_fail`, `teacher_error`; `difficulty_removed > 0`; all four
domains present in `train_final.by_domain`; and a non-zero token share per domain.

**Then look at the oracle drop rate before accepting the number.** Compute
`(all_pass + all_fail) / difficulty_judged` and, separately, the ratio of `all_pass` to the
judged count. The filter is spec §10.2's rejection sampling and dropping all-pass rows is
correct behaviour — but the oracle-bearing set is only about 1,075 candidates (the 1,001
seeds plus the maths golds), so a reasoning pass rate near 1.0 deletes nearly all of it in one
step and leaves the corpus with almost no mechanically verified rows. That is a *recorded
decision point*, not a bug:

- below ~80% removed, ship as-is and note the rate;
- above it, say so explicitly in the guide and decide whether to ship the all-pass rows for
  the prompts the teacher solved. That change is **merge-only and costs no GPU time** — the
  cache holds the completions, so re-admitting them is a re-run of `--mode difficulty`
  followed by `--mode merge`, with no teacher calls. Do not silently ship a corpus whose
  verified core has been filtered to a few hundred rows.

Then apply spec §11.5's actual criterion, which is a tolerance rather than an eyeball:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import json; from pathlib import Path; m=json.loads(Path('datasets/qwen35-4b-sft/manifest.json').read_text(encoding='utf-8')); want={'reasoning':0.30,'coding':0.30,'roleplay':0.20,'uncensored':0.20}; got=m['train_final']['example_share']; print('target', want); print('actual', got); print('within +/-10pct', all(abs(got.get(d,0)-w) <= 0.10*w for d,w in want.items())); print('shortfall', {d: round(w-got.get(d,0),4) for d,w in want.items() if got.get(d,0) < w})"
```

Expected: `within +/-10pct` is `True`, or the `shortfall` map names which domains missed and
by how much.

A `False` here is a decision, not a bug: the uncensored column has a hard supply ceiling of
about 4,370 candidates against a 5,000 accepted target, so a shortfall there is expected and
is exactly what spec §7.5 predicts. Record it against §11.5 in the guide and, if the
shortfall is large enough to matter, rebalance the other domains' caps and re-run. Do not
"solve" it by loosening a filter — that changes what ships without a measurement behind it.

`image_bearing_share` will sit near 0.09 rather than §7.6's 0.12: M1 realises less than the estimate because the image configs are quota-limited, per the rebuild guide. Record the deviation rather than adjusting a filter to chase the number.

- [ ] **Step 5: Verify every accepted record and its image refs**

Run from the repo root, one line:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset import canonical; from tools.dataset.imgstore import ImageStore; from pathlib import Path; root=Path('datasets/qwen35-4b-sft'); store=ImageStore(root/'images'); rows=[canonical.example_from_dict(r) for rel in ('train.jsonl','val.jsonl') for r in canonical.iter_jsonl(root/rel)]; bad=[(ex.id) for ex in rows if canonical.validate(ex) or any(store.resolve(i.sha256) is None for i in ex.images)]; print('checked', len(rows), 'bad', len(bad), bad[:3])"
```

Expected: `checked N bad 0` — 100% schema-valid and every image sha resolves (spec §11.3).

- [ ] **Step 6: Confirm the difficulty filter actually removed rows**

The filter's claim is that it dropped every all-pass and all-fail prompt. Verify against the
*verdicts*, not the survivors — a check written against the survivors passes vacuously, which
is how the original version of this plan shipped a filter that did nothing:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import json; from pathlib import Path; from tools.dataset import canonical; root=Path('datasets/qwen35-4b-sft'); v=[json.loads(l) for l in (root/'m2'/'difficulty.verdicts.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; train={e.id for e in (canonical.example_from_dict(r) for r in canonical.iter_jsonl(root/'train.jsonl'))}; drop={x['id'] for x in v if x['verdict'] in ('all_pass','all_fail')}; keep={x['id'] for x in v if x['verdict']=='keep'}; print('judged', len(v)); print('dropped', len(drop), '| of those shipped anyway', len(drop & train)); print('kept', len(keep), '| of those missing', len(keep - train))"
```

Expected: `judged N`, `dropped D | of those shipped anyway 0`, `kept K | of those missing 0`.

Both zeros matter. The first is the bug this task fixes: a non-zero count means merge is not
consulting the verdicts. The second means the survivors are actually in the corpus. `D` equal
to `all_pass + all_fail` is the consistency check on the stats block.

- [ ] **Step 7: Export calibration chunks**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.calibrate --dataset datasets/qwen35-4b-sft/train.jsonl --out datasets/qwen35-4b-sft/calibration.txt --chunks 200
```

Expected: `calibration chunks written: N -> ...` with `N > 0`. Spec §3 sizes this at ~200 chunks for Phase 3's `llama-imatrix`. If `N` is below ~150, record the shortfall and the token total in the guide: imatrix quality scales with coverage, and the fix is more data, not a different chunk size.

- [ ] **Step 8: Re-run the render gate and the contamination guard**

Two separate render invocations. `--multiturn` returns from the multi-turn report before the
single-turn checks run, so one combined command cannot produce both sets of output:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft/train.jsonl --template tools/dataset/student.jinja --sample 200
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft/train.jsonl --template tools/dataset/student.jinja --sample 200 --multiturn
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.contaminate --dataset datasets/qwen35-4b-sft/train.jsonl --extra-eval datasets/qwen35-4b-sft/eval/refusal.jsonl
```

Expected:

- the first prints `0 template failures`, a balanced `think tags in rendered output` line,
  `probe think block survived: True`, `generation prompt opens thinking: True`;
- the second prints `multi-turn problems: 0` plus which per-turn think blocks survive;
- the third prints a `collisions:` count.

**Read the contaminate exit code, not just the counts.** `contaminate` returns non-zero for
NEAR collisions as well as exact ones, so a non-zero exit alongside a zero exact count is the
expected outcome, not a failed gate. The acceptance condition is the *exact* count:

**Acceptance on contamination: 0 exact collisions.** A small non-zero NEAR count is correct:
`seeds/uncensored.jsonl` and the quarantined `eval/refusal.jsonl` are two hash-selected slices
of one corpus, so spec §8's intended in-distribution overlap surfaces as NEAR. A non-zero
*exact* count is the failure — do not ship it, and do not "fix" it by loosening the guard.

- [ ] **Step 9: Generate the sample audit and read it**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.audit --dataset datasets/qwen35-4b-sft/train.jsonl --per-domain 10 --out datasets/qwen35-4b-sft/audit.md
```

Expected: `audit written: ... | {'coding': 10, 'reasoning': 10, 'roleplay': 10, 'uncensored': 10}`.

That is 40 rows, a deliberate superset of §11.7's "20 random examples rendered across all
domains". Say in the guide that it is the 10-per-domain superset and why (40 rows is still a
20-minute read, and the un-oracled columns are the ones worth the extra rows) — otherwise a
later reader compares 40 against the spec's 20 and assumes one of them is wrong.

Then **actually read `datasets/qwen35-4b-sft/audit.md`** and fill in the verdict slots. This is the step the skipped judge would have replaced, and it is the only quality signal the un-oracled columns get. Record the outcome — how many `ok`, `fix`, `drop`, and what the failures had in common — because it goes in the guide and it is the evidence for Task 6's caps next milestone.

- [ ] **Step 10: Update the rebuild guide**

Append a `## M3 — scale-out` section to `docs/qwen35-4b-sft-rebuild.md`, the tracked guide M2 already extended. Do **not** write it to `datasets/qwen35-4b-sft/README.md`: `.gitignore` line 8 is `datasets/`, so that file is untrackable and Step 11's `git add` would refuse it. The section contains:

- the Colab launch: which engine served the accepted smoke in Task 1 Step 6, the tokenizer
  repo it was started with, and the compute units it consumed;
- the driver command from Step 3 with the flags actually used and the `--concurrency` value;
- the measured `pass_rate`, per-domain example share and token share, and image-bearing share;
- the drop-reason table, with `all_pass` and `all_fail` labelled as the difficulty filter's
  expected rejections rather than defects, and `teacher_error` labelled as infrastructure;
- the uncensored shortfall from Step 4: the ceiling (~4,370 candidates), the realised accepted
  count, and that spec §7.5 predicted it;
- whether the simulated pass produced any schema-seeded rows, per Task 5 Step 7;
- the audit outcome from Step 9, including the un-oracled columns' coverage;
- the calibrate command and the chunk count;
- a note that the teacher was served from the *same* GGUF M2 used, so M2's pass rates remain
  the anchor;
- the deferred-judge note, verbatim: the roleplay/trajectory judge is deliberately not built,
  and the audit plus §11.7's spot-check stand in for it.

The final artefact is `train.jsonl` plus `val.jsonl` on your machine, which is where Phase 2A
reads it from. Spec §12's "the 2.7 GB artefact must leave the Colab runtime" is moot under
this architecture: nothing of the dataset ever lived on Colab. Say so in one line so a later
reader does not go looking for a Drive copy.

- [ ] **Step 11: Back up the irreplaceable state, then commit**

Everything that cost GPU time lives on this machine and only here: `m2/cache.jsonl` (every
completion), `images/`, `verification/seeds.jsonl`, `prompts/`, `train.jsonl`/`val.jsonl`,
`m2/difficulty.verdicts.jsonl` and `manifest.json`. None of it is in git — `.gitignore` line 8
is `datasets/` — and the cache is a single point of failure for both resume and re-derivation.
Copy the small, high-value ones somewhere that is not this disk (Drive, which is already
mounted for the teacher, or the HF Hub) before finishing:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$dest = "$HOME\Drive\m3-backup-$stamp"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item "datasets/qwen35-4b-sft/m2/cache.jsonl" $dest
Copy-Item "datasets/qwen35-4b-sft/manifest.json" $dest
Copy-Item "datasets/qwen35-4b-sft/m2/difficulty.verdicts.jsonl" $dest
Copy-Item "datasets/qwen35-4b-sft/train.jsonl","datasets/qwen35-4b-sft/val.jsonl" $dest
"backed up to $dest"
```

The images directory is the exception to "copy it all to Drive": it is content-addressed and
large, so archive it (`Compress-Archive`) rather than syncing thousands of small files.

Then commit the code and the guide:

```bash
git add tools/dataset/audit.py docs/qwen35-4b-sft-rebuild.md
git commit -m "docs(dataset): M3 scale-out results and sample audit"
```

---

## Deferred to a later milestone

Recorded here so the decision does not silently rot into "we forgot".

**Roleplay / trajectory judge.** Spec §10.1 and §10.2 call for judge-scored action coherence where no oracle exists — the roleplay trajectory is "judge-scored plus sample-audited" — and that is the requirement, not an option: the word "optional" nowhere attaches to the judge. It is deliberately not built, and this is the plan's largest conscious deviation from the spec. Reopening it requires: a second served model that is *not* the teacher (otherwise the judge shares the teacher's bugs and passes exactly the failures worth catching); a rubric prompt; a hand-labelled calibration set; and an agreement measurement before any threshold is allowed to gate data. Budget a comparable amount of teacher time again. The audit (`tools/dataset/audit.py`) is the cheap substitute, and Task 8 Step 9 is where its result is recorded — the guide section must state the deviation in those words rather than implying the judge was optional.

**Multi-turn robustness.** The spec accepts that SFT teaches format and first-turn behaviour, and that robustness needs a later RL/DPO stage (§13, §14). Not in scope here.

**`error_laden` tool results.** `verify.py` flags any tool result containing an `"error"` key; a well-formed API error is a legal observation. Measured at ~1.6% on ToolACE in M2. Narrowing the pattern to a top-level `error` field is a one-line change deferred until Task 8 Step 4 shows the rate in the trajectory drops.

**vLLM GGUF plugin.** If Task 1 Step 6 fell back to `llama.cpp`, the throughput figures in the guide are llama.cpp's. Re-measure before comparing M3's cost against the spec's §12 budget.

**The difficulty filter covers single-turn oracle prompts only.** `run_difficulty` requires a `verify` spec, so it applies to the MBPP/APPS seeds and the maths golds. Trajectories are excluded: their oracle lives on the final assistant turn, a multi-turn rollout costs N calls rather than 1, and `K` rollouts of the same prebuilt trajectory would mostly measure the prebuilt prefix. Widening it to trajectories is a real option, not an oversight, and it needs its own cost estimate first.

**Uncensored demand.** The column wants 5,000 accepted examples and the prompt pool tops out near 4,370 candidates. Closing the gap that filtering opens would mean inventing prompts without a seed — a different generator with a different quality story, not a cap tweak.

## Open risks carried into execution

| Risk | Why it is acceptable | What would change the plan |
|---|---|---|
| Colab prunes the runtime mid-run | The cache replays every completed call; the cost is the calls in flight | If losses exceed roughly one session per hour, move the driver onto Colab so the dataset is local to the teacher |
| Cloudflare's edge abandons a request that sends no bytes for ~100 s, which is every non-streaming completion long enough to matter | The client streams (`stream: true`, SSE) so bytes flow continuously, and Task 1 Step 6 proves a >100 s request through the tunnel before any bulk work | If a long request still fails through the tunnel, move the driver onto Colab; do not raise `--timeout`, which cannot help |
| The measured cost lands 2-6x over spec §12's 5-8 h / 20-30 units | Task 8 phase 1 measures the real pass rates and tokens/second before the full pass, and the candidate total can be cut while the corpus is cheap to rebuild | If phase 1's throughput cannot finish the full pass inside the remaining session budget, cut `--target-train` and record the shortfall against §4 rather than starting a fourth session |
| vLLM cannot map the `qwen3_5` hybrid architecture from GGUF, cannot see images, cannot return a reasoning trace, or will not pass a `tools` schema to the chat template | All four are canaried in Task 1 Step 6 before any bulk work, and `llama.cpp` serves the identical weights, projector and template | If neither engine passes all four canaries, stop. No vision deletes the §7.6 image column; no reasoning trace breaks spec §5.2 and every reasoning row; no `tools` deletes the trajectory and simulated columns. None of the three is substitutable with text data |
| The 12-hour session ceiling with a long ingest pass | Ingest (Task 7) is local and unaffected; generation (Task 8) is resumable | Split generation by pool via `--mode` and merge at the end |
| The uncensored column cannot reach its §4 share | Known before the run: the pool ceiling is ~4,370 candidates (1,939 kept seeds plus one invention each) for a 5,000 target, so even a pass rate of 1.0 lands ~12% short, and spec §7.5 flags the column as the tight one | Record the shortfall and rebalance the other domains' caps; inventing unseeded prompts is out of scope (see Deferred) |
| Any other domain lands under its §4 share | This is exactly what Task 8 Step 4's ±10% check measures, rather than smoothing over | Re-anchor that domain's cap with `anchor.py`, then re-run Tasks 7–8; do not tune a filter to hide it |
| The audit finds systematic roleplay failures | The audit exists precisely to find them | That is the trigger to build the deferred judge, with a proper calibration set |
| Teacher divergence between Colab and the local M2 teacher | Same GGUF, same sampling, same template; the engine differs, not the weights | If Task 8's pass rates differ from M2's by more than a few points, re-run an M2-sized slice on both engines and compare before scaling further |
| Batching changes numerics relative to the sequential path | Acceptance is order-independent by construction; only the sampled text can differ | Compare accepted counts, not text or `pass_rate`, when validating Task 3 Step 7 |

## Self-review

**Spec coverage.** §4 mix and the 25,000/1,000 split: Tasks 6–8. §7 caps and their re-anchoring on measured pass rates: Task 6. §10 teacher, sampling, resume, OpenAI-compatible serving with a different `--base-url`: Tasks 1, 3. §10.1 verification before training, per-source pass rates as the M3 cap guide: Tasks 5, 6, 8. §10.2 schema-conditioned invention, the 8-turn bounds (unchanged, `MAX_TOOL_TURNS`/`MAX_USER_TURNS`), the difficulty filter, and schema-seeded simulated trajectories: Tasks 4, 5. §11.1 contamination guard: Task 8 Step 8. §11.2 render gate: Task 8 Step 8. §11.3 validator and image resolution: Task 8 Step 5. §11.5 distribution report and its ±10% criterion: Task 8 Step 4. §7.6 vision and §14's unverified "`tools` through the Qwen3.5 template" path: the four canaries in Task 1 Step 6. §11.7 human spot-check: Task 8 Step 9 (10 per domain, a stated superset of the spec's 20). §12 M3 milestone, calibration and artefact persistence: Tasks 7, 8. Deliberately out of scope and recorded in "Deferred": §10.1/§10.2's judge — stated there as a deviation, because the spec does not mark it optional — §11.6 behavioural deltas (Phase 2A), the RL gap, and the difficulty filter's exclusion of trajectories.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code and names the exact lines it changes. Every run step states its expected output. The two steps that wait on measurement — Task 6 Step 4's cap values and Task 8's numbers — are deterministic procedures with the arithmetic, the command that prints the inputs, and a worked example, not blanks: `candidates_for` is asserted against three known values in Task 6 Step 2 (`10715`, `8065`, `1539`) and the resulting pool against a fourth (`41703`), so the scale factor is verified independently of M2. Server-dependent checks are confined to Task 1 Step 6, Task 3 Step 7, Task 5 Step 6 and Task 8; every other step runs offline.

**Type consistency.** `TeacherClient.__init__(base_url, model, timeout, retries, backoff, api_key)` is used consistently in `generate.py`, `magpie.py` and the module smokes. `GenCache.get/put/__len__` keep their signatures; `put` gains only locking and first-write-wins. `generate_one(prompt, client, cache, *, store, thinking, rollout=0)` matches every call site. `_map(fn, items, concurrency)` and `_progress(rel, done, total, stats, every)` are each defined once. `run_seeded`/`run_trajectory`/`run_simulated`/`run_difficulty`/`run_all` take `concurrency`, and `run_all` also takes `schema_limit`; `run_merge` is sequential by construction. `run_difficulty` returns a stats dict whose `drops` keys are `all_pass`, `all_fail` and `teacher_error`, and writes `difficulty.jsonl` plus `difficulty.verdicts.jsonl`, whose `verdict` values are `keep`, `all_pass`, `all_fail` and `teacher_error` — the same four strings `run_merge` filters on. `run_simulated`'s `prepared` tuples are `(id, domain, source, tools, first_user)` everywhere they are unpacked. `magpie.invent_from_schema(client, cache, *, schema, example, thinking, max_tokens)`, `magpie.build(domain, seeds, client, cache, *, limit, source_name, deduper)`, `magpie.build_schema_prompts(items, client, cache, *, limit, source_name, deduper)`, `magpie.run(*, root, client, cache, limit, dry_run, schema_limit=0)`, `magpie._prompt_id(domain, text)` and `magpie._extract_schemas(path)` are each defined once and called with matching shapes; `run_all`'s `magpie.run` call site keeps working because `schema_limit` defaults to 0. `anchor.per_domain_pass_rate`, `anchor.candidates_for`, `anchor.anchored_caps` (returning `(Source, int)` pairs) and `anchor.driver_flags` are each defined once and called with matching shapes. `audit.sample(records, per_domain, seed)` and `audit.render_markdown(records, total)` align with their call sites and the smoke. `Source.anchored_on` is a new optional field defaulting to `None`, so every existing construction keeps working. Three invariants are stated because breaking any of them is silent: `_run_pool`, `run_trajectory` and `run_simulated` all keep their `root` parameter and their `_record_reject` calls, so every drop still lands in `m2/rejected.jsonl` with its completion — `generate.smoke()` asserts exactly that, and Task 3 Step 6 checks the printed rejected line as well as the exit code. The four pool limits (`--limit`, `--val-limit`, `--multi-limit`, `--sim-limit`) all default to `None`, and the guards are `if limit is not None`, so `0` means none and omitted means every candidate; `--magpie-limit` is the exception and keeps an integer default because `magpie.build` compares it. `TeacherClient`'s `timeout` default is `1800` in both the constructor and the CLI, because a timeout is terminal and 900 killed rows in M2. `_trajectory_reject(traj)` is the only trajectory-rejection helper — there is no boolean `_accept_trajectory` — and `anchor.driver_flags` returns `val_size` alongside `val_limit` so Task 7's pipeline and Task 8's driver cannot disagree about the val split. Every other existing construction site keeps working. The four invariants added by the adversarial review, each of which is silent when broken: the client streams (`stream: true`) so a long completion survives the tunnel; the cache stores the *completion* and the verdict is recomputed on replay, so a filter edit applies to the whole cache; `GenCache` reloads on `"\n"` and skips unparseable lines, so one U+2028 does not make the cache unloadable; and `--magpie-limit` defaults to `0` with a refuse-to-shrink guard, so a forgotten flag cannot replace a 6,000-row pool with 40.

**Known risk.** `_run_pool`'s worker returns a verdict that the main thread folds into `stats`; if a later change moves any `stats` mutation inside a worker, the statistics become order-dependent and `--concurrency 8` will stop matching `--concurrency 1`. Task 3 Step 7 is the check that catches it, and it is deliberately a comparison of accepted counts rather than a timing measurement. The second: `run_difficulty` treats a `TeacherError` on *any* rollout as a drop of the whole prompt. That is the conservative choice — a prompt whose rollouts were interrupted has no measured difficulty — but it means a flaky endpoint shows up as difficulty drops rather than as errors. Task 8 Step 4 reads `drops` for exactly that reason; a `teacher_error` count that scales with the number of Colab sessions is an infrastructure signal, not a data-quality one. The third: the correctness of `run_merge` now depends on `difficulty.verdicts.jsonl` existing. If that file is deleted while `difficulty.jsonl` survives, merge falls through to the seeded rows and the filter silently stops applying — Task 8 Step 6's `dropped` check is what catches it, which is why it asserts against the verdicts rather than the survivors. The fourth: the reject ledger is an invariant, not a side effect. `_record_reject` is called from the sequential fold of all three stages, and the plan's concurrency rewrite is the one change that could quietly delete it — a `_run_pool` without `root` still compiles and still reports the same accepted counts, so nothing but `generate.smoke()` and Task 3 Step 6's rejected line would notice. If a future edit needs the ledger elsewhere, move the calls; do not drop them. The fifth: the difficulty filter's verdict file is only valid for the corpus that produced it, which is why it now carries an `oracle_in_pools` fingerprint and `run_merge` refuses to filter without a match. A partial `--mode difficulty` run (Task 5 Step 6 against the real root instead of `$fixture`) would otherwise re-admit every row the full pass had dropped while dropping the survivors it never saw — and Task 8 Step 6's own check would still pass, because it validates the verdict file against itself.
