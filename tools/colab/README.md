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
2. The notebook will ask to connect your Drive. Approve it; you may need to copy an auth
   code back into the cell.
3. Run the launcher cell. The tokenizer repo is fixed for this teacher — Task 1 Step 5 read it out of the GGUF:

   ```
   !python /content/serve_teacher.py --tokenizer XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
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

- `CUDA out of memory` on load — another runtime is still alive. **Runtime → Manage
  sessions**, terminate the others, retry.
- The launcher exits with a weight-mapping error — the GGUF plugin could not map this
  architecture. Re-run with `--engine llama-cpp`.
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
