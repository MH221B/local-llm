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
