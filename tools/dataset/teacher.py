"""OpenAI-compatible client for the local teacher served by llama-server (spec section 10).

The teacher runs without speculative decoding, so it is served on its own port. The M2
driver is strictly sequential, so one slot (`-np 1`) suffices; batched serving is M3's
Colab problem. Requests carry `chat_template_kwargs.enable_thinking`, matching the
invocation the project already uses for the student bench. Images go as base64 data URIs,
resolved from the content-addressed store by sha256.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request

from .canonical import normalize_tools

# Spec section 10 sampling, with the anti-repetition pair the teacher's parent model
# publishes (ornith-ai/Ornith-1.5-9B): presence_penalty 1.5 for general tasks and 0.0 for
# precise coding, min_p 0.0 (not llama.cpp's 0.05 default). The teacher inherits
# Ornith-1.5's agentic half and, with no anti-repetition pressure at all, loops on
# open-ended prompts (measured: one 505-char prompt generated 21,038 tokens without
# stopping, and a second ran past the client timeout).
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5


class TeacherError(RuntimeError):
    pass


def _fold_reasoning(message: dict) -> dict:
    """Fold a server-split `reasoning_content` back into `content` as a `<think>` block.

    llama-server's default `--reasoning-format auto` extracts the model's thoughts into a
    separate `message.reasoning_content` field and leaves `content` as the answer only.
    The canonical schema (spec section 5.2) stores reasoning verbatim *inside* `content`
    as `<think>...</think>`, and the response filters, the render gate, and both chat
    templates all expect it there — `student.jinja` reads `reasoning_content` only as a
    fallback. Fold it back at this single transport boundary so no caller can drop it.
    """
    reasoning = message.get("reasoning_content")
    content = message.get("content") or ""
    if isinstance(reasoning, str) and reasoning.strip() and "<think>" not in content:
        message = {**message,
                   "content": f"<think>\n{reasoning.strip()}\n</think>\n\n{content}".rstrip()}
    return message


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """The server's own message for an HTTP error.

    `str(HTTPError)` is only "HTTP Error 400: Bad Request"; llama.cpp's diagnosis (a
    tool-call/tool-response mismatch, an over-long context, a schema it still dislikes) is
    carried in the response body, which urllib never reads for us. Drain it here so the
    failure is diagnosable instead of opaque.
    """
    try:
        body = (exc.read() or b"").decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    return f"HTTP {exc.code}: {body[:500]}" if body else f"HTTP {exc.code}: {exc.reason or exc}"


def to_wire_messages(messages: list[dict], store=None) -> list[dict]:
    """Canonical messages -> OpenAI chat messages, resolving image shas to data URIs."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        wire: dict = {"role": m["role"]}
        if isinstance(content, str):
            wire["content"] = content
        elif isinstance(content, list):
            parts: list[dict] = []
            for p in content:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image":
                    if store is None:
                        raise TeacherError("image part present but no image store was given")
                    path = store.resolve(p["sha256"])
                    if path is None:
                        raise TeacherError(f"image {p['sha256']} does not resolve in the store")
                    mime = mimetypes.guess_type(path.name)[0] or "image/png"
                    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{mime};base64,{b64}"}})
            wire["content"] = parts
        else:
            wire["content"] = ""
        if m.get("tool_calls"):
            wire["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            wire["tool_call_id"] = m["tool_call_id"]
        out.append(wire)
    return out


class TeacherClient:
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
        fix it, because the edge gives up regardless. Measured through the tunnel: a
        4000-word reply returned `http 200 | 131.4s | 1.39 MB` of SSE.
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
        attempts = 0
        detail = "no attempt made"
        for attempt in range(1, self.retries + 1):
            attempts = attempt
            try:
                return self._stream(payload) if stream else self._post(path, payload)
            except urllib.error.HTTPError as exc:
                # HTTPError subclasses URLError, so catch it first to read the body before
                # anything else touches it. A 4xx is deterministic (bad request, context
                # too long, a schema the grammar builder rejects): replaying the identical
                # request burns the retry budget for nothing, so it is terminal. 5xx may be
                # the edge or the tunnel and keeps the retries.
                detail = _http_error_detail(exc)
                if exc.code < 500:
                    break
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as exc:
                detail = str(exc)
                # A timeout means the teacher is still generating. Retrying replays the
                # same deterministic request and burns the same wall-clock again
                # (measured: 3 x 900s = 45 min lost to one uncapped coding row), so a
                # timeout is terminal. Retries stay for connection-level failures.
                if isinstance(exc, TimeoutError):
                    break
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise TeacherError(f"{path} failed after {attempts} attempt(s): {detail}")

    def props(self) -> dict:
        return self._get("/props")

    def chat_template(self) -> str:
        template = self.props().get("chat_template")
        if not template:
            raise TeacherError("server returned no chat_template in /props")
        return template

    def complete(self, messages: list[dict], *, store=None, tools=None, thinking: bool = True,
                 max_tokens: int | None = None, temperature: float = TEMPERATURE,
                 top_p: float = TOP_P, top_k: int = TOP_K, seed: int | None = None) -> dict:
        """One teacher turn. Returns the assistant message dict (content, tool_calls)."""
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
            payload["tools"] = normalize_tools(tools)
        if seed is not None:
            payload["seed"] = seed
        reply = self._post_retry("/v1/chat/completions", payload, stream=True)
        return _fold_reasoning(reply)


if __name__ == "__main__":
    import io
    import os
    import sys
    import tempfile
    from pathlib import Path

    from PIL import Image

    from .imgstore import ImageStore

    # Offline check first (no server needed): the HTTP-error body must be captured, or a
    # 400 degrades to "HTTP Error 400: Bad Request" -- which is exactly what hid the
    # toolace diagnosis during the M3 run.
    import urllib.error
    _err = urllib.error.HTTPError(
        "http://x/v1/chat/completions", 400, "Bad Request", {},
        io.BytesIO(b'{"error":{"message":"tool call count mismatch"}}'))
    print("http error detail:", _http_error_detail(_err))
    _err_nobody = urllib.error.HTTPError(
        "http://x/v1/chat/completions", 500, "Internal Server Error", {}, io.BytesIO(b""))
    print("http error detail (no body):", _http_error_detail(_err_nobody))

    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8086"
    client = TeacherClient(url, api_key=os.environ.get("TEACHER_API_KEY") or None)
    info = client.props()
    print("model:", str(info.get("model_path", "?"))[-48:])
    print("template chars:", len(client.chat_template() or ""))
    msg = client.complete(
        [{"role": "user", "content": [{"type": "text", "text": "Reply with the single word: ok"}]}],
        thinking=False, max_tokens=16)
    print("reply:", repr((msg.get("content") or "").strip())[:60])

    # Image canary (spec section 10): the base64 data-URI path must work before bulk
    # generation, not be assumed. A solid-colour PNG is enough to prove the round trip.
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (200, 30, 30)).save(buf, format="PNG")
    store = ImageStore(Path(tempfile.mkdtemp()) / "images")
    sha, _, _, _ = store.put(buf.getvalue())
    msg = client.complete(
        [{"role": "user", "content": [
            {"type": "text", "text": "What single colour fills this image? One word."},
            {"type": "image", "sha256": sha}]}],
        store=store, thinking=False, max_tokens=16)
    print("image reply:", repr((msg.get("content") or "").strip())[:60])
