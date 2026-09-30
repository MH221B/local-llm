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

# Spec section 10 sampling. No max_tokens cap: the teacher must fit its reasoning trace
# AND the answer into one completion, and a 4096 cap truncated many rows mid-reasoning
# with no answer at all. The request omits `max_tokens` so the model stops on its own.
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


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
                 model: str = "local-teacher", timeout: int = 900,
                 retries: int = 3, backoff: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post_retry(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._post(path, payload)
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise TeacherError(f"{path} failed after {self.retries} attempts: {last}")

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
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        reply = self._post_retry("/v1/chat/completions", payload)
        return _fold_reasoning(reply["choices"][0]["message"])


if __name__ == "__main__":
    import io
    import sys
    import tempfile
    from pathlib import Path

    from PIL import Image

    from .imgstore import ImageStore

    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8086"
    client = TeacherClient(url)
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
