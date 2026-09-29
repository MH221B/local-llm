"""Client for a local llama-server: /props for the chat template, /tokenize for counts."""
from __future__ import annotations

import json
import urllib.request


class LlamaServer:
    def __init__(self, base_url: str = "http://127.0.0.1:8085"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat_template(self) -> str:
        props = self._get("/props")
        template = props.get("chat_template")
        if not template:
            raise RuntimeError("server returned no chat_template in /props")
        return template

    def encode(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text}).get("tokens", [])


class ServerTokenizer:
    """Adapter matching textutil.set_tokenizer's expected .encode(text).ids shape."""

    class _Ids:
        def __init__(self, ids: list[int]):
            self.ids = ids

    def __init__(self, server: LlamaServer):
        self.server = server

    def encode(self, text: str) -> "ServerTokenizer._Ids":
        return ServerTokenizer._Ids(self.server.encode(text))


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
    server = LlamaServer(url)
    template = server.chat_template()
    print("template chars:", len(template))
    print("ids for 'hello world':", len(server.encode("hello world")))
