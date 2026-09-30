"""Append-only resume cache. One JSONL line per completed request; re-runs replay.

Spec section 10: re-running is idempotent. Single-turn rows key on the request payload;
a trajectory keys on the whole conversation prefix sent so far, so an interrupted
trajectory resumes at its next missing turn rather than restarting.

ponytail: keys cover `{kind, messages}` only. Changing a sampling parameter or the `tools`
field between runs replays the cached completion rather than regenerating. That is fine
while the spec fixes sampling; if a run needs a different parameter set, delete the cache.
"""
from __future__ import annotations

import hashlib
import json
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
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, key: str, value: dict) -> None:
        rec = {"key": key, **value}
        self.records[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.records)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "cache.jsonl"
    c = GenCache(tmp)
    k1 = prefix_key([{"role": "user", "content": "hi"}], kind="k")
    c.put(k1, {"completion": {"content": "hello"}})
    c.put(prefix_key([{"role": "user", "content": "yo"}], kind="k"),
          {"completion": {"content": "hey"}})

    replay = GenCache(tmp)
    print("records after reload:", len(replay))
    print("replayed value:", replay.get(k1)["completion"]["content"])
    print("same key twice:", prefix_key([{"role": "user", "content": "hi"}], kind="k") == k1)
