# Phase 1 / M1 — Prompt corpus pipeline (teacher answers in M2) Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the M1 half of the dataset pipeline: ingest the frozen prebuilt sources as **user prompts only**, normalise them into the canonical prompt schema, dedup, filter, split into a four-domain prompt mix, and emit `prompts/train.jsonl` / `prompts/val.jsonl` plus a manifest — with no teacher inference and no prebuilt assistant text anywhere. Every response the student eventually trains on is written by the 9B teacher in M2; M1 also builds the unit-test verification seeds and checker that filter the teacher's failed attempts before they can reach the student.

**Architecture:** A stage pipeline over one canonical prompt record. One adapter module per source shape parses the source (including its prebuilt answers, where the format needs them for shape validation); one shared step, `strip_to_prompt()`, discards every prebuilt assistant turn before anything is persisted. Images ride along as raw bytes on content parts and get materialised into a content-addressed store by one shared step. Dedup, filtering, splitting, and reporting only ever see `Prompt`. A single frozen source table drives everything, so M3 is this same pipeline plus M2's teacher rows and a larger `--target-train`. Distillation hygiene is built in: `verification/seeds.jsonl` attaches unit tests or gold answers to prompts, and `verify.check_record` runs a teacher completion against them (the student must not inherit the teacher's bugs), while think tags are stored verbatim so the student mimics the teacher's reasoning process.

**Tech Stack:** Python 3.12 in a conda environment named `dataset` (miniconda at `$HOME\miniconda3`); `datasets` for streaming ingest, `pillow` for image decode, `jinja2` for the render gate. `llama-server` (already installed at `projects/local-llm/llama-cpp/`) supplies the student's real chat template and exact token counts; the abliterated 9B teacher is served in M2 (spec §10).

**Testing:** No new unit tests (not opted in). Every task verifies by running its own `__main__` smoke check or the pipeline CLI and reading the stated output. The spec's §11 verification gates are Tasks 14–16 and 18, implemented as code rather than left as advice.

**How to run a module:** conda environments run modules directly; smoke checks use the env's interpreter from the repo root:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.<module> [args...]
```

The run blocks below use that form.

**Spec:** `docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md`

---

## Scope

This plan covers **M1 only** — the prompt half. Three things in the spec are deliberately
deferred to their own plans:

| Deferred | Why |
|---|---|
| Teacher generation (`generate.py`, M2) | Consumes `prompts/` and `verification/seeds.jsonl` and must obey the M2 contract at the end of this plan |
| Cauldron VQA reference answers | The VQA configs are the §8 vision probe (Task 17) and keep their prebuilt short answers as eval labels; every other prebuilt response is discarded |
| Colab scale-out (M3) | The same CLI with M2's teacher rows and a larger `--target-train`; an operational runbook, not new code |

## File structure

```
tools/dataset/
  __init__.py
  canonical.py        Example / Prompt / ImageRef, validators, JSONL read+write
  imgstore.py         content-addressed store; materialise_images()
  textutil.py         normalising, hashing, minhash, loop detection, token counting
  dedup.py            exact + near-duplicate (minhash banding) index
  filters.py          prompt predicates (M1) + response predicates (M2)
  domains.py          keyword fallback table + assign_domain()
  split.py            source-stratified, image-disjoint split
  report.py           manifest builder
  sources.py          the frozen source table + quota allocation
  tokenserver.py      llama-server /props and /tokenize client
  render.py           the chat-template render gate (spec §11.2)
  contaminate.py      eval-set disjointness guard (spec §11.1)
  calibrate.py        calibration chunk exporter for Phase 3 (runs after M2)
  seeds.py            uncensored prompt seeds + 500-row refusal holdout
  testsets.py         unit-test verification seeds (MBPP, APPS)
  verify.py           execute a teacher completion against a verify spec
  pipeline.py         CLI: fetch -> normalise -> strip to prompt -> dedup -> filter -> split -> report
  adapters/
    __init__.py       ADAPTERS registry
    base.py           Adapter dataclass, Ctx, message helpers
    smoltalk.py       messages shape
    sharegpt.py       conversations shape (OpenHermes-2.5, ToolACE)
    plainmath.py      NuminaMath-1.5, gsm8k
    codefeedback.py   m-a-p/CodeFeedback-Filtered-Instruction
    toolcalls.py      hermes-function-calling-v1
    cauldron.py       Cauldron configs (images; VQA builder for the holdout)
```

Every module has a `__main__` smoke check so verification is one command with a
predictable line of output.

---

### Task 1: Profile, package scaffold, canonical schema

**Files:**
- Create: `tools/dataset/__init__.py`
- Create: `tools/dataset/canonical.py`

- [x] **Step 0: Ensure the project is a git repository (one-time)**

Every task commits, so version control must exist before Task 1's commit.

```powershell
git rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0) { git init }
```

Expected: the check succeeds, or `Initialized empty Git repository in ...`.

- [x] **Step 1: Create the conda environment and install dependencies**

```powershell
& "$HOME\miniconda3\Scripts\conda.exe" create -y -n dataset python=3.12
& "$HOME\miniconda3\envs\dataset\python.exe" -m pip install datasets pillow jinja2
```

- [x] **Step 2: Verify the environment imports its dependencies**

Run:
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "import datasets, PIL, jinja2; print('deps ok')"
```
Expected: `deps ok`

- [x] **Step 3: Create the package marker**

Create `tools/dataset/__init__.py` as an empty file.

- [x] **Step 4: Write `tools/dataset/canonical.py`**

```python
"""Canonical records. Adapters emit Example; the M1 pipeline persists Prompt (spec section 5)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

DOMAINS = ("reasoning", "coding", "roleplay", "uncensored")
ORIGINS = ("prebuilt", "teacher")
PART_TYPES = ("text", "image")
ROLES = ("system", "user", "assistant")

# Opening pleasantries with no request in them. Compared lowercased with trailing
# "!" / "." stripped. See _is_bare_greeting for why this exists.
_BARE_GREETINGS = frozenset({
    "hi", "hi there", "hello", "hello there", "hey", "hey there",
    "good morning", "good afternoon", "good evening", "how are you",
    "how are you doing", "how's it going", "greetings", "yo",
})


@dataclass
class ImageRef:
    sha256: str
    path: str
    w: int
    h: int
    origin_url: str | None = None

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "path": self.path, "w": self.w,
                "h": self.h, "origin_url": self.origin_url}


@dataclass
class Example:
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    images: list[ImageRef] = field(default_factory=list)
    tools: Any = None
    tokens: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "origin": self.origin,
            "source": self.source,
            "messages": self.messages,
            "tools": self.tools,
            "images": [i.to_dict() for i in self.images],
            "tokens": self.tokens,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Prompt:
    """A user-side record: leading system turn + first user turn, images and tools attached.

    M2 stems the teacher completion onto `messages` (same content-part shape), runs the
    attached `verify` spec, and only then may the row become training data.
    """
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    images: list[ImageRef] = field(default_factory=list)
    tools: Any = None
    tokens: int = 0
    verify: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "origin": self.origin,
            "source": self.source,
            "messages": self.messages,
            "tools": self.tools,
            "images": [i.to_dict() for i in self.images],
            "tokens": self.tokens,
            "verify": self.verify,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _is_bare_greeting(m: dict) -> bool:
    """True for an opener that carries no request, e.g. smoltalk's "Hi there".

    everyday-conversations opens with a greeting handshake before the topic; keeping
    that turn as the prompt yields a 1-2 token record that prompt_too_short rejects,
    so the source would contribute nothing. Checked against the dataset: every one of
    270 sampled rows opened with one of the five strings below and all had a later
    user turn. Text-only and short by construction, so a real request never matches.
    """
    content = m.get("content")
    if isinstance(content, list):
        if any(p.get("type") != "text" for p in content):
            return False
        text = "".join(p.get("text", "") for p in content)
    elif isinstance(content, str):
        text = content
    else:
        return False
    return text.strip().lower().rstrip("!.") in _BARE_GREETINGS


def strip_to_prompt(ex: Example) -> Prompt | None:
    """Reduce an adapter Example to its prompt: leading system + first user turn.

    Every prebuilt assistant turn is discarded here, before anything is written: the
    source datasets supply prompts only, and M2's teacher supplies every response.
    A leading greeting turn is dropped when a later user turn exists, so the prompt
    carries an actual request. Returns None when the row has no usable user turn.
    """
    messages = list(ex.messages)
    # Drop a leading greeting handshake (greeting, assistant reply, greeting, ...) so
    # the scan starts at the first turn that carries a request. A leading system turn
    # is preserved, so the greeting is found at index 1 in that case.
    while True:
        i = 1 if messages and messages[0].get("role") == "system" else 0
        if len(messages) > i + 1 and messages[i].get("role") == "user" \
                and _is_bare_greeting(messages[i]) \
                and any(n.get("role") == "user" for n in messages[i + 1:]):
            del messages[i]
            if i < len(messages) and messages[i].get("role") == "assistant":
                del messages[i]
        else:
            break

    kept: list[dict] = []
    for m in messages:
        if m.get("role") == "system" and not kept:
            kept.append(m)
            continue
        if m.get("role") == "user":
            kept.append(m)
            used = {p["sha256"] for msg in kept
                    for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                    if p.get("type") == "image" and "sha256" in p}
            return Prompt(
                id=ex.id, domain=ex.domain, origin=ex.origin, source=ex.source,
                messages=kept, images=[i for i in ex.images if i.sha256 in used],
                tools=ex.tools, tokens=ex.tokens, verify=ex.meta.get("verify"), meta=ex.meta)
        return None
    return None


def image_shas(ex: Example | Prompt) -> list[str]:
    """sha256 of every image part in the example or prompt, in order."""
    out: list[str] = []
    for m in ex.messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image" and "sha256" in part:
                    out.append(part["sha256"])
    return out


def validate(ex: Example) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    if not ex.id:
        problems.append("empty id")
    if ex.domain not in DOMAINS:
        problems.append(f"bad domain: {ex.domain!r}")
    if ex.origin not in ORIGINS:
        problems.append(f"bad origin: {ex.origin!r}")
    if not isinstance(ex.source, dict) or "name" not in ex.source:
        problems.append("source missing name")
    if not ex.messages:
        problems.append("no messages")
        return problems

    roles = [m.get("role") for m in ex.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "assistant":
        problems.append("last message is not assistant")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")

    refs = {i.sha256 for i in ex.images}
    for m in ex.messages:
        content = m.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            problems.append("content is neither str nor list")
            continue
        for part in content:
            ptype = part.get("type")
            if ptype not in PART_TYPES:
                problems.append(f"bad part type: {ptype!r}")
            if ptype == "image" and part.get("sha256") not in refs:
                problems.append(f"image part {part.get('sha256')!r} has no ImageRef")

    if ex.tools is not None and not isinstance(ex.tools, list):
        problems.append("tools is neither list nor null")
    if ex.tokens < 0:
        problems.append("negative token count")
    return problems


def validate_prompt(p: Prompt) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    if not p.id:
        problems.append("empty id")
    if p.domain not in DOMAINS:
        problems.append(f"bad domain: {p.domain!r}")
    if p.origin not in ORIGINS:
        problems.append(f"bad origin: {p.origin!r}")
    if not isinstance(p.source, dict) or "name" not in p.source:
        problems.append("source missing name")
    if not p.messages:
        problems.append("no messages")
        return problems

    roles = [m.get("role") for m in p.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "user":
        problems.append("last message is not user")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")
    if "assistant" in roles:
        problems.append("prebuilt assistant turn was not stripped")

    refs = {i.sha256 for i in p.images}
    for m in p.messages:
        content = m.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            problems.append("content is neither str nor list")
            continue
        for part in content:
            ptype = part.get("type")
            if ptype not in PART_TYPES:
                problems.append(f"bad part type: {ptype!r}")
            if ptype == "image" and part.get("sha256") not in refs:
                problems.append(f"image part {part.get('sha256')!r} has no ImageRef")

    if p.tools is not None and not isinstance(p.tools, list):
        problems.append("tools is neither list nor null")
    if p.tokens < 0:
        problems.append("negative token count")
    if p.verify is not None and not isinstance(p.verify, dict):
        problems.append("verify is neither dict nor null")
    return problems


def write_jsonl(path: Path, records: Iterable[Example | Prompt]) -> int:
    count = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    good = Example(
        id="smoke-1", domain="reasoning", origin="prebuilt",
        source={"name": "smoke", "row": 0, "license": None},
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                  {"role": "assistant", "content": "hello"}],
    )
    bad = Example(
        id="smoke-2", domain="nope", origin="prebuilt",
        source={"name": "smoke"},
        messages=[{"role": "user", "content": "hi"}],
    )
    prompt = strip_to_prompt(good)
    print("good:", validate(good))
    print("bad:", len(validate(bad)), "problems")
    print("prompt:", validate_prompt(prompt), [m["role"] for m in prompt.messages])
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.canonical
```
Expected: `good: []`, `bad: 2 problems` — `nope` is not a domain, and the only message is a `user` turn, so the example has no assistant reply — then `prompt: [] ['user']`, proving the prebuilt answer reduces to a prompt.

- [x] **Step 5: Commit**

```powershell
git add tools/dataset/__init__.py tools/dataset/canonical.py
git commit -m "feat(dataset): canonical example schema and validator"
```

---

### Task 2: Content-addressed image store

**Files:**
- Create: `tools/dataset/imgstore.py`

- [x] **Step 1: Write `tools/dataset/imgstore.py`**

```python
"""Content-addressed image store, plus the one step that moves raw bytes into it."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from .canonical import Example, ImageRef

EXT_BY_FORMAT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


class ImageStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_for(self, sha: str) -> Path:
        return self.root / sha[:2]

    def resolve(self, sha: str) -> Path | None:
        d = self.dir_for(sha)
        if not d.is_dir():
            return None
        for p in sorted(d.glob(f"{sha}.*")):
            return p
        return None

    def put(self, data: bytes) -> tuple[str, int, int, str]:
        """Store bytes, return (sha256, width, height, path relative to the store root)."""
        sha = hashlib.sha256(data).hexdigest()
        existing = self.resolve(sha)
        if existing is not None:
            with Image.open(existing) as im:
                return sha, im.width, im.height, str(existing.relative_to(self.root))
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "PNG").upper()
            w, h = im.width, im.height
        ext = EXT_BY_FORMAT.get(fmt, ".png")
        target = self.dir_for(sha) / f"{sha}{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return sha, w, h, str(target.relative_to(self.root))

    def put_pil(self, im: Image.Image) -> tuple[str, int, int, str]:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return self.put(buf.getvalue())


def image_part(data: bytes, origin_url: str | None = None) -> dict:
    """A not-yet-materialised image content part. Adapters emit these."""
    return {"type": "image", "data": data, "origin_url": origin_url}


def materialise_images(ex: Example, store: ImageStore) -> Example:
    """Replace every {'type':'image','data':...} part with a sha256 reference and fill ex.images.

    Idempotent: refs that were already materialised are preserved rather than cleared.
    """
    refs: list[ImageRef] = list(ex.images)
    for m in ex.messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        rebuilt: list[dict] = []
        for part in content:
            if part.get("type") == "image" and "data" in part:
                sha, w, h, rel = store.put(part["data"])
                refs.append(ImageRef(sha, rel, w, h, part.get("origin_url")))
                rebuilt.append({"type": "image", "sha256": sha})
            else:
                rebuilt.append(part)
        m["content"] = rebuilt

    seen: set[str] = set()
    unique: list[ImageRef] = []
    for r in refs:
        if r.sha256 not in seen:
            seen.add(r.sha256)
            unique.append(r)
    ex.images = unique
    ex.meta["n_images"] = len(unique)
    return ex


if __name__ == "__main__":
    import tempfile

    from .canonical import validate

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, format="PNG")
    payload = buf.getvalue()

    tmp = Path(tempfile.mkdtemp())
    store = ImageStore(tmp / "images")
    ex = Example(
        id="smoke-img", domain="reasoning", origin="prebuilt",
        source={"name": "smoke"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "what colour?"},
            image_part(payload),
        ]}, {"role": "assistant", "content": "dark blue"}],
    )
    materialise_images(ex, store)
    # storing the same bytes twice must deduplicate
    sha_a = ex.images[0].sha256
    materialise_images(
        Example(id="again", domain="reasoning", origin="prebuilt", source={"name": "smoke"},
                messages=[{"role": "user", "content": [image_part(payload)]},
                          {"role": "assistant", "content": "x"}]), store)
    files = [p for p in store.root.rglob(f"{sha_a}.*")]
    print("refs:", ex.images[0].w, "x", ex.images[0].h, "files:", len(files), "valid:", validate(ex))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.imgstore
```
Expected: `refs: 64 x 48 files: 1 valid: []` — one file on disk despite two identical stores.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/imgstore.py
git commit -m "feat(dataset): content-addressed image store"
```

---

### Task 3: Text utilities

**Files:**
- Create: `tools/dataset/textutil.py`

- [x] **Step 1: Write `tools/dataset/textutil.py`**

```python
"""Normalising, hashing, near-duplicate signatures, loop detection, token counting."""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

_WS = re.compile(r"\s+")
_MERSENNE = 2147483647
_PLANES: list[tuple[int, int]] = []
_TOKENIZER = None


def set_tokenizer(tok) -> None:
    """Install an object exposing .encode(text).ids; used for exact token counts."""
    global _TOKENIZER
    _TOKENIZER = tok


def normalise(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text).ids)
    return max(1, len(text) // 4)


def _planes(k: int, seed: int = 1) -> list[tuple[int, int]]:
    global _PLANES
    if len(_PLANES) != k:
        rnd = random.Random(seed)
        _PLANES = [(rnd.randrange(1, _MERSENNE), rnd.randrange(1, _MERSENNE))
                   for _ in range(k)]
    return _PLANES


def _shingle_hash(s: str) -> int:
    """Stable shingle hash: no PYTHONHASHSEED salt, so signatures are reproducible."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _shingles(text: str, n: int = 5) -> set[int]:
    t = normalise(text)
    if len(t) < n:
        return {_shingle_hash(t)}
    return {_shingle_hash(t[i:i + n]) for i in range(len(t) - n + 1)}


def minhash(text: str, k: int = 64) -> tuple[int, ...]:
    shingles = _shingles(text)
    return tuple(min((a * x + b) % _MERSENNE for x in shingles) for a, b in _planes(k))


def banded(sig: tuple[int, ...], bands: int = 16) -> list[tuple[int, ...]]:
    """Split a signature into bands; equal bands are near-duplicate candidates."""
    width = len(sig) // bands
    return [tuple(sig[i * width:(i + 1) * width]) for i in range(bands)]


def jaccard_est(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def repeating_ngram_ratio(text: str, n: int = 8) -> float:
    """Fraction of word n-grams that are repeats. High values mean degenerate looping."""
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    repeats = sum(c - 1 for c in grams.values() if c > 1)
    return repeats / len(grams)


if __name__ == "__main__":
    a = "The quick brown fox jumps over the lazy dog, again and again."
    b = "the QUICK brown fox jumps   over the lazy dog, again and again."
    c = "Completely unrelated content about marine biology and tides."
    print("hash_equal:", prompt_hash(a) == prompt_hash(b))
    print("near_ab:", round(jaccard_est(minhash(a), minhash(b)), 3))
    print("near_ac:", round(jaccard_est(minhash(a), minhash(c)), 3))
    loop = " ".join(["step one two three four five six seven"] * 20)
    print("loop_ratio:", round(repeating_ngram_ratio(loop), 3))
    print("tokens_est:", count_tokens(a))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.textutil
```
Expected: `hash_equal: True` (the two strings differ only in case and whitespace, which is what `normalise` collapses), `near_ab` above 0.5, `near_ac` well below it, `loop_ratio` above 0.05 and at most 1.0, and a non-zero estimated token count.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/textutil.py
git commit -m "feat(dataset): hashing, minhash, loop detection, token counting"
```

---

### Task 4: Adapter base and registry

**Files:**
- Create: `tools/dataset/adapters/__init__.py`
- Create: `tools/dataset/adapters/base.py`

- [x] **Step 1: Write `tools/dataset/adapters/base.py`**

```python
"""Adapter protocol and helpers shared by every source module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..canonical import Example


@dataclass
class Ctx:
    """Per-source run context handed to an adapter's build()."""
    name: str
    config: str | None
    domain: str
    license: str | None = None

    def source(self, row_index: int | None = None) -> dict:
        s = {"name": self.name, "license": self.license}
        if self.config is not None:
            s["config"] = self.config
        if row_index is not None:
            s["row"] = row_index
        return s


@dataclass(frozen=True)
class Adapter:
    """One entry per (dataset, config). `build` returns None to drop the row."""
    key: str
    build: Callable[[dict, int, Ctx], Example | None]


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def user_msg(parts: list[dict]) -> dict:
    return {"role": "user", "content": parts}


def assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def system_msg(text: str) -> dict:
    return {"role": "system", "content": text}


_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
             "user": "user", "assistant": "assistant"}

_MARKUP = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>")


def clean_markup(text: str) -> str:
    for token in _MARKUP:
        text = text.replace(token, "")
    return text.strip()


def sharegpt_turns(conversations: list, *, system: str | None = None) -> list[dict] | None:
    """Convert ShareGPT turns to canonical messages. Accepts from/value or role/content."""
    messages: list[dict] = []
    if system and system.strip():
        messages.append(system_msg(system.strip()))
    for turn in conversations:
        raw_role = turn.get("from") or turn.get("role")
        text = turn.get("value")
        if text is None:
            text = turn.get("content")
        role = _ROLE_MAP.get(raw_role)
        if role is None or not isinstance(text, str):
            return None
        messages.append({"role": role, "content": [text_part(clean_markup(text))]})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages


def plain_pair(question: str, answer: str) -> list[dict]:
    return [user_msg([text_part(question.strip())]), assistant_msg(answer.strip())]
```

Adapters parse the full source shape — some formats need the prebuilt answer for shape
validation — but the pipeline immediately reduces every row to its prompt with
`canonical.strip_to_prompt()`. No prebuilt assistant text is ever written to any artifact.

- [x] **Step 2: Write `tools/dataset/adapters/__init__.py`**

```python
"""Adapter registry. Keys match the adapter column in sources.py."""
from __future__ import annotations

from .base import Adapter
from . import cauldron, codefeedback, plainmath, sharegpt, smoltalk, toolcalls

ADAPTERS: dict[str, Adapter] = {
    "smoltalk": Adapter("smoltalk", smoltalk.build),
    "sharegpt_openhermes": Adapter("sharegpt_openhermes", sharegpt.build_openhermes),
    "sharegpt_toolace": Adapter("sharegpt_toolace", sharegpt.build_toolace),
    "numina": Adapter("numina", plainmath.build_numina),
    "gsm8k": Adapter("gsm8k", plainmath.build_gsm8k),
    "codefeedback": Adapter("codefeedback", codefeedback.build),
    "toolcalls": Adapter("toolcalls", toolcalls.build),
    "cauldron": Adapter("cauldron", cauldron.build),
}


def get(key: str) -> Adapter:
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter {key!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]
```

The package will not import until Tasks 5–9 land. That is expected; Task 5 completes it
enough to smoke-test one path.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/adapters/
git commit -m "feat(dataset): adapter protocol, helpers, and registry"
```

---

### Task 5: smoltalk adapter (messages shape)

**Files:**
- Create: `tools/dataset/adapters/smoltalk.py`

- [x] **Step 1: Write `tools/dataset/adapters/smoltalk.py`**

```python
"""HuggingFaceTB/smoltalk configs. Already in the `messages` shape (spec section 6)."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, text_part

# Spec section 7.3 asks for the smol-magpie-ultra slice to be filtered on
# `quality` / `reward_model_score`. Observed values run excellent/good/average/poor,
# with scores in the 0.15-0.20 band.
MAGPIE_MIN_SCORE = 0.15
MAGPIE_QUALITIES = ("excellent", "good")


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    if ctx.config == "smol-magpie-ultra":
        score = row.get("reward_model_score")
        if row.get("quality") not in MAGPIE_QUALITIES:
            return None
        if not isinstance(score, (int, float)) or score < MAGPIE_MIN_SCORE:
            return None
    messages: list[dict] = []
    for m in row.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": [text_part(content)]})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    label = ctx.config or "default"
    return Example(
        id=f"smoltalk-{label}-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


if __name__ == "__main__":
    from .base import Ctx

    ctx = Ctx("HuggingFaceTB/smoltalk", "metamathqa-50k", "reasoning", "apache-2.0")
    row = {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
    ]}
    ex = build(row, 7, ctx)
    print(ex.id, ex.domain, [m["role"] for m in ex.messages])
    print("dropped:", build({"messages": [{"role": "user", "content": "x"}]}, 0, ctx))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.smoltalk
```
Expected: `smoltalk-metamathqa-50k-7 reasoning ['user', 'assistant']` then `dropped: None`.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/adapters/smoltalk.py
git commit -m "feat(dataset): smoltalk adapter"
```

---

### Task 6: ShareGPT adapters (OpenHermes-2.5, ToolACE)

**Files:**
- Create: `tools/dataset/adapters/sharegpt.py`

- [x] **Step 1: Write `tools/dataset/adapters/sharegpt.py`**

```python
"""`conversations` shaped sources: OpenHermes-2.5 (roleplay slice) and ToolACE."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, sharegpt_turns


def build_openhermes(row: dict, index: int, ctx: Ctx) -> Example | None:
    if row.get("category") != "roleplay":
        return None
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system_prompt"))
    if messages is None:
        return None
    return Example(
        id=f"openhermes-roleplay-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


def build_toolace(row: dict, index: int, ctx: Ctx) -> Example | None:
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system"))
    if messages is None:
        return None
    return Example(
        id=f"toolace-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
    )


if __name__ == "__main__":
    ctx = Ctx("teknium/OpenHermes-2.5", None, "roleplay", "apache-2.0")
    rp = {"category": "roleplay", "system_prompt": "You are Ada.",
          "conversations": [{"from": "human", "value": "Hello<|im_end|>"},
                            {"from": "gpt", "value": "Hi there."}]}
    ex = build_openhermes(rp, 3, ctx)
    sys_content = ex.messages[0]["content"]
    sys_text = sys_content if isinstance(sys_content, str) else sys_content[0]["text"]
    print(ex.id, [m["role"] for m in ex.messages], repr(sys_text))
    print("non-roleplay dropped:", build_openhermes({"category": "coding", "conversations": []}, 1, ctx))
    print("unterminated dropped:", build_openhermes(
        {"category": "roleplay", "conversations": [{"from": "human", "value": "hi"}]}, 2, ctx))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.sharegpt
```
Expected: the id, `['system', 'user', 'assistant']`, the system prompt text with markup
stripped, then `None` twice.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/adapters/sharegpt.py
git commit -m "feat(dataset): OpenHermes and ToolACE adapters"
```

---

### Task 7: Plain-column adapters (NuminaMath-1.5, gsm8k, CodeFeedback)

**Files:**
- Create: `tools/dataset/adapters/plainmath.py`
- Create: `tools/dataset/adapters/codefeedback.py`

- [x] **Step 1: Write `tools/dataset/adapters/plainmath.py`**

```python
"""NuminaMath-1.5 and gsm8k. Plain columns plus the validity flags (spec section 7.1)."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, plain_pair


def build_numina(row: dict, index: int, ctx: Ctx) -> Example | None:
    # NuminaMath-1.5 ships these as strings ("Yes" / "Incomplete" / "No"), not bools.
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return None
    problem, solution = row.get("problem"), row.get("solution")
    if not problem or not solution:
        return None
    return Example(
        id=f"numina-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(problem, solution),
        meta={"problem_type": row.get("problem_type"), "synthetic": row.get("synthetic")},
    )


def build_gsm8k(row: dict, index: int, ctx: Ctx) -> Example | None:
    question, answer = row.get("question"), row.get("answer")
    if not question or not answer:
        return None
    return Example(
        id=f"gsm8k-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(question, answer),
    )


if __name__ == "__main__":
    ctx = Ctx("AI-MO/NuminaMath-1.5", None, "reasoning", "apache-2.0")
    good = {"problem": "Solve x.", "solution": "x = 1", "problem_is_valid": "Yes",
            "solution_is_valid": "Yes", "problem_type": "algebra", "synthetic": False}
    bad = {"problem": "Solve y.", "solution": "y = 2", "problem_is_valid": "Incomplete",
           "solution_is_valid": "Yes"}
    print(build_numina(good, 5, ctx).id, build_numina(good, 5, ctx).meta)
    print("invalid dropped:", build_numina(bad, 6, ctx))

    gctx = Ctx("openai/gsm8k", "main", "reasoning", "mit")
    print(build_gsm8k({"question": "1+1?", "answer": "#### 2"}, 1, gctx).messages[-1]["content"])
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.plainmath
```
Expected: `numina-5 {'problem_type': 'algebra', 'synthetic': False}`, `None`, then `#### 2`.

- [x] **Step 2: Write `tools/dataset/adapters/codefeedback.py`**

```python
"""m-a-p/CodeFeedback-Filtered-Instruction: query/answer/resource/lang columns."""
from __future__ import annotations

from ..canonical import Example
from .base import Ctx, plain_pair


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    query, answer = row.get("query"), row.get("answer")
    if not query or not answer:
        return None
    return Example(
        id=f"codefeedback-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(query, answer),
        meta={"lang": row.get("lang"), "resource": row.get("resource")},
    )


if __name__ == "__main__":
    ctx = Ctx("m-a-p/CodeFeedback-Filtered-Instruction", None, "coding", "apache-2.0")
    row = {"query": "Write a reverse function.", "answer": "def rev(s): return s[::-1]",
           "lang": "python", "resource": "codefeedback"}
    ex = build(row, 9, ctx)
    print(ex.id, ex.meta, "|", ex.messages[-1]["content"][:20])
    print("missing answer dropped:", build({"query": "x"}, 0, ctx))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.codefeedback
```
Expected: `codefeedback-9 {'lang': 'python', 'resource': 'codefeedback'} | def rev(s): return s` then `None`.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/adapters/plainmath.py tools/dataset/adapters/codefeedback.py
git commit -m "feat(dataset): NuminaMath, gsm8k, and CodeFeedback adapters"
```

---

### Task 8: Tool-calling adapter (hermes-function-calling-v1)

**Files:**
- Create: `tools/dataset/adapters/toolcalls.py`

The `tools` column is the reason the canonical schema has a `tools` field at all
(spec §5). It must survive, on the prompt record, to the JSONL unchanged, so the teacher
generates tool calls against the real schema.

- [x] **Step 1: Write `tools/dataset/adapters/toolcalls.py`**

```python
"""NousResearch/hermes-function-calling-v1. Carries a `tools` schema per example."""
from __future__ import annotations

import json

from ..canonical import Example
from .base import Ctx, sharegpt_turns


def _valid_tool_schema(tools: list) -> bool:
    """A tool list must hold OpenAI-style function objects; anything else breaks rendering."""
    for entry in tools:
        if not isinstance(entry, dict):
            return False
        fn = entry.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            return False
    return True


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    messages = sharegpt_turns(row.get("conversations") or [],
                              system=row.get("system"))
    if messages is None:
        return None
    # The column ships JSON-encoded tool schemas as a string.
    tools = row.get("tools")
    if isinstance(tools, str):
        tools = tools.strip() or None
        if tools is not None:
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                return None
    if tools is not None:
        if not isinstance(tools, list) or not _valid_tool_schema(tools):
            return None
    return Example(
        id=f"hermesfc-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=messages,
        tools=tools or None,
        meta={"category": row.get("category"), "task": row.get("task")},
    )


if __name__ == "__main__":
    ctx = Ctx("NousResearch/hermes-function-calling-v1", "func_calling", "coding", "apache-2.0")
    row = {
        "conversations": [{"from": "human", "value": "What is the weather in Paris?"},
                          {"from": "gpt", "value": "<tool_call>{\"name\": \"get_weather\"}</tool_call>"}],
        "tools": json.dumps([{"type": "function", "function": {"name": "get_weather"}}]),
        "category": "weather", "task": "tool_call",
    }
    ex = build(row, 11, ctx)
    print(ex.id, "tools:", len(ex.tools), "meta:", ex.meta)
    print("bad tools dropped:", build({**row, "tools": "not-a-list"}, 12, ctx))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.toolcalls
```
Expected: `hermesfc-11 tools: 1 meta: {'category': 'weather', 'task': 'tool_call'}` then `None`.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/adapters/toolcalls.py
git commit -m "feat(dataset): hermes function-calling adapter with tools passthrough"
```

---

### Task 9: Cauldron adapter (generation configs, images)

**Files:**
- Create: `tools/dataset/adapters/cauldron.py`

All eight configs become prompts; prebuilt responses are discarded centrally, like every
other source's. The four VQA configs additionally form the §8 vision probe — `build_vqa`
keeps their short reference answers for the holdout (Task 17), which is the one place a
prebuilt answer survives, as an eval label rather than a training target.

- [x] **Step 1: Write `tools/dataset/adapters/cauldron.py`**

```python
"""HuggingFaceM4/the_cauldron configs. Images ride on prompts; the VQA builder keeps reference answers for the holdout only."""
from __future__ import annotations

import io
from typing import Any

from ..canonical import Example
from ..imgstore import image_part
from .base import Ctx, assistant_msg, user_msg

# Prompt configs: long-form description tasks whose prompts carry images.
GENERATION_CONFIGS = ("chart2text", "diagram_image_to_text", "tabmwp", "screen2words")

# Holdout configs: short reference answers are kept for the section 8 vision probe only.
VQA_CONFIGS = ("chartqa", "ai2d", "tqa", "scienceqa")


def _as_bytes(image: Any) -> bytes | None:
    if isinstance(image, dict):
        if image.get("bytes"):
            return image["bytes"]
        path = image.get("path")
        if path and isinstance(path, (bytes, bytearray)):
            return bytes(path)
        return None
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "save"):  # a PIL image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    return None


def _text_pair(texts: list) -> tuple[str, str] | None:
    """Cauldron `texts` are single-element lists of {user, assistant, source} structs.

    Plain [prompt, response] lists are accepted too, so hand-built fixtures stay usable.
    Values must be strings: str()-coercing a dict or a number would write a Python repr
    into the prompt, which no filter can tell apart from real content.
    """
    if not texts:
        return None
    first = texts[0]
    if isinstance(first, dict):
        user, assistant = first.get("user"), first.get("assistant")
    else:
        user = texts[0]
        assistant = texts[1] if len(texts) > 1 else None
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    prompt, response = user.strip(), assistant.strip()
    if not prompt or not response:
        return None
    return prompt, response


def build(row: dict, index: int, ctx: Ctx) -> Example | None:
    if ctx.config not in GENERATION_CONFIGS:
        return None
    pair = _text_pair(row.get("texts") or [])
    if pair is None:
        return None
    prompt, response = pair

    parts: list[dict] = [{"type": "text", "text": prompt}]
    for image in row.get("images") or []:
        data = _as_bytes(image)
        if data is None:
            return None
        parts.append(image_part(data))

    return Example(
        id=f"cauldron-{ctx.config}-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=[user_msg(parts), assistant_msg(response)],
    )


def build_vqa(row: dict, index: int, ctx: Ctx) -> Example | None:
    """Holdout builder for the section 8 vision probe. Used by Task 17, never by the pipeline."""
    if ctx.config not in VQA_CONFIGS:
        return None
    pair = _text_pair(row.get("texts") or [])
    if pair is None:
        return None
    question, answer = pair

    parts: list[dict] = [{"type": "text", "text": question}]
    for image in row.get("images") or []:
        data = _as_bytes(image)
        if data is None:
            return None
        parts.append(image_part(data))

    return Example(
        id=f"visionholdout-{ctx.config}-{index}",
        domain="reasoning",
        origin="prebuilt",
        source=ctx.source(index),
        messages=[user_msg(parts), assistant_msg(answer)],
    )


if __name__ == "__main__":
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (200, 30, 30)).save(buf, format="PNG")
    payload = buf.getvalue()

    ctx = Ctx("HuggingFaceM4/the_cauldron", "chart2text", "reasoning", "apache-2.0")
    row = {"images": [{"bytes": payload}],
           "texts": [{"user": "Describe the chart.", "assistant": "It rises steadily.",
                      "source": "chart2text"}]}
    ex = build(row, 4, ctx)
    print(ex.id, "parts:", [p["type"] for p in ex.messages[0]["content"]])
    print("vqa config skipped:", build(row, 5, Ctx("HuggingFaceM4/the_cauldron", "chartqa", "reasoning")))
    print("single-text skipped:", build({"images": [{"bytes": payload}], "texts": ["only"]}, 6, ctx))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.cauldron
```
Expected: `cauldron-chart2text-4 parts: ['text', 'image']` then `None` twice.

- [x] **Step 2: Commit**

```powershell
git add tools/dataset/adapters/cauldron.py
git commit -m "feat(dataset): Cauldron generation-config adapter"
```

---

### Task 10: Uncensored prompt-seed ingest

**Files:**
- Create: `tools/dataset/seeds.py`

M1 has no teacher, so the uncensored column arrives as prompts. This task fetches the
seed corpus (spec §7.4), hash-selects the 500-row refusal slice that §8 quarantines from
the same corpus (1,500 train / 500 held out), and exposes the trainable rows as canonical
`Prompt` records for the pipeline to merge as the uncensored fourth domain. The real
harvest runs here so source availability is validated in M1, not deferred.

- [x] **Step 1: Write `tools/dataset/seeds.py`**

```python
"""Fetch the uncensored prompt seeds and quarantine the refusal eval slice.

Prompts only; M2 supplies every response. The in-the-wild harvest also carves the
500-row held-out slice from spec section 8, so it can never train.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from . import textutil
from .canonical import Prompt, iter_jsonl

# (dataset, config, split, field, cap) - spec section 7.4.
# Caps respect what each config actually ships: in-the-wild 2023_12_25 has 1,405 rows,
# 2023_05_07 has 666, JBB-Behaviors ships 100 harmful + 100 benign.
SEED_SOURCES = [
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train", "prompt", 1400),
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_05_07", "train", "prompt", 600),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", "Goal", 100),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "benign", "Goal", 100),
    ("mlabonne/harmful_behaviors", None, "train", "text", 200),
    ("mlabonne/harmless_alpaca", None, "train", "text", 600),
]

IN_THE_WILD = "TrustAIRLab/in-the-wild-jailbreak-prompts"
REFUSAL_HOLDOUT = 500          # spec section 8: 1,500 train / 500 quarantined


def harvest(*, dry_run: bool = False) -> list[dict]:
    seeds: list[dict] = []
    for name, config, split, field, cap in SEED_SOURCES:
        if dry_run:
            print(f"[dry] {name}:{config}/{split} cap={cap} field={field}")
            continue
        stream = load_dataset(name, config, split=split, streaming=True)
        taken = 0
        for row in stream:
            text = (row.get(field) or "").strip()
            if not text:
                continue
            seeds.append({"id": f"{name.split('/')[-1]}-{config or 'default'}-{split}-{taken}",
                          "source": name, "config": config, "text": text})
            taken += 1
            if taken >= cap:
                break
        print(f"{name}:{config}/{split} -> {taken}")
    return seeds


def split_refusal_holdout(seeds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hash-select 500 in-the-wild prompts for the section 8 refusal eval."""
    inwild = [s for s in seeds if s["source"] == IN_THE_WILD]
    if len(inwild) <= REFUSAL_HOLDOUT:
        return seeds, []
    order = sorted(inwild, key=lambda s: textutil.prompt_hash(s["text"]))
    heldout = {id(s) for s in order[:REFUSAL_HOLDOUT]}
    return ([s for s in seeds if id(s) not in heldout],
            [s for s in seeds if id(s) in heldout])


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def load_prompts(root: Path) -> list[Prompt]:
    """The trainable seeds as canonical prompts; the pipeline merges them as uncensored."""
    path = Path(root) / "seeds" / "uncensored.jsonl"
    if not path.exists():
        print(f"WARNING: {path} missing; run tools.dataset.seeds before pipeline.py")
        return []
    return [
        Prompt(id=row["id"], domain="uncensored", origin="prebuilt",
               source={"name": row["source"], "config": row["config"]},
               messages=[{"role": "user",
                          "content": [{"type": "text", "text": row["text"]}]}])
        for row in iter_jsonl(path)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="uncensored prompt seeds")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    print("seed sources:", len(SEED_SOURCES),
          "total cap:", sum(s[4] for s in SEED_SOURCES))
    seeds = harvest(dry_run=args.dry_run)
    if args.dry_run:
        return 0

    trainable, holdout = split_refusal_holdout(seeds)
    n_train = write_jsonl(args.root / "seeds" / "uncensored.jsonl", trainable)
    n_eval = write_jsonl(args.root / "eval" / "refusal.jsonl", holdout)
    print(f"seeds trainable: {n_train}, refusal holdout: {n_eval}")
    return 0 if n_train and n_eval else 1


if __name__ == "__main__":
    sys.exit(main())
```
- [x] **Step 2: Dry run, then the real harvest**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.seeds --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.seeds --root datasets/qwen35-4b-sft
```
Expected: `seed sources: 6 total cap: 3000` then six `[dry]` lines; then one
`<dataset>:<config>/<split> -> N` line per source (1400, 600, 100, 100, 200, 600) and
`seeds trainable: 2500, refusal holdout: 500`. The caps sum to 3,000, matching the
prebuilt half of spec §7.4; the 500 held-out rows are the §8 refusal eval slice.
`pipeline.py` later loads `seeds/uncensored.jsonl` as the uncensored prompt column.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/seeds.py
git commit -m "feat(dataset): uncensored prompt-seed harvest"
```

---

### Task 11: Frozen source table, quota allocation, domain assignment

**Files:**
- Create: `tools/dataset/sources.py`
- Create: `tools/dataset/domains.py`

- [x] **Step 1: Write `tools/dataset/sources.py`**

```python
"""The frozen source table (spec section 7) and the M1 prompt quota allocator."""
from __future__ import annotations

from dataclasses import dataclass

# Domain shares of the final mix (spec section 4).
DOMAIN_SHARE = {"reasoning": 0.30, "coding": 0.30, "roleplay": 0.20, "uncensored": 0.20}
TOTAL_CANDIDATES = 39_000


@dataclass(frozen=True)
class Source:
    dataset: str
    config: str | None
    split: str
    domain: str
    cap: int
    adapter: str
    teacher: bool = False          # True -> M2 territory, excluded from M1


# M1 prompt sources: every row supplies a user prompt only. Prebuilt answers are
# discarded centrally (canonical.strip_to_prompt); spec sections 7.1-7.3, 7.6.
SOURCES: list[Source] = [
    Source("AI-MO/NuminaMath-1.5", None, "train", "reasoning", 8000, "numina"),
    Source("openai/gsm8k", "main", "train", "reasoning", 1000, "gsm8k"),
    Source("HuggingFaceTB/smoltalk", "metamathqa-50k", "train", "reasoning", 500, "smoltalk"),
    Source("HuggingFaceM4/the_cauldron", "chart2text", "train", "reasoning", 1500, "cauldron"),
    # diagram_image_to_text ships only 300 rows; the cap is a maximum, not a promise.
    Source("HuggingFaceM4/the_cauldron", "diagram_image_to_text", "train", "reasoning", 1000, "cauldron"),
    Source("HuggingFaceM4/the_cauldron", "tabmwp", "train", "reasoning", 500, "cauldron"),
    Source("HuggingFaceM4/the_cauldron", "screen2words", "train", "coding", 1000, "cauldron"),
    Source("m-a-p/CodeFeedback-Filtered-Instruction", None, "train", "coding", 5000, "codefeedback"),
    Source("NousResearch/hermes-function-calling-v1", "func_calling", "train", "coding", 1500, "toolcalls"),
    Source("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "coding", 1500, "toolcalls"),
    Source("Team-ACE/ToolACE", None, "train", "coding", 3000, "sharegpt_toolace"),
    Source("teknium/OpenHermes-2.5", None, "train", "roleplay", 2500, "sharegpt_openhermes"),
    Source("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "systemchats-30k", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", "roleplay", 1500, "smoltalk"),
    Source("HuggingFaceTB/smoltalk", "longalign", "train", "roleplay", 500, "smoltalk"),
]

# M2 teacher rows, counted so the pool arithmetic stays honest and visible.
TEACHER_SOURCES: list[Source] = [
    Source("teacher:tools", None, "-", "coding", 1000, "teacher", teacher=True),
    Source("teacher:uncensored", None, "-", "uncensored", 6000, "teacher", teacher=True),
]


def per_domain_caps(sources: list[Source]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for s in sources:
        caps[s.domain] = caps.get(s.domain, 0) + s.cap
    return caps


def allocate(target_train: int, targets: list[Source]) -> dict[tuple[str, str | None], int]:
    """Split `target_train` across sources in proportion to their caps within each domain."""
    domain_budget = {d: round(target_train * share) for d, share in DOMAIN_SHARE.items()}
    by_domain: dict[str, list[Source]] = {}
    for s in targets:
        by_domain.setdefault(s.domain, []).append(s)

    quotas: dict[tuple[str, str | None], int] = {}
    for domain, sources in by_domain.items():
        budget = domain_budget.get(domain, 0)
        total_cap = sum(s.cap for s in sources)
        for s in sources:
            quotas[(s.dataset, s.config)] = int(budget * s.cap / total_cap) if total_cap else 0
    return quotas


if __name__ == "__main__":
    prebuilt_caps = per_domain_caps(SOURCES)
    teacher_caps = per_domain_caps(TEACHER_SOURCES)
    m1_total = sum(prebuilt_caps.values())
    pool = sum(prebuilt_caps.get(d, 0) + teacher_caps.get(d, 0) for d in DOMAIN_SHARE)
    print("M1 prebuilt caps:", prebuilt_caps, "=", m1_total)
    print("teacher caps:", teacher_caps)
    print("full pool:", pool, "(spec target 39000)")
    q = allocate(3000, SOURCES)
    print("M1 quota total:", sum(q.values()),
          "(uncensored prompts come from seeds.py; downsample renormalises)")
    print("first three:", list(q.items())[:3])
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.sources
```
Expected: M1 prompt caps summing to 32,000; the full pool printing **39000**; and a
`SOURCES` quota total of **2,399** — 80% of 3,000, because the uncensored fifth has no
entry in this table (its prompts come from Task 10's seeds). With seeds present,
`downsample` trims the accepted pool to the full four-domain 30 / 30 / 20 / 20 mix. If
the pool is not 39,000, a cap above contradicts the spec — fix the cap, not the printout.

- [x] **Step 2: Write `tools/dataset/domains.py`**

```python
"""Domain assignment. Sources are single-domain; the table classifies mixed sources."""
from __future__ import annotations

from .canonical import DOMAINS

KEYWORDS = {
    "reasoning": ("solve", "prove", "theorem", "derive", "calculate", "equation", "integral"),
    "coding": ("function", "def ", "class ", "compile", "refactor", "traceback", "import "),
    "roleplay": ("pretend", "roleplay", "character", "story", "persona", "*smiles*"),
    "uncensored": ("jailbreak", "bypass", "unrestricted", "no restrictions", "as an ai without"),
}


def assign_domain(text: str, declared: str) -> str:
    """Trust a source's declared domain; keyword-classify only mixed/unmapped sources.

    Spec section 9.5: rule-based per source. Substring keyword hits must never silently
    move a single-domain source (e.g. a CodeFeedback prompt mentioning "solve"), so the
    fallback applies only when the source declares something outside DOMAINS.
    """
    if declared in DOMAINS:
        return declared
    low = text.lower()
    for domain, words in KEYWORDS.items():
        if any(w in low for w in words):
            return domain
    return declared


def prompt_text(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


if __name__ == "__main__":
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Solve x^2 = 4"}]}]
    print("declared kept:", assign_domain(prompt_text(msgs), "reasoning"))
    print("mixed classified:", assign_domain("write a def and refactor", "mixed"))
    print("mixed fallback:", assign_domain("pretend you are a prince", "mixed"))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.domains
```
Expected: `reasoning`, `coding`, `roleplay` — declared domains are never overridden, and
rows from mixed/unmapped sources use the keyword table.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/sources.py tools/dataset/domains.py
git commit -m "feat(dataset): frozen source table, quota allocator, domain rules"
```

---

### Task 12: Dedup and filters

**Files:**
- Create: `tools/dataset/dedup.py`
- Create: `tools/dataset/filters.py`

- [x] **Step 1: Write `tools/dataset/dedup.py`**

```python
"""Exact and near-duplicate rejection, plus image-sha collapse."""
from __future__ import annotations

from typing import Iterable

from .canonical import Example, Prompt, image_shas
from . import textutil


class Deduper:
    """Check-then-commit so rows rejected by later stages never enter the index."""

    def __init__(self, threshold: float = 0.85, bands: int = 16):
        self.threshold = threshold
        self.bands = bands
        self._hashes: set[str] = set()
        self._images: set[str] = set()
        # band -> [(signature, image identity)]. The image identity travels with the
        # signature so a near-duplicate hit only counts when the images also agree.
        self._band_index: dict[tuple[int, ...], list[tuple[tuple[int, ...], str]]] = {}

    @staticmethod
    def _image_key(images: Iterable[str] = ()) -> str:
        """Identity of a row's images; empty string for text-only rows."""
        return ",".join(sorted(images))

    @classmethod
    def _key(cls, prompt: str, images: Iterable[str] = ()) -> str:
        """Identity of a row: its normalised text plus the images it carries.

        Image configs such as Cauldron's chart2text and screen2words reuse one
        templated instruction ("Summarize the main components in this picture.")
        across hundreds of rows that each carry a *different* image. Those are not
        duplicates: they are the same question about different visual inputs, and
        collapsing them on text alone discards almost the whole image column. So the
        images are part of the identity, and two rows collide only when both match.
        """
        image_key = cls._image_key(images)
        base = textutil.prompt_hash(prompt)
        return base + "|" + image_key if image_key else base

    def check_prompt(self, prompt: str, images: Iterable[str] = ()) -> str | None:
        """Return a rejection reason without mutating the index."""
        image_key = self._image_key(images)
        if self._key(prompt, images) in self._hashes:
            return "duplicate_prompt"
        sig = textutil.minhash(prompt)
        for band in textutil.banded(sig, self.bands):
            for other, other_images in self._band_index.get(band, ()):
                # Text-only rows compare as before (both keys empty). A row carrying
                # images only matches another row carrying the same images, so one
                # instruction over many distinct images is not collapsed.
                if other_images != image_key:
                    continue
                if textutil.jaccard_est(sig, other) >= self.threshold:
                    return "near_duplicate"
        return None

    def commit_prompt(self, prompt: str, images: Iterable[str] = ()) -> None:
        """Record a prompt the pipeline accepted (call only after filters and validate)."""
        image_key = self._image_key(images)
        sig = textutil.minhash(prompt)
        self._hashes.add(self._key(prompt, images))
        for band in textutil.banded(sig, self.bands):
            self._band_index.setdefault(band, []).append((sig, image_key))

    def reject_reason(self, ex: Example, prompt: str) -> str | None:
        """`check_prompt` + `commit_prompt`, for standalone use and smoke checks."""
        images = image_shas(ex)
        reason = self.check_prompt(prompt, images)
        if reason is None:
            self.commit_prompt(prompt, images)
        return reason

    def has_new_images(self, ex: Example | Prompt) -> bool:
        """True when the example has no images, or carries at least one unseen image."""
        shas = image_shas(ex)
        if not shas:
            return True
        return any(s not in self._images for s in shas)

    def commit_images(self, ex: Example | Prompt) -> None:
        self._images.update(image_shas(ex))


if __name__ == "__main__":
    from .canonical import Example

    def mk(i, text):
        return Example(id=f"e{i}", domain="reasoning", origin="prebuilt",
                       source={"name": "smoke"},
                       messages=[{"role": "user", "content": [{"type": "text", "text": text}]},
                                 {"role": "assistant", "content": "ok"}])

    d = Deduper()
    print(d.reject_reason(mk(0, "What is the capital of France?"), "What is the capital of France?"))
    print(d.reject_reason(mk(1, "What is the capital of France?  "), "What is the capital of France?  "))
    print(d.reject_reason(mk(2, "Explain photosynthesis in plants."), "Explain photosynthesis in plants."))

    from .canonical import ImageRef

    def with_image(sha):
        ex = mk(3, "Describe this chart.")
        ex.images = [ImageRef(sha, f"images/{sha[:2]}/{sha}.png", 8, 8)]
        ex.messages[0]["content"].append({"type": "image", "sha256": sha})
        return ex

    print("text-only kept:", d.has_new_images(mk(4, "plain text")))
    print("first image kept:", d.has_new_images(with_image("aa11")))
    d.commit_images(with_image("aa11"))
    print("repeated image dropped:", d.has_new_images(with_image("aa11")))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.dedup
```
Expected: `None`, `duplicate_prompt`, `None`, then `True`, `True`, `False`.

- [x] **Step 2: Write `tools/dataset/filters.py`**

```python
"""Reject predicates.

Prompt-side predicates run in M1; response-side predicates run in M2 on teacher
completions, before they can become student data. Each returns a reason string when the
record is dropped.
"""
from __future__ import annotations

import re

from .canonical import Example
from . import textutil

MIN_PROMPT_TOKENS = 4
MAX_PROMPT_TOKENS = 8192
MIN_ANSWER_TOKENS = 16
MAX_ANSWER_TOKENS = 16384
LOOP_RATIO_MAX = 0.05

# The student template stores the CoT as angle-bracket delimited tags, read from the
# GGUF's own chat template in Task 15 rather than assumed: the template writes
# '<think>' / '</think>', each on its own line.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Spec section 9.4's language filter: reject rows dominated by non-Latin scripts.
# Symbols common in maths (Greek, arrows) are deliberately not in the class.
_NON_ENGLISH = re.compile(
    r"[\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
NON_ENGLISH_RATIO_MAX = 0.02

# The spec (section 9.4) calls this an inverted refusal filter: relative to a guardrail
# pipeline, which keeps refusals, this one drops them.
REFUSAL_MARKERS = (
    "i cannot", "i can't", "i can not", "i'm unable", "i am unable", "i won't",
    "i will not", "i must decline", "as an ai language model", "i'm sorry, but",
)


def assistant_text(ex: Example) -> str:
    for m in reversed(ex.messages):
        if m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def prompt_reject_reason(text: str, tokens: int) -> str | None:
    """Prompt-side predicates, used by the M1 pipeline."""
    if not text.strip():
        return "empty_prompt"
    if tokens < MIN_PROMPT_TOKENS:
        return "prompt_too_short"
    if tokens > MAX_PROMPT_TOKENS:
        return "prompt_too_long"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
    return None


def example_reject_reason(ex: Example) -> str | None:
    """Response-side predicates. M2 runs these on teacher completions before writing."""
    text = assistant_text(ex)
    if not text.strip():
        return "empty_assistant"
    if text.count(THINK_OPEN) != text.count(THINK_CLOSE):
        return "unbalanced_think_tags"
    if ex.tokens < MIN_ANSWER_TOKENS:
        return "too_short"
    if ex.tokens > MAX_ANSWER_TOKENS:
        return "too_long"
    if textutil.repeating_ngram_ratio(text) > LOOP_RATIO_MAX:
        return "looping"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
    low = text.lower()
    if any(marker in low for marker in REFUSAL_MARKERS):
        return "refusal"
    return None


if __name__ == "__main__":
    from .canonical import Example

    def mk(text, tokens=200):
        return Example(id="f", domain="reasoning", origin="prebuilt", source={"name": "s"},
                       messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]},
                                 {"role": "assistant", "content": text}], tokens=tokens)

    print("prompt clean:", prompt_reject_reason("Solve x^2 = 4.", 200))
    print("prompt empty:", prompt_reject_reason("", 0))
    print("clean:", example_reject_reason(mk("A clear worked answer.")))
    print("refusal:", example_reject_reason(mk("I cannot help with that request.")))
    print("unbalanced:", example_reject_reason(mk("<think>reasoning without a close tag")))
    print("short:", example_reject_reason(mk("ok", tokens=3)))
    print("loop:", example_reject_reason(mk(" ".join(["a b c d e f g h"] * 30))))
    print("non-english:", example_reject_reason(mk("这是一段中文回答，用于测试语言过滤。")))
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.filters
```
Expected: `prompt clean: None`, `prompt empty: empty_prompt`, then `clean: None`,
`refusal`, `unbalanced`, `short`, `loop`, and `non_english` response reasons.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/dedup.py tools/dataset/filters.py
git commit -m "feat(dataset): dedup index and reject filters"
```

---

### Task 13: Split and manifest

**Files:**
- Create: `tools/dataset/split.py`
- Create: `tools/dataset/report.py`

- [x] **Step 1: Write `tools/dataset/split.py`**

```python
"""Source-stratified, image-disjoint train/val split (spec section 9.6)."""
from __future__ import annotations

import random
from collections import defaultdict

from .canonical import Prompt, image_shas


def stratified_split(records: list[Prompt], val_size: int, seed: int = 17
                     ) -> tuple[list[Prompt], list[Prompt]]:
    rnd = random.Random(seed)
    by_domain: dict[str, list[Prompt]] = defaultdict(list)
    for ex in records:
        by_domain[ex.domain].append(ex)

    total = len(records)
    val: list[Prompt] = []
    train: list[Prompt] = []
    val_images: set[str] = set()

    for domain in sorted(by_domain):
        items = by_domain[domain][:]
        rnd.shuffle(items)
        want = round(val_size * len(items) / total) if total else 0
        taken = 0
        for ex in items:
            shas = set(image_shas(ex))
            if taken < want and not (shas & val_images):
                val.append(ex)
                val_images |= shas
                taken += 1
            else:
                train.append(ex)

    train = [ex for ex in train if not (set(image_shas(ex)) & val_images)]
    return train, val


def downsample(records: list[Prompt], target: int, seed: int = 17) -> list[Prompt]:
    """Trim to `target`, weighted by spec section 4's shares over the domains present.

    Uncensored prompts arrive from seeds.py, so the accepted pool covers four domains and
    targets the full 30 / 30 / 20 / 20 mix.
    """
    from .sources import DOMAIN_SHARE

    rnd = random.Random(seed)
    by_domain: dict[str, list[Prompt]] = defaultdict(list)
    for ex in records:
        by_domain[ex.domain].append(ex)
    weights = {d: DOMAIN_SHARE.get(d, 0.0) for d in by_domain}
    total_weight = sum(weights.values()) or 1.0

    out: list[Prompt] = []
    for domain, items in by_domain.items():
        items = items[:]
        rnd.shuffle(items)
        out.extend(items[:round(target * weights[domain] / total_weight)])

    # `round` is half-to-even, so the per-domain allocations can sum to one more than
    # target. Trim back rather than return `target + 1` from a function that says "trim".
    if len(out) > target:
        rnd.shuffle(out)
        out = out[:target]

    if len(out) < target:
        chosen = {id(e) for e in out}
        leftovers = [e for e in records if id(e) not in chosen]
        rnd.shuffle(leftovers)
        out.extend(leftovers[:target - len(out)])
    return out


if __name__ == "__main__":
    from .canonical import ImageRef, Prompt

    def mk(i, domain, sha=None):
        imgs = [ImageRef(sha, f"images/{sha[:2]}/{sha}.png", 8, 8)] if sha else []
        content = [{"type": "text", "text": f"q{i}"}]
        if sha:
            content.append({"type": "image", "sha256": sha})
        return Prompt(id=f"s{i}", domain=domain, origin="prebuilt", source={"name": "s"},
                      messages=[{"role": "user", "content": content}], images=imgs)

    rows = [mk(i, ["reasoning", "coding", "roleplay"][i % 3]) for i in range(60)]
    rows.append(mk(999, "reasoning", "deadbeef"))
    train, val = stratified_split(rows, val_size=6)
    val_shas = {s for ex in val for s in image_shas(ex)}
    train_shas = {s for ex in train for s in image_shas(ex)}
    print("train:", len(train), "val:", len(val))
    print("image overlap:", val_shas & train_shas)
    print("val domains:", sorted({ex.domain for ex in val}))

    trimmed = downsample(rows[:60], target=30)
    counts: dict[str, int] = {}
    for ex in trimmed:
        counts[ex.domain] = counts.get(ex.domain, 0) + 1
    print("downsample 30 ->", counts)
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.split
```
Expected: `train: 55 val: 6`, `image overlap: set()`, the val domain list, and
`downsample 30 -> {'reasoning': 11, 'coding': 11, 'roleplay': 8}` — the 60 raw rows are
20 per domain; with three domains present the 3:3:2 weighting renormalises exactly as
before, while the full pipeline adds uncensored and targets 30 / 30 / 20 / 20.

- [x] **Step 2: Write `tools/dataset/report.py`**

```python
"""Manifest: counts, caps, filter drops, token share, image totals (spec section 9.7)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .canonical import Prompt


def summarise(records: list[Prompt], label: str) -> dict:
    by_domain = Counter(ex.domain for ex in records)
    tokens_by_domain: Counter = Counter()
    images = 0
    image_examples = 0
    for ex in records:
        tokens_by_domain[ex.domain] += ex.tokens
        if ex.images:
            image_examples += 1
            images += len(ex.images)
    total_tokens = sum(tokens_by_domain.values())
    return {
        "split": label,
        "examples": len(records),
        "by_domain": dict(sorted(by_domain.items())),
        "example_share": {d: round(n / len(records), 4) for d, n in sorted(by_domain.items())}
        if records else {},
        "tokens": total_tokens,
        "token_share": {d: round(n / total_tokens, 4) for d, n in sorted(tokens_by_domain.items())}
        if total_tokens else {},
        "image_bearing_examples": image_examples,
        "image_bearing_share": round(image_examples / len(records), 4) if records else 0.0,
        "images": images,
    }


def build_manifest(*, run: dict, train: list[Prompt], val: list[Prompt],
                   sources: list[dict], drops: Counter) -> dict:
    return {
        "run": run,
        "sources": sources,
        "drops": dict(drops.most_common()),
        "train": summarise(train, "train"),
        "val": summarise(val, "val"),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    from .canonical import Prompt

    rows = [Prompt(id=f"r{i}", domain="reasoning", origin="prebuilt", source={"name": "s"},
                   messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                   tokens=100)
            for i in range(3)]
    rows += [Prompt(id="c", domain="coding", origin="prebuilt", source={"name": "s"},
                    messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                    tokens=300)]
    s = summarise(rows, "train")
    print("examples:", s["examples"], "token_share:", s["token_share"])
    print("image_bearing_share:", s["image_bearing_share"])
```
Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.report
```
Expected: `examples: 4 token_share: {'coding': 0.5, 'reasoning': 0.5}` and
`image_bearing_share: 0.0`.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/split.py tools/dataset/report.py
git commit -m "feat(dataset): stratified image-disjoint split and manifest"
```

---

### Task 14: Pipeline CLI and M1 dry run

**Files:**
- Create: `tools/dataset/pipeline.py`

- [x] **Step 1: Write `tools/dataset/pipeline.py`**

```python
"""M1 pipeline: fetch -> normalise -> strip to prompt -> dedup -> filter -> split -> report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from . import canonical, domains, filters, sources, textutil
from .adapters import get as get_adapter
from .adapters.base import Ctx
from .dedup import Deduper
from .imgstore import ImageStore, materialise_images
from .report import build_manifest, write_manifest
from .seeds import load_prompts as load_seed_prompts
from .split import downsample, stratified_split


def fetch_meta(dataset: str) -> tuple[str | None, str | None]:
    """License and revision from the HF API, for the manifest (best effort)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(dataset)
        return (info.cardData or {}).get("license"), info.sha
    except Exception:
        return None, None


def stream_source(src: sources.Source, quota: int, store: ImageStore, ctx: Ctx):
    adapter = get_adapter(src.adapter)
    stream = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
    produced = 0
    seen = 0
    for row in stream:
        if produced >= quota:
            return
        seen += 1
        ex = adapter.build(row, seen, ctx)
        if ex is None:
            continue
        materialise_images(ex, store)
        yield ex, adapter
        produced += 1


def run(*, root: Path, target_train: int, val_size: int, dry_run: bool) -> dict:
    store = ImageStore(root / "images")
    quotas = sources.allocate(target_train + val_size, sources.SOURCES)
    deduper = Deduper()
    drops: Counter = Counter()
    accepted: list[canonical.Prompt] = []
    source_stats: list[dict] = []

    for src in sources.SOURCES:
        quota = quotas.get((src.dataset, src.config), 0) * 2  # candidate overshoot
        kept = 0
        if dry_run:
            print(f"[dry] {src.dataset}:{src.config} domain={src.domain} quota={quota}")
            continue
        license_, revision = fetch_meta(src.dataset)
        ctx = Ctx(src.dataset, src.config, src.domain, license_)
        for ex, _adapter in stream_source(src, quota, store, ctx):
            prompt_rec = canonical.strip_to_prompt(ex)
            if prompt_rec is None:
                drops["no_user_turn"] += 1
                continue
            prompt = domains.prompt_text(prompt_rec.messages)
            images = canonical.image_shas(prompt_rec)
            reason = deduper.check_prompt(prompt, images) if prompt else "empty_prompt"
            if reason:
                drops[reason] += 1
                continue
            if not deduper.has_new_images(prompt_rec):
                drops["duplicate_images"] += 1
                continue
            prompt_rec.domain = domains.assign_domain(prompt, prompt_rec.domain)
            prompt_rec.tokens = textutil.count_tokens(prompt)
            reason = filters.prompt_reject_reason(prompt, prompt_rec.tokens)
            if reason:
                drops[reason] += 1
                continue
            problems = canonical.validate_prompt(prompt_rec)
            if problems:
                drops["invalid"] += 1
                if drops["invalid"] <= 3:
                    print(f"  invalid {prompt_rec.id}: {problems[:2]}")
                continue
            # Accepted: only now does it enter the dedup index, so rejected rows
            # cannot suppress a later viable row.
            deduper.commit_prompt(prompt, images)
            deduper.commit_images(prompt_rec)
            accepted.append(prompt_rec)
            kept += 1
        source_stats.append({"dataset": src.dataset, "config": src.config,
                             "domain": src.domain, "cap": src.cap, "kept": kept,
                             "license": license_, "revision": revision})
        print(f"{src.dataset}:{src.config} kept {kept}")

    if dry_run:
        print("[dry] seeds:uncensored (local file from Task 10)")
        return {}

    # Uncensored prompts come from the seed harvest (Task 10), not from SOURCES.
    seed_records = load_seed_prompts(root)
    seeds_kept = 0
    for prompt_rec in seed_records:
        prompt = domains.prompt_text(prompt_rec.messages)
        images = canonical.image_shas(prompt_rec)   # seeds are text-only; empty here
        reason = deduper.check_prompt(prompt, images) if prompt else "empty_prompt"
        if reason:
            drops[reason] += 1
            continue
        prompt_rec.tokens = textutil.count_tokens(prompt)
        reason = filters.prompt_reject_reason(prompt, prompt_rec.tokens)
        if reason:
            drops[reason] += 1
            continue
        if canonical.validate_prompt(prompt_rec):
            drops["invalid"] += 1
            continue
        deduper.commit_prompt(prompt, images)
        accepted.append(prompt_rec)
        seeds_kept += 1
    if seed_records:
        source_stats.append({"dataset": "seeds:uncensored", "config": None,
                             "domain": "uncensored", "cap": len(seed_records),
                             "kept": seeds_kept, "license": None, "revision": None})
    print(f"seeds:uncensored kept {seeds_kept}")

    # Spec section 11.3: every image sha must resolve to a file on disk.
    missing = [(ex.id, ref.sha256) for ex in accepted for ref in ex.images
               if store.resolve(ref.sha256) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} image refs do not resolve to files; "
                           f"first: {missing[:3]}")

    trimmed = downsample(accepted, target_train + val_size)
    train, val = stratified_split(trimmed, val_size=val_size)

    write_count = canonical.write_jsonl(root / "prompts" / "train.jsonl", train)
    val_count = canonical.write_jsonl(root / "prompts" / "val.jsonl", val)
    if write_count < target_train or val_count < val_size:
        print(f"WARNING: short of target — train {write_count}/{target_train}, "
              f"val {val_count}/{val_size}, pool {len(accepted)}")
    manifest = build_manifest(
        run={"root": str(root), "artifact": "prompts", "target_train": target_train,
             "val_size": val_size, "train_written": write_count, "val_written": val_count,
             "token_counter": "exact" if textutil._TOKENIZER else "chars/4 estimate",
             "caveats": ["NuminaMath-1.5 may contain GSM8K-derived problems; "
                         "gsm8k-test is not fully clean (spec section 8)."]},
        train=train, val=val, sources=source_stats, drops=drops)
    write_manifest(root / "manifest.json", manifest)
    print(f"prompts train {write_count}, val {val_count}, manifest written")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 prompt pipeline")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--target-train", type=int, default=3000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    manifest = run(root=args.root, target_train=args.target_train,
                   val_size=args.val_size, dry_run=args.dry_run)
    if manifest:
        print(json.dumps(manifest["train"]["by_domain"], indent=2))
        if (manifest["run"]["train_written"] < args.target_train
                or manifest["run"]["val_written"] < args.val_size):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: Dry run — no network, checks allocation only**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --dry-run
```
Expected: one `[dry]` line per source in `sources.SOURCES` (16 lines), each with a
non-zero quota, plus one `[dry] seeds:uncensored` line, and no exception.

- [x] **Step 3: Small live run — 200 rows, exercises real streaming end to end**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.seeds --root datasets/qwen35-4b-sft-smoke
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --target-train 200 --val-size 20 --root datasets/qwen35-4b-sft-smoke
```
Expected: a `kept N` line per source plus `seeds:uncensored kept N`, then
`prompts train 200, val 20, manifest written`. The `SOURCES` candidate pool is
`0.8 × 220 × 2 = 352` rows and the seeds add the uncensored column, so the run needs
roughly two thirds of its text candidates to survive dedup/filters; a shortfall prints a
`WARNING` and exits 1 instead of shipping silently. (Skipping the seeds command leaves
uncensored empty and warns.) Inspect:
```powershell
Get-Content datasets/qwen35-4b-sft-smoke/manifest.json -TotalCount 40
```
Expected: `drops` shows only legitimate reasons, `train.by_domain` covers all four
domains including `uncensored` (from seeds), and the `sources` entries carry `license`
and `revision`.

- [x] **Step 4: Commit**

```powershell
git add tools/dataset/pipeline.py
git commit -m "feat(dataset): M1 pipeline CLI"
```

---

### Task 15: Render gate (spec §11.2)

**Files:**
- Create: `tools/dataset/tokenserver.py`
- Create: `tools/dataset/render.py`

This task settles the think-tag spelling the spec defers, using the student's real
template rather than an assumption. It must pass before the full run in Task 16.

- [x] **Step 1: Write `tools/dataset/tokenserver.py`**

```python
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
```
Run (with a server already up on that port, or expect a connection error — this smoke
check requires the server, see Task 16 Step 1):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.tokenserver http://127.0.0.1:8085
```
Expected: a non-zero template length and a small token count.

- [x] **Step 2: Write `tools/dataset/render.py`**

```python
"""Render gate: apply the student's real chat template to a sample (spec section 11.2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jinja2 import Environment
from jinja2.exceptions import TemplateError

from .canonical import iter_jsonl

# The student template's literal delimiters (verified against the GGUF in Task 16).
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def render(template: str, messages: list[dict], add_generation_prompt: bool = False) -> str:
    env = Environment(trim_blocks=False, lstrip_blocks=False)
    tpl = env.from_string(template)
    return tpl.render(messages=messages, add_generation_prompt=add_generation_prompt,
                      bos_token="", eos_token="")


def strip_image_data(messages: list[dict]) -> list[dict]:
    """Templates that cannot handle image parts get a text-only view for this check."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(m)
            continue
        parts = [p for p in content if p.get("type") == "text"]
        out.append({"role": m["role"], "content": parts or [{"type": "text", "text": ""}]})
    return out


def run(*, dataset: Path, template: Path, sample: int) -> int:
    template_text = Path(template).read_text(encoding="utf-8")
    rows = list(iter_jsonl(dataset))[:sample]
    failures = 0
    tags = {"open": 0, "close": 0}
    for row in rows:
        try:
            text = render(template_text, strip_image_data(row["messages"]))
        except TemplateError as exc:
            failures += 1
            print(f"FAIL {row['id']}: {exc}")
            continue
        tags["open"] += text.count(THINK_OPEN)
        tags["close"] += text.count(THINK_CLOSE)

    # M1 prompts carry no think tags at all, so the gate also proves the template
    # preserves a stored think block (spec section 11.2).
    probe = [{"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
             {"role": "assistant",
              "content": f"{THINK_OPEN}\nOne plus one is two.\n{THINK_CLOSE}\n\nTwo."}]
    try:
        probe_text = render(template_text, strip_image_data(probe))
        probe_ok = (THINK_OPEN in probe_text and THINK_CLOSE in probe_text
                    and "One plus one is two." in probe_text)
    except TemplateError as exc:
        probe_ok = False
        print(f"FAIL probe: {exc}")

    # The student must mimic the teacher's reasoning process, so the generation prompt
    # itself has to open a think block (spec sections 5 and 11.2).
    try:
        gen_text = render(template_text, strip_image_data(probe[:1]), add_generation_prompt=True)
        thinking_ok = THINK_OPEN in gen_text
    except TemplateError as exc:
        thinking_ok = False
        print(f"FAIL generation prompt: {exc}")

    print(f"rendered {len(rows)} examples, {failures} template failures")
    print(f"think tags in rendered output: open={tags['open']} close={tags['close']}")
    print(f"probe think block survived: {probe_ok}")
    print(f"generation prompt opens thinking: {thinking_ok}")
    if failures or not probe_ok or not thinking_ok:
        print("VERDICT: template cannot render this data as written")
        return 1
    balanced = tags["open"] == tags["close"]
    print("VERDICT:", "think tags balanced" if balanced else "think tags UNBALANCED")
    return 0 if balanced else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chat-template render gate")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args(argv)
    return run(dataset=args.dataset, template=args.template, sample=args.sample)


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/tokenserver.py tools/dataset/render.py
git commit -m "feat(dataset): llama-server client and render gate"
```

---

### Task 16: Contamination guard, calibration export, full M1 run

**Files:**
- Create: `tools/dataset/contaminate.py`
- Create: `tools/dataset/calibrate.py`

- [ ] **Step 1: Start the student model server for the render gate**

```powershell
Start-Process -FilePath "C:\Users\tnmh\projects\local-llm\llama-cpp\llama-server.exe" `
  -ArgumentList @("-m","C:\Users\tnmh\projects\local-llm\models\qwen35-mtp\Qwen3.5-4B-MTP-Heretic.i1-Q4_K_S.gguf",
                  "-ngl","99","-fa","on","-c","8192","-np","1","--port","8085") `
  -WindowStyle Hidden
```
Wait for health:
```powershell
Invoke-RestMethod http://127.0.0.1:8085/health
```
Expected: `status` is `ok`. Then extract the template:
```powershell
Invoke-RestMethod http://127.0.0.1:8085/props | Select-Object -ExpandProperty chat_template | Set-Content tools/dataset/student.jinja -Encoding utf8
Get-Item tools/dataset/student.jinja | Select-Object Length
```
Expected: a non-zero file length.

- [ ] **Step 2: Run the render gate**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft-smoke/prompts/train.jsonl --template tools/dataset/student.jinja
```
Expected: `0 template failures`, `probe think block survived: True`,
`generation prompt opens thinking: True`, and a `VERDICT:` line. **Record the verdict and
the observed tag spelling in the spec's §5 before proceeding** — the template extracted
above uses the literal pair `<think>` / `</think>`, and M2 relies on the generation prompt
opening the think block so the student can mimic the teacher's process. Task 16 Step 5 is
gated on both checks; if the probe fails, thinking does not open, or the verdict is
unbalanced, reconcile the tag constants (`THINK_OPEN` / `THINK_CLOSE` in `filters.py` and
`render.py`) with the template rather than continuing.

- [ ] **Step 3: Write `tools/dataset/contaminate.py`**

```python
"""Eval-set disjointness guard (spec section 11.1). Non-negotiable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from .canonical import iter_jsonl
from . import textutil

# (dataset, config, split, field) for the remote eval holdouts in spec section 8.
# The refusal slice is local: seeds.py writes eval/refusal.jsonl, joined via --extra-eval.
EVAL_SETS = [
    ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    ("bigcode/humanevalpack", "python", "test", "prompt"),
    ("openai/gsm8k", "main", "test", "question"),
    # The section 8 vision probe. Excluded from training by construction, so this is a
    # belt-and-braces check that no future cap pulls them back into the pool.
    ("HuggingFaceM4/the_cauldron", "chartqa", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "ai2d", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "tqa", "train", "texts"),
    ("HuggingFaceM4/the_cauldron", "scienceqa", "train", "texts"),
]


def _field_text(row: dict, field: str) -> str:
    """Cauldron's `texts` is a list of {user, assistant, source} structs; others are strings."""
    value = row.get(field)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
        if isinstance(value, dict):
            value = value.get("user") or value.get("assistant") or ""
    return str(value or "").strip()


def eval_prompts(limit_per_set: int = 2000, skip: tuple[str, ...] = ()) -> list[str]:
    out: list[str] = []
    for name, config, split, field in EVAL_SETS:
        if name in skip or f"{name}:{config}" in skip:
            print(f"skip {name}:{config}")
            continue
        stream = load_dataset(name, config, split=split, streaming=True)
        taken = 0
        for i, row in enumerate(stream):
            if i >= limit_per_set:
                break
            text = _field_text(row, field)
            if text:
                out.append(text)
                taken += 1
        print(f"{name}:{config} -> {taken} eval prompts")
    return out


def _local_prompts(path: Path) -> list[str]:
    """Prompt text from a local JSONL: seeds use `text`, canonical rows use messages."""
    out: list[str] = []
    for row in iter_jsonl(path):
        if isinstance(row.get("text"), str):
            out.append(row["text"])
            continue
        text = "".join(p.get("text", "") for m in row.get("messages", [])
                       if m.get("role") == "user"
                       for p in (m["content"] if isinstance(m["content"], list)
                                 else [{"type": "text", "text": m["content"]}]))
        if text:
            out.append(text)
    return out


def check(*, dataset: Path, threshold: float = 0.85,
          skip: tuple[str, ...] = (), extra_eval: tuple[Path, ...] = ()) -> int:
    prompts = eval_prompts(skip=skip)  # fetched once; both derived lists reuse it
    for path in extra_eval:
        prompts.extend(_local_prompts(path))
    eval_hashes = {textutil.prompt_hash(p) for p in prompts}
    eval_sigs = [textutil.minhash(p) for p in prompts]
    collisions = 0
    for row in iter_jsonl(dataset):
        prompt = "".join(
            p.get("text", "") for m in row["messages"] if m["role"] == "user"
            for p in (m["content"] if isinstance(m["content"], list)
                      else [{"type": "text", "text": m["content"]}]))
        if not prompt:
            continue
        if textutil.prompt_hash(prompt) in eval_hashes:
            collisions += 1
            print(f"EXACT COLLISION {row['id']}")
            continue
        sig = textutil.minhash(prompt)
        for other in eval_sigs:
            if textutil.jaccard_est(sig, other) >= threshold:
                collisions += 1
                print(f"NEAR COLLISION {row['id']}")
                break
    print(f"collisions: {collisions}")
    return 1 if collisions else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="eval contamination guard")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--skip", action="append", default=[],
                    help="dataset or dataset:config to leave out of the eval side")
    ap.add_argument("--extra-eval", action="append", type=Path, default=[],
                    help="local JSONL whose prompts join the eval side")
    args = ap.parse_args(argv)
    return check(dataset=args.dataset, threshold=args.threshold,
                 skip=tuple(args.skip), extra_eval=tuple(args.extra_eval))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `tools/dataset/calibrate.py`**

```python
"""Carve calibration chunks for Phase 3's llama-imatrix (spec section 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical import iter_jsonl


def assistant_text(row: dict) -> str:
    for m in reversed(row["messages"]):
        if m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


# Spec section 3: chunks of 2-4k tokens. At the planning ratio (~4 chars/token) that is
# roughly 8k-16k characters; exact token counts are not material for imatrix calibration.
MIN_CHARS = 8000
MAX_CHARS = 16000


def run(*, dataset: Path, out: Path, chunks: int,
        min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> int:
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in iter_jsonl(dataset):
            text = assistant_text(row)
            if len(text) < min_chars:
                continue
            fh.write(text[:max_chars].replace("\n", " ") + "\n")
            written += 1
            if written >= chunks:
                break
    print(f"calibration chunks written: {written} -> {out}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibration set exporter")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("calibration.txt"))
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args(argv)
    return run(dataset=args.dataset, out=args.out, chunks=args.chunks,
               min_chars=args.min_chars, max_chars=args.max_chars)


if __name__ == "__main__":
    sys.exit(main())
```

Calibration needs completed answers, so it runs after M2 on the teacher-written
`train.jsonl`; the module is built here so Phase 3 is not blocked later.

- [ ] **Step 5: Full M1 run**

Only after Step 2's render verdict is balanced. Wire the exact token counter in first by
adding to the top of `run()` in `pipeline.py`, immediately after `store = ImageStore(...)`:

```python
    if token_server_url:
        from .tokenserver import LlamaServer, ServerTokenizer
        textutil.set_tokenizer(ServerTokenizer(LlamaServer(token_server_url)))
```

and add the matching parameter and flag:

```python
def run(*, root: Path, target_train: int, val_size: int, dry_run: bool,
        token_server_url: str | None = None) -> dict:
```
```python
    ap.add_argument("--token-server", default="http://127.0.0.1:8085")
```
```python
    manifest = run(root=args.root, target_train=args.target_train,
                   val_size=args.val_size, dry_run=args.dry_run,
                   token_server_url=args.token_server)
```

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --target-train 3000 --val-size 200 --root datasets/qwen35-4b-sft
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.contaminate --dataset datasets/qwen35-4b-sft/prompts/train.jsonl --extra-eval datasets/qwen35-4b-sft/eval/refusal.jsonl
```
Expected: `prompts train 3000, val 200`; `collisions: 0`. (Calibration and the
teacher-written `train.jsonl` arrive with M2.)

- [ ] **Step 6: Accept against the spec**

Check the manifest against spec §4 and §12:
```powershell
$m = Get-Content datasets/qwen35-4b-sft/manifest.json | ConvertFrom-Json
$m.train.examples; $m.train.by_domain; $m.train.token_share; $m.train.image_bearing_share; $m.drops
```
Expected: 3,000 prompt train; all four domains present, shares near **30 / 30 / 20 / 20**
before the trim to `--target-train` (uncensored comes from Task 10's seeds);
image-bearing share at or near **0.12**; token counter reported as `exact`. This is the
prompt mix; M2's accepted teacher completions will shift the final one.

- [ ] **Step 7: Record the rebuild instructions (spec section 3)**

Write `datasets/qwen35-4b-sft/README.md` containing the commands above (seeds, testsets,
pipeline, contaminate, visionholdout) plus the post-M2 ones (verify usage, calibrate),
the manifest's caveats, and a note that `raw/` is empty by design: ingestion streams from
the Hub and materialises images straight into `images/`. `prompts/` plus
`verification/seeds.jsonl` are M2's inputs; the teacher-written `train.jsonl` /
`val.jsonl` are M2's outputs.

- [ ] **Step 8: Commit**

```powershell
git add tools/dataset/contaminate.py tools/dataset/calibrate.py tools/dataset/pipeline.py
git commit -m "feat(dataset): contamination guard, calibration export, full M1 run"
```

---

### Task 17: Vision holdout (spec §8)

**Files:**
- Create: `tools/dataset/visionholdout.py`

The image column exists to keep visual grounding from eroding under a text-heavy SFT
mix (§7.6). That guard is unverifiable without a probe, and the four Cauldron VQA
configs are excluded from training by construction — so they are the probe. Phase 1
emits the holdout; the before/after measurement is Phase 2A's §11.6.

Depends on Tasks 1–2 and 9.

- [ ] **Step 1: Write `tools/dataset/visionholdout.py`**

```python
"""Emit the section 8 vision holdout: held-out Cauldron VQA, never trained on."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from . import canonical
from .adapters.base import Ctx
from .adapters.cauldron import VQA_CONFIGS, build_vqa
from .imgstore import ImageStore, materialise_images

DATASET = "HuggingFaceM4/the_cauldron"
DEFAULT_PER_CONFIG = 125          # 4 configs x 125 = 500, per spec section 8


def harvest(root: Path, per_config: int, dry_run: bool = False) -> list[canonical.Example]:
    store = ImageStore(root / "images")
    out: list[canonical.Example] = []
    for config in VQA_CONFIGS:
        if dry_run:
            print(f"[dry] {config}: {per_config} rows")
            continue
        ctx = Ctx(DATASET, config, "reasoning", "apache-2.0")
        stream = load_dataset(DATASET, config, split="train", streaming=True)
        kept = 0
        for index, row in enumerate(stream):
            ex = build_vqa(row, index, ctx)
            if ex is None:
                continue
            materialise_images(ex, store)
            if canonical.validate(ex):
                continue
            out.append(ex)
            kept += 1
            if kept >= per_config:
                break
        print(f"{config} -> {kept}")
    return out


def run(*, root: Path, per_config: int, dry_run: bool) -> int:
    examples = harvest(root, per_config, dry_run)
    if dry_run:
        return 0
    path = root / "eval" / "vision.jsonl"
    written = canonical.write_jsonl(path, examples)
    print(f"vision holdout: {written} rows -> {path}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vision holdout exporter")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--per-config", type=int, default=DEFAULT_PER_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(root=args.root, per_config=args.per_config, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
```

The holdout shares `root/images` with training. The store is content-addressed, so a
chart appearing in both would be stored once — but the two sets are disjoint by
config, so in practice no sharing occurs. The manifest's image totals report the
*training* pool only, which is why Task 16 Step 6 reads them rather than a raw
`du -sh` of the store.

- [ ] **Step 2: Dry run, then the real export**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.visionholdout --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.visionholdout --root datasets/qwen35-4b-sft
```
Expected: four `[dry]` lines naming `chartqa`, `ai2d`, `tqa`, `scienceqa`; then four
`<config> -> 125` lines and `vision holdout: 500 rows`.

- [ ] **Step 3: Confirm it is disjoint from training**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.contaminate --dataset datasets/qwen35-4b-sft/eval/vision.jsonl --skip HuggingFaceM4/the_cauldron
```
Expected: `collisions: 0`. The VQA configs are excluded by construction from
`sources.SOURCES`, so this checks the holdout against the *other* §8 eval sets; `--skip`
keeps the guard from comparing the holdout with its own source configs (which would
always collide).

- [ ] **Step 4: Commit**

```powershell
git add tools/dataset/visionholdout.py
git commit -m "feat(dataset): section 8 vision holdout exporter"
```

---

### Task 18: Unit-test distillation seeds and verifier

**Files:**
- Create: `tools/dataset/testsets.py`
- Create: `tools/dataset/verify.py`

Distillation transfers the teacher's bugs as faithfully as its skills. The defence in M1
is a seed corpus that carries its own oracle: MBPP ships assert-style unit tests
(`test_list`), APPS ships stdin/stdout pairs (`input_output`). Both become canonical
`Prompt` records with a `verify` spec attached, so M2 can execute the teacher's completion
and drop failures *before* they can become student data. The checker is built and
smoke-tested here; M2 calls it per generation.

- [ ] **Step 1: Write `tools/dataset/testsets.py`**

```python
"""Unit-test verification seeds: prompts whose tests filter teacher attempts.

MBPP ships assert-style tests; APPS ships stdin/stdout pairs. Each seed becomes a
canonical Prompt with a `verify` spec attached, so M2 can execute the teacher's
completion and drop failures before they can reach the student.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from .canonical import Prompt, write_jsonl

MBPP = {"dataset": "google-research-datasets/mbpp", "config": "sanitized",
        "split": "train", "cap": 400}
APPS = {"dataset": "codeparrot/apps", "config": None, "split": "train", "cap": 1000}


def mbpp_prompt(row: dict, index: int) -> Prompt | None:
    prompt, tests = row.get("prompt"), row.get("test_list")
    if not prompt or not isinstance(tests, list) or not tests:
        return None
    return Prompt(
        id=f"mbpp-{row.get('task_id', index)}",
        domain="coding",
        origin="prebuilt",
        source={"name": MBPP["dataset"], "config": MBPP["config"], "row": index},
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        verify={"type": "python_tests",
                "setup": "\n".join(row.get("test_imports") or []),
                "tests": tests},
        meta={"difficulty": row.get("difficulty")},
    )


def apps_prompt(row: dict, index: int) -> Prompt | None:
    question, raw = row.get("question"), row.get("input_output")
    if not question or not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    inputs, outputs = data.get("inputs") or [], data.get("outputs") or []
    pairs = [[i, o] for i, o in zip(inputs, outputs)
             if isinstance(i, str) and isinstance(o, str)]
    if not pairs:
        return None
    return Prompt(
        id=f"apps-{row.get('problem_id', index)}",
        domain="coding",
        origin="prebuilt",
        source={"name": APPS["dataset"], "row": index},
        messages=[{"role": "user", "content": [{"type": "text", "text": question}]}],
        verify={"type": "python_io", "pairs": pairs[:20]},
        meta={"difficulty": row.get("difficulty")},
    )


def harvest(spec: dict, mapper) -> list[Prompt]:
    out: list[Prompt] = []
    stream = load_dataset(spec["dataset"], spec["config"], split=spec["split"], streaming=True)
    seen = 0
    for row in stream:
        seen += 1
        rec = mapper(row, seen)
        if rec is None:
            continue
        out.append(rec)
        if len(out) >= spec["cap"]:
            break
    print(f"{spec['dataset']} -> {len(out)} verification seeds")
    return out


def smoke() -> int:
    mb = mbpp_prompt({"task_id": 2, "prompt": "Write add(a, b).", "test_imports": [],
                      "test_list": ["assert add(1, 2) == 3"]}, 0)
    apps = apps_prompt({"problem_id": 1, "question": "Read two ints and print the sum.",
                        "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]})}, 0)
    print("mbpp verify:", mb.verify["type"], len(mb.verify["tests"]))
    print("apps verify:", apps.verify["type"], len(apps.verify["pairs"]))
    print("no-test row dropped:", mbpp_prompt({"prompt": "x", "test_list": []}, 0))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="unit-test verification seeds")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--smoke", action="store_true", help="fixture checks, no network")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke()
    seeds = harvest(MBPP, mbpp_prompt) + harvest(APPS, apps_prompt)
    if not seeds:
        print("no verification seeds; nothing written")
        return 1
    written = write_jsonl(args.root / "verification" / "seeds.jsonl", seeds)
    print(f"verification seeds: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `tools/dataset/verify.py`**

```python
"""Execute a teacher completion against a seed's verification spec.

Used by M2 to drop failed distillation attempts before they become student data. This
runs generated code with the local interpreter; it is a correctness check, not a security
sandbox — run generation and verification on a disposable environment/profile.
"""
from __future__ import annotations

import re
import subprocess
import sys

_FENCE = "`" * 3
_CODE_FENCE = re.compile(_FENCE + r"(?:python)?\s*(.*?)" + _FENCE, re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_FINAL = re.compile(r"####\s*([^\n]+)")


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_code(completion: str) -> str:
    blocks = _CODE_FENCE.findall(completion)
    return (blocks[-1] if blocks else strip_think(completion)).strip()


def run_python(source: str, stdin: str = "", timeout: float = 10.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run([sys.executable, "-c", source], input=stdin,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"exit {proc.returncode}: {tail[-1] if tail else 'no output'}"
    return True, proc.stdout


def check_python_tests(completion: str, tests: list[str], setup: str = "",
                       timeout: float = 10.0) -> tuple[bool, str]:
    body = "\n".join([setup, extract_code(completion), *tests])
    ok, info = run_python(body, timeout=timeout)
    return (True, "") if ok else (False, info)


def check_python_io(completion: str, pairs: list[list[str]],
                    timeout: float = 10.0) -> tuple[bool, str]:
    code = extract_code(completion)
    for stdin, expected in pairs[:20]:
        ok, info = run_python(code, stdin=stdin, timeout=timeout)
        if not ok:
            return False, info
        if info.strip() != expected.strip():
            return False, f"output mismatch: {info.strip()[:40]!r} != {expected.strip()[:40]!r}"
    return True, ""


def extract_answer(text: str) -> str:
    clean = strip_think(text)
    boxed = _BOXED.findall(clean)
    if boxed:
        return boxed[-1].strip()
    finals = _HASH_FINAL.findall(clean)
    if finals:
        return finals[-1].strip()
    return clean.strip().splitlines()[-1].strip() if clean.strip() else ""


def normalise_answer(text: str) -> str:
    text = text.strip().lower().replace(",", "").replace("$", "").replace("\\%", "%")
    return re.sub(r"\s+", " ", text).rstrip(".")


def check_answer(completion: str, gold: str) -> bool:
    return normalise_answer(extract_answer(completion)) == normalise_answer(gold)


def check_record(verify: dict | None, completion: str) -> tuple[bool, str]:
    """Dispatch on the seed's verify spec. Returns (passed, reason)."""
    if verify is None:
        return True, ""
    vtype = verify.get("type")
    if vtype == "python_tests":
        return check_python_tests(completion, verify.get("tests") or [],
                                  setup=verify.get("setup") or "")
    if vtype == "python_io":
        return check_python_io(completion, verify.get("pairs") or [])
    if vtype == "answer_match":
        ok = check_answer(completion, str(verify.get("gold") or ""))
        return (True, "") if ok else (False, "answer mismatch")
    return False, f"unknown verify type {vtype!r}"


if __name__ == "__main__":
    ok, _ = check_python_tests("def add(a, b):\n    return a + b",
                               ["assert add(1, 2) == 3"])
    bad, why_bad = check_python_tests("def add(a, b):\n    return a - b",
                                      ["assert add(1, 2) == 3"])
    io_ok, _ = check_python_io("print(sum(map(int, input().split())))", [["1 2", "3"]])
    print("tests positive:", ok, "| negative:", bad, "|", why_bad)
    print("io positive:", io_ok)
    print("answer match:", check_answer("<think>\n4\n</think>\n\n\\boxed{4}", "4"))
    print("answer mismatch:", check_answer("The answer is 5.", "4"))
```

- [ ] **Step 3: Smoke, then fetch the seeds**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --smoke
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.verify
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.testsets --root datasets/qwen35-4b-sft
```
Expected: `mbpp verify: python_tests 1`, `apps verify: python_io 1`,
`no-test row dropped: None`; `tests positive: True | negative: False | exit 1: AssertionError`,
`io positive: True`, `answer match: True`, `answer mismatch: False`; then
`google-research-datasets/mbpp -> 400 verification seeds`,
`codeparrot/apps -> 1000 verification seeds`, `verification seeds: 1400`.

- [ ] **Step 4: Commit**

```powershell
git add tools/dataset/testsets.py tools/dataset/verify.py
git commit -m "feat(dataset): unit-test verification seeds and completion checker"
```

---

## M2 contract — teacher generation (next plan, fixed here)

M1 stops at prompts. The generation plan (`generate.py`) must honour this contract,
because everything the student learns in Phase 1 is teacher-written:

1. **Inputs:** `prompts/train.jsonl` and `verification/seeds.jsonl`; `prompts/val.jsonl`
   is never generated for training.
2. **Verbatim reasoning:** request `enable_thinking: true` and store the completion as
   the assistant message exactly as returned, `<think>...</think>` block included. The
   student mimics the teacher's reasoning process because the tags are data, not framing
   (spec §5); the render gate (Task 15) proves the template carries them.
3. **Verification before training:** every prompt carrying a `verify` spec runs through
   `verify.check_record`; failed completions (wrong code, wrong answer) are dropped and
   counted per source. Unit-test seeds are the code oracle; gsm8k / NuminaMath golds
   support `answer_match` for maths.
4. **Response-side filters:** accepted rows still pass `filters.example_reject_reason`
   (balanced think tags, loop, refusal, length) before being written.
5. **Residual risk:** domains without an oracle (roleplay, chat-style coding,
   uncensored) pass filters and review only; teacher bugs survive there. Sample-audit
   those columns before scaling M3.
6. **Acceptance metric:** report per-source and per-domain pass rates in the M2 manifest
   — that number is the verifier's value and the guide for M3 caps.

---

## Spec sync — applied 2026-09-29

The design doc (`2026-09-28-qwen35-phase1-dataset-design.md`) has been updated to match
this plan: prompt-only ingestion, teacher-written responses for all four domains, the
verification corpus (§10.1, MBPP/APPS plus `verify.py`), the canonical `Prompt` record,
and the verbatim `<think>` / `</think>` preservation contract. This plan remains the
implementation source of truth for M1.

---

## Self-review

**Spec coverage.** Every M1-relevant section maps to a task: §3/§9 deliverables as prompts
(Tasks 1–3, 13–17), §5 schema (Task 1: `Prompt` added, `Example` kept for holdouts),
§6 adapters (Tasks 5–9), §7.1–7.3 sources including the smol-magpie quality filter
(Tasks 5–9, 11), §7.4 seed caps and §8 refusal quarantine (Task 10), §7.6 multimodal
prompts (Tasks 9, 11, verified in Task 16 Step 6), §8 eval holdout (Task 10 quarantine,
Task 16 Step 3 via `--extra-eval`, Tasks 17–18), §9 stages 1–7 as prompt stages
(Tasks 14, 15), §11.1–11.5 verification gates (Tasks 14–16, 18), and the distillation
hygiene this revision adds: unit-test seeds + verifier (Task 18) and the think-tag
preservation contract (Tasks 15, 18). Deliberately uncovered in this plan: §7.4 responses,
§10 teacher generation (bounded by the M2 contract), §11.6–11.7 (post-M2 measurements
and review), and M3 — all declared out of scope at the top.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code. Task 16 Step 5
shows the exact edit to `pipeline.py` rather than saying "wire in the token counter".

**Type consistency.** `Example` is adapter output and the holdout shape; `Prompt` is what
M1 persists; `strip_to_prompt()` is the only bridge. `Ctx.source(index)` is the single
source-dict builder. `Adapter.build(row, index, ctx)` is the same signature in every
adapter. `textutil.banded`, `Deduper.check_prompt`/`commit_prompt`,
`filters.prompt_reject_reason` (M1) and `filters.example_reject_reason` (M2), and
`verify.check_record` are referenced consistently. `calibrate.py` runs after M2.

**Known risk.** Task 16 Step 2 may return a failed probe, a closed generation prompt, or
an unbalanced verdict. That is a *finding*, not a failure: it would mean the stored tag
form and the template disagree, which is exactly what the gate exists to detect before
25,000 rows get generated. Stop, reconcile the `THINK_OPEN` / `THINK_CLOSE` constants
with the template (they are `<think>` / `</think>` in the current GGUF), and re-run.

**Distillation risk.** Unit tests cover the coding seeds and gold answers cover maths via
`answer_match`; roleplay, chat coding, and uncensored completions have no oracle and
inherit the teacher (M2 contract item 5). Sample-audit those columns before M3.

**Not committed:** this plan document.
