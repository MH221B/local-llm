# Phase 1 / M2 — Teacher generation, verification, and multi-turn trajectories

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn M1's prompt corpus into trainable records by making the local 9B teacher write **every** response — single-turn, Magpie-invented, and multi-turn — then verify each completion (unit tests, golds, structural turn checks) and apply response filters before anything is written to `train.jsonl` / `val.jsonl`. Failed attempts are dropped and counted per source so M3's caps are set from measured pass rates, not hope.

**Architecture:** One OpenAI-compatible teacher client and one append-only resume cache sit under three generation shapes: seeded single-turn (`generate.py`), Magpie prompt invention (`magpie.py`), and a multi-turn loop (`trajectory.py`) with two provenance modes — *prebuilt-backed* (source user turns and tool results kept as context, every assistant turn regenerated) and *simulated* (the teacher plays user, agent, and tool-environment). All three funnel through the same verification stack: `verify.check_record` / `verify.check_trajectory` for oracles and structure, `filters.example_reject_reason` / `filters.trajectory_reject_reason` for response hygiene. Only rows that pass both are written. Multi-turn needs a fourth canonical role (`tool`) and a `Trajectory` record; both are added here, not in M1.

**Tech Stack:** Python 3.12 in the `dataset` conda env; stdlib `urllib` for the client (no new dependency), `jinja2` for the multi-turn render gate. Teacher is the local `MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf` + `.mmproj-BF16.gguf` served by `projects/local-llm/llama-cpp/llama-server.exe` on port 8086 (spec §10 names `mradermacher/Ornith-1.5-9B-Abliterated-i1-GGUF`; the local box has the earlier MiMo-Ornith build, which PIPELINE.md Phase 1 already names — the server launch is parameterised so the spec's GGUF can be swapped in later). Teacher sampling is temp 0.6 / top_p 0.95 / top_k 20 / `enable_thinking: true` (spec §10).

**Testing:** No new unit tests (inherited from M1 — not opted in). Every task verifies with its own `__main__` smoke check or a CLI run and a stated expected output. Server-dependent checks are isolated to the launch/canary steps; every other smoke injects a `FakeTeacher` so it runs offline and deterministically. The spec's §11.4 positive/negative verifier controls are Task 8's smoke.

**Prerequisites (must be true before Task 1):** M1 Tasks 14–18 are complete, so `datasets/qwen35-4b-sft/prompts/train.jsonl`, `prompts/val.jsonl`, `verification/seeds.jsonl`, `seeds/uncensored.jsonl`, `eval/refusal.jsonl`, `eval/vision.jsonl`, `tools/dataset/student.jinja`, and `tools/dataset/{verify,render,tokenserver,calibrate,contaminate}.py` all exist. Task 1 fixes two M1 defects the M2 work depends on.

**Spec:** `docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md` (§5, §5.1, §5.2, §10, §10.1, §10.2, §11.2, §11.4, §12)

**How to run a module:** from the repo root (`projects/local-llm`), using the env interpreter:

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.<module> [args...]
```

---

## Scope

This plan covers the **M2 milestone**: teacher-written responses for all four columns, single-turn and multi-turn, verified and filtered, at M2 scale (~500 examples per spec §12) with pass rates recorded. Deferred:

| Deferred | Why |
|---|---|
| Full 25,000-example M3 run | Same CLIs with larger `--target` and Colab vLLM as `--base-url`; operational, not new code (§12) |
| Magpie-invented **tool schemas** at scale | M2 invents a small number; the cap arithmetic for `teacher:tools`/`teacher:uncensored` is M3's (§7.2, §7.4) |
| Model-judge scoring of roleplay coherence | No oracle exists (§10.1 residual risk); M2 records the gap and sample-audits, it does not build a judge |
| RL/DPO for multi-turn robustness | Accepted residual risk; no RL stage is planned (§14) |

### Known state discrepancies this plan resolves

1. **Think tag spelling.** The authoritative tag is `<think>` / `</think>` (§5.2, §11.2; M1 plan). `filters.py` may be correct at HEAD, but the spec still carries two typos (`" thinking"` on lines 107 and 195). Task 1 verifies the constant and fixes both spec lines.
2. **Maths golds are discarded.** M1's `plainmath.py` builds gsm8k / NuminaMath prompts with no `verify`, so `strip_to_prompt` drops the answer and `answer_match` has nothing to check against (§7.7, §10.1). Task 1 attaches the golds and re-runs the prompt pipeline.

## File structure

```
tools/dataset/
  teacher.py        OpenAI-compatible teacher client: chat completion, image data URIs, retries
  gencache.py       append-only resume cache keyed by request / trajectory-prefix hash
  canonical.py      + tool role, Trajectory, trajectory helpers, *_from_dict loaders
  trajparse.py      prebuilt multi-turn sources -> prompt trajectories (Hermes-FC, ToolACE, chat)
  trajbuild.py      fetch those sources -> prompts/trajectories.jsonl
  trajectory.py     multi-turn teacher loop: prebuilt-backed and simulated
  magpie.py         teacher-invented user prompts (coding/tools, uncensored)
  generate.py       M2 CLI: all | merge
  verify.py         + check_turns, check_trajectory
  filters.py        + response_reject_reason, trajectory_reject_reason
  render.py         + multi-turn gate (mask markers, per-turn think survival, tool round-trip)
  report.py         + merge_manifest (pass rates)
  prompts/
    magpie_tools.md       Magpie instruction template: tool-use requests
    magpie_uncensored.md  Magpie instruction template: adversarial/borderline requests
  m2/               run outputs: single.jsonl, val.jsonl, trajectory.jsonl, simulated.jsonl, *.stats.json
```

Run artefacts under `datasets/qwen35-4b-sft/`: `train.jsonl`, `val.jsonl`, `manifest.json` (updated in place), `calibration.txt`.

Every new module has a `__main__` smoke check; each task's verification is one command with a predictable line of output.

---

### Task 1: Repair the M1 defects M2 depends on

**Files:**
- Modify: `tools/dataset/filters.py` (only if the constant is not already `<think>`)
- Modify: `tools/dataset/adapters/plainmath.py`
- Modify: `docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md:107`
- Modify: `docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md:195`

- [x] **Step 1: Confirm the think-tag constant and fix the spec typos**

First check the working tree (M1 may already have corrected it):

```powershell
Select-String -Path tools/dataset/filters.py -Pattern 'THINK_'
```
If `THINK_OPEN` is not `"<think>"`, set it to match the real template pair (spec §5.2, §11.2):

```python
# The student template stores the CoT as literal ASCII tags; confirmed against the
# GGUF in M1 Task 16 (spec sections 5 and 11.2).
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
```

The spec still carries two typos where `<think>` lost its `<`. Fix both:

```json
    {"role": "assistant", "content": "<think>The blue bar reaches 40...</think>\n\nThe blue bar."}
```
and on line 195 change `` a ` thinking` block `` to `` a `<think>` block ``.

- [x] **Step 2: Attach maths golds as `answer_match` verify specs**

Replace `tools/dataset/adapters/plainmath.py` with this version. `strip_to_prompt` reads `ex.meta["verify"]`, so the specs ride the prompt into `prompts/train.jsonl` and become M2's oracle:

```python
"""NuminaMath-1.5 and gsm8k. Plain columns plus the validity flags (spec section 7.1)."""
from __future__ import annotations

import re

from ..canonical import Example
from .base import Ctx, plain_pair

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_FINAL = re.compile(r"####\s*([^\n]+)")


def _gsm8k_gold(answer: str) -> str | None:
    """gsm8k solutions end with the final answer after `####`."""
    finals = _HASH_FINAL.findall(answer)
    return finals[-1].strip() if finals else None


def _numina_gold(solution: str) -> str | None:
    """NuminaMath solutions put the final answer in the last \\boxed{}.

    Measured against the live dataset: NuminaMath-1.5's `solution` is competition-maths
    prose that states its answer inline ("Answer: 0 or 3", "the answer is no") and does
    **not** use `\\boxed{}` — 0 of 2,830 valid rows sampled carried one. So this extractor
    attaches almost no golds in practice; gsm8k below is what actually supplies them.
    Kept because the cost is nil and a future Numina config may ship boxed answers.
    """
    boxed = _BOXED.findall(solution)
    return boxed[-1].strip() if boxed else None


def build_numina(row: dict, index: int, ctx: Ctx) -> Example | None:
    # NuminaMath-1.5 ships these as strings ("Yes" / "Incomplete" / "No"), not bools.
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return None
    problem, solution = row.get("problem"), row.get("solution")
    if not problem or not solution:
        return None
    meta = {"problem_type": row.get("problem_type"), "synthetic": row.get("synthetic")}
    gold = _numina_gold(solution)
    if gold:
        meta["verify"] = {"type": "answer_match", "gold": gold}
    return Example(
        id=f"numina-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(problem, solution),
        meta=meta,
    )


def build_gsm8k(row: dict, index: int, ctx: Ctx) -> Example | None:
    question, answer = row.get("question"), row.get("answer")
    if not question or not answer:
        return None
    meta: dict = {}
    gold = _gsm8k_gold(answer)
    if gold:
        meta["verify"] = {"type": "answer_match", "gold": gold}
    return Example(
        id=f"gsm8k-{index}",
        domain=ctx.domain,
        origin="prebuilt",
        source=ctx.source(index),
        messages=plain_pair(question, answer),
        meta=meta,
    )


if __name__ == "__main__":
    ctx = Ctx("AI-MO/NuminaMath-1.5", None, "reasoning", "apache-2.0")
    good = {"problem": "Solve x.", "solution": "Thus \\boxed{1}.", "problem_is_valid": "Yes",
            "solution_is_valid": "Yes", "problem_type": "algebra", "synthetic": False}
    bad = {"problem": "Solve y.", "solution": "y = 2", "problem_is_valid": "Incomplete",
           "solution_is_valid": "Yes"}
    ex = build_numina(good, 5, ctx)
    print(ex.id, ex.meta["verify"])
    print("invalid dropped:", build_numina(bad, 6, ctx))

    gctx = Ctx("openai/gsm8k", "main", "reasoning", "mit")
    g = build_gsm8k({"question": "1+1?", "answer": "Sum is 1+1.\n#### 2"}, 1, gctx)
    print(g.id, g.meta["verify"], "| tail:", g.messages[-1]["content"][-6:])
```

- [x] **Step 3: Smoke the fixed adapter**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.adapters.plainmath
```
Expected: `numina-5 {'type': 'answer_match', 'gold': '1'}`, `invalid dropped: None`, then `gsm8k-1 {'type': 'answer_match', 'gold': '2'} | tail: #### 2`.

- [x] **Step 4: Re-run the M1 prompt pipeline so the golds land in the prompt corpus**

Precondition: the student server must be up on port 8085 for the exact token counter — `pipeline.py` defaults `--token-server http://127.0.0.1:8085` with no off switch, and `LlamaServer` has no fallback. Start it per M1 Task 16 Step 1 if it is not running.

Run:
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.pipeline --target-train 3000 --val-size 200 --root datasets/qwen35-4b-sft
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset.canonical import iter_jsonl; n=sum(1 for r in iter_jsonl('datasets/qwen35-4b-sft/prompts/train.jsonl') if r.get('verify')); print('prompts with verify specs:', n)"
```
Expected: `prompts train 3000, val 200`; then a small non-zero count of verify specs — only
the gsm8k rows carry one (its `#### gold` is present on ~100% of rows), and 74 gsm8k prompts
are in the 3,000-prompt corpus, so expect **roughly 70–80, not "a few hundred"**. The
NuminaMath rows contribute zero: their solutions do not use `\boxed{}` (see `_numina_gold`).
A zero print would mean the golds did not ride through `strip_to_prompt`; a number near 74
is the pass.

- [x] **Step 5: Commit**

```powershell
git add tools/dataset/filters.py tools/dataset/adapters/plainmath.py docs/superpowers/specs/2026-09-28-qwen35-phase1-dataset-design.md
git commit -m "fix(dataset): correct think tag spelling and attach maths golds"
```

---

### Task 2: Teacher client (spec §10)

**Files:**
- Create: `tools/dataset/teacher.py`

- [x] **Step 1: Write `tools/dataset/teacher.py`**

```python
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

# Spec section 10 sampling.
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 4096


class TeacherError(RuntimeError):
    pass


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
                 max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE,
                 top_p: float = TOP_P, top_k: int = TOP_K, seed: int | None = None) -> dict:
        """One teacher turn. Returns the assistant message dict (content, tool_calls)."""
        payload = {
            "model": self.model,
            "messages": to_wire_messages(messages, store),
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        reply = self._post_retry("/v1/chat/completions", payload)
        return reply["choices"][0]["message"]


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
```

- [x] > **Reasoning-trace capture (found while running Task 15).** llama-server's default `--reasoning-format auto` extracts the model's thoughts into a separate `message.reasoning_content` field and leaves `content` as the answer only. The canonical schema (spec §5.2) and both chat templates expect the reasoning *inside* `content` as `<think>…</think>`, so `TeacherClient.complete` folds `reasoning_content` back into `content` via `_fold_reasoning` at that single boundary. Without it, every generated row silently lost its trace (measured: 0 of 24 accepted rows carried a `<think>` block). Server-side alternative: launch with `--reasoning-format deepseek-legacy`, which keeps the tags in `content`; the client-side fold is kept so capture does not depend on the launch flag. The cache was pruned and the seeded phase re-run after the fix.

**Step 2: Launch the teacher server**

Teacher GGUF and projector are already on disk (PIPELINE.md Phase 1). Port 8086 is distinct from the student's 8085. The M2 driver is sequential, so a single slot is enough; `-c 32768` keeps long multi-turn prefixes (which resend images) inside one slot's context.

```powershell
Start-Process -FilePath "C:\Users\tnmh\projects\local-llm\llama-cpp\llama-server.exe" `
  -ArgumentList @("-m","C:\Users\tnmh\projects\local-llm\models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf",
                  "--mmproj","C:\Users\tnmh\projects\local-llm\models\MiMo-Ornith-9B-AGSI-Abliterated-HQ.mmproj-BF16.gguf",
                  "-ngl","99","-fa","on","-c","32768","-np","1","--port","8086") `
  -WindowStyle Hidden
Start-Sleep -Seconds 20
Invoke-RestMethod http://127.0.0.1:8086/health
```
Expected: `status` is `ok`.

- [x] **Step 3: Smoke the client — text and image — against the live server**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.teacher http://127.0.0.1:8086
```
Expected: a model path ending `.gguf`, a non-zero template length, a short text reply such as `'ok'`, and a short image reply naming a colour (e.g. `'red'`) — the §10 image data-URI canary.

- [x] **Step 4: Commit**

```powershell
git add tools/dataset/teacher.py
git commit -m "feat(dataset): OpenAI-compatible teacher client"
```

---

### Task 3: Resume cache (spec §10)

**Files:**
- Create: `tools/dataset/gencache.py`

- [x] **Step 1: Write `tools/dataset/gencache.py`**

```python
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
```

- [x] **Step 2: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.gencache
```
Expected: `records after reload: 2`, `replayed value: hello`, `same key twice: True`.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/gencache.py
git commit -m "feat(dataset): append-only resume cache"
```

---

### Task 4: Trajectory schema and validators (spec §5.1)

**Files:**
- Modify: `tools/dataset/canonical.py`

- [x] **Step 1: Add the `tool` role, `Trajectory`, helpers, and JSONL loaders**

In `tools/dataset/canonical.py`, change the role tuple and add the trajectory record after `Prompt`:

```python
ROLES = ("system", "user", "assistant", "tool")
```

```python
@dataclass
class Trajectory:
    """A multi-turn record: system + user/assistant/tool turns, ending on an assistant turn.

    M2 stores the whole conversation. Prebuilt tool results are observations and are kept;
    every assistant turn is teacher-written before the row can be trained (spec section 5.1).
    """
    id: str
    domain: str
    origin: str
    source: dict
    messages: list[dict]
    tools: Any = None
    images: list[ImageRef] = field(default_factory=list)
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


def tool_calls_of(message: dict) -> list[dict]:
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def tool_call_arguments(call: dict) -> dict:
    """A tool call's arguments as a mapping, for template rendering (spec sections 5.1, 11.2).

    Canonical records and the OpenAI wire format both store `arguments` as a JSON string,
    but the student template iterates it as a mapping (`tool_call.arguments|items`). This
    is the single place that converts between the two, so the schema stays a string.
    Returns `{}` when the arguments are absent or unparseable rather than raising; a
    malformed call is caught by `validate_trajectory`, not silently mis-rendered.
    """
    args = (call.get("function") or {}).get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def for_template(messages: list[dict]) -> list[dict]:
    """A render-time view: tool-call `arguments` become mappings, everything else is copied.

    The student template reads `tool_call.arguments|items`, which requires a mapping; the
    stored form is a JSON string. Convert here rather than mutating the record so the
    schema and the written JSONL keep the OpenAI string form (spec section 5.1).
    """
    out: list[dict] = []
    for m in messages:
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        rendered = []
        for call in calls:
            fn = dict(call.get("function") or {})
            fn["arguments"] = tool_call_arguments(call)
            rendered.append({**call, "function": fn})
        out.append({**m, "tool_calls": rendered})
    return out


def declared_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    for entry in tools or []:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def final_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return message_text(m)
    return ""


def validate_trajectory(t: Trajectory) -> list[str]:
    """Return a list of problems. Empty list means valid (spec sections 5.1, 11.3)."""
    problems: list[str] = []
    if not t.id:
        problems.append("empty id")
    if t.domain not in DOMAINS:
        problems.append(f"bad domain: {t.domain!r}")
    if t.origin not in ORIGINS:
        problems.append(f"bad origin: {t.origin!r}")
    if not isinstance(t.source, dict) or "name" not in t.source:
        problems.append("source missing name")
    if not t.messages:
        return problems + ["no messages"]

    roles = [m.get("role") for m in t.messages]
    for r in roles:
        if r not in ROLES:
            problems.append(f"bad role: {r!r}")
    if roles[-1] != "assistant":
        problems.append("last message is not assistant")
    if "system" in roles and roles[0] != "system":
        problems.append("system message is not first")

    calls: dict[str, int] = {}
    answered: set[str] = set()
    for i, m in enumerate(t.messages):
        if m.get("role") == "assistant":
            for call in tool_calls_of(m):
                cid = call.get("id")
                if not cid or call.get("type") != "function":
                    problems.append(f"bad tool_call at message {i}")
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    problems.append(f"tool_call without function name at message {i}")
                calls[cid] = i
                args = (fn or {}).get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        problems.append(f"tool_call arguments are not JSON at message {i}")
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if not cid:
                problems.append(f"tool message without tool_call_id at message {i}")
            elif cid not in calls:
                problems.append(f"tool result {cid!r} has no preceding call")
            else:
                answered.add(cid)
    for cid, idx in calls.items():
        if cid not in answered:
            problems.append(f"call {cid!r} at message {idx} has no tool result")

    refs = {i.sha256 for i in t.images}
    for m in t.messages:
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

    if t.tools is not None and not isinstance(t.tools, list):
        problems.append("tools is neither list nor null")
    if t.tokens < 0:
        problems.append("negative token count")
    if t.verify is not None and not isinstance(t.verify, dict):
        problems.append("verify is neither dict nor null")
    return problems


def _images_from(raw: list) -> list[ImageRef]:
    return [ImageRef(**{k: d.get(k) for k in ("sha256", "path", "w", "h", "origin_url")})
            for d in raw or []]


def prompt_from_dict(row: dict) -> Prompt:
    return Prompt(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        verify=row.get("verify"), meta=row.get("meta") or {})


def trajectory_from_dict(row: dict) -> Trajectory:
    return Trajectory(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        verify=row.get("verify"), meta=row.get("meta") or {})


def example_from_dict(row: dict) -> Example:
    return Example(
        id=row["id"], domain=row["domain"], origin=row["origin"], source=row["source"],
        messages=row["messages"], images=_images_from(row.get("images")),
        tools=row.get("tools"), tokens=row.get("tokens", 0),
        meta=row.get("meta") or {})
```

- [x] **Step 2: Extend the smoke block**

Replace the `if __name__ == "__main__":` block at the end of `canonical.py` with:

```python
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

    ok_traj = Trajectory(
        id="traj-1", domain="coding", origin="teacher", source={"name": "smoke"},
        messages=[
            {"role": "user", "content": "Book a table for two."},
            {"role": "assistant", "content": "Checking.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "book", "arguments": "{\"n\": 2}"}}]},
            {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Booked — T-991."},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}],
        tokens=42,
    )
    bad_traj = Trajectory(
        id="traj-2", domain="coding", origin="teacher", source={"name": "smoke"},
        messages=[
            {"role": "assistant", "content": "x",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "book", "arguments": "{bad"}}]},
            {"role": "tool", "content": "r", "tool_call_id": "call_9"},
        ],
    )
    print("traj ok:", validate_trajectory(ok_traj),
          [m["role"] for m in ok_traj.messages])
    print("traj bad:", len(validate_trajectory(bad_traj)), "problems")

    call = ok_traj.messages[1]["tool_calls"][0]
    print("args as mapping:", tool_call_arguments(call))
    print("args unparseable:", tool_call_arguments(
        {"function": {"name": "book", "arguments": "{bad"}}))
    rendered = for_template(ok_traj.messages)
    print("for_template converts:", isinstance(
        rendered[1]["tool_calls"][0]["function"]["arguments"], dict),
        "| record unchanged:", isinstance(call["function"]["arguments"], str))
```

- [x] **Step 3: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.canonical
```
Expected: `good: []`, `bad: 2 problems`, `prompt: [] ['user']`, then
`traj ok: [] ['user', 'assistant', 'tool', 'assistant']` and `traj bad: 4 problems`
(non-JSON arguments, orphan tool result, unanswered call, and last message not assistant);
then `args as mapping: {'n': 2}`, `args unparseable: {}`, and
`for_template converts: True | record unchanged: True`.

- [x] **Step 4: Commit**

```powershell
git add tools/dataset/canonical.py
git commit -m "feat(dataset): trajectory record, tool role, and trajectory validator"
```

---

### Task 5: Trajectory ingestion — Hermes-FC (spec §5.1)

**Files:**
- Create: `tools/dataset/trajparse.py`

Hermes-FC ships `conversations` with `from` in `{system, human, gpt, tool}`. Assistant turns embed one or more `<tool_call>{json}</tool_call>` blocks as **text**; tool turns embed `<tool_response>{json}</tool_response>`. This task converts them to the structured OpenAI shape and emits a prompt-side trajectory: system + user context + the source's observations, with assistant turns left as scaffolds for M2 to regenerate.

- [x] **Step 1: Write `trajparse.py` with the Hermes parser**

```python
"""Prebuilt multi-turn sources -> prompt trajectories for M2.

Assistant turns are teacher-written in M2, so this module keeps only the context the
teacher needs: system and user turns, the declared `tools` schema, and — for tool-calling
sources — the observations (`tool` results) that follow each call. Assistant turns are
kept as inert scaffolds (their content is ignored, their `tool_calls` only exist so a
following observation has a valid `tool_call_id`) and are marked `"_scaffold": true`.
`trajectory.generate_prebuilt` regenerates them and drops the flag before writing.
"""
from __future__ import annotations

import json

from .canonical import Trajectory
from .adapters.base import Ctx, system_msg, user_msg, assistant_msg, text_part

HERMES_DATASET = "NousResearch/hermes-function-calling-v1"
_TAG = "<tool_call>"


def _blocks(text: str, tag: str) -> list[str]:
    out, chunks = [], text.split(f"<{tag}>")
    for chunk in chunks[1:]:
        out.append(chunk.split(f"</{tag}>")[0].strip())
    return out


def _hermes_tool_schema(raw) -> list | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    return raw


def _hermes_messages(conversations: list) -> list[dict] | None:
    messages: list[dict] = []
    call_seq = 0
    for turn in conversations:
        frm, value = turn.get("from"), turn.get("value")
        if not isinstance(value, str):
            return None
        if frm == "system":
            text = value
            if "<tools>" in text:
                text = text.split("<tools>")[0].strip()
            messages.append(system_msg(text))
        elif frm == "human":
            messages.append(user_msg([text_part(value)]))
        elif frm == "gpt":
            blocks = _blocks(value, "tool_call")
            prose = value
            for b in blocks:
                prose = prose.replace(f"{_TAG}\n{b}\n</tool_call>", "")
                prose = prose.replace(f"{_TAG}{b}</tool_call>", "")
            prose = prose.strip()
            if blocks:
                calls = []
                for b in blocks:
                    try:
                        obj = json.loads(b)
                    except json.JSONDecodeError:
                        return None
                    name = obj.get("name")
                    args = obj.get("arguments", {})
                    if not isinstance(name, str) or not name:
                        return None
                    calls.append({"id": f"call_{call_seq}", "type": "function",
                                  "function": {"name": name,
                                               "arguments": json.dumps(args, ensure_ascii=False)}})
                    call_seq += 1
                msg = assistant_msg(prose)
                msg["tool_calls"] = calls
                msg["_scaffold"] = True
                messages.append(msg)
            else:
                messages.append(assistant_msg(value))
        elif frm == "tool":
            responses = _blocks(value, "tool_response")
            content = responses[0] if responses else value
            messages.append({"role": "tool", "content": content, "tool_call_id": ""})
        else:
            return None
    if not messages or messages[-1]["role"] != "assistant":
        return None
    # Pair each scaffold call with the tool message that follows it, in order.
    pending: list[str] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("_scaffold"):
            pending = [c["id"] for c in m["tool_calls"]]
        elif m["role"] == "tool":
            if not pending:
                return None
            m["tool_call_id"] = pending.pop(0)
    if any(m["role"] == "tool" and not m["tool_call_id"] for m in messages):
        return None
    return messages


def build_hermes(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    tools = _hermes_tool_schema(row.get("tools"))
    messages = _hermes_messages(row.get("conversations") or [])
    if messages is None:
        return None
    if tools is not None and any(
            m.get("tool_calls") and c["function"]["name"] not in
            {e["function"]["name"] for e in tools if isinstance(e.get("function"), dict)}
            for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []):
        return None
    return Trajectory(
        id=f"hermesfc-{index}", domain=ctx.domain, origin="prebuilt",
        source=ctx.source(index), messages=messages, tools=tools,
        meta={"category": row.get("category"), "task": row.get("task"), "simulated": False},
    )


if __name__ == "__main__":
    ctx = Ctx(HERMES_DATASET, "func_calling", "coding", "apache-2.0")
    row = {
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model.\n<tools>\n[{}]\n</tools>"},
            {"from": "human", "value": "Check the front door camera."},
            {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera", "arguments": {"id": "front"}}\n</tool_call>'},
            {"from": "tool", "value": '<tool_response>\n{"status": "ok"}\n</tool_response>'},
            {"from": "gpt", "value": "The front door camera is live."},
        ],
        "tools": json.dumps([{"type": "function", "function": {"name": "get_camera"}}]),
        "category": "security", "task": "tool_call",
    }
    traj = build_hermes(row, 3, ctx)
    print("id:", traj.id, "roles:", [m["role"] for m in traj.messages],
          "calls:", len(traj.messages[2].get("tool_calls") or []))
    print("tool_call_id paired:", traj.messages[3]["tool_call_id"])
    print("system stripped:", "<tools>" not in traj.messages[0]["content"])
    print("no-last-assistant dropped:", build_hermes(
        {"conversations": [{"from": "human", "value": "hi"}]}, 4, ctx))
```

- [x] **Step 2: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajparse
```
Expected: `id: hermesfc-3 roles: ['system', 'user', 'assistant', 'tool', 'assistant'] calls: 1`,
`tool_call_id paired: call_0`, `system stripped: True`, `no-last-assistant dropped: None`.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/trajparse.py
git commit -m "feat(dataset): hermes function-calling trajectory parser"
```

---

### Task 6: Trajectory ingestion — ToolACE (spec §5.1)

**Files:**
- Modify: `tools/dataset/trajparse.py`

ToolACE ships a `system` column (prose that embeds `[{name, description, parameters, required}]`) and `conversations` with assistant calls in a **Python-call DSL**: `[Name(arg="x"), Name2(a=1)]`, where names may contain spaces and parentheses (`User Feed (Video Posts) V2`). Tool results are JSON, often a list.

- [x] **Step 1: Append the ToolACE parser to `trajparse.py`**

Add before the `if __name__ == "__main__":` block:

```python
import re

TOOLACE_DATASET = "Team-ACE/ToolACE"
# Argument values are quoted strings or scalars and contain no parentheses, so a call can
# be split at its outermost parens. Function names may contain spaces and balanced parens
# ("User Feed (Video Posts) V2"), so we split the body on top-level commas (tracking paren
# depth and quote state) and take the last "(" as the call's opening paren.
_ARG = re.compile(r"(\w+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\d+|True|False|None)")


def _split_calls(body: str) -> list[str]:
    """Split `A(x), B(y)` into ['A(x)', 'B(y)'] at top-level commas only."""
    parts: list[str] = []
    depth, quote, start = 0, "", 0
    for i, ch in enumerate(body):
        if quote:
            if ch == quote and body[i - 1] != "\\":
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p.strip() for p in parts if p.strip()]


def _toolace_calls(text: str) -> list[dict] | None:
    """Parse ToolACE's `[Name(arg=val), ...]` DSL into structured calls."""
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return None
    calls: list[dict] = []
    for chunk in _split_calls(body[1:-1]):
        if not chunk.endswith(")"):
            return None
        name, args_blob = chunk[:-1].rsplit("(", 1)
        name = name.strip()
        if not name:
            return None
        args: dict = {}
        for key, raw in _ARG.findall(args_blob):
            try:
                args[key] = json.loads(raw)
            except json.JSONDecodeError:
                args[key] = raw.strip("'\"")
        calls.append({"id": f"call_{len(calls)}", "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls or None


def _toolace_schema(system: str) -> list | None:
    """Extract the embedded JSON function list and wrap it in the OpenAI shape."""
    start = system.find("[")
    end = system.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(system[start:end + 1])
    except json.JSONDecodeError:
        return None
    tools = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        tools.append({"type": "function", "function": {
            "name": entry["name"], "description": entry.get("description", ""),
            "parameters": entry.get("parameters", {})}})
    return tools or None


def build_toolace(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    system = row.get("system") or ""
    tools = _toolace_schema(system)
    messages: list[dict] = []
    # Keep the prose preamble before the embedded JSON function list as the system turn;
    # the function list itself becomes the structured `tools` field.
    if "[" in system:
        preamble = system[:system.find("[")].strip()
        if preamble:
            messages.append(system_msg(preamble))
    pending: list[str] = []
    for turn in row.get("conversations") or []:
        frm, value = turn.get("from"), turn.get("value")
        if not isinstance(value, str):
            return None
        if frm == "user":
            messages.append(user_msg([text_part(value)]))
        elif frm == "assistant":
            calls = _toolace_calls(value)
            if calls:
                msg = assistant_msg("")
                msg["tool_calls"] = calls
                msg["_scaffold"] = True
                pending = [c["id"] for c in calls]
                messages.append(msg)
            else:
                messages.append(assistant_msg(value))
        elif frm == "tool":
            if not pending:
                return None
            messages.append({"role": "tool", "content": value, "tool_call_id": pending.pop(0)})
        else:
            return None
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return Trajectory(
        id=f"toolace-{index}", domain=ctx.domain, origin="prebuilt",
        source=ctx.source(index), messages=messages, tools=tools,
        meta={"simulated": False},
    )
```

Also extend the `__main__` smoke with a ToolACE fixture:

```python
    actx = Ctx(TOOLACE_DATASET, None, "coding", "apache-2.0")
    trow = {
        "system": 'You are an expert in composing functions. Here is a list of functions:\n'
                  '[{"name": "Timezones", "description": "Get times.", "parameters": {}}]',
        "conversations": [
            {"from": "user", "value": "What time is it in New York and Tokyo?"},
            {"from": "assistant", "value": '[Timezones(timezone="New York"), Timezones(timezone="Tokyo")]'},
            {"from": "tool", "value": '[{"name": "Timezones", "results": {}}]'},
            {"from": "assistant", "value": "New York is 3pm; Tokyo is 4am."},
        ],
    }
    tt = build_toolace(trow, 9, actx)
    print("toolace roles:", [m["role"] for m in tt.messages],
          "calls:", len(tt.messages[2]["tool_calls"]),
          "tool:", tt.messages[2]["tool_calls"][1]["function"]["arguments"])
    print("toolace tools:", len(tt.tools), "no-call row dropped:", build_toolace(
        {"system": "x", "conversations": [{"from": "user", "value": "hi"}]}, 10, actx))
```

- [x] **Step 2: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajparse
```
Expected: the Task 5 lines, then
`toolace roles: ['system', 'user', 'assistant', 'tool', 'assistant'] calls: 2 tool: {"timezone": "Tokyo"}` and
`toolace tools: 1 no-call row dropped: None`.

- [x] **Step 3: Commit**

```powershell
git add tools/dataset/trajparse.py
git commit -m "feat(dataset): ToolACE DSL trajectory parser"
```

---

### Task 7: Trajectory ingestion — roleplay chat, corpus export

**Files:**
- Modify: `tools/dataset/trajparse.py`
- Create: `tools/dataset/trajbuild.py`

Roleplay multi-turn comes from SmolTalk `systemchats-30k` and `everyday-conversations` (plain `messages` shape) and OpenHermes roleplay (`conversations`). No tool turns: the teacher fills every assistant slot.

- [x] **Step 1: Add the chat parser and corpus builder to `trajparse.py`**

Add before `__main__`:

```python
def _chat_messages(conversations: list) -> list[dict] | None:
    """SmolTalk/OpenHermes chat turns -> trajectories whose assistant turns are scaffolds."""
    messages: list[dict] = []
    for turn in conversations:
        frm = turn.get("from") or turn.get("role")
        value = turn.get("value")
        if value is None:
            value = turn.get("content")
        if not isinstance(value, str):
            return None
        if frm in ("system",):
            messages.append(system_msg(value))
        elif frm in ("human", "user"):
            messages.append(user_msg([text_part(value)]))
        elif frm in ("gpt", "assistant"):
            msg = assistant_msg("")
            msg["_scaffold"] = True
            messages.append(msg)
        else:
            return None
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages


def build_chat(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    """For a `messages` shaped row use `messages`; for ShareGPT use `conversations`."""
    turns = row.get("messages") or row.get("conversations") or []
    messages = _chat_messages(turns)
    if messages is None:
        return None
    if sum(1 for m in messages if m["role"] == "user") < 2:
        return None  # not actually multi-turn
    return Trajectory(
        id=f"chat-{ctx.name.split('/')[-1]}-{ctx.config or 'default'}-{index}",
        domain=ctx.domain, origin="prebuilt", source=ctx.source(index),
        messages=messages, meta={"simulated": False},
    )


def build_openhermes(row: dict, index: int, ctx: Ctx) -> Trajectory | None:
    """OpenHermes-2.5 roleplay conversations (spec section 5.1)."""
    if row.get("category") != "roleplay":
        return None
    messages = _chat_messages(row.get("conversations") or [])
    if messages is None or sum(1 for m in messages if m["role"] == "user") < 2:
        return None
    return Trajectory(
        id=f"openhermes-roleplay-{index}", domain=ctx.domain, origin="prebuilt",
        source=ctx.source(index), messages=messages, meta={"simulated": False},
    )
```

- [x] **Step 2: Write `tools/dataset/trajbuild.py`**

```python
"""Fetch the prebuilt multi-turn sources and emit prompts/trajectories.jsonl (spec 5.1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

from . import canonical, trajparse
from .adapters.base import Ctx

# (dataset, config, split, domain, parser, cap). Caps are prompt-candidate maxima; the
# M2 run samples a subset. spec sections 7.2, 7.3.
SOURCES = [
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "coding", "hermes", 800),
    ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "coding", "hermes", 400),
    ("Team-ACE/ToolACE", None, "train", "coding", "toolace", 800),
    ("HuggingFaceTB/smoltalk", "systemchats-30k", "train", "roleplay", "chat", 400),
    ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "roleplay", "chat", 400),
    ("teknium/OpenHermes-2.5", None, "train", "roleplay", "openhermes", 400),
]

PARSERS = {
    "hermes": trajparse.build_hermes,
    "toolace": trajparse.build_toolace,
    "chat": trajparse.build_chat,
    "openhermes": trajparse.build_openhermes,
}


def harvest(limit_per_source: int, dry_run: bool = False) -> list[canonical.Trajectory]:
    out: list[canonical.Trajectory] = []
    for dataset, config, split, domain, parser, cap in SOURCES:
        want = min(cap, limit_per_source)
        if dry_run:
            print(f"[dry] {dataset}:{config} domain={domain} parser={parser} take={want}")
            continue
        ctx = Ctx(dataset, config, domain, None)
        build = PARSERS[parser]
        stream = load_dataset(dataset, config, split=split, streaming=True)
        kept = 0
        for index, row in enumerate(stream):
            traj = build(row, index, ctx)
            if traj is None:
                continue
            problems = canonical.validate_trajectory(traj)
            if problems:
                continue
            out.append(traj)
            kept += 1
            if kept >= want:
                break
        print(f"{dataset}:{config} -> {kept} trajectories")
    return out


def run(*, root: Path, limit_per_source: int, dry_run: bool) -> int:
    trajectories = harvest(limit_per_source, dry_run)
    if dry_run:
        return 0
    path = root / "prompts" / "trajectories.jsonl"
    written = canonical.write_jsonl(path, trajectories)
    sim = sum(1 for t in trajectories if t.meta.get("simulated"))
    print(f"trajectory prompts: {written} -> {path} (simulated: {sim})")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prebuilt trajectory prompt corpus")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--limit-per-source", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(root=args.root, limit_per_source=args.limit_per_source, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 3: Smoke, then a small live harvest**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajparse
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajbuild --dry-run
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajbuild --root datasets/qwen35-4b-sft --limit-per-source 40
```
Expected: the Task 5–6 smoke lines; six `[dry]` lines naming the two Hermes configs, ToolACE, the two SmolTalk configs, and OpenHermes-2.5; then six `<dataset>:<config> -> N` lines and `trajectory prompts: N -> ...` with a non-zero `N` (a few hundred; measured 170 at `--limit-per-source 40`, and OpenHermes contributed 0 — see the note below).

> **OpenHermes-2.5 yields 0 — measured, not a parser bug.** Across the whole dataset: 2,093 roleplay rows, every one a single `human`/`gpt` exchange (user-turn histogram `{1: 2093}`), so a `>= 2 user turns` guard rejects all of them — the roleplay slice is single-turn, though spec §5.1 listed it as a multi-turn source. Not a parser bug: the source simply has no multi-turn roleplay. Roleplay trajectories therefore come from SmolTalk alone (measured 80). **Dropped from `SOURCES` and `build_openhermes` deleted** (commit `872949a`); OpenHermes is not lost — M1's `sources.py` keeps it as the single-turn roleplay source (adapter `sharegpt_openhermes`, 426 prompts kept).

- [x] **Step 4: Commit**

```powershell
git add tools/dataset/trajparse.py tools/dataset/trajbuild.py
git commit -m "feat(dataset): roleplay chat trajectories and prompt-corpus builder"
```

---

### Task 8: Multi-turn verification (spec §10.1, §11.4)

**Files:**
- Modify: `tools/dataset/verify.py`

- [x] **Step 1: Append the trajectory checks to `verify.py`**

Add after `check_record` and before `__main__`:

```python
_JSON_ERROR = re.compile(r'"error"\s*:')
# Known limitation: a legitimate tool result may carry an `"error"` key (a well-formed
# API error response is still a valid observation). Measured on ToolACE, ~1.6% of tool
# results match and are dropped as "error-laden". Left as specified; the rate shows up in
# the trajectory drops, and M3 can narrow the pattern to a top-level error field.


def check_turns(trajectory) -> list[str]:
    """Turn-level structural checks (spec section 10.2). Empty list means valid."""
    from .canonical import declared_tool_names
    problems: list[str] = []
    names = declared_tool_names(trajectory.tools) if trajectory.tools else None
    calls: dict[str, int] = {}
    answered: set[str] = set()
    for i, m in enumerate(trajectory.messages):
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if not (strip_think(str(m.get("content") or "")).strip() or tcs):
                problems.append(f"empty assistant turn at {i}")
            for tc in tcs:
                cid = tc.get("id")
                fn = tc.get("function") or {}
                if not cid:
                    problems.append(f"tool_call without id at {i}")
                    continue
                if names is not None and fn.get("name") not in names:
                    problems.append(f"undeclared tool {fn.get('name')!r} at {i}")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        problems.append(f"tool_call arguments not JSON at {i}")
                calls[cid] = i
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in calls:
                problems.append(f"tool result without a preceding call at {i}")
            else:
                answered.add(cid)
            text = str(m.get("content") or "")
            if text.strip().lower().startswith("error:") or _JSON_ERROR.search(text):
                problems.append(f"error-laden tool result at {i}")
    for cid, idx in calls.items():
        if cid not in answered:
            problems.append(f"call {cid!r} at {idx} has no tool result")
    return problems


def check_trajectory(verify_spec: dict | None, trajectory) -> tuple[bool, str]:
    """Trajectory-level check: structure first, then the final visible answer (spec 10.1).

    `think` blocks are stripped before any oracle runs, so reasoning cannot satisfy a test
    by accident (spec section 10.1).
    """
    problems = check_turns(trajectory)
    if problems:
        return False, problems[0]
    if verify_spec is not None:
        from .canonical import final_assistant_text
        return check_record(verify_spec, final_assistant_text(trajectory.messages))
    return True, ""
```

> The §10.2 difficulty filter (drop all-pass / all-fail) only applies where a prompt is rolled out more than once. M2 rolls out once, so it is not implemented here; it belongs to M3's rejection sampling.

Add `import json` to the imports at the top of `verify.py` (it currently imports `re`, `subprocess`, `sys`).

- [x] **Step 2: Extend the smoke block**

Replace the `__main__` block with:

```python
if __name__ == "__main__":
    from .canonical import Trajectory

    ok, _ = check_python_tests("def add(a, b):\n    return a + b",
                               ["assert add(1, 2) == 3"])
    bad, why_bad = check_python_tests("def add(a, b):\n    return a - b",
                                      ["assert add(1, 2) == 3"])
    io_ok, _ = check_python_io("print(sum(map(int, input().split())))", [["1 2", "3"]])
    print("tests positive:", ok, "| negative:", bad, "|", why_bad)
    print("io positive:", io_ok)
    print("answer match:", check_answer("<think>\n4\n</think>\n\n\\boxed{4}", "4"))
    print("answer mismatch:", check_answer("The answer is 5.", "4"))

    good = Trajectory(
        id="t1", domain="coding", origin="teacher", source={"name": "s"},
        messages=[
            {"role": "user", "content": "Book a table."},
            {"role": "assistant", "content": "<think>need the tool</think>",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "book", "arguments": "{\"n\": 2}"}}]},
            {"role": "tool", "content": "{\"ok\": true}", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Booked."},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}])
    orphan = Trajectory(
        id="t2", domain="coding", origin="teacher", source={"name": "s"},
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "nope", "arguments": "{bad"}}]},
            {"role": "assistant", "content": "z"},
        ],
        tools=[{"type": "function", "function": {"name": "book"}}])
    print("turns good:", check_turns(good))
    print("turns bad:", len(check_turns(orphan)), "problems")
    print("traj verify pass:", check_trajectory({"type": "answer_match", "gold": "Booked."}, good))
```

- [x] **Step 3: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.verify
```
Expected: `tests positive: True | negative: False | exit 1: AssertionError`, `io positive: True`,
`answer match: True`, `answer mismatch: False`; `turns good: []`; `turns bad: 3 problems`
(undeclared tool, non-JSON arguments, unanswered call); `traj verify pass: (True, '')`.

- [x] **Step 4: Commit**

```bash
git add tools/dataset/verify.py
git commit -m "feat(dataset): turn and trajectory verification"
```

---

### Task 9: Per-turn response filters (spec §10.1)

> **Refusal predicate removed (decided during Task 15).** Once the reasoning trace is stored inside `content` (see the Task 2 note), substring refusal matching false-fired on incidental phrases in the reasoning — measured: a coding row was rejected because its reasoning quoted the example `"I want to go but I can't because I'm busy"`. Per the user's instruction the `refusal` predicate and `REFUSAL_MARKERS` were deleted outright. The response filters now cover empty / unbalanced-think / too_short / too_long / looping / non_english only.
>
> **All response-quality predicates removed outright (decided during Task 15, after auditing every drop).** Each one false-fires on the trace-in-`content` shape: refusal markers are quoted inside the reasoning, the n-gram loop check trips on the trace echoing its own test fixtures (measured: a coding row's trace scored 0.058, driven by a run of the word `zebra` the model used as its own test input, while its answer scored 0.045), and the language check trips on legitimately non-Latin content. Measured: **2 of the first 3 seeded rows were false positives.** Decision: accept those rows outright. `filters.response_reject_reason` is now **structure only** (`empty_assistant` / `unbalanced_think_tags` / `too_short` / `too_long`); `filters.py` has no `refusal`, `looping` or `non_english` predicate, and the smoke pins all four removed classes as accepted. M3's reward filter owns response quality.
>
> **`answer_match` accepts the answer's last numeric token (Task 15, same audit).** The pool prompts carry no "end with just the answer" instruction, so a correct reasoning answer ends with prose. Measured: a GSM8K row whose final value was right ended `James made **$126** from selling all the water.`, and the last-line rule dropped it as `verify_failed`. `verify.check_answer` now also accepts the last numeric token when the gold normalises to a number, and still rejects a wrong final number (smoke covers both).

**Files:**
- Modify: `tools/dataset/filters.py`

- [x] **Step 1: Refactor `example_reject_reason` and add trajectory filtering**

In `tools/dataset/filters.py`, add `import json` to the imports at the top, then replace `example_reject_reason` with:

```python
MIN_TURN_TOKENS = 4


def _loop_ratio(text: str) -> float:
    """Degenerate-repeat ratio for one assistant span, scored on each part separately.

    The teacher's reasoning trace frequently drafts the answer verbatim before emitting
    it, so scoring the concatenation counts the draft as a repeat and flags legitimate
    rows (measured: a row whose trace and answer each scored 0.0 scored 0.070 joined,
    over the 0.05 threshold, and was dropped as `looping`).
    """
    if THINK_CLOSE in text:
        trace, answer = text.split(THINK_CLOSE, 1)
    else:
        trace, answer = "", text
    return max(textutil.repeating_ngram_ratio(trace), textutil.repeating_ngram_ratio(answer))


def response_reject_reason(text: str, tokens: int,
                           min_tokens: int = MIN_ANSWER_TOKENS) -> str | None:
    """Response-side predicates on one assistant span (M2, spec sections 10.1, 11.3)."""
    if not text.strip():
        return "empty_assistant"
    if text.count(THINK_OPEN) != text.count(THINK_CLOSE):
        return "unbalanced_think_tags"
    if tokens < min_tokens:
        return "too_short"
    if tokens > MAX_ANSWER_TOKENS:
        return "too_long"
    if _loop_ratio(text) > LOOP_RATIO_MAX:
        return "looping"
    if len(_NON_ENGLISH.findall(text)) / max(1, len(text)) > NON_ENGLISH_RATIO_MAX:
        return "non_english"
    return None


def _response_text(message: dict) -> str:
    """Assistant text for filtering. A tool-call-only turn contributes its calls, so a
    legitimate function call is not mistaken for an empty turn."""
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    else:
        text = ""
    if not text.strip() and message.get("tool_calls"):
        return json.dumps(message["tool_calls"], ensure_ascii=False)
    return text


def example_reject_reason(ex: Example) -> str | None:
    """Single-turn response predicates. M2 runs these on teacher completions."""
    for m in reversed(ex.messages):
        if m.get("role") == "assistant":
            return response_reject_reason(_response_text(m), ex.tokens)
    return "empty_assistant"


def trajectory_reject_reason(traj) -> str | None:
    """Run the response predicates on every assistant turn; reject if any turn fails.

    Chat turns are shorter than standalone answers, so the per-turn minimum is
    `MIN_TURN_TOKENS`.
    """
    for i, m in enumerate(traj.messages):
        if m.get("role") != "assistant":
            continue
        text = _response_text(m)
        reason = response_reject_reason(text, textutil.count_tokens(text),
                                        min_tokens=MIN_TURN_TOKENS)
        if reason:
            return f"turn {i}: {reason}"
    return None
```

- [x] **Step 2: Extend the smoke block**

Replace `__main__` with:

```python
if __name__ == "__main__":
    from .canonical import Example, Trajectory

    def mk(text, tokens=200):
        return Example(id="f", domain="reasoning", origin="prebuilt", source={"name": "s"},
                       messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]},
                                 {"role": "assistant", "content": text}], tokens=tokens)

    print("prompt clean:", prompt_reject_reason("Solve x^2 = 4.", 200))
    print("prompt empty:", prompt_reject_reason("", 0))
    print("clean:", example_reject_reason(mk("A clear worked answer.")))
    print("unbalanced:", example_reject_reason(mk("<think>reasoning without a close tag")))
    print("short:", example_reject_reason(mk("ok", tokens=3)))
    print("loop:", example_reject_reason(mk(" ".join(["a b c d e f g h"] * 30))))
    # Regression: the trace drafts the answer, so the joined text repeats. Scored per
    # span this is clean; scored joined it would trip `looping`.
    drafted = " ".join(f"word{i}" for i in range(20))
    print("draft-echo not looping:",
          example_reject_reason(mk(f"<think>Draft: {drafted}</think>\n\n{drafted}")))
    print("repeating answer still looping:",
          example_reject_reason(mk(f"<think>short plan</think>\n\n{' '.join(['a b c d e f g h'] * 30)}")))
    print("non-english:", example_reject_reason(mk("这是一段中文回答，用于测试语言过滤。")))

    good = Trajectory(id="t1", domain="roleplay", origin="teacher", source={"name": "s"},
                      messages=[{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "Hello there, how are you?"}])
    print("traj clean:", trajectory_reject_reason(good))

    tool_only = Trajectory(
        id="t3", domain="coding", origin="teacher", source={"name": "s"},
        messages=[{"role": "user", "content": "book it"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "c", "type": "function",
                                   "function": {"name": "book", "arguments": "{\"n\": 2}"}}]}])
    print("traj tool-only kept:", trajectory_reject_reason(tool_only))
```

- [x] **Step 3: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.filters
```
Expected: `prompt clean: None`, `prompt empty: empty_prompt`, `clean: None`,
`unbalanced: unbalanced_think_tags`, `short: too_short`, `loop: looping`,
`draft-echo not looping: None`, `repeating answer still looping: looping`,
`non-english: non_english`, then `traj clean: None` and `traj tool-only kept: None`.

- [x] **Step 4: Commit**

```bash
git add tools/dataset/filters.py
git commit -m "feat(dataset): per-turn response filters for trajectories"
```

---

### Task 10: Prebuilt-backed trajectory generation (spec §10.2)

**Files:**
- Create: `tools/dataset/trajectory.py`

Design decision (spec §10.2, §5.1): **the teacher writes every assistant turn, including its `tool_calls`; the source's observations are spliced in after the teacher's call, with the call id remapped.** This keeps M1's invariant ("no prebuilt assistant text is trained") and matches "the teacher re-writes the assistant turns … with the source's tool results spliced back in". User turns and `tool` results are prebuilt context. This reading supersedes §5.1's table row ("prebuilt turns kept; teacher answers the final assistant turn"): §10's every-response-teacher-written rule wins.

- [x] **Step 1: Write `tools/dataset/trajectory.py`**

```python
"""Multi-turn teacher generation: prebuilt-backed and simulated (spec sections 5.1, 10.2).

`_scaffold` assistant turns coming from trajparse are never sent to the teacher; they only
tell the walk how many assistant turns the source had. Every generated turn is teacher text.
"""
from __future__ import annotations

import json

from . import gencache, textutil
from .canonical import Trajectory, final_assistant_text, message_text
from .teacher import TeacherClient

MAX_TOOL_TURNS = 8
MAX_USER_TURNS = 8
SIM_USER_SYSTEM = (
    "You are the user in a tool-use conversation. Write the user's next message: a natural, "
    "specific request that one of the available tools can satisfy. Reply with only the message.")
SIM_TOOL_SYSTEM = (
    "You are the tool environment. Given a tool call, reply with only a short JSON object a "
    "plausible API would return. No prose and no code fences.")


def _gen(client, cache, key: str, messages: list[dict], **kw) -> dict:
    """One cached teacher turn. Cached value: {'completion': {...}}."""
    cached = cache.get(key)
    if cached is None:
        msg = client.complete(messages, **kw)
        cached = {"completion": {"content": msg.get("content") or "",
                                 "tool_calls": msg.get("tool_calls") or []}}
        cache.put(key, cached)
    return cached["completion"]


def _remap(calls: list[dict], traj_id: str, start: int) -> list[dict]:
    out = []
    for k, tc in enumerate(calls):
        fn = tc.get("function") or {}
        out.append({"id": f"{traj_id}-call-{start + k}", "type": "function",
                    "function": {"name": fn.get("name", ""),
                                 "arguments": fn.get("arguments", "{}")}})
    return out


def _truncate(messages: list[dict], max_user: int, max_tool: int) -> list[dict]:
    """Cut at the first turn that breaks a bound, then back up to an assistant turn."""
    users = tools = 0
    cut = len(messages)
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            users += 1
        elif m.get("role") == "tool":
            tools += 1
        if users > max_user or tools > max_tool:
            cut = i
            break
    trimmed = [dict(m) for m in messages[:cut]]
    while trimmed and trimmed[-1].get("role") != "assistant":
        trimmed.pop()
    return trimmed


def _source_run(messages: list[dict], i: int) -> list[dict]:
    """The source `tool` messages immediately following index `i`."""
    out, j = [], i + 1
    while j < len(messages) and messages[j].get("role") == "tool":
        out.append(messages[j])
        j += 1
    return out


def generate_prebuilt(prompt_traj: Trajectory, client: TeacherClient, cache,
                      *, store=None, thinking: bool = True) -> Trajectory | None:
    """Regenerate every assistant turn; splice the source's observations after teacher calls."""
    source = _truncate(prompt_traj.messages, MAX_USER_TURNS, MAX_TOOL_TURNS)
    new: list[dict] = []
    calls_made = 0
    i = 0
    while i < len(source):
        m = source[i]
        if m.get("role") != "assistant":
            new.append({"role": m["role"], "content": m.get("content", ""),
                        **({"tool_call_id": m["tool_call_id"]} if m.get("tool_call_id") else {})})
            i += 1
            continue

        key = gencache.prefix_key(new, kind=f"prebuilt:{prompt_traj.id}")
        comp = _gen(client, cache, key, new, store=store, tools=prompt_traj.tools,
                    thinking=thinking)
        calls = comp.get("tool_calls") or []
        run = _source_run(source, i)
        assistant = {"role": "assistant", "content": comp["content"]}
        if calls:
            if not run:
                return None  # a call with no observation to splice in
            assistant["tool_calls"] = _remap(calls, prompt_traj.id, calls_made)
            calls_made += len(assistant["tool_calls"])
            new.append(assistant)
            for call, src_tool in zip(assistant["tool_calls"], run):
                new.append({"role": "tool", "content": src_tool.get("content", ""),
                            "tool_call_id": call["id"]})
        else:
            new.append(assistant)
        i += 1 + len(run)

    while new and new[-1].get("role") == "tool":
        new.pop()
    if not new or new[-1].get("role") != "assistant":
        return None
    text = "\n".join(message_text(m) for m in new if m.get("role") == "assistant")
    return Trajectory(
        id=prompt_traj.id, domain=prompt_traj.domain, origin="teacher",
        source=dict(prompt_traj.source), messages=new, tools=prompt_traj.tools,
        images=prompt_traj.images, tokens=textutil.count_tokens(text),
        verify=prompt_traj.verify, meta={**prompt_traj.meta, "simulated": False})


if __name__ == "__main__":
    from .gencache import GenCache
    from .canonical import validate_trajectory
    import tempfile
    from pathlib import Path

    class FakeTeacher:
        def __init__(self, script):
            self.script = list(script)
            self.sent = []

        def complete(self, messages, **kw):
            self.sent.append(messages)
            return self.script.pop(0)

    src = Trajectory(
        id="toolace-1", domain="coding", origin="prebuilt", source={"name": "smoke"},
        messages=[
            {"role": "user", "content": "What time is it in Tokyo?"},
            {"role": "assistant", "content": "", "_scaffold": True,
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": "clock", "arguments": "{\"city\": \"Tokyo\"}"}}]},
            {"role": "tool", "content": '{"time": "04:00"}', "tool_call_id": "call_0"},
            {"role": "assistant", "content": "", "_scaffold": True},
        ],
        tools=[{"type": "function", "function": {"name": "clock"}}])

    fake = FakeTeacher([
        {"content": "<think>look up the time</think>",
         "tool_calls": [{"id": "x", "type": "function",
                         "function": {"name": "clock", "arguments": "{\"city\": \"Tokyo\"}"}}]},
        {"content": "It is 04:00 in Tokyo.", "tool_calls": []},
    ])
    cache = GenCache(Path(tempfile.mkdtemp()) / "c.jsonl")
    out = generate_prebuilt(src, fake, cache)
    print("roles:", [m["role"] for m in out.messages])
    print("tool id:", out.messages[2]["tool_call_id"], "| problems:", validate_trajectory(out))
    print("teacher never saw the scaffold:", all(
        not any(m.get("_scaffold") for m in msgs) for msgs in fake.sent))

    # Render the generated trajectory through the student template. This is the check that
    # catches the string-vs-mapping `arguments` mismatch: the template iterates
    # `tool_call.arguments|items`, so a stored JSON string raises a TemplateError here.
    # Skips cleanly when the template has not been pinned yet.
    template_path = Path(__file__).parent / "student.jinja"
    if template_path.exists():
        from . import canonical
        from .render import render
        try:
            text = render(template_path.read_text(encoding="utf-8"),
                          canonical.for_template(out.messages))
            print("renders through student.jinja: True |",
                  "<function=clock>" in text)
        except Exception as exc:
            print("renders through student.jinja: False |", type(exc).__name__, exc)
    else:
        print("renders through student.jinja: skipped (no student.jinja)")
```

- [x] **Step 2: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajectory
```
Expected: `roles: ['user', 'assistant', 'tool', 'assistant']`,
`tool id: toolace-1-call-0 | problems: []`, `teacher never saw the scaffold: True`, then
`renders through student.jinja: True | True` (skipped with a `skipped` line if
`student.jinja` has not been pinned yet — M1 Task 16 writes it, so it should be present).

- [x] **Step 3: Commit**

```bash
git add tools/dataset/trajectory.py
git commit -m "feat(dataset): prebuilt-backed multi-turn generation"
```

---

### Task 11: Simulated trajectory generation (spec §10.2)

**Files:**
- Modify: `tools/dataset/trajectory.py`

The teacher plays three roles: user, agent, and tool-environment. Seeded from a real first user turn plus a `tools` schema. Bounds: 8 tool turns, 8 user turns. A trajectory that never reaches a final assistant turn is dropped.

- [x] **Step 1: Append the simulated loop to `trajectory.py`**

Add before `__main__`:

```python
def generate_simulated(client: TeacherClient, cache, *, id: str, domain: str, source: dict,
                       tools: list, first_user: str | None = None,
                       max_tool_turns: int = MAX_TOOL_TURNS, thinking: bool = True,
                       store=None) -> Trajectory | None:
    """End-to-end simulated trajectory: teacher is user, agent, and tool environment."""
    messages: list[dict] = []
    if first_user is None:
        key = gencache.prefix_key([{"role": "system", "content": SIM_USER_SYSTEM}],
                                  kind=f"simuser:{id}")
        first_user = _gen(client, cache, key,
                          [{"role": "system", "content": SIM_USER_SYSTEM},
                           {"role": "user", "content": "Start a request that uses a tool."}],
                          thinking=False, max_tokens=256)["content"].strip()
    if not first_user:
        return None
    messages.append({"role": "user", "content": first_user})

    for _ in range(max_tool_turns):
        key = gencache.prefix_key(messages, kind=f"simagent:{id}")
        comp = _gen(client, cache, key, messages, store=store, tools=tools,
                    thinking=thinking, max_tokens=1024)
        calls = comp.get("tool_calls") or []
        assistant = {"role": "assistant", "content": comp["content"]}
        if not calls:
            messages.append(assistant)
            break
        assistant["tool_calls"] = calls
        messages.append(assistant)
        for call in calls:
            tkey = gencache.prefix_key(
                [{"role": "system", "content": SIM_TOOL_SYSTEM},
                 {"role": "user", "content": json.dumps(call.get("function") or {})}],
                kind=f"simtool:{id}")
            result = _gen(client, cache, tkey,
                          [{"role": "system", "content": SIM_TOOL_SYSTEM},
                           {"role": "user", "content": json.dumps(call.get("function") or {})}],
                          thinking=False, max_tokens=256)["content"].strip()
            messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
    else:
        return None  # hit the turn budget without a final answer: not answer-bearing

    if not messages or messages[-1].get("role") != "assistant":
        return None
    text = "\n".join(message_text(m) for m in messages if m.get("role") == "assistant")
    return Trajectory(
        id=id, domain=domain, origin="teacher", source=dict(source), messages=messages,
        tools=tools, tokens=textutil.count_tokens(text),
        meta={"simulated": True})
```

Extend the `__main__` smoke with a simulated run:

```python
    sim_fake = FakeTeacher([
        {"content": "<think>find the city</think>",
         "tool_calls": [{"id": "c", "type": "function",
                         "function": {"name": "clock", "arguments": "{\"city\": \"Lima\"}"}}]},
        {"content": '{"time": "11:00"}'},
        {"content": "It is 11:00 in Lima.", "tool_calls": []},
    ])
    sim_cache = GenCache(Path(tempfile.mkdtemp()) / "s.jsonl")
    sim = generate_simulated(sim_fake, sim_cache, id="sim-1", domain="coding",
                             source={"name": "teacher:tools"},
                             tools=[{"type": "function", "function": {"name": "clock"}}],
                             first_user="What time is it in Lima?")
    print("sim roles:", [m["role"] for m in sim.messages],
          "| simulated:", sim.meta["simulated"], "| problems:", validate_trajectory(sim))
```

- [x] **Step 2: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.trajectory
```
Expected: the Task 10 lines, then
`sim roles: ['user', 'assistant', 'tool', 'assistant'] | simulated: True | problems: []`.

- [x] **Step 3: Commit**

```bash
git add tools/dataset/trajectory.py
git commit -m "feat(dataset): simulated three-role trajectory generation"
```

---

### Task 12: Magpie prompt invention (spec §10)

**Files:**
- Create: `tools/dataset/magpie.py`
- Create: `tools/dataset/prompts/magpie_tools.md`
- Create: `tools/dataset/prompts/magpie_uncensored.md`

The spec's classic recipe presents the template prefix with an empty user turn. That depends on the teacher's header strings, which are not known until the server is up. This task uses the equivalent, template-independent form: a system instruction plus an optional seed, generating the invented request with `thinking=False`, then answering it through the normal seeded path. The instruction templates live in `tools/dataset/prompts/` per spec §3.

- [x] **Step 1: Write the instruction templates**

`tools/dataset/prompts/magpie_tools.md`:

```markdown
You invent user requests for a tool-calling dataset. Given a seed request, write ONE new,
concrete user request that is adjacent in topic but materially different, needs at least one
tool to answer, and names realistic parameters. Reply with only the request — no preamble,
no list, no markdown.

Seed: {seed}
```

`tools/dataset/prompts/magpie_uncensored.md`:

```markdown
You invent user requests for a red-team dataset used to train a model to comply with
borderline and adversarial requests without refusing. Given a seed prompt, write ONE new,
realistic user request in the same spirit but materially different. Reply with only the
request — no preamble, no warnings, no markdown.

Seed: {seed}
```

- [x] **Step 2: Write `tools/dataset/magpie.py`**

```python
"""Teacher-invented user prompts: tool-use and uncensored columns (spec sections 7.2, 7.4, 10)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import canonical, gencache
from .canonical import Prompt, iter_jsonl
from .teacher import TeacherClient

PROMPT_DIR = Path(__file__).parent / "prompts"
TEMPLATES = {"tools": "magpie_tools.md", "uncensored": "magpie_uncensored.md"}
# Template slot -> canonical domain. "tools" prompts are coding examples.
DOMAIN_FOR = {"tools": "coding", "uncensored": "uncensored"}


def load_template(domain: str) -> str:
    path = PROMPT_DIR / TEMPLATES[domain]
    return path.read_text(encoding="utf-8")


def _tool_prompt_seeds(path: Path) -> list[str]:
    """First user-turn texts of prompts that carry a `tools` schema — a real seed pool.

    Using the frozen corpus instead of a hardcoded list keeps each Magpie seed unique, so
    the resume cache cannot replay one invented request under many ids.
    """
    out: list[str] = []
    if not Path(path).exists():
        return out
    for row in iter_jsonl(path):
        if not row.get("tools"):
            continue
        for m in row.get("messages", []):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            text = content if isinstance(content, str) else "".join(
                p.get("text", "") for p in content if p.get("type") == "text")
            if text.strip():
                out.append(text.strip())
            break
    return out


def invent(client: TeacherClient, cache, *, domain: str, seed: str,
           thinking: bool = False, max_tokens: int = 384) -> str | None:
    template = load_template(domain)
    instruction = template.replace("{seed}", seed)
    key = gencache.request_key({"magpie": domain, "seed": seed, "v": 1})
    cached = cache.get(key)
    if cached is None:
        msg = client.complete(
            [{"role": "system", "content": "You are a dataset generator."},
             {"role": "user", "content": instruction}],
            thinking=thinking, max_tokens=max_tokens)
        cached = {"completion": {"content": (msg.get("content") or "").strip()}}
        cache.put(key, cached)
    text = cached["completion"]["content"].splitlines()[0].strip() if cached[
        "completion"]["content"] else ""
    return text or None


def build(domain: str, seeds: list[str], client: TeacherClient, cache, *,
          limit: int, source_name: str) -> list[Prompt]:
    out: list[Prompt] = []
    for seed in seeds:
        if len(out) >= limit:
            break
        text = invent(client, cache, domain=domain, seed=seed)
        if not text:
            continue
        out.append(Prompt(
            id=f"magpie-{domain}-{len(out)}", domain=DOMAIN_FOR[domain], origin="teacher",
            source={"name": source_name, "license": None},
            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
            meta={"magpie": True, "seed": seed[:200]}))
    return out


def run(*, root: Path, client: TeacherClient, cache, limit: int, dry_run: bool) -> int:
    if dry_run:
        for domain in TEMPLATES:
            print(f"[dry] magpie {domain}: limit={limit}")
        return 0
    seed_rows = [r for r in iter_jsonl(root / "seeds" / "uncensored.jsonl")] \
        if (root / "seeds" / "uncensored.jsonl").exists() else []
    uncensored_seeds = [r["text"] for r in seed_rows[:limit]]
    tool_seeds = _tool_prompt_seeds(root / "prompts" / "train.jsonl")[:limit]
    prompts = build("uncensored", uncensored_seeds, client, cache, limit=limit,
                    source_name="teacher:uncensored")
    prompts += build("tools", tool_seeds, client, cache, limit=limit,
                     source_name="teacher:tools")
    path = root / "prompts" / "magpie.jsonl"
    written = canonical.write_jsonl(path, prompts)
    print(f"magpie prompts: {written} -> {path}")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    from .gencache import GenCache
    ap = argparse.ArgumentParser(description="Magpie prompt invention")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8086")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cache_path = args.cache or (args.root / "m2" / "cache.jsonl")
    return run(root=args.root, client=TeacherClient(args.base_url),
               cache=GenCache(cache_path), limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 3: Replace the smoke dispatch with the fake-client check**

Replace the trailing block of `magpie.py` (the `if __name__ == "__main__": sys.exit(main())` from Step 2) with:

```python
if __name__ == "__main__":
    if len(sys.argv) == 1:
        class FakeTeacher:
            def complete(self, messages, **kw):
                return {"content": "Plan a three-day trip to Kyoto using the train tool."}

        import tempfile
        cache = gencache.GenCache(Path(tempfile.mkdtemp()) / "c.jsonl")
        prompt = build("tools", ["Book a train"], FakeTeacher(), cache,
                       limit=1, source_name="teacher:tools")
        print("template:", load_template("tools").splitlines()[0])
        print("invented:", prompt[0].messages[0]["content"][0]["text"])
        print("domain:", prompt[0].domain)
    else:
        sys.exit(main())
```

- [x] **Step 4: Smoke it**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.magpie
```
Expected: `template: You invent user requests for a tool-calling dataset. Given a seed request, write ONE new,` then
`invented: Plan a three-day trip to Kyoto using the train tool.`, `domain: coding`.

- [x] **Step 5: Commit**

```bash
git add tools/dataset/magpie.py tools/dataset/prompts/
git commit -m "feat(dataset): Magpie prompt invention"
```

---

### Task 13: Multi-turn render gate (spec §11.2)

**Files:**
- Modify: `tools/dataset/render.py`

The single-turn gate already asserts the think block survives and the generation prompt opens one. This task adds the multi-turn assertions: which per-turn think blocks survive rendering (Qwen3 keeps only the latest user-delimited segment), whether the template carries `{% generation %}` markers (or relies on TRL's Qwen3 auto-patch), and that `tool_calls` arguments round-trip without double-escaping.

- [x] **Step 1: Add the multi-turn check to `render.py`**

Add near the top:

```python
import re
```

`render.py` already imports `from .canonical import iter_jsonl`; add the package import so
the render-time argument conversion is reachable, and keep `iter_jsonl` (the single-turn
body calls it unqualified):

```python
from . import canonical
from .canonical import iter_jsonl
```

Add after `strip_image_data`:

```python
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def tool_calls_in(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            out.extend(m.get("tool_calls") or [])
    return out


def check_multiturn(template: str, rows: list[dict], sample: int = 50) -> int:
    """Multi-turn render gate: think survival, mask markers, tool round-trip."""
    has_markers = "{% generation %}" in template and "{% endgeneration %}" in template
    problems: list[str] = []
    think_stored = think_kept = 0
    for row in rows[:sample]:
        messages = strip_image_data(row["messages"])
        stored = sum(m["content"].count(THINK_OPEN) for m in messages
                     if m["role"] == "assistant" and isinstance(m.get("content"), str))
        try:
            # The template iterates `tool_call.arguments|items`, which needs a mapping;
            # stored arguments are JSON strings. `for_template` is that one conversion.
            text = render(template, canonical.for_template(messages))
        except TemplateError as exc:
            problems.append(f"{row['id']}: {exc}")
            continue
        kept = len(_THINK_BLOCK.findall(text))
        think_stored += stored
        think_kept += kept
        # The template fabricates a `<think>` block for every assistant turn after the
        # last user turn (student.jinja splits a stored block out of `content` and re-emits
        # it in its own delimiters). So the expected count is those turns, NOT the number
        # of stored tag pairs: a content-less scaffold legitimately renders more than it
        # stores. More than one block per qualifying turn is the real failure.
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"),
                        default=-1)
        expected = sum(1 for i, m in enumerate(messages)
                       if m.get("role") == "assistant" and i > last_user)
        if kept > expected:
            problems.append(
                f"{row['id']}: rendered {kept} think blocks, expected {expected}")
        for call in tool_calls_in(messages):
            args = canonical.tool_call_arguments(call)
            name = (call.get("function") or {}).get("name")
            if name and f"<function={name}>" not in text:
                problems.append(f"{row['id']}: tool name {name!r} did not render")
            for key, value in args.items():
                # The template writes `<parameter=key>\nvalue\n</parameter>`; check the
                # parameter survived rather than the raw JSON string, which it cannot.
                if f"<parameter={key}>" not in text:
                    problems.append(f"{row['id']}: parameter {key!r} did not render")
    print(f"generation mask markers present: {has_markers}")
    print(f"think blocks stored={think_stored} kept={think_kept} (Qwen3 keeps the latest segment)")
    print(f"multi-turn problems: {len(problems)}")
    for p in problems[:5]:
        print("  ", p)
    return 1 if problems else 0
```

- [x] **Step 2: Add a `--multiturn` flag to the CLI**

In `render.py`, change the existing `run` signature (at M1 HEAD it is `run(*, dataset, template, sample)`) to `template_path` and add the `multiturn` flag, inserting the branch immediately after the rows are loaded. The rename is deliberate: `template_path` says it is a path, and `main` is updated in the same step to match, so the two cannot drift:

```python
def run(*, dataset: Path, template_path: Path, sample: int,
        multiturn: bool = False) -> int:
    template_text = template_path.read_text(encoding="utf-8")
    rows = list(iter_jsonl(dataset))[:sample]
    if multiturn:
        return check_multiturn(template_text, rows, sample=sample)
    failures = 0
    # ... the rest of the existing single-turn body is unchanged ...
```

and in `main`, add the flag and pass the renamed keyword through:

```python
    ap.add_argument("--multiturn", action="store_true",
                    help="trajectory gates: think survival, mask markers, tool round-trip")
    args = ap.parse_args(argv)
    return run(dataset=args.dataset, template_path=args.template, sample=args.sample,
               multiturn=args.multiturn)
```

> The rename touches **two** places, not one: the signature, and `Path(template).read_text(...)` in the body, which becomes `Path(template_path)`. The local variable is `template_text`; `main` is updated in the same step so the two cannot drift.

- [x] **Step 3: Run the gate on the trajectory corpus**

Requires the teacher/student template file. Use the teacher's template for the teacher-side gate and the student's for the student-side; the student template is the binding one for training.

```powershell
Invoke-RestMethod http://127.0.0.1:8085/props | Select-Object -ExpandProperty chat_template | Set-Content tools/dataset/student.jinja -Encoding utf8
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft/prompts/trajectories.jsonl --template tools/dataset/student.jinja --multiturn --sample 20
```
Expected: `generation mask markers present: False` (Qwen3.5 relies on TRL's auto-patch unless the GGUF template carries markers; record whichever prints), `think blocks stored=0 kept=M` — **`kept > stored` is expected here**: prompt-side trajectories carry content-less `_scaffold` assistant turns, and the template fabricates one `<think>` block per assistant turn after the last user turn, so `stored` counts nothing while `kept` counts those turns. The check fails only if a turn renders *more* blocks than the template can emit (`kept > expected`). `multi-turn problems: 0`, exit 0. Measured: `stored=0 kept=224` over all 170 rows, 0 problems.

- [x] **Step 4: Commit**

```bash
git add tools/dataset/render.py
git commit -m "feat(dataset): multi-turn render gate"
```

---

### Task 14: M2 driver and pass-rate manifest (spec §10.1, §12)

**Files:**
- Create: `tools/dataset/generate.py`
- Modify: `tools/dataset/report.py`

`generate.py` orchestrates the three generation shapes and a merge step. Each shape writes its own shard (`m2/single.jsonl`, `m2/val.jsonl`, `m2/trajectory.jsonl`) plus a `*.stats.json`; `merge` stitches them into `train.jsonl` / `val.jsonl` and folds pass rates into `manifest.json`.

- [x] **Step 1: Append `merge_manifest` to `report.py`**

Add before `__main__` (and add `from . import canonical` is not needed; use plain dicts):

```python
def merge_manifest(path: Path, *, train: list, val: list, m2: dict) -> dict:
    """Read the M1 manifest, add the M2 pass-rate block, and refresh the split summaries."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest["m2"] = m2
    manifest["train_final"] = summarise(train, "train")
    manifest["val_final"] = summarise(val, "val")
    return manifest
```

- [x] **Step 2: Write `tools/dataset/generate.py`**

```python
"""M2 driver: teacher generation and verification over every M1 prompt column.

`all` runs magpie -> seeded (single-turn) -> trajectory (prebuilt) -> simulated -> merge.
Only verified, filtered rows are written; drops and pass rates are counted per source and
per domain so M3 can re-anchor its caps on measured numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import canonical, filters, gencache, split, textutil, verify
from .canonical import Example, iter_jsonl, prompt_from_dict, trajectory_from_dict
from .gencache import GenCache
from .imgstore import ImageStore
from .report import merge_manifest, write_manifest
from .teacher import TeacherClient, TeacherError
from . import magpie, trajectory

TRAIN_POOLS = ["prompts/train.jsonl", "verification/seeds.jsonl", "prompts/magpie.jsonl"]
VAL_POOLS = ["prompts/val.jsonl"]


def _seed(text: str) -> int:
    return int(text[:8], 16)


def _source_key(record) -> str:
    return f"{record.source.get('name', '?')}|{record.domain}"


def generate_one(prompt, client, cache, *, store, thinking: bool) -> dict:
    """One cached single-turn completion, verified and filtered. Cached value is the verdict."""
    key = gencache.prefix_key(prompt.messages, kind=f"single:{prompt.id}")
    cached = cache.get(key)
    if cached is not None:
        return cached
    msg = client.complete(prompt.messages, store=store, tools=prompt.tools,
                          thinking=thinking, seed=_seed(key))
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    if prompt.verify:
        ok, why = verify.check_record(prompt.verify, content)
        if not ok:
            cached = {"accepted": False, "reason": "verify_failed", "detail": why}
            cache.put(key, cached)
            return cached
    assistant = {"role": "assistant", "content": content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    tokens = textutil.count_tokens(content) + (
        textutil.count_tokens(json.dumps(tool_calls, ensure_ascii=False)) if tool_calls else 0)
    ex = Example(id=prompt.id, domain=prompt.domain, origin="teacher",
                 source=prompt.source, messages=prompt.messages + [assistant],
                 images=prompt.images, tools=prompt.tools,
                 tokens=tokens, meta=dict(prompt.meta))
    reason = filters.example_reject_reason(ex)
    cached = {"accepted": reason is None, "reason": reason, "example": ex.to_dict()}
    cache.put(key, cached)
    return cached


def _run_pool(pool: Path, limit: int | None, client, cache, *, store, thinking, stats):
    records = [prompt_from_dict(r) for r in iter_jsonl(pool)]
    if limit:
        records = split.downsample(records, limit)
    out = []
    for prompt in records:
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            verdict = generate_one(prompt, client, cache, store=store, thinking=thinking)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}")
            continue
        if verdict["accepted"]:
            out.append(canonical.example_from_dict(verdict["example"]))
            stats["by_domain"][prompt.domain] += 1
            stats["accepted"] += 1
            stats["accepted_by_source"][key] += 1
        else:
            stats["drops"][verdict["reason"]] += 1
    return out


def run_seeded(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
               thinking: bool = True):
    store = ImageStore(root / "images")
    stats = {"attempted": 0, "accepted": 0, "by_domain": Counter(),
             "attempted_by_source": Counter(), "accepted_by_source": Counter(),
             "drops": Counter()}
    train: list[Example] = []
    for rel in TRAIN_POOLS:
        path = root / rel
        if path.exists():
            train.extend(_run_pool(path, limit, client, cache, store=store,
                                   thinking=thinking, stats=stats))
    val: list[Example] = []
    for rel in VAL_POOLS:
        path = root / rel
        if path.exists():
            val.extend(_run_pool(path, val_limit, client, cache, store=store,
                                 thinking=thinking, stats=stats))
    canonical.write_jsonl(root / "m2" / "single.jsonl", train)
    canonical.write_jsonl(root / "m2" / "val.jsonl", val)
    _dump_stats(root / "m2" / "seeded.stats.json", stats)
    print(f"seeded: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def _accept_trajectory(traj, stats) -> bool:
    """Validate, verify, and filter one trajectory; record the drop reason."""
    problems = canonical.validate_trajectory(traj)
    if problems:
        stats["drops"]["invalid"] += 1
        return False
    ok, _ = verify.check_trajectory(traj.verify, traj)
    if not ok:
        stats["drops"]["verify_failed"] += 1
        return False
    reason = filters.trajectory_reject_reason(traj)
    if reason:
        stats["drops"][reason.split(":")[-1].strip()] += 1
        return False
    return True


def _traj_stats() -> dict:
    return {"attempted": 0, "accepted": 0, "simulated": 0, "by_domain": Counter(),
            "attempted_by_source": Counter(), "accepted_by_source": Counter(),
            "drops": Counter()}


def run_trajectory(*, root: Path, client, cache, limit: int | None, thinking: bool = True):
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    records = [trajectory_from_dict(r) for r in iter_jsonl(path)] if path.exists() else []
    if limit:
        records = split.downsample(records, limit)
    out = []
    for prompt in records:
        key = _source_key(prompt)
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            traj = trajectory.generate_prebuilt(prompt, client, cache, store=store,
                                                thinking=thinking)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {prompt.id}: {exc}")
            continue
        if traj is None:
            stats["drops"]["turn_structure"] += 1
            continue
        if not _accept_trajectory(traj, stats):
            continue
        out.append(traj)
        stats["accepted"] += 1
        stats["by_domain"][prompt.domain] += 1
        stats["accepted_by_source"][key] += 1
    canonical.write_jsonl(root / "m2" / "trajectory.jsonl", out)
    _dump_stats(root / "m2" / "trajectory.stats.json", stats)
    print(f"trajectories: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def run_simulated(*, root: Path, client, cache, limit: int | None, thinking: bool = True):
    """Simulated trajectories (spec section 5.1): the teacher plays user, agent, and tool
    environment, seeded from the real first user turns and tool schemas of the prebuilt
    trajectory prompts. Rows carry `meta.simulated = true`."""
    store = ImageStore(root / "images")
    stats = _traj_stats()
    path = root / "prompts" / "trajectories.jsonl"
    seeds = [t for t in (trajectory_from_dict(r) for r in iter_jsonl(path))
             if t.tools] if path.exists() else []
    if limit:
        seeds = split.downsample(seeds, limit)
    out = []
    for seed in seeds:
        first_user = next((canonical.message_text(m) for m in seed.messages
                           if m.get("role") == "user"), "")
        if not first_user:
            continue
        key = f"teacher:simulated|{seed.domain}"
        stats["attempted"] += 1
        stats["attempted_by_source"][key] += 1
        try:
            traj = trajectory.generate_simulated(
                client, cache, id=f"sim-{seed.id}", domain=seed.domain,
                source={"name": "teacher:simulated"}, tools=seed.tools,
                first_user=first_user, thinking=thinking, store=store)
        except TeacherError as exc:
            stats["drops"]["teacher_error"] += 1
            print(f"  teacher error on {seed.id}: {exc}")
            continue
        if traj is None:
            stats["drops"]["turn_structure"] += 1
            continue
        if not _accept_trajectory(traj, stats):
            continue
        out.append(traj)
        stats["accepted"] += 1
        stats["by_domain"][seed.domain] += 1
        stats["accepted_by_source"][key] += 1
        stats["simulated"] += 1
    canonical.write_jsonl(root / "m2" / "simulated.jsonl", out)
    _dump_stats(root / "m2" / "simulated.stats.json", stats)
    print(f"simulated: accepted {stats['accepted']} of {stats['attempted']}")
    return stats


def _dump_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    att = serial.get("attempted_by_source", {})
    acc = serial.get("accepted_by_source", {})
    serial["pass_rate_by_source"] = {
        k: round(acc.get(k, 0) / n, 4) for k, n in att.items() if n}
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def run_merge(*, root: Path) -> dict:
    # Training rows are Examples: `example_from_dict` is the right loader for all three
    # shards, including the trajectory ones. A `Trajectory`'s `verify` spec is a
    # generation-time filter (`check_trajectory` already ran in `_accept_trajectory`) and
    # is deliberately not carried into the written row — `Example.to_dict` has no
    # `verify` field. If a downstream stage ever needs to re-run an oracle, it must read
    # the originating prompt, not train.jsonl.
    def load(rel):
        p = root / "m2" / rel
        return [canonical.example_from_dict(r) for r in iter_jsonl(p)] if p.exists() else []

    train = load("single.jsonl") + load("trajectory.jsonl") + load("simulated.jsonl")
    val = load("val.jsonl")
    canonical.write_jsonl(root / "train.jsonl", train)
    canonical.write_jsonl(root / "val.jsonl", val)

    def stats(rel):
        p = root / "m2" / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    m2 = {"seeded": stats("seeded.stats.json"), "trajectory": stats("trajectory.stats.json"),
          "simulated": stats("simulated.stats.json"),
          "train_written": len(train), "val_written": len(val)}
    shards = ("seeded", "trajectory", "simulated")
    passed = sum(m2[k].get("accepted", 0) for k in shards)
    attempted = sum(m2[k].get("attempted", 0) for k in shards)
    m2["pass_rate"] = round(passed / attempted, 4) if attempted else 0.0
    manifest = merge_manifest(root / "manifest.json", train=train, val=val, m2=m2)
    write_manifest(root / "manifest.json", manifest)
    print(f"merged: train {len(train)}, val {len(val)}, pass_rate {m2['pass_rate']}")
    return m2


def run_all(*, root: Path, client, cache, limit: int | None, val_limit: int | None,
            multi_limit: int | None, sim_limit: int | None, magpie_limit: int,
            thinking: bool, dry_run: bool):
    if dry_run:
        print("[dry] magpie:", magpie_limit)
        print("[dry] trajectories:", multi_limit)
        print("[dry] simulated:", sim_limit)
        print("[dry] single-turn:", limit, "val:", val_limit)
        return
    magpie.run(root=root, client=client, cache=cache, limit=magpie_limit, dry_run=False)
    run_seeded(root=root, client=client, cache=cache, limit=limit, val_limit=val_limit,
               thinking=thinking)
    run_trajectory(root=root, client=client, cache=cache, limit=multi_limit, thinking=thinking)
    run_simulated(root=root, client=client, cache=cache, limit=sim_limit, thinking=thinking)
    run_merge(root=root)


def smoke() -> int:
    """Offline end-to-end: one prompt, a fake teacher, a temp root. No network."""
    import tempfile

    from .canonical import Prompt, validate

    tmp = Path(tempfile.mkdtemp())
    (tmp / "prompts").mkdir(parents=True, exist_ok=True)
    canonical.write_jsonl(tmp / "prompts" / "train.jsonl", [Prompt(
        id="p1", domain="reasoning", origin="prebuilt", source={"name": "smoke"},
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}])])

    class FakeTeacher:
        def complete(self, messages, **kw):
            return {"content": "Four is the sum of two and two. Working it through: two plus "
                               "two equals four, so the answer is four."}

    cache = GenCache(tmp / "m2" / "cache.jsonl")
    stats = run_seeded(root=tmp, client=FakeTeacher(), cache=cache, limit=10,
                       val_limit=0, thinking=True)
    kept = canonical.iter_jsonl(tmp / "m2" / "single.jsonl")
    first = next(kept)
    print("smoke accepted:", stats["accepted"],
          "| valid:", validate(canonical.example_from_dict(first)) == [])
    return 0 if stats["accepted"] == 1 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 teacher generation")
    ap.add_argument("--mode", choices=["all", "merge"], default="all")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8086")
    ap.add_argument("--model", default="local-teacher")
    ap.add_argument("--limit", type=int, default=300,
                    help="single-turn attempts PER POOL (three pools: "
                         "prompts/train.jsonl, verification/seeds.jsonl, prompts/magpie.jsonl)")
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--multi-limit", type=int, default=100)
    ap.add_argument("--sim-limit", type=int, default=40)
    ap.add_argument("--magpie-limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # Spec section 10 fixes enable_thinking: true; the non-thinking roles (Magpie, the
    # simulated user, the simulated tool environment) hardcode thinking=False internally.
    cache = GenCache(args.root / "m2" / "cache.jsonl")
    client = TeacherClient(args.base_url, model=args.model)
    if args.dry_run:
        run_all(root=args.root, client=client, cache=cache, limit=args.limit,
                val_limit=args.val_limit, multi_limit=args.multi_limit,
                sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
                thinking=True, dry_run=True)
        return 0
    if args.mode == "merge":
        run_merge(root=args.root)
        return 0
    run_all(root=args.root, client=client, cache=cache, limit=args.limit,
            val_limit=args.val_limit, multi_limit=args.multi_limit,
            sim_limit=args.sim_limit, magpie_limit=args.magpie_limit,
            thinking=True, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(smoke() if len(sys.argv) == 1 else main())
```

- [x] **Step 3: Dry run (no network)**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --dry-run
```
Expected: `[dry] magpie: 20`, `[dry] trajectories: 100`, `[dry] simulated: 40`,
`[dry] single-turn: 300 val: 100`, no exception.

- [x] **Step 4: Offline smoke of the driver with a fake teacher**

Run (from the repo root):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate
```
Expected: `seeded: accepted 1 of 1` then `smoke accepted: 1 | valid: True`. This exercises the
full single-turn path — cache, verify, filters, JSONL write — with no server.

- [x] **Step 5: Commit**

```bash
git add tools/dataset/generate.py tools/dataset/report.py
git commit -m "feat(dataset): M2 driver and pass-rate manifest"
```

---

### Task 15: Full M2 run, calibration export, acceptance

**Files:**
- Modify: `datasets/qwen35-4b-sft/README.md` (created in M1 Task 16 Step 7)

- [ ] **Step 1: Confirm the teacher is up and the template is current**

```powershell
Invoke-RestMethod http://127.0.0.1:8086/health
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.teacher http://127.0.0.1:8086
```
Expected: `status ok`; then the Task 2 smoke lines.

- [ ] **Step 2: Run M2 end to end**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.generate --mode all --root datasets/qwen35-4b-sft --limit 300 --val-limit 100 --multi-limit 100 --sim-limit 40 --magpie-limit 20
```
Expected: `magpie prompts: 40 -> ...`, `seeded: accepted N of M`, `trajectories: accepted N of M`, `simulated: accepted N of M`, `merged: train X, val Y, pass_rate 0.xx`, with `X >= 400`. Because every call is cached, a re-run prints the same numbers.

> **The local M2 run is a pipeline smoke, not the corpus (decided during Task 15).** The teacher emits a long reasoning trace per call (~4k tokens, measured ~5 min/call on the local iGPU), so a full local run is tens of hours for no added signal: the point locally is to prove every shape, verifier, filter, merge and manifest step fires end-to-end. The acceptance run therefore uses `--limit 5 --val-limit 2 --multi-limit 3 --sim-limit 2 --magpie-limit 3` (≈40 calls). The ~25k corpus is M3 on Colab vLLM, so Task 15's `X >= 400` expectation is a scale-out target, not a local gate. Separately, `TeacherClient.complete` no longer sends `max_tokens` — a 4096 cap truncated rows mid-reasoning with no answer at all. The cache was cleared before this run.

Two scale notes the flags understate:

- `--limit` is **per pool**, so `--limit 300` attempts up to **900** single-turn rows (300 each from `prompts/train.jsonl`, `verification/seeds.jsonl`, `prompts/magpie.jsonl`). `M` in the seeded line is ~900 + 100 val, not 300.
- `verification/seeds.jsonl` is **100% coding and 100% oracle-bearing** (1,001 rows, all `python_tests`/`python_io`) and has no domain mix, so its 300 attempts land entirely in `coding` and pass or fail on executed code. That drags the seeded pass rate below the `prompts/train.jsonl` rate and skews `train_final.by_domain` toward coding; read `pass_rate_by_source` rather than the single blended `pass_rate` when setting M3 caps.

- [ ] **Step 3: Read the pass rates and drop reasons**

```powershell
$m = Get-Content datasets/qwen35-4b-sft/manifest.json | ConvertFrom-Json
$m.m2.pass_rate
$m.m2.seeded.pass_rate_by_source; $m.m2.trajectory.pass_rate_by_source; $m.m2.simulated.pass_rate_by_source
$m.m2.seeded.drops; $m.m2.trajectory.drops; $m.m2.simulated.drops
$m.train_final.by_domain; $m.train_final.token_share
```
Expected: a `pass_rate` between 0 and 1; a `pass_rate_by_source` map keyed by `dataset|domain` (the M3 cap guide, spec §10.1); `drops` keys limited to legitimate reasons (`verify_failed`, `looping`, `too_short`, `too_long`, `non_english`, `unbalanced_think_tags`, `empty_assistant`, `turn_structure`, `invalid`, `teacher_error`); `train_final.by_domain` covering all four domains; non-zero token share.

- [ ] **Step 4: Verify every accepted record and its image refs**

Run (from the repo root; one line):
```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -c "from tools.dataset import canonical; from tools.dataset.imgstore import ImageStore; from pathlib import Path; root=Path('datasets/qwen35-4b-sft'); store=ImageStore(root/'images'); rows=[canonical.example_from_dict(r) for rel in ('train.jsonl','val.jsonl') for r in canonical.iter_jsonl(root/rel)]; bad=[(rel, ex.id) for rel in ('train.jsonl','val.jsonl') for ex in [canonical.example_from_dict(r) for r in canonical.iter_jsonl(root/rel)] if canonical.validate(ex) or any(store.resolve(i.sha256) is None for i in ex.images)]; print('checked', len(rows), 'bad', len(bad), bad[:3])"
```
Expected: `checked N bad 0` — 100% schema-valid and every image sha resolves (spec §11.3).

- [ ] **Step 5: Export calibration chunks (now that answers exist)**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.calibrate --dataset datasets/qwen35-4b-sft/train.jsonl --out datasets/qwen35-4b-sft/calibration.txt --chunks 20
```
Expected: `calibration chunks written: N -> ...` with `N > 0`. This is Phase 3's `llama-imatrix` input (spec §3).

- [ ] **Step 6: Re-run the render gate on real records and the contamination guard**

```powershell
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.render --dataset datasets/qwen35-4b-sft/train.jsonl --template tools/dataset/student.jinja --sample 200
& "$HOME\miniconda3\envs\dataset\python.exe" -m tools.dataset.contaminate --dataset datasets/qwen35-4b-sft/train.jsonl --extra-eval datasets/qwen35-4b-sft/eval/refusal.jsonl
```
Expected: `0 template failures`, `probe think block survived: True`, `generation prompt opens thinking: True`, a `VERDICT:` line; then a `collisions:` count. Expect **0 exact** collisions and a small non-zero **NEAR** count: the in-the-wild jailbreak corpus supplies both `seeds/uncensored.jsonl` prompts and the quarantined refusal slice, so spec §8's intended in-distribution overlap shows as NEAR, not as a leak. A non-zero **exact** count is the failure.

- [ ] **Step 7: Update the README with the M2 commands and measured numbers**

Append to `datasets/qwen35-4b-sft/README.md` a `## M2 — teacher generation` section containing: the teacher launch command (Task 2 Step 2), the `generate.py --mode all` command with the flags actually used, the measured `pass_rate` and per-domain token share, the drop-reason table, the calibrate command, and a note that `prompts/train.jsonl` + `verification/seeds.jsonl` are M2's inputs while `train.jsonl` / `val.jsonl` are its outputs. Note the Teacher model divergence: spec §10 names `mradermacher/Ornith-1.5-9B-Abliterated-i1-GGUF`; the local run used `MiMo-Ornith-9B-AGSI-Abliterated-HQ.i1-Q4_K_S.gguf` + `.mmproj-BF16.gguf`.

- [ ] **Step 8: Commit**

```bash
git add datasets/qwen35-4b-sft/README.md
git commit -m "docs(dataset): M2 rebuild instructions and measured pass rates"
```

---

## M3 handoff — what M2 leaves for the scale-out

M2's measured numbers set M3's caps. M3 is the same CLIs with `--base-url` pointing at a vLLM-served teacher and larger targets, plus the full Magpie tool-schema generator:

1. **Inputs:** `prompts/train.jsonl`, `prompts/val.jsonl`, `prompts/magpie.jsonl`, `prompts/trajectories.jsonl`, `verification/seeds.jsonl`, `seeds/uncensored.jsonl`.
2. **Scale:** `--limit 20000 --multi-limit 6000 --sim-limit 3000 --val-limit 1000` (spec §7 candidates), teacher on Colab vLLM.
3. **Caps:** re-anchor `sources.TEACHER_SOURCES` on M2's `pass_rate_by_source`; a domain at 0.7 pass rate needs 1.43× candidates for the same accepted count (spec §10.1). The §10.2 difficulty filter (drop all-pass / all-fail) arrives here, where prompts are rolled out more than once.
4. **Residual risk carried forward:** roleplay, chat-style coding, and uncensored have no oracle and inherit teacher bugs; sample-audit those columns before scaling (spec §10.1, §14).
5. **Judge:** only if a roleplay/multi-turn quality signal is wanted; M2 deliberately did not build one.

---

## Self-review

**Spec coverage.** §5/§5.1/§5.2 schema: Task 4 (tool role, `Trajectory`, validators). §6 ingestion: Tasks 5–7. §7.1–7.7 sources and the verification corpus: Tasks 1 (golds), 5–7 (trajectories), 8 (oracles), 12 (Magpie). §10 teacher, sampling, resume, **image canary**, Magpie: Tasks 2, 3, 12. §10.1 verification before training (per-source and per-domain pass rates): Tasks 8, 9, 14, 15. §10.2 multi-turn — prebuilt-backed and simulated, bounds, answer-bearing drop: Tasks 4–7, 10, 11, 14. §11.2 multi-turn render: Task 13. §11.3 validator + image resolution: Task 15 Step 4. §11.4 verifier controls: Task 8. §11.5 distribution/pass rates: Tasks 14–15. §12 M2 milestone and calibration: Task 15. Deliberately out of scope: §10.2's difficulty filter (needs multi-rollout; M3 rejection sampling), §11.6–11.7 behavioural deltas and human spot-check (Phase 2A and review), M3 scale-out, RL/DPO.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code and shows the exact lines to change; every run step has an expected output. Server-dependent checks are confined to Task 2 Steps 2–3 (text + image canary), Task 13 Step 3 (render gate) and Task 15; every other smoke injects a `FakeTeacher` and runs offline.

**Type consistency.** `TeacherClient.complete`, `GenCache.get/put`, `prefix_key`, `canonical.Trajectory`/`validate_trajectory`/`message_text`/`final_assistant_text`/`declared_tool_names`/`tool_calls_of`/`tool_call_arguments`/`for_template`, `prompt_from_dict`/`trajectory_from_dict`/`example_from_dict`, `verify.check_record`/`check_turns`/`check_trajectory`, `filters.response_reject_reason`/`_response_text`/`example_reject_reason`/`trajectory_reject_reason`, `trajectory.generate_prebuilt`/`generate_simulated`, `magpie.invent`/`build`/`DOMAIN_FOR`, `generate.generate_one`/`_run_pool`/`run_seeded`/`run_trajectory`/`run_simulated`/`run_merge`/`run_all`, `report.merge_manifest` are each defined once and referenced with the same signature. `tool_call.arguments` is a **JSON string** everywhere it is stored (`trajparse.py`, `trajectory.py`, the OpenAI wire format) and a **mapping** only at the render boundary, via `canonical.for_template`, which is the single conversion point. `THINK_OPEN`/`THINK_CLOSE` are the real pair in Task 1 and used consistently.

**Known risk.** The student template iterates `tool_call.arguments|items`, which requires a mapping, while canonical records store a JSON string. `canonical.for_template` is the one conversion point, applied in `check_multiturn` and exercised by Task 10's smoke; if a future template variant expects the string form instead, that helper is where the change goes. Task 13 Step 3 may report no `{% generation %}` markers; that is a finding, not a failure — it means TRL's Qwen3 auto-patch supplies the mask, and the gate records it. If the tool-parameter round-trip fails, the student template transforms `tool_calls` and the multi-turn records must be reshaped before M3; stop and reconcile rather than scaling.

**Committed:** this plan document is tracked at `docs/superpowers/plans/`. Implementation fixes found during M2 are synced into the affected tasks (Task 1, Task 7, Task 13).
