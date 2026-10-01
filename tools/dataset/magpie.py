"""Teacher-invented user prompts: tool-use and uncensored columns (spec sections 7.2, 7.4, 10)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
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
           thinking: bool = False) -> str | None:
    template = load_template(domain)
    instruction = template.replace("{seed}", seed)
    key = gencache.request_key({"magpie": domain, "seed": seed, "v": 1})
    cached = cache.get(key)
    if cached is None:
        msg = client.complete(
            [{"role": "system", "content": "You are a dataset generator."},
             {"role": "user", "content": instruction}],
            thinking=thinking)
        cached = {"completion": {"content": (msg.get("content") or "").strip()}}
        cache.put(key, cached)
    text = cached["completion"]["content"].splitlines()[0].strip() if cached[
        "completion"]["content"] else ""
    return text or None


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


def _prompt_id(domain: str, text: str) -> str:
    """Content-derived id, so the same shipped text always yields the same id.

    Hashing the *invented text* rather than the seed makes an id identify the row that
    actually ships, and keeps it stable across a `--concurrency` change and across re-runs.
    A length-derived id (`magpie-{domain}-{len(out)}`) is wrong once invention is
    concurrent: two different requests can receive the same id.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"magpie-{domain}-{digest}"


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


def main(argv: list[str] | None = None) -> int:
    from .gencache import GenCache
    ap = argparse.ArgumentParser(description="Magpie prompt invention")
    ap.add_argument("--root", type=Path, default=Path("datasets/qwen35-4b-sft"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8086")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="inventions per seeded domain; 0 leaves the existing pool alone")
    ap.add_argument("--api-key", default=None, help="Bearer token for the teacher endpoint")
    ap.add_argument("--schema-limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cache_path = args.cache or (args.root / "m2" / "cache.jsonl")
    return run(root=args.root, client=TeacherClient(args.base_url, api_key=args.api_key),
               cache=GenCache(cache_path), limit=args.limit,
               schema_limit=args.schema_limit, dry_run=args.dry_run)


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
    else:
        sys.exit(main())
