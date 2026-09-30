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
