"""Render a stratified sample of the finished corpus for human review.

Spec sections 10.1 and 11.7: roleplay and uncensored have no mechanical oracle at all, so the
substitute is a measured sample audit; coding is mixed, with the seed corpus carrying a
`verify` spec that already filtered it and the chat-style rows readable only by a human. This
module is that artefact: it writes one Markdown file with N examples per domain, and reports
how many of them came from the two fully un-oracled columns so the audit's coverage is
explicit rather than implied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import canonical

ORACLE_LESS = ("roleplay", "uncensored")


def sample(records: list, per_domain: int, seed: int = 17) -> dict[str, list]:
    import random

    rnd = random.Random(seed)
    by_domain: dict[str, list] = {}
    for ex in records:
        by_domain.setdefault(ex.domain, []).append(ex)
    out = {}
    for domain, items in by_domain.items():
        items = items[:]
        rnd.shuffle(items)
        out[domain] = items[:per_domain]
    return out


def render_markdown(records: dict[str, list], total: int) -> str:
    lines = ["# M3 sample audit", "",
             f"Corpus: {total} examples. Sampled uniformly at random per domain.",
             "Read each response and record a verdict in the table at the end.", ""]
    for domain in sorted(records):
        lines += [f"## {domain}", ""]
        for i, ex in enumerate(records[domain], start=1):
            lines.append(f"### {domain} {i} — `{ex.id}`")
            lines.append("")
            for m in ex.messages:
                role = m.get("role")
                content = m.get("content")
                text = content if isinstance(content, str) else "".join(
                    p.get("text", "") for p in content if p.get("type") == "text")
                if m.get("images") or (isinstance(content, list)
                                       and any(p.get("type") == "image" for p in content)):
                    text = f"[image] {text}"
                lines.append(f"**{role}:** {text}")
                lines.append("")
                if m.get("tool_calls"):
                    lines.append(f"**tool_calls:** `{m['tool_calls']}`")
                    lines.append("")
            lines += [f"- verdict: <!-- ok | fix | drop -->", ""]
    ours = sum(len(v) for k, v in records.items() if k in ORACLE_LESS)
    lines += ["## Coverage", "",
              f"- examples drawn from un-oracled columns: {ours}",
              f"- `roleplay` and `uncensored` have no oracle at all, so every row here is a",
              f"  judgement call rather than a re-check",
              f"- `coding` is mixed: the seed corpus carries a `verify` spec and was filtered by",
              f"  it, while the chat-style rows are read here like the other two",
              f"- nothing in this file overrides `verify.py`; it is a human signal, not a gate", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3 sample audit")
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/qwen35-4b-sft/train.jsonl"))
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path("datasets/qwen35-4b-sft/audit.md"))
    args = ap.parse_args(argv)
    records = [canonical.example_from_dict(r)
               for r in canonical.iter_jsonl(args.dataset)]
    picked = sample(records, args.per_domain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(picked, len(records)), encoding="utf-8")
    print("audit written:", args.out, "|",
          {d: len(v) for d, v in sorted(picked.items())})
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from .canonical import Example
        rows = [Example(id=f"r{i}", domain="roleplay", origin="teacher",
                        source={"name": "s"},
                        messages=[{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": f"answer {i}"}])
                for i in range(4)]
        rows[0].messages[1]["tool_calls"] = [{"id": "c", "type": "function",
                                              "function": {"name": "t",
                                                           "arguments": "{}"}}]
        picked = sample(rows, 2)
        md = render_markdown(picked, len(rows))
        print("sampled:", {d: len(v) for d, v in picked.items()})
        print("has tool_calls line:", "**tool_calls:**" in md)
        print("has verdict slot:", "- verdict:" in md)
        print("coverage line:", [line for line in md.splitlines()
                                 if "un-oracled" in line])
    else:
        sys.exit(main())
