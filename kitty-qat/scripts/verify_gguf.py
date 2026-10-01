"""Assert 2: the converted GGUF is 33 blocks with the 15 nextn tensors.

The exact command that ran is itself a deliverable (the sibling project's one
gap); its output goes verbatim into bench/qat-results.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--expect-tensors", type=int, default=441)
    args = parser.parse_args()

    from gguf import GGUFReader

    reader = GGUFReader(args.gguf)
    names = [t.name for t in reader.tensors]
    blocks = sorted({int(n.split(".")[1]) for n in names
                     if n.startswith("blk.") and n.split(".")[1].isdigit()})
    nextn = [n for n in names if "nextn" in n]
    print("tensors:", len(names))
    print(f"block range: {blocks[0]} .. {blocks[-1]} ({len(blocks)} blocks)")
    print("nextn tensors:", len(nextn), nextn[:5])
    assert len(names) == args.expect_tensors, len(names)
    assert len(blocks) == 33, blocks
    assert len(nextn) == 15, len(nextn)
    print("assert 2 ok: block_count == 33, 15 nextn tensors present")


if __name__ == "__main__":
    main()
