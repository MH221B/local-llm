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
