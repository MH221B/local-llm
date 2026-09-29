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
    """Fraction of word n-grams that are repeats. High values mean degenerate looping.

    Bounded in [0, 1]: repeats over the *total* n-gram count, not the distinct count.
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    repeats = sum(c - 1 for c in grams.values() if c > 1)
    total = sum(grams.values())
    return repeats / total


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
