"""Exact and near-duplicate rejection, plus image-sha collapse."""
from __future__ import annotations

from .canonical import Example, Prompt, image_shas
from . import textutil


class Deduper:
    """Check-then-commit so rows rejected by later stages never enter the index."""

    def __init__(self, threshold: float = 0.85, bands: int = 16):
        self.threshold = threshold
        self.bands = bands
        self._hashes: set[str] = set()
        self._images: set[str] = set()
        self._band_index: dict[tuple[int, ...], list[tuple[int, ...]]] = {}

    def check_prompt(self, prompt: str) -> str | None:
        """Return a rejection reason without mutating the index."""
        if textutil.prompt_hash(prompt) in self._hashes:
            return "duplicate_prompt"
        sig = textutil.minhash(prompt)
        for band in textutil.banded(sig, self.bands):
            for other in self._band_index.get(band, ()):
                if textutil.jaccard_est(sig, other) >= self.threshold:
                    return "near_duplicate"
        return None

    def commit_prompt(self, prompt: str) -> None:
        """Record a prompt the pipeline accepted (call only after filters and validate)."""
        sig = textutil.minhash(prompt)
        self._hashes.add(textutil.prompt_hash(prompt))
        for band in textutil.banded(sig, self.bands):
            self._band_index.setdefault(band, []).append(sig)

    def reject_reason(self, ex: Example, prompt: str) -> str | None:
        """`check_prompt` + `commit_prompt`, for standalone use and smoke checks."""
        reason = self.check_prompt(prompt)
        if reason is None:
            self.commit_prompt(prompt)
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
