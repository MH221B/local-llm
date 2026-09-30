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
