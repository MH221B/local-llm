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
