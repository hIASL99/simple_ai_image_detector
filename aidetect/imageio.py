"""Robust image loading.

A forensic detector is only as good as its decoder: silent failures here
(wrong orientation, a resize that smooths away the evidence, a truncated
file that decodes to grey) turn into silent accuracy loss downstream.
Everything that touches pixels goes through this module.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

# Left at Pillow's default (False) on purpose. With it globally True, load()
# never raises on a truncated file -- it silently pads the undecodable tail with
# flat grey -- so there is no way left to tell that it happened. open_image()
# instead decodes strictly first and only then retries permissively, recording
# the fact. The flag is process-global and not thread-safe, which is why it is
# toggled inside one narrow block rather than set once at import.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Pillow's default bomb guard is ~89 MPix. Generative output is routinely
# larger than that, so raise it, but keep a hard ceiling.
Image.MAX_IMAGE_PIXELS = 512_000_000

SUPPORTED_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".gif", ".ppm", ".pgm", ".avif", ".heic", ".heif",
}


class ImageLoadError(Exception):
    """Raised when a file cannot be decoded as a still image."""


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image plus the provenance of the decode itself."""

    image: Image.Image
    path: Path | None
    original_mode: str
    original_size: tuple[int, int]
    format: str | None
    n_frames: int
    exif_orientation: int | None
    truncated: bool

    @property
    def is_animated(self) -> bool:
        return self.n_frames > 1


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite transparency onto white.

    Converting RGBA straight to RGB discards alpha and leaves whatever garbage
    the encoder happened to store in fully transparent pixels -- which is a
    strong, entirely spurious signal for a forensic model.
    """
    background = Image.new("RGB", img.size, (255, 255, 255))
    rgba = img if img.mode == "RGBA" else img.convert("RGBA")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def to_rgb(img: Image.Image) -> Image.Image:
    """Normalise any Pillow mode to 8-bit RGB without inventing signal."""
    mode = img.mode
    if mode == "RGB":
        return img
    if mode in ("RGBA", "LA", "PA"):
        return _flatten_alpha(img)
    if mode == "La":
        # Premultiplied-alpha greyscale. Pillow refuses to convert it to
        # anything, so undo the premultiplication by hand rather than letting
        # a ValueError escape as an unhandled crash.
        arr = np.asarray(img).astype(np.float64)
        grey, alpha = arr[..., 0], np.clip(arr[..., 1], 1, 255) / 255.0
        straight = np.clip(grey / alpha, 0, 255).astype(np.uint8)
        rgba = np.dstack([straight, straight, straight, arr[..., 1].astype(np.uint8)])
        return _flatten_alpha(Image.fromarray(rgba, mode="RGBA"))
    if mode == "P":
        converted = img.convert("RGBA")
        return _flatten_alpha(converted) if "transparency" in img.info else converted.convert("RGB")
    if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # 16-bit/float: scale by the observed range, not a fixed /256, so that
        # low-dynamic-range scientific images do not collapse to black.
        arr = np.asarray(img).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr) if hi <= lo else (arr - lo) * (255.0 / (hi - lo))
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    return img.convert("RGB")


def open_image(source: str | os.PathLike | bytes | io.IOBase) -> LoadedImage:
    """Decode `source` into RGB, applying EXIF orientation.

    Accepts a path, raw bytes, or a file object.
    """
    path: Path | None = None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.exists():
            raise ImageLoadError(f"no such file: {path}")
        if path.is_dir():
            raise ImageLoadError(f"is a directory: {path}")
        if path.stat().st_size == 0:
            raise ImageLoadError(f"empty file: {path}")
        blob = path.read_bytes()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        blob = bytes(source)
    else:
        blob = source.read()  # type: ignore[union-attr]
    if not blob:
        raise ImageLoadError("empty input")

    # Keeping the bytes lets a truncated file be reopened for the second,
    # permissive decode; a consumed stream could not be rewound reliably.
    def _open() -> Image.Image:
        return Image.open(io.BytesIO(blob))

    # Pillow reports the BytesIO wrapper rather than the file, which is useless
    # in a batch run, so name the source ourselves.
    where = str(path) if path is not None else f"{len(blob)} bytes"
    try:
        img = _open()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types
        raise ImageLoadError(f"cannot decode {where} as an image "
                             f"({type(exc).__name__})") from exc

    fmt = img.format
    original_mode = img.mode
    original_size = img.size
    n_frames = int(getattr(img, "n_frames", 1) or 1)

    orientation: int | None = None
    try:
        exif = img.getexif()
        raw = exif.get(0x0112)
        orientation = int(raw) if raw is not None else None
    except Exception:  # noqa: BLE001 - malformed EXIF is common and non-fatal
        orientation = None

    truncated = False
    try:
        img.load()
    except OSError:
        # Strict decode failed: the file is short. Retry permissively so the
        # caller still gets pixels, but remember that the tail was invented.
        truncated = True
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = _open()
            img.load()
        except Exception as exc:  # noqa: BLE001
            raise ImageLoadError(f"cannot decode pixels of {where}: {exc}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"cannot decode pixels of {where}: {exc}") from exc

    if img.size[0] < 1 or img.size[1] < 1:
        raise ImageLoadError(f"degenerate image size: {img.size}")

    rgb = to_rgb(img)
    # transpose() reads orientation from the *source* image's EXIF, so it must
    # be given an image that still carries it.
    try:
        rgb.info = getattr(img, "info", {})
        rgb = ImageOps.exif_transpose(rgb) or rgb
    except Exception:  # noqa: BLE001
        pass

    return LoadedImage(
        image=rgb,
        path=path,
        original_mode=original_mode,
        original_size=original_size,
        format=fmt,
        n_frames=n_frames,
        exif_orientation=orientation,
        truncated=truncated,
    )


def load_rgb(source: str | os.PathLike | bytes | io.IOBase) -> Image.Image:
    """Convenience wrapper returning just the RGB image."""
    return open_image(source).image


def iter_image_files(root: str | os.PathLike, recursive: bool = True):
    """Yield image-looking files under `root` in a stable order."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p
