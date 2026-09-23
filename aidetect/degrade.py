"""What happens to an image between the generator and the person looking at it.

Almost no image arrives untouched: it is re-encoded by a messenger, resized to
fit a timeline, or photographed off a screen. Detectors that read compression
statistics or high-frequency residue lose most of their signal to exactly that
trip -- published figures put NPR at 41.9% -> 0.2% fake-accuracy under JPEG
q50 and SAFE at 63.0% -> 0.0% -- and the failure is asymmetric. Real-accuracy
rises towards 100% while fake-accuracy falls to zero, so a detector run at a
fixed threshold quietly turns into one that answers "real" to everything, with
confidence scores that still look reasonable. These functions exist to make
that visible.

Everything here is a pure function of the input pixels and the parameters:
same image in, same bytes out, no wall-clock or global RNG state, so scores
cached under a degradation name stay valid across runs.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .imageio import to_rgb

# 4:2:0 chroma subsampling is what phones, browsers and every social platform
# write. Encoding the benchmark at 4:4:4 would understate the damage.
JPEG_SUBSAMPLING = "4:2:0"


def _roundtrip(img: Image.Image, fmt: str, **params) -> Image.Image:
    """Encode and decode again.

    The artefacts that break a detector are in the *decoded* pixels, not in the
    file, so a degradation that only writes bytes has done nothing. The decode
    is forced here while the buffer is still alive.
    """
    buf = io.BytesIO()
    to_rgb(img).save(buf, format=fmt, **params)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return to_rgb(out)


def jpeg(img: Image.Image, quality: int = 75) -> Image.Image:
    return _roundtrip(img, "JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING)


def webp(img: Image.Image, quality: int = 80) -> Image.Image:
    # method=4 is Pillow's default effort level; pinning it keeps the output
    # identical if that default ever changes.
    return _roundtrip(img, "WEBP", quality=int(quality), method=4)


def double_jpeg(img: Image.Image, q1: int = 85, q2: int = 60) -> Image.Image:
    """Saved, re-opened, re-saved -- what a re-shared image has been through.

    Re-quantising an already-quantised image on the same 8x8 grid leaves the
    double-quantisation comb in the DCT coefficient histograms. That is a
    fingerprint of the sharing, not of the generator, and it sits in precisely
    the statistics the frequency-domain detectors read.
    """
    return jpeg(jpeg(img, q1), q2)


def _resize(img: Image.Image, size: tuple[int, int], resample) -> Image.Image:
    return img.resize((max(1, size[0]), max(1, size[1])), resample)


def downscale(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """Shrink both sides by `scale` (0.5 = half). Lanczos, as a decent pipeline uses."""
    if not 0 < scale <= 1:
        raise ValueError(f"downscale expects 0 < scale <= 1, got {scale}")
    w, h = img.size
    return _resize(img, (round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def downscale_upscale(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """Shrink then restore the original pixel dimensions.

    The "resized for the web" case, and the more instructive of the two: the
    image still has its original size, so nothing downstream can tell from the
    dimensions that the high frequencies -- where the generator fingerprint
    lives -- were thrown away. Bicubic on the way back up because that is what
    browsers and apps use to fit an image to a container.
    """
    w, h = img.size
    return _resize(downscale(img, scale), (w, h), Image.Resampling.BICUBIC)


def gaussian_blur(img: Image.Image, sigma: float = 1.0) -> Image.Image:
    # Pillow's GaussianBlur `radius` is the standard deviation, not a kernel
    # half-width, despite the name.
    return to_rgb(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def gaussian_noise(img: Image.Image, sigma: float = 2.0, seed: int = 0) -> Image.Image:
    """Additive white noise, sigma in 8-bit levels."""
    arr = np.asarray(to_rgb(img), dtype=np.float32)
    # Seed from the pixels so the result is a function of the image alone: the
    # same file gets the same noise whatever order or batch it is scored in,
    # while different files still get independent noise.
    mixed = (zlib.crc32(arr.tobytes()) ^ (int(seed) * 0x9E3779B1)) & 0xFFFFFFFF
    rng = np.random.default_rng(mixed)
    noisy = arr + rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    # rint, not a bare cast: truncation would darken every pixel by half a
    # level and add a DC shift that has nothing to do with the noise.
    return Image.fromarray(np.rint(np.clip(noisy, 0, 255)).astype(np.uint8), mode="RGB")


def screenshot(img: Image.Image, scale: float = 0.6, blur_sigma: float = 0.5,
               brightness: float = 1.06, contrast: float = 0.94,
               quality: int = 80) -> Image.Image:
    """A screenshot-and-reshare chain: downscale, slight blur, tone shift, JPEG.

    The parameters are a defensible middle, not a measurement -- there is no
    recapture corpus on this disk to fit them to:

      scale 0.6    a screenshot is taken at display resolution, which is
                   usually below the image's own.
      sigma 0.5    the display's and the capture path's resampling are never
                   pixel-exact; half a pixel of blur is the mildest defocus
                   that is still visible in a residual.
      1.06 / 0.94  screens are brighter and lower-contrast than the source,
                   and any auto-exposure between them compounds it.
      q80          what phone screenshot and share pipelines write.

    What it does NOT model: moire against the screen's pixel grid, the
    screen-door pattern, perspective, glare, or camera sensor noise. It is a
    clean digital recapture, so it is an upper bound on photo-of-screen
    performance -- the literature's ~0.55 accuracy for real recapture is
    worse than anything this chain will produce.
    """
    out = downscale(img, scale)
    out = gaussian_blur(out, blur_sigma)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return jpeg(out, quality)


def identity(img: Image.Image) -> Image.Image:
    return img


@dataclass(frozen=True)
class Degradation:
    """A named, parameter-bound degradation."""

    name: str
    apply: Callable[[Image.Image], Image.Image]
    note: str = ""

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.apply(img)


def _reg(name: str, fn: Callable[..., Image.Image], note: str, **kw) -> Degradation:
    return Degradation(name=name, apply=partial(fn, **kw) if kw else fn, note=note)


REGISTRY: dict[str, Degradation] = {d.name: d for d in [
    _reg("clean", identity, "the benchmark image as built, for the reference numbers"),
    _reg("jpeg90", jpeg, "light re-encode, the best case for a re-shared image", quality=90),
    _reg("jpeg70", jpeg, "typical web upload", quality=70),
    _reg("jpeg50", jpeg, "the level at which the published frequency detectors collapse", quality=50),
    _reg("jpeg30", jpeg, "aggressive; visibly blocky", quality=30),
    _reg("webp80", webp, "what most platforms now re-encode to", quality=80),
    _reg("webp50", webp, "aggressive WebP", quality=50),
    _reg("downscale_50", downscale, "half size, as delivered to a feed", scale=0.5),
    _reg("downscale_25", downscale, "thumbnail size", scale=0.25),
    _reg("rescale_50", downscale_upscale,
         "halved and restored: original dimensions, high frequencies gone", scale=0.5),
    _reg("rescale_25", downscale_upscale, "quarter-size round trip", scale=0.25),
    _reg("double_jpeg_85_60", double_jpeg, "saved, re-opened, re-saved", q1=85, q2=60),
    _reg("screenshot", screenshot, "digital recapture chain; an upper bound on photo-of-screen"),
    _reg("noise_2", gaussian_noise, "mild sensor-grade noise", sigma=2.0),
    _reg("noise_5", gaussian_noise, "heavy noise", sigma=5.0),
    _reg("blur_1", gaussian_blur, "slight defocus", sigma=1.0),
    _reg("blur_2", gaussian_blur, "clearly soft", sigma=2.0),
]}

# The cross-product with backends and protocols is expensive, so the default
# sweep keeps one parameter per failure mode plus the JPEG ladder, which is
# the axis the literature reports and the one that breaks detectors hardest.
DEFAULT_SUITE: tuple[str, ...] = (
    "clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "webp50",
    "downscale_50", "rescale_50", "double_jpeg_85_60", "screenshot",
    "noise_2", "blur_1",
)


def get(name: str) -> Degradation:
    if name not in REGISTRY:
        raise KeyError(f"unknown degradation {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def apply_chain(img: Image.Image, names) -> Image.Image:
    """Compose degradations by name, left to right."""
    for n in names:
        img = get(n)(img)
    return img
