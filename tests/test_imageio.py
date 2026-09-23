"""Decoding has to be right before anything downstream can be.

A forensic detector reads pixel statistics, so a loader that quietly changes
them -- dropping alpha, skipping EXIF rotation, padding a truncated file --
degrades accuracy without ever raising.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from aidetect.imageio import ImageLoadError, open_image, to_rgb
from conftest import encode


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "La", "L", "P", "PA",
                                  "CMYK", "I;16", "F", "1", "YCbCr", "HSV"])
def test_every_mode_reaches_rgb(mode):
    img = Image.new(mode, (16, 12))
    out = to_rgb(img)
    assert out.mode == "RGB"
    assert out.size == (16, 12)


def test_transparent_pixels_are_composited_not_dropped():
    """Fully transparent pixels hold arbitrary colour that must not survive.

    Converting RGBA straight to RGB keeps whatever the encoder left under the
    transparency, which is a strong and entirely spurious signal.
    """
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., 0] = 255          # loud red hiding under the alpha
    arr[..., 3] = 0            # fully transparent
    out = np.asarray(to_rgb(Image.fromarray(arr, "RGBA")))
    assert (out == 255).all(), "transparent region should composite to white"


def test_palette_transparency_is_honoured():
    img = Image.new("P", (8, 8))
    img.info["transparency"] = 0
    assert to_rgb(img).mode == "RGB"


def test_sixteen_bit_uses_the_observed_range():
    """A fixed /256 would crush a low-dynamic-range 16-bit image to black."""
    arr = np.full((8, 8), 1000, dtype=np.uint16)
    arr[0, 0] = 2000
    out = np.asarray(to_rgb(Image.fromarray(arr, mode="I;16")))
    assert out.max() == 255 and out.min() == 0


def test_exif_orientation_actually_moves_pixels():
    arr = np.zeros((20, 10, 3), dtype=np.uint8)
    arr[0, :, 0] = 255                      # red stripe along the top edge
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6                        # rotate 90 CW on display
    blob = encode(img, "JPEG", exif=exif.tobytes(), quality=95)

    loaded = open_image(blob)
    assert loaded.exif_orientation == 6
    assert loaded.image.size == (20, 10), "orientation 6 must transpose the axes"


def test_truncated_file_is_flagged_not_hidden():
    rng = np.random.default_rng(1)
    full = encode(Image.fromarray((rng.random((128, 128, 3)) * 255).astype("uint8")),
                  "JPEG", quality=90)
    loaded = open_image(full[: len(full) // 2])
    assert loaded.truncated, "callers must be able to see that pixels were invented"


def test_animated_inputs_report_their_frame_count():
    # Frames must actually differ, or the encoder collapses them into one.
    frames = [Image.new("RGB", (8, 8), (i * 80, 0, 0)) for i in range(3)]
    blob = io.BytesIO()
    frames[0].save(blob, "GIF", save_all=True, append_images=frames[1:])
    loaded = open_image(blob.getvalue())
    assert loaded.is_animated and loaded.n_frames == 3


@pytest.mark.parametrize("blob,reason", [
    (b"", "empty"),
    (b"not an image at all, just text", "garbage"),
    (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", "a PDF"),
])
def test_undecodable_input_raises_our_error(blob, reason):
    with pytest.raises(ImageLoadError):
        open_image(blob)


def test_missing_path_and_directory_raise(tmp_path):
    with pytest.raises(ImageLoadError):
        open_image(tmp_path / "nope.jpg")
    with pytest.raises(ImageLoadError):
        open_image(tmp_path)


def test_zero_byte_file_raises(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        open_image(p)


def test_one_pixel_image_is_fine():
    assert open_image(encode(Image.new("RGB", (1, 1)))).image.size == (1, 1)


def test_bytes_and_path_agree(tmp_path, rgb_image):
    blob = encode(rgb_image, "PNG")
    p = tmp_path / "x.png"
    p.write_bytes(blob)
    assert np.array_equal(np.asarray(open_image(p).image),
                          np.asarray(open_image(blob).image))
