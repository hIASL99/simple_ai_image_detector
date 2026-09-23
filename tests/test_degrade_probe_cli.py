"""Degradations, the linear probe, and the command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from aidetect import degrade
from aidetect.probe import LinearProbe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def photo():
    rng = np.random.default_rng(3)
    return Image.fromarray((rng.random((128, 96, 3)) * 255).astype("uint8"))


@pytest.mark.parametrize("name", sorted(degrade.REGISTRY))
def test_every_degradation_returns_a_usable_image(name, photo):
    out = degrade.REGISTRY[name].apply(photo)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert min(out.size) > 0


@pytest.mark.parametrize("name", sorted(degrade.REGISTRY))
def test_degradations_are_deterministic(name, photo):
    """Scores are cached per degradation name, so the same name must mean the
    same pixels forever -- including the ones that add noise."""
    a = np.asarray(degrade.REGISTRY[name].apply(photo))
    b = np.asarray(degrade.REGISTRY[name].apply(photo))
    assert np.array_equal(a, b)


def test_clean_is_an_identity(photo):
    assert np.array_equal(np.asarray(degrade.REGISTRY["clean"].apply(photo)),
                          np.asarray(photo.convert("RGB")))


def test_rescale_preserves_size_but_destroys_detail(photo):
    """The instructive case: nothing downstream can tell from the dimensions."""
    out = degrade.downscale_upscale(photo, 0.5)
    assert out.size == photo.size
    before = np.asarray(photo.convert("RGB")).astype(float)
    after = np.asarray(out).astype(float)
    assert np.abs(np.diff(after, axis=1)).mean() < np.abs(np.diff(before, axis=1)).mean()


def test_lower_jpeg_quality_loses_more(photo):
    ref = np.asarray(photo.convert("RGB")).astype(float)
    errs = [np.abs(np.asarray(degrade.jpeg(photo, q)).astype(float) - ref).mean()
            for q in (90, 70, 50, 30)]
    assert errs == sorted(errs), f"error should grow as quality falls, got {errs}"


def test_unknown_degradation_name_raises():
    with pytest.raises(KeyError):
        degrade.get("definitely-not-a-degradation")


def test_probe_round_trip_preserves_predictions(tmp_path):
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(200, 32))
    labels = (feats[:, 0] + rng.normal(0, 0.3, 200) > 0).astype(int)
    probe = LinearProbe.fit(feats, labels)
    path = tmp_path / "probe.json"
    probe.save(path)
    assert LinearProbe.load(path)(feats) == pytest.approx(probe(feats))


def test_probe_learns_the_right_direction():
    rng = np.random.default_rng(6)
    feats = np.r_[rng.normal(-1, 0.3, (100, 8)), rng.normal(1, 0.3, (100, 8))]
    labels = np.r_[np.zeros(100), np.ones(100)].astype(int)
    p = LinearProbe.fit(feats, labels)
    assert p(feats[:100]).mean() < 0.5 < p(feats[100:]).mean()


def _run(*args):
    return subprocess.run([sys.executable, "-m", "aidetect", *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_metadata_only_reports_a_generator(tmp_path):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat, Steps: 30, Sampler: Euler a, CFG scale: 7")
    p = tmp_path / "gen.png"
    Image.new("RGB", (32, 32)).save(p, pnginfo=meta)

    r = _run(str(p), "--metadata-only", "--json")
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got[0]["metadata_says_ai"] is True
    assert "Stable Diffusion" in (got[0]["generator"] or "")


def test_cli_metadata_only_is_silent_on_a_clean_file(tmp_path):
    p = tmp_path / "clean.png"
    Image.new("RGB", (32, 32)).save(p)
    got = json.loads(_run(str(p), "--metadata-only", "--json").stdout)
    assert got[0]["metadata_says_ai"] is False


def test_cli_reports_unreadable_files_instead_of_crashing(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"this is not an image")
    r = _run(str(bad), "--metadata-only", "--json")
    assert r.returncode == 0, r.stderr
    assert "error" in json.loads(r.stdout)[0]


def test_cli_exits_cleanly_when_nothing_matches(tmp_path):
    r = _run(str(tmp_path))
    assert r.returncode == 2
    assert "no images found" in r.stderr
