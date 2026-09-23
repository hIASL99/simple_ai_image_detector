"""predict_paths must return exactly one result per input, in input order.

Worth its own file: the first implementation reordered results by looking each
one up in a dict keyed on the path string. A file listed twice in one call --
which happens the moment a user globs overlapping patterns -- collapsed into a
single key, and every result after it came back against the wrong image. The
scores were right; they were attached to the wrong filenames, which is the kind
of wrong that nobody notices.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aidetect.detector import Detector


class StubBackend:
    """Scores an image by its mean brightness, so results are identifiable."""

    name = "stub"

    def score_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([float(np.asarray(im).mean()) / 255.0 for im in images])


@pytest.fixture
def detector():
    return Detector(backends={"stub": StubBackend()}, fusion=None,
                    threshold=0.5, use_metadata=False)


def _write(tmp_path, name: str, value: int):
    p = tmp_path / name
    Image.new("RGB", (16, 16), (value, value, value)).save(p)
    return p


def test_results_follow_input_order(detector, tmp_path):
    values = [10, 200, 90, 250]
    paths = [_write(tmp_path, f"{v:03d}.png", v) for v in values]
    got = detector.predict_paths([str(p) for p in paths], batch_size=2)

    assert [r.path for r in got] == [str(p) for p in paths]
    # The stub scores brightness, so each result must carry its own image's
    # value -- this is what catches a result landing on the wrong filename.
    assert [round(r.p_ai * 255) for r in got] == values


def test_a_repeated_path_gets_its_own_result(detector, tmp_path):
    a = _write(tmp_path, "a.png", 10)
    b = _write(tmp_path, "b.png", 200)
    order = [str(a), str(b), str(a), str(b), str(a)]
    got = detector.predict_paths(order, batch_size=2)
    assert [r.path for r in got] == order
    # Same file, same score, wherever it appears.
    assert got[0].p_ai == got[2].p_ai == got[4].p_ai
    assert got[1].p_ai == got[3].p_ai
    assert got[0].p_ai != got[1].p_ai


def test_unreadable_inputs_keep_their_slot(detector, tmp_path):
    good = _write(tmp_path, "good.png", 200)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    missing = tmp_path / "gone.png"

    order = [str(bad), str(good), str(missing), str(good), str(bad)]
    got = detector.predict_paths(order, batch_size=2)
    assert [r.path for r in got] == order
    assert [r.verdict for r in got] == ["ERROR", "AI-GENERATED", "ERROR",
                                        "AI-GENERATED", "ERROR"]
    assert all(r.error for r in (got[0], got[2], got[4]))


def test_every_input_yields_exactly_one_result(detector, tmp_path):
    paths = [str(_write(tmp_path, f"{i}.png", i * 20)) for i in range(11)]
    for batch in (1, 2, 3, 8, 32):
        got = detector.predict_paths(paths, batch_size=batch)
        assert len(got) == len(paths)
        assert [r.path for r in got] == paths
