"""API tests.

A stub detector stands in for the real ensemble so these run in milliseconds
with no checkpoints and no network. What is being tested is the HTTP contract --
status codes, limits, ordering, error shape -- not the model, which the
benchmark covers.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi", reason="API extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from aidetect import api as api_module  # noqa: E402
from aidetect.calibration import FusionModel, PlattCalibrator  # noqa: E402
from aidetect.detector import Detector  # noqa: E402

pytestmark = pytest.mark.asgi


class StubBackend:
    """Bright images score high, so results are traceable to their input."""

    name = "stub"

    def score_batch(self, images):
        return np.array([float(np.asarray(im).mean()) / 255.0 for im in images])


@pytest.fixture
def client(monkeypatch):
    fusion = FusionModel(backends=["stub"], calibrators={"stub": PlattCalibrator(1.0, 0.0)},
                         thresholds={"fpr5": 0.6, "fpr1": 0.9, "balanced": 0.5},
                         dtype="float32")
    detector = Detector(backends={"stub": StubBackend()}, fusion=fusion,
                        threshold=0.6, low_threshold=0.4, use_metadata=False)
    # TestClient runs the lifespan, whose job is to load the real ensemble off
    # disk. Replacing _load keeps these tests fast and independent of whether
    # the checkpoints have been downloaded at all.
    def fake_load() -> None:
        api_module._state.update(detector=detector, error=None, loaded_at=0.1)

    monkeypatch.setattr(api_module, "_load", fake_load)
    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


def png(value: int, size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buf, "PNG")
    return buf.getvalue()


def upload(name: str, blob: bytes):
    return ("files", (name, io.BytesIO(blob), "image/png"))


def test_health_reports_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_health_reports_a_load_failure_instead_of_crashing(client, monkeypatch):
    monkeypatch.setitem(api_module._state, "detector", None)
    monkeypatch.setitem(api_module._state, "error", "OSError: no checkpoints")
    r = client.get("/health")
    assert r.status_code == 503
    assert "no checkpoints" in r.json()["error"]


def test_info_lists_the_ensemble_and_limits(client):
    body = client.get("/info").json()
    assert "stub" in body["ensemble"]
    assert body["limits"]["max_files_per_request"] == api_module.MAX_FILES
    assert "fpr5" in body["available_operating_points"]


def test_detect_returns_one_result_per_file_in_order(client):
    files = [upload(f"{v}.png", png(v)) for v in (10, 240, 90)]
    body = client.post("/detect", files=files).json()
    assert [r["filename"] for r in body["results"]] == ["10.png", "240.png", "90.png"]
    assert [round(r["p_ai"] * 255) for r in body["results"]] == [10, 240, 90]


def test_bright_and_dark_land_on_opposite_verdicts(client):
    body = client.post("/detect", files=[upload("dark.png", png(5)),
                                         upload("bright.png", png(250))]).json()
    assert body["results"][0]["verdict"] == "REAL"
    assert body["results"][1]["verdict"] == "AI-GENERATED"


def test_a_bad_file_fails_alone_and_keeps_its_slot(client):
    files = [upload("good.png", png(250)),
             upload("bad.png", b"not an image"),
             upload("good2.png", png(240))]
    body = client.post("/detect", files=files).json()
    verdicts = [r["verdict"] for r in body["results"]]
    assert verdicts == ["AI-GENERATED", "ERROR", "AI-GENERATED"]
    assert "bad.png" in body["results"][1]["error"]


def test_empty_upload_is_rejected(client):
    r = client.post("/detect", files=[upload("empty.png", b"")])
    assert r.status_code == 400


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_BYTES", 1024)
    r = client.post("/detect", files=[upload("big.png", png(200, (400, 400)))])
    assert r.status_code == 413


def test_too_many_files_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_FILES", 2)
    files = [upload(f"{i}.png", png(100)) for i in range(3)]
    assert client.post("/detect", files=files).status_code == 413


def test_missing_files_field_is_a_validation_error(client):
    assert client.post("/detect").status_code == 422


def test_operating_point_changes_the_threshold(client):
    files = [upload("mid.png", png(180))]
    loose = client.post("/detect?operating_point=balanced", files=files).json()
    strict = client.post("/detect?operating_point=fpr1", files=files).json()
    assert loose["results"][0]["threshold"] == 0.5
    assert strict["results"][0]["threshold"] == 0.9
    # Same image, same score; only where the line sits changed.
    assert loose["results"][0]["p_ai"] == strict["results"][0]["p_ai"]


def test_unknown_operating_point_is_rejected(client):
    r = client.post("/detect?operating_point=nope", files=[upload("a.png", png(100))])
    assert r.status_code == 400


def test_detect_is_unavailable_until_the_models_load(client, monkeypatch):
    monkeypatch.setitem(api_module._state, "detector", None)
    monkeypatch.setitem(api_module._state, "error", None)
    r = client.post("/detect", files=[upload("a.png", png(100))])
    assert r.status_code == 503
