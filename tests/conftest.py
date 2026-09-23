"""Shared fixtures, plus a guard that keeps the default suite offline.

Nothing in tests/ may touch the network or load a real checkpoint unless it is
marked `slow` and explicitly opted into with --runslow. A test that silently
downloads a model is a test that passes on this machine and fails on any other.
"""

from __future__ import annotations

import io
import socket
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="also run tests that load the downloaded checkpoints")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs the checkpoints; pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Make any socket call raise, so an accidental download fails loudly."""
    if "slow" in request.keywords:
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("this test tried to use the network")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def encode(img: Image.Image, fmt: str = "PNG", **params) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **params)
    return buf.getvalue()


@pytest.fixture
def rgb_image() -> Image.Image:
    import numpy as np
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((64, 48, 3)) * 255).astype("uint8"))
