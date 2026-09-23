"""Label polarity is the highest-consequence detail in the whole registry.

Index 0 means "artificial" in most of these checkpoints and "human" in others.
Getting it backwards produces a detector that is confidently inverted and looks
entirely healthy, so the mapping is resolved from config.id2label at load time
and every registry entry is checked here against the real label sets.
"""

from __future__ import annotations

import pytest

from aidetect.backends import REGISTRY, _resolve_indices, default_backend_names


@pytest.mark.parametrize("id2label,ai,real,expect_ai,expect_real", [
    ({0: "artificial", 1: "human"}, ("artificial",), ("human",), (0,), (1,)),
    ({0: "human", 1: "AI-generated"}, ("ai-generated",), ("human",), (1,), (0,)),
    ({0: "ai", 1: "hum"}, ("ai",), ("hum", "human"), (0,), (1,)),
    ({0: "Real", 1: "AI"}, ("ai",), ("real",), (1,), (0,)),
    ({0: "FAKE", 1: "REAL"}, ("fake",), ("real",), (0,), (1,)),
    ({0: "dalle", 1: "real", 2: "sd"}, ("dalle", "sd"), ("real",), (0, 2), (1,)),
])
def test_polarity_resolution(id2label, ai, real, expect_ai, expect_real):
    assert _resolve_indices(id2label, ai) == expect_ai
    assert _resolve_indices(id2label, real) == expect_real


def test_unmappable_labels_resolve_to_nothing_so_the_loader_can_refuse():
    """Returning () is what makes load_backend raise instead of guessing."""
    assert _resolve_indices({0: "cat", 1: "dog"}, ("artificial",)) == ()


def test_every_registry_entry_has_disjoint_label_sets():
    for name, spec in REGISTRY.items():
        overlap = {a.lower() for a in spec.ai_labels} & {r.lower() for r in spec.real_labels}
        assert not overlap, f"{name} claims {overlap} is both AI and real"
        assert spec.ai_labels and spec.real_labels, f"{name} is missing a label set"


def test_face_deepfake_models_are_excluded_from_the_default_ensemble():
    """They answer a different question and score at chance on general images."""
    assert "prithiv-deepfake" not in default_backend_names()
    assert "prithiv-deepfake" in default_backend_names(include_specialised=True)


def test_registry_records_a_real_licence_for_every_checkpoint():
    """Several are CC-BY-NC, which callers need to know before shipping."""
    for name, spec in REGISTRY.items():
        assert spec.license != "unknown", f"{name} has no licence recorded"


def test_non_commercial_checkpoints_are_identifiable():
    """The README's licensing section is generated from this, so it must hold."""
    nc = {n for n, s in REGISTRY.items() if "-nc" in s.license}
    assert {"organika-sdxl", "smogy"} <= nc
