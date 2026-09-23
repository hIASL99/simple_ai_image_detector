"""Calibration and thresholding decide what the probability *means*."""

from __future__ import annotations

import numpy as np
import pytest

from aidetect.calibration import (FusionModel, PlattCalibrator, logit, metrics,
                                  sigmoid, threshold_at_fpr, threshold_at_tpr)


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(400), np.ones(400)].astype(int)
    scores = sigmoid(np.r_[rng.normal(-1.5, 1, 400), rng.normal(1.5, 1, 400)])
    return scores, labels


def test_platt_improves_calibration_without_changing_the_ranking(separable):
    from sklearn.metrics import roc_auc_score
    scores, labels = separable
    cal = PlattCalibrator.fit(scores, labels)
    out = cal(scores)
    assert roc_auc_score(labels, out) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)
    brier_before = float(np.mean((scores - labels) ** 2))
    brier_after = float(np.mean((out - labels) ** 2))
    assert brier_after < brier_before


def test_platt_on_a_single_class_split_is_a_no_op():
    """A leave-one-generator-out fold can hand us one class; it must not crash."""
    cal = PlattCalibrator.fit(np.array([0.2, 0.3, 0.4]), np.array([1, 1, 1]))
    assert (cal.a, cal.b) == (1.0, 0.0)


def test_platt_stays_finite_on_perfectly_separable_input():
    """Unregularised logistic regression sends the slope to infinity here."""
    scores = np.r_[np.full(50, 0.01), np.full(50, 0.99)]
    labels = np.r_[np.zeros(50), np.ones(50)].astype(int)
    cal = PlattCalibrator.fit(scores, labels)
    assert np.isfinite(cal.a) and np.isfinite(cal.b)
    assert 0.0 < cal(np.array([0.5]))[0] < 1.0


@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.2, 0.5])
def test_threshold_at_fpr_never_exceeds_the_target(separable, target):
    scores, labels = separable
    thr = threshold_at_fpr(scores, labels, target)
    achieved = float((scores[labels == 0] >= thr).mean())
    assert achieved <= target + 1e-9


@pytest.mark.parametrize("target", [0.9, 0.95, 0.98, 1.0])
def test_threshold_at_tpr_reaches_the_target(separable, target):
    scores, labels = separable
    thr = threshold_at_tpr(scores, labels, target)
    assert float((scores[labels == 1] >= thr).mean()) >= target - 1e-9


def test_thresholds_survive_ties_and_tiny_samples():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert np.isfinite(threshold_at_fpr(scores, labels, 0.05))
    assert np.isfinite(threshold_at_tpr(scores, labels, 0.98))
    assert threshold_at_fpr(np.array([0.3]), np.array([1]), 0.05) == 0.5


def test_metrics_matches_a_hand_computed_confusion_matrix():
    scores = np.array([0.9, 0.8, 0.4, 0.1])
    labels = np.array([1, 0, 1, 0])
    m = metrics(scores, labels, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["tpr"] == 0.5 and m["fpr"] == 0.5
    assert m["balanced_accuracy"] == 0.5


def test_metrics_reports_nan_auroc_when_a_class_is_missing():
    m = metrics(np.array([0.2, 0.4]), np.array([1, 1]), 0.5)
    assert np.isnan(m["auroc"]) and np.isnan(m["ap"])


def test_fusion_is_order_independent():
    f = FusionModel(backends=["a", "b", "c"],
                    calibrators={n: PlattCalibrator(1.0, 0.0) for n in "abc"})
    s = {"a": np.array([0.9]), "b": np.array([0.2]), "c": np.array([0.6])}
    shuffled = {k: s[k] for k in ("c", "a", "b")}
    assert f.fuse(s) == pytest.approx(f.fuse(shuffled))


def test_fusion_warns_when_a_member_is_missing():
    f = FusionModel(backends=["a", "b"],
                    calibrators={n: PlattCalibrator() for n in "ab"})
    with pytest.warns(RuntimeWarning, match="fusing without"):
        f.fuse({"a": np.array([0.9])})


def test_fusion_json_round_trip_preserves_predictions():
    f = FusionModel(backends=["a", "b"],
                    calibrators={"a": PlattCalibrator(2.0, -1.0),
                                 "b": PlattCalibrator(0.5, 0.25)},
                    thresholds={"fpr5": 0.7}, dtype="bfloat16")
    g = FusionModel.from_json(f.to_json())
    s = {"a": np.array([0.3, 0.9]), "b": np.array([0.6, 0.1])}
    assert g.fuse(s) == pytest.approx(f.fuse(s))
    assert g.thresholds == f.thresholds and g.dtype == "bfloat16"


def test_logit_saturation_stays_finite():
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()
    assert np.isfinite(sigmoid(np.array([-1e6, 1e6]))).all()
