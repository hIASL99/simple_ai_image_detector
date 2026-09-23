"""Turning raw detector scores into probabilities you can act on.

Two separate problems, often confused:

  calibration  a raw score of 0.9 does not mean "90% likely generated". Platt
               scaling on the log-odds fixes the shape of the score so that it
               does.
  thresholding 0.5 is only the right cut-off if the two classes are equally
               frequent and the two error types equally costly. Neither is true
               here -- calling a real photo fake is the expensive mistake -- so
               the operating point is chosen from a target false-positive rate.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60, 60)))


@dataclass
class PlattCalibrator:
    """One-dimensional logistic recalibration: p' = sigmoid(a * logit(p) + b)."""

    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = logit(scores).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            return cls()  # nothing to learn from a single-class split
        # Mild regularisation keeps `a` finite when a detector separates the
        # calibration set perfectly, which would otherwise send it to infinity
        # and produce 0/1 probabilities that no longer rank anything.
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        return cls(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * logit(scores) + self.b)


@dataclass
class FusionModel:
    """Combines several calibrated detectors into one probability.

    Averaging log-odds (rather than probabilities) is the right default: it is
    the correct combination rule for independent evidence, and it stops a
    single saturated detector from dominating the vote.
    """

    backends: list[str]
    calibrators: dict[str, PlattCalibrator]
    weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    # bfloat16 autocast shifts the raw logits, so a calibration fitted under one
    # dtype does not transfer to the other. Recording it lets the detector warn
    # instead of silently reporting a miscalibrated probability.
    dtype: str = "unknown"

    def calibrated(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {n: self.calibrators[n](np.asarray(scores[n]))
                for n in self.backends if n in scores}

    def missing(self, scores: dict[str, np.ndarray]) -> list[str]:
        return [n for n in self.backends if n not in scores]

    def fuse(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        cal = self.calibrated(scores)
        if not cal:
            raise ValueError("no backend scores supplied")
        absent = self.missing(scores)
        if absent:
            # The stored thresholds were fitted for the full ensemble. Dropping
            # a member still produces a usable ranking, but the probability is
            # no longer calibrated and the operating point no longer holds, so
            # say so instead of returning a number that looks trustworthy.
            warnings.warn(
                f"fusing without {absent}: the probability and the stored "
                f"thresholds are no longer calibrated for this subset",
                RuntimeWarning, stacklevel=2)
        total_w = sum(self.weights.get(n, 1.0) for n in cal)
        stacked = sum(self.weights.get(n, 1.0) * logit(v) for n, v in cal.items())
        return sigmoid(stacked / max(total_w, _EPS))

    def to_json(self) -> str:
        return json.dumps({
            "backends": self.backends,
            "calibrators": {k: asdict(v) for k, v in self.calibrators.items()},
            "weights": self.weights,
            "thresholds": self.thresholds,
            "dtype": self.dtype,
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "FusionModel":
        d = json.loads(text)
        return cls(backends=d["backends"],
                   calibrators={k: PlattCalibrator(**v) for k, v in d["calibrators"].items()},
                   weights=d.get("weights", {}),
                   thresholds=d.get("thresholds", {}),
                   dtype=d.get("dtype", "unknown"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "FusionModel":
        return cls.from_json(Path(path).read_text())


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate is <= target_fpr.

    labels: 1 = AI-generated, 0 = real. A "false positive" is a real photo
    called generated -- the error that destroys trust in the tool.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    negatives = np.sort(scores[labels == 0])
    if negatives.size == 0:
        return 0.5
    # Keep at most floor(target_fpr * n) negatives above the threshold.
    allowed = int(np.floor(target_fpr * negatives.size))
    if allowed <= 0:
        return float(np.nextafter(negatives[-1], 1.0))
    return float(negatives[-allowed])


def threshold_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    """Largest threshold that still catches at least `target_tpr` of the AI images.

    The mirror of threshold_at_fpr, and the lower edge of the abstention band:
    below it, so few generated images land that calling the image real is a
    controlled decision rather than a guess.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    positives = np.sort(scores[labels == 1])
    if positives.size == 0:
        return 0.5
    # Keep at least ceil(target_tpr * n) positives at or above the threshold.
    keep = int(np.ceil(target_tpr * positives.size))
    keep = min(max(keep, 1), positives.size)
    return float(positives[positives.size - keep])


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy/AUROC/AP plus the error rates that actually matter."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= threshold).astype(int)

    pos, neg = labels == 1, labels == 0
    tp = int((pred[pos] == 1).sum()) if pos.any() else 0
    fn = int((pred[pos] == 0).sum()) if pos.any() else 0
    fp = int((pred[neg] == 1).sum()) if neg.any() else 0
    tn = int((pred[neg] == 0).sum()) if neg.any() else 0

    out = {
        "n": int(labels.size), "n_ai": int(pos.sum()), "n_real": int(neg.sum()),
        "threshold": float(threshold),
        "accuracy": float((pred == labels).mean()) if labels.size else float("nan"),
        "tpr": tp / max(tp + fn, 1),   # recall on generated images
        "fpr": fp / max(fp + tn, 1),   # real photos wrongly flagged
        "precision": tp / max(tp + fp, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    out["balanced_accuracy"] = 0.5 * (out["tpr"] + (1 - out["fpr"]))
    if pos.any() and neg.any():
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["ap"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out
