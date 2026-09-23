"""A linear probe on frozen CLIP features.

Deliberately the simplest possible head: logistic regression on L2-normalised
embeddings. The capacity is the point -- a linear head cannot memorise a
generator's fingerprint, so what it learns tends to transfer to generators it
has never seen. Anything heavier reliably scores better in-distribution and
worse out of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import sigmoid


@dataclass
class LinearProbe:
    """w . x + b, squashed. `classes_positive` documents which side is AI."""

    weights: np.ndarray
    bias: float
    dim: int
    backbone: str = "openai/clip-vit-large-patch14"
    crops: int = 1
    trained_on: list[str] | None = None

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, *, C: float = 1.0,
            backbone: str = "openai/clip-vit-large-patch14", crops: int = 1,
            trained_on: list[str] | None = None) -> "LinearProbe":
        from sklearn.linear_model import LogisticRegression

        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        # class_weight balances the loss so an unequal real/AI split in the
        # training pool does not bake a prior into the decision boundary.
        lr = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")
        lr.fit(features, labels)
        return cls(weights=lr.coef_[0].astype(np.float64), bias=float(lr.intercept_[0]),
                   dim=features.shape[1], backbone=backbone, crops=crops,
                   trained_on=list(trained_on) if trained_on else None)

    def decision(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.weights + self.bias

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision(features))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "weights": self.weights.tolist(), "bias": self.bias, "dim": self.dim,
            "backbone": self.backbone, "crops": self.crops, "trained_on": self.trained_on,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        d = json.loads(Path(path).read_text())
        return cls(weights=np.asarray(d["weights"], dtype=np.float64), bias=float(d["bias"]),
                   dim=int(d["dim"]), backbone=d.get("backbone", "openai/clip-vit-large-patch14"),
                   crops=int(d.get("crops", 1)), trained_on=d.get("trained_on"))
