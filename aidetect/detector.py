"""The assembled detector: metadata evidence plus a calibrated model ensemble."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .backends import REGISTRY, ClassifierBackend, load_backend
from .calibration import FusionModel, logit, sigmoid
from .features import load_features
from .imageio import open_image
from .metadata import MetadataVerdict, read_metadata
from .probe import LinearProbe

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models"

CLIP_KEY = "clip-probe"


@dataclass
class Prediction:
    """One image's verdict, with the reasoning kept attached."""

    path: str
    p_ai: float
    threshold: float
    # Below this, so few generated images land that "real" is a controlled call
    # rather than a guess. Between the two is the abstention band.
    low_threshold: float | None = None
    per_backend: dict[str, float] = field(default_factory=dict)
    per_backend_raw: dict[str, float] = field(default_factory=dict)
    metadata: MetadataVerdict | None = None
    truncated: bool = False
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        # Only structured, hard-to-fake provenance overrides the pixels: an
        # A1111 parameter block, a ComfyUI node graph, a C2PA/IPTC
        # trainedAlgorithmicMedia assertion. A bare "Software: Midjourney"
        # string is one EXIF write away from appearing on a real photo, so it
        # stays a weak hint (metadata.weak_ai_evidence) and never decides.
        # Camera-shaped EXIF never overrides in the other direction, for the
        # same reason: editors copy it onto generated files routinely.
        if self.metadata is not None and self.metadata.says_ai:
            # A generative fill on a photograph sets says_ai too. Reporting that
            # as "wholly AI-generated" is a different, wrong claim, so a
            # partial-edit marker only overrides toward the edited verdict.
            return "AI-EDITED" if self.metadata.partial_ai else "AI-GENERATED"
        if self.truncated:
            # Pillow fills the undecodable tail of a truncated file with flat
            # grey (measured: mean exactly 128). That is a large synthetic
            # region the detector will happily read as evidence, so the score
            # is about the padding, not the photograph.
            return "UNCERTAIN"
        if self.p_ai >= self.threshold:
            return "AI-GENERATED"
        if self.low_threshold is not None and self.p_ai > self.low_threshold:
            # Neither error rate is controlled in here. Saying so beats guessing:
            # under heavy JPEG or a screenshot the whole score distribution
            # slides toward "real", and a forced answer is then silently wrong.
            return "UNCERTAIN"
        return "REAL"

    @property
    def basis(self) -> str:
        if self.error:
            return self.error
        if self.metadata is not None and self.metadata.says_ai:
            kind = "a generated region in an otherwise real image" if self.metadata.partial_ai \
                else "the whole image"
            return f"metadata describes {kind}: " + "; ".join(self.metadata.ai_evidence)
        if self.metadata is not None and self.metadata.weak_ai_evidence:
            weak = "; ".join(self.metadata.weak_ai_evidence)
            return f"pixel ensemble; weak metadata hint ({weak})"
        if self.truncated:
            return ("file is truncated; the missing part decoded as flat grey, so the "
                    "score would describe the padding")
        if self.verdict == "UNCERTAIN":
            return (f"pixel ensemble is between the operating points "
                    f"({self.low_threshold:.2f} < p <= {self.threshold:.2f})")
        margin = abs(self.p_ai - self.threshold)
        strength = "borderline" if margin < 0.1 else ("clear" if margin > 0.3 else "moderate")
        return f"pixel ensemble ({strength}, {len(self.per_backend)} models)"

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "p_ai": round(self.p_ai, 4),
            "threshold": round(self.threshold, 4),
            "low_threshold": None if self.low_threshold is None else round(self.low_threshold, 4),
            "basis": self.basis,
            "per_model": {k: round(v, 4) for k, v in sorted(self.per_backend.items())},
            "metadata_says_ai": bool(self.metadata and self.metadata.says_ai),
            "metadata_partial_ai": bool(self.metadata and self.metadata.partial_ai),
            "metadata_weak_hints": list(self.metadata.weak_ai_evidence) if self.metadata else [],
            "metadata_generator": self.metadata.generator if self.metadata else None,
            "metadata_camera_exif": bool(self.metadata and self.metadata.says_camera),
            "has_c2pa": bool(self.metadata and self.metadata.has_c2pa),
            "truncated": self.truncated,
            "error": self.error,
        }


@dataclass
class Detector:
    """Loads the ensemble once, then scores many images."""

    backends: dict[str, ClassifierBackend]
    fusion: FusionModel | None
    clip: object | None = None
    probe: LinearProbe | None = None
    threshold: float = 0.5
    low_threshold: float | None = None
    use_metadata: bool = True

    @classmethod
    def load(cls, *, model_dir: str | os.PathLike = DEFAULT_MODEL_DIR,
             backend_names: list[str] | None = None,
             operating_point: str = "fpr5",
             local_files_only: bool = True,
             num_threads: int | None = None,
             use_metadata: bool = True,
             bf16: bool | None = None) -> "Detector":
        from .runtime import cpu_supports_bf16

        # The shipped calibration was fitted on bf16 scores, so bf16 is the
        # default wherever the CPU supports it: matching numerics matters more
        # than the ~0.005 average shift, and it is ~2.5x faster.
        if bf16 is None:
            bf16 = cpu_supports_bf16()
        model_dir = Path(model_dir)
        fusion_path = model_dir / "fusion.json"
        probe_path = model_dir / "probe.json"

        fusion = FusionModel.load(fusion_path) if fusion_path.exists() else None
        probe = LinearProbe.load(probe_path) if probe_path.exists() else None

        if backend_names is None:
            backend_names = list(fusion.backends) if fusion else [
                n for n, s in REGISTRY.items() if s.general_purpose]

        clip = None
        wanted = [n for n in backend_names if n != CLIP_KEY]
        if probe is not None and (fusion is None or CLIP_KEY in fusion.backends):
            clip = load_features(probe.backbone, crops=probe.crops,
                                 local_files_only=local_files_only,
                                 num_threads=num_threads, bf16=bf16)
        backends = {n: load_backend(n, local_files_only=local_files_only,
                                    num_threads=num_threads, bf16=bf16) for n in wanted}

        if fusion is not None and fusion.dtype not in ("unknown", ""):
            running = "bfloat16" if bf16 else "float32"
            if running != fusion.dtype:
                import warnings
                warnings.warn(
                    f"models/fusion.json was calibrated under {fusion.dtype} but this run uses "
                    f"{running}; the probability and thresholds will be slightly off",
                    RuntimeWarning, stacklevel=2)
        threshold, low = 0.5, None
        if fusion and fusion.thresholds:
            threshold = float(fusion.thresholds.get(operating_point,
                                                    fusion.thresholds.get("fpr5", 0.5)))
            low = fusion.thresholds.get("abstain_low")
            low = None if low is None else float(low)
        return cls(backends=backends, fusion=fusion, clip=clip, probe=probe,
                   threshold=threshold, low_threshold=low, use_metadata=use_metadata)

    # -- scoring ---------------------------------------------------------

    def raw_scores(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        scores = {name: b.score_batch(images) for name, b in self.backends.items()}
        if self.clip is not None and self.probe is not None:
            scores[CLIP_KEY] = self.probe(self.clip.embed_batch(images))
        return scores

    def fuse(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        if self.fusion is not None:
            return self.fusion.fuse({k: v for k, v in scores.items()
                                     if k in self.fusion.backends})
        # No fitted fusion on disk: fall back to an unweighted log-odds mean.
        # Honest but uncalibrated, so the probability is a ranking, not a rate.
        stacked = np.mean([logit(v) for v in scores.values()], axis=0)
        return sigmoid(stacked)

    def predict_paths(self, paths, batch_size: int = 8) -> list[Prediction]:
        """Score every path, returning one Prediction per input, in input order.

        Results are written into a slot reserved by position rather than looked
        up by path: the same file may legitimately appear twice in one call, and
        keying the reordering on the path string would collapse those into one
        slot and shuffle everything after it.
        """
        paths = [Path(p) for p in paths]
        out: list[Prediction | None] = [None] * len(paths)

        for start in range(0, len(paths), batch_size):
            chunk = list(enumerate(paths[start:start + batch_size], start=start))
            images, ok, truncated = [], [], []
            for index, p in chunk:
                try:
                    loaded = open_image(p)
                except Exception as exc:  # noqa: BLE001
                    out[index] = Prediction(path=str(p), p_ai=float("nan"),
                                            threshold=self.threshold,
                                            low_threshold=self.low_threshold,
                                            error=f"{type(exc).__name__}: {exc}")
                    continue
                images.append(loaded.image)
                ok.append((index, p))
                truncated.append(loaded.truncated)

            if not images:
                continue
            raw = self.raw_scores(images)
            cal = (self.fusion.calibrated({k: v for k, v in raw.items()
                                           if k in self.fusion.backends})
                   if self.fusion else raw)
            fused = self.fuse(raw)
            for j, (index, p) in enumerate(ok):
                meta = None
                if self.use_metadata:
                    try:
                        meta = read_metadata(p)
                    except Exception:  # noqa: BLE001 - metadata is best-effort
                        meta = None
                out[index] = Prediction(
                    path=str(p), p_ai=float(fused[j]), threshold=self.threshold,
                    low_threshold=self.low_threshold, truncated=truncated[j],
                    per_backend={k: float(v[j]) for k, v in cal.items()},
                    per_backend_raw={k: float(v[j]) for k, v in raw.items()},
                    metadata=meta)

        assert all(r is not None for r in out), "every input must yield a prediction"
        return out  # type: ignore[return-value]

    def predict(self, path: str | os.PathLike) -> Prediction:
        return self.predict_paths([path])[0]
