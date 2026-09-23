"""Pretrained image-classification detectors, wrapped behind one interface.

Every backend answers the same question -- p(this image was generated) -- but
the underlying checkpoints disagree about almost everything else: label order,
input size, architecture, even how many classes they have. The registry below
records those differences explicitly, because a flipped label index produces a
detector that is confidently wrong and still looks plausible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .imageio import load_rgb
from .runtime import autocast_cpu, configure_threads

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".hf_cache"
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE))


@dataclass(frozen=True)
class BackendSpec:
    """Everything needed to turn a checkpoint into a p(AI) score."""

    name: str
    repo_id: str
    # Labels whose probability mass counts as "generated". Matched
    # case-insensitively against config.id2label, so the mapping survives a
    # checkpoint being re-uploaded with a different label order.
    ai_labels: tuple[str, ...]
    real_labels: tuple[str, ...]
    notes: str = ""
    # Some checkpoints ship without preprocessor_config.json. Naming a repo to
    # borrow the processor from is safer than silently falling back to
    # whatever defaults happen to be in scope.
    processor_repo: str | None = None
    # Some checkpoints are face/deepfake detectors rather than general
    # generated-image detectors; they are kept for comparison but excluded
    # from the default ensemble.
    general_purpose: bool = True
    license: str = "unknown"
    # "hf" = a transformers image-classification head; "commforensics" = the
    # single-logit ViT published through PyTorchModelHubMixin, which needs its
    # own loader and its own evaluation transform.
    kind: str = "hf"


REGISTRY: dict[str, BackendSpec] = {
    s.name: s for s in [
        BackendSpec("organika-sdxl", "Organika/sdxl-detector",
                    license="cc-by-nc-3.0",
                    ai_labels=("artificial",), real_labels=("human",),
                    notes="Swin-tiny fine-tuned on SDXL output; the most downloaded general detector."),
        BackendSpec("umm-maybe", "umm-maybe/AI-image-detector",
                    license="cc-by-4.0",
                    ai_labels=("artificial",), real_labels=("human",),
                    notes="Swin-tiny, 2022-era training data (GAN + early diffusion)."),
        BackendSpec("ateeqq-siglip", "Ateeqq/ai-vs-human-image-detector",
                    license="apache-2.0",
                    ai_labels=("ai",), real_labels=("hum", "human"),
                    notes="SigLIP backbone; trained on a broader generator mix."),
        BackendSpec("haywoodsloan", "haywoodsloan/ai-image-detector-deploy",
                    license="apache-2.0",
                    ai_labels=("artificial",), real_labels=("real",),
                    notes="SwinV2 at 256px, community-maintained, frequently retrained."),
        BackendSpec("smogy", "Smogy/SMOGY-Ai-images-detector",
                    license="cc-by-nc-4.0",
                    ai_labels=("artificial",), real_labels=("human",),
                    notes="Swin, derived from the umm-maybe lineage."),
        BackendSpec("dima806-vit", "dima806/ai_vs_human_generated_image_detection",
                    license="apache-2.0",
                    ai_labels=("ai-generated",), real_labels=("human",),
                    notes="ViT-base. NOTE the reversed label order: index 0 is 'human' here."),
        BackendSpec("nyuad", "NYUAD-ComNets/NYUAD_AI-generated_images_detector",
                    license="apache-2.0",
                    ai_labels=("dalle", "sd"), real_labels=("real",),
                    processor_repo="google/vit-base-patch16-224-in21k",
                    notes="Three-way (dalle / real / sd); p(AI) sums the two generator classes. "
                          "Ships no preprocessor_config.json, so it borrows the stock ViT one."),
        BackendSpec("prithiv-deepfake", "prithivMLmods/Deep-Fake-Detector-v2-Model",
                    ai_labels=("deepfake",), real_labels=("realism",),
                    general_purpose=False, license="apache-2.0",
                    notes="Face-deepfake detector, not a general generated-image detector."),
        BackendSpec("sadra-sdxl", "SadraCoding/SDXL-Deepfake-Detector",
                    license="mit",
                    ai_labels=("artificial",), real_labels=("human",),
                    processor_repo="google/vit-base-patch16-224-in21k",
                    notes="Another SDXL-era fine-tune; kept for ensemble diversity."),
        # Deliberately not registered: yaya36095/ai-image-detector ships a
        # ResNet-50 checkpoint behind a ViT config, so every ViT parameter is
        # randomly initialised at load time. The guard in load_backend() catches
        # this class of repo; this one is simply left out.
        BackendSpec("mmanikanta-swin", "mmanikanta/SWIN-AI-Image-Detector",
                    license="apache-2.0",
                    ai_labels=("fake",), real_labels=("real",),
                    notes="Small Swin with FAKE/REAL labels."),
        BackendSpec("mmanikanta-convnext", "mmanikanta/ConvNeXT_AI_image_detector",
                    license="apache-2.0",
                    ai_labels=("fake",), real_labels=("real",),
                    notes="ConvNeXt -- the only convolutional member, so it fails differently "
                          "from the transformer majority."),
        BackendSpec("commforensics384", "OwensLab/commfor-model-384",
                    ai_labels=("ai",), real_labels=("real",), kind="commforensics",
                    license="mit",
                    notes="Community Forensics (Park & Owens, CVPR 2025): ViT-S/16 @384 trained "
                          "on output from ~4.8k generators. Widest training distribution of any "
                          "public checkpoint."),
        BackendSpec("commforensics224", "OwensLab/commfor-model-224",
                    ai_labels=("ai",), real_labels=("real",), kind="commforensics",
                    license="mit",
                    notes="The 224px sibling: ~3x cheaper, slightly weaker."),
        BackendSpec("aiornot-siglip2", "prithivMLmods/AIorNot-SigLIP2",
                    license="apache-2.0",
                    ai_labels=("ai",), real_labels=("not ai", "not_ai", "real"),
                    notes="SigLIP2. Same family as several checkpoints reported to never "
                          "predict 'real' -- included so the benchmark can confirm or clear it."),
    ]
}


@dataclass
class ClassifierBackend:
    """A loaded checkpoint that scores images."""

    spec: BackendSpec
    model: object = field(repr=False)
    processor: object = field(repr=False)
    ai_indices: tuple[int, ...]
    real_indices: tuple[int, ...]
    id2label: dict[int, str]
    # bfloat16 autocast is ~2.5x faster on an AVX-512-BF16 CPU and shifts p(AI)
    # by ~0.005 on average, but by as much as 0.2 on individual images for the
    # SwinV2 checkpoints. Fine for interactive use, not for producing the
    # numbers a calibrator is fitted on -- hence off by default.
    bf16: bool = False

    @property
    def name(self) -> str:
        return self.spec.name

    @torch.inference_mode()
    def score_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Return p(AI) for each image, one float in [0, 1]."""
        if not images:
            return np.zeros(0, dtype=np.float64)
        inputs = self.processor(images=images, return_tensors="pt")
        with autocast_cpu(self.bf16):
            logits = self.model(**inputs).logits.to(torch.float64)
        probs = torch.softmax(logits, dim=-1).numpy()
        ai = probs[:, list(self.ai_indices)].sum(axis=1)
        real = probs[:, list(self.real_indices)].sum(axis=1)
        # Renormalise over the classes we actually understand, so a checkpoint
        # with extra classes is not silently penalised.
        total = ai + real
        return np.where(total > 0, ai / np.maximum(total, 1e-12), 0.5)

    def score_paths(self, paths, batch_size: int = 16, progress=None) -> np.ndarray:
        out: list[np.ndarray] = []
        paths = list(paths)
        for i in range(0, len(paths), batch_size):
            chunk = [load_rgb(p) for p in paths[i:i + batch_size]]
            out.append(self.score_batch(chunk))
            if progress is not None:
                progress(min(i + batch_size, len(paths)), len(paths))
        return np.concatenate(out) if out else np.zeros(0)


def _resolve_indices(id2label: dict[int, str], wanted: tuple[str, ...]) -> tuple[int, ...]:
    lowered = {i: str(lbl).strip().lower() for i, lbl in id2label.items()}
    want = {w.lower() for w in wanted}
    hits = tuple(sorted(i for i, lbl in lowered.items() if lbl in want))
    if not hits:
        # Fall back to substring matching before giving up -- some checkpoints
        # use 'Real'/'realism'/'human face' style variants.
        hits = tuple(sorted(i for i, lbl in lowered.items()
                            if any(w in lbl or lbl in w for w in want)))
    return hits


def load_backend(name: str, *, local_files_only: bool = False,
                 num_threads: int | None = None,
                 bf16: bool = False, flip_tta: bool = False):
    """Load a registered backend by name.

    Returns either a ClassifierBackend or a CommunityForensics instance; both
    expose the same score_batch/score_paths/name interface.
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    configure_threads(num_threads)

    if spec.kind == "commforensics":
        from .commforensics import CommunityForensics

        size = 384 if spec.repo_id.endswith("384") else 224
        return CommunityForensics.load(size, local_files_only=local_files_only,
                                       num_threads=num_threads, bf16=bf16,
                                       flip_tta=flip_tta)

    try:
        processor = AutoImageProcessor.from_pretrained(
            spec.repo_id, local_files_only=local_files_only)
    except (OSError, ValueError):
        if not spec.processor_repo:
            raise
        processor = AutoImageProcessor.from_pretrained(
            spec.processor_repo, local_files_only=local_files_only)
    model, info = AutoModelForImageClassification.from_pretrained(
        spec.repo_id, local_files_only=local_files_only, output_loading_info=True)
    model.eval()

    # A checkpoint whose weights do not match its own config loads *successfully*
    # with randomly initialised parameters, and then produces confident nonsense.
    # yaya36095/ai-image-detector is exactly this: a ResNet checkpoint behind a
    # ViT config. Refuse rather than score images with noise.
    missing = [k for k in info.get("missing_keys", []) if not k.endswith("position_ids")]
    if missing:
        raise RuntimeError(
            f"{name}: {len(missing)} parameters are missing from the checkpoint and were "
            f"randomly initialised (e.g. {missing[:3]}). The published weights do not match "
            f"the published config, so this model cannot be trusted.")

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    ai_idx = _resolve_indices(id2label, spec.ai_labels)
    real_idx = _resolve_indices(id2label, spec.real_labels)
    if not ai_idx or not real_idx:
        raise RuntimeError(
            f"{name}: could not map labels {id2label} onto ai={spec.ai_labels} "
            f"real={spec.real_labels}. Refusing to guess -- a wrong polarity here "
            f"inverts every prediction.")
    if set(ai_idx) & set(real_idx):
        raise RuntimeError(f"{name}: label sets overlap: ai={ai_idx} real={real_idx}")

    return ClassifierBackend(spec=spec, model=model, processor=processor,
                             ai_indices=ai_idx, real_indices=real_idx, id2label=id2label,
                             bf16=bf16)


def default_backend_names(include_specialised: bool = False) -> list[str]:
    return [n for n, s in REGISTRY.items() if s.general_purpose or include_specialised]
