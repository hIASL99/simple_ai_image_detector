"""The Community Forensics detector (Park & Owens, CVPR 2025).

A ViT-S/16 trained on output from ~4,800 generators -- by far the widest
training distribution of any public checkpoint, which is why independent 2026
benchmarks put it ahead of every other off-the-shelf detector on generators
nobody has seen before.

The checkpoint is published through PyTorchModelHubMixin, so the Hub holds the
weights but not the class that defines them. The architecture and the exact
evaluation transform below are transcribed from the authors' repository
(models.py and dataloader.get_transform(mode="val")); getting the transform
wrong is the easy way to silently lose most of the model's accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .runtime import autocast_cpu, configure_threads

REPOS = {384: "OwensLab/commfor-model-384", 224: "OwensLab/commfor-model-224"}
# Pinned so a silent re-upload upstream cannot change what the shipped
# calibration was fitted against. Set REVISIONS[size] = None to track main.
REVISIONS = {384: None, 224: None}
TIMM_NAMES = {384: "vit_small_patch16_384", 224: "vit_small_patch16_224"}
# Resize the short side to this, then centre-crop to input_size.
RESIZE = {384: 440, 224: 256}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CommunityForensics:
    """Single-logit detector; sigmoid(logit) is p(generated)."""

    model: torch.nn.Module
    input_size: int
    bf16: bool = False
    flip_tta: bool = False

    @classmethod
    def load(cls, input_size: int = 384, *, local_files_only: bool = False,
             num_threads: int | None = None, bf16: bool = False,
             flip_tta: bool = False) -> "CommunityForensics":
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if input_size not in REPOS:
            raise ValueError(f"input_size must be one of {sorted(REPOS)}")
        configure_threads(num_threads)

        path = hf_hub_download(REPOS[input_size], "model.safetensors",
                               revision=REVISIONS.get(input_size),
                               local_files_only=local_files_only)
        state = load_file(path)
        # The published keys are prefixed with the attribute name of the inner
        # timm model ("vit."); strip it to load into a bare timm ViT.
        state = {k[len("vit."):]: v for k, v in state.items() if k.startswith("vit.")}

        model = timm.create_model(TIMM_NAMES[input_size], pretrained=False, num_classes=1)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Community Forensics weights do not match {TIMM_NAMES[input_size]}: "
                f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        model.eval()
        return cls(model=model, input_size=input_size, bf16=bf16, flip_tta=flip_tta)

    @property
    def name(self) -> str:
        return f"commforensics{self.input_size}"

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Resize short side -> RESIZE, centre-crop to input_size, ImageNet norm."""
        from torchvision.transforms import functional as TF

        target = self.input_size
        tensors = []
        for img in images:
            resized = TF.resize(img, RESIZE[target])
            cropped = TF.center_crop(resized, [target, target])
            t = TF.to_tensor(cropped)
            tensors.append(TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD))
        return torch.stack(tensors)

    @torch.inference_mode()
    def score_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros(0, dtype=np.float64)
        batch = self.preprocess(images)
        with autocast_cpu(self.bf16):
            logits = self.model(batch).to(torch.float64).reshape(-1)
            if self.flip_tta:
                flipped = self.model(torch.flip(batch, dims=[3])).to(torch.float64).reshape(-1)
                # Averaging logits, not probabilities: the two views are the
                # same evidence seen twice, so their log-odds add and average.
                logits = 0.5 * (logits + flipped)
        return torch.sigmoid(logits).numpy()

    def score_paths(self, paths, batch_size: int = 16, progress=None) -> np.ndarray:
        from .imageio import load_rgb

        paths = list(paths)
        out = []
        for i in range(0, len(paths), batch_size):
            out.append(self.score_batch([load_rgb(p) for p in paths[i:i + batch_size]]))
            if progress is not None:
                progress(min(i + batch_size, len(paths)), len(paths))
        return np.concatenate(out) if out else np.zeros(0)
