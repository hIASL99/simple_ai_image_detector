"""Frozen-backbone image embeddings.

The UniversalFakeDetect line of work (Ojha et al., CVPR 2023) found that a
linear classifier on *frozen* CLIP features generalises to generators it never
saw far better than a network fine-tuned end-to-end on one generator -- the
fine-tuned model latches onto that generator's fingerprint, the frozen features
do not. This module supplies those features.

What has moved on since is the backbone, not the head: "Simplicity Prevails"
(arXiv 2602.01738) reports the same linear probe at 0.842 in-the-wild AUROC on
a 2025-vintage backbone against ~0.62 on the original CLIP that UnivFD used.
So the backbone is a parameter here rather than a constant. DINOv3 is gated
(HTTP 401 without a token), so the modern option we can actually run offline is
Meta's Perception Encoder, PE-Core.
"""

from __future__ import annotations

import contextlib
import math
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

CLIP_REPO = "openai/clip-vit-large-patch14"

# Short names for the backbones we hold weights for. Anything not listed is
# passed to timm verbatim, so trying another checkpoint needs no edit here.
ALIASES = {
    "clip-vit-l14": CLIP_REPO,
    "pe-core-b16-224": "vit_pe_core_base_patch16_224.fb",
}


@contextlib.contextmanager
def _offline(enabled: bool):
    """Force huggingface_hub offline for the duration.

    timm has no local_files_only parameter -- it calls huggingface_hub itself --
    and the hub library reads HF_HUB_OFFLINE into a module constant at import
    time, so setting the environment variable alone is too late to have any
    effect once anything has already imported it.
    """
    if not enabled:
        yield
        return
    from huggingface_hub import constants

    prev_env = os.environ.get("HF_HUB_OFFLINE")
    prev_flag = constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = prev_flag
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env


def _crop_boxes(w: int, h: int, side: int, n: int) -> list[tuple[int, int, int, int]]:
    """Up to n side x side windows: centre first, then corners, then a grid.

    Centre first so that a small n still contains the view a single-crop
    detector would have had, and so the set only grows as n grows. Duplicate
    origins are dropped, which is why a nearly-square-crop-sized image yields
    fewer than n windows.
    """
    origins = [((w - side) // 2, (h - side) // 2),
               (0, 0), (w - side, 0), (0, h - side), (w - side, h - side)]
    if n > len(origins):
        k = math.ceil(math.sqrt(n))
        step_x = (w - side) / max(k - 1, 1)
        step_y = (h - side) / max(k - 1, 1)
        origins += [(round(i * step_x), round(j * step_y))
                    for j in range(k) for i in range(k)]
    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in origins:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        boxes.append((x, y, x + side, y + side))
        if len(boxes) == n:
            break
    return boxes


def native_tiles(img: Image.Image, side: int, n: int) -> list[Image.Image]:
    """Up to n crops of exactly side x side, taken at the image's own scale.

    Returns [] when the image is smaller than one tile; the caller then has to
    fall back to the backbone's resize transform. Upscaling instead would be
    worse than the fallback: it invents high-frequency content with the
    interpolator's own signature, in the band the detector reads.
    """
    w, h = img.size
    if w < side or h < side:
        return []
    return [img.crop(b) for b in _crop_boxes(w, h, side, n)]


def _tensorise(images: list[Image.Image], mean, std) -> torch.Tensor:
    """Scale to [0,1] and normalise, with no resampling of any kind.

    This is what the backbone's own transform does once its resize and crop
    become no-ops, which is the case exactly when the input is already the
    model's input size. Verified against CLIPImageProcessor on 224x224 input:
    max abs difference 2.4e-7.
    """
    from torchvision.transforms import functional as TF

    return torch.stack([TF.normalize(TF.to_tensor(im), mean, std) for im in images])


@dataclass(kw_only=True)
class FeatureExtractor:
    """One frozen backbone behind one interface.

    crops == 1 uses the backbone's own resize transform, i.e. the whole frame
    squashed to the input size. That is the setting the linear-probe papers
    report and what the cached bench features were built with, so it stays the
    default. crops > 1 switches to native-resolution tiles and no resampling at
    all, because a resize is a low-pass filter over precisely the
    high-frequency residue that separates rendered pixels from sensor pixels.
    """

    name: str
    dim: int
    input_size: int
    crops: int = 1
    flip_tta: bool = False
    bf16: bool = False
    # Images smaller than one tile cannot be tiled and go through the resize
    # transform instead. Counted rather than raised: it is a property of the
    # data, not an error, but it silently changes what the features mean, so
    # callers have to be able to see that it happened.
    resize_fallbacks: int = field(default=0, init=False)

    def _preprocess_resized(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess_tiles(self, tiles: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """L2-normalised embeddings, shape (n, dim)."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        tiles: list[Image.Image] = []
        whole: list[Image.Image] = []
        counts: list[int] = []
        for img in images:
            got = native_tiles(img, self.input_size, self.crops) if self.crops > 1 else []
            if got:
                tiles.extend(got)
                counts.append(len(got))
            else:
                if self.crops > 1:
                    self.resize_fallbacks += 1
                whole.append(img)
                counts.append(0)  # 0 marks "one embedding, from the resize path"

        tiled = self._embed(self._preprocess_tiles(tiles)) if tiles else None
        resized = self._embed(self._preprocess_resized(whole)) if whole else None

        # Mean of the *unnormalised* embeddings. Normalising each tile first
        # and renormalising the mean is a different operator: it discards each
        # tile's norm, which is how strongly the backbone responded to that
        # tile, so an empty patch of sky would pull the average exactly as hard
        # as a detailed one.
        out = np.empty((len(images), self.dim), dtype=np.float64)
        ti = wi = 0
        for row, n in enumerate(counts):
            if n:
                out[row] = tiled[ti:ti + n].mean(axis=0)
                ti += n
            else:
                out[row] = resized[wi]
                wi += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-12)).astype(np.float32)

    def _embed(self, batch: torch.Tensor) -> np.ndarray:
        if self.flip_tta:
            # Mirroring the preprocessed tensor, not the file: this is exactly
            # the mirror of the view the model was given, and it costs no second
            # decode or resize. It is bit-identical to mirroring the image first
            # for tiles, and for the resize path whenever the centre crop lands
            # evenly; where it does not (512x342 -> 335 wide, an odd 111px
            # margin) the two differ by the one-pixel crop offset, which is a
            # difference between two equally valid views rather than an error.
            # Averaged in embedding space, before any normalisation, for the
            # same reason the tiles are.
            n = batch.shape[0]
            both = self._forward(torch.cat([batch, torch.flip(batch, dims=[3])]))
            embedded = 0.5 * (both[:n] + both[n:])
        else:
            embedded = self._forward(batch)
        if embedded.shape[1] != self.dim:
            raise RuntimeError(
                f"{self.name}: backbone returned {embedded.shape[1]}-d features but the "
                f"extractor was built for {self.dim}-d; a probe fitted on one and applied "
                f"to the other would score noise.")
        return embedded.astype(np.float64)

    def embed_paths(self, paths, batch_size: int = 16, progress=None) -> np.ndarray:
        paths = list(paths)
        out = []
        for i in range(0, len(paths), batch_size):
            out.append(self.embed_batch([load_rgb(p) for p in paths[i:i + batch_size]]))
            if progress is not None:
                progress(min(i + batch_size, len(paths)), len(paths))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)


@dataclass(kw_only=True)
class ClipFeatures(FeatureExtractor):
    """CLIP ViT-L/14 image embeddings -- the original UnivFD backbone."""

    model: object = field(repr=False)
    processor: object = field(repr=False)

    @classmethod
    def load(cls, repo_id: str = CLIP_REPO, *, crops: int = 1, flip_tta: bool = False,
             local_files_only: bool = False, num_threads: int | None = None,
             bf16: bool = False) -> "ClipFeatures":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        configure_threads(num_threads)
        processor = CLIPImageProcessor.from_pretrained(repo_id, local_files_only=local_files_only)
        model = CLIPVisionModelWithProjection.from_pretrained(
            repo_id, local_files_only=local_files_only)
        model.eval()
        return cls(model=model, processor=processor, name=repo_id,
                   dim=int(model.config.projection_dim),
                   input_size=int(processor.crop_size["height"]),
                   crops=crops, flip_tta=flip_tta, bf16=bf16)

    def _preprocess_resized(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def _preprocess_tiles(self, tiles: list[Image.Image]) -> torch.Tensor:
        return _tensorise(tiles, self.processor.image_mean, self.processor.image_std)

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        with autocast_cpu(self.bf16):
            return self.model(pixel_values=batch).image_embeds.to(torch.float32).numpy()


@dataclass(kw_only=True)
class TimmFeatures(FeatureExtractor):
    """Any timm backbone with its head removed, preprocessed its own way.

    The preprocessing is not a detail. PE-Core normalises with mean = std = 0.5
    and resamples bicubic at crop_pct 1.0, not with the ImageNet statistics most
    of timm uses; feeding it ImageNet-normalised pixels shifts every embedding
    and the probe on top of them degrades quietly rather than failing.
    """

    model: torch.nn.Module = field(repr=False)
    transform: object = field(repr=False)
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def load(cls, model_name: str, *, crops: int = 1, flip_tta: bool = False,
             local_files_only: bool = False, num_threads: int | None = None,
             bf16: bool = False) -> "TimmFeatures":
        import timm
        from timm.data import create_transform, resolve_data_config

        configure_threads(num_threads)
        if model_name.startswith("timm/"):
            model_name = f"hf-hub:{model_name}"
        with _offline(local_files_only):
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()

        # timm's ordinary loader raises when a name has no weights, but its
        # custom-load path (and an empty pretrained_cfg) only *warns* and hands
        # back a randomly initialised network -- which still looks like a
        # working feature extractor and just scores AUROC 0.5. Same source keys
        # timm itself resolves against.
        cfg = dict(getattr(model, "pretrained_cfg", None) or {})
        if not any(cfg.get(k) for k in ("url", "hf_hub_id", "file", "state_dict")):
            raise RuntimeError(
                f"{model_name}: timm has no pretrained weights for this name, so the "
                f"backbone is randomly initialised. Refusing to embed with it.")

        data_cfg = resolve_data_config({}, model=model)
        transform = create_transform(**data_cfg, is_training=False)
        size = int(data_cfg["input_size"][-1])
        if data_cfg["input_size"][-2] != size:
            raise RuntimeError(f"{model_name}: non-square input {data_cfg['input_size']} "
                               f"is not supported by the tiling path.")
        return cls(model=model, transform=transform, name=model_name,
                   dim=int(model.num_features), input_size=size,
                   mean=tuple(data_cfg["mean"]), std=tuple(data_cfg["std"]),
                   crops=crops, flip_tta=flip_tta, bf16=bf16)

    def _preprocess_resized(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(im) for im in images])

    def _preprocess_tiles(self, tiles: list[Image.Image]) -> torch.Tensor:
        return _tensorise(tiles, self.mean, self.std)

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        with autocast_cpu(self.bf16):
            return self.model(batch).to(torch.float32).numpy()


def load_features(name: str = "clip-vit-l14", *, crops: int = 1, flip_tta: bool = False,
                  local_files_only: bool = False, num_threads: int | None = None,
                  bf16: bool = False) -> FeatureExtractor:
    """Load a backbone by short name, CLIP repo id, or timm model name."""
    resolved = ALIASES.get(name, name)
    kwargs = dict(crops=crops, flip_tta=flip_tta, local_files_only=local_files_only,
                  num_threads=num_threads, bf16=bf16)
    # A plain repo id such as openai/clip-... is a transformers CLIP; a timm
    # model is either bare ("vit_pe_core_base_patch16_224.fb") or hub-qualified.
    if resolved.startswith(("hf-hub:", "timm/")) or "/" not in resolved:
        return TimmFeatures.load(resolved, **kwargs)
    return ClipFeatures.load(resolved, **kwargs)
