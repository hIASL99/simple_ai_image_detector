#!/usr/bin/env python
"""One-time download of the checkpoints into the project-local cache.

After this runs, everything else works with HF_HOME pointing at .hf_cache and
local_files_only=True, so no part of the tool touches the network again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(CACHE))

sys.path.insert(0, str(ROOT))
from aidetect.backends import REGISTRY  # noqa: E402

# Only what inference reads. Several of these repos were uploaded straight from
# a training run and carry an optimiser state and intermediate checkpoints --
# haywoodsloan ships a 1.5 GB optimizer.pt and a duplicate checkpoint directory,
# Organika a 694 MB one -- none of which is loaded. Fetching the whole repo
# costs 9.7 GB; fetching this list costs ~1.7 GB for the same ensemble.
ALLOW = [
    "*.safetensors", "*.json", "*.txt", "*.model",
]
SKIP = [
    "*.msgpack", "*.h5", "*.ot", "*flax*", "*tf_model*", "*.onnx", "*.pb",
    "*optimizer.pt", "*scheduler.pt", "*rng_state*", "*trainer_state*",
    "checkpoint-*/*", "runs/*", "*.bin",   # .bin always duplicates .safetensors here
]

EXTRA = {
    "openai/clip-vit-large-patch14": "CLIP ViT-L/14 backbone for the linear probe",
    "timm/vit_pe_core_base_patch16_224.fb": "PE-Core backbone (Apache-2.0) for the linear probe",
    "google/vit-base-patch16-224-in21k": "stock ViT processor, borrowed by checkpoints that ship none",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble-only", action="store_true",
                    help="fetch only the checkpoints listed in models/fusion.json")
    ap.add_argument("--model-dir", default="models",
                    help="which fitted ensemble to resolve (models or models_permissive)")
    ap.add_argument("--all", action="store_true",
                    help="fetch every registered checkpoint, including the ones the benchmark "
                         "rejected (useful for reproducing the benchmark)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    wanted: dict[str, str] = {}
    model_dir = ROOT / args.model_dir
    fusion = model_dir / "fusion.json"
    if args.ensemble_only and fusion.exists():
        import json
        members = set(json.loads(fusion.read_text())["backends"])
        wanted = {REGISTRY[m].repo_id: REGISTRY[m].notes for m in members if m in REGISTRY}
        # "clip-probe" is not a checkpoint -- it is a linear head we trained,
        # stored in probe.json, that runs on a frozen backbone. Without the
        # backbone the ensemble cannot load at all, and the failure only shows
        # up at startup, so resolve it here rather than trusting the registry.
        probe = model_dir / "probe.json"
        if probe.exists():
            backbone = json.loads(probe.read_text()).get("backbone", "")
            if backbone:
                from aidetect.features import ALIASES
                repo = ALIASES.get(backbone, backbone)
                if "/" not in repo:
                    repo = f"timm/{repo}"
                wanted[repo] = "frozen backbone for the linear probe"
    else:
        wanted = {s.repo_id: s.notes for n, s in REGISTRY.items()
                  if args.all or s.general_purpose}
        wanted.update(EXTRA)

    print(f"cache: {os.environ['HF_HOME']}")
    failures = []
    for repo, why in wanted.items():
        try:
            snapshot_download(repo, allow_patterns=ALLOW, ignore_patterns=SKIP)
            print(f"  ok      {repo}")
        except Exception as exc:  # noqa: BLE001
            failures.append((repo, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED  {repo}  ({type(exc).__name__})")
    if failures:
        print("\nsome checkpoints could not be fetched:")
        for repo, why in failures:
            print(f"  {repo}: {why[:160]}")
        print("Gated or removed repositories are expected; the rest of the ensemble still works.")
    # Report the cache actually in use, and count each blob once: the snapshot
    # tree is symlinks into blobs/, so following them doubles every number.
    cache = Path(os.environ["HF_HOME"])
    total = sum(f.stat().st_size for f in cache.rglob("*")
                if f.is_file() and not f.is_symlink())
    print(f"\n{cache} now holds {total / 1e9:.2f} GB")
    return 1 if len(failures) == len(wanted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
