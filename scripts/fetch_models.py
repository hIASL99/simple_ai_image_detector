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

# Weight formats we never load; downloading them would triple the transfer.
SKIP = ["*.msgpack", "*.h5", "*.ot", "*flax*", "*tf_model*", "*.onnx", "*.pb"]

EXTRA = {
    "openai/clip-vit-large-patch14": "CLIP ViT-L/14 backbone for the linear probe",
    "timm/vit_pe_core_base_patch16_224.fb": "PE-Core backbone (Apache-2.0) for the linear probe",
    "google/vit-base-patch16-224-in21k": "stock ViT processor, borrowed by checkpoints that ship none",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble-only", action="store_true",
                    help="fetch only the checkpoints listed in models/fusion.json")
    ap.add_argument("--all", action="store_true",
                    help="fetch every registered checkpoint, including the ones the benchmark "
                         "rejected (useful for reproducing the benchmark)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    wanted: dict[str, str] = {}
    fusion = ROOT / "models" / "fusion.json"
    if args.ensemble_only and fusion.exists():
        import json
        members = set(json.loads(fusion.read_text())["backends"])
        wanted = {REGISTRY[m].repo_id: REGISTRY[m].notes for m in members if m in REGISTRY}
    else:
        wanted = {s.repo_id: s.notes for n, s in REGISTRY.items()
                  if args.all or s.general_purpose}
        wanted.update(EXTRA)

    print(f"cache: {CACHE}")
    failures = []
    for repo, why in wanted.items():
        try:
            snapshot_download(repo, ignore_patterns=SKIP)
            print(f"  ok      {repo}")
        except Exception as exc:  # noqa: BLE001
            failures.append((repo, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED  {repo}  ({type(exc).__name__})")
    if failures:
        print("\nsome checkpoints could not be fetched:")
        for repo, why in failures:
            print(f"  {repo}: {why[:160]}")
        print("Gated or removed repositories are expected; the rest of the ensemble still works.")
    total = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file())
    print(f"\ncache now holds {total / 1e9:.1f} GB")
    return 1 if len(failures) == len(wanted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
