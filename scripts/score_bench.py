#!/usr/bin/env python
"""Score every benchmark image with every detector and cache the results.

Scoring is the expensive part -- fourteen checkpoints across three protocols
over a few thousand images is hours of CPU -- and it is pure: the same
checkpoint on the same file always gives the same number. So it is done once,
written to data/scores/, and every later analysis reads from there.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidetect.backends import REGISTRY, load_backend  # noqa: E402
from aidetect.features import load_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
SCORES = ROOT / "data" / "scores"
PROTOCOLS = ("native", "matched", "crop")


def read_manifest(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _progress(tag: str):
    start = time.time()
    state = {"last": 0.0}

    def report(done: int, total: int) -> None:
        now = time.time()
        if now - state["last"] < 5 and done < total:
            return
        state["last"] = now
        rate = done / max(now - start, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        print(f"\r    {tag}: {done}/{total}  {rate:5.1f} img/s  eta {eta:5.0f}s",
              end="", flush=True)
        if done >= total:
            print()
    return report


def score_backend(name: str, rows: list[dict], protocol: str, batch_size: int,
                  threads: int, bf16: bool = False) -> np.ndarray:
    backend = load_backend(name, local_files_only=True, num_threads=threads, bf16=bf16)
    paths = [ROOT / r[protocol] for r in rows]
    return backend.score_paths(paths, batch_size=batch_size,
                               progress=_progress(f"{name}/{protocol}"))


def embed_features(rows: list[dict], protocol: str, backbone: str, crops: int,
                   batch_size: int, threads: int, bf16: bool = False) -> np.ndarray:
    extractor = load_features(backbone, crops=crops, local_files_only=True,
                              num_threads=threads, bf16=bf16)
    paths = [ROOT / r[protocol] for r in rows]
    return extractor.embed_paths(paths, batch_size=batch_size,
                                 progress=_progress(f"{backbone}x{crops}/{protocol}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=BENCH / "manifest.csv")
    ap.add_argument("--out", type=Path, default=SCORES)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--backends", nargs="+", default=["all"],
                    help='registered backend names, "all", or "none" for features only')
    ap.add_argument("--feature-backbone", default="pe-core-b16-224",
                    help="frozen backbone for the linear probe. pe-core-b16-224 is ~4.5x "
                         "cheaper than clip-vit-l14 at the same 768-d width")
    ap.add_argument("--clip-crops", type=int, default=1,
                    help="native-resolution crops averaged per image for the probe features")
    ap.add_argument("--skip-clip", action="store_true",
                    help="score the classifiers only, no probe features")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=max(1, (torch.get_num_threads() or 4)))
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 autocast: ~2.5x faster on an AVX-512-BF16 CPU, and shifts "
                         "p(AI) by ~0.005 on average")
    ap.add_argument("--suffix", default="",
                    help="tag appended to the cache filenames, so an fp32 control run does not "
                         "overwrite the bf16 sweep")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.backends == ["all"]:
        names = sorted(REGISTRY)
    elif args.backends == ["none"]:
        names = []  # features only
    else:
        names = args.backends
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise SystemExit(f"unknown backends: {unknown}; known: {sorted(REGISTRY)}")
    args.out.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["id"] for r in rows])
    print(f"{len(rows)} images | backends: {', '.join(names)} | protocols: {', '.join(args.protocols)}")

    for protocol in args.protocols:
        for name in names:
            dest = args.out / f"{name}{args.suffix}__{protocol}.npz"
            if dest.exists() and not args.force:
                print(f"  cached {dest.name}")
                continue
            t0 = time.time()
            try:
                scores = score_backend(name, rows, protocol, args.batch_size,
                                       args.threads, bf16=args.bf16)
            except Exception as exc:  # noqa: BLE001 - one bad checkpoint must not stop the sweep
                print(f"  FAILED {name}/{protocol}: {type(exc).__name__}: {exc}")
                continue
            np.savez_compressed(dest, ids=ids, scores=scores)
            print(f"  wrote {dest.name} in {time.time() - t0:.0f}s")

        if not args.skip_clip:
            tag = f"feat-{args.feature_backbone}x{args.clip_crops}{args.suffix}"
            dest = args.out / f"{tag}__{protocol}.npz"
            if dest.exists() and not args.force:
                print(f"  cached {dest.name}")
                continue
            t0 = time.time()
            try:
                feats = embed_features(rows, protocol, args.feature_backbone,
                                       args.clip_crops, args.batch_size,
                                       args.threads, bf16=args.bf16)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED features/{protocol}: {type(exc).__name__}: {exc}")
                continue
            np.savez_compressed(dest, ids=ids, features=feats)
            print(f"  wrote {dest.name} in {time.time() - t0:.0f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
