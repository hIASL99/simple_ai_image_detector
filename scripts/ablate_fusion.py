#!/usr/bin/env python
"""Does ensembling actually help, and does calibrating first matter?

Worth settling with numbers, because the intuition cuts both ways. Several of
these checkpoints saturate -- their softmax reaches 1.000 on ordinary
photographs -- so averaging raw log-odds lets the most over-confident member
dominate, and adding members genuinely makes the ranking worse. Fitting a Platt
scale per member first removes that, and then the same members help.

Every row is leave-one-generator-out, so nothing here is fitted on what it is
scored on.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aidetect.calibration import PlattCalibrator, logit, metrics, sigmoid, threshold_at_fpr  # noqa: E402
from train_fusion import load_manifest, load_scores, logo_folds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out_of_fold(scores, labels, generators, members, calibrate, seed):
    out = np.full(len(labels), np.nan)
    for _, train, test in logo_folds(generators, labels, seed):
        if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
            continue
        if calibrate:
            cals = {n: PlattCalibrator.fit(scores[n][train], labels[train]) for n in members}
            z = np.mean([logit(cals[n](scores[n])) for n in members], axis=0)
        else:
            z = np.mean([logit(scores[n]) for n in members], axis=0)
        out[test] = sigmoid(z)[test]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "bench" / "manifest.csv")
    ap.add_argument("--scores", type=Path, default=ROOT / "data" / "scores")
    ap.add_argument("--protocol", default="crop")
    ap.add_argument("--members", nargs="+",
                    default=["commforensics384", "haywoodsloan", "organika-sdxl"])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    ids = np.array([r["id"] for r in rows])
    labels = np.array([1 if r["label"] == "ai" else 0 for r in rows])
    generators = np.array([r["generator"] for r in rows])
    scores = load_scores(args.protocol, args.scores, ids)
    members = [m for m in args.members if m in scores]
    if not members:
        print("none of the requested members have cached scores")
        return 1

    configs = [(f"{members[0]} alone, calibrated", members[:1], True)]
    for i in range(2, len(members) + 1):
        configs.append((f"+ {members[i - 1]}, calibrated", members[:i], True))
    configs += [
        (f"the same {len(members)}, UNCALIBRATED", members, False),
        ("every detector, calibrated", sorted(scores), True),
        ("every detector, UNCALIBRATED", sorted(scores), False),
    ]

    print(f"protocol={args.protocol}  n={len(ids)}  leave-one-generator-out\n")
    print(f"{'configuration':50s} {'AUROC':>7s} {'bAcc':>7s} {'TPR':>7s} {'FPR':>7s}")
    print("-" * 82)
    for tag, subset, calibrate in configs:
        s = out_of_fold(scores, labels, generators, subset, calibrate, args.seed)
        ok = ~np.isnan(s)
        thr = threshold_at_fpr(s[ok], labels[ok], args.target_fpr)
        m = metrics(s[ok], labels[ok], thr)
        print(f"{tag:50s} {m['auroc']:7.4f} {m['balanced_accuracy']:7.4f} "
              f"{m['tpr']:7.3f} {m['fpr']:7.3f}")
    print(f"\nbAcc/TPR/FPR are at a threshold fitted for {args.target_fpr:.0%} FPR. "
          f"AUROC is threshold-free, so a change there is not an artefact of where the "
          f"cut-off landed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
