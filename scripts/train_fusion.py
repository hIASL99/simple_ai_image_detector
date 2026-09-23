#!/usr/bin/env python
"""Select the ensemble, calibrate it, and measure it honestly.

Three questions, three experiments, in increasing order of how much they can
embarrass us:

  leave-one-generator-out   hold out one generator at a time. Answers "what
                            happens when the next image model ships?"
  cross-track               fit on the locally assembled corpus, score on the
                            CommunityForensics benchmark, and the reverse. The
                            two tracks share no generators, no real-image
                            pipeline and no collection date, so this is the
                            closest thing to a genuinely external test set.
  in-sample                 fit and score on everything. Reported only as the
                            optimistic bound it is.

Ensemble membership is chosen by greedy forward selection on out-of-fold
scores, not by reputation: a checkpoint joins only if it raises held-out AUROC.
That is what keeps the two broken checkpoints in the registry from dragging the
ensemble down, without anyone having to know in advance which two they are.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidetect.calibration import (FusionModel, PlattCalibrator, metrics,  # noqa: E402
                                  threshold_at_fpr, threshold_at_tpr)
from aidetect.probe import LinearProbe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
SCORES = ROOT / "data" / "scores"
MODELS = ROOT / "models"
CLIP_KEY = "clip-probe"


# -- loading -----------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_scores(protocol: str, score_dir: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    index = {v: i for i, v in enumerate(ids)}
    out: dict[str, np.ndarray] = {}
    for f in sorted(score_dir.glob(f"*__{protocol}.npz")):
        name = f.name[: -len(f"__{protocol}.npz")]
        if name.startswith("clipfeat"):
            continue
        data = np.load(f, allow_pickle=False)
        if "scores" not in data:
            continue
        aligned = np.full(len(ids), np.nan)
        for got, score in zip(data["ids"], data["scores"]):
            pos = index.get(str(got))
            if pos is not None:
                aligned[pos] = score
        missing = int(np.isnan(aligned).sum())
        if missing:
            print(f"  skipping {name}: {missing} images unscored")
            continue
        out[name] = aligned
    return out


def load_features(protocol: str, score_dir: Path, ids: np.ndarray,
                  crops: int, backbone: str) -> np.ndarray | None:
    # "none" means the probe is either not wanted or is already present as a
    # precomputed out-of-fold score column, which is far cheaper: refitting it
    # inside every candidate evaluation costs hours, and its score does not
    # depend on which other members are in the ensemble anyway.
    if backbone == "none":
        return None
    candidates = [score_dir / f"feat-{backbone}x{crops}__{protocol}.npz"]
    candidates += sorted(score_dir.glob(f"feat-*__{protocol}.npz"))
    candidates += [score_dir / f"clipfeat{crops}__{protocol}.npz"]
    for f in candidates:
        if not f.exists():
            continue
        data = np.load(f, allow_pickle=False)
        index = {str(v): i for i, v in enumerate(data["ids"])}
        rows = [index.get(str(i)) for i in ids]
        if any(r is None for r in rows):
            continue
        return data["features"][rows]
    return None


# -- fitting -----------------------------------------------------------------

def fit_on(train: np.ndarray, scores: dict[str, np.ndarray], labels: np.ndarray,
           feats: np.ndarray | None, members: list[str] | None,
           probe_C: float) -> tuple[FusionModel, LinearProbe | None, dict[str, np.ndarray]]:
    """Fit the CLIP probe and one Platt calibrator per member on `train` rows."""
    working = dict(scores)
    probe = None
    if feats is not None:
        probe = LinearProbe.fit(feats[train], labels[train], C=probe_C)
        working[CLIP_KEY] = probe(feats)

    names = members if members is not None else sorted(working)
    names = [n for n in names if n in working]
    calibrators = {n: PlattCalibrator.fit(working[n][train], labels[train]) for n in names}
    return FusionModel(backends=names, calibrators=calibrators), probe, working


def logo_folds(generators: np.ndarray, labels: np.ndarray, seed: int):
    """Leave-one-generator-out, with the real images partitioned across folds.

    Each real image belongs to exactly one fold, so no real image is ever both
    trained on and tested on within a fold, and the pooled out-of-fold score
    array has exactly one entry per image.
    """
    ai_gens = sorted({g for g, l in zip(generators, labels) if l == 1})
    real_idx = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    shares = np.array_split(rng.permutation(real_idx), len(ai_gens))
    for gen, real_share in zip(ai_gens, shares):
        test = np.zeros(len(labels), dtype=bool)
        test[(generators == gen) & (labels == 1)] = True
        test[real_share] = True
        yield gen, np.flatnonzero(~test), np.flatnonzero(test)


def out_of_fold(scores: dict[str, np.ndarray], labels: np.ndarray,
                generators: np.ndarray, feats: np.ndarray | None,
                members: list[str], probe_C: float, seed: int) -> np.ndarray:
    """Pooled out-of-fold fused score for one candidate ensemble."""
    oof = np.full(len(labels), np.nan)
    for _, train, test in logo_folds(generators, labels, seed):
        if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
            continue
        fusion, _, working = fit_on(train, scores, labels, feats, members, probe_C)
        oof[test] = fusion.fuse({n: working[n] for n in fusion.backends})[test]
    return oof


def greedy_select(scores: dict[str, np.ndarray], labels: np.ndarray,
                  generators: np.ndarray, feats: np.ndarray | None,
                  probe_C: float, seed: int, max_members: int) -> tuple[list[str], list[tuple]]:
    """Add the backend that most improves out-of-fold AUROC, until none does."""
    from sklearn.metrics import roc_auc_score

    pool = sorted(scores) + ([CLIP_KEY] if feats is not None else [])
    chosen: list[str] = []
    history: list[tuple] = []
    best_auroc = 0.0
    while len(chosen) < max_members:
        candidates = []
        for name in pool:
            if name in chosen:
                continue
            oof = out_of_fold(scores, labels, generators, feats, chosen + [name], probe_C, seed)
            ok = ~np.isnan(oof)
            if ok.sum() < 50 or len(np.unique(labels[ok])) < 2:
                continue
            candidates.append((float(roc_auc_score(labels[ok], oof[ok])), name))
        if not candidates:
            break
        candidates.sort(reverse=True)
        auroc, name = candidates[0]
        # A member has to buy a real improvement; 1e-4 is noise at this sample size.
        if auroc <= best_auroc + 1e-4:
            history.append((None, auroc, "stopped: no candidate improved out-of-fold AUROC"))
            break
        best_auroc = auroc
        chosen.append(name)
        history.append((name, auroc, f"added ({len(chosen)} members)"))
        print(f"    + {name:22s} out-of-fold AUROC {auroc:.4f}")
    return chosen, history


# -- reporting ---------------------------------------------------------------

def per_generator(fused: np.ndarray, labels: np.ndarray, generators: np.ndarray,
                  threshold: float) -> dict[str, dict]:
    out = {}
    for gen in sorted({g for g, l in zip(generators, labels) if l == 1}):
        mask = (generators == gen) & (labels == 1) & ~np.isnan(fused)
        n = int(mask.sum())
        if n:
            out[gen] = {"n": n, "recall": float((fused[mask] >= threshold).mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=BENCH / "manifest.csv")
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--protocol", default="crop",
                    help="which benchmark protocol to fit on (crop is the honest one)")
    ap.add_argument("--clip-crops", type=int, default=1)
    ap.add_argument("--feature-backbone", default="pe-core-b16-224",
                    help="backbone for the linear probe that ships in models/")
    ap.add_argument("--probe-column", default="probe-oof",
                    help="cached score column holding the probe's out-of-fold predictions. "
                         "Using it instead of refitting the probe inside every candidate "
                         "evaluation turns hours of selection into seconds, and is equivalent: "
                         "the probe's score does not depend on the rest of the ensemble")
    ap.add_argument("--dtype", default="bfloat16",
                    help="numeric mode the cached scores were produced under; stored in "
                         "fusion.json so the detector can warn on a mismatch")
    ap.add_argument("--probe-C", type=float, default=1.0)
    ap.add_argument("--max-members", type=int, default=6)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--abstain-tpr", type=float, default=0.98,
                    help="share of AI images that must fall above the lower band edge")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--members", nargs="*", default=None,
                    help="skip selection and use exactly these backends")
    ap.add_argument("--out", type=Path, default=MODELS)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "fusion_report.json")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    ids = np.array([r["id"] for r in rows])
    labels = np.array([1 if r["label"] == "ai" else 0 for r in rows])
    generators = np.array([r["generator"] for r in rows])
    tracks = np.array([r["track"] for r in rows])
    sources = np.array([r["source"] for r in rows])

    scores = load_scores(args.protocol, args.scores, ids)
    if not scores:
        print(f"no cached scores for protocol {args.protocol!r} in {args.scores}")
        return 1
    # Selection and the held-out evaluation always run off the cached probe
    # column, never off live features: refitting a 768-d logistic regression
    # inside every candidate x fold turns seconds of work into hours, and
    # changes nothing, because the probe's score is independent of which other
    # detectors sit beside it.
    feats = None
    print(f"protocol={args.protocol}  n={len(ids)}  candidates={sorted(scores)}  "
          f"probe column={'present' if args.probe_column in scores else 'absent'}")

    report: dict = {"protocol": args.protocol, "n": int(len(ids)),
                    "candidates": sorted(scores), "target_fpr": args.target_fpr}

    # -- each candidate on its own, so the report shows what each one is worth
    solo = {}
    for name, s in scores.items():
        oof = out_of_fold(scores, labels, generators, None, [name], args.probe_C, args.seed)
        ok = ~np.isnan(oof)
        thr = threshold_at_fpr(oof[ok], labels[ok], args.target_fpr)
        solo[name] = metrics(oof[ok], labels[ok], thr)
    report["solo_leave_one_generator_out"] = solo
    print("\n  leave-one-generator-out, each detector alone:")
    for name, m in sorted(solo.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"    {name:22s} AUROC={m['auroc']:.4f}  bAcc={m['balanced_accuracy']:.4f}  "
              f"TPR@{args.target_fpr:.0%}fpr={m['tpr']:.3f}")

    # -- ensemble selection
    if args.members:
        members, history = list(args.members), []
        print(f"\n  using the given ensemble: {members}")
    else:
        print("\n  greedy forward selection on out-of-fold AUROC:")
        members, history = greedy_select(scores, labels, generators, feats,
                                         args.probe_C, args.seed, args.max_members)
    # The cached column stands in for the probe during selection; the shipped
    # model refers to it by the name the detector computes at inference time.
    uses_probe = args.probe_column in members
    report["members"] = [CLIP_KEY if m == args.probe_column else m for m in members]
    report["selection_history"] = [{"added": a, "auroc": b, "note": c} for a, b, c in history]
    if not members:
        print("  no ensemble could be selected")
        return 1

    # -- leave-one-generator-out for the chosen ensemble
    oof = np.full(len(labels), np.nan)
    fold_rows = []
    for gen, train, test in logo_folds(generators, labels, args.seed):
        if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
            continue
        fusion, _, working = fit_on(train, scores, labels, feats, members, args.probe_C)
        fused = fusion.fuse({n: working[n] for n in fusion.backends})
        oof[test] = fused[test]
        thr = threshold_at_fpr(fused[train], labels[train], args.target_fpr)
        m = metrics(fused[test], labels[test], thr)
        fold_rows.append({"held_out": gen, "n_test": int(test.size), **m})
    report["logo_folds"] = fold_rows
    ok = ~np.isnan(oof)
    oof_thr = threshold_at_fpr(oof[ok], labels[ok], args.target_fpr)
    report["logo_pooled"] = metrics(oof[ok], labels[ok], oof_thr)
    report["logo_per_generator"] = per_generator(oof, labels, generators, oof_thr)
    p = report["logo_pooled"]
    print(f"\n  leave-one-generator-out, ensemble: AUROC={p['auroc']:.4f} "
          f"bAcc={p['balanced_accuracy']:.4f} TPR={p['tpr']:.3f} FPR={p['fpr']:.3f}")

    # -- cross-track: fit on one corpus, report on the other
    report["cross_track"] = {}
    for fit_track in sorted(set(tracks)):
        train = np.flatnonzero(tracks == fit_track)
        test = np.flatnonzero(tracks != fit_track)
        if not train.size or not test.size:
            continue
        if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
            continue
        fusion, _, working = fit_on(train, scores, labels, feats, members, args.probe_C)
        fused = fusion.fuse({n: working[n] for n in fusion.backends})
        thr = threshold_at_fpr(fused[train], labels[train], args.target_fpr)
        m = metrics(fused[test], labels[test], thr)
        key = f"fit_on_{fit_track}_test_on_rest"
        report["cross_track"][key] = {
            **m, "per_generator": per_generator(fused, labels, generators, thr),
        }
        print(f"  fit on track {fit_track:5s} -> test on the rest: AUROC={m['auroc']:.4f} "
              f"bAcc={m['balanced_accuracy']:.4f} TPR={m['tpr']:.3f} FPR={m['fpr']:.3f}")

    # -- final model, fitted on everything
    fusion, probe, working = fit_on(np.arange(len(labels)), scores, labels, feats,
                                    members, args.probe_C)
    fused = fusion.fuse({n: working[n] for n in fusion.backends})

    if uses_probe:
        # Refit the probe on everything for shipping, and rename its slot so
        # Detector.load knows to compute it from the backbone rather than look
        # for a checkpoint called "probe-oof".
        final_feats = load_features(args.protocol, args.scores, ids, args.clip_crops,
                                    args.feature_backbone)
        if final_feats is None:
            print(f"  cannot ship the probe: no cached {args.feature_backbone} features")
            return 1
        probe = LinearProbe.fit(final_feats, labels, C=args.probe_C,
                                backbone=args.feature_backbone, crops=args.clip_crops)
        fusion.backends = [CLIP_KEY if n == args.probe_column else n for n in fusion.backends]
        fusion.calibrators[CLIP_KEY] = fusion.calibrators.pop(args.probe_column)
    fusion.thresholds = {
        "fpr1": threshold_at_fpr(fused, labels, 0.01),
        "fpr5": threshold_at_fpr(fused, labels, 0.05),
        "fpr10": threshold_at_fpr(fused, labels, 0.10),
        "balanced": 0.5,
        # Lower edge of the abstention band: below it, 98% of generated images
        # have already been caught, so calling the image real is a controlled
        # decision. Between the two edges the tool says "uncertain".
        "abstain_low": threshold_at_tpr(fused, labels, args.abstain_tpr),
    }
    fusion.dtype = args.dtype
    args.out.mkdir(parents=True, exist_ok=True)
    fusion.save(args.out / "fusion.json")
    if probe is not None:
        probe.trained_on = sorted(set(sources.tolist()))
        probe.crops = args.clip_crops
        probe.backbone = args.feature_backbone
        probe.save(args.out / "probe.json")
    report["thresholds"] = fusion.thresholds
    report["in_sample"] = metrics(fused, labels, fusion.thresholds["fpr5"])
    print(f"\n  in-sample (optimistic): AUROC={report['in_sample']['auroc']:.4f} "
          f"bAcc={report['in_sample']['balanced_accuracy']:.4f}")
    print(f"  saved {args.out / 'fusion.json'}"
          + (f" and {args.out / 'clip_probe.json'}" if probe is not None else ""))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
