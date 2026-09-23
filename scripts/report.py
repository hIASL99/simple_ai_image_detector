#!/usr/bin/env python
"""Turn the cached scores in data/scores/ into a readable report.

Reads whatever is present, so it can be run while the sweep is still going.
Three things are reported and they answer different questions:

  per-backend, per-protocol   which checkpoints actually work, and which are
                              broken or inverted
  native vs crop              how much of a backend's score comes from the
                              container rather than the image
  per-generator               where the ensemble fails, which is the only way
                              to say anything honest about a generator nobody
                              has tested yet
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidetect.calibration import metrics, threshold_at_fpr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
SCORES = ROOT / "data" / "scores"
PROTOCOLS = ("native", "matched", "crop")


def load_manifest(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_score_files(score_dir: Path, ids: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    index = {v: i for i, v in enumerate(ids)}
    out: dict[tuple[str, str], np.ndarray] = {}
    for f in sorted(score_dir.glob("*__*.npz")):
        stem = f.name[: -len(".npz")]
        name, _, protocol = stem.partition("__")
        if name.startswith("clipfeat") or not protocol:
            continue
        data = np.load(f, allow_pickle=False)
        if "scores" not in data:
            continue
        aligned = np.full(len(ids), np.nan)
        for got, score in zip(data["ids"], data["scores"]):
            pos = index.get(str(got))
            if pos is not None:
                aligned[pos] = score
        out[(name, protocol)] = aligned
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- honest at the small per-generator sample sizes."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(x: float, nd: int = 3) -> str:
    return "   -  " if x != x else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=BENCH / "manifest.csv")
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--track", default=None, help="restrict to one benchmark track (cf | local)")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "benchmark.md")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if args.track:
        rows = [r for r in rows if r["track"] == args.track]
    ids = np.array([r["id"] for r in rows])
    labels = np.array([1 if r["label"] == "ai" else 0 for r in rows])
    tracks = np.array([r["track"] for r in rows])
    generators = np.array([r["generator"] for r in rows])

    scores = load_score_files(args.scores, ids)
    if not scores:
        print(f"no cached scores in {args.scores}")
        return 1
    backends = sorted({n for n, _ in scores})
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Benchmark report")
    emit()
    emit(f"- images: {len(rows)} ({int((labels == 0).sum())} real / {int(labels.sum())} AI)")
    emit(f"- generators: {len({g for g, l in zip(generators, labels) if l == 1})}")
    emit(f"- tracks: {', '.join(sorted(set(tracks)))}")
    emit(f"- operating point: threshold set to {args.target_fpr:.0%} false-positive rate "
         f"on this same data (in-sample; see the fusion report for held-out numbers)")
    emit()

    emit("## Per-detector, per-protocol")
    emit()
    emit("AUROC is threshold-free. `bAcc` is balanced accuracy at a threshold fitted for the "
         "target FPR. `TPR@fpr` is the share of AI images caught at that operating point.")
    emit()
    header = f"| {'detector':22s} | protocol | n    | AUROC | bAcc  | TPR@fpr | FPR   |"
    emit(header)
    emit("|" + "-" * 24 + "|----------|------|-------|-------|---------|-------|")
    auroc_table: dict[str, dict[str, float]] = defaultdict(dict)
    for name in backends:
        for protocol in PROTOCOLS:
            s = scores.get((name, protocol))
            if s is None:
                continue
            ok = ~np.isnan(s)
            if ok.sum() < 20 or len(np.unique(labels[ok])) < 2:
                continue
            thr = threshold_at_fpr(s[ok], labels[ok], args.target_fpr)
            m = metrics(s[ok], labels[ok], thr)
            auroc_table[name][protocol] = m["auroc"]
            emit(f"| {name:22s} | {protocol:8s} | {int(ok.sum()):4d} | {fmt(m['auroc'])} | "
                 f"{fmt(m['balanced_accuracy'])} | {fmt(m['tpr']):7s} | {fmt(m['fpr'])} |")
    emit()

    emit("## What each detector is actually reading")
    emit()
    emit("The two steps between the protocols remove different things, so the two gaps mean "
         "different things.")
    emit()
    emit("`native -> matched` equalises resolution and codec across every source. What it "
         "removes is the container shortcut -- the chance to score well by noticing that real "
         "photos arrive as small JPEGs and generator output as large PNGs.")
    emit()
    emit("`matched -> crop` swaps a downscaled whole image for a native-resolution window. It "
         "takes away global composition and gives back the high-frequency detail that "
         "downscaling destroys. A detector that reads local pixel statistics barely notices; one "
         "that reads the scene loses a lot.")
    emit()
    emit(f"| {'detector':22s} | native | matched | crop  | container | context |")
    emit("|" + "-" * 24 + "|--------|---------|-------|-----------|---------|")
    for name in backends:
        t = auroc_table.get(name, {})
        if not {"native", "matched", "crop"} <= set(t):
            continue
        emit(f"| {name:22s} | {fmt(t['native'])}  | {fmt(t['matched'])}   | {fmt(t['crop'])} | "
             f"{t['native'] - t['matched']:+.3f}    | {t['matched'] - t['crop']:+.3f}  |")
    emit()

    best = max(backends, key=lambda n: auroc_table.get(n, {}).get("crop", 0.0), default=None)
    if best:
        emit(f"## Per-generator recall -- {best}, crop protocol")
        emit()
        emit("Threshold fitted once on all real images at the target FPR, then applied per "
             "generator. Intervals are Wilson 95%.")
        emit()
        s = scores.get((best, "crop"))
        if s is not None:
            ok = ~np.isnan(s)
            thr = threshold_at_fpr(s[ok], labels[ok], args.target_fpr)
            emit(f"| {'generator':28s} | n   | recall | 95% CI          |")
            emit("|" + "-" * 30 + "|-----|--------|-----------------|")
            for gen in sorted({g for g, l in zip(generators, labels) if l == 1}):
                mask = ok & (generators == gen) & (labels == 1)
                n = int(mask.sum())
                if not n:
                    continue
                k = int((s[mask] >= thr).sum())
                lo, hi = wilson(k, n)
                emit(f"| {gen:28s} | {n:3d} | {k / n:.3f}  | [{lo:.3f}, {hi:.3f}] |")
            emit()
            for src in sorted({r["source"] for r in rows if r["label"] == "real"}):
                mask = ok & np.array([r["source"] == src for r in rows]) & (labels == 0)
                n = int(mask.sum())
                if n:
                    fp = int((s[mask] >= thr).sum())
                    emit(f"- real source `{src}`: {n} images, {fp} wrongly flagged "
                         f"({fp / n:.1%})")
            emit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
