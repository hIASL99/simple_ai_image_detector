#!/usr/bin/env python
"""Score the benchmark under degradation and show where each detector breaks.

Clean-benchmark accuracy is the number everyone publishes and the number that
means least: the images people actually ask about have been through a
messenger, a resize, or a screenshot. The interesting column here is not
AUROC -- ranking often survives -- but accuracy at the threshold fitted on
CLEAN data, because that is what a deployed detector uses. When a degradation
shifts the whole score distribution downwards, that fixed threshold turns the
detector into one that calls everything real: FPR goes to 0, TPR goes to 0,
and accuracy stays respectable because half the set is real. Reading TPR and
FPR together is the only way to see it happen.

Scores are cached per image id, so re-running with a larger --limit only pays
for the images that are new.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidetect.backends import REGISTRY as BACKENDS, default_backend_names, load_backend  # noqa: E402
from aidetect.calibration import metrics, threshold_at_fpr  # noqa: E402
from aidetect.degrade import DEFAULT_SUITE, REGISTRY as DEGRADATIONS, get as get_degradation  # noqa: E402
from aidetect.imageio import load_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
SCORES = ROOT / "data" / "scores_robust"
REPORTS = ROOT / "reports"
PROTOCOLS = ("native", "matched", "crop")


def read_manifest(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _allocate(sizes: dict[str, int], budget: int) -> dict[str, int]:
    """Largest-remainder split of `budget` across groups, proportional to size."""
    total = sum(sizes.values())
    budget = min(budget, total)
    if total == 0:
        return {}
    quota = {k: budget * sizes[k] / total for k in sizes}
    take = {k: min(sizes[k], int(quota[k])) for k in sizes}
    order = sorted(sizes, key=lambda k: (-(quota[k] - int(quota[k])), k))
    i = 0
    while sum(take.values()) < budget:
        k = order[i % len(order)]
        if take[k] < sizes[k]:
            take[k] += 1
        i += 1
    return {k: v for k, v in take.items() if v > 0}


def stratified_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """A reproducible subset, balanced real/ai and spread over the sources.

    Balanced rather than proportional because both the threshold and the FPR
    it is fitted to are estimated from the real half; a subset that inherits
    the corpus's 2:1 AI skew spends its budget on the half that matters less.
    With a limit below the number of sources the biggest sources take the
    slots, so not every generator is represented -- adequate for a smoke test,
    not for a headline number.
    """
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for label, budget in (("real", limit // 2), ("ai", limit - limit // 2)):
        pool: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["label"] == label:
                pool[r["source"]].append(r)
        for src, n in _allocate({k: len(v) for k, v in pool.items()}, budget).items():
            candidates = sorted(pool[src], key=lambda r: r["id"])
            picked = rng.permutation(len(candidates))[:n]
            out.extend(candidates[i] for i in sorted(picked))
    return sorted(out, key=lambda r: r["id"])


def _progress(tag: str):
    start = time.time()
    state = {"last": 0.0}

    def report(done: int, total: int) -> None:
        now = time.time()
        if now - state["last"] < 5 and done < total:
            return
        state["last"] = now
        rate = done / max(now - start, 1e-9)
        print(f"\r    {tag}: {done}/{total}  {rate:5.2f} img/s"
              f"  eta {(total - done) / max(rate, 1e-9):5.0f}s", end="", flush=True)
        if done >= total:
            print()
    return report


def score_degraded(backend, paths, degradation, batch_size: int, progress=None) -> np.ndarray:
    """Load, degrade, score. The degradation happens after decoding, so the
    protocol's own encoding is part of the input exactly as it is in a clean run."""
    out = []
    for i in range(0, len(paths), batch_size):
        batch = [degradation(load_rgb(p)) for p in paths[i:i + batch_size]]
        out.append(backend.score_batch(batch))
        if progress is not None:
            progress(min(i + batch_size, len(paths)), len(paths))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {str(i): float(s) for i, s in zip(data["ids"], data["scores"])}


def save_cache(path: Path, table: dict[str, float]) -> None:
    ids = np.array(sorted(table))
    np.savez_compressed(path, ids=ids,
                        scores=np.array([table[i] for i in ids], dtype=np.float64))


def scores_for(backend, rows, protocol: str, name: str, dest: Path,
               batch_size: int, force: bool) -> np.ndarray:
    """Cached scores for `rows`, computing only what is missing.

    The cache is keyed by image id rather than by the subset, so a run with a
    bigger --limit reuses everything an earlier run paid for.
    """
    table = {} if force else load_cache(dest)
    todo = [r for r in rows if r["id"] not in table]
    if todo:
        degradation = get_degradation(name)
        t0 = time.time()
        fresh = score_degraded(backend, [ROOT / r[protocol] for r in todo], degradation,
                               batch_size, _progress(f"{backend.name}/{protocol}/{name}"))
        if fresh.size != len(todo) or not np.isfinite(fresh).all():
            raise RuntimeError(f"{backend.name}/{name}: got {fresh.size} usable scores "
                               f"for {len(todo)} images")
        table.update(zip((r["id"] for r in todo), fresh.tolist()))
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_cache(dest, table)
        print(f"    +{len(todo)} scores in {time.time() - t0:.0f}s -> {dest.name}")
    else:
        print(f"    cached {dest.name}")
    return np.array([table[r["id"]] for r in rows], dtype=np.float64)


def resolve_backends(names: list[str]) -> list[str]:
    if names == ["all"]:
        return sorted(BACKENDS)
    if names == ["default"]:
        return default_backend_names()
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s): {unknown}; known: {sorted(BACKENDS)}")
    return names


def resolve_degradations(names: list[str]) -> list[str]:
    chosen = sorted(DEGRADATIONS) if names == ["all"] else list(names)
    unknown = [n for n in chosen if n not in DEGRADATIONS]
    if unknown:
        raise SystemExit(f"unknown degradation(s): {unknown}; known: {sorted(DEGRADATIONS)}")
    # The clean pass defines the threshold every other row is judged at, so it
    # is never optional.
    return ["clean"] + [n for n in chosen if n != "clean"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=BENCH / "manifest.csv")
    ap.add_argument("--out", type=Path, default=SCORES)
    ap.add_argument("--protocol", default="matched", choices=PROTOCOLS)
    ap.add_argument("--backends", nargs="+", default=["commforensics224"],
                    help="'all', 'default', or explicit names")
    ap.add_argument("--degradations", nargs="+", default=list(DEFAULT_SUITE),
                    help="'all' or explicit names; 'clean' is always included")
    ap.add_argument("--limit", type=int, default=400,
                    help="images per run, stratified; 0 means the whole benchmark "
                         "(backends x degradations x 3756 images is hours of CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="false-positive rate the clean threshold is fitted to")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4,
                    help="deliberately low: this sweep is meant to run beside other work")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--report", type=Path, default=None,
                    help=f"JSON output; default {REPORTS}/robustness__<protocol>.json")
    args = ap.parse_args()

    rows = stratified_subset(read_manifest(args.manifest), args.limit, args.seed)
    if not rows:
        raise SystemExit("no rows selected")
    labels = np.array([1 if r["label"] == "ai" else 0 for r in rows])
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        raise SystemExit("subset has only one class; raise --limit")
    names = resolve_backends(args.backends)
    degradations = resolve_degradations(args.degradations)
    by_source = defaultdict(int)
    for r in rows:
        by_source[f"{r['label']}/{r['source']}"] += 1

    print(f"{len(rows)} images ({int(labels.sum())} ai / {int((labels == 0).sum())} real) "
          f"from {len(by_source)} sources | protocol {args.protocol}")
    print(f"backends: {', '.join(names)}")
    print(f"degradations: {', '.join(degradations)}")

    report = {"protocol": args.protocol, "n": len(rows), "limit": args.limit,
              "seed": args.seed, "target_fpr": args.target_fpr,
              "sources": dict(sorted(by_source.items())),
              "degradations": {n: get_degradation(n).note for n in degradations},
              "backends": {}}

    for name in names:
        print(f"\n{name}")
        try:
            backend = load_backend(name, local_files_only=True, num_threads=args.threads)
        except Exception as exc:  # noqa: BLE001 - a broken checkpoint must not stop the sweep
            print(f"  FAILED to load: {type(exc).__name__}: {exc}")
            continue

        per_degradation: dict[str, np.ndarray] = {}
        for deg in degradations:
            dest = args.out / f"{name}__{args.protocol}__{deg}.npz"
            try:
                per_degradation[deg] = scores_for(backend, rows, args.protocol, deg, dest,
                                                  args.batch_size, args.force)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {deg}: {type(exc).__name__}: {exc}")

        if "clean" not in per_degradation:
            print("  no clean scores; cannot fit a threshold")
            continue
        clean = per_degradation["clean"]
        threshold = threshold_at_fpr(clean, labels, args.target_fpr)
        base = metrics(clean, labels, threshold)

        # The threshold is fitted and evaluated on the same clean scores, which
        # flatters the clean row. That is deliberate: it makes the clean row the
        # best case, so every drop below it is a floor on the real damage.
        print(f"  threshold {threshold:.6f} fitted on clean at FPR<={args.target_fpr:.0%} "
              f"(clean AUROC {base['auroc']:.4f})")
        # floor(fpr * n_real) negatives are allowed above the threshold, so on a
        # small subset the operating point rests on a single image -- and if that
        # image sits in a softmax's saturated tail there is no score resolution
        # around it either. Both make the acc/TPR/dTPR columns coarse in a way
        # AUROC never shows.
        allowed = int(np.floor(args.target_fpr * int((labels == 0).sum())))
        if allowed <= 1:
            print(f"  warning: at FPR<={args.target_fpr:.0%} only {allowed} of the "
                  f"{int((labels == 0).sum())} real images may sit above the threshold, so the "
                  f"operating point rests on one image; raise --limit before quoting it")
        if threshold > 1 - 1e-4:
            print("  warning: the threshold is inside this backend's saturated softmax tail, "
                  "where the scores carry no usable resolution")
        print(f"  {'degradation':<20}{'AUROC':>8}{'dAUROC':>9}{'acc':>8}{'TPR':>8}{'FPR':>8}"
              f"{'dTPR':>9}{'mean_ai':>9}{'acc@0.5':>9}{'TPR@0.5':>9}")
        rows_out = {}
        for deg, scores in per_degradation.items():
            m = metrics(scores, labels, threshold)
            m_half = metrics(scores, labels, 0.5)
            # The mean score on the AI half is the early warning: it falls
            # before the threshold crossings do, and it is what turns a
            # detector into one that answers "real" to everything.
            mean_ai = float(scores[labels == 1].mean())
            rows_out[deg] = {"at_clean_threshold": m, "at_half": m_half,
                             "auroc_drop": base["auroc"] - m["auroc"],
                             "tpr_drop": base["tpr"] - m["tpr"],
                             "mean_score": float(scores.mean()), "mean_score_ai": mean_ai,
                             "mean_score_real": float(scores[labels == 0].mean())}
            print(f"  {deg:<20}{m['auroc']:>8.4f}{m['auroc'] - base['auroc']:>+9.4f}"
                  f"{m['accuracy']:>8.3f}{m['tpr']:>8.3f}{m['fpr']:>8.3f}"
                  f"{m['tpr'] - base['tpr']:>+9.3f}{mean_ai:>9.3f}"
                  f"{m_half['accuracy']:>9.3f}{m_half['tpr']:>9.3f}")
        report["backends"][name] = {"threshold": float(threshold), "degradations": rows_out}

    dest = args.report or (REPORTS / f"robustness__{args.protocol}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
