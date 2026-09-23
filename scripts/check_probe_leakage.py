"""Does reusing one out-of-fold probe column leak into the held-out estimate?

Row i's probe score is clean for the fold that tests it, but the Platt
calibrator fitted on that fold's training rows sees scores from probes that did
look at the test rows. That is a two-parameter channel. Two checks: refit the
probe strictly inside every fold (the expensive, correct thing), and compare.
"""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aidetect.calibration import PlattCalibrator, logit, metrics, sigmoid, threshold_at_fpr
from aidetect.probe import LinearProbe
import train_fusion as tf
R = Path(__file__).resolve().parent.parent
rows=tf.load_manifest(R/'data/bench/manifest.csv')
ids=np.array([r['id'] for r in rows]); labels=np.array([1 if r['label']=='ai' else 0 for r in rows])
gens=np.array([r['generator'] for r in rows])
scores=tf.load_scores('crop', R/'data/scores', ids)
feats=tf.load_features('crop', R/'data/scores', ids, 1, 'pe-core-b16-224')
CLASSIFIERS=['commforensics384','haywoodsloan','commforensics224','organika-sdxl']
SEED=20260922

def run(strict):
    oof=np.full(len(labels), np.nan)
    for gen, tr, te in tf.logo_folds(gens, labels, SEED):
        if labels[tr].sum()==0 or (labels[tr]==0).sum()==0: continue
        cols={n:scores[n] for n in CLASSIFIERS}
        if strict:
            # Probe fitted only on this fold's training rows, and the calibrator
            # fitted on that same probe's in-fold predictions. No channel at all.
            pr=LinearProbe.fit(feats[tr], labels[tr], C=1.0)
            cols['probe']=pr(feats)
        else:
            cols['probe']=scores['probe-oof']
        cals={n:PlattCalibrator.fit(cols[n][tr], labels[tr]) for n in cols}
        z=np.mean([logit(cals[n](cols[n])) for n in cols], axis=0)
        oof[te]=sigmoid(z)[te]
    ok=~np.isnan(oof)
    thr=threshold_at_fpr(oof[ok], labels[ok], 0.05)
    return metrics(oof[ok], labels[ok], thr)

for strict in (False, True):
    t=time.time(); m=run(strict)
    tag="strict: probe refitted inside every fold" if strict else "cached out-of-fold probe column"
    print(f"{tag:44s} AUROC={m['auroc']:.4f} bAcc={m['balanced_accuracy']:.4f} TPR={m['tpr']:.3f} ({time.time()-t:.0f}s)")
