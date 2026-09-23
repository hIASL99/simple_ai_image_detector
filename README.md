# aidetect — local detection of AI-generated images

Given an image, decide whether it was produced by a text-to-image model or captured by a camera,
and return a calibrated probability. Everything runs on CPU, and after a one-time model download
nothing touches the network.

Scope: **fully synthetic images** from generators like Stable Diffusion, SDXL, FLUX, Midjourney,
DALL·E, Imagen, Firefly and Ideogram. It is *not* a deepfake-face detector, and AI-upscaled or
AI-retouched photographs are a documented blind spot — see [Limitations](#limitations).

## Results

Leave-one-generator-out across **3,756 images and 26 generators**: every generator is held out in
turn, the calibration is fitted on the rest, and the held-out generator is scored by a model that
has never seen it. This is the number that estimates what happens when the next image model ships.

| | AUROC | balanced acc. | recall @ 5% FPR |
|---|---|---|---|
| **Ensemble (default, 5 members)** | **0.981** | **0.931** | **0.911** |
| Ensemble, permissive licences only | 0.981 | 0.927 | 0.904 |
| Best single detector (Community Forensics) | 0.944 | 0.841 | 0.731 |
| PE-Core linear probe alone | 0.921 | 0.801 | 0.653 |

Fitting on one corpus and testing on a completely independent one (different generators, different
real-image pipeline, different collection date) gives AUROC **0.986** and **0.977** in the two
directions, so the result is not an artefact of one dataset.

Median per-generator recall is 0.97, but the spread is what matters:

| hardest | recall | easiest | recall |
|---|---|---|---|
| Hourglass (pixel diffusion) | 0.30 | SDXL | 1.00 |
| Imagen 3 | 0.74 | Midjourney v4 | 1.00 |
| Ideogram V2 | 0.75 | FLUX.1-schnell | 1.00 |
| Midjourney (GenImage) | 0.79 | Stable Cascade | 1.00 |
| DALL·E 2 / Ideogram V1 | 0.83 | DALL·E 3 | 1.00 |

Hourglass is a pixel-space diffusion model, architecturally unlike everything else in the
benchmark, and it is the clearest illustration of the limitation: held out, the ensemble catches
fewer than a third of its images.

Reproduce with `scripts/report.py`, `scripts/train_fusion.py` and `scripts/ablate_fusion.py`.

## Install

```
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e .
./venv/bin/python scripts/fetch_models.py --ensemble-only    # ~1.5 GB, once
```

Checkpoints land in `./.hf_cache`. Every later run sets `HF_HOME` there and passes
`local_files_only=True`, so the tool works with the network unplugged.

## Use

```
$ ./venv/bin/python -m aidetect examples/
00_real_coco.jpg                REAL          p(ai)= 0.250  [#####...............]  pixel ensemble (clear, 3 models)
02_real_div2k.png               UNCERTAIN     p(ai)= 0.608  [############........]  pixel ensemble is between the operating points (0.53 < p <= 0.74)
04_real_openimages.jpg          REAL          p(ai)= 0.334  [#######.............]  pixel ensemble (clear, 3 models)
05_ai_flux_dev.png              AI-GENERATED  p(ai)= 0.947  [###################.]  pixel ensemble (moderate, 3 models)
08_ai_cf_ai_Dalle3.png          AI-GENERATED  p(ai)= 0.996  [####################]  pixel ensemble (moderate, 3 models)
09_ai_cf_ai_MidjourneyV6_1.png  AI-GENERATED  p(ai)= 0.947  [###################.]  pixel ensemble (moderate, 3 models)

11 image(s): 6 flagged AI-generated, 4 real, 1 uncertain, 0 unreadable
```

On this eleven-image demo set the full ensemble is right eleven times out of eleven. The same set
scored by the strongest *single* model gets 8/11, missing FLUX, Midjourney v6.1 and Imagen 3
outright; the three-classifier ensemble gets 10/11 and abstains on the high-resolution DIV2K
photograph that the probe later resolves correctly.

Useful flags: `--json`, `--verbose` (per-model scores), `-r` (recurse), `--operating-point
{fpr1,fpr5,fpr10,balanced}`, `--no-abstain`, `--metadata-only` (milliseconds, no models),
`--fp32`, `--threads N`.

```python
from aidetect import Detector

d = Detector.load()
print(d.predict("photo.jpg").as_dict())
```

## HTTP API and container

```
podman build -t aidetect .          # or docker build
podman run -p 8000:8000 aidetect
curl -F "files=@photo.jpg" localhost:8000/detect
```

```json
{"results": [{"filename": "photo.jpg", "verdict": "AI-GENERATED", "p_ai": 0.9694,
              "threshold": 0.7115, "low_threshold": 0.5438,
              "basis": "pixel ensemble (moderate, 5 models)",
              "per_model": {"clip-probe": 0.9966, "commforensics384": 0.9377, "...": 0.0},
              "metadata_says_ai": false, "truncated": false, "error": null}],
 "elapsed_ms": 807, "operating_point": "fpr5"}
```

| endpoint | purpose |
|---|---|
| `POST /detect` | multipart upload, one or many files. `?operating_point=fpr1\|fpr5\|fpr10\|balanced`, `?metadata=false` |
| `GET /health` | 200 when the ensemble is loaded, 503 with the reason while it is not |
| `GET /info` | ensemble members and their licences, thresholds, limits |
| `GET /docs` | generated OpenAPI browser |

Without a container: `pip install -e ".[api]"` then
`uvicorn aidetect.api:app --host 0.0.0.0 --port 8000`.

Configuration is environment variables: `AIDETECT_MODEL_DIR` (point it at
`/app/models_permissive` for the commercially usable ensemble),
`AIDETECT_OPERATING_POINT`, `AIDETECT_THREADS` (0 = physical core count),
`AIDETECT_MAX_BYTES`, `AIDETECT_MAX_FILES`.

Notes on the image, all of them load-bearing:

- **The weights are baked in and `HF_HUB_OFFLINE=1`**, so the container never
  reaches the network. Build with `--build-arg BAKE_MODELS=false` for a 1.5 GB
  image instead and mount a populated HF cache at `/opt/models`.
- **Only the files inference reads are fetched.** Several of these checkpoints
  were pushed straight from a training run: `haywoodsloan` carries a 1.5 GB
  `optimizer.pt` and a duplicate checkpoint directory, `Organika` a 694 MB one,
  and `timm` ships a `.bin` duplicating its `.safetensors`. Pulling the repos
  whole costs **9.7 GB**; pulling what is loaded costs **1.7 GB**.
- **One uvicorn worker, and one request scored at a time.** Inference is
  CPU-bound and already uses every core; a second worker halves the threads each
  request gets, finishes no sooner, and doubles peak memory.
- Runs as a non-root user (uid 10001). Uploads are capped at 32 MB and 16 files
  per request before anything reaches the decoder, which has its own
  decompression-bomb limit.
- Podman ignores `HEALTHCHECK` under the default OCI format; add
  `--format docker` if you want it honoured.

## How it works

**1. Provenance metadata, first and cheaply.** A1111 `parameters` blocks, ComfyUI node graphs,
C2PA/JUMBF manifests, IPTC `digitalSourceType`, XMP history, EXIF `Software` and `UserComment`.
Structured markers are near-conclusive and override the pixel score; a bare `Software: Midjourney`
string is one EXIF write away from appearing on a real photo, so it stays a *weak hint* and never
decides. A generative-fill marker yields `AI-EDITED`, not `AI-GENERATED` — a different claim.

The invariant: **metadata can only argue for a conclusion, never against one.** Its absence carries
no information, because every platform strips it. Measured here: across 900 real and AI images from
the benchmark corpora, metadata produced **zero false positives and zero true positives** — the
corpora are stripped. The pixel ensemble does the work; metadata is a free, high-precision bonus on
files that come straight from a generator.

**2. A calibrated ensemble of pixel detectors.** Each member is Platt-scaled on held-out data, then
their log-odds are averaged. Calibrating *before* fusing is the whole trick: several public
checkpoints saturate — their softmax reaches 1.000 on ordinary photographs — and averaging raw
log-odds lets the most over-confident member dominate.

| configuration (leave-one-generator-out) | AUROC | recall @ 5% FPR |
|---|---|---|
| Community Forensics alone | 0.944 | 0.731 |
| + PE-Core linear probe | 0.976 | — |
| + haywoodsloan | 0.979 | — |
| + commforensics224 | 0.981 | — |
| + organika-sdxl | **0.981** | **0.911** |
| three classifiers, **uncalibrated** | 0.959 | 0.798 |
| all fourteen, uncalibrated | 0.922 | 0.695 |

The largest single gain comes from adding a **linear probe on a frozen PE-Core backbone** — a
logistic regression over 768-d embeddings, trained here, with no fine-tuning. It is weaker alone
(0.921) than the best checkpoint, but it fails differently, which is what an ensemble needs. This
matches the 2026 finding that a frozen modern backbone plus a plain linear head beats specialised
detectors, and that fine-tuning the backbone makes generalisation worse.

Members were chosen by greedy forward selection on out-of-fold AUROC — a checkpoint joins only if
it measurably helps — not by download count or reputation.

**3. Thresholds from a target false-positive rate, not 0.5.** Calling a real photo fake is the
expensive error. `fpr5` (the default) puts the cut where 5% of real photos are flagged. Below a
second threshold, 98% of generated images have already been caught, so "real" is a controlled
call; between the two the tool answers **UNCERTAIN** rather than guessing. Truncated files also
return UNCERTAIN: Pillow pads the undecodable tail with flat grey, and the score would describe
the padding.

## The benchmark

3,756 images, 26 generators, two independent tracks:

- **local** — real: COCO val2017, DIV2K, Open Images. AI: FLUX.1 dev/schnell, SDXL, Midjourney v4,
  Google nano-banana, GenImage-Midjourney.
- **cf** — [CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)
  (Park & Owens, CVPR 2025): 20 generators including DALL·E 2/3, Midjourney v5.2/v6.1, Imagen 3,
  Firefly 2/3, Ideogram V1/V2, Stable Cascade, Kandinsky, LCM-LoRA variants, and GAN-era models.

Each image is written three ways, because **scoring raw files measures the container, not the
image**. Real photos arrive as ~600px JPEGs and generator output as 1024px PNGs, so a detector can
score ~95% from file size alone and have learned nothing.

| protocol | what it is | what it tells you |
|---|---|---|
| `native` | the original file, byte for byte | realistic, and fully contaminated by the shortcut |
| `matched` | longest side 512, JPEG q90 | resolution and codec equalised |
| `crop` | 224×224 centre crop at **native** resolution, JPEG q90 | identical dimensions and codec for every source; the honest number |

The two steps between protocols remove different things, and separating them says what each
detector is actually reading. `native -> matched` takes away the container shortcut.
`matched -> crop` takes away global composition and gives back the native-resolution detail that
downscaling destroys.

| detector | native | matched | crop | container | context |
|---|---|---|---|---|---|
| commforensics384 | 0.977 | 0.952 | 0.950 | +0.025 | **+0.002** |
| commforensics224 | 0.952 | 0.908 | 0.906 | +0.044 | +0.002 |
| haywoodsloan | 0.912 | 0.897 | 0.848 | +0.014 | +0.049 |
| smogy | 0.867 | 0.845 | 0.678 | +0.021 | **+0.167** |
| organika-sdxl | 0.793 | 0.794 | 0.791 | −0.001 | +0.003 |
| ateeqq-siglip | 0.780 | 0.777 | 0.719 | +0.004 | +0.057 |
| umm-maybe | 0.563 | 0.528 | 0.497 | +0.035 | +0.031 |

Two things fall out of this. The container shortcut is small in this corpus (≤0.044 for every
detector), so the benchmark is not the file-size test that scoring raw public eval sets can
degenerate into. And the ensemble's two strongest members read **local pixel statistics** — Community
Forensics loses 0.002 AUROC when global composition is removed — whereas `smogy` loses 0.167, so
most of its apparent skill is reading the *scene*, not the generation artefacts. That is a
difference that only shows up if you build the protocols to separate them.

Every headline figure in this README is on `crop`, the strictest protocol.

### Two checkpoints that do not work

Measured, not asserted. On the crop protocol, out-of-fold:

- `umm-maybe/AI-image-detector` — the most-downloaded AI-image detector on the Hub — scores AUROC
  **0.29**, i.e. *anti-correlated* with the truth on modern generators. It was trained in 2022,
  before SDXL, Midjourney v5 and DALL·E 3.
- `dima806/ai_vs_human_generated_image_detection` scores **0.50**, chance. It answers "human" to
  almost everything.

A third, `yaya36095/ai-image-detector`, ships ResNet-50 weights behind a ViT config, so every
parameter is randomly initialised at load. `load_backend()` now refuses any checkpoint with missing
parameters rather than scoring images with noise.

## Robustness: where it actually breaks

`scripts/robustness.py` re-scores the benchmark through the degradations a shared image really
goes through, using the threshold fitted on **clean** data — which is what a deployed detector
uses. 400 images, native protocol, per detector:

**Community Forensics-384**

| degradation | AUROC | recall | FPR |
|---|---|---|---|
| clean | 0.980 | 0.880 | 0.050 |
| JPEG q90 | 0.957 | 0.790 | 0.035 |
| JPEG q70 | 0.949 | 0.730 | 0.020 |
| JPEG q50 | 0.934 | 0.610 | 0.005 |
| **JPEG q30** | 0.899 | **0.400** | **0.010** |
| double JPEG 85→60 | 0.935 | 0.630 | 0.005 |
| WebP q50 | 0.931 | 0.770 | 0.070 |
| screenshot chain | 0.917 | 0.755 | 0.075 |
| downscale ½ | 0.957 | 0.850 | 0.095 |
| blur σ=1 | 0.976 | 0.905 | 0.100 |

Read the last two columns together. At JPEG q30 recall falls from 0.88 to 0.40 while the
false-positive rate falls too, from 0.050 to 0.010. The detector has not become noisy — **the whole
score distribution slides toward "real"**, so a fixed threshold quietly turns it into a detector
that finds almost nothing and still looks well-behaved. AUROC only drops 0.08, which is exactly why
AUROC alone is a misleading headline: the *ranking* mostly survives, the *operating point* does not.
`haywoodsloan` behaves the same way (recall 0.71 → 0.40 at q30). `organika-sdxl` is the exception —
flat, even slightly better under compression — which is the diversity argument for keeping a weaker
member in the ensemble.

Practical consequence: **on heavily re-compressed images, lower the threshold or treat the answer
as UNCERTAIN.** `--operating-point fpr10` recovers some recall at a known cost. Do not run the
fpr1 operating point on social-media images and conclude there is nothing there.

## Reproducing

```
./venv/bin/python scripts/build_bench.py --per-source 250   # needs data/raw; see scripts/
./venv/bin/python scripts/score_bench.py --bf16 --threads 11
./venv/bin/python scripts/train_fusion.py --protocol crop
./venv/bin/python scripts/report.py
./venv/bin/python scripts/ablate_fusion.py
./venv/bin/python scripts/robustness.py --limit 400
./venv/bin/python scripts/check_probe_leakage.py
```

`check_probe_leakage.py` exists because the probe is evaluated from a cached out-of-fold score
column rather than refitted inside every candidate evaluation — the shortcut that turns hours of
ensemble selection into seconds. Each row's probe score is clean for the fold that tests it, but
the Platt calibrator for that fold is fitted on scores from probes that did see the test rows, so
there is a two-parameter channel. Refitting the probe strictly inside every fold gives an
**identical AUROC of 0.9813** and a TPR 0.006 lower, so the shortcut is sound.

Scoring is the expensive step — hours of CPU for the full cross-product — and is cached per
(checkpoint, protocol) in `data/scores/`.

Tests: `./venv/bin/python -m pytest` (268 tests, no network, no model downloads; a socket guard
fails any test that tries). `--runslow` additionally runs the checkpoint-dependent tests.

## Limitations

Read this section before trusting any number above.

- **Unfamiliar architectures are harder.** Published surveys covering 291 generators report mean
  detector accuracy falling from ~79% on 2020–21 models to ~38% on 2024 models. Held out, this
  ensemble catches 0.30 of Hourglass (pixel-space diffusion) and 0.74–0.83 of Imagen 3, Ideogram
  and DALL·E 2, against 1.00 for the latent-diffusion family it has seen most of. A generator
  released after this was calibrated may do worse than anything in the table.
- **Re-encoding destroys the operating point, silently and asymmetrically.** Measured here, not
  quoted: at JPEG q30 the best member's recall falls 0.88 → 0.40 while its false-positive rate
  falls 0.050 → 0.010. See [Robustness](#robustness-where-it-actually-breaks). Published figures
  for frequency-based detectors are far worse (fake-recall 0.0–1.5%). Run
  `scripts/robustness.py` on your own data before deploying on re-shared images.
- **Photo-of-a-screen and print recapture are near coin-flip** (~0.55 for the best published
  methods).
- **A real photo passed through an AI pipeline** — upscaled, denoised, VAE round-tripped,
  generatively retouched — is a known blind spot for frozen-feature detectors. Metadata may catch
  it (`AI-EDITED`); the pixel ensemble will often not.
- **The benchmark is not the world.** 250 images per source, drawn from public corpora that were
  re-encoded when they were assembled. Per-generator recalls come with Wilson intervals roughly
  ±0.12 at n=60 — see `reports/benchmark.md`.
- **bfloat16 by default.** ~2.5× faster on this CPU and what the shipped calibration was fitted
  under. `--fp32` is available; the detector warns if the numeric mode does not match the
  calibration.

**Do not** use a single score to accuse anyone of anything, to make an automated moderation
decision, or as evidence. A 5% false-positive rate means one real photograph in twenty is flagged.
Use `--operating-point fpr1` when a false accusation is costly, treat UNCERTAIN as the answer it
is, and keep a human in the loop.

## Licensing

The code here is yours to use. The checkpoints are not uniformly permissive:

| checkpoint | licence | in default ensemble |
|---|---|---|
| `OwensLab/commfor-model-384` | MIT | yes |
| `haywoodsloan/ai-image-detector-deploy` | Apache-2.0 | yes |
| `Organika/sdxl-detector` | **CC-BY-NC-3.0** | yes (and only just) |
| `timm/vit_pe_core_base_patch16_224.fb` (probe backbone) | Apache-2.0 | yes |
| `OwensLab/commfor-model-224` | MIT | yes |
| `Smogy/SMOGY-Ai-images-detector` | **CC-BY-NC-4.0** | no |
| `Ateeqq/…`, `NYUAD-ComNets/…`, `mmanikanta/…` | Apache-2.0 | no |

**The default ensemble is therefore non-commercial**, because of `organika-sdxl`. For commercial
use load `models_permissive/` instead — MIT + Apache-2.0 only, and it costs almost nothing:
AUROC 0.9805 against 0.9813, recall 0.904 against 0.911. `organika-sdxl` is the last member greedy
selection added and it barely earns its place, so dropping it is close to free.

```
./venv/bin/python -m aidetect IMAGE --model-dir models_permissive
```

CommunityForensics-Eval is CC-BY-NC-SA-4.0: local evaluation only, do not redistribute the images.
COCO, Open Images and DIV2K carry their own upstream terms.

## Credits

- Community Forensics — Park & Owens, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125)).
- The calibrate-then-fuse and target-FPR design follows the 2024–2026 literature on detector
  robustness; the crop protocol follows the "crop, don't resize" finding, which preserves the
  high-frequency evidence that downscaling discards.
