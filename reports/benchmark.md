# Benchmark report

- images: 3756 (1122 real / 2634 AI)
- generators: 26
- tracks: cf, local
- operating point: threshold set to 5% false-positive rate on this same data (in-sample; see the fusion report for held-out numbers)

## Per-detector, per-protocol

AUROC is threshold-free. `bAcc` is balanced accuracy at a threshold fitted for the target FPR. `TPR@fpr` is the share of AI images caught at that operating point.

| detector               | protocol | n    | AUROC | bAcc  | TPR@fpr | FPR   |
|------------------------|----------|------|-------|-------|---------|-------|
| aiornot-siglip2        | crop     | 3756 | 0.570 | 0.527 | 0.103   | 0.050 |
| ateeqq-siglip          | native   | 3756 | 0.780 | 0.643 | 0.336   | 0.050 |
| ateeqq-siglip          | matched  | 3756 | 0.777 | 0.642 | 0.333   | 0.050 |
| ateeqq-siglip          | crop     | 3756 | 0.719 | 0.566 | 0.182   | 0.050 |
| commforensics224       | native   | 3756 | 0.952 | 0.863 | 0.777   | 0.050 |
| commforensics224       | matched  | 3756 | 0.908 | 0.816 | 0.683   | 0.050 |
| commforensics224       | crop     | 3756 | 0.906 | 0.797 | 0.643   | 0.050 |
| commforensics384       | native   | 3756 | 0.977 | 0.911 | 0.872   | 0.050 |
| commforensics384       | matched  | 3756 | 0.952 | 0.865 | 0.781   | 0.051 |
| commforensics384       | crop     | 3756 | 0.950 | 0.844 | 0.737   | 0.050 |
| dima806-vit            | crop     | 3756 | 0.558 | 0.527 | 0.103   | 0.050 |
| haywoodsloan           | native   | 3756 | 0.912 | 0.840 | 0.730   | 0.050 |
| haywoodsloan           | matched  | 3756 | 0.897 | 0.822 | 0.694   | 0.050 |
| haywoodsloan           | crop     | 3756 | 0.848 | 0.751 | 0.552   | 0.050 |
| mmanikanta-convnext    | crop     | 3756 | 0.553 | 0.533 | 0.116   | 0.050 |
| mmanikanta-swin        | crop     | 3756 | 0.613 | 0.534 | 0.118   | 0.050 |
| nyuad                  | crop     | 3756 | 0.561 | 0.535 | 0.119   | 0.050 |
| organika-sdxl          | native   | 3756 | 0.793 | 0.659 | 0.368   | 0.050 |
| organika-sdxl          | matched  | 3756 | 0.794 | 0.644 | 0.338   | 0.050 |
| organika-sdxl          | crop     | 3756 | 0.791 | 0.674 | 0.398   | 0.050 |
| prithiv-deepfake       | crop     | 3756 | 0.502 | 0.493 | 0.037   | 0.050 |
| sadra-sdxl             | crop     | 3756 | 0.696 | 0.527 | 0.105   | 0.051 |
| smogy                  | native   | 3756 | 0.867 | 0.760 | 0.571   | 0.050 |
| smogy                  | matched  | 3756 | 0.845 | 0.736 | 0.521   | 0.050 |
| smogy                  | crop     | 3756 | 0.678 | 0.570 | 0.191   | 0.050 |
| umm-maybe              | native   | 3756 | 0.563 | 0.523 | 0.095   | 0.050 |
| umm-maybe              | matched  | 3756 | 0.528 | 0.520 | 0.091   | 0.050 |
| umm-maybe              | crop     | 3756 | 0.497 | 0.522 | 0.093   | 0.050 |

## What each detector is actually reading

The two steps between the protocols remove different things, so the two gaps mean different things.

`native -> matched` equalises resolution and codec across every source. What it removes is the container shortcut -- the chance to score well by noticing that real photos arrive as small JPEGs and generator output as large PNGs.

`matched -> crop` swaps a downscaled whole image for a native-resolution window. It takes away global composition and gives back the high-frequency detail that downscaling destroys. A detector that reads local pixel statistics barely notices; one that reads the scene loses a lot.

| detector               | native | matched | crop  | container | context |
|------------------------|--------|---------|-------|-----------|---------|
| ateeqq-siglip          | 0.780  | 0.777   | 0.719 | +0.004    | +0.057  |
| commforensics224       | 0.952  | 0.908   | 0.906 | +0.044    | +0.002  |
| commforensics384       | 0.977  | 0.952   | 0.950 | +0.025    | +0.002  |
| haywoodsloan           | 0.912  | 0.897   | 0.848 | +0.014    | +0.049  |
| organika-sdxl          | 0.793  | 0.794   | 0.791 | -0.001    | +0.003  |
| smogy                  | 0.867  | 0.845   | 0.678 | +0.021    | +0.167  |
| umm-maybe              | 0.563  | 0.528   | 0.497 | +0.035    | +0.031  |

## Per-generator recall -- commforensics384, crop protocol

Threshold fitted once on all real images at the target FPR, then applied per generator. Intervals are Wilson 95%.

| generator                    | n   | recall | 95% CI          |
|------------------------------|-----|--------|-----------------|
| cf:DFGAN                     | 116 | 0.897  | [0.828, 0.940] |
| cf:Dalle2                    |  60 | 0.717  | [0.592, 0.815] |
| cf:Dalle3                    |  38 | 0.974  | [0.865, 0.995] |
| cf:DeciDiffusionV2           |  60 | 0.933  | [0.841, 0.974] |
| cf:FLUX-dev                  |  61 | 0.902  | [0.802, 0.954] |
| cf:FLUX-schnell              |  60 | 0.883  | [0.778, 0.942] |
| cf:Firefly_Image2            |  60 | 1.000  | [0.940, 1.000] |
| cf:Firefly_Image3            |  60 | 0.917  | [0.819, 0.964] |
| cf:GALIP                     |  60 | 0.967  | [0.886, 0.991] |
| cf:Hourglass                 |  60 | 0.267  | [0.171, 0.390] |
| cf:IdeogramV1                |  60 | 0.533  | [0.409, 0.654] |
| cf:IdeogramV2                |  60 | 0.317  | [0.213, 0.442] |
| cf:Imagen3                   |  61 | 0.557  | [0.433, 0.675] |
| cf:LCM_lora_sdv15            |  58 | 0.966  | [0.883, 0.990] |
| cf:LCM_lora_sdxl             |  60 | 0.683  | [0.558, 0.787] |
| cf:LCM_lora_ssd1b            |  60 | 0.850  | [0.739, 0.919] |
| cf:MidjourneyV5_2            |  60 | 0.950  | [0.863, 0.983] |
| cf:MidjourneyV6_1            |  60 | 0.667  | [0.541, 0.773] |
| cf:kandinsky_2_2             |  60 | 0.783  | [0.664, 0.869] |
| cf:stable_cascade            |  60 | 0.983  | [0.911, 0.997] |
| flux.1-dev                   | 200 | 0.505  | [0.436, 0.574] |
| flux.1-schnell               | 200 | 0.870  | [0.816, 0.910] |
| midjourney-genimage          | 250 | 0.868  | [0.820, 0.904] |
| midjourney-v4                | 250 | 0.912  | [0.870, 0.941] |
| nano-banana                  | 250 | 0.200  | [0.155, 0.254] |
| sdxl-base-1.0                | 250 | 0.796  | [0.742, 0.841] |

- real source `cf_real_src-LAION`: 250 images, 9 wrongly flagged (3.6%)
- real source `cf_real_src-coco`: 63 images, 1 wrongly flagged (1.6%)
- real source `cf_real_src-imagenet`: 59 images, 1 wrongly flagged (1.7%)
- real source `coco`: 250 images, 2 wrongly flagged (0.8%)
- real source `div2k`: 250 images, 25 wrongly flagged (10.0%)
- real source `openimages`: 250 images, 18 wrongly flagged (7.2%)

