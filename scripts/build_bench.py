#!/usr/bin/env python
"""Assemble the benchmark corpus from the raw downloads in data/raw.

Every source of real images and every generator has its own resolution and
codec fingerprint. A detector can score beautifully by learning "1024px PNG ==
fake" and tell you nothing about authenticity, so the corpus is written three
ways and results are reported for all three:

  native   verbatim original files. Realistic, and the shortcut is fully
           available -- treat a high score here as an upper bound.
  matched  longest side scaled to 512, re-encoded JPEG q=90 4:4:4. Resolution
           and codec are equalised; whole-image semantics survive.
  crop     a 320x320 centre crop taken at *native* resolution, JPEG q=90 4:4:4.
           Pixel dimensions and codec are identical across every source and the
           native high-frequency statistics are untouched, so this is the
           honest test of whether a detector reads the image rather than its
           container. It does cost the semantic detectors their global context.

The gap between native and crop is the size of the shortcut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidetect.imageio import ImageLoadError, iter_image_files, open_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BENCH = ROOT / "data" / "bench"

MATCHED_MAX_SIDE = 512
# 224 is both the input size of most of the backbones and small enough that
# 256px GAN output survives the crop, which keeps the GAN-era generators in
# the benchmark instead of silently dropping them.
CROP_SIDE = 224
JPEG_QUALITY = 90

# track, name, label, generator, kind, location
#
# Two independent tracks. `local` is assembled here from primary sources;
# `cf` is the CommunityForensics-Eval benchmark (Park & Owens, CVPR 2025).
# Keeping them separate makes a genuinely held-out test possible: fit the
# calibration on one track, report the numbers on the other.
BASE_SOURCES = [
    ("local", "coco",         "real", "camera:coco-val2017",   "dir",     RAW / "coco" / "val2017"),
    ("local", "div2k",        "real", "camera:div2k-hr",       "dir",     RAW / "div2k"),
    ("local", "openimages",   "real", "camera:open-images-v7", "parquet", RAW / "parquet" / "openimages.parquet"),
    ("local", "midjourney",   "ai",   "midjourney-v4",         "dir",     RAW / "midjourney"),
    ("local", "flux_dev",     "ai",   "flux.1-dev",            "dir",     RAW / "flux_dev"),
    ("local", "flux_schnell", "ai",   "flux.1-schnell",        "dir",     RAW / "flux_schnell"),
    ("local", "sdxl",         "ai",   "sdxl-base-1.0",         "parquet", RAW / "parquet" / "sdxl.parquet"),
    ("local", "nanobanana",   "ai",   "nano-banana",           "parquet", RAW / "parquet" / "nanobanana.parquet"),
    ("local", "genimage_mj",  "ai",   "midjourney-genimage",   "parquet", RAW / "parquet" / "genimage_mj.parquet"),
]

CF_ROOT = RAW / "commforensics"


def community_forensics_sources() -> list[tuple]:
    """One source per (label, generator) group in the CommunityForensics pull.

    Splitting by generator rather than lumping the whole set together is what
    makes leave-one-generator-out possible downstream.
    """
    meta = CF_ROOT / "meta.jsonl"
    if not meta.exists():
        return []
    groups: dict[tuple[str, str], list[Path]] = {}
    for line in meta.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        label = "ai" if rec["label"] == 1 else "real"
        model = str(rec.get("model") or "unknown").replace("/", "-")
        # For a real row `model` names the generator it was *paired with*, not
        # how the photo was made, so real rows are grouped by their true source.
        key = model if label == "ai" else f"src-{rec.get('real_source') or 'unknown'}"
        groups.setdefault((label, key), []).append(CF_ROOT / rec["path"])
    out = []
    for (label, key), paths in sorted(groups.items()):
        prefix = "cf_ai" if label == "ai" else "cf_real"
        out.append(("cf", f"{prefix}_{key}", label,
                    (f"cf:{key}" if label == "ai" else f"camera:{key}"), "files", paths))
    return out


SOURCES = BASE_SOURCES


def _parquet_images(path: Path, limit: int):
    """Yield raw image bytes from an HF-style parquet shard, a row group at a time."""
    f = pq.ParquetFile(path)
    seen = 0
    for rg in range(f.metadata.num_row_groups):
        for rec in f.read_row_group(rg, columns=["image"]).column("image").to_pylist():
            if seen >= limit:
                return
            if rec and rec.get("bytes"):
                seen += 1
                yield rec["bytes"]


def _source_items(kind, location, limit: int, rng: random.Random):
    if kind == "files":
        paths = list(location)
        rng.shuffle(paths)
        for p in paths[:limit]:
            yield p.name, p.read_bytes()
    elif kind == "dir":
        files = list(iter_image_files(location))
        rng.shuffle(files)
        for p in files[:limit]:
            yield p.name, p.read_bytes()
    else:
        for i, blob in enumerate(_parquet_images(location, limit)):
            yield f"{location.stem}_{i:05d}", blob


def _centre_crop(img: Image.Image, side: int) -> Image.Image:
    w, h = img.size
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _write_jpeg(img: Image.Image, path: Path) -> None:
    img.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=300)
    ap.add_argument("--crop-side", type=int, default=CROP_SIDE)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", type=Path, default=BENCH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    digests: set[str] = set()

    sources = list(SOURCES) + community_forensics_sources()
    for track, name, label, generator, kind, location in sources:
        if kind != "files" and not location.exists():
            print(f"  SKIP {name}: {location} missing")
            continue
        per_source = args.per_source if kind != "files" else min(args.per_source, len(location))
        dirs = {p: args.out / p / label / name for p in ("native", "matched", "crop")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        kept = failed = dupes = toosmall = 0
        for stem, blob in _source_items(kind, location, per_source * 3, rng):
            if kept >= per_source:
                break
            digest = hashlib.sha256(blob).hexdigest()
            if digest in digests:
                dupes += 1
                continue
            try:
                loaded = open_image(blob)
            except ImageLoadError:
                failed += 1
                continue
            img = loaded.image
            if loaded.is_animated:
                failed += 1
                continue
            # The crop protocol only works if every image can supply a full
            # CROP_SIDE square without upscaling; enforce it corpus-wide so all
                # three protocols cover exactly the same images.
            if min(img.size) < args.crop_side:
                toosmall += 1
                continue
            digests.add(digest)

            base = Path(stem).stem
            ext = ".png" if (loaded.format or "").upper() in ("PNG", "TIFF", "BMP", "WEBP") else ".jpg"
            native = dirs["native"] / f"{base}{ext}"
            matched = dirs["matched"] / f"{base}.jpg"
            crop = dirs["crop"] / f"{base}.jpg"
            try:
                # native keeps the file byte-for-byte: any re-encode here would
                # overwrite the very compression history a detector may use.
                native.write_bytes(blob)

                w, h = img.size
                scale = min(1.0, MATCHED_MAX_SIDE / max(w, h))
                small = img if scale == 1.0 else img.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
                _write_jpeg(small, matched)
                _write_jpeg(_centre_crop(img, args.crop_side), crop)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"    write failed for {base}: {exc}")
                continue

            kept += 1
            rows.append({
                "id": f"{name}/{base}",
                "track": track,
                "label": label,
                "source": name,
                "generator": generator,
                "native": str(native.relative_to(ROOT)),
                "matched": str(matched.relative_to(ROOT)),
                "crop": str(crop.relative_to(ROOT)),
                "format": loaded.format or "",
                "width": img.size[0],
                "height": img.size[1],
                "sha256": digest[:16],
            })
        print(f"  {track:5s} {name:26s} {label:4s} kept={kept:4d} small={toosmall:4d} "
              f"failed={failed:3d} dup={dupes:3d}")

    if not rows:
        print("no images collected")
        return 1

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_real = sum(r["label"] == "real" for r in rows)
    print(f"\nwrote {manifest}: {len(rows)} images ({n_real} real / {len(rows) - n_real} ai)")
    for track in sorted({r["track"] for r in rows}):
        sub = [r for r in rows if r["track"] == track]
        gens = len({r["generator"] for r in sub if r["label"] == "ai"})
        print(f"  track {track}: {len(sub)} images, "
              f"{sum(r['label'] == 'real' for r in sub)} real, {gens} generators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
