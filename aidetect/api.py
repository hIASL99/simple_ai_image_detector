"""HTTP API around the detector.

Inference here is CPU-bound and already uses every core, so the server does not
try to run requests concurrently: a single lock serialises scoring and the
worker count stays at one. Two requests in parallel on one box would each get
half the threads and finish no sooner, while doubling peak memory.

    uvicorn aidetect.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .backends import REGISTRY
from .detector import DEFAULT_MODEL_DIR, Detector
from .imageio import ImageLoadError, open_image
from .metadata import read_metadata

# An image arriving over HTTP is untrusted input. The decoder already guards
# against decompression bombs; these bound what reaches it in the first place.
MAX_BYTES = int(os.getenv("AIDETECT_MAX_BYTES", 32 * 1024 * 1024))
MAX_FILES = int(os.getenv("AIDETECT_MAX_FILES", 16))
MODEL_DIR = os.getenv("AIDETECT_MODEL_DIR", str(DEFAULT_MODEL_DIR))
OPERATING_POINT = os.getenv("AIDETECT_OPERATING_POINT", "fpr5")
THREADS = int(os.getenv("AIDETECT_THREADS", 0)) or None

_state: dict = {"detector": None, "error": None, "loaded_at": None}
_lock = asyncio.Lock()


def _load() -> None:
    t0 = time.time()
    try:
        _state["detector"] = Detector.load(
            model_dir=MODEL_DIR, operating_point=OPERATING_POINT,
            local_files_only=True, num_threads=THREADS)
        _state["loaded_at"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001 - reported through /health, not a crash loop
        _state["error"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading takes seconds and every request needs it, so pay it at startup
    # rather than making the first caller wait. A failure is recorded and
    # surfaced by /health instead of killing the process, so an operator can
    # see *why* rather than watching a container restart in a loop.
    await asyncio.get_running_loop().run_in_executor(None, _load)
    yield
    _state["detector"] = None


app = FastAPI(
    title="aidetect",
    version=__version__,
    summary="Local detection of AI-generated images",
    description="POST an image to /detect. Runs entirely on CPU, offline.",
    lifespan=lifespan,
)


def require_detector() -> Detector:
    if _state["detector"] is None:
        raise HTTPException(status_code=503,
                            detail=_state["error"] or "models are still loading")
    return _state["detector"]


class ModelScore(BaseModel):
    model_config = {"protected_namespaces": ()}


class Result(BaseModel):
    filename: str
    verdict: str = Field(description="AI-GENERATED, AI-EDITED, REAL, UNCERTAIN or ERROR")
    p_ai: float | None = Field(description="calibrated probability the image was generated")
    threshold: float
    low_threshold: float | None
    basis: str
    per_model: dict[str, float] = {}
    metadata_says_ai: bool = False
    metadata_generator: str | None = None
    truncated: bool = False
    error: str | None = None


class DetectResponse(BaseModel):
    results: list[Result]
    elapsed_ms: int
    operating_point: str


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    if not files:
        raise HTTPException(400, "send at least one file in the 'files' field")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} files per request")
    out = []
    for f in files:
        blob = await f.read()
        if len(blob) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: larger than {MAX_BYTES} bytes")
        if not blob:
            raise HTTPException(400, f"{f.filename}: empty upload")
        out.append((f.filename or "upload", blob))
    return out


@app.get("/health", summary="Readiness and what is loaded")
def health() -> JSONResponse:
    ready = _state["detector"] is not None
    body = {
        "status": "ok" if ready else "unavailable",
        "version": __version__,
        "models_loaded": ready,
        "load_seconds": round(_state["loaded_at"], 2) if _state["loaded_at"] else None,
        "error": _state["error"],
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/info", summary="Which detectors are in the ensemble, and at what operating point")
def info(detector: Annotated[Detector, Depends(require_detector)]) -> dict:
    fusion = detector.fusion
    return {
        "version": __version__,
        "ensemble": sorted(detector.backends) + (["clip-probe"] if detector.probe else []),
        "licences": {n: REGISTRY[n].license for n in detector.backends if n in REGISTRY},
        "operating_point": OPERATING_POINT,
        "threshold": detector.threshold,
        "abstain_below": detector.low_threshold,
        "available_operating_points": sorted(fusion.thresholds) if fusion else [],
        "calibrated_for_dtype": fusion.dtype if fusion else "unknown",
        "limits": {"max_bytes_per_file": MAX_BYTES, "max_files_per_request": MAX_FILES},
    }


@app.post("/detect", response_model=DetectResponse,
          summary="Score one or more images")
async def detect(
    detector: Annotated[Detector, Depends(require_detector)],
    files: Annotated[list[UploadFile], File(description="image files")],
    operating_point: Annotated[str | None, Query(
        description="override the configured operating point")] = None,
    metadata: Annotated[bool, Query(
        description="read EXIF/PNG/C2PA provenance as well as the pixels")] = True,
) -> DetectResponse:
    uploads = await _read_uploads(files)

    threshold, low = detector.threshold, detector.low_threshold
    if operating_point:
        if not detector.fusion or operating_point not in detector.fusion.thresholds:
            raise HTTPException(400, f"unknown operating point {operating_point!r}")
        threshold = float(detector.fusion.thresholds[operating_point])

    t0 = time.time()
    # One request at a time: see the module docstring.
    async with _lock:
        results = await asyncio.get_running_loop().run_in_executor(
            None, _score, detector, uploads, threshold, low, metadata)
    return DetectResponse(results=results, elapsed_ms=int((time.time() - t0) * 1000),
                          operating_point=operating_point or OPERATING_POINT)


def _score(detector: Detector, uploads: list[tuple[str, bytes]], threshold: float,
           low: float | None, use_metadata: bool) -> list[Result]:
    """Decode, score, and package. Runs on a worker thread."""
    images, ok, truncated = [], [], []
    out: dict[int, Result] = {}

    for index, (name, blob) in enumerate(uploads):
        try:
            loaded = open_image(blob)
        except ImageLoadError as exc:
            # The decoder only saw bytes, so it can only say "12 bytes"; the
            # caller needs to know which upload failed.
            why = f"{name}: {exc}"
            out[index] = Result(filename=name, verdict="ERROR", p_ai=None,
                                threshold=threshold, low_threshold=low,
                                basis=why, error=why)
            continue
        images.append(loaded.image)
        ok.append((index, name, blob))
        truncated.append(loaded.truncated)

    if images:
        raw = detector.raw_scores(images)
        cal = (detector.fusion.calibrated({k: v for k, v in raw.items()
                                           if k in detector.fusion.backends})
               if detector.fusion else raw)
        fused = detector.fuse(raw)
        for j, (index, name, blob) in enumerate(ok):
            meta = None
            if use_metadata:
                try:
                    meta = read_metadata(blob)
                except Exception:  # noqa: BLE001 - provenance is best-effort
                    meta = None
            # Reuse Prediction so the API and the CLI can never disagree about
            # what a score means or where the verdict boundaries sit.
            from .detector import Prediction
            pred = Prediction(path=name, p_ai=float(fused[j]), threshold=threshold,
                              low_threshold=low, truncated=truncated[j],
                              per_backend={k: float(v[j]) for k, v in cal.items()},
                              metadata=meta)
            out[index] = Result(
                filename=name, verdict=pred.verdict, p_ai=round(pred.p_ai, 4),
                threshold=round(threshold, 4),
                low_threshold=None if low is None else round(low, 4),
                basis=pred.basis,
                per_model={k: round(v, 4) for k, v in sorted(pred.per_backend.items())},
                metadata_says_ai=bool(meta and meta.says_ai),
                metadata_generator=meta.generator if meta else None,
                truncated=truncated[j])

    return [out[i] for i in range(len(uploads))]
