# Two stages so the ~700 MB of build wheels and pip metadata never reach the
# runtime image; only the installed site-packages and the model cache do.
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models

WORKDIR /src

# torch is pulled from the CPU index: the default wheel carries the bundled
# CUDA runtime, which is ~2 GB of libraries this image can never use.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^(torch|torchvision|pytest|pandas|pyarrow)' requirements.txt > runtime-requirements.txt \
 && pip install -r runtime-requirements.txt \
 && pip install "fastapi==0.141.1" "uvicorn[standard]==0.53.0" "python-multipart==0.0.32"

COPY aidetect ./aidetect
COPY scripts ./scripts
COPY models ./models
COPY models_permissive ./models_permissive
COPY pyproject.toml README.md ./
RUN pip install --no-deps -e .

# Bake the checkpoints in so the container never needs the network. Only the
# files inference reads: several of these repos ship an optimiser state and
# intermediate training checkpoints, which is the difference between 1.7 GB
# and 9.7 GB. Set BAKE_MODELS=false to build a slim image and mount /opt/models
# at run time instead.
ARG BAKE_MODELS=true
# models         -> the 5-member ensemble, which includes Organika/sdxl-detector
#                   (CC-BY-NC-3.0). Fine locally; NOT redistributable in a public
#                   image, because anyone pulling it may use it commercially.
# models_permissive -> MIT + Apache-2.0 only, and costs 0.0008 AUROC. This is the
#                   one to publish.
ARG MODEL_DIR=models
# mkdir unconditionally: the runtime stage copies this path either way, and a
# COPY from a directory that was never created fails the whole build.
RUN mkdir -p /opt/models \
 && if [ "$BAKE_MODELS" = "true" ]; then \
        python scripts/fetch_models.py --ensemble-only --model-dir "$MODEL_DIR"; \
    else \
        echo "skipping model download; mount a populated cache at /opt/models"; \
    fi


FROM python:3.12-slim AS runtime

# Re-declared: build args do not cross stages.
ARG MODEL_DIR=models
ARG VERSION=0.1.0
ARG LICENSES="MIT AND Apache-2.0"

LABEL org.opencontainers.image.title="aidetect" \
      org.opencontainers.image.description="Local, offline detection of AI-generated images" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/hIASL99/simple_ai_image_detector" \
      org.opencontainers.image.licenses="${LICENSES}" \
      io.aidetect.ensemble="${MODEL_DIR}"

# libgomp is OpenMP, which torch needs for CPU threading; the slim base omits it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 aidetect

ENV HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=0 \
    AIDETECT_MODEL_DIR=/app/${MODEL_DIR} \
    AIDETECT_OPERATING_POINT=fpr5

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=build --chown=aidetect:aidetect /opt/models /opt/models
COPY --from=build --chown=aidetect:aidetect /src/aidetect /app/aidetect
COPY --from=build --chown=aidetect:aidetect /src/scripts /app/scripts
COPY --from=build --chown=aidetect:aidetect /src/models /app/models
COPY --from=build --chown=aidetect:aidetect /src/models_permissive /app/models_permissive

WORKDIR /app
USER aidetect

EXPOSE 8000

# Loading the ensemble takes a few seconds, so give it a start period rather
# than letting an orchestrator kill the container before it is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# One worker on purpose: scoring is CPU-bound and already uses every core, so a
# second worker halves the threads each request gets and doubles peak memory.
CMD ["uvicorn", "aidetect.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
