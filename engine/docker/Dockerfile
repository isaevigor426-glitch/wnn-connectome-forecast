# Scorer-like image (public part of the organizers' Dockerfile).
# Put the starter pack's requirements.txt next to this file before building.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba

WORKDIR /work
