#!/usr/bin/env bash
# Run any script of engine/ inside the scorer-like container with the platform's limits:
# 1 CPU, 16 GB RAM, no network.
#
#   ./docker/run.sh python bench_step.py
#   ./docker/run.sh python local_eval.py --solution-dir /starter/baseline \
#       --data /datasets/valid.parquet --max-seqs 20
#
# Mounts: project -> /work, $DATASETS (default ./datasets) -> /datasets,
#         $STARTER (default ./starter) -> /starter
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="${DATASETS:-$PWD/datasets}"
STARTER="${STARTER:-$PWD/starter}"
IMAGE="${IMAGE:-wunder-scorer:local}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [ -f docker/requirements.txt ] || { echo "copy the starter pack's requirements.txt to docker/requirements.txt"; exit 1; }
  docker build -t "$IMAGE" docker
fi

docker run --rm --cpus=1 --memory=16g --network=none \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD":/work -v "$DATASETS":/datasets:ro -v "$STARTER":/starter:ro \
  -w /work "$IMAGE" "$@"
