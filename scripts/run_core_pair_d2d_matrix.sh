#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/core-pair-d2d-matrix-$(date +%Y%m%d-%H%M%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
mkdir -p "$OUT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m cluster_health_detect.core_pair_d2d_matrix \
  --out-dir "$OUT_DIR" \
  --backend "${BACKEND:-auto}" \
  --device-kind "${DEVICE_KIND:-auto}" \
  --device-pair "${DEVICE_PAIR:-0,1}" \
  --rank0-cpus "${RANK0_CPUS:-auto}" \
  --rank1-cpus "${RANK1_CPUS:-auto}" \
  --sizes-mb "${SIZES_MB:-16,64,256}" \
  --dtype "${DTYPE:-float16}" \
  --iters "${ITERS:-20}" \
  --warmup "${WARMUP:-5}" \
  --repeats "${REPEATS:-3}"

python3 -m cluster_health_detect.matrix_excel "$OUT_DIR" \
  --affinity-json "$OUT_DIR/affinity.json" \
  --out "$OUT_DIR/core-pair-d2d-matrix.xlsx"

echo "Wrote $OUT_DIR"

