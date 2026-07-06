#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/core-h2d-matrix-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

python3 -m cluster_health_detect.core_h2d_matrix \
  --out-dir "$OUT_DIR" \
  --device-kind "${DEVICE_KIND:-auto}" \
  --devices "${DEVICES:-auto}" \
  --cpus "${CPUS:-auto}" \
  --sizes-mb "${SIZES_MB:-16,64,256}" \
  --dtype "${DTYPE:-float16}" \
  --iters "${ITERS:-20}" \
  --warmup "${WARMUP:-5}" \
  --repeats "${REPEATS:-3}"

python3 -m cluster_health_detect.matrix_excel "$OUT_DIR" \
  --affinity-json "$OUT_DIR/affinity.json" \
  --out "$OUT_DIR/core-h2d-matrix.xlsx"

echo "Wrote $OUT_DIR"

