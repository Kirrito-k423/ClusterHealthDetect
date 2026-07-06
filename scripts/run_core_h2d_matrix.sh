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
  --repeats "${REPEATS:-3}" \
  --checkpoint-every-cpus "${CHECKPOINT_EVERY_CPUS:-1}"

python3 -m cluster_health_detect.matrix_excel "$OUT_DIR" \
  --affinity-json "$OUT_DIR/affinity.json" \
  --out "$OUT_DIR/core-h2d-matrix.xlsx"

if python3 - <<'PY' >/dev/null 2>&1
import PIL
PY
then
  python3 -m cluster_health_detect.matrix_png "$OUT_DIR" --out-dir "$OUT_DIR/png"
else
  echo "Pillow not available; skip PNG generation on this host."
fi

echo "Wrote $OUT_DIR"
