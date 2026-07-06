#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUT_DIR="${OUT_DIR:-results/numa-affinity-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
P2P_ARGS=()
if [[ "${ENABLE_P2P:-0}" == "1" ]]; then
  P2P_ARGS+=(--enable-p2p)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m cluster_health_detect.numa_affinity_benchmark \
  --out-dir "$OUT_DIR" \
  --backend "${BACKEND:-auto}" \
  --device "${DEVICE:-auto}" \
  --tests "${TESTS:-h2d,d2d,collective}" \
  --policies "${BIND_POLICIES:-auto}" \
  --cpus-per-rank "${CPUS_PER_RANK:-0}" \
  --sizes-mb "${SIZES_MB:-16,64,256}" \
  --dtype "${DTYPE:-float16}" \
  --iters "${ITERS:-20}" \
  --warmup "${WARMUP:-5}" \
  --repeats "${REPEATS:-3}" \
  "${P2P_ARGS[@]}"

python3 -m cluster_health_detect.excel_heatmap "$OUT_DIR" \
  --affinity-json "$OUT_DIR/affinity.json" \
  --out "$OUT_DIR/numa-affinity-heatmap.xlsx"

python3 -m cluster_health_detect.summarize "$OUT_DIR" \
  --title "ClusterHealthDetect NUMA Affinity Report" \
  --out "$OUT_DIR/report.md"

echo "Wrote $OUT_DIR"
