#!/usr/bin/env bash
set -euo pipefail

NNODES="${NNODES:?set NNODES, for example 2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK, 0 on master and 1 on worker}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the rank-0 host IP}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUT_DIR="${OUT_DIR:-results/multinode-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
P2P_ARGS=()
if [[ "${ENABLE_P2P:-0}" == "1" ]]; then
  P2P_ARGS+=(--enable-p2p)
fi

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m cluster_health_detect.benchmark \
  --out-dir "$OUT_DIR" \
  --backend "${BACKEND:-auto}" \
  --device "${DEVICE:-auto}" \
  --profiles "${PROFILES:-idle,cpu:2,device}" \
  --tests "${TESTS:-all}" \
  --sizes-mb "${SIZES_MB:-16,64,256}" \
  --cpu-sizes "${CPU_SIZES:-512,1024,2048}" \
  --device-sizes "${DEVICE_SIZES:-1024,2048,4096}" \
  --seconds "${SECONDS_PER_SIZE:-2}" \
  --iters "${ITERS:-20}" \
  --warmup "${WARMUP:-5}" \
  "${P2P_ARGS[@]}"

if [[ "$NODE_RANK" == "0" ]]; then
  python3 -m cluster_health_detect.summarize "$OUT_DIR" \
    --title "ClusterHealthDetect Multi Node Report" \
    --out "$OUT_DIR/report.md"
fi

echo "Wrote $OUT_DIR"
