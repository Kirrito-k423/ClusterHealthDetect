#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/affinity-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

python3 -m cluster_health_detect.affinity_probe --out "$OUT_DIR/affinity.json"

echo "Wrote $OUT_DIR/affinity.json"

