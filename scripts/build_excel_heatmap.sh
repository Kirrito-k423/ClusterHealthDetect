#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <results-dir-or-json> [more paths...]" >&2
  exit 2
fi

OUT="${OUT:-reports/cluster-health-heatmap.xlsx}"
AFFINITY_JSON="${AFFINITY_JSON:-}"

args=(python3 -m cluster_health_detect.excel_heatmap "$@" --out "$OUT")
if [[ -n "$AFFINITY_JSON" ]]; then
  args+=(--affinity-json "$AFFINITY_JSON")
fi

"${args[@]}"

echo "Wrote $OUT"

