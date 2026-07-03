#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/env-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

{
  echo "## date"
  date -Is
  echo
  echo "## host"
  hostname
  echo
  echo "## uname"
  uname -a
  echo
  echo "## lscpu"
  lscpu || true
  echo
  echo "## lscpu -e"
  lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ || true
  echo
  echo "## /proc/self/status"
  grep -E 'Cpus_allowed|Mems_allowed' /proc/self/status || true
  echo
  echo "## npu-smi"
  npu-smi info || true
  echo
  echo "## torch stack"
  python3 - <<'PY'
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
try:
    import torch
    print("torch", torch.__version__)
    print("distributed", torch.distributed.is_available())
    print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
except Exception as exc:
    print("torch import failed", repr(exc))
try:
    import torch_npu
    print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
    print("npu", torch.npu.is_available(), torch.npu.device_count())
except Exception as exc:
    print("torch_npu import failed", repr(exc))
PY
} > "$OUT_DIR/env.md" 2>&1

echo "$OUT_DIR/env.md"

