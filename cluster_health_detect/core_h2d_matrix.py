#!/usr/bin/env python3
"""Measure H2D bandwidth for every bound CPU core against every NPU/GPU card."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

from .affinity import bindable_cpus, collect_cpu_records, current_allowed_cpus, format_cpu_list, parse_cpu_list, set_current_affinity
from .benchmark import env_snapshot, import_torch_stack, now_utc, parse_csv_ints, run_cmd, synchronize, torch_dtype, write_json

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def parse_ids(raw: str, default: list[int]) -> list[int]:
    if raw.strip() == "auto":
        return default
    return parse_cpu_list(raw)


def set_device(torch: Any, kind: str, device_id: int) -> str:
    if kind == "npu":
        torch.npu.set_device(device_id)
        return f"npu:{device_id}"
    if kind == "cuda":
        torch.cuda.set_device(device_id)
        return f"cuda:{device_id}"
    raise ValueError(f"unsupported device kind {kind}")


def available_device_ids(stack: dict[str, Any], kind: str) -> list[int]:
    if kind == "npu":
        return list(range(int(stack.get("npu_count") or 0)))
    if kind == "cuda":
        return list(range(int(stack.get("cuda_count") or 0)))
    return []


def resolve_device_kind(stack: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if stack.get("npu_available") and int(stack.get("npu_count") or 0) > 0:
        return "npu"
    if stack.get("cuda_available") and int(stack.get("cuda_count") or 0) > 0:
        return "cuda"
    return "none"


def measure_h2d(torch: Any, device: str, dtype_name: str, size_mb: int, iters: int, warmup: int) -> dict[str, Any]:
    dtype = torch_dtype(torch, dtype_name)
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, size_mb * 1024 * 1024 // element_size)
    src = torch.empty(numel, dtype=dtype, device="cpu")
    try:
        src = src.pin_memory()
        pinned = True
    except Exception:
        pinned = False
    dst = torch.empty(numel, dtype=dtype, device=device)
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    synchronize(torch, device)
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    synchronize(torch, device)
    elapsed = max(time.perf_counter() - start, 1e-12)
    bytes_moved = numel * element_size * iters
    return {
        "status": "ok",
        "operation": "copy",
        "direction": "host_to_device",
        "size_mb": size_mb,
        "iterations": iters,
        "seconds": elapsed,
        "gbps": bytes_moved / elapsed / 1e9,
        "pinned": pinned,
    }


def metric_base(cpu: int | None, device_id: int | None, repeat: int | None, **extra: Any) -> dict[str, Any]:
    metric = {
        "ts": now_utc(),
        "test": "core_h2d_matrix",
        "host_cpu": cpu,
        "device_id": device_id,
        "repeat": repeat,
    }
    metric.update(extra)
    return metric


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-core by NPU-card H2D bandwidth matrix")
    parser.add_argument("--out-dir", default="results/core-h2d-matrix/latest")
    parser.add_argument("--device-kind", default="auto", choices=["auto", "npu", "cuda"])
    parser.add_argument("--devices", default="auto", help="Device ids, e.g. auto or 0-15 or 0,2,4")
    parser.add_argument("--cpus", default="auto", help="CPU ids, e.g. auto or 0-639 or 0,80,160")
    parser.add_argument("--sizes-mb", default="16,64,256")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_affinity = current_allowed_cpus()
    records, affinity_summary = collect_cpu_records(check_bindable=True)
    cpu_ids = parse_ids(args.cpus, bindable_cpus(records))
    stack = import_torch_stack()
    device_kind = resolve_device_kind(stack, args.device_kind)
    device_ids = parse_ids(args.devices, available_device_ids(stack, device_kind))
    sizes_mb = parse_csv_ints(args.sizes_mb)

    metrics: list[dict[str, Any]] = []
    torch = None
    if stack["torch_ok"] and device_kind in {"npu", "cuda"} and device_ids:
        import torch as torch_module  # type: ignore

        torch = torch_module
    else:
        reason = "torch/device unavailable"
        if stack.get("torch_error"):
            reason = str(stack["torch_error"])
        metrics.append(metric_base(None, None, None, status="skip", reason=reason))

    if torch is not None:
        for repeat in range(args.repeats):
            for cpu in cpu_ids:
                ok, actual, err = set_current_affinity({cpu})
                if not ok or cpu not in actual:
                    metrics.append(metric_base(cpu, None, repeat, status="error", operation="bind_cpu", actual_cpus=actual, reason=err))
                    continue
                actual_list = format_cpu_list(actual)
                for device_id in device_ids:
                    try:
                        device = set_device(torch, device_kind, device_id)
                        for size_mb in sizes_mb:
                            result = measure_h2d(torch, device, args.dtype, size_mb, args.iters, args.warmup)
                            metrics.append(
                                metric_base(
                                    cpu,
                                    device_id,
                                    repeat,
                                    **result,
                                    device=device,
                                    dtype=args.dtype,
                                    bind_actual_cpus=actual,
                                    bind_actual_cpus_list=actual_list,
                                )
                            )
                    except Exception as exc:
                        metrics.append(
                            metric_base(
                                cpu,
                                device_id,
                                repeat,
                                status="error",
                                operation="copy",
                                device_kind=device_kind,
                                error=repr(exc),
                                bind_actual_cpus=actual,
                                bind_actual_cpus_list=actual_list,
                            )
                        )

    if original_affinity is not None:
        set_current_affinity(original_affinity)

    meta = {
        "created_at": now_utc(),
        "argv": sys.argv,
        "env": env_snapshot(),
        "torch_stack": stack,
        "device_kind": device_kind,
        "device_ids": device_ids,
        "cpu_ids": cpu_ids,
        "affinity_summary": affinity_summary,
        "npu_smi_head": run_cmd(["npu-smi", "info"], timeout=10),
    }
    payload = {"meta": meta, "metrics": metrics}
    write_json(out_dir / "results.json", payload)
    write_json(out_dir / "affinity.json", {"created_at": now_utc(), "summary": affinity_summary, "cpus": [record.to_dict() for record in records]})
    ok_values = [float(metric["gbps"]) for metric in metrics if metric.get("status") == "ok" and isinstance(metric.get("gbps"), (int, float))]
    if ok_values:
        print(f"metrics={len(metrics)} ok={len(ok_values)} gbps_max={max(ok_values):.3f} gbps_median={median(ok_values):.3f}")
    else:
        print(f"metrics={len(metrics)} ok=0")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

