#!/usr/bin/env python3
"""Measure a fixed two-card communication path under every CPU-core pair."""

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
from .benchmark import (
    env_snapshot,
    import_torch_stack,
    init_dist,
    now_utc,
    parse_csv_ints,
    rank_barrier,
    rank_info,
    run_cmd,
    synchronize,
    torch_dtype,
    write_json,
)

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def parse_ids(raw: str, default: list[int]) -> list[int]:
    if raw.strip() == "auto":
        return default
    return parse_cpu_list(raw)


def parse_pair(raw: str) -> tuple[int, int]:
    values = parse_cpu_list(raw)
    if len(values) != 2:
        raise ValueError(f"expected exactly two ids, got {raw}")
    return values[0], values[1]


def set_rank_device(torch: Any, device_kind: str, npu_pair: tuple[int, int], local_rank: int) -> str:
    device_id = npu_pair[local_rank]
    if device_kind == "npu":
        torch.npu.set_device(device_id)
        return f"npu:{device_id}"
    if device_kind == "cuda":
        torch.cuda.set_device(device_id)
        return f"cuda:{device_id}"
    raise ValueError(f"unsupported device kind {device_kind}")


def resolve_device_kind(stack: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if stack.get("npu_available") and int(stack.get("npu_count") or 0) > 0:
        return "npu"
    if stack.get("cuda_available") and int(stack.get("cuda_count") or 0) > 0:
        return "cuda"
    return "none"


def make_all_gather_buffers(torch: Any, device: str, dtype_name: str, size_mb: int) -> tuple[Any, list[Any], int]:
    dtype = torch_dtype(torch, dtype_name)
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, size_mb * 1024 * 1024 // element_size)
    tensor = torch.empty(numel, dtype=dtype, device=device)
    output = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    return tensor, output, numel * element_size


def measure_all_gather(torch: Any, device: str, tensor: Any, output: list[Any], bytes_per_rank: int, size_mb: int, iters: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        torch.distributed.all_gather(output, tensor)
    synchronize(torch, device)
    torch.distributed.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        torch.distributed.all_gather(output, tensor)
    synchronize(torch, device)
    torch.distributed.barrier()
    elapsed = max(time.perf_counter() - start, 1e-12)
    avg_seconds = elapsed / iters
    return {
        "status": "ok",
        "operation": "all_gather",
        "size_mb": size_mb,
        "iterations": iters,
        "seconds": elapsed,
        "avg_seconds": avg_seconds,
        "alg_gbps": bytes_per_rank / avg_seconds / 1e9,
        "rank_exchange_gbps": bytes_per_rank * (torch.distributed.get_world_size() - 1) / avg_seconds / 1e9,
    }


def metric_base(info: dict[str, Any], cpu0: int | None, cpu1: int | None, repeat: int | None, **extra: Any) -> dict[str, Any]:
    metric = {
        "ts": now_utc(),
        "test": "core_pair_d2d_matrix",
        "rank0_cpu": cpu0,
        "rank1_cpu": cpu1,
        "repeat": repeat,
        **info,
    }
    metric.update(extra)
    return metric


def main() -> int:
    parser = argparse.ArgumentParser(description="Core-pair by core-pair D2D/allgather matrix for two fixed devices")
    parser.add_argument("--out-dir", default="results/core-pair-d2d-matrix/latest")
    parser.add_argument("--backend", default="auto", choices=["auto", "hccl", "nccl", "gloo"])
    parser.add_argument("--device-kind", default="auto", choices=["auto", "npu", "cuda"])
    parser.add_argument("--device-pair", default="0,1", help="Two local device ids, e.g. 0,1")
    parser.add_argument("--rank0-cpus", default="auto")
    parser.add_argument("--rank1-cpus", default="auto")
    parser.add_argument("--sizes-mb", default="16,64,256")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint-every-pairs", type=int, default=256, help="Write rank-local partial JSON after this many CPU pairs; 0 disables checkpoints")
    parser.add_argument("--record-ranks", default="rank0", choices=["rank0", "all"], help="Record metrics from rank0 only, or from all ranks")
    args = parser.parse_args()

    info = rank_info()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_affinity = current_allowed_cpus()
    records, affinity_summary = collect_cpu_records(check_bindable=True)
    all_cpus = bindable_cpus(records)
    rank0_cpus = parse_ids(args.rank0_cpus, all_cpus)
    rank1_cpus = parse_ids(args.rank1_cpus, all_cpus)
    sizes_mb = parse_csv_ints(args.sizes_mb)
    device_pair = parse_pair(args.device_pair)
    stack = import_torch_stack()
    device_kind = resolve_device_kind(stack, args.device_kind)

    meta: dict[str, Any] = {
        "created_at": now_utc(),
        "argv": sys.argv,
        "env": env_snapshot(),
        "rank_info": info,
        "torch_stack": stack,
        "device_kind": device_kind,
        "device_pair": list(device_pair),
        "rank0_cpus": rank0_cpus,
        "rank1_cpus": rank1_cpus,
        "sizes_mb": sizes_mb,
        "record_ranks": args.record_ranks,
        "affinity_summary": affinity_summary,
        "npu_smi_head": run_cmd(["npu-smi", "info"], timeout=10),
    }
    metrics: list[dict[str, Any]] = []
    record_this_rank = args.record_ranks == "all" or info["rank"] == 0

    def save_partial(pair_count: int, cpu0: int | None = None, cpu1: int | None = None) -> None:
        payload = {
            "meta": {
                **meta,
                "partial": True,
                "updated_at": now_utc(),
                "completed_pairs_per_rank": pair_count,
                "last_rank0_cpu": cpu0,
                "last_rank1_cpu": cpu1,
            },
            "metrics": metrics,
        }
        write_json(out_dir / f"rank_{info['rank']:05d}_partial.json", payload)

    torch = None
    device = "cpu"
    if stack["torch_ok"] and device_kind in {"npu", "cuda"}:
        import torch as torch_module  # type: ignore

        torch = torch_module
    else:
        metrics.append(metric_base(info, None, None, None, status="skip", reason=stack.get("torch_error") or "torch/device unavailable"))

    if torch is not None:
        if info["world_size"] != 2 or info["local_world_size"] < 2:
            metrics.append(metric_base(info, None, None, None, status="error", reason="run with torchrun --nproc_per_node=2"))
        else:
            device = set_rank_device(torch, device_kind, device_pair, info["local_rank"])
            ok, backend_or_error = init_dist(torch, args.backend, device)
            if not ok:
                metrics.append(metric_base(info, None, None, None, status="error", reason=backend_or_error, device=device))
            else:
                rank_barrier(torch)
                buffers = {size_mb: make_all_gather_buffers(torch, device, args.dtype, size_mb) for size_mb in sizes_mb}
                pair_count = 0
                for repeat in range(args.repeats):
                    for cpu0 in rank0_cpus:
                        for cpu1 in rank1_cpus:
                            target_cpu = cpu0 if info["local_rank"] == 0 else cpu1
                            bind_ok, actual, bind_err = set_current_affinity({target_cpu})
                            bind_actual_cpus_list = format_cpu_list(actual)
                            if not bind_ok or target_cpu not in actual:
                                if record_this_rank:
                                    metrics.append(
                                        metric_base(
                                            info,
                                            cpu0,
                                            cpu1,
                                            repeat,
                                            status="error",
                                            operation="bind_cpu",
                                            target_cpu=target_cpu,
                                            actual_cpus=actual,
                                            reason=bind_err,
                                            device=device,
                                        )
                                    )
                                continue
                            rank_barrier(torch)
                            for size_mb in sizes_mb:
                                try:
                                    tensor, output, bytes_per_rank = buffers[size_mb]
                                    result = measure_all_gather(torch, device, tensor, output, bytes_per_rank, size_mb, args.iters, args.warmup)
                                    if record_this_rank:
                                        metrics.append(
                                            metric_base(
                                                info,
                                                cpu0,
                                                cpu1,
                                                repeat,
                                                **result,
                                                backend=backend_or_error,
                                                device=device,
                                                device_kind=device_kind,
                                                device_pair=list(device_pair),
                                                dtype=args.dtype,
                                                target_cpu=target_cpu,
                                                bind_actual_cpus=actual,
                                                bind_actual_cpus_list=bind_actual_cpus_list,
                                            )
                                        )
                                except Exception as exc:
                                    if record_this_rank:
                                        metrics.append(
                                            metric_base(
                                                info,
                                                cpu0,
                                                cpu1,
                                                repeat,
                                                status="error",
                                                operation="all_gather",
                                                size_mb=size_mb,
                                                error=repr(exc),
                                                device=device,
                                                target_cpu=target_cpu,
                                            )
                                        )
                            rank_barrier(torch)
                            pair_count += 1
                            if args.checkpoint_every_pairs and pair_count % args.checkpoint_every_pairs == 0:
                                save_partial(pair_count, cpu0, cpu1)
                                if info["rank"] == 0:
                                    total_pairs = len(rank0_cpus) * len(rank1_cpus) * args.repeats
                                    print(f"progress repeat={repeat + 1}/{args.repeats} pairs={pair_count}/{total_pairs} metrics_rank={len(metrics)}", flush=True)

    if original_affinity is not None:
        set_current_affinity(original_affinity)

    meta["finished_at"] = now_utc()
    write_json(out_dir / f"rank_{info['rank']:05d}.json", {"meta": meta, "metrics": metrics})
    merged: list[dict[str, Any]] = metrics
    if args.record_ranks == "all" and torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[Any] = [None for _ in range(info["world_size"])]
        torch.distributed.all_gather_object(gathered, metrics)
        merged = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
    if info["rank"] == 0:
        write_json(out_dir / "results.json", {"meta": meta, "metrics": merged})
        write_json(out_dir / "affinity.json", {"created_at": now_utc(), "summary": affinity_summary, "cpus": [record.to_dict() for record in records]})
        values = [float(metric["alg_gbps"]) for metric in merged if metric.get("status") == "ok" and isinstance(metric.get("alg_gbps"), (int, float))]
        if values:
            print(f"metrics={len(merged)} ok={len(values)} alg_gbps_max={max(values):.3f} alg_gbps_median={median(values):.3f}")
        else:
            print(f"metrics={len(merged)} ok=0")
        print(out_dir)
    try:
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
