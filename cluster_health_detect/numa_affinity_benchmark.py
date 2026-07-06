#!/usr/bin/env python3
"""Torchrun benchmark for CPU/NUMA affinity effects on H2D, D2D, and allgather."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import median
from typing import Any

from .affinity import apply_policy, collect_cpu_records, current_allowed_cpus, format_cpu_list, resolve_auto_policies
from .benchmark import (
    bandwidth_probe,
    base_metric,
    choose_device,
    collective_probe,
    env_snapshot,
    gather_metrics,
    import_torch_stack,
    init_dist,
    now_utc,
    parse_csv_ints,
    rank_barrier,
    rank_info,
    run_cmd,
    summarize_for_stdout,
    write_json,
)

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def parse_policies(raw: str, records: list[Any]) -> list[str]:
    if raw.strip() == "auto":
        return resolve_auto_policies(records)
    return [item.strip() for item in raw.split(";") if item.strip()]


def annotate(metrics: list[dict[str, Any]], bind_result: dict[str, Any], repeat: int) -> list[dict[str, Any]]:
    for metric in metrics:
        metric["bind_policy"] = bind_result.get("policy")
        metric["bind_mode"] = bind_result.get("mode")
        metric["bind_numa_node"] = bind_result.get("numa_node")
        metric["bind_status"] = bind_result.get("status")
        metric["bind_actual_cpus"] = bind_result.get("actual_cpus")
        metric["bind_actual_cpus_list"] = bind_result.get("actual_cpus_list")
        metric["repeat"] = repeat
    return metrics


def numeric_summary(metrics: list[dict[str, Any]]) -> str:
    ok = [m for m in metrics if m.get("status") == "ok"]
    parts = [f"metrics={len(metrics)} ok={len(ok)}"]
    for key in ["gbps", "alg_gbps", "rank_exchange_gbps"]:
        values = [float(m[key]) for m in ok if isinstance(m.get(key), (int, float))]
        if values:
            parts.append(f"{key}: max={max(values):.3f} median={median(values):.3f}")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure H2D/D2D/allgather under CPU/NUMA binding policies")
    parser.add_argument("--out-dir", default="results/numa-affinity/latest")
    parser.add_argument("--backend", default="auto", choices=["auto", "hccl", "nccl", "gloo"])
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    parser.add_argument("--tests", default="h2d,d2d,collective", help="Comma-separated: h2d,d2d,collective")
    parser.add_argument("--policies", default="auto", help="Semicolon-separated policies or auto. Examples: none;local_rank;numa:0;numa:1:shard")
    parser.add_argument("--cpus-per-rank", type=int, default=0, help="Limit CPUs per rank for sharded policies; 0 keeps the full shard")
    parser.add_argument("--sizes-mb", default="16,64,256")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--enable-p2p", action="store_true")
    args = parser.parse_args()

    info = rank_info()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_affinity = current_allowed_cpus()
    records, affinity_summary = collect_cpu_records(check_bindable=True)
    policies = parse_policies(args.policies, records)
    tests = {item.strip().lower() for item in args.tests.split(",") if item.strip()}
    sizes_mb = parse_csv_ints(args.sizes_mb)

    stack = import_torch_stack()
    torch = None
    if stack["torch_ok"]:
        import torch as torch_module  # type: ignore

        torch = torch_module
    device = "cpu"
    device_warning = None
    if torch is not None:
        device, device_warning = choose_device(torch, stack, args.device, info["local_rank"])

    if torch is not None and "collective" in tests:
        ok, _ = init_dist(torch, args.backend, device)
        if ok:
            rank_barrier(torch)

    metrics: list[dict[str, Any]] = [
        base_metric(
            info,
            "meta",
            "environment",
            status="ok",
            selected_device=device,
            torch_stack=stack,
            device_warning=device_warning,
            original_affinity=sorted(original_affinity) if original_affinity is not None else None,
            original_affinity_list=format_cpu_list(original_affinity or set()),
        )
    ]

    for repeat in range(args.repeats):
        for policy in policies:
            bind_result = apply_policy(policy, records, info["local_rank"], info["local_world_size"], args.cpus_per_rank, original_affinity)
            bind_metric = base_metric(info, f"bind:{policy}", "affinity", operation="apply_bind", repeat=repeat, **bind_result)
            metrics.append(bind_metric)
            rank_barrier(torch)
            probe_metrics: list[dict[str, Any]] = []
            if bind_result.get("status") != "ok":
                probe_metrics.append(
                    base_metric(info, f"bind:{policy}", "affinity", status="skip", operation="benchmark", repeat=repeat, reason=bind_result.get("reason"))
                )
            else:
                if "h2d" in tests:
                    probe_metrics.extend(bandwidth_probe(torch, info, f"bind:{policy}", device, "h2d", args.dtype, sizes_mb, args.iters, args.warmup))
                if "d2d" in tests:
                    probe_metrics.extend(bandwidth_probe(torch, info, f"bind:{policy}", device, "d2d", args.dtype, sizes_mb, args.iters, args.warmup))
                if "collective" in tests:
                    probe_metrics.extend(
                        collective_probe(torch, info, f"bind:{policy}", device, args.dtype, sizes_mb, args.iters, args.warmup, args.backend, args.enable_p2p)
                    )
            metrics.extend(annotate(probe_metrics, bind_result, repeat))
            rank_barrier(torch)

    if original_affinity is not None:
        apply_policy("none", records, info["local_rank"], info["local_world_size"], 0, original_affinity)

    meta = {
        "created_at": now_utc(),
        "argv": sys.argv,
        "rank_info": info,
        "env": env_snapshot(),
        "torch_stack": stack,
        "selected_device": device,
        "device_warning": device_warning,
        "affinity_summary": affinity_summary,
        "policies": policies,
        "npu_smi_head": run_cmd(["npu-smi", "info"], timeout=10),
    }
    write_json(out_dir / f"rank_{info['rank']:05d}.json", {"meta": meta, "metrics": metrics})
    merged = gather_metrics(torch, metrics, info)
    if info["rank"] == 0:
        write_json(out_dir / "results.json", {"meta": meta, "metrics": merged})
        write_json(out_dir / "affinity.json", {"created_at": now_utc(), "summary": affinity_summary, "cpus": [record.to_dict() for record in records]})
        print(numeric_summary(merged), flush=True)
        print(summarize_for_stdout(merged), flush=True)

    try:
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

