#!/usr/bin/env python3
"""Create Excel heatmaps for per-core H2D and core-pair D2D matrices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .excel_heatmap import col_name, write_xlsx
from .summarize import load_metrics


def metric_number(metric: dict[str, Any]) -> tuple[str, float] | None:
    for field in ["gbps", "alg_gbps", "rank_exchange_gbps"]:
        value = metric.get(field)
        if isinstance(value, (int, float)):
            return field, float(value)
    return None


def h2d_sheets(metrics: list[dict[str, Any]]) -> list[tuple[str, list[list[Any]], str | None]]:
    grouped: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for metric in metrics:
        if metric.get("test") != "core_h2d_matrix" or metric.get("status") != "ok":
            continue
        if metric.get("host_cpu") is None or metric.get("device_id") is None or metric.get("size_mb") is None:
            continue
        value = metric.get("gbps")
        if not isinstance(value, (int, float)):
            continue
        grouped[int(metric["size_mb"])][(int(metric["host_cpu"]), int(metric["device_id"]))].append(float(value))

    sheets: list[tuple[str, list[list[Any]], str | None]] = []
    for size_mb, cells in sorted(grouped.items()):
        cpus = sorted({cpu for cpu, _ in cells})
        devices = sorted({device for _, device in cells})
        rows: list[list[Any]] = [[f"CPU core -> NPU H2D GB/s ({size_mb} MB)", *[f"npu:{device}" for device in devices]]]
        for cpu in cpus:
            row: list[Any] = [cpu]
            for device in devices:
                values = cells.get((cpu, device), [])
                row.append(median(values) if values else None)
            rows.append(row)
        heatmap_range = None
        if len(rows) > 1 and len(rows[0]) > 1:
            heatmap_range = f"B2:{col_name(len(rows[0]))}{len(rows)}"
        sheets.append((f"H2D_{size_mb}MB", rows, heatmap_range))
    return sheets


def d2d_sheets(metrics: list[dict[str, Any]]) -> list[tuple[str, list[list[Any]], str | None]]:
    grouped: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for metric in metrics:
        if metric.get("test") != "core_pair_d2d_matrix" or metric.get("status") != "ok":
            continue
        if metric.get("rank0_cpu") is None or metric.get("rank1_cpu") is None or metric.get("size_mb") is None:
            continue
        pair = metric_number(metric)
        if pair is None:
            continue
        _, value = pair
        grouped[int(metric["size_mb"])][(int(metric["rank0_cpu"]), int(metric["rank1_cpu"]))].append(value)

    sheets: list[tuple[str, list[list[Any]], str | None]] = []
    for size_mb, cells in sorted(grouped.items()):
        rank0_cpus = sorted({cpu0 for cpu0, _ in cells})
        rank1_cpus = sorted({cpu1 for _, cpu1 in cells})
        rows: list[list[Any]] = [[f"rank0 CPU -> rank1 CPU all_gather alg GB/s ({size_mb} MB)", *rank1_cpus]]
        for cpu0 in rank0_cpus:
            row: list[Any] = [cpu0]
            for cpu1 in rank1_cpus:
                values = cells.get((cpu0, cpu1), [])
                row.append(median(values) if values else None)
            rows.append(row)
        heatmap_range = None
        if len(rows) > 1 and len(rows[0]) > 1:
            heatmap_range = f"B2:{col_name(len(rows[0]))}{len(rows)}"
        sheets.append((f"D2D_{size_mb}MB", rows, heatmap_range))
    return sheets


def summary_rows(metrics: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = [["test", "size_mb", "row", "column", "metric", "count", "min", "median", "max", "max/min"]]
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for metric in metrics:
        if metric.get("status") != "ok":
            continue
        pair = metric_number(metric)
        if pair is None:
            continue
        field, value = pair
        if metric.get("test") == "core_h2d_matrix":
            key = ("core_h2d_matrix", metric.get("size_mb"), metric.get("host_cpu"), metric.get("device_id"), field)
        elif metric.get("test") == "core_pair_d2d_matrix":
            key = ("core_pair_d2d_matrix", metric.get("size_mb"), metric.get("rank0_cpu"), metric.get("rank1_cpu"), field)
        else:
            continue
        grouped[key].append(value)
    for key, values in sorted(grouped.items()):
        min_v = min(values)
        max_v = max(values)
        rows.append([*key, len(values), min_v, median(values), max_v, max_v / min_v if min_v else None])
    return rows


def affinity_rows(path: Path | None) -> list[list[Any]]:
    rows = [["cpu", "numa_node", "socket_id", "core_id", "online", "allowed", "bindable", "thread_siblings", "reason"]]
    if path is None or not path.exists():
        return rows
    data = json.loads(path.read_text())
    for cpu in data.get("cpus", []):
        rows.append(
            [
                cpu.get("cpu"),
                cpu.get("numa_node"),
                cpu.get("socket_id"),
                cpu.get("core_id"),
                cpu.get("online"),
                cpu.get("allowed"),
                cpu.get("bindable"),
                ",".join(str(v) for v in cpu.get("thread_siblings") or []),
                cpu.get("reason"),
            ]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build matrix heatmap XLSX for core H2D or core-pair D2D results")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--affinity-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/core-matrix-heatmap.xlsx"))
    args = parser.parse_args()

    _, metrics = load_metrics(args.paths)
    sheets = [*h2d_sheets(metrics), *d2d_sheets(metrics)]
    if not sheets:
        sheets = [("Heatmap", [["No core matrix metrics found"]], None)]
    sheets.append(("Summary", summary_rows(metrics), None))
    sheets.append(("Affinity", affinity_rows(args.affinity_json), None))
    write_xlsx(args.out, sheets)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

