#!/usr/bin/env python3
"""Summarize ClusterHealthDetect JSON outputs into Markdown."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def load_metrics(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metas: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            if (path / "results.json").exists():
                files.append(path / "results.json")
            else:
                files.extend(sorted(path.glob("rank_*.json")))
        else:
            files.append(path)
    seen: set[Path] = set()
    for file in files:
        if file in seen or not file.exists():
            continue
        seen.add(file)
        data = json.loads(file.read_text())
        if isinstance(data.get("meta"), dict):
            metas.append(data["meta"])
        if isinstance(data.get("metrics"), list):
            metrics.extend(data["metrics"])
    return metas, metrics


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_metric_groups(metrics: list[dict[str, Any]]) -> list[str]:
    numeric_keys = ["tflops", "gbps", "alg_gbps", "rank_exchange_gbps"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if metric.get("status") != "ok":
            continue
        key = (
            metric.get("profile"),
            metric.get("test"),
            metric.get("operation"),
            metric.get("scope", ""),
            metric.get("size_mb", metric.get("size_n", "")),
            metric.get("dtype", ""),
        )
        groups[key].append(metric)

    rows: list[list[Any]] = []
    for key, vals in sorted(groups.items()):
        best_key = next((k for k in numeric_keys if any(isinstance(v.get(k), (int, float)) for v in vals)), None)
        if not best_key:
            continue
        nums = [float(v[best_key]) for v in vals if isinstance(v.get(best_key), (int, float))]
        ranks = sorted({v.get("rank") for v in vals})
        hosts = sorted({v.get("host") for v in vals})
        skew = ""
        if nums and max(nums) > 0 and len(nums) > 1:
            skew = f"{max(nums) / max(min(nums), 1e-12):.2f}x"
        rows.append(
            [
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
                key[5],
                best_key,
                fmt(max(nums) if nums else None),
                fmt(median(nums) if nums else None),
                fmt(pct(nums, 0.1) if nums else None),
                skew,
                len(ranks),
                ",".join(str(h) for h in hosts[:4]),
            ]
        )
    if not rows:
        return ["未发现成功的数值型 benchmark 指标。"]
    return md_table(
        ["profile", "test", "op", "scope", "size", "dtype", "metric", "max", "median", "p10", "max/min", "ranks", "hosts"],
        rows,
    )


def summarize_errors(metrics: list[dict[str, Any]]) -> list[str]:
    errors = [m for m in metrics if m.get("status") in {"error", "skip"}]
    if not errors:
        return ["未记录 error/skip。"]
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    examples: dict[tuple[Any, ...], str] = {}
    for metric in errors:
        key = (metric.get("status"), metric.get("test"), metric.get("operation", ""), metric.get("scope", ""))
        counts[key] += 1
        examples.setdefault(key, str(metric.get("error") or metric.get("reason") or "")[:220])
    rows = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append([*key, count, examples[key]])
    return md_table(["status", "test", "op", "scope", "count", "example"], rows)


def summarize_environment(metas: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[str]:
    rows: list[list[Any]] = []
    env_metrics = [m for m in metrics if m.get("test") == "environment"]
    for metric in env_metrics:
        stack = metric.get("torch_stack") or {}
        rows.append(
            [
                metric.get("host"),
                metric.get("rank"),
                metric.get("selected_device"),
                stack.get("torch_ok"),
                stack.get("torch_version", stack.get("torch_error", "")),
                stack.get("torch_npu_ok"),
                stack.get("torch_npu_version", stack.get("torch_npu_error", "")),
                stack.get("npu_count"),
            ]
        )
    if not rows and metas:
        for meta in metas:
            stack = meta.get("torch_stack") or {}
            info = meta.get("rank_info") or {}
            rows.append(
                [
                    info.get("host", meta.get("hostname")),
                    info.get("rank"),
                    meta.get("selected_device"),
                    stack.get("torch_ok"),
                    stack.get("torch_version", stack.get("torch_error", "")),
                    stack.get("torch_npu_ok"),
                    stack.get("torch_npu_version", stack.get("torch_npu_error", "")),
                    stack.get("npu_count"),
                ]
            )
    if not rows:
        return ["无环境元信息。"]
    return md_table(["host", "rank", "device", "torch_ok", "torch", "torch_npu_ok", "torch_npu", "npu_count"], rows[:64])


def bottleneck_notes(metrics: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if metric.get("status") != "ok":
            continue
        key = (
            metric.get("profile"),
            metric.get("test"),
            metric.get("operation"),
            metric.get("scope", ""),
            metric.get("size_mb", metric.get("size_n", "")),
        )
        by_group[key].append(metric)

    for key, vals in sorted(by_group.items()):
        for metric_key in ["tflops", "gbps", "alg_gbps", "rank_exchange_gbps"]:
            nums = [(float(v[metric_key]), v) for v in vals if isinstance(v.get(metric_key), (int, float))]
            if len(nums) < 2:
                continue
            low, high = min(nums, key=lambda item: item[0]), max(nums, key=lambda item: item[0])
            if high[0] > 0 and low[0] / high[0] < 0.90:
                notes.append(
                    "- "
                    + f"`{key}` 的 rank 间差异超过 10%：最低 rank={low[1].get('rank')} host={low[1].get('host')} "
                    + f"{metric_key}={low[0]:.3f}，最高 rank={high[1].get('rank')} host={high[1].get('host')} {metric_key}={high[0]:.3f}。"
                )
                break
    if not notes:
        notes.append("- 未从当前样本中发现超过 10% 的 rank 间数值离散；这不代表生产 SFT 场景无问题，只说明本轮基准未复现。")
    return notes[:30]


def build_report(title: str, metas: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"- 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}",
        f"- 输入 metrics：{len(metrics)} 条",
        "",
        "## 环境概览",
        "",
        *summarize_environment(metas, metrics),
        "",
        "## 关键上限指标",
        "",
        *summarize_metric_groups(metrics),
        "",
        "## 候选瓶颈提示",
        "",
        *bottleneck_notes(metrics),
        "",
        "## Error / Skip",
        "",
        *summarize_errors(metrics),
        "",
        "## 解读口径",
        "",
        "- `cpu`/`device` 的 `tflops` 是 matmul 实测吞吐上限，适合比较 CPU 线程、NPU AICore 或虚拟化影响。",
        "- `h2d` 是 host 到 device copy 带宽；`d2d` 是同 rank device 内 copy 带宽。",
        "- `collective/all_gather/global` 反映训练中跨所有 rank 的 allgather；`local_host` 反映本机多卡；`sendrecv/local_host_pair` 与 `sendrecv/inter_host_pair` 需要显式开启 P2P。",
        "- `max/min` 超过 `1.10x` 时，需要继续查对应 rank 的 CPU 绑核、NPU health、HCCL/NIC 绑定、容器 cpuset、NUMA 和后台负载。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ClusterHealthDetect JSON outputs")
    parser.add_argument("paths", nargs="+", type=Path, help="results.json, rank_*.json, or result directories")
    parser.add_argument("--title", default="ClusterHealthDetect Benchmark Report")
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark-report.md"))
    args = parser.parse_args()

    metas, metrics = load_metrics(args.paths)
    report = build_report(args.title, metas, metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
