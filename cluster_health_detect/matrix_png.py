#!/usr/bin/env python3
"""Render core-matrix benchmark JSON files as PNG heatmaps."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .summarize import load_metrics


def import_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont

        return Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise SystemExit("PNG export requires Pillow: python3 -m pip install pillow") from exc


def color_scale(value: float, lo: float, mid: float, hi: float) -> tuple[int, int, int]:
    red = (248, 105, 107)
    yellow = (255, 235, 132)
    green = (99, 190, 123)
    if hi <= lo:
        return green
    if value <= mid:
        t = 0.0 if mid <= lo else (value - lo) / (mid - lo)
        a, b = red, yellow
    else:
        t = 0.0 if hi <= mid else (value - mid) / (hi - mid)
        a, b = yellow, green
    return tuple(int(a[i] + (b[i] - a[i]) * max(0.0, min(1.0, t))) for i in range(3))


def grouped_h2d(metrics: list[dict[str, Any]]) -> dict[int, dict[tuple[int, int], list[float]]]:
    grouped: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for metric in metrics:
        if metric.get("test") != "core_h2d_matrix" or metric.get("status") != "ok":
            continue
        if metric.get("host_cpu") is None or metric.get("device_id") is None or metric.get("size_mb") is None:
            continue
        value = metric.get("gbps")
        if isinstance(value, (int, float)):
            grouped[int(metric["size_mb"])][(int(metric["host_cpu"]), int(metric["device_id"]))].append(float(value))
    return grouped


def grouped_d2d(metrics: list[dict[str, Any]]) -> dict[int, dict[tuple[int, int], list[float]]]:
    grouped: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for metric in metrics:
        if metric.get("test") != "core_pair_d2d_matrix" or metric.get("status") != "ok":
            continue
        if metric.get("rank0_cpu") is None or metric.get("rank1_cpu") is None or metric.get("size_mb") is None:
            continue
        value = metric.get("alg_gbps", metric.get("gbps"))
        if isinstance(value, (int, float)):
            grouped[int(metric["size_mb"])][(int(metric["rank0_cpu"]), int(metric["rank1_cpu"]))].append(float(value))
    return grouped


def draw_heatmap(
    title: str,
    row_label: str,
    col_label: str,
    rows: list[int],
    cols: list[int],
    values: dict[tuple[int, int], float],
    out: Path,
) -> None:
    Image, ImageDraw, ImageFont = import_pillow()
    cell_w = 52 if len(cols) <= 64 else max(3, min(18, 3200 // max(1, len(cols))))
    cell_h = 14 if len(rows) <= 128 else max(3, min(10, 3200 // max(1, len(rows))))
    left = 96
    top = 64
    right = 24
    bottom = 60
    width = left + len(cols) * cell_w + right
    height = top + len(rows) * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 11)
        title_font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    numeric = [v for v in values.values()]
    lo = min(numeric) if numeric else 0.0
    hi = max(numeric) if numeric else 1.0
    mid = median(numeric) if numeric else 0.5

    draw.text((12, 10), title, fill=(0, 0, 0), font=title_font)
    draw.text((left, 36), col_label, fill=(0, 0, 0), font=font)
    draw.text((8, top - 18), row_label, fill=(0, 0, 0), font=font)
    draw.text((left + len(cols) * cell_w - 210, 10), f"min={lo:.3f} median={mid:.3f} max={hi:.3f}", fill=(0, 0, 0), font=font)

    col_step = max(1, len(cols) // 16)
    for c_idx, col in enumerate(cols):
        x = left + c_idx * cell_w
        if c_idx % col_step == 0 or len(cols) <= 32:
            draw.text((x + 2, top - 18), str(col), fill=(0, 0, 0), font=font)

    row_step = max(1, len(rows) // 32)
    for r_idx, row in enumerate(rows):
        y = top + r_idx * cell_h
        if r_idx % row_step == 0 or len(rows) <= 64:
            draw.text((left - 48, y), str(row), fill=(0, 0, 0), font=font)
        for c_idx, col in enumerate(cols):
            x = left + c_idx * cell_w
            value = values.get((row, col))
            color = (238, 238, 238) if value is None else color_scale(value, lo, mid, hi)
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=color)
            if value is not None and cell_w >= 42 and cell_h >= 12:
                draw.text((x + 2, y + 1), f"{value:.2f}", fill=(0, 0, 0), font=font)

    # Legend.
    legend_x = left
    legend_y = top + len(rows) * cell_h + 20
    legend_w = min(360, len(cols) * cell_w)
    for i in range(legend_w):
        v = lo + (hi - lo) * i / max(1, legend_w - 1)
        draw.line((legend_x + i, legend_y, legend_x + i, legend_y + 12), fill=color_scale(v, lo, mid, hi))
    draw.text((legend_x, legend_y + 16), f"{lo:.3f}", fill=(0, 0, 0), font=font)
    draw.text((legend_x + legend_w // 2 - 20, legend_y + 16), f"{mid:.3f}", fill=(0, 0, 0), font=font)
    draw.text((legend_x + legend_w - 48, legend_y + 16), f"{hi:.3f}", fill=(0, 0, 0), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def render_h2d(grouped: dict[int, dict[tuple[int, int], list[float]]], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for size_mb, cells in sorted(grouped.items()):
        rows = sorted({cpu for cpu, _ in cells})
        cols = sorted({device for _, device in cells})
        values = {key: median(vals) for key, vals in cells.items()}
        out = out_dir / f"h2d_cpu_core_by_npu_{size_mb}MB.png"
        draw_heatmap(f"H2D GB/s by CPU core and NPU ({size_mb} MB)", "CPU core", "NPU id", rows, cols, values, out)
        outputs.append(out)
    return outputs


def render_d2d(grouped: dict[int, dict[tuple[int, int], list[float]]], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for size_mb, cells in sorted(grouped.items()):
        rows = sorted({cpu0 for cpu0, _ in cells})
        cols = sorted({cpu1 for _, cpu1 in cells})
        values = {key: median(vals) for key, vals in cells.items()}
        out = out_dir / f"d2d_rank0_core_by_rank1_core_{size_mb}MB.png"
        draw_heatmap(f"Two-card all_gather alg GB/s ({size_mb} MB)", "rank0 CPU core", "rank1 CPU core", rows, cols, values, out)
        outputs.append(out)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render core matrix benchmark results as PNG heatmaps")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/png"))
    args = parser.parse_args()

    _, metrics = load_metrics(args.paths)
    outputs = [*render_h2d(grouped_h2d(metrics), args.out_dir), *render_d2d(grouped_d2d(metrics), args.out_dir)]
    for output in outputs:
        print(output)
    if not outputs:
        print("No core matrix metrics found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

