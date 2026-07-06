#!/usr/bin/env python3
"""Create an Excel heatmap workbook from benchmark JSON outputs.

This writer uses only the Python standard library. It emits a small XLSX with
typed numeric cells and Excel conditional-format color scales, so it can run on
remote training nodes without installing openpyxl/xlsxwriter.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any
from xml.sax.saxutils import escape

from .summarize import load_metrics


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row: int, col: int, value: Any) -> str:
    ref = f"{col_name(col)}{row}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value):.12g}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def sheet_xml(rows: list[list[Any]], freeze_top_row: bool = True, heatmap_range: str | None = None) -> str:
    max_cols = max((len(row) for row in rows), default=1)
    widths: list[float] = []
    for col in range(max_cols):
        max_len = max((len(str(row[col])) for row in rows if col < len(row) and row[col] is not None), default=8)
        if col == 0:
            width = min(max(18, max_len + 2), 32)
        else:
            width = min(max(12, max_len + 2), 45)
        widths.append(width)
    sheet_rows: list[str] = []
    for r_idx, row in enumerate(rows, 1):
        cells = "".join(cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(row, 1))
        sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')
    cols = "".join(f'<col min="{idx}" max="{idx}" width="{widths[idx - 1]:.1f}" customWidth="1"/>' for idx in range(1, max_cols + 1))
    views = ""
    if freeze_top_row:
        views = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    conditional = ""
    if heatmap_range:
        conditional = f"""
<conditionalFormatting sqref="{heatmap_range}">
  <cfRule type="colorScale" priority="1">
    <colorScale>
      <cfvo type="min"/>
      <cfvo type="percentile" val="50"/>
      <cfvo type="max"/>
      <color rgb="FFF8696B"/>
      <color rgb="FFFFEB84"/>
      <color rgb="FF63BE7B"/>
    </colorScale>
  </cfRule>
</conditionalFormatting>"""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  {views}
  <cols>{cols}</cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  {conditional}
</worksheet>"""


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{sheets}</sheets>
</workbook>"""


def workbook_rels(sheet_count: int) -> str:
    rels = [
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    ]
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def content_types(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>"""


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>"""


def metric_value(metric: dict[str, Any]) -> tuple[str, float] | None:
    for field in ["alg_gbps", "rank_exchange_gbps", "gbps", "tflops"]:
        value = metric.get(field)
        if isinstance(value, (int, float)):
            return field, float(value)
    return None


def metric_label(metric: dict[str, Any], field: str) -> str:
    op = metric.get("operation") or metric.get("test")
    scope = metric.get("scope") or metric.get("direction") or ""
    size = metric.get("size_mb", metric.get("size_n", ""))
    unit_size = f"{size}MB" if metric.get("size_mb") is not None else str(size)
    return "|".join(str(part) for part in [metric.get("test"), op, scope, unit_size, field] if part not in ("", None))


def build_summary(metrics: list[dict[str, Any]]) -> tuple[list[list[Any]], list[list[Any]]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for metric in metrics:
        if metric.get("status") != "ok":
            continue
        pair = metric_value(metric)
        if pair is None:
            continue
        field, value = pair
        row_key = str(metric.get("bind_policy") or metric.get("profile") or "unknown")
        label = metric_label(metric, field)
        grouped[(row_key, label)].append(value)

    row_keys = sorted({key[0] for key in grouped})
    labels = sorted({key[1] for key in grouped})
    summary_rows = [["policy/profile", "metric", "count", "min", "median", "max", "max/min"]]
    for row_key in row_keys:
        for label in labels:
            values = grouped.get((row_key, label))
            if not values:
                continue
            min_v = min(values)
            max_v = max(values)
            summary_rows.append([row_key, label, len(values), min_v, median(values), max_v, max_v / min_v if min_v else None])

    heatmap_rows = [["policy/profile", *labels]]
    for row_key in row_keys:
        row: list[Any] = [row_key]
        for label in labels:
            values = grouped.get((row_key, label), [])
            row.append(median(values) if values else None)
        heatmap_rows.append(row)
    return summary_rows, heatmap_rows


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


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]], str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(sheets)))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("xl/workbook.xml", workbook_xml([sheet[0] for sheet in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(len(sheets)))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, (_, rows, heatmap_range) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows, heatmap_range=heatmap_range))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an XLSX heatmap from ClusterHealthDetect JSON results")
    parser.add_argument("paths", nargs="+", type=Path, help="results.json, rank JSONs, or result directories")
    parser.add_argument("--affinity-json", type=Path, default=None, help="Optional affinity.json from affinity_probe or numa_affinity_benchmark")
    parser.add_argument("--out", type=Path, default=Path("reports/cluster-health-heatmap.xlsx"))
    args = parser.parse_args()

    _, metrics = load_metrics(args.paths)
    summary_rows, heatmap_rows = build_summary(metrics)
    heatmap_range = None
    if len(heatmap_rows) > 1 and len(heatmap_rows[0]) > 1:
        heatmap_range = f"B2:{col_name(len(heatmap_rows[0]))}{len(heatmap_rows)}"
    sheets = [
        ("Heatmap", heatmap_rows, heatmap_range),
        ("Summary", summary_rows, None),
        ("Affinity", affinity_rows(args.affinity_json), None),
    ]
    write_xlsx(args.out, sheets)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
