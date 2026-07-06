#!/usr/bin/env python3
"""Probe whether every visible CPU can be targeted by sched_setaffinity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .affinity import collect_cpu_records
from .benchmark import now_utc


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe CPU affinity feasibility and NUMA topology")
    parser.add_argument("--out", type=Path, default=Path("results/affinity/affinity.json"))
    parser.add_argument("--no-bind-check", action="store_true", help="Only collect topology; do not try sched_setaffinity per CPU")
    args = parser.parse_args()

    records, summary = collect_cpu_records(check_bindable=not args.no_bind_check)
    payload = {
        "created_at": now_utc(),
        "summary": summary,
        "cpus": [record.to_dict() for record in records],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "cpu_count={cpu_count} allowed={allowed_count} bindable={bindable_count} unavailable={unavailable_count} out={out}".format(
            out=args.out,
            **summary,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

