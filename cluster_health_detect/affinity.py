"""CPU affinity and NUMA helpers.

The functions in this module intentionally use only the Python standard
library so they can run on stripped-down training nodes before torch is
healthy.
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CpuRecord:
    cpu: int
    online: bool | None
    allowed: bool | None
    bindable: bool | None
    numa_node: int | None
    socket_id: int | None
    core_id: int | None
    thread_siblings: list[int]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_cpu_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    cpus: list[int] = []
    for part in raw.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus: list[int] | set[int] | tuple[int, ...]) -> str:
    values = sorted(set(int(cpu) for cpu in cpus))
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = values[0]
    for cpu in values[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except Exception:
        return None


def read_int(path: Path) -> int | None:
    raw = read_text(path)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def current_allowed_cpus() -> set[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    return None


def set_current_affinity(cpus: list[int] | set[int]) -> tuple[bool, list[int], str | None]:
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        return False, [], "sched_setaffinity is unavailable"
    try:
        os.sched_setaffinity(0, set(cpus))  # type: ignore[attr-defined]
        actual = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        requested = sorted(set(cpus))
        ok = bool(set(requested) & set(actual))
        return ok, actual, None if ok else f"requested {requested}, actual {actual}"
    except Exception as exc:
        return False, [], repr(exc)


def cpu_online(cpu: int) -> bool | None:
    online_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/online")
    if not online_path.exists():
        return True
    value = read_int(online_path)
    if value is None:
        return None
    return value == 1


def cpu_numa_node(cpu: int) -> int | None:
    cpu_dir = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    for child in cpu_dir.glob("node[0-9]*"):
        try:
            return int(child.name.replace("node", ""))
        except ValueError:
            continue
    return None


def cpu_topology_ids(cpu: int) -> tuple[int | None, int | None, list[int]]:
    topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    socket_id = read_int(topo / "physical_package_id")
    core_id = read_int(topo / "core_id")
    siblings = parse_cpu_list(read_text(topo / "thread_siblings_list"))
    return socket_id, core_id, siblings


def present_cpus() -> list[int]:
    present = parse_cpu_list(read_text(Path("/sys/devices/system/cpu/present")))
    if present:
        return present
    count = os.cpu_count() or 0
    return list(range(count))


def collect_cpu_records(check_bindable: bool = True) -> tuple[list[CpuRecord], dict[str, Any]]:
    allowed = current_allowed_cpus()
    original = set(allowed) if allowed is not None else None
    records: list[CpuRecord] = []
    errors: list[str] = []
    for cpu in present_cpus():
        online = cpu_online(cpu)
        socket_id, core_id, siblings = cpu_topology_ids(cpu)
        bindable: bool | None = None
        reason: str | None = None
        is_allowed = cpu in allowed if allowed is not None else None
        if check_bindable:
            ok, actual, err = set_current_affinity({cpu})
            bindable = ok and cpu in actual
            reason = err
            if not bindable and reason is None:
                reason = f"actual affinity {actual}"
            if original is not None:
                restore_ok, _, restore_err = set_current_affinity(original)
                if not restore_ok:
                    errors.append(f"restore affinity failed after cpu {cpu}: {restore_err}")
        records.append(
            CpuRecord(
                cpu=cpu,
                online=online,
                allowed=is_allowed,
                bindable=bindable,
                numa_node=cpu_numa_node(cpu),
                socket_id=socket_id,
                core_id=core_id,
                thread_siblings=siblings,
                reason=reason,
            )
        )

    if original is not None:
        set_current_affinity(original)

    summary = summarize_records(records)
    summary.update(
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "allowed_cpus": sorted(allowed) if allowed is not None else None,
            "allowed_cpus_list": format_cpu_list(allowed or set()),
            "errors": errors,
        }
    )
    return records, summary


def summarize_records(records: list[CpuRecord]) -> dict[str, Any]:
    by_numa: dict[str, dict[str, Any]] = {}
    for record in records:
        key = "unknown" if record.numa_node is None else str(record.numa_node)
        bucket = by_numa.setdefault(key, {"cpus": [], "allowed_cpus": [], "bindable_cpus": []})
        bucket["cpus"].append(record.cpu)
        if record.allowed:
            bucket["allowed_cpus"].append(record.cpu)
        if record.bindable:
            bucket["bindable_cpus"].append(record.cpu)
    for bucket in by_numa.values():
        for key in ["cpus", "allowed_cpus", "bindable_cpus"]:
            bucket[key] = sorted(bucket[key])
            bucket[f"{key}_list"] = format_cpu_list(bucket[key])
        bucket["cpu_count"] = len(bucket["cpus"])
        bucket["allowed_count"] = len(bucket["allowed_cpus"])
        bucket["bindable_count"] = len(bucket["bindable_cpus"])
    return {
        "cpu_count": len(records),
        "online_count": sum(1 for r in records if r.online is True),
        "allowed_count": sum(1 for r in records if r.allowed is True),
        "bindable_count": sum(1 for r in records if r.bindable is True),
        "unavailable_count": sum(1 for r in records if r.bindable is False),
        "numa_nodes": by_numa,
    }


def bindable_cpus(records: list[CpuRecord]) -> list[int]:
    values = [record.cpu for record in records if record.bindable is True]
    if values:
        return sorted(values)
    return sorted(record.cpu for record in records if record.allowed is not False and record.online is not False)


def numa_groups(records: list[CpuRecord]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for record in records:
        if record.numa_node is None:
            continue
        if record.bindable is False or record.allowed is False or record.online is False:
            continue
        groups.setdefault(record.numa_node, []).append(record.cpu)
    return {node: sorted(cpus) for node, cpus in sorted(groups.items()) if cpus}


def split_for_rank(cpus: list[int], rank: int, world: int, cpus_per_rank: int = 0) -> list[int]:
    if not cpus:
        return []
    world = max(1, world)
    rank = rank % world
    chunks: list[list[int]] = [[] for _ in range(world)]
    for idx, cpu in enumerate(cpus):
        chunks[idx % world].append(cpu)
    selected = chunks[rank] or [cpus[rank % len(cpus)]]
    if cpus_per_rank > 0:
        selected = selected[:cpus_per_rank] or [cpus[rank % len(cpus)]]
    return sorted(selected)


def resolve_auto_policies(records: list[CpuRecord]) -> list[str]:
    policies = ["none", "local_rank"]
    groups = numa_groups(records)
    for node in sorted(groups):
        policies.append(f"numa:{node}")
        policies.append(f"numa:{node}:shard")
    return policies


def resolve_policy_cpus(policy: str, records: list[CpuRecord], local_rank: int, local_world_size: int, cpus_per_rank: int) -> tuple[list[int], dict[str, Any]]:
    policy = policy.strip()
    all_bindable = bindable_cpus(records)
    groups = numa_groups(records)
    meta: dict[str, Any] = {"policy": policy, "numa_node": None, "mode": None}
    if policy == "none":
        meta["mode"] = "none"
        return [], meta
    if policy == "all":
        meta["mode"] = "all"
        return all_bindable, meta
    if policy == "local_rank":
        meta["mode"] = "local_rank"
        return split_for_rank(all_bindable, local_rank, local_world_size, cpus_per_rank), meta
    if policy.startswith("cpu:"):
        cpus = parse_cpu_list(policy.split(":", 1)[1])
        meta["mode"] = "explicit_cpu_list"
        return cpus, meta
    if policy.startswith("numa:"):
        parts = policy.split(":")
        if len(parts) < 2:
            raise ValueError(f"invalid NUMA policy: {policy}")
        node = int(parts[1])
        cpus = groups.get(node, [])
        meta["numa_node"] = node
        if len(parts) >= 3 and parts[2] == "shard":
            meta["mode"] = "numa_shard"
            return split_for_rank(cpus, local_rank, local_world_size, cpus_per_rank), meta
        meta["mode"] = "numa_all"
        if cpus_per_rank > 0:
            cpus = cpus[:cpus_per_rank]
        return cpus, meta
    raise ValueError(f"unknown affinity policy: {policy}")


def apply_policy(policy: str, records: list[CpuRecord], local_rank: int, local_world_size: int, cpus_per_rank: int, original: set[int] | None) -> dict[str, Any]:
    requested, meta = resolve_policy_cpus(policy, records, local_rank, local_world_size, cpus_per_rank)
    if policy == "none":
        if original is None:
            return {**meta, "status": "ok", "requested_cpus": [], "actual_cpus": [], "actual_cpus_list": "", "reason": "affinity unsupported"}
        ok, actual, err = set_current_affinity(original)
        return {
            **meta,
            "status": "ok" if ok else "error",
            "requested_cpus": sorted(original),
            "requested_cpus_list": format_cpu_list(original),
            "actual_cpus": actual,
            "actual_cpus_list": format_cpu_list(actual),
            "reason": err,
        }
    if not requested:
        return {**meta, "status": "error", "requested_cpus": [], "actual_cpus": [], "actual_cpus_list": "", "reason": "policy resolved to no CPUs"}
    ok, actual, err = set_current_affinity(requested)
    return {
        **meta,
        "status": "ok" if ok else "error",
        "requested_cpus": requested,
        "requested_cpus_list": format_cpu_list(requested),
        "actual_cpus": actual,
        "actual_cpus_list": format_cpu_list(actual),
        "reason": err,
    }

