#!/usr/bin/env python3
"""Torchrun entrypoint for CPU/NPU and communication health probes.

The benchmark intentionally records failures as structured metrics. On shared
training clusters a broken torch_npu import, missing HCCL library, or invalid
CPU affinity map is itself a useful health signal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_tests(raw: str) -> set[str]:
    tests = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "all" in tests:
        return {"affinity", "cpu", "device", "h2d", "d2d", "collective"}
    return tests


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def run_cmd(cmd: list[str], timeout: int = 5) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[:20000],
            "stderr": proc.stderr[:20000],
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cmd": cmd, "error": repr(exc)}


def env_snapshot() -> dict[str, Any]:
    keys = [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "HCCL_IF_IP",
        "GLOO_SOCKET_IFNAME",
        "NCCL_SOCKET_IFNAME",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
    ]
    return {key: os.environ.get(key) for key in keys if key in os.environ}


def import_torch_stack() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_ok": False,
        "torch_error": None,
        "torch_npu_ok": False,
        "torch_npu_error": None,
        "cuda_available": False,
        "cuda_count": 0,
        "npu_available": False,
        "npu_count": 0,
    }
    try:
        import torch  # type: ignore

        result["torch_ok"] = True
        result["torch_version"] = getattr(torch, "__version__", "unknown")
        result["torch_distributed_available"] = bool(torch.distributed.is_available())
        try:
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_count"] = int(torch.cuda.device_count())
            result["cuda_version"] = getattr(torch.version, "cuda", None)
        except Exception as exc:
            result["cuda_error"] = repr(exc)

        try:
            import torch_npu  # type: ignore  # noqa: F401

            result["torch_npu_ok"] = True
            result["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
            result["npu_available"] = bool(torch.npu.is_available())
            result["npu_count"] = int(torch.npu.device_count())
        except Exception as exc:
            result["torch_npu_error"] = repr(exc)
    except Exception as exc:
        result["torch_error"] = repr(exc)
        result["torch_traceback"] = traceback.format_exc(limit=8)
    return result


def choose_device(torch: Any, stack: dict[str, Any], requested: str, local_rank: int) -> tuple[str, str | None]:
    if requested != "auto":
        device_kind = requested
    elif stack.get("npu_available") and stack.get("npu_count", 0) > 0:
        device_kind = "npu"
    elif stack.get("cuda_available") and stack.get("cuda_count", 0) > 0:
        device_kind = "cuda"
    else:
        device_kind = "cpu"

    if device_kind == "npu":
        count = int(stack.get("npu_count") or 0)
        if count <= 0:
            return "cpu", "requested npu but no NPU is visible"
        index = local_rank % count
        torch.npu.set_device(index)
        return f"npu:{index}", None
    if device_kind == "cuda":
        count = int(stack.get("cuda_count") or 0)
        if count <= 0:
            return "cpu", "requested cuda but no CUDA GPU is visible"
        index = local_rank % count
        torch.cuda.set_device(index)
        return f"cuda:{index}", None
    return "cpu", None


def device_kind(device: str) -> str:
    return device.split(":", 1)[0]


def synchronize(torch: Any, device: str) -> None:
    kind = device_kind(device)
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "npu":
        torch.npu.synchronize()


def torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[name.lower()]


def rank_info() -> dict[str, Any]:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def base_metric(info: dict[str, Any], profile: str, test: str, **extra: Any) -> dict[str, Any]:
    metric = {
        "ts": now_utc(),
        "profile": profile,
        "test": test,
        **info,
    }
    metric.update(extra)
    return metric


def cpu_affinity_probe(info: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    cpu_count = os.cpu_count()
    status: dict[str, Any] = {
        "cpu_count": cpu_count,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "lscpu_e": run_cmd(["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"], timeout=5),
        "proc_status": None,
        "allowed_cpus": None,
        "bindable_cpus": [],
        "unavailable_cpus": [],
        "errors": [],
    }
    try:
        status["proc_status"] = Path("/proc/self/status").read_text(errors="replace")[:20000]
    except Exception as exc:
        status["errors"].append(f"read /proc/self/status failed: {exc!r}")

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity") and cpu_count:
        original = set(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        status["allowed_cpus"] = sorted(original)
        for cpu in range(cpu_count):
            try:
                os.sched_setaffinity(0, {cpu})  # type: ignore[attr-defined]
                actual = set(os.sched_getaffinity(0))  # type: ignore[attr-defined]
                if cpu in actual:
                    status["bindable_cpus"].append(cpu)
                else:
                    status["unavailable_cpus"].append(cpu)
            except Exception as exc:
                status["unavailable_cpus"].append(cpu)
                status["errors"].append(f"cpu {cpu}: {exc!r}")
            finally:
                try:
                    os.sched_setaffinity(0, original)  # type: ignore[attr-defined]
                except Exception as exc:
                    status["errors"].append(f"restore affinity failed: {exc!r}")
        status["bindable_count"] = len(status["bindable_cpus"])
        status["unavailable_count"] = len(status["unavailable_cpus"])
    else:
        status["errors"].append("os.sched_getaffinity/sched_setaffinity is unavailable on this platform")

    metrics.append(base_metric(info, profile, "affinity", status="ok", **status))
    return metrics


def cpu_compute_probe(torch: Any, info: dict[str, Any], profile: str, sizes: list[int], seconds: float, threads: int | None) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if torch is None:
        return [base_metric(info, profile, "cpu", status="skip", reason="torch import failed")]

    old_threads = None
    try:
        old_threads = torch.get_num_threads()
        if threads:
            torch.set_num_threads(max(1, threads))
    except Exception:
        pass

    for n in sizes:
        try:
            a = torch.randn((n, n), dtype=torch.float32, device="cpu")
            b = torch.randn((n, n), dtype=torch.float32, device="cpu")
            _ = a @ b
            start = time.perf_counter()
            iters = 0
            while time.perf_counter() - start < seconds:
                _ = a @ b
                iters += 1
            elapsed = max(time.perf_counter() - start, 1e-12)
            flops = 2.0 * n * n * n * iters
            metrics.append(
                base_metric(
                    info,
                    profile,
                    "cpu",
                    status="ok",
                    operation="matmul",
                    dtype="float32",
                    size_n=n,
                    iterations=iters,
                    seconds=elapsed,
                    tflops=flops / elapsed / 1e12,
                    torch_threads=torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
                )
            )
        except Exception as exc:
            metrics.append(base_metric(info, profile, "cpu", status="error", operation="matmul", size_n=n, error=repr(exc)))

    if old_threads:
        try:
            torch.set_num_threads(old_threads)
        except Exception:
            pass
    return metrics


def device_compute_probe(
    torch: Any,
    info: dict[str, Any],
    profile: str,
    device: str,
    dtype_name: str,
    sizes: list[int],
    seconds: float,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if torch is None:
        return [base_metric(info, profile, "device", status="skip", reason="torch import failed")]
    if device_kind(device) == "cpu":
        return [base_metric(info, profile, "device", status="skip", reason="no accelerator device selected", device=device)]

    dtype = torch_dtype(torch, dtype_name)
    for n in sizes:
        try:
            a = torch.randn((n, n), dtype=dtype, device=device)
            b = torch.randn((n, n), dtype=dtype, device=device)
            for _ in range(3):
                _ = a @ b
            synchronize(torch, device)
            start = time.perf_counter()
            iters = 0
            while time.perf_counter() - start < seconds:
                _ = a @ b
                iters += 1
            synchronize(torch, device)
            elapsed = max(time.perf_counter() - start, 1e-12)
            flops = 2.0 * n * n * n * iters
            metrics.append(
                base_metric(
                    info,
                    profile,
                    "device",
                    status="ok",
                    operation="matmul",
                    device=device,
                    dtype=dtype_name,
                    size_n=n,
                    iterations=iters,
                    seconds=elapsed,
                    tflops=flops / elapsed / 1e12,
                )
            )
        except Exception as exc:
            metrics.append(
                base_metric(
                    info,
                    profile,
                    "device",
                    status="error",
                    operation="matmul",
                    device=device,
                    dtype=dtype_name,
                    size_n=n,
                    error=repr(exc),
                )
            )
    return metrics


def bandwidth_probe(
    torch: Any,
    info: dict[str, Any],
    profile: str,
    device: str,
    test: str,
    dtype_name: str,
    sizes_mb: list[int],
    iters: int,
    warmup: int,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if torch is None:
        return [base_metric(info, profile, test, status="skip", reason="torch import failed")]
    if device_kind(device) == "cpu":
        return [base_metric(info, profile, test, status="skip", reason="no accelerator device selected", device=device)]

    dtype = torch_dtype(torch, dtype_name)
    element_size = torch.tensor([], dtype=dtype).element_size()
    for size_mb in sizes_mb:
        numel = max(1, size_mb * 1024 * 1024 // element_size)
        try:
            if test == "h2d":
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
                extra = {"pinned": pinned, "direction": "host_to_device"}
            elif test == "d2d":
                src = torch.empty(numel, dtype=dtype, device=device)
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
                extra = {"direction": "device_to_device_same_rank"}
            else:
                raise ValueError(f"unknown bandwidth test {test}")
            metrics.append(
                base_metric(
                    info,
                    profile,
                    test,
                    status="ok",
                    operation="copy",
                    device=device,
                    dtype=dtype_name,
                    size_mb=size_mb,
                    iterations=iters,
                    seconds=elapsed,
                    gbps=bytes_moved / elapsed / 1e9,
                    **extra,
                )
            )
        except Exception as exc:
            metrics.append(
                base_metric(
                    info,
                    profile,
                    test,
                    status="error",
                    operation="copy",
                    device=device,
                    dtype=dtype_name,
                    size_mb=size_mb,
                    error=repr(exc),
                )
            )
    return metrics


def init_dist(torch: Any, backend: str, device: str) -> tuple[bool, str | None]:
    if torch is None:
        return False, "torch import failed"
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, "WORLD_SIZE <= 1"
    if not torch.distributed.is_available():
        return False, "torch.distributed is not available"
    if torch.distributed.is_initialized():
        try:
            return True, str(torch.distributed.get_backend())
        except Exception:
            return True, "initialized"

    selected = backend
    if backend == "auto":
        kind = device_kind(device)
        if kind == "npu":
            selected = "hccl"
        elif kind == "cuda":
            selected = "nccl"
        else:
            selected = "gloo"
    try:
        torch.distributed.init_process_group(
            backend=selected,
            timeout=dt.timedelta(seconds=int(os.environ.get("CHD_DIST_TIMEOUT", "300"))),
        )
        return True, selected
    except Exception as exc:
        return False, f"init_process_group({selected}) failed: {exc!r}"


def gather_hosts(torch: Any, info: dict[str, Any]) -> list[dict[str, Any]]:
    world = info["world_size"]
    payload = {"rank": info["rank"], "host": info["host"], "local_rank": info["local_rank"]}
    values: list[Any] = [None for _ in range(world)]
    torch.distributed.all_gather_object(values, payload)
    return [dict(v) for v in values if v is not None]


def make_host_groups(torch: Any, host_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[int]]:
    by_host: dict[str, list[int]] = {}
    for row in host_rows:
        by_host.setdefault(row["host"], []).append(int(row["rank"]))
    local_group = None
    local_ranks: list[int] = []
    for host in sorted(by_host):
        ranks = sorted(by_host[host])
        group = torch.distributed.new_group(ranks=ranks)
        if socket.gethostname() == host:
            local_group = group
            local_ranks = ranks
    return local_group, local_ranks


def collective_probe(
    torch: Any,
    info: dict[str, Any],
    profile: str,
    device: str,
    dtype_name: str,
    sizes_mb: list[int],
    iters: int,
    warmup: int,
    backend: str,
    enable_p2p: bool,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    ok, backend_or_error = init_dist(torch, backend, device)
    if not ok:
        return [base_metric(info, profile, "collective", status="skip", reason=backend_or_error, device=device)]

    dist = torch.distributed
    selected_backend = backend_or_error
    try:
        host_rows = gather_hosts(torch, info)
        local_group, local_ranks = make_host_groups(torch, host_rows)
    except Exception as exc:
        host_rows = [{"rank": info["rank"], "host": info["host"], "local_rank": info["local_rank"]}]
        local_group = None
        local_ranks = []
        metrics.append(base_metric(info, profile, "collective", status="error", operation="gather_hosts", error=repr(exc)))

    dtype = torch_dtype(torch, dtype_name)
    element_size = torch.tensor([], dtype=dtype).element_size()

    def run_all_gather(scope: str, group: Any, group_size: int) -> None:
        if group_size <= 1:
            metrics.append(
                base_metric(info, profile, "collective", status="skip", operation="all_gather", scope=scope, reason="group size <= 1")
            )
            return
        for size_mb in sizes_mb:
            numel = max(1, size_mb * 1024 * 1024 // element_size)
            try:
                tensor = torch.empty(numel, dtype=dtype, device=device)
                output = [torch.empty_like(tensor) for _ in range(group_size)]
                for _ in range(warmup):
                    dist.all_gather(output, tensor, group=group)
                synchronize(torch, device)
                dist.barrier(group=group)
                start = time.perf_counter()
                for _ in range(iters):
                    dist.all_gather(output, tensor, group=group)
                synchronize(torch, device)
                dist.barrier(group=group)
                elapsed = max(time.perf_counter() - start, 1e-12)
                bytes_per_rank = numel * element_size
                avg_seconds = elapsed / iters
                metrics.append(
                    base_metric(
                        info,
                        profile,
                        "collective",
                        status="ok",
                        operation="all_gather",
                        scope=scope,
                        backend=selected_backend,
                        device=device,
                        dtype=dtype_name,
                        group_size=group_size,
                        size_mb=size_mb,
                        iterations=iters,
                        seconds=elapsed,
                        avg_seconds=avg_seconds,
                        alg_gbps=bytes_per_rank / avg_seconds / 1e9,
                        rank_exchange_gbps=bytes_per_rank * (group_size - 1) / avg_seconds / 1e9,
                    )
                )
            except Exception as exc:
                metrics.append(
                    base_metric(
                        info,
                        profile,
                        "collective",
                        status="error",
                        operation="all_gather",
                        scope=scope,
                        backend=selected_backend,
                        device=device,
                        dtype=dtype_name,
                        size_mb=size_mb,
                        error=repr(exc),
                    )
                )

    run_all_gather("global", None, info["world_size"])
    if local_group is not None and local_ranks:
        run_all_gather("local_host", local_group, len(local_ranks))

    if enable_p2p:
        metrics.extend(p2p_probe(torch, info, profile, device, dtype_name, sizes_mb, iters, warmup, host_rows, selected_backend))
    else:
        metrics.append(
            base_metric(
                info,
                profile,
                "collective",
                status="skip",
                operation="sendrecv",
                reason="p2p disabled; set --enable-p2p after validating backend support",
            )
        )
    return metrics


def p2p_probe(
    torch: Any,
    info: dict[str, Any],
    profile: str,
    device: str,
    dtype_name: str,
    sizes_mb: list[int],
    iters: int,
    warmup: int,
    host_rows: list[dict[str, Any]],
    backend: str | None,
) -> list[dict[str, Any]]:
    dist = torch.distributed
    metrics: list[dict[str, Any]] = []
    by_host: dict[str, list[int]] = {}
    for row in host_rows:
        by_host.setdefault(row["host"], []).append(int(row["rank"]))
    dtype = torch_dtype(torch, dtype_name)
    element_size = torch.tensor([], dtype=dtype).element_size()

    def run_pair(scope: str, rank_a: int, rank_b: int) -> None:
        active = info["rank"] in {rank_a, rank_b}
        dist.barrier()
        for size_mb in sizes_mb:
            numel = max(1, size_mb * 1024 * 1024 // element_size)
            if active:
                src = torch.empty(numel, dtype=dtype, device=device)
                dst = torch.empty(numel, dtype=dtype, device=device)
                peer = rank_b if info["rank"] == rank_a else rank_a
                for _ in range(warmup):
                    req_recv = dist.irecv(dst, src=peer)
                    req_send = dist.isend(src, dst=peer)
                    req_send.wait()
                    req_recv.wait()
                synchronize(torch, device)
            dist.barrier()
            if active:
                start = time.perf_counter()
                for _ in range(iters):
                    req_recv = dist.irecv(dst, src=peer)
                    req_send = dist.isend(src, dst=peer)
                    req_send.wait()
                    req_recv.wait()
                synchronize(torch, device)
                elapsed = max(time.perf_counter() - start, 1e-12)
                bytes_moved = numel * element_size * iters
                metrics.append(
                    base_metric(
                        info,
                        profile,
                        "collective",
                        status="ok",
                        operation="sendrecv",
                        scope=scope,
                        backend=backend,
                        peer_rank=peer,
                        device=device,
                        dtype=dtype_name,
                        size_mb=size_mb,
                        iterations=iters,
                        seconds=elapsed,
                        gbps=bytes_moved / elapsed / 1e9,
                    )
                )
            dist.barrier()

    try:
        local_pair_count = 0
        for host in sorted(by_host):
            ranks = sorted(by_host[host])
            if len(ranks) >= 2:
                local_pair_count += 1
                run_pair("local_host_pair", ranks[0], ranks[1])
        if local_pair_count == 0:
            metrics.append(base_metric(info, profile, "collective", status="skip", operation="sendrecv", scope="local_host_pair", reason="no host has >=2 ranks"))

        if len(by_host) >= 2:
            host_names = sorted(by_host)
            rank_a = sorted(by_host[host_names[0]])[0]
            rank_b = sorted(by_host[host_names[1]])[0]
            run_pair("inter_host_pair", rank_a, rank_b)
        else:
            metrics.append(base_metric(info, profile, "collective", status="skip", operation="sendrecv", scope="inter_host_pair", reason="only one host"))
    except Exception as exc:
        metrics.append(
            base_metric(
                info,
                profile,
                "collective",
                status="error",
                operation="sendrecv",
                scope="inter_host_pair",
                backend=backend,
                device=device,
                error=repr(exc),
            )
        )
    return metrics


def cpu_burn(stop: mp.Event) -> None:
    seed = b"cluster-health-detect"
    while not stop.is_set():
        seed = hashlib.sha256(seed).digest()


class CpuLoad:
    def __init__(self, workers: int) -> None:
        self.workers = max(0, workers)
        self.stop = mp.Event()
        self.children: list[mp.Process] = []

    def __enter__(self) -> "CpuLoad":
        for _ in range(self.workers):
            proc = mp.Process(target=cpu_burn, args=(self.stop,), daemon=True)
            proc.start()
            self.children.append(proc)
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        for proc in self.children:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()


class DeviceLoad:
    def __init__(self, torch: Any, device: str, dtype_name: str, n: int) -> None:
        self.torch = torch
        self.device = device
        self.dtype_name = dtype_name
        self.n = n
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def __enter__(self) -> "DeviceLoad":
        if self.torch is None or device_kind(self.device) == "cpu":
            return self
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            dtype = torch_dtype(self.torch, self.dtype_name)
            a = self.torch.randn((self.n, self.n), dtype=dtype, device=self.device)
            b = self.torch.randn((self.n, self.n), dtype=dtype, device=self.device)
            while not self.stop.is_set():
                _ = a @ b
                synchronize(self.torch, self.device)
        except Exception as exc:
            self.error = repr(exc)

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)


def parse_profile(raw: str) -> tuple[str, int, bool]:
    name = raw.strip()
    if not name or name == "idle":
        return "idle", 0, False
    if name.startswith("cpu"):
        parts = name.split(":", 1)
        return name, int(parts[1]) if len(parts) == 2 and parts[1] else max(1, (os.cpu_count() or 2) // 8), False
    if name.startswith("device") or name.startswith("npu") or name.startswith("gpu"):
        return name, 0, True
    raise ValueError(f"unsupported load profile: {raw}")


def rank_barrier(torch: Any) -> None:
    try:
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass


def gather_metrics(torch: Any, metrics: list[dict[str, Any]], info: dict[str, Any]) -> list[dict[str, Any]]:
    if torch is None or info["world_size"] <= 1:
        return metrics
    try:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return metrics
        gathered: list[Any] = [None for _ in range(info["world_size"])]
        torch.distributed.all_gather_object(gathered, metrics)
        merged: list[dict[str, Any]] = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
        return merged
    except Exception:
        return metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize_for_stdout(metrics: list[dict[str, Any]]) -> str:
    ok = [m for m in metrics if m.get("status") == "ok"]
    errors = [m for m in metrics if m.get("status") == "error"]
    lines = [f"metrics={len(metrics)} ok={len(ok)} errors={len(errors)}"]
    for key in ["tflops", "gbps", "alg_gbps", "rank_exchange_gbps"]:
        vals = [float(m[key]) for m in ok if key in m and isinstance(m.get(key), (int, float))]
        if vals:
            lines.append(f"{key}: max={max(vals):.3f} median={median(vals):.3f} p10={quantile(vals, 0.1):.3f}")
    return "; ".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Torchrun CPU/NPU cluster health and bandwidth benchmark")
    parser.add_argument("--out-dir", default="results/latest", help="Directory for JSON outputs")
    parser.add_argument("--backend", default="auto", choices=["auto", "hccl", "nccl", "gloo"], help="Distributed backend")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--tests", default="all", help="Comma-separated: affinity,cpu,device,h2d,d2d,collective,all")
    parser.add_argument("--profiles", default="idle,cpu:2,device", help="Comma-separated load profiles: idle,cpu:N,device")
    parser.add_argument("--sizes-mb", default="16,64,256", help="Tensor sizes for copy/collective probes")
    parser.add_argument("--cpu-sizes", default="512,1024,2048", help="CPU matmul N values")
    parser.add_argument("--device-sizes", default="1024,2048,4096", help="Device matmul N values")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds per compute size")
    parser.add_argument("--iters", type=int, default=20, help="Iterations for copy/collective probes")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--cpu-threads", type=int, default=0, help="torch CPU threads per rank; 0 keeps default")
    parser.add_argument("--device-load-size", type=int, default=2048, help="Matmul N for background device load")
    parser.add_argument("--enable-p2p", action="store_true", help="Enable distributed send/recv pair probes. Some HCCL stacks may be slow or unsupported.")
    args = parser.parse_args()

    info = rank_info()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = import_torch_stack()
    torch = None
    if stack["torch_ok"]:
        import torch as torch_module  # type: ignore

        torch = torch_module

    device = "cpu"
    device_warning = None
    if torch is not None:
        device, device_warning = choose_device(torch, stack, args.device, info["local_rank"])

    meta = {
        "created_at": now_utc(),
        "argv": sys.argv,
        "rank_info": info,
        "env": env_snapshot(),
        "torch_stack": stack,
        "selected_device": device,
        "device_warning": device_warning,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "npu_smi_head": run_cmd(["npu-smi", "info"], timeout=10),
    }

    metrics: list[dict[str, Any]] = [
        base_metric(info, "meta", "environment", status="ok", selected_device=device, torch_stack=stack, device_warning=device_warning)
    ]
    tests = parse_tests(args.tests)
    sizes_mb = parse_csv_ints(args.sizes_mb)
    cpu_sizes = parse_csv_ints(args.cpu_sizes)
    device_sizes = parse_csv_ints(args.device_sizes)

    for raw_profile in [p for p in args.profiles.split(",") if p.strip()]:
        profile, cpu_workers, use_device_load = parse_profile(raw_profile)
        with CpuLoad(cpu_workers), DeviceLoad(torch, device, args.dtype, args.device_load_size) if use_device_load else nullcontext():
            if torch is not None:
                # Initialize distributed before collective probes. This also lets
                # later barriers align load profiles across ranks.
                if "collective" in tests:
                    ok, _ = init_dist(torch, args.backend, device)
                    if ok:
                        rank_barrier(torch)
            if "affinity" in tests:
                metrics.extend(cpu_affinity_probe(info, profile))
            if "cpu" in tests:
                metrics.extend(cpu_compute_probe(torch, info, profile, cpu_sizes, args.seconds, args.cpu_threads or None))
            if "device" in tests:
                metrics.extend(device_compute_probe(torch, info, profile, device, args.dtype, device_sizes, args.seconds))
            if "h2d" in tests:
                metrics.extend(bandwidth_probe(torch, info, profile, device, "h2d", args.dtype, sizes_mb, args.iters, args.warmup))
            if "d2d" in tests:
                metrics.extend(bandwidth_probe(torch, info, profile, device, "d2d", args.dtype, sizes_mb, args.iters, args.warmup))
            if "collective" in tests:
                metrics.extend(
                    collective_probe(
                        torch,
                        info,
                        profile,
                        device,
                        args.dtype,
                        sizes_mb,
                        args.iters,
                        args.warmup,
                        args.backend,
                        args.enable_p2p,
                    )
                )
            rank_barrier(torch)

    write_json(out_dir / f"rank_{info['rank']:05d}.json", {"meta": meta, "metrics": metrics})
    merged = gather_metrics(torch, metrics, info)
    if info["rank"] == 0:
        write_json(out_dir / "results.json", {"meta": meta, "metrics": merged})
        print(summarize_for_stdout(merged), flush=True)

    try:
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    return 0


class nullcontext:
    def __enter__(self) -> "nullcontext":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
