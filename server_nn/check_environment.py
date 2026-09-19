"""Read-only CPU, memory, and software preflight for the portable CPU study."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

GIB = 1024 ** 3
EXPECTED_PYTHON = "3.13.11"
EXPECTED_PACKAGES = {
    "numpy": "2.2.6", "pandas": "2.3.3", "torch": "2.14.0",
    "scikit-learn": "1.8.0", "scipy": "1.16.3", "joblib": "1.5.3",
    "threadpoolctl": "3.5.0", "filelock": "3.20.0",
    "psutil": "7.0.0", "matplotlib": "3.10.8",
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def effective_cpu_count(logical, affinity=None, environ=None):
    """Use the tightest known allocation; a host CPU count is not an allocation."""
    environ = os.environ if environ is None else environ
    limits = {"host_logical": int(logical or 1)}
    if affinity:
        limits["process_affinity"] = len(affinity)
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = environ.get(name, "")
        if raw.isdigit() and int(raw) > 0:
            limits[name] = int(raw)
    return min(limits.values()), limits


def slurm_memory_bytes(raw):
    """Slurm memory environment variables use MiB when no suffix is present."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)\s*", str(raw), re.I)
    if not match:
        return None
    value = float(match.group(1))
    powers = {"K": 1, "": 2, "M": 2, "G": 3, "T": 4}
    result = int(value * 1024 ** powers[match.group(2).upper()])
    return result if result > 0 else None


def cgroup_memory_limits():
    """Read common cgroup v2/v1 memory limits, including the current cgroup."""
    pairs = [(Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
             (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
              Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))]
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                _, controllers, relative = line.split(":", 2)
                # Traversal components are not used as filesystem paths.
                parts = [part for part in relative.split("/") if part]
                if ".." in parts:
                    continue
                if controllers == "":
                    anchor = Path("/sys/fs/cgroup")
                    names = ("memory.max", "memory.current")
                elif "memory" in controllers.split(","):
                    anchor = Path("/sys/fs/cgroup/memory")
                    names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
                else:
                    continue
                base = anchor.joinpath(*parts)
                # A job's limit may live above its step/process cgroup.
                while base.is_relative_to(anchor):
                    pairs.append((base / names[0], base / names[1]))
                    if base == anchor:
                        break
                    base = base.parent
        except (OSError, ValueError):
            pass
    limits = []
    for maximum, current in dict.fromkeys(pairs):
        try:
            limit = int(maximum.read_text().strip())
            usage = int(current.read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < (1 << 60):
            limits.append({"source": str(maximum), "limit_bytes": limit,
                           "remaining_bytes": max(0, limit - usage)})
    return limits


def memory_budget(total, available, cpus, environ=None, cgroups=()):
    environ = os.environ if environ is None else environ
    limits = {"host_total": int(total)}
    remaining = {"host_available": int(available)}
    node = slurm_memory_bytes(environ.get("SLURM_MEM_PER_NODE", ""))
    per_cpu = slurm_memory_bytes(environ.get("SLURM_MEM_PER_CPU", ""))
    if node:
        limits["SLURM_MEM_PER_NODE"] = node
    elif per_cpu:
        limits["SLURM_MEM_PER_CPU_times_allowed_cpus"] = per_cpu * cpus
    for item in cgroups:
        limits[item["source"]] = int(item["limit_bytes"])
        remaining[item["source"]] = int(item["remaining_bytes"])
    return min(*limits.values(), *remaining.values()), limits, remaining


def inspect_environment(workers=24, threads=2, memory_per_worker_gib=6.0,
                        reserve_gib=24.0, allow_runtime_difference=False):
    errors, warnings = [], []
    versions = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual is None:
            errors.append(f"Missing package: {name}=={expected}")
        elif actual.split("+", 1)[0] != expected:
            message = f"Runtime differs: {name} is {actual}, expected {expected}"
            (warnings if allow_runtime_difference else errors).append(message)
    if platform.python_version() != EXPECTED_PYTHON:
        message = f"Python is {platform.python_version()}, source used {EXPECTED_PYTHON}"
        (warnings if allow_runtime_difference else errors).append(message)
    try:
        import psutil
    except ImportError:
        return {"ok": False, "errors": errors, "warnings": warnings,
                "versions": versions, "python": platform.python_version()}
    affinity = None
    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, OSError, psutil.Error):
        if hasattr(os, "sched_getaffinity"):
            affinity = sorted(os.sched_getaffinity(0))
    cpus, cpu_limits = effective_cpu_count(os.cpu_count(), affinity)
    vm = psutil.virtual_memory()
    usable, memory_limits, memory_remaining = memory_budget(
        vm.total, vm.available, cpus, cgroups=cgroup_memory_limits())
    if workers * threads > cpus:
        errors.append(f"Requested {workers * threads} compute threads exceed {cpus} allowed CPUs")
    estimated_bytes = (workers * memory_per_worker_gib + reserve_gib) * GIB
    if estimated_bytes > usable:
        errors.append(f"Planning memory {estimated_bytes / GIB:.1f} GiB exceeds "
                      f"currently usable/allotted {usable / GIB:.1f} GiB; reduce workers, "
                      "or use a measured NN worker budget")
    if sys.platform.startswith("linux") and not affinity and "SLURM_CPUS_PER_TASK" not in os.environ:
        warnings.append("CPU affinity/allocation was unavailable; confirm the host allocation manually")
    warnings.append("Memory is a planning estimate, not an observed NN peak; benchmark before scaling up")
    if allow_runtime_difference:
        warnings.append("Runtime differences require train --no-reuse-local and a separate output directory")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "allowed_cpus": cpus, "cpu_limits": cpu_limits,
            "memory_limits_bytes": memory_limits, "memory_remaining_bytes": memory_remaining,
            "usable_memory_gib": usable / GIB, "workers": workers, "threads": threads,
            "planning_memory_gib": estimated_bytes / GIB,
            "memory_per_worker_gib": memory_per_worker_gib, "reserve_gib": reserve_gib,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=positive_int, default=24)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--memory-per-worker-gib", type=positive_float, default=6)
    parser.add_argument("--reserve-gib", type=positive_float, default=24)
    parser.add_argument("--allow-runtime-difference", action="store_true",
                        help="Only for fresh runs with --no-reuse-local; missing packages still fail")
    args = parser.parse_args()
    result = inspect_environment(**vars(args))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
