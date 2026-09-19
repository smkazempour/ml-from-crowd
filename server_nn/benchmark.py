"""Full-budget resource benchmark, isolated from study outputs and scientific ranking.

The default is a small *workload*, not reduced epochs, seeds, or penalties. A
resource sweep deliberately expands the workload enough to exercise concurrency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .common import (PACKAGE, data_identity, phase_models, read_json, runtime_info, training_config,
                     training_hashes, verify_bundle, write_json)
from .train import Heartbeat, expected_months, load_bundle, make_tasks, model_spec, month_job
from nn_checkpoint import fingerprint
from filelock import FileLock
from joblib import Parallel, delayed
import psutil

ANCHOR_MONTHS = ["2014-01", "2015-03", "2022-12"]


def benchmark_jobs(profile, minimum_jobs=0):
    if profile not in {"core_textcore", "largest"}:
        raise ValueError("Unknown benchmark profile")
    models = [m for m in phase_models("nn3") if m["target"] == "raw" and
              (m["fit_days"], m["feature_set"]) in
              ([(504, "core"), (504, "textcore")] if profile == "core_textcore" else [(756, "textall")])]
    jobs = [(model, month) for month in ANCHOR_MONTHS for model in models]
    existing = {(m["model_id"], month) for m, month in jobs}
    for month in expected_months():
        for model in models:
            if len(jobs) >= max(minimum_jobs, len(ANCHOR_MONTHS) * len(models)):
                return jobs
            if (model["model_id"], month) not in existing:
                jobs.append((model, month))
                existing.add((model["model_id"], month))
    if len(jobs) < minimum_jobs:
        raise ValueError("Requested benchmark concurrency exceeds unique registered workload")
    return jobs


def cpu_allocation():
    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, OSError, psutil.Error):
        affinity = None
    return {"logical_cpus_host": psutil.cpu_count(logical=True),
            "physical_cores_host": psutil.cpu_count(logical=False),
            "cpu_affinity": affinity,
            "affinity_logical_cpus": len(affinity) if affinity is not None else None,
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE")}


class ResourceSampler:
    """Measure this process and descendants; never include unrelated local runs."""
    def __init__(self, root, interval=2):
        self.root, self.interval = Path(root), interval
        self.parent = psutil.Process()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = []

    def sample(self):
        try:
            processes = [self.parent, *self.parent.children(recursive=True)]
        except psutil.Error:
            processes = [self.parent]
        rss, private, private_count, alive = 0, 0, 0, 0
        pids = set()
        for process in processes:
            try:
                memory = process.memory_info()
                rss += memory.rss
                if hasattr(memory, "private"):
                    private += memory.private
                    private_count += 1
                pids.add(process.pid)
                alive += 1
            except (psutil.Error, OSError):
                pass
        active = 0
        for marker in self.root.glob("models/*/*/months/*.running.json"):
            try:
                import json
                active += int(json.loads(marker.read_text(encoding="utf-8"))["pid"] in pids)
            except (OSError, ValueError, KeyError):
                pass
        self.samples.append({"time_seconds": time.monotonic(), "rss_bytes": rss,
                             "private_bytes": private if private_count == alive and alive else None,
                             "processes": alive, "active_month_jobs": active})

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def summary(self):
        private = [s["private_bytes"] for s in self.samples if s["private_bytes"] is not None]
        return {"peak_rss_bytes_sum_parent_and_children": max(s["rss_bytes"] for s in self.samples),
                "peak_private_bytes_sum_parent_and_children": max(private) if private else None,
                "peak_active_month_jobs": max(s["active_month_jobs"] for s in self.samples),
                "peak_process_count": max(s["processes"] for s in self.samples),
                "sample_count": len(self.samples), "sample_interval_seconds": self.interval,
                "rss_note": "Summed process RSS can double-count shared/mapped pages; private bytes are reported only where available. Sampling may miss brief peaks."}


def run_configuration(bundle_root, configuration_dir, workload, workers, threads):
    configuration_dir = Path(configuration_dir)
    configuration_dir.mkdir(parents=True, exist_ok=False)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    bundle = load_bundle(Path(bundle_root) / "prepared")
    identity, runtime, code = data_identity(bundle_root), runtime_info(), training_hashes()
    task_sets = {f: make_tasks(bundle, fit_days=f) for f in {m["fit_days"] for m, _ in workload}}
    dispatched = []
    for model, month in workload:
        config = training_config(model["widths"], threads)
        spec = model_spec(model, task_sets[model["fit_days"]], config, identity, runtime=runtime, code=code)
        signature = fingerprint(spec)
        run_dir = configuration_dir / "models" / model["model_id"] / signature[:16]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(spec, run_dir / "run_spec.json")
        task = next(t for t in task_sets[model["fit_days"]] if t["month"] == month)
        dispatched.append((model, task, config, run_dir, signature))
    del bundle
    started = time.monotonic()
    receipts = []
    with Heartbeat(configuration_dir / "status.json", {"status": "running", "workers": workers,
                   "threads": threads, "expected_monthly_jobs": len(dispatched),
                   "completed_monthly_jobs": 0, "pid": os.getpid()}) as heartbeat, ResourceSampler(configuration_dir) as sampler:
        # Each configuration runs in a fresh interpreter, so a later resource
        # setting cannot inherit this interpreter's reusable joblib worker pool.
        with Parallel(n_jobs=workers, backend="loky", pre_dispatch=workers,
                      return_as="generator_unordered") as parallel:
            for receipt in parallel(delayed(month_job)(bundle_root, model, task, config, run_dir,
                    signature, import_legacy=False) for model, task, config, run_dir, signature in dispatched):
                receipts.append(receipt)
                heartbeat.update(completed_monthly_jobs=len(receipts))
        heartbeat.update(status="complete")
    elapsed = time.monotonic() - started
    resources = sampler.summary()
    result = {"workers": workers, "threads": threads, "requested_compute_threads": workers * threads,
              "monthly_jobs": len(receipts), "candidate_seed_fits": len(receipts) * 15,
              "elapsed_seconds": elapsed, "monthly_jobs_per_hour": len(receipts) * 3600 / elapsed,
              "training_budget": "standard", "scope": "resource benchmark; not scientific model selection",
              "all_jobs_newly_fitted": all(r["action"] == "fit" for r in receipts),
              "enough_jobs_for_two_waves": len(receipts) >= 2 * workers,
              "observed_requested_concurrency": resources["peak_active_month_jobs"] >= workers,
              "cpu_allocation": cpu_allocation(), "resources": resources}
    write_json(result, configuration_dir / "result.json")
    write_json(sampler.samples, configuration_dir / "resource_samples.json")
    return result


def isolated_configuration(bundle_root, configuration_dir, workload, workers, threads):
    configuration_dir = Path(configuration_dir)
    spec_path = configuration_dir.with_suffix(".json")
    write_json({"bundle_root": str(Path(bundle_root).resolve()),
                "configuration_dir": str(configuration_dir.resolve()),
                "workload": workload, "workers": workers, "threads": threads}, spec_path)
    environment = os.environ.copy()
    environment.update({variable: str(threads) for variable in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    environment.update(KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUTF8="1")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    subprocess.run([sys.executable, "-B", "-u", "-m", "server_nn.benchmark",
                    "_run-configuration", str(spec_path.resolve())],
                   cwd=PACKAGE.parent, env=environment, check=True, **options)
    return read_json(configuration_dir / "result.json")


def main(argv=None):
    from .check_environment import inspect_environment, positive_float
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_run-configuration":
        if len(argv) != 2:
            raise ValueError("Internal configuration invocation requires exactly one specification")
        return run_configuration(**read_json(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["core_textcore", "largest"], default="core_textcore")
    parser.add_argument("--workers", nargs="+", type=int, choices=[1, 3, 8, 16, 24, 32])
    parser.add_argument("--threads", nargs="+", type=int, choices=[2, 4], default=[2])
    parser.add_argument("--resource-sweep", action="store_true", help="Expand to at least two waves at max workers; expensive opt-in")
    parser.add_argument("--memory-per-worker-gib", type=positive_float, default=6,
                        help="Planning allowance, not a measured peak; lower only after benchmarking")
    parser.add_argument("--reserve-gib", type=positive_float, default=24)
    parser.add_argument("--allow-runtime-difference", action="store_true",
                        help="Benchmark a fresh alternative software environment; legacy imports are always disabled")
    args = parser.parse_args(argv)
    workers = args.workers or ([8, 16, 24, 32] if args.resource_sweep else [1])
    if len(workers) != len(set(workers)) or len(args.threads) != len(set(args.threads)):
        parser.error("Duplicate resource configurations are not useful")
    preflight = inspect_environment(workers=max(workers), threads=max(args.threads),
        memory_per_worker_gib=args.memory_per_worker_gib, reserve_gib=args.reserve_gib,
        allow_runtime_difference=args.allow_runtime_difference)
    if not preflight["ok"]:
        parser.error("Resource/runtime preflight failed: " + "; ".join(preflight["errors"]))
    bundle_root, output_root = args.bundle.resolve(), args.output.resolve()
    verify_bundle(bundle_root, full_hash=True)
    workload = benchmark_jobs(args.profile, 2 * max(workers) if args.resource_sweep else 0)
    benchmark_root = output_root / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / "training.lock"), timeout=0):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_root = benchmark_root / run_id
        run_root.mkdir()
        manifest = {"profile": args.profile, "workload": [{"model_id": m["model_id"], "month": month}
                    for m, month in workload], "workers": workers, "threads": args.threads,
                    "resource_sweep": args.resource_sweep, "training_budget": "standard", "preflight": preflight,
                    "warnings": ["Benchmark checkpoints are isolated and never imported into study results.",
                                 "Compare resource throughput only; do not select models using benchmark test outcomes.",
                                 "Default three-month workloads cannot establish saturation of 24 or 32 workers."]}
        write_json(manifest, run_root / "benchmark_spec.json")
        results = []
        print(f"BENCHMARK {run_root}; {len(workload)} full-budget monthly jobs per configuration", flush=True)
        for count in workers:
            for threads in args.threads:
                results.append(isolated_configuration(bundle_root, run_root / f"workers_{count}_threads_{threads}",
                                                 workload, count, threads))
                write_json({"specification": manifest, "configurations": results,
                            "complete": len(results) == len(workers) * len(args.threads)}, run_root / "summary.json")
        print(f"Benchmark report: {run_root / 'summary.json'}", flush=True)
        return run_root / "summary.json"


if __name__ == "__main__":
    main()
