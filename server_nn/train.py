"""Portable, checkpointed NN study; train only from a verified exported bundle.

Run ``python -m server_nn.train --help`` from the exported package directory.
The numerical fitting function is the frozen local-study implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time
import traceback

from .common import (data_identity, ensure_vendor, hash_file, phase_models,
                     read_json, runtime_info, training_config, training_hashes,
                     verify_bundle, write_json)

ensure_vendor()
import numpy as np
import pandas as pd
from filelock import FileLock
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from nn_checkpoint import atomic_pickle, fingerprint
from protocol_data import load_bundle, make_block, make_tasks
from protocol_nn import clean_json, fit_block, validate_checkpoint


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def expected_months():
    return [str(m) for m in pd.period_range("2014-01", "2022-12", freq="M")]


def select_tasks(tasks, months):
    if [t["month"] for t in tasks] != expected_months():
        raise ValueError("Prepared data must supply all 108 registered test months")
    if months is None:
        return tasks
    if not months or len(months) != len(set(months)) or set(months) - set(expected_months()):
        raise ValueError("Pilot months must be unique YYYY-MM values within 2014-2022")
    return [t for t in tasks if t["month"] in months]


def model_spec(model, tasks, config, identity, *, runtime=None, code=None):
    """Phase, worker count and filesystem location cannot change a model's identity.

    All 108 tasks enter the identity even for a partial pilot. A successful pilot
    therefore supplies exactly reusable monthly checkpoints to a later full run.
    """
    return {"schema": "server_nn_model_v1", "protocol": "v1.1", "model": model,
            "validation_days": 126, "horizon": 1, "tasks": tasks,
            "config": config, "data_identity": identity,
            "runtime": runtime_info() if runtime is None else runtime,
            "training_code_sha256": training_hashes() if code is None else code}


def expected_rows(bundle, task):
    return np.flatnonzero((bundle["codes"] >= task["test_first"]) &
                          (bundle["codes"] <= task["test_last"]))


def load_checkpoint(path, task, signature, rows):
    payload = pd.read_pickle(path)
    if payload["fingerprint"] != signature or payload["task"] != task:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    validate_checkpoint(payload["result"], task, rows)
    return payload


def month_job(bundle_root, model, task, config, run_dir, signature, *, import_legacy=True):
    """Return a small receipt; never send saved network states to the parent."""
    bundle_root, run_dir = Path(bundle_root), Path(run_dir)
    checkpoint = run_dir / "months" / f"{task['month']}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    marker = checkpoint.with_suffix(".running.json")
    with FileLock(str(checkpoint) + ".lock", timeout=0):
        bundle = load_bundle(bundle_root / "prepared")
        rows = expected_rows(bundle, task)
        if checkpoint.exists():
            load_checkpoint(checkpoint, task, signature, rows)
            print(f"RESUME {model['model_id']} {task['month']}", flush=True)
            return {"checkpoint": str(checkpoint), "month": task["month"], "action": "resume"}
        started = time.monotonic()
        write_json({"pid": os.getpid(), "started_at": utc_now(),
                    "model_id": model["model_id"], "month": task["month"]}, marker)
        try:
            imported = None
            if import_legacy:
                from .migration import try_import_checkpoint
                imported = try_import_checkpoint(bundle_root, model, task, config,
                                                 checkpoint, signature, bundle)
            if imported is not None:
                load_checkpoint(checkpoint, task, signature, rows)
                action = "import"
            else:
                with threadpool_limits(limits=config["threads"]):
                    block = make_block(bundle, task, model["feature_set"], model["target"])
                    result = fit_block(block, {**config, "log_label":
                        f"{model['model_id']} {task['month']}"})
                result.update(month=task["month"], elapsed_seconds=time.monotonic() - started)
                validate_checkpoint(result, task, rows)
                atomic_pickle({"fingerprint": signature, "task": task, "result": result}, checkpoint)
                action = "fit"
            print(f"SAVED {model['model_id']} {task['month']} ({action}): "
                  f"{time.monotonic() - started:.1f}s", flush=True)
            return {"checkpoint": str(checkpoint), "month": task["month"],
                    "action": action, "elapsed_seconds": time.monotonic() - started}
        finally:
            marker.unlink(missing_ok=True)


def publish_model(bundle, model, tasks, config, run_dir, signature, output_root, *, pilot):
    """Validate and read one checkpoint at a time; keep only predictions/diagnostics."""
    run_dir, output_root = Path(run_dir), Path(output_root)
    indices, predictions, diagnostics = [], [], []
    import_count = 0
    for task in tasks:
        payload = load_checkpoint(run_dir / "months" / f"{task['month']}.pkl", task,
                                  signature, expected_rows(bundle, task))
        result = payload["result"]
        indices.append(np.asarray(result["test_indices"]).copy())
        predictions.append(np.asarray(result["prediction"]).copy())
        diagnostic = {k: v for k, v in result.items() if k not in {
            "prediction", "test_indices", "model_states", "mean", "scale", "constant"}}
        if "import_provenance" in payload:
            diagnostic["import_provenance"] = payload["import_provenance"]
            import_count += 1
        diagnostics.append(diagnostic)
        del result, payload
    all_indices = np.concatenate(indices)
    expected = np.concatenate([expected_rows(bundle, task) for task in tasks])
    if not np.array_equal(all_indices, expected):
        raise ValueError("Final predictions do not cover exactly the requested monthly rows")
    frame = bundle["keys"].iloc[all_indices].copy()
    frame["prediction"] = np.concatenate(predictions)
    if frame.duplicated(["date", "permno"]).any() or not np.isfinite(frame["prediction"]).all():
        raise ValueError("Invalid economic keys or predictions at publication")
    coverage = [t["month"] for t in tasks]
    publication = run_dir / "publications" / fingerprint({"months": coverage, "pilot": pilot})[:16]
    publication.mkdir(parents=True, exist_ok=True)
    prediction_path = publication / "predictions.pkl"
    # Stable repeated resumes need not rewrite already verified predictions.
    if prediction_path.exists():
        pd.testing.assert_frame_equal(pd.read_pickle(prediction_path), frame, check_exact=True)
    else:
        atomic_pickle(frame, prediction_path)
    write_json(clean_json({"fingerprint": signature, "months": coverage,
                           "month_diagnostics": diagnostics}), publication / "diagnostics.json")
    input_features = len(bundle["manifest"]["feature_sets"][model["feature_set"]])
    widths = [input_features, *model["widths"], 1]
    parameter_count = sum((a + 1) * b for a, b in zip(widths[:-1], widths[1:])) + 2 * sum(model["widths"])
    record = {**model, "validation_days": 126, "horizon": 1,
              "input_features": input_features, "parameter_count": parameter_count,
              "target_column": {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]],
              "predictions": prediction_path.relative_to(output_root).as_posix(),
              "run_dir": run_dir.relative_to(output_root).as_posix(),
              "diagnostics": (publication / "diagnostics.json").relative_to(output_root).as_posix(),
              "run_fingerprint": signature, "kind": "pilot" if pilot else "full",
              "coverage_scope": "partial" if pilot else "2014-2022", "training_budget": "standard",
              "rows": len(frame), "months": len(tasks), "month_list": coverage,
              "sha256": hash_file(prediction_path), "threads": config["threads"],
              "imported_checkpoint_months": import_count,
              "server_fitted_checkpoint_months": len(tasks) - import_count}
    write_json(record, publication / "complete.json")
    if not pilot:
        write_json(record, run_dir / "complete.json")
    return record


class Heartbeat:
    def __init__(self, path, initial, interval=10):
        self.path, self.state, self.interval = Path(path), dict(initial), interval
        self.stop_event, self.mutex = threading.Event(), threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def update(self, **values):
        with self.mutex:
            self.state.update(values)
            write_json({**self.state, "updated_at": utc_now()}, self.path)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.update()

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join()
        if exc is not None:
            self.update(status="failed", error=str(exc), traceback="".join(traceback.format_exception(exc_type, exc, tb)))


def run_study(bundle_root, output_root, phase, workers, threads, months=None, *, reuse_local=True):
    if min(workers, threads) < 1:
        raise ValueError("Worker and thread counts must be positive")
    bundle_root, output_root = Path(bundle_root).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # The root lock also serializes overlapping phases that share NN3 checkpoints.
    with FileLock(str(output_root / "training.lock"), timeout=0):
        verify_bundle(bundle_root, full_hash=True)
        if reuse_local:
            from .migration import require_bridge
            require_bridge(bundle_root, threads)
        bundle = load_bundle(bundle_root / "prepared")
        identity, runtime, code = data_identity(bundle_root), runtime_info(), training_hashes()
        models = phase_models(phase)
        task_sets = {f: make_tasks(bundle, fit_days=f) for f in {m["fit_days"] for m in models}}
        selected = {f: select_tasks(tasks, months) for f, tasks in task_sets.items()}
        pilot = months is not None
        registry_spec = {"phase": phase, "model_ids": [m["model_id"] for m in models],
                         "months": [t["month"] for t in next(iter(selected.values()))],
                         "data_identity": identity, "runtime": runtime, "code": code,
                         "threads": threads, "kind": "pilot" if pilot else "full"}
        registry_id = fingerprint(registry_spec)
        registry_dir = output_root / "registries"
        registry_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_root / "status.json"
        initial = {"status": "running", "phase": phase, "pid": os.getpid(), "started_at": utc_now(),
                   "workers": workers, "threads": threads, "kind": registry_spec["kind"],
                   "expected_models": len(models), "completed_models": 0,
                   "expected_monthly_jobs": sum(len(selected[m["fit_days"]]) for m in models),
                   "completed_monthly_jobs": 0, "registry_fingerprint": registry_id}
        records, receipts = [], []
        with Heartbeat(status_path, initial) as heartbeat:
            for model in models:
                config = training_config(model["widths"], threads)
                spec = model_spec(model, task_sets[model["fit_days"]], config, identity, runtime=runtime, code=code)
                signature = fingerprint(spec)
                run_dir = output_root / "models" / model["model_id"] / signature[:16]
                run_dir.mkdir(parents=True, exist_ok=True)
                spec_path = run_dir / "run_spec.json"
                if spec_path.exists() and read_json(spec_path) != spec:
                    raise ValueError(f"Model directory has a conflicting specification: {run_dir}")
                if not spec_path.exists():
                    write_json(spec, spec_path)
                heartbeat.update(current_model=model["model_id"], current_run_dir=run_dir.relative_to(output_root).as_posix())
                print(f"RUN {model['model_id']} {len(selected[model['fit_days']])} months; {run_dir}", flush=True)
                completed = Parallel(n_jobs=workers, backend="loky", pre_dispatch=workers, return_as="generator_unordered")(
                    delayed(month_job)(bundle_root, model, task, config, run_dir, signature,
                                       import_legacy=reuse_local)
                    for task in selected[model["fit_days"]])
                for receipt in completed:
                    receipts.append(receipt)
                    heartbeat.update(completed_monthly_jobs=len(receipts), last_completed=receipt["month"],
                                     last_action=receipt["action"])
                record = publish_model(bundle, model, selected[model["fit_days"]], config,
                                       run_dir, signature, output_root, pilot=pilot)
                records.append(record)
                heartbeat.update(completed_models=len(records))
            if {r["model_id"] for r in records} != {m["model_id"] for m in models} or len(records) != len(models):
                raise ValueError("Phase registry does not contain the exact registered model matrix")
            if not pilot and any(r["month_list"] != expected_months() for r in records):
                raise ValueError("A full registry cannot contain partial monthly coverage")
            verify_bundle(bundle_root, full_hash=True)
            if training_hashes() != code or runtime_info() != runtime or data_identity(bundle_root) != identity:
                raise ValueError("Training source, runtime or bundle identity changed during this invocation")
            registry = {"schema_version": "server_nn_registry_v1", "protocol": "v1.1", "phase": phase,
                        "kind": registry_spec["kind"], "path_base": "output_root", "models": records,
                        "fingerprint": registry_id, "specification": registry_spec,
                        "data_identity": identity, "created_at": utc_now(),
                        "expected_models": len(models), "expected_months": len(next(iter(selected.values())))}
            registry_path = registry_dir / f"{phase}_{registry_spec['kind']}_{registry_id[:16]}.json"
            write_json(registry, registry_path)
            if not pilot:
                write_json(registry, registry_dir / f"{phase}.json")
            heartbeat.update(status="complete", current_model=None, completed_at=utc_now(),
                             registry=registry_path.relative_to(output_root).as_posix(),
                             checkpoint_actions={a: sum(r["action"] == a for r in receipts) for a in ("fit", "resume", "import")})
            print(f"Registry: {registry_path}", flush=True)
            return registry_path


def main(argv=None):
    from .check_environment import inspect_environment, positive_float
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["nn3", "depth", "width"], required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--months", nargs="+", help="Full-budget pilot months; never published as a full study")
    parser.add_argument("--no-reuse-local", action="store_true", help="Train fresh when no server checkpoint exists; skip legacy imports")
    parser.add_argument("--memory-per-worker-gib", type=positive_float, default=6,
                        help="Planning allowance, not a measured peak; lower only after benchmarking")
    parser.add_argument("--reserve-gib", type=positive_float, default=24)
    args = parser.parse_args(argv)
    if min(args.workers, args.threads) < 1:
        parser.error("Worker and thread counts must be positive")
    preflight = inspect_environment(workers=args.workers, threads=args.threads,
        memory_per_worker_gib=args.memory_per_worker_gib, reserve_gib=args.reserve_gib,
        allow_runtime_difference=args.no_reuse_local)
    if not preflight["ok"]:
        parser.error("Resource/runtime preflight failed: " + "; ".join(preflight["errors"]))
    for warning in preflight["warnings"]:
        print(f"PREFLIGHT {warning}", flush=True)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return run_study(args.bundle, args.output, args.phase, args.workers, args.threads, args.months,
                     reuse_local=not args.no_reuse_local)


if __name__ == "__main__":
    main()
