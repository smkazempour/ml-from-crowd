"""Resumable study of explicit stock-characteristic, social and text interactions.

This module owns execution and provenance. The isolated linear_interaction_core
module owns numerical fitting; completed earlier studies remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import time

from filelock import FileLock
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

try:
    from . import linear_interaction_core as core, protocol_data as data, protocol_linear as linear
    from . import linear_design_core as legacy_core, linear_design_runner as legacy
    from .nn_checkpoint import atomic_json, atomic_pickle, fingerprint
except ImportError:
    import linear_interaction_core as core
    import linear_design_core as legacy_core
    import linear_design_runner as legacy
    import protocol_data as data
    import protocol_linear as linear
    from nn_checkpoint import atomic_json, atomic_pickle, fingerprint


SCHEMA = "linear_interactions_v1"
FIT_DAYS = (504,)
FEATURE_SETS = ("characteristics", "characteristics_core", "characteristics_textcore")
ESTIMATORS = ("ols", "ridge", "enet")
BASELINE_RTOL = 1e-7
BASELINE_ATOL = 1e-10
BASELINE_DAILY_IC_ATOL = 1e-7


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__)


def code_identity():
    names = ("linear_interaction_runner.py", "linear_interaction_core.py",
             "linear_interaction_evaluate.py", "linear_design_runner.py", "linear_design_core.py", "linear_optimization_core.py", "protocol_data.py",
             "protocol_linear.py", "characteristic_data.py", "characteristic_linear.py",
             "prediction_metrics.py", "nn_checkpoint.py")
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def json_safe(value):
    """Deterministic diagnostic serialization, including unavailable scores."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def planned_models(targets=("raw", "dgtw")):
    models = []
    for target in targets:
        for spec in core.procedure_specs():
            models.append(dict(**spec, model_id=core.model_name(spec, target), target=target,
                target_column=linear.TARGET_COLUMNS[target],
                history_policy=spec["history"], refit_policy=spec["refit"],
                fit_days=504 if spec["history"] == "fixed504" else 0,
                validation_days=126, horizon=1))
    if len({model["model_id"] for model in models}) != len(models):
        raise ValueError("Duplicate planned model identifiers")
    return models


def month_tasks(bundle, start="2014-01", end="2022-12", months=None):
    by_month = {}
    for fit in FIT_DAYS:
        for task in data.make_tasks(bundle, fit_days=fit, validation_days=126,
                                    horizon=1, start=start, end=end, months=months):
            by_month.setdefault(task["month"], []).append(task)
    result = [by_month[month] for month in sorted(by_month)]
    if not result:
        raise ValueError("No requested test months")
    for tasks in result:
        validate_tasks(tasks)
    return result


def validate_tasks(tasks):
    if len(tasks) != 1 or [task["fit_days"] for task in tasks] != list(FIT_DAYS):
        raise ValueError("Each monthly job requires the fixed 504-session task")
    same = ("month", "validation_days", "horizon", "valid_first", "valid_last", "test_first", "test_last")
    if any(any(task[field] != tasks[0][field] for field in same) for task in tasks):
        raise ValueError("Candidate histories do not share validation and test dates")
    if tasks[0]["validation_days"] != 126 or tasks[0]["horizon"] != 1:
        raise ValueError("The frozen experiment requires 126 validation sessions and h=1")


def expected_indices(bundle, tasks):
    task = tasks[0]
    return np.flatnonzero((bundle["codes"] >= task["test_first"])
                          & (bundle["codes"] <= task["test_last"]))


def verify_prepared(bundle):
    path = Path(bundle["path"])
    manifest = bundle["manifest"]
    records = manifest.get("artifact_records", {})
    required = {"X.npy", "y.npy", "q.npy", "codes.npy", "calendar.npy", "keys.pkl", "evaluation.pkl"}
    if not required.issubset(records):
        raise ValueError("Prepared artifact certificates are incomplete")
    verified = {}
    for name, record in records.items():
        if Path(name).name != name:
            raise ValueError("Prepared artifact must be a direct cache child")
        artifact = path / name
        actual = dict(sha256=sha256_file(artifact), size=artifact.stat().st_size)
        if actual != record:
            raise ValueError(f"Prepared artifact changed: {name}")
        verified[name] = actual
    for feature in FEATURE_SETS:
        columns = manifest["feature_sets"][feature]
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Invalid prepared feature set: {feature}")
    if manifest["targets"] != ["f_cumret1", "ar_dgtw_1"]:
        raise ValueError("Prepared target order differs")
    return dict(prepared_manifest_sha256=sha256_file(path / "manifest.json"), artifacts=verified)


def verify_production_schema(bundle):
    manifest = bundle["manifest"]
    expected = dict(characteristics=34, characteristics_core=36, characteristics_textcore=422)
    for feature, count in expected.items():
        if len(manifest["feature_sets"][feature]) != count:
            raise ValueError(f"Unexpected production feature count: {feature}")
    names = [manifest["feature_names"][column] for column in manifest["feature_sets"]["characteristics_textcore"]]
    embeddings = [name for name in names if re.fullmatch(r"embed_\d{3}", name)]
    if len(embeddings) != 384 or not {"embed_norm", "embed_cos"}.issubset(names):
        raise ValueError("Expected 384 embedding coordinates and two separate agreement measures")


def verify_baseline(bundle, registry_path, tasks_by_month, targets):
    """Certify saved monthly historical fits before using them as exact controls."""
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    old = registry["config"]
    if (registry.get("schema_version") != "linear_design_v2"
            or Path(registry["prepared"]).resolve() != Path(bundle["path"]).resolve()
            or old.get("data_manifest") != bundle["manifest"]
            or old.get("prepared_manifest_sha256") != sha256_file(Path(bundle["path"]) / "manifest.json")
            or old.get("validation_days") != 126 or old.get("horizon") != 1
            or old.get("grid") != legacy_core.grid_specification()):
        raise ValueError("Historical interaction configuration differs")
    for name, digest in old["code"].items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Historical numerical source differs: {name}")
    for name, version in versions().items():
        if str(old.get("versions", {}).get(name, "")).split()[0:1] != [version]:
            raise ValueError(f"Historical numerical version differs: {name}")
    old_tasks = {(task["month"], task["fit_days"]): task for group in old["tasks"] for task in group}
    old_groups = {group[0]["month"]: group for group in old["tasks"]}
    records = {}
    for tasks in tasks_by_month:
        validate_tasks(tasks)
        for task in tasks:
            if old_tasks.get((task["month"], task["fit_days"])) != task:
                raise ValueError("Historical calendar task differs")
        task = tasks[0]
        for target in targets:
            path = registry_path.parent / "checkpoints" / f"{target}_{task['month']}.pkl"
            with path.open("rb") as source:
                saved = pickle.load(source)
            result = saved["result"]
            predictions = np.asarray(result["predictions"])
            if (saved["fingerprint"] != registry["fingerprint"] or saved["tasks"] != old_groups[task["month"]]
                    or result["target"] != target or result["month"] != task["month"]
                    or not np.array_equal(result["test_indices"], expected_indices(bundle, tasks))
                    or legacy.result_certificate(result) != saved.get("result_certificate")
                    or sha256_file(path) != registry["checkpoint_sha256"].get(path.name)
                    or not np.isfinite(predictions).all()):
                raise ValueError(f"Historical checkpoint is invalid: {path}")
            for spec in core.procedure_specs():
                old_spec = core.legacy_spec(spec)
                if old_spec is not None and legacy_core.model_name(old_spec, target) not in result["model_names"]:
                    raise ValueError("Historical checkpoint omits a required baseline")
            records[f"{target}/{task['month']}"] = dict(path=str(path), sha256=sha256_file(path))
    return dict(registry=str(registry_path), registry_sha256=sha256_file(registry_path),
                fingerprint=registry["fingerprint"], checkpoints=records,
                comparison=dict(rtol=BASELINE_RTOL, atol=BASELINE_ATOL, daily_ic_atol=BASELINE_DAILY_IC_ATOL))


def verify_baseline_predictions(result, baseline_record, bundle):
    path = Path(baseline_record["path"])
    if sha256_file(path) != baseline_record["sha256"]:
        raise ValueError("Historical checkpoint changed after certification")
    with path.open("rb") as source:
        previous = pickle.load(source)["result"]
    if not np.array_equal(previous["test_indices"], result["test_indices"]):
        raise ValueError("Historical baseline prediction rows differ")
    audit = {}
    indices = np.asarray(result["test_indices"])
    codes = np.asarray(bundle["codes"])[indices]
    target_index = bundle["manifest"]["targets"].index(linear.TARGET_COLUMNS[result["target"]])
    targets = np.asarray(bundle["q"])[indices, target_index]
    boundaries = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
    for spec in core.procedure_specs():
        old_spec = core.legacy_spec(spec)
        if old_spec is None:
            continue
        name = core.model_name(spec, result["target"])
        old_name = legacy_core.model_name(old_spec, result["target"])
        actual = result["predictions"][:, result["model_names"].index(name)]
        expected = previous["predictions"][:, previous["model_names"].index(old_name)]
        difference = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=BASELINE_RTOL, atol=BASELINE_ATOL):
            raise ValueError(f"Historical baseline predictions were not reproduced: {name}; max_abs={difference}")
        max_ic_difference, scored_dates = 0., 0
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            finite = np.isfinite(targets[first:last])
            if finite.sum() < 10 or np.unique(targets[first:last][finite]).size < 2:
                continue
            scores, _ = linear.validation_scores(np.column_stack([actual[first:last], expected[first:last]]),
                                                  targets[first:last], codes[first:last])
            max_ic_difference = max(max_ic_difference, float(abs(scores[0] - scores[1])))
            scored_dates += 1
        if max_ic_difference > BASELINE_DAILY_IC_ATOL:
            raise ValueError(f"Historical baseline daily IC differs: {name}; max_abs={max_ic_difference}")
        scale = max(float(np.std(expected)), np.finfo(float).tiny)
        audit[name] = dict(max_abs_difference=difference, max_abs_difference_over_prediction_sd=difference / scale,
                          max_daily_ic_difference=max_ic_difference, compared_dates=scored_dates,
                          historical_model_id=old_name)
    if len(audit) != 9:
        raise ValueError("The baseline audit must compare all nine original additive procedures")
    return audit


def _array_sha(array):
    values = np.asarray(array)
    return fingerprint(dict(dtype=str(values.dtype), shape=list(values.shape),
                            bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest()))


def result_certificate(result):
    metadata = {key: value for key, value in result.items() if key not in ("predictions", "test_indices")}
    return dict(predictions=_array_sha(result["predictions"]),
                test_indices=_array_sha(result["test_indices"]),
                metadata=fingerprint(json_safe(metadata)))


def validate_month(result, tasks, target, bundle):
    validate_tasks(tasks)
    expected = expected_indices(bundle, tasks)
    names = [core.model_name(spec, target) for spec in core.procedure_specs()]
    values = np.asarray(result["predictions"])
    if (result["month"] != tasks[0]["month"] or result["target"] != target
            or not np.array_equal(result["test_indices"], expected)
            or result["model_names"] != names or values.shape != (len(expected), len(names))
            or not np.isfinite(values).all() or set(result["diagnostics"]) != set(names)):
        raise ValueError(f"Invalid interaction checkpoint: {target}/{tasks[0]['month']}")
    for spec, name in zip(core.procedure_specs(), names):
        diagnostic = result["diagnostics"][name]
        chosen = diagnostic["selected"]
        fit = int(chosen["selected_fit_days"])
        if fit not in FIT_DAYS or (spec["history"] == "fixed504" and fit != 504):
            raise ValueError(f"Invalid selected history: {name}")
        if "final" not in diagnostic:
            raise ValueError(f"Missing final-fit diagnostic: {name}")


def checkpoint_month(prepared, tasks, target, run_dir, run_fingerprint, baseline_record, threads=2):
    bundle = data.load_bundle(prepared)
    path = Path(run_dir) / "checkpoints" / f"{target}_{tasks[0]['month']}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists():
            with path.open("rb") as source:
                saved = pickle.load(source)
            if (saved["fingerprint"] != run_fingerprint or saved["tasks"] != tasks
                    or saved.get("baseline_record") != baseline_record):
                raise ValueError(f"Interaction checkpoint identity mismatch: {path}")
            validate_month(saved["result"], tasks, target, bundle)
            if result_certificate(saved["result"]) != saved.get("result_certificate"):
                raise ValueError(f"Interaction checkpoint checksum mismatch: {path}")
            verify_baseline_predictions(saved["result"], baseline_record, bundle)
            print(f"RESUME {target} {tasks[0]['month']}", flush=True)
            return str(path)
        begin = time.monotonic()
        with threadpool_limits(limits=threads):
            result = core.fit_month(bundle, tasks, target)
        validate_month(result, tasks, target, bundle)
        result["baseline_audit"] = verify_baseline_predictions(result, baseline_record, bundle)
        result["elapsed_seconds"] = time.monotonic() - begin
        atomic_pickle(dict(fingerprint=run_fingerprint, tasks=tasks, result=result,
                           baseline_record=baseline_record, result_certificate=result_certificate(result)), path)
        print(f"SAVED {target} {tasks[0]['month']}: {result['elapsed_seconds']:.1f}s", flush=True)
        return str(path)


def _csv_value(value):
    value = json_safe(value)
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def _selection_row(result, name, specification):
    diagnostic = result["diagnostics"][name]
    row = {**specification, "month": result["month"], "test_rows": len(result["test_indices"]),
           "elapsed_month_seconds": result["elapsed_seconds"]}
    for key, value in diagnostic["selected"].items():
        if key not in ("coefficient", "intercept"):
            row[key] = _csv_value(value)
    for key, value in diagnostic["final"].items():
        if key not in ("coefficient", "intercept") and np.isscalar(value):
            row[f"final_{key}"] = _csv_value(value)
    for key, value in diagnostic.get("stability", {}).items():
        row[f"stability_{key}"] = _csv_value(value)
    return row


def publish_run(bundle, run_dir, config, checkpoint_paths):
    run_dir = Path(run_dir)
    specs = {model["model_id"]: model for model in planned_models(config["targets"])}
    expected_tasks = {(tasks[0]["month"], target): tasks
                      for tasks in config["tasks"] for target in config["targets"]}
    grouped, monthly, seen, checkpoint_records = {}, [], set(), {}
    for checkpoint in checkpoint_paths:
        path = Path(checkpoint)
        with path.open("rb") as source:
            saved = pickle.load(source)
        result, tasks = saved["result"], saved["tasks"]
        key = (result["month"], result["target"])
        record = config["baseline_proof"]["checkpoints"].get(f"{result['target']}/{result['month']}")
        if (key in seen or expected_tasks.get(key) != tasks or saved["fingerprint"] != config["fingerprint"]
                or saved.get("baseline_record") != record):
            raise ValueError("Publication checkpoint identity or coverage mismatch")
        validate_month(result, tasks, result["target"], bundle)
        if result_certificate(result) != saved.get("result_certificate"):
            raise ValueError("Publication checkpoint checksum mismatch")
        verify_baseline_predictions(result, record, bundle)
        seen.add(key)
        checkpoint_records[path.name] = sha256_file(path)
        for column, name in enumerate(result["model_names"]):
            grouped.setdefault(name, []).append((result["test_indices"], result["predictions"][:, column]))
            monthly.append(_selection_row(result, name, specs[name]))
    if seen != set(expected_tasks):
        raise ValueError("Cannot publish an incomplete interaction run")
    if code_identity() != config["code"]:
        raise ValueError("Interaction fitting source changed during the run")
    if versions() != config["versions"]:
        raise ValueError("Numerical versions changed during fitting")
    if sha256_file(Path(bundle["path"]) / "manifest.json") != config["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest changed during fitting")
    proof = config["baseline_proof"]
    if sha256_file(proof["registry"]) != proof["registry_sha256"]:
        raise ValueError("Historical registry changed during fitting")
    if config.get("experiment_path") and sha256_file(config["experiment_path"]) != config["experiment_sha256"]:
        raise ValueError("Experiment specification changed during fitting")
    expected = np.concatenate([expected_indices(bundle, tasks) for tasks in config["tasks"]])
    models = []
    for name, specification in specs.items():
        pieces = grouped.pop(name)
        indices = np.concatenate([piece[0] for piece in pieces])
        if not np.array_equal(np.sort(indices), np.sort(expected)) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Prediction row coverage differs: {name}")
        frame = bundle["keys"].iloc[indices].copy()
        frame["prediction"] = np.concatenate([piece[1] for piece in pieces])
        frame.sort_values(["date", "permno"], inplace=True)
        if not frame.index.is_unique or frame.duplicated(["date", "permno"]).any():
            raise ValueError(f"Duplicate prediction keys: {name}")
        path = run_dir / f"predictions_{name}.pkl"
        atomic_pickle(frame, path)
        entry = {**specification, "predictions": str(path.resolve()), "rows": len(frame),
                 "months": int(frame.date.dt.to_period("M").nunique()), "sha256": sha256_file(path)}
        atomic_json(entry | dict(fingerprint=config["fingerprint"], schema_version=SCHEMA), path.with_suffix(".json"))
        models.append(entry)
    if grouped or len(models) != len(specs):
        raise ValueError("Unexpected model publication coverage")
    summary = run_dir / "monthly_selection.csv"
    temporary = summary.with_suffix(".csv.tmp")
    pd.DataFrame(monthly).to_csv(temporary, index=False)
    os.replace(temporary, summary)
    registry = dict(schema_version=SCHEMA, prepared=config["prepared"], fingerprint=config["fingerprint"],
                    config=config, models=models, monthly_selection=str(summary.resolve()),
                    monthly_selection_sha256=sha256_file(summary), checkpoint_sha256=checkpoint_records,
                    baseline_audit=str((run_dir / "baseline_audit.json").resolve()))
    atomic_json(registry, run_dir / "registry.json")
    atomic_json(dict(fingerprint=config["fingerprint"], models=len(models), checkpoints=len(seen),
                     registry_sha256=sha256_file(run_dir / "registry.json")), run_dir / "complete.json")
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-registry", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parents[1] / ".runs" / SCHEMA / "linear")
    parser.add_argument("--targets", nargs="+", choices=["raw", "dgtw"], default=["raw", "dgtw"])
    parser.add_argument("--start", default="2014-01")
    parser.add_argument("--end", default="2022-12")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    args.targets = list(dict.fromkeys(args.targets))
    bundle = data.load_bundle(args.prepared)
    verify_production_schema(bundle)
    tasks = month_tasks(bundle, args.start, args.end, args.months)
    prepared_proof = verify_prepared(bundle)
    baseline_proof = verify_baseline(bundle, args.baseline_registry, tasks, args.targets)
    grid = core.grid_specification()
    config = dict(schema_version=SCHEMA, prepared=str(args.prepared.resolve()),
        prepared_manifest_sha256=prepared_proof["prepared_manifest_sha256"], data_manifest=bundle["manifest"],
        prepared_proof=prepared_proof, baseline_proof=baseline_proof, fit_days=list(FIT_DAYS),
        validation_days=126, horizon=1, targets=args.targets, tasks=tasks,
        feature_sets=list(FEATURE_SETS), estimators=list(ESTIMATORS),
        procedures=core.procedure_specs(), grid=grid, threads=args.threads,
        code=code_identity(), versions=versions(), run_kind="pilot" if args.months else "full")
    try:
        from .linear_interaction_evaluate import family_counts
    except ImportError:
        from linear_interaction_evaluate import family_counts
    config["planned_family_counts"] = family_counts(planned_models(args.targets))
    if args.spec:
        config.update(experiment_path=str(args.spec.resolve()), experiment_sha256=sha256_file(args.spec),
                      experiment=json.loads(args.spec.read_text(encoding="utf-8")))
    config["fingerprint"] = fingerprint(json_safe(config))
    run_dir = args.out_root / f"{config['run_kind']}_{config['fingerprint'][:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_dir) + ".lock", timeout=0):
        atomic_json(json_safe(config), run_dir / "config.json")
        atomic_json(baseline_proof, run_dir / "baseline_audit.json")
        work = [(group, target) for group in tasks for target in args.targets]
        print(f"RUN {run_dir.resolve()} | {len(work)} month-target jobs | workers={args.workers}, threads={args.threads}", flush=True)
        arguments = [(str(args.prepared), group, target, str(run_dir), config["fingerprint"],
                      baseline_proof["checkpoints"][f"{target}/{group[0]['month']}"], args.threads)
                     for group, target in work]
        if args.workers == 1:
            paths = [checkpoint_month(*values) for values in arguments]
        else:
            paths = Parallel(n_jobs=args.workers, backend="loky", pre_dispatch=args.workers)(
                delayed(checkpoint_month)(*values) for values in arguments)
        publish_run(bundle, run_dir, config, paths)
        print(f"REGISTRY {(run_dir / 'registry.json').resolve()}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()

