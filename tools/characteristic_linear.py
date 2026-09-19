"""Characteristic-controlled linear study with verified reuse of social-only fits.

Numerical fitting, transformations, loss, grids and chronological splits are the
unchanged protocol-v1.1 kernels. This runner owns its separate configuration,
checkpoints and registry; historical artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import time

from filelock import FileLock
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

try:
    from . import protocol_data as data, protocol_linear as linear
    from .nn_checkpoint import atomic_json, atomic_pickle, fingerprint
except ImportError:
    import protocol_data as data
    import protocol_linear as linear
    from nn_checkpoint import atomic_json, atomic_pickle, fingerprint


SCHEMA = "characteristics_v1_linear"
SOCIAL_FEATURE_SETS = tuple(linear.FEATURE_SETS)
FEATURE_SETS = ("characteristics", "characteristics_sentiment", "characteristics_attention",
                "characteristics_core", "characteristics_all", "characteristics_textcore",
                "characteristics_textall")
ESTIMATORS = tuple(linear.ESTIMATORS)
UNION_FEATURE_SET = "characteristics_textall"
PRESERVED_FILES = ("keys.pkl", "evaluation.pkl", "y.npy", "q.npy", "codes.npy", "calendar.npy")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    names = ("characteristic_linear.py", "characteristic_data.py", "protocol_linear.py",
             "protocol_data.py", "nn_checkpoint.py", "prediction_metrics.py")
    result = {}
    for name in names:
        path = Path(__file__).with_name(name)
        if path.exists():
            result[name] = sha256_file(path)
        elif name != "characteristic_data.py":
            raise FileNotFoundError(path)
    return result


def _resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def validate_parent_identity(bundle, registry):
    """Independently certify that all historical fitting dependencies survived.

    Same outcomes alone are insufficient: scaler inputs, date ranks, row order,
    split calendar, features and code must also agree before forecasts are reused.
    """
    manifest = bundle["manifest"]
    derivation = manifest.get("derivation", {})
    if derivation.get("kind") != "append_market_characteristics":
        raise ValueError("Prepared cache lacks characteristic parent provenance")
    parent_path = Path(derivation["parent_path"]).resolve()
    if Path(registry["prepared"]).resolve() != parent_path:
        raise ValueError("Linear registry and characteristic cache have different parents")
    parent = data.load_bundle(parent_path)
    old_config = registry.get("config", {})
    if (old_config.get("data_manifest") != parent["manifest"]
            or derivation.get("parent_fingerprint") != parent["manifest"]["fingerprint"]
            or derivation.get("parent_manifest_sha256") != sha256_file(parent_path / "manifest.json")):
        raise ValueError("Historical parent manifest identity mismatch")
    for name, digest in linear.code_identity().items():
        if old_config.get("code", {}).get(name) != digest:
            raise ValueError(f"Historical numerical source identity mismatch: {name}")
    current_versions = {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for name, version in current_versions.items():
        saved_version = str(old_config.get("versions", {}).get(name, "")).split()
        if not saved_version or saved_version[0] != version:
            raise ValueError(f"Historical numerical version differs: {name}")
    if (old_config.get("grid") != linear.grid_specification()
            or old_config.get("validation_days") != 126 or old_config.get("horizon") != 1
            or tuple(old_config.get("estimators", [])) != ESTIMATORS
            or tuple(old_config.get("feature_sets", [])) != SOCIAL_FEATURE_SETS):
        raise ValueError("Historical linear fitting procedure differs")
    prefix = len(parent["manifest"]["feature_names"])
    if (derivation.get("parent_feature_count") != prefix
            or manifest["feature_names"][:prefix] != parent["manifest"]["feature_names"]
            or bundle["X"].shape[0] != parent["X"].shape[0]
            or bundle["X"].shape[1] <= prefix or bundle["X"].dtype != parent["X"].dtype):
        raise ValueError("Historical feature prefix schema mismatch")
    for feature in SOCIAL_FEATURE_SETS:
        if manifest["feature_sets"].get(feature) != parent["manifest"]["feature_sets"][feature]:
            raise ValueError(f"Historical feature set differs: {feature}")
    if manifest["targets"] != parent["manifest"]["targets"]:
        raise ValueError("Historical target column order differs")
    current_path = Path(bundle["path"])
    preserved = {}
    for name in PRESERVED_FILES:
        old_sha, new_sha = sha256_file(parent_path / name), sha256_file(current_path / name)
        if old_sha != new_sha:
            raise ValueError(f"Historical data identity mismatch: {name}")
        preserved[name] = {"sha256": old_sha, "bytes": (current_path / name).stat().st_size}
    parent_X_sha = sha256_file(parent_path / "X.npy")
    if parent_X_sha != derivation.get("parent_X_sha256"):
        raise ValueError("Historical feature file hash differs from provenance")
    for first in range(0, len(parent["X"]), 4096):
        if not np.array_equal(parent["X"][first:first + 4096].view(np.uint8),
                              bundle["X"][first:first + 4096, :prefix].view(np.uint8)):
            raise ValueError(f"Historical feature values changed near row {first}")
    # These transformation metadata affect make_block, despite identical X bytes.
    original_names = set(parent["manifest"]["feature_names"])
    for field in ("binary_features", "scalar_rank_features", "unranked_features"):
        old_names = set(parent["manifest"].get(field, []))
        new_names = set(manifest.get(field, [])) & original_names
        if old_names != new_names:
            raise ValueError(f"Historical transformation metadata differs: {field}")
    current_X_sha = sha256_file(current_path / "X.npy")
    artifact_records = manifest.get("artifact_records", {})
    if "X.npy" not in artifact_records:
        raise ValueError("Characteristic prepared artifact certificates are missing")
    verified_artifacts = {}
    for name, record in artifact_records.items():
        path = current_path / name
        if Path(name).name != name:
            raise ValueError("Prepared artifact must be a direct cache child")
        digest = (current_X_sha if name == "X.npy" else preserved[name]["sha256"]
                  if name in preserved else sha256_file(path))
        actual = {"sha256": digest, "size": path.stat().st_size}
        if actual != record:
            raise ValueError(f"Characteristic prepared artifact changed: {name}")
        verified_artifacts[name] = actual
    return {"parent_path": str(parent_path), "parent_fingerprint": parent["manifest"]["fingerprint"],
            "parent_manifest_sha256": derivation["parent_manifest_sha256"],
            "parent_X_sha256": parent_X_sha, "parent_feature_count": prefix,
            "parent_feature_prefix_verified": True, "preserved_files": preserved,
            "current_X_sha256": current_X_sha, "prepared_artifact_records": verified_artifacts,
            "historical_kernel_sha256": old_config["code"]}


def _expected_indices(bundle, tasks):
    pieces = [np.flatnonzero((bundle["codes"] >= task["test_first"])
                            & (bundle["codes"] <= task["test_last"])) for task in tasks]
    indices = np.concatenate(pieces)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("Test tasks overlap")
    return indices


def verify_reused_models(bundle, registry_path, tasks, targets):
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    proof = validate_parent_identity(bundle, registry)
    old_models = registry["models"]
    lookup = {(m["estimator"], m["feature_set"], m["target"], int(m["fit_days"])): m
              for m in old_models}
    if len(lookup) != len(old_models):
        raise ValueError("Historical registry has duplicate model specifications")
    fit_days = list(dict.fromkeys(task["fit_days"] for task in tasks))
    parent_tasks = {(int(t["fit_days"]), t["month"]): t for t in registry["config"]["tasks"]}
    for task in tasks:
        if parent_tasks.get((task["fit_days"], task["month"])) != task:
            raise ValueError(f"Historical calendar task differs: {task['month']}")
    verified = []
    for fit in fit_days:
        expected = bundle["keys"].iloc[_expected_indices(bundle, [t for t in tasks if t["fit_days"] == fit])]
        for target in targets:
            for feature in SOCIAL_FEATURE_SETS:
                for estimator in ESTIMATORS:
                    key = (estimator, feature, target, fit)
                    if key not in lookup:
                        raise ValueError(f"Missing historical model: {key}")
                    entry = dict(lookup[key])
                    if (entry.get("model_id") != linear.model_name(estimator, feature, target, fit)
                            or entry.get("target_column") != linear.TARGET_COLUMNS[target]):
                        raise ValueError(f"Historical model metadata mismatch: {key}")
                    path = _resolve(registry_path.parent, entry["predictions"])
                    if sha256_file(path) != entry.get("sha256"):
                        raise ValueError(f"Historical prediction hash mismatch: {path}")
                    if entry.get("validation_days") != 126:
                        raise ValueError("Historical validation period differs")
                    frame = pd.read_pickle(path)
                    if not frame.index.is_unique or len(frame) != entry["rows"]:
                        raise ValueError(f"Invalid historical prediction rows: {path}")
                    aligned = frame.reindex(expected.index)
                    if (not aligned[["date", "permno"]].equals(expected[["date", "permno"]])
                            or not np.isfinite(aligned["prediction"].to_numpy(dtype=float)).all()):
                        raise ValueError(f"Historical prediction key/coverage mismatch: {path}")
                    entry.update(predictions=str(path), reused=True,
                                 reused_from_registry=str(registry_path), requested_rows=len(expected))
                    verified.append(entry)
        print(f"VERIFIED social-only F{fit}: {len(targets) * len(SOCIAL_FEATURE_SETS) * len(ESTIMATORS)} models", flush=True)
    proof.update(registry=str(registry_path), registry_sha256=sha256_file(registry_path),
                 models=[{"model_id": m["model_id"], "sha256": m["sha256"],
                          "predictions": m["predictions"], "requested_rows": m["requested_rows"]}
                         for m in verified])
    return verified, proof


def local_feature_columns(bundle, feature_set, union=UNION_FEATURE_SET):
    """Translate manifest-global indices to the actual make_block column order."""
    union_columns = data.feature_columns(bundle, union)
    if len(set(union_columns)) != len(union_columns):
        raise ValueError("Duplicate columns in union feature set")
    inverse = {column: index for index, column in enumerate(union_columns)}
    selected = data.feature_columns(bundle, feature_set)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Empty or duplicate feature columns: {feature_set}")
    try:
        return [inverse[column] for column in selected]
    except KeyError as error:
        raise ValueError(f"Feature set {feature_set} is not contained in union") from error


def _model_names(task, target):
    return [linear.model_name(estimator, feature, target, task["fit_days"], task["validation_days"])
            for feature in FEATURE_SETS for estimator in ESTIMATORS]


def fit_month(bundle, task, target):
    block = data.make_block(bundle, task, UNION_FEATURE_SET, target)
    moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
    predictions, diagnostics = [], {}
    for feature in FEATURE_SETS:
        columns = local_feature_columns(bundle, feature)
        candidates = linear.fit_candidates(moments, columns)
        selected = linear.select_candidates(candidates, block["X_valid"][:, columns],
                                             block["y_valid"], block["codes_valid"])
        coefficients = np.column_stack([selected[est]["coefficient"] for est in ESTIMATORS])
        intercepts = np.asarray([selected[est]["intercept"] for est in ESTIMATORS])
        predictions.append(block["X_test"][:, columns] @ coefficients + intercepts)
        for estimator in ESTIMATORS:
            name = linear.model_name(estimator, feature, target, task["fit_days"], task["validation_days"])
            diagnostics[name] = {
                "selected": selected[estimator],
                "grid": [{key: value for key, value in candidate.items()
                          if key not in ("coefficient", "intercept")}
                         for candidate in candidates if candidate["estimator"] == estimator],
                "features": [block["feature_names"][j] for j in columns],
                "feature_mean": block["mean"][columns], "feature_scale": block["scale"][columns],
                "constant": block["constant"][columns], "all_missing": block["all_missing"][columns]}
    values = np.column_stack(predictions)
    return {"month": task["month"], "target": target, "split": block["split"],
            "test_indices": np.asarray(block["test_indices"], dtype=np.int64),
            "predictions": values, "prediction_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "model_names": _model_names(task, target), "diagnostics": diagnostics,
            "fit_rows": len(block["X_fit"]), "validation_rows": len(block["X_valid"]),
            "fit_dates": int(len(np.unique(block["codes_fit"]))),
            "validation_dates": int(len(np.unique(block["codes_valid"])))}


def validate_month(result, task, target, bundle):
    expected, names = _expected_indices(bundle, [task]), _model_names(task, target)
    values = np.asarray(result["predictions"])
    if (result["month"] != task["month"] or result["target"] != target
            or not np.array_equal(result["test_indices"], expected)
            or result["model_names"] != names or values.shape != (len(expected), len(names))
            or not np.isfinite(values).all()
            or hashlib.sha256(values.tobytes()).hexdigest() != result.get("prediction_sha256")
            or set(result["diagnostics"]) != set(names)):
        raise ValueError(f"Invalid characteristic checkpoint: {target}/{task['month']}")
    for feature in FEATURE_SETS:
        expected_names = [bundle["manifest"]["feature_names"][j]
                          for j in data.feature_columns(bundle, feature)]
        for estimator in ESTIMATORS:
            name = linear.model_name(estimator, feature, target, task["fit_days"], task["validation_days"])
            diagnostic = result["diagnostics"][name]
            if diagnostic["features"] != expected_names:
                raise ValueError(f"Checkpoint feature order differs: {name}")


def checkpoint_month(prepared, task, target, run_dir, run_fingerprint, threads=2):
    bundle = data.load_bundle(prepared)
    path = Path(run_dir) / "checkpoints" / f"fit{task['fit_days']}_{target}_{task['month']}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists():
            with path.open("rb") as source:
                saved = pickle.load(source)
            if saved["fingerprint"] != run_fingerprint or saved["task"] != task:
                raise ValueError(f"Characteristic checkpoint identity mismatch: {path}")
            validate_month(saved["result"], task, target, bundle)
            print(f"RESUME F{task['fit_days']} {target} {task['month']}", flush=True)
            return str(path)
        begin = time.monotonic()
        with threadpool_limits(limits=threads):
            result = fit_month(bundle, task, target)
        validate_month(result, task, target, bundle)
        result["elapsed_seconds"] = time.monotonic() - begin
        atomic_pickle({"fingerprint": run_fingerprint, "task": task, "result": result}, path)
        print(f"SAVED F{task['fit_days']} {target} {task['month']}: {result['elapsed_seconds']:.1f}s", flush=True)
        return str(path)


def publish_run(bundle, run_dir, config, checkpoint_paths, reused_models):
    run_dir = Path(run_dir)
    grouped, monthly, seen = {}, [], set()
    expected_tasks = {(t["fit_days"], t["month"], target): t
                      for t in config["tasks"] for target in config["targets"]}
    for path in checkpoint_paths:
        with Path(path).open("rb") as source:
            saved = pickle.load(source)
        task, result = saved["task"], saved["result"]
        key = (task["fit_days"], task["month"], result["target"])
        if (saved["fingerprint"] != config["fingerprint"] or key in seen
                or expected_tasks.get(key) != task):
            raise ValueError("Publication checkpoint identity or task mismatch")
        validate_month(result, task, result["target"], bundle)
        seen.add(key)
        for column, name in enumerate(result["model_names"]):
            grouped.setdefault(name, []).append((result["test_indices"], result["predictions"][:, column]))
            chosen = result["diagnostics"][name]["selected"]
            monthly.append({"model_id": name, "month": result["month"],
                            "fit_rows": result["fit_rows"], "validation_rows": result["validation_rows"],
                            "test_rows": len(result["test_indices"]),
                            "elapsed_month_seconds": result["elapsed_seconds"],
                            **{key: value for key, value in chosen.items()
                               if key not in ("coefficient", "intercept")}})
    if seen != set(expected_tasks):
        raise ValueError("Cannot publish an incomplete characteristic run")
    if code_identity() != config["code"]:
        raise ValueError("Fitting source files changed during the run")
    models = list(reused_models)
    for entry in models:
        if sha256_file(entry["predictions"]) != entry["sha256"]:
            raise ValueError(f"Reused prediction changed during fitting: {entry['model_id']}")
    for fit in config["fit_days"]:
        for target in config["targets"]:
            for feature in FEATURE_SETS:
                for estimator in ESTIMATORS:
                    name = linear.model_name(estimator, feature, target, fit, config["validation_days"])
                    pieces = grouped.pop(name)
                    indices = np.concatenate([p[0] for p in pieces])
                    frame = bundle["keys"].iloc[indices].copy()
                    frame["prediction"] = np.concatenate([p[1] for p in pieces])
                    frame.sort_values(["date", "permno"], inplace=True)
                    if not frame.index.is_unique or frame.duplicated(["date", "permno"]).any():
                        raise ValueError(f"Duplicate published keys: {name}")
                    path = run_dir / f"predictions_{name}.pkl"
                    atomic_pickle(frame, path)
                    entry = dict(model_id=name, estimator=estimator, feature_set=feature, target=target,
                                 target_column=linear.TARGET_COLUMNS[target], fit_days=fit,
                                 validation_days=config["validation_days"],
                                 predictions=str(path.resolve()), rows=len(frame),
                                 months=int(frame.date.dt.to_period("M").nunique()),
                                 sha256=sha256_file(path), reused=False)
                    atomic_json(entry | {"fingerprint": config["fingerprint"], "schema_version": SCHEMA},
                                path.with_suffix(".json"))
                    models.append(entry)
    if grouped or len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Unexpected or duplicate published models")
    expected_models = len(config["fit_days"]) * len(config["targets"]) * len(ESTIMATORS) * (len(FEATURE_SETS) + len(SOCIAL_FEATURE_SETS))
    if len(models) != expected_models:
        raise ValueError("Publication does not contain the complete matched design")
    summary = run_dir / "monthly_selection.csv"
    temporary = summary.with_suffix(".csv.tmp")
    pd.DataFrame(monthly).to_csv(temporary, index=False)
    os.replace(temporary, summary)
    registry = dict(schema_version=SCHEMA, prepared=config["prepared"], fingerprint=config["fingerprint"],
                    config=config, models=models, monthly_selection=str(summary.resolve()),
                    reuse_audit=str((run_dir / "reuse_audit.json").resolve()))
    atomic_json(registry, run_dir / "registry.json")
    atomic_json(dict(fingerprint=config["fingerprint"], models=len(models),
                     reused_models=len(reused_models), checkpoints=len(seen)), run_dir / "complete.json")
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--linear-registry", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parents[1] / ".runs" / "characteristics_v1" / "linear")
    parser.add_argument("--fit-days", nargs="+", type=int, choices=[504, 252, 756], default=[504, 252, 756])
    parser.add_argument("--targets", nargs="+", choices=["raw", "dgtw"], default=["raw", "dgtw"])
    parser.add_argument("--start", default="2014-01")
    parser.add_argument("--end", default="2022-12")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    args.fit_days, args.targets = list(dict.fromkeys(args.fit_days)), list(dict.fromkeys(args.targets))
    bundle = data.load_bundle(args.prepared)
    for feature in FEATURE_SETS:
        local_feature_columns(bundle, feature)
    tasks = [task for fit in args.fit_days for task in data.make_tasks(
        bundle, fit_days=fit, validation_days=126, horizon=1,
        start=args.start, end=args.end, months=args.months)]
    if not tasks:
        raise ValueError("No test tasks were requested")
    reused, proof = verify_reused_models(bundle, args.linear_registry, tasks, args.targets)
    config = dict(schema_version=SCHEMA, prepared=str(args.prepared.resolve()),
                  prepared_manifest_sha256=sha256_file(args.prepared / "manifest.json"),
                  data_manifest=bundle["manifest"], fit_days=args.fit_days, validation_days=126,
                  horizon=1, targets=args.targets, feature_sets=list(FEATURE_SETS),
                  reused_feature_sets=list(SOCIAL_FEATURE_SETS), estimators=list(ESTIMATORS),
                  tasks=tasks, grid=linear.grid_specification(), threads=args.threads,
                  code=code_identity(), reuse_proof=proof,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__),
                  run_kind="pilot" if args.months else "full")
    config["fingerprint"] = fingerprint(config)
    run_dir = args.out_root / f"{config['run_kind']}_{config['fingerprint'][:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_dir) + ".lock", timeout=0):
        atomic_json(config, run_dir / "config.json")
        atomic_json(proof, run_dir / "reuse_audit.json")
        work = [(task, target) for task in tasks for target in args.targets]
        print(f"RUN {run_dir.resolve()} | {len(work)} month-target jobs | workers={args.workers}, threads={args.threads}", flush=True)
        if args.workers == 1:
            paths = [checkpoint_month(str(args.prepared), task, target, run_dir,
                                      config["fingerprint"], args.threads) for task, target in work]
        else:
            paths = Parallel(n_jobs=args.workers, backend="loky", pre_dispatch=args.workers)(
                delayed(checkpoint_month)(str(args.prepared), task, target, str(run_dir),
                                          config["fingerprint"], args.threads) for task, target in work)
        if sha256_file(args.prepared / "manifest.json") != config["prepared_manifest_sha256"]:
            raise ValueError("Prepared manifest changed during fitting")
        publish_run(bundle, run_dir, config, paths, reused)
        print(f"REGISTRY {(run_dir / 'registry.json').resolve()}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
