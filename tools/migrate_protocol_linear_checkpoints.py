"""Stage provably unaffected linear checkpoints after the bounded FTNW repair.

This helper never trains models or publishes prediction series. It verifies the
data/code delta, copies only month-target tasks whose fitting AND validation
blocks exclude all changed label/rank dates, and leaves dependent tasks to the
unchanged protocol_linear CLI. Old runs and source inputs remain untouched.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

from filelock import FileLock
import numpy as np
import pandas as pd
import scipy
import sklearn

from nn_checkpoint import atomic_json, atomic_pickle, fingerprint
import protocol_data as data
import protocol_linear as linear


RUNTIME_FUNCTIONS = ("load_bundle", "make_tasks", "make_block", "feature_columns", "centered_rank",
                     "equal_date_weights", "_weighted_scaler", "_transform")
INPUT_FILES = ("X.npy", "keys.pkl", "codes.npy", "calendar.npy")
BAD_MM_INDEX = 9900658
BAD_PERMNO = 17182
BAD_DATE = "2019-03-13"


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def runtime_ast(source):
    nodes = ast.parse(source).body
    functions = {node.name: ast.dump(node, include_attributes=False) for node in nodes
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in RUNTIME_FUNCTIONS}
    if set(functions) != set(RUNTIME_FUNCTIONS):
        raise ValueError("A fitting runtime function is absent from the source snapshot")
    return functions


def verify_runtime_code(old_config, source_run):
    current = linear.code_identity()
    old_data_path = Path(source_run)/"code"/"protocol_data.py"
    if file_hash(old_data_path) != old_config["code"]["protocol_data.py"]:
        raise ValueError("Archived original data code does not match the source run")
    for name, digest in old_config["code"].items():
        if name != "protocol_data.py" and current.get(name) != digest:
            raise ValueError(f"Training/scoring/checkpoint code changed: {name}")
    original = runtime_ast(old_data_path.read_text(encoding="utf-8"))
    revised = runtime_ast(Path(data.__file__).read_text(encoding="utf-8"))
    if original != revised:
        raise ValueError("Data preparation correction changed a fitting runtime function")
    versions = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    if old_config["versions"] != versions or old_config["grid"] != linear.grid_specification():
        raise ValueError("Numerical libraries or hyperparameter/solver specification changed")
    return {"original_data_code": str(old_data_path.resolve()),
            "original_data_code_sha256": file_hash(old_data_path), "current_code": current,
            "ast_equal_functions": list(RUNTIME_FUNCTIONS),
            "runtime_ast_sha256": fingerprint(original)}


def changed_cells(old, new, chunk_rows=65536):
    if old.shape != new.shape or old.dtype != new.dtype or old.ndim != 2:
        raise ValueError("Label/rank array shape or dtype changed")
    result = []
    for start in range(0, len(old), chunk_rows):
        a, b = np.asarray(old[start:start+chunk_rows]), np.asarray(new[start:start+chunk_rows])
        different = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        rows, columns = np.where(different)
        result.extend((int(row+start), int(column)) for row, column in zip(rows, columns))
    return result


def verify_array_delta(old_bundle, new_bundle, declaration):
    """Require exact declared changes, then verify the bounded repair semantics."""
    actual = {}
    for name in ("y", "q"):
        actual[name] = changed_cells(old_bundle[name], new_bundle[name])
        declared = [tuple(map(int, pair)) for pair in declaration[name]]
        if len(set(declared)) != len(declared) or set(actual[name]) != set(declared):
            raise ValueError(f"Unexpected or missing changes in {name}")
    keys = old_bundle["keys"]
    if not keys.index.is_unique or BAD_MM_INDEX not in keys.index:
        raise ValueError("Bounded FTNW repair key is absent or duplicated")
    row = int(keys.index.get_loc(BAD_MM_INDEX))
    security = keys.iloc[row]
    if security["permno"] != BAD_PERMNO or str(pd.Timestamp(security["date"]).date()) != BAD_DATE:
        raise ValueError("Bounded FTNW repair key does not identify the expected security/date")
    expected_y_cells = {(row, j) for j in range(2) if not np.isnan(old_bundle["y"][row, j])}
    if set(actual["y"]) != expected_y_cells or not np.isnan(new_bundle["y"][row]).all():
        raise ValueError("Raw labels changed beyond nulling the two FTNW one-day targets")
    code = int(old_bundle["codes"][row])
    same_date = np.flatnonzero(old_bundle["codes"] == code)
    for column in range(2):
        expected_q = data.centered_rank(new_bundle["y"][same_date, column],
                                       new_bundle["codes"][same_date], target=True)
        if not np.array_equal(expected_q, new_bundle["q"][same_date, column], equal_nan=True):
            raise ValueError("Corrected ranks do not equal freshly computed same-date target ranks")
    if any(int(old_bundle["codes"][r]) != code for r, _ in actual["q"]):
        raise ValueError("Rank corrections extend beyond the declared FTNW date")
    changed_dates = {}
    for column, target in enumerate(("raw", "dgtw")):
        rows = {r for name in ("y", "q") for r, c in actual[name] if c == column}
        changed_dates[target] = sorted({int(old_bundle["codes"][r]) for r in rows})
    return {"changed_cells": {name: [list(pair) for pair in values] for name, values in actual.items()},
            "changed_date_codes": changed_dates, "affected_mm_index": BAD_MM_INDEX,
            "affected_date": BAD_DATE}


def depends_on_changed_labels(task, changed_dates):
    return any(task[f"{part}_first"] <= date <= task[f"{part}_last"]
               for part in ("fit", "valid") for date in changed_dates)


def verify_inputs(old_path, new_path, old_bundle, new_bundle):
    hashes = {}
    for name in INPUT_FILES:
        old_file, new_file = Path(old_path)/name, Path(new_path)/name
        previous = file_hash(old_file)
        current = previous if os.path.samefile(old_file, new_file) else file_hash(new_file)
        if previous != current:
            raise ValueError(f"Predictor/universe/calendar file changed: {name}")
        hashes[name] = {"old_sha256": previous, "new_sha256": current,
                        "same_file": os.path.samefile(old_file, new_file)}
    for key in ("protocol", "rows", "feature_names", "feature_sets", "binary_features", "targets",
                "rank_min_labels", "source", "crsp_sources", "source_timing", "numpy", "pandas"):
        if old_bundle["manifest"].get(key) != new_bundle["manifest"].get(key):
            raise ValueError(f"Fitting-relevant manifest field changed: {key}")
    return hashes


def corrected_config(old_config, new_path, bundle):
    new = copy.deepcopy(old_config)
    new.pop("fingerprint", None)
    if old_config["schema_version"] != linear.SCHEMA or old_config["run_kind"] != "full":
        raise ValueError("Migration requires the full protocol linear experiment")
    tasks = old_config["tasks"]
    months = sorted({task["month"] for task in tasks})
    generated = [task for fit_days in old_config["fit_days"] for task in data.make_tasks(
        bundle, fit_days=fit_days, validation_days=old_config["validation_days"],
        horizon=old_config["horizon"], start=months[0], end=months[-1])]
    if generated != tasks:
        raise ValueError("Corrected run's fitting, validation or test tasks differ")
    new.update(prepared=str(Path(new_path).resolve()), data_manifest=bundle["manifest"], code=linear.code_identity())
    new["fingerprint"] = fingerprint(new)
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--corrected-prepared", required=True, type=Path)
    parser.add_argument("--correction-audit", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1]/".runs"/"protocol_v1_1"/"linear")
    args = parser.parse_args(argv)
    source_run = args.source_run.resolve()
    old = json.loads((source_run/"config.json").read_text(encoding="utf-8"))
    if fingerprint({k: v for k, v in old.items() if k != "fingerprint"}) != old["fingerprint"]:
        raise ValueError("Source configuration fingerprint is invalid")
    old_bundle, new_bundle = data.load_bundle(old["prepared"]), data.load_bundle(args.corrected_prepared)
    if old_bundle["manifest"] != old["data_manifest"]:
        raise ValueError("Original prepared manifest changed since the source run")
    code_audit = verify_runtime_code(old, source_run)
    input_audit = verify_inputs(old["prepared"], args.corrected_prepared, old_bundle, new_bundle)
    correction = json.loads(args.correction_audit.read_text(encoding="utf-8"))
    delta = verify_array_delta(old_bundle, new_bundle, correction["changed_cells"])
    new = corrected_config(old, args.corrected_prepared, new_bundle)
    destination = args.run_root.resolve()/f"full_{new['fingerprint'][:16]}"
    if destination == source_run:
        raise ValueError("Migration must target a distinct corrected run")
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination/"migration.lock"), timeout=0):
        if (destination/"config.json").exists():
            if json.loads((destination/"config.json").read_text(encoding="utf-8")) != new:
                raise ValueError("Destination configuration mismatch")
        atomic_json(new, destination/"config.json")
        audit = {"source_run": str(source_run), "source_fingerprint": old["fingerprint"],
                 "destination": str(destination), "destination_fingerprint": new["fingerprint"],
                 "correction_audit": str(args.correction_audit.resolve()),
                 "correction_audit_sha256": file_hash(args.correction_audit),
                 "migration_code_sha256": file_hash(__file__), "runtime_code": code_audit,
                 "unchanged_inputs": input_audit, "label_delta": delta, "jobs": []}
        (destination/"checkpoints").mkdir(exist_ok=True)
        for task in old["tasks"]:
            for target in old["targets"]:
                name = f"fit{task['fit_days']}_{target}_{task['month']}.pkl"
                source, dest = source_run/"checkpoints"/name, destination/"checkpoints"/name
                dependent = depends_on_changed_labels(task, delta["changed_date_codes"][target])
                record = {"month": task["month"], "fit_days": task["fit_days"], "target": target,
                          "dependent": dependent, "checkpoint": str(dest)}
                with FileLock(str(dest)+".lock", timeout=0):
                    if dest.exists():
                        saved = pd.read_pickle(dest)
                        if saved["fingerprint"] != new["fingerprint"] or saved["task"] != task:
                            raise ValueError(f"Existing corrected checkpoint has incompatible identity: {dest}")
                        linear.validate_month(saved["result"], task, target, new_bundle)
                        if saved.get("migration"):
                            if dependent or saved["migration"].get("source_fingerprint") != old["fingerprint"]:
                                raise ValueError(f"Existing migration has incompatible dependency/provenance: {dest}")
                            record.update(action="already_reused", **saved["migration"])
                        else:
                            record["action"] = "already_present"
                    elif dependent:
                        record["action"] = "retrain_dependency"
                    elif not source.exists():
                        record["action"] = "retrain_missing_source"
                    else:
                        saved = pd.read_pickle(source)
                        if saved["fingerprint"] != old["fingerprint"] or saved["task"] != task:
                            raise ValueError(f"Original checkpoint identity mismatch: {source}")
                        linear.validate_month(saved["result"], task, target, old_bundle)
                        linear.validate_month(saved["result"], task, target, new_bundle)
                        provenance = {"source_checkpoint": str(source), "source_sha256": file_hash(source),
                                      "source_fingerprint": old["fingerprint"],
                                      "source_result_sha256": hashlib.sha256(pickle.dumps(saved["result"], protocol=pickle.HIGHEST_PROTOCOL)).hexdigest(),
                                      "reason": "No changed label or rank date enters this target's fitting or validation block; test labels do not affect linear predictions.",
                                      "changed_date_codes": delta["changed_date_codes"][target]}
                        atomic_pickle({"fingerprint": new["fingerprint"], "task": task,
                                       "result": saved["result"], "migration": provenance}, dest)
                        record.update(action="reused", **provenance)
                audit["jobs"].append(record)
        counts = pd.Series([job["action"] for job in audit["jobs"]]).value_counts().to_dict()
        audit["counts"] = {str(k): int(v) for k, v in counts.items()}
        audit["resume_command"] = [sys.executable, "-B", str(Path(linear.__file__).resolve()),
            "--prepared", str(args.corrected_prepared.resolve()), "--fit-days", *map(str, new["fit_days"]),
            "--targets", *new["targets"], "--start", min(t["month"] for t in new["tasks"]),
            "--end", max(t["month"] for t in new["tasks"]), "--threads", str(new["threads"]),
            "--run-root", str(args.run_root.resolve())]
        atomic_json(audit, destination/"migration_audit.json")
        print(json.dumps({"destination": str(destination), "counts": audit["counts"],
                          "resume_command": audit["resume_command"]}, indent=2), flush=True)
    return destination


if __name__ == "__main__":
    main()
