"""Run the authorized full NN3 study after verifying all revised linear results.

This foreground controller can itself be launched asynchronously. It supervises
local child processes, publishes atomic progress, and resumes NN monthly caches
when the same command is rerun. It never starts a scheduler or network service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import numpy as np
import pandas as pd

from nn_checkpoint import atomic_json, fingerprint


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("core", "all", "textcore", "textall")
TARGETS = ("raw", "dgtw")
WINDOWS = (252, 504, 756)
LINEAR = ("ols", "ridge", "lasso", "enet")
NN = ("nn3",)
MONTHS = tuple(str(x) for x in pd.period_range("2014-01", "2022-12", freq="M"))


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base)/path).resolve()


def read_registry(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    for model in document.get("models", []):
        model["predictions"] = str(resolve_path(path.parent, model["predictions"]))
    if document.get("monthly_selection"):
        document["monthly_selection"] = str(resolve_path(path.parent, document["monthly_selection"]))
    return document


def validate_matrix(document, prepared, family):
    """Reject partial grids, pilots, alternative histories and budget changes."""
    if family not in ("linear", "nn"):
        raise ValueError("Unknown model family")
    if Path(document.get("prepared", "")).resolve() != Path(prepared).resolve():
        raise ValueError(f"{family} registry uses a different prepared dataset")
    estimators = LINEAR if family == "linear" else NN
    expected = set(itertools.product(estimators, FEATURES, TARGETS, WINDOWS))
    models = document.get("models", [])
    specifications, identifiers = [], []
    for record in models:
        try:
            specification = (record["estimator"], record["feature_set"], record["target"], int(record["fit_days"]))
            identifiers.append(record["model_id"])
            if int(record["months"]) != len(MONTHS) or int(record["rows"]) <= 0:
                raise ValueError("Every model must cover all 108 months with positive row counts")
            if int(record["validation_days"]) != 126 or int(record.get("horizon", 1)) != 1:
                raise ValueError("Nonprotocol validation length or target horizon")
            expected_target = {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[record["target"]]
            if record["target_column"] != expected_target:
                raise ValueError("Target alias and target column disagree")
            if not record.get("sha256") or not record.get("predictions"):
                raise ValueError("Prediction path or file digest is absent")
            if record.get("kind", "full") != "full":
                raise ValueError("Pilot predictions cannot enter the full study")
            if family == "nn" and (record.get("coverage_scope") != "2014-2022"
                                    or record.get("training_budget") != "standard"):
                raise ValueError("NN coverage or training budget is incomplete")
            specifications.append(specification)
        except KeyError as exc:
            raise ValueError(f"{family} registry is missing {exc}") from exc
    if (set(specifications) != expected or len(specifications) != len(expected)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError(f"{family} registry must contain the exact unique {len(expected)}-model matrix")
    if family == "linear":
        if document.get("config", {}).get("run_kind") != "full":
            raise ValueError("Linear registry is not a full training run")
    else:
        config = document.get("config", {})
        if (document.get("kind") != "full" or config.get("max_epochs") != 100
                or config.get("seeds") != [7, 1007, 2007, 3007, 4007]
                or config.get("widths") != [128, 64, 32]):
            raise ValueError("NN registry must use the declared NN3 five-seed/100-epoch recipe")
    return models


def verify_prediction_files(models, prepared):
    """Check actual keyed coverage and hashes, rather than trusting month counts."""
    keys = pd.read_pickle(Path(prepared)/"keys.pkl")
    keys["date"] = pd.to_datetime(keys["date"])
    expected = keys.loc[keys.date.between("2014-01-01", "2022-12-31"), ["date", "permno"]]
    if (expected.empty or expected.isna().any().any() or expected.duplicated(["date", "permno"]).any()
            or tuple(sorted(expected.date.dt.to_period("M").astype(str).unique())) != MONTHS):
        raise ValueError("Prepared universe does not cover all 108 required months")
    expected = expected.sort_values(["date", "permno"])
    dates = expected.date.to_numpy(dtype="datetime64[ns]")
    securities = expected.permno.to_numpy()
    checked = {}
    for model in models:
        path = Path(model["predictions"])
        if path in checked:
            actual_hash = checked[path]
        else:
            actual_hash = file_hash(path)
            frame = pd.read_pickle(path)
            required = {"date", "permno", "prediction"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing prediction columns: {path}")
            if (len(frame) != len(expected)
                    or not np.array_equal(frame.date.to_numpy(dtype="datetime64[ns]"), dates)
                    or not np.array_equal(frame.permno.to_numpy(), securities)
                    or not np.isfinite(frame.prediction.to_numpy(dtype=float)).all()):
                raise ValueError(f"Prediction coverage/order/finiteness does not match prepared full span: {path}")
            checked[path] = actual_hash
        if actual_hash != model["sha256"] or int(model["rows"]) != len(expected):
            raise ValueError(f"Prediction digest or row-count mismatch: {path}")
    return {str(path): digest for path, digest in checked.items()}


def combine_registries(linear, neural, prepared, linear_path, nn_path):
    validate_matrix(linear, prepared, "linear")
    validate_matrix(neural, prepared, "nn")
    models = linear["models"]+neural["models"]
    if len({model["model_id"] for model in models}) != 120:
        raise ValueError("Combined registry must contain 120 unique model IDs")
    result = {"schema_version": "protocol_v1_1_combined", "kind": "full",
              "coverage_scope": "2014-2022", "prepared": str(Path(prepared).resolve()),
              "models": models, "monthly_selection": linear.get("monthly_selection"),
              "config": {"linear": linear["config"], "nn": neural["config"]},
              "source_registries": [{"family": family, "path": str(Path(path).resolve()),
                                     "fingerprint": document["fingerprint"]}
                                    for family, path, document in [("linear", linear_path, linear),
                                                                  ("nn", nn_path, neural)]]}
    result["fingerprint"] = fingerprint(result)
    return result


def last_registry(log_path):
    found = None
    with Path(log_path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("Registry:"):
                found = Path(line.partition(":")[2].strip()).resolve()
    if found is None or not found.is_file():
        raise ValueError("NN subprocess did not print an existing final registry path")
    return found


def code_hashes():
    names = ["run_protocol_nn_study.py", "protocol_nn.py", "protocol_nn_metrics.py",
             "protocol_data.py", "prediction_metrics.py", "nn_checkpoint.py",
             "protocol_evaluate.py", "protocol_report.py"]
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def update_status(path, state, **changes):
    state.update(changes)
    state["updated_at"] = timestamp()
    atomic_json(state, path)


def run_child(command, log_path, phase, status_path, state, environment):
    """Block this controller, not the assistant, until its supervised child exits."""
    command = [str(value) for value in command]
    update_status(status_path, state, phase=phase, command=command, log=str(log_path), child_pid=None)
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, shell=False,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        update_status(status_path, state, child_pid=process.pid)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    update_status(status_path, state, child_pid=process.pid)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
    state.setdefault("commands", []).append({"phase": phase, "command": command,
                                             "log": str(log_path), "exit_code": returncode})
    update_status(status_path, state, child_pid=None, last_exit_code=returncode)
    if returncode:
        raise RuntimeError(f"{phase} failed with exit code {returncode}; see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-registry", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=ROOT/".runs"/"protocol_v1_1"/"nn_study")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.threads < 1:
        parser.error("Use 1–8 workers and a positive thread count")
    args.prepared, args.linear_registry = args.prepared.resolve(), args.linear_registry.resolve()
    linear = read_registry(args.linear_registry)
    identity = {"schema": "protocol_v1_1_nn_study", "prepared": str(args.prepared),
                "prepared_manifest_sha256": file_hash(args.prepared/"manifest.json"),
                "linear_registry": str(args.linear_registry),
                "linear_fingerprint": linear.get("fingerprint"),
                "linear_registry_sha256": file_hash(args.linear_registry),
                "code": code_hashes(), "threads": args.threads, "python": sys.version}
    signature = fingerprint(identity)
    run_dir = args.out_root.resolve()/signature[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir/"status.json"
    with FileLock(str(run_dir/"study.lock"), timeout=0):
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        attempt = int(previous.get("attempt", 0))+1
        state = {"status": "running", "phase": "verify_linear", "pid": os.getpid(),
                 "child_pid": None, "attempt": attempt, "started_at": timestamp(),
                 "fingerprint": signature, "run_dir": str(run_dir), "commands": [],
                 "linear_registry": str(args.linear_registry), "prepared": str(args.prepared)}
        atomic_json(identity, run_dir/"study_spec.json")
        update_status(status_path, state)
        print(f"STUDY {run_dir}; status: {status_path}", flush=True)
        try:
            snapshot = run_dir/"code"
            snapshot.mkdir(exist_ok=True)
            for name, expected in identity["code"].items():
                content = Path(__file__).with_name(name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Code changed while creating its snapshot: {name}")
                destination = snapshot/name
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f"Existing code snapshot differs: {destination}")
                destination.write_bytes(content)
            validate_matrix(linear, args.prepared, "linear")
            completion = json.loads((args.linear_registry.parent/"complete.json").read_text())
            if completion.get("fingerprint") != linear["fingerprint"] or completion.get("models") != 96:
                raise ValueError("Linear completion marker does not certify the 96-model registry")
            source_hashes = verify_prediction_files(linear["models"], args.prepared)
            env = os.environ.copy()
            env.update({name: str(args.threads) for name in
                        ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]})
            env.update(KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            nn_command = [sys.executable, "-B", ROOT/"tools"/"protocol_nn.py",
                          "--prepared", args.prepared, "--workers", args.workers, "--threads", args.threads,
                          "--architecture", 3, "--fit-days", 504, 252, 756,
                          "--features", "core", "textcore", "all", "textall",
                          "--targets", "raw", "dgtw", "--start", "2014-01", "--end", "2022-12",
                          "--seeds", 5, "--max-epochs", 100, "--out-root", run_dir/"nn"]
            nn_log = run_dir/f"nn_attempt_{attempt:03}.log"
            run_child(nn_command, nn_log, "train_nn", status_path, state, env)
            update_status(status_path, state, phase="verify_and_merge")
            if code_hashes() != identity["code"]:
                raise ValueError("Study code changed during training; rerun under its new identity")
            if (file_hash(args.linear_registry) != identity["linear_registry_sha256"]
                    or file_hash(args.prepared/"manifest.json") != identity["prepared_manifest_sha256"]):
                raise ValueError("A source registry or prepared manifest changed during training")
            for path, digest in source_hashes.items():
                if file_hash(path) != digest:
                    raise ValueError(f"A linear prediction source changed during training: {path}")
            nn_path = last_registry(nn_log)
            neural = read_registry(nn_path)
            validate_matrix(neural, args.prepared, "nn")
            verify_prediction_files(neural["models"], args.prepared)
            combined = combine_registries(linear, neural, args.prepared, args.linear_registry, nn_path)
            combined_path = run_dir/"combined_registry.json"
            atomic_json(combined, combined_path)
            evaluation = ROOT/"reports"/"data"/"protocol_v1_1_combined"
            report = ROOT/"reports"/"08_protocol_nn_results.md"
            update_status(status_path, state, nn_registry=str(nn_path), combined_registry=str(combined_path),
                          evaluation=str(evaluation), report=str(report))
            run_child([sys.executable, "-B", ROOT/"tools"/"protocol_evaluate.py", "--registry", combined_path,
                       "--prepared", args.prepared, "--out", evaluation],
                      run_dir/f"evaluation_attempt_{attempt:03}.log", "evaluate", status_path, state, env)
            run_child([sys.executable, "-B", ROOT/"tools"/"protocol_report.py", "--registry", combined_path,
                       "--evaluation", evaluation, "--out", report, "--title", "Protocol v1.1: NN3 and matched linear results"],
                      run_dir/f"report_attempt_{attempt:03}.log", "report", status_path, state, env)
            if not evaluation.with_suffix(".json").is_file() or not report.is_file() or not report.stat().st_size:
                raise ValueError("Evaluation or report artifact is missing after successful subprocess exit")
            update_status(status_path, state, status="complete", phase="complete", finished_at=timestamp(),
                          models=120, artifact_sha256={str(p): file_hash(p) for p in
                              [combined_path, evaluation.with_suffix(".json"), report]})
            print(f"COMPLETE {status_path}", flush=True)
        except BaseException as exc:
            update_status(status_path, state, status="failed", finished_at=timestamp(),
                          error=f"{type(exc).__name__}: {exc}")
            raise
    return run_dir


if __name__ == "__main__":
    main()
