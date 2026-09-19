"""Supervise the fixed-design characteristic linear experiment, with restart support.

Preparation, a two-month implementation pilot, full linear training, evaluation and
reporting run in separate processes. No model or history is selected using pilot
test outcomes. Scientific kernels from the completed study remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from filelock import FileLock
import numpy as np
import pandas as pd
import psutil

from nn_checkpoint import atomic_json, fingerprint
from resume_protocol_nn_study import ResilientStatusWriter
import characteristic_data as characteristics


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "run_characteristic_study.py", "characteristic_data.py", "characteristic_linear.py",
    "characteristic_evaluate.py", "characteristic_report.py", "protocol_data.py",
    "protocol_linear.py", "protocol_evaluate.py", "prediction_metrics.py",
    "nn_checkpoint.py", "resume_protocol_nn_study.py",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_identity():
    return {name: sha256(Path(__file__).with_name(name)) for name in SOURCES}


def _update_status(path, state, **changes):
    state.update(changes)
    state["updated_at"] = timestamp()
    atomic_json(state, path)


update_status = ResilientStatusWriter(_update_status)


def stop_owned_tree(process):
    """Remove this controller's owned subprocess tree after an actual phase failure."""
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for child in owned:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def run_phase(command, log_path, phase, status_path, state, environment):
    command = [str(value) for value in command]
    update_status(status_path, state, phase=phase, child_pid=None,
                  command=command, log=str(log_path))
    started = timestamp()
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            update_status(status_path, state, child_pid=process.pid)
            while True:
                try:
                    exit_code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    update_status(status_path, state, child_pid=process.pid)
        except BaseException:
            stop_owned_tree(process)
            raise
    state.setdefault("commands", []).append({
        "phase": phase, "command": command, "log": str(log_path),
        "started_at": started, "finished_at": timestamp(), "exit_code": exit_code,
    })
    update_status(status_path, state, child_pid=None, last_exit_code=exit_code)
    if exit_code:
        raise RuntimeError(f"{phase} exited with {exit_code}; see {log_path}")


def output_path(log_path, label):
    found = None
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        head, _, tail = line.partition(" ")
        if head.rstrip(":").lower() == label.lower():
            candidate = Path(tail.strip()).resolve()
            if candidate.exists():
                found = candidate
    if found is None:
        raise ValueError(f"No existing {label} path in {log_path}")
    return found


def validate_prepared(prepared, parent, source, identity):
    """Verify a supplied cache against this run's inputs, not just its location."""
    prepared, parent, source = map(lambda p: Path(p).resolve(), (prepared, parent, source))
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    derivation, recipe = manifest["derivation"], manifest["characteristics"]
    parent_manifest = json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
    if (Path(derivation["parent_path"]).resolve() != parent
            or derivation["parent_manifest_sha256"] != identity["parent_manifest_sha256"]
            or derivation["parent_fingerprint"] != parent_manifest["fingerprint"]):
        raise ValueError("Characteristic cache derives from a different parent")
    origin = recipe["primitive_source"]
    observed = source.stat()
    if (Path(origin["path"]).resolve() != source or observed.st_size != origin["size"]
            or observed.st_mtime_ns != origin["mtime_ns"] or sha256(source) != origin["sha256"]):
        raise ValueError("Characteristic cache uses a different primitive CRSP source")
    if (recipe["code_sha256"] != identity["code"]["characteristic_data.py"]
            or recipe["recipes"] != characteristics.RECIPES
            or recipe["timing"] != characteristics.TIMING):
        raise ValueError("Characteristic cache uses a different recipe or builder")
    for name, expected in recipe["kernel_code_sha256"].items():
        if identity["code"].get(name) != expected:
            raise ValueError("Characteristic transformation kernel changed")
    names, sets = characteristics.feature_schema(parent_manifest)
    if manifest["feature_names"] != names or manifest["feature_sets"] != sets:
        raise ValueError("Characteristic feature schema differs from the declared design")
    for name, record in manifest["artifact_records"].items():
        path = prepared/name
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Characteristic prepared artifact changed: {name}")
    return manifest


def validate_registry(path, prepared, specification):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(registry["prepared"]).resolve() != Path(prepared).resolve():
        raise ValueError("Registry refers to a different prepared dataset")
    expected = {
        (e, f, t, w) for e in specification["estimators"]
        for f in specification["feature_sets"] for t in specification["targets"]
        for w in specification["fit_days"]
    }
    models = registry["models"]
    actual = [(m["estimator"], m["feature_set"], m["target"], int(m["fit_days"])) for m in models]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Registry does not cover the unique complete declared model grid")
    if len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Duplicate model identifiers")
    keys = pd.read_pickle(Path(prepared)/"keys.pkl")
    expected_keys = keys.loc[keys.date.between("2014-01-01", "2022-12-31"), ["date", "permno"]]
    expected_keys = expected_keys.sort_values(["date", "permno"])
    months = tuple(sorted(expected_keys.date.dt.to_period("M").astype(str).unique()))
    if (expected_keys.duplicated(["date", "permno"]).any()
            or months != tuple(str(x) for x in pd.period_range("2014-01", "2022-12", freq="M"))):
        raise ValueError("Prepared prediction keys are duplicate or do not cover all 108 months")
    dates = expected_keys.date.to_numpy(dtype="datetime64[ns]")
    permnos = expected_keys.permno.to_numpy()
    for model in models:
        source = Path(model["predictions"])
        if not source.is_absolute():
            source = Path(path).parent/source
        if (model["months"] != 108 or model["rows"] != len(expected_keys)
                or model["validation_days"] != 126 or model.get("horizon", 1) != 1
                or model["target_column"] != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]]
                or sha256(source) != model["sha256"]):
            raise ValueError(f"Invalid prediction certificate: {model['model_id']}")
        frame = pd.read_pickle(source)
        if (len(frame) != len(expected_keys)
                or not np.array_equal(frame.date.to_numpy(dtype="datetime64[ns]"), dates)
                or not np.array_equal(frame.permno.to_numpy(), permnos)
                or not np.isfinite(frame.prediction.to_numpy()).all()):
            raise ValueError(f"Invalid prediction coverage: {model['model_id']}")
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path,
                        default=ROOT/"reports/data/characteristics_v1_experiment.json")
    parser.add_argument("--parent", type=Path,
                        default=ROOT/".runs/protocol_v1_1/prepared/8f3f4eb44b399771")
    parser.add_argument("--linear-registry", type=Path,
                        default=ROOT/".runs/protocol_v1_1/linear/full_3314fb5ca2fe170a/registry.json")
    parser.add_argument("--source", type=Path, default=ROOT.parent/"Data/CRSP/dsf.pkl")
    parser.add_argument("--prepared", type=Path, help="Previously prepared, verified characteristic cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=ROOT/".runs/characteristics_v1/study")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.threads < 1:
        parser.error("Use 1-8 workers and a positive thread count")
    spec = json.loads(args.specification.read_text(encoding="utf-8"))
    if (spec["fit_days"] != [504, 252, 756] or spec["validation_days"] != 126
            or spec["test_months"] != 108 or spec["total_model_specifications"] != 264):
        raise ValueError("This controller requires the declared complete fixed-design study")
    source_stat = args.source.stat()
    identity = {
        "experiment": spec, "specification_sha256": sha256(args.specification),
        "parent": str(args.parent.resolve()), "parent_manifest_sha256": sha256(args.parent/"manifest.json"),
        "linear_registry": str(args.linear_registry.resolve()), "linear_registry_sha256": sha256(args.linear_registry),
        "source": str(args.source.resolve()), "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns, "threads": args.threads,
        "python": sys.version, "code": code_identity(),
        "supplied_prepared": str(args.prepared.resolve()) if args.prepared else None,
        "supplied_prepared_manifest_sha256": sha256(args.prepared/"manifest.json") if args.prepared else None,
    }
    signature = fingerprint(identity)
    run_dir = args.out_root.resolve()/signature[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir/"status.json"
    environment = os.environ.copy()
    environment.update({name: str(args.threads) for name in
                        ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]})
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", KMP_DUPLICATE_LIB_OK="TRUE")
    base_command = [sys.executable, "-B"]
    with FileLock(str(run_dir/"study.lock"), timeout=0):
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if previous.get("status") == "complete":
            for path, expected in previous["artifact_sha256"].items():
                if sha256(path) != expected:
                    raise ValueError("A previously completed output changed")
            print(f"COMPLETE {status_path}", flush=True)
            return run_dir
        attempt = int(previous.get("attempt", 0))+1
        if previous:
            archive = run_dir/f"status_before_attempt_{attempt:03}.json"
            atomic_json(previous, archive)
        state = {"status": "running", "phase": "initialize", "pid": os.getpid(),
                 "child_pid": None, "attempt": attempt, "started_at": timestamp(),
                 "fingerprint": signature, "run_dir": str(run_dir), "commands": []}
        atomic_json(identity, run_dir/"study_spec.json")
        snapshot = run_dir/"code"
        snapshot.mkdir(exist_ok=True)
        for name, digest in identity["code"].items():
            source = Path(__file__).with_name(name)
            if sha256(source) != digest:
                raise ValueError("Source changed while freezing the run")
            shutil.copyfile(source, snapshot/name)
        update_status(status_path, state)
        print(f"STUDY {run_dir}", flush=True)
        try:
            prepared = args.prepared.resolve() if args.prepared else None
            if prepared is None:
                log = run_dir/f"prepare_attempt_{attempt:03}.log"
                run_phase(base_command+[ROOT/"tools/characteristic_data.py", "--parent", args.parent,
                          "--source", args.source, "--out-root", ROOT/".runs/characteristics_v1/prepared"],
                          log, "prepare", status_path, state, environment)
                prepared = output_path(log, "prepared")
            validate_prepared(prepared, args.parent, args.source, identity)
            prepared_hash = sha256(prepared/"manifest.json")
            update_status(status_path, state, prepared=str(prepared), prepared_manifest_sha256=prepared_hash)
            common = [ROOT/"tools/characteristic_linear.py", "--prepared", prepared,
                      "--linear-registry", args.linear_registry, "--threads", args.threads,
                      "--out-root", run_dir/"linear"]
            log = run_dir/f"pilot_attempt_{attempt:03}.log"
            run_phase(base_command+common+["--fit-days", 504, "--months", "2014-01", "2022-12",
                      "--workers", min(2, args.workers)], log, "implementation_pilot", status_path, state, environment)
            pilot_registry = output_path(log, "registry")
            update_status(status_path, state, pilot_registry=str(pilot_registry))
            log = run_dir/f"linear_attempt_{attempt:03}.log"
            run_phase(base_command+common+["--fit-days", 504, 252, 756, "--workers", args.workers],
                      log, "train_linear", status_path, state, environment)
            registry_path = output_path(log, "registry")
            update_status(status_path, state, registry=str(registry_path), phase="verify_predictions")
            validate_registry(registry_path, prepared, spec)
            if (code_identity() != identity["code"]
                    or sha256(prepared/"manifest.json") != prepared_hash
                    or sha256(args.specification) != identity["specification_sha256"]
                    or sha256(args.parent/"manifest.json") != identity["parent_manifest_sha256"]
                    or sha256(args.linear_registry) != identity["linear_registry_sha256"]):
                raise ValueError("Study sources changed during training")
            prefix = ROOT/spec["evaluation_prefix"]
            report = ROOT/spec["report"]
            update_status(status_path, state, evaluation=str(prefix), report=str(report))
            run_phase(base_command+[ROOT/"tools/characteristic_evaluate.py", "--registry", registry_path,
                      "--prepared", prepared, "--out", prefix, "--threads", args.threads], run_dir/f"evaluation_attempt_{attempt:03}.log",
                      "evaluate", status_path, state, environment)
            run_phase(base_command+[ROOT/"tools/characteristic_report.py", "--registry", registry_path,
                      "--evaluation", prefix, "--out", report], run_dir/f"report_attempt_{attempt:03}.log",
                      "report", status_path, state, environment)
            if code_identity() != identity["code"]:
                raise ValueError("Study sources changed during evaluation/reporting")
            artifacts = [registry_path, prefix.with_suffix(".json"), report]
            evaluation = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
            for item in evaluation["output_files"].values():
                artifact = Path(item["path"])
                if sha256(artifact) != evaluation["output_sha256"][artifact.name]:
                    raise ValueError(f"Evaluation output changed: {artifact}")
                artifacts.append(artifact)
            if any(not path.is_file() or path.stat().st_size == 0 for path in artifacts):
                raise ValueError("A required completion artifact is absent or empty")
            update_status(status_path, state, status="complete", phase="complete",
                          finished_at=timestamp(), models=264,
                          artifact_sha256={str(path): sha256(path) for path in artifacts})
            print(f"COMPLETE {status_path}", flush=True)
        except BaseException as error:
            update_status(status_path, state, status="failed", finished_at=timestamp(),
                          error=f"{type(error).__name__}: {error}")
            raise
    return run_dir


if __name__ == "__main__":
    main()
