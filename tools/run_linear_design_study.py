"""Run the declared linear-design pilot, training, verification and reporting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from filelock import FileLock
import numpy as np
import pandas as pd

from nn_checkpoint import atomic_json, fingerprint
from run_characteristic_study import run_phase, output_path, update_status
import linear_design_core as core


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "run_linear_design_study.py", "linear_design_core.py", "linear_design_runner.py",
    "linear_design_evaluate.py", "linear_design_report.py", "run_characteristic_study.py",
    "resume_protocol_nn_study.py", "characteristic_data.py", "characteristic_linear.py",
    "characteristic_evaluate.py", "protocol_data.py", "protocol_linear.py",
    "protocol_evaluate.py", "prediction_metrics.py", "nn_checkpoint.py",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_identity():
    return {name: sha256(ROOT/"tools"/name) for name in SOURCES}


def validate_specification(spec):
    """Reject a declaration which disagrees with the executable candidate menu."""
    grid = core.grid_specification()
    groups = spec["group_ridge"]
    if (spec["total_models"] != 228 or spec["test_start"] != "2014-01"
            or spec["test_end"] != "2022-12" or spec["test_months"] != 108
            or spec["validation_days"] != 126 or spec["targets"] != ["raw", "dgtw"]
            or len(core.procedure_specs()) != 114
            or spec["selected_fit_days"] != grid["fit_windows"]
            or spec["representations"]["pca_dimensions"] != grid["pca_dimensions"]
            or groups["base_alphas"] != grid["group_ridge"]["base_alphas"]
            or groups["social_penalty_multipliers"] != grid["group_ridge"]["social_multipliers"]
            or groups["text_penalty_multipliers"] != grid["group_ridge"]["text_multipliers"]):
        raise ValueError("The declared study differs from the executable design")


def validate_prepared(path):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("derivation", {}).get("kind") != "append_market_characteristics":
        raise ValueError("Expected the certified characteristic-derived cache")
    for name, record in manifest["artifact_records"].items():
        artifact = path/name
        if artifact.stat().st_size != record["size"] or sha256(artifact) != record["sha256"]:
            raise ValueError(f"Prepared artifact changed: {artifact}")
    return manifest


def validate_registry(path, prepared, targets=("raw", "dgtw")):
    """Completion checks actual keyed forecasts, not only registry declarations."""
    path, prepared = Path(path), Path(prepared)
    registry = json.loads(path.read_text(encoding="utf-8"))
    expected = {core.model_name(spec, target) for spec in core.procedure_specs() for target in targets}
    models = registry["models"]
    if len(models) != len(expected) or {m["model_id"] for m in models} != expected:
        raise ValueError("Registry does not contain the complete declared model grid")
    if Path(registry["prepared"]).resolve() != prepared.resolve():
        raise ValueError("Registry uses a different prepared cache")
    keys = pd.read_pickle(prepared/"keys.pkl")
    keys = keys.loc[keys.date.between("2014-01-01", "2022-12-31"), ["date", "permno"]]
    keys = keys.sort_values(["date", "permno"])
    months = tuple(sorted(keys.date.dt.to_period("M").astype(str).unique()))
    if (keys.duplicated(["date", "permno"]).any()
            or months != tuple(str(x) for x in pd.period_range("2014-01", "2022-12", freq="M"))):
        raise ValueError("Prepared test keys are duplicate or do not cover all 108 months")
    dates, permnos = keys.date.to_numpy(dtype="datetime64[ns]"), keys.permno.to_numpy()
    for model in models:
        prediction = Path(model["predictions"])
        if not prediction.is_absolute():
            prediction = path.parent/prediction
        if (model["months"] != 108 or model["rows"] != len(keys)
                or model["validation_days"] != 126 or model.get("horizon", 1) != 1
                or model["target_column"] != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]]
                or sha256(prediction) != model["sha256"]):
            raise ValueError(f"Invalid prediction certificate: {model['model_id']}")
        frame = pd.read_pickle(prediction)
        if (len(frame) != len(keys)
                or not np.array_equal(frame.date.to_numpy(dtype="datetime64[ns]"), dates)
                or not np.array_equal(frame.permno.to_numpy(), permnos)
                or not np.isfinite(frame.prediction.to_numpy()).all()):
            raise ValueError(f"Invalid prediction coverage: {model['model_id']}")
    return registry


def evaluation_artifacts(prefix, report, registry):
    prefix, report, registry = map(Path, (prefix, report, registry))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    required = {"summary", "_daily", "_contrasts", "_coverage", "_yearly", "_periods"}
    if not required.issubset(metadata["output_files"]):
        raise ValueError("Missing required evaluation table declarations")
    artifacts = [registry, report, prefix.with_suffix(".json")]
    for key, record in metadata["output_files"].items():
        if record is None:
            if key in required:
                raise ValueError(f"Missing required table: {key}")
            continue
        artifact = Path(record["path"])
        if sha256(artifact) != metadata["output_sha256"][artifact.name]:
            raise ValueError(f"Evaluation output changed: {artifact}")
        artifacts.append(artifact)
    if any(not path.is_file() or not path.stat().st_size for path in artifacts):
        raise ValueError("A required output is absent or empty")
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT/"reports/data/linear_design_v2_experiment.json")
    parser.add_argument("--prepared", type=Path, default=ROOT/".runs/characteristics_v1/prepared/35ef3a5eb0cb5e42")
    parser.add_argument("--baseline-registry", type=Path,
                        default=ROOT/".runs/characteristics_v1/study/d20631f4712e3eb5/linear/full_fc16bf210fe45a57/registry.json")
    parser.add_argument("--out-root", type=Path, default=ROOT/".runs/linear_design_v2/study")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6 or args.threads < 1:
        parser.error("Use one to six workers and a positive thread count")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    validate_specification(spec)
    identity = {
        "experiment": spec, "spec_sha256": sha256(args.spec),
        "prepared": str(args.prepared.resolve()), "prepared_manifest_sha256": sha256(args.prepared/"manifest.json"),
        "baseline_registry": str(args.baseline_registry.resolve()), "baseline_registry_sha256": sha256(args.baseline_registry),
        "code": code_identity(), "python": sys.version, "threads": args.threads,
    }
    signature = fingerprint(identity)
    run = args.out_root.resolve()/signature[:16]
    run.mkdir(parents=True, exist_ok=True)
    status = run/"status.json"
    environment = os.environ.copy()
    environment.update({k: str(args.threads) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]})
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", KMP_DUPLICATE_LIB_OK="TRUE")
    base = [sys.executable, "-B"]
    with FileLock(str(run/"study.lock"), timeout=0):
        previous = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        if previous.get("status") == "complete":
            for path, digest in previous["artifact_sha256"].items():
                if sha256(path) != digest:
                    raise ValueError(f"Completed artifact changed: {path}")
            print(f"COMPLETE {status}", flush=True)
            return run
        attempt = int(previous.get("attempt", 0))+1
        if previous:
            atomic_json(previous, run/f"status_before_attempt_{attempt:03}.json")
        state = dict(status="running", phase="preflight", pid=os.getpid(), child_pid=None,
                     attempt=attempt, started_at=timestamp(), fingerprint=signature,
                     run_dir=str(run), workers=args.workers, threads=args.threads, commands=[])
        atomic_json(identity, run/"study_spec.json")
        snapshot = run/"code"
        snapshot.mkdir(exist_ok=True)
        for name, digest in identity["code"].items():
            source = ROOT/"tools"/name
            if sha256(source) != digest:
                raise ValueError("Source changed while freezing the study")
            shutil.copyfile(source, snapshot/name)
        update_status(status, state)
        print(f"STUDY {run}", flush=True)
        try:
            validate_prepared(args.prepared)
            common = [ROOT/"tools/linear_design_runner.py", "--prepared", args.prepared,
                      "--baseline-registry", args.baseline_registry, "--spec", args.spec,
                      "--out-root", run/"linear", "--threads", args.threads]
            pilot_log = run/f"pilot_attempt_{attempt:03}.log"
            run_phase(base+common+["--months", "2014-01", "2022-12", "--workers", min(2, args.workers)],
                      pilot_log, "implementation_pilot", status, state, environment)
            update_status(status, state, pilot_registry=str(output_path(pilot_log, "registry")))
            fit_log = run/f"linear_attempt_{attempt:03}.log"
            run_phase(base+common+["--workers", args.workers], fit_log,
                      "train_linear", status, state, environment)
            registry_path = output_path(fit_log, "registry")
            update_status(status, state, registry=str(registry_path), phase="verify_predictions")
            validate_registry(registry_path, args.prepared)
            if (code_identity() != identity["code"] or sha256(args.spec) != identity["spec_sha256"]
                    or sha256(args.prepared/"manifest.json") != identity["prepared_manifest_sha256"]
                    or sha256(args.baseline_registry) != identity["baseline_registry_sha256"]):
                raise ValueError("Study sources changed during training")
            prefix, report = ROOT/spec["evaluation_prefix"], ROOT/spec["report"]
            update_status(status, state, evaluation=str(prefix), report=str(report))
            run_phase(base+[ROOT/"tools/linear_design_evaluate.py", "--registry", registry_path,
                      "--prepared", args.prepared, "--out", prefix, "--threads", args.threads],
                      run/f"evaluation_attempt_{attempt:03}.log", "evaluate", status, state, environment)
            run_phase(base+[ROOT/"tools/linear_design_report.py", "--registry", registry_path,
                      "--evaluation", prefix, "--out", report], run/f"report_attempt_{attempt:03}.log",
                      "report", status, state, environment)
            if code_identity() != identity["code"]:
                raise ValueError("Study sources changed during evaluation/reporting")
            artifacts = evaluation_artifacts(prefix, report, registry_path)
            update_status(status, state, status="complete", phase="complete", finished_at=timestamp(),
                          models=228, artifact_sha256={str(path): sha256(path) for path in artifacts})
            print(f"COMPLETE {status}", flush=True)
        except BaseException as error:
            update_status(status, state, status="failed", finished_at=timestamp(), error=f"{type(error).__name__}: {error}")
            raise
    return run


if __name__ == "__main__":
    main()
