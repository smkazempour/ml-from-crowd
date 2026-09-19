"""Run and verify the standard-budget, six-task protocol-v1.1 NN3 pilot.

Use after completing the revised linear comparison. This checks convergence,
capacity, restart integrity and saved-state reproducibility; it does not select
models from pilot test returns. --dry-run prints the exact invocation only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import uuid

from nn_checkpoint import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ["2014-01", "2015-03", "2022-12"]
FEATURES = ["core", "textcore"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError(f"Artifact is outside its required directory: {path}")
    return path


def command(args):
    return [sys.executable, "-B", str(ROOT / "tools" / "protocol_nn.py"),
            "--prepared", str(args.prepared.resolve()), "--fit-days", "504",
            "--targets", "raw", "--features", *FEATURES, "--months", *MONTHS,
            "--architecture", "3", "--seeds", "5", "--max-epochs", "100",
            "--workers", str(args.workers), "--threads", str(args.threads),
            "--out-root", str(args.out_root.resolve())]


def run_logged(argv, environment, log_path):
    started = time.monotonic()
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
             if os.name == "nt" else 0)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(argv, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True, creationflags=flags)
    return time.monotonic() - started


def registry_from_log(log_path, out_root, prepared):
    markers = [line.removeprefix("Registry: ").strip()
               for line in log_path.read_text(encoding="utf-8").splitlines()
               if line.startswith("Registry: ")]
    if not markers:
        raise ValueError(f"No completed registry in {log_path}")
    registry_path = inside(markers[-1], out_root)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["kind"] != "pilot" or Path(registry["prepared"]).resolve() != prepared.resolve():
        raise ValueError("Registry scope or prepared dataset differs from the pilot")
    config = registry["config"]
    if len(config["seeds"]) != 5 or config["max_epochs"] != 100:
        raise ValueError("Pilot did not use the declared standard training budget")
    models = registry["models"]
    if len(models) != 2 or {r["feature_set"] for r in models} != set(FEATURES):
        raise ValueError("The pilot requires exactly core and textcore models")
    checkpoints, predictions, records = {}, {}, []
    for model in models:
        if any(model[key] != value for key, value in {
                "kind": "pilot", "coverage_scope": "partial", "training_budget": "standard",
                "estimator": "nn3", "target": "raw", "fit_days": 504,
                "validation_days": 126, "horizon": 1, "months": 3}.items()):
            raise ValueError("Unexpected model specification in pilot registry")
        run_dir = inside(model["run_dir"], out_root)
        prediction_path = inside(model["predictions"], run_dir)
        prediction_hash = sha256(prediction_path)
        if prediction_hash != model["sha256"]:
            raise ValueError("Prediction hash disagrees with registry")
        predictions[str(prediction_path)] = prediction_hash
        expected = {f"{month}.pkl" for month in MONTHS}
        if {p.name for p in (run_dir / "months").glob("*.pkl")} != expected:
            raise ValueError("Unexpected or missing monthly pilot checkpoints")
        spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
        for month in MONTHS:
            path = inside(run_dir / "months" / f"{month}.pkl", run_dir)
            with path.open("rb") as source:
                payload = pickle.load(source)
            if payload["fingerprint"] != model["run_fingerprint"] or payload["task"]["month"] != month:
                raise ValueError("Checkpoint identity differs from registry")
            result = payload["result"]
            diagnostics = []
            for candidate in result["candidate_diagnostics"]:
                histories = candidate["curves"]
                if len(histories) != len(config["seeds"]):
                    raise ValueError("Missing seed diagnostics")
                diagnostics.append({"penalty": candidate["penalty"],
                    "selected": candidate["penalty"] == result["chosen_penalty"],
                    "ensemble_validation_ic": candidate["ensemble_validation_ic"],
                    "seeds": [{"seed": seed, "best_epoch": best, "epochs_run": len(curve),
                               "hit_epoch_cap": len(curve) == config["max_epochs"],
                               "final_training_data_loss": curve[-1]["training_data_loss"]}
                              for seed, best, curve in zip(config["seeds"], candidate["best_epochs"], histories)]})
            selected = next(d for d in diagnostics if d["selected"])
            stat = path.stat()
            checkpoints[str(path)] = {"sha256": sha256(path), "mtime_ns": stat.st_mtime_ns}
            records.append({"feature_set": model["feature_set"], "month": month,
                "checkpoint": str(path), "chosen_penalty": result["chosen_penalty"],
                "selected_seed_epochs": [s["best_epoch"] for s in selected["seeds"]],
                "candidate_seed_fits_at_epoch_cap": sum(s["hit_epoch_cap"] for d in diagnostics for s in d["seeds"]),
                "selected_seed_fits_at_epoch_cap": sum(s["hit_epoch_cap"] for s in selected["seeds"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "elapsed_scope": "shared block construction and training; excludes initial bundle load",
                "fit_rows": result["n_fit"], "validation_rows": result["n_valid"],
                "prediction_rows": result["n_test"], "feature_count": len(result["feature_names"]),
                "input_array_bytes": 4 * len(result["feature_names"]) * sum(result[k] for k in ("n_fit", "n_valid", "n_test")),
                "capacity_note": "input-array size only, not measured peak process memory",
                "split": result["split"], "candidates": diagnostics,
                "code_sha256": spec["code_sha256"]})
    return registry_path, registry, checkpoints, predictions, records


def verify_restored_states(registry):
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    from protocol_data import load_bundle, make_block
    from protocol_nn import restore_network, predict

    config = registry["config"]
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    bundle = load_bundle(registry["prepared"])
    checks = []
    for feature_set, month in (("core", MONTHS[0]), ("textcore", MONTHS[-1])):
        model_record = next(r for r in registry["models"] if r["feature_set"] == feature_set)
        checkpoint = Path(model_record["run_dir"]) / "months" / f"{month}.pkl"
        payload = pd.read_pickle(checkpoint)
        result = payload["result"]
        with threadpool_limits(limits=config["threads"]):
            block = make_block(bundle, payload["task"], feature_set, "raw")
            if (block["feature_names"] != result["feature_names"]
                    or not np.array_equal(block["mean"], result["mean"])
                    or not np.array_equal(block["scale"], result["scale"])
                    or not np.array_equal(block["test_indices"], result["test_indices"])):
                raise ValueError("Restored feature transformation differs from saved model")
            selected = next(s for s in result["model_states"] if s["penalty"] == result["chosen_penalty"])
            if [s["seed"] for s in selected["seeds"]] != config["seeds"]:
                raise ValueError("Selected seed states differ from declared ensemble")
            x = torch.from_numpy(np.ascontiguousarray(block["X_test"], dtype=np.float32))
            predicted = np.stack([predict(restore_network(s["state_dict"], x.shape[1], config["widths"]), x)
                                  for s in selected["seeds"]]).mean(axis=0)
            final = pd.read_pickle(model_record["predictions"])
            expected_keys = bundle["keys"].iloc[result["test_indices"]]
            saved = final.loc[expected_keys.index]
            if (not saved[["date", "permno", "ticker"]].equals(expected_keys)
                    or not np.array_equal(predicted, result["prediction"])
                    or not np.array_equal(predicted, saved["prediction"].to_numpy())):
                raise ValueError("Restored ensemble does not exactly reproduce saved predictions")
            checks.append({"feature_set": feature_set, "month": month, "seeds_restored": 5,
                           "prediction_rows": len(predicted), "exact_equal": True,
                           "max_abs_difference": 0.0, "threads": config["threads"]})
            del block, x, final
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=ROOT / ".runs" / "protocol_v1_1" / "nn_pilot")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "data" / "protocol_v1_1_nn_pilot.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads) < 1:
        parser.error("Workers and threads must be positive")
    args.out_root, args.report = inside(args.out_root, ROOT), inside(args.report, ROOT)
    overrides = {name: str(args.threads) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    overrides.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    invocation = command(args)
    if args.dry_run:
        print(json.dumps({"command": invocation, "environment": overrides, "report": str(args.report)}, indent=2))
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    first_log, resume_log = [args.out_root / f"{stamp}_{stage}.log" for stage in ("first", "resume")]
    report = {"schema": "protocol_v1_1_nn_pilot", "kind": "pilot", "training_budget": "standard",
              "coverage_scope": "partial", "status": "running", "months": MONTHS,
              "purpose": "execution, convergence, capacity and reproducibility checks; no model ranking from pilot test outcomes",
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": invocation,
              "environment_overrides": overrides, "wrapper_sha256": sha256(__file__),
              "prepared": str(args.prepared.resolve()), "logs": [str(first_log), str(resume_log)]}
    os.environ.update(overrides)
    try:
        atomic_json(report, args.report)
        print(f"Pilot log: {first_log}", flush=True)
        report["first_pass_wall_seconds"] = run_logged(invocation, os.environ.copy(), first_log)
        registry_path, registry, checkpoints, predictions, records = registry_from_log(first_log, args.out_root, args.prepared)
        report.update(registry=str(registry_path), registry_sha256=sha256(registry_path),
                      prepared_fingerprint=json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))["fingerprint"],
                      checkpoint_snapshots=checkpoints, prediction_sha256=predictions, month_diagnostics=records)
        atomic_json(report, args.report)
        print(f"Resume log: {resume_log}", flush=True)
        report["resume_wall_seconds"] = run_logged(invocation, os.environ.copy(), resume_log)
        replay_path, _, replay_checkpoints, replay_predictions, _ = registry_from_log(resume_log, args.out_root, args.prepared)
        replay_text = resume_log.read_text(encoding="utf-8")
        resumed = sum(line.startswith("RESUME nn ") for line in replay_text.splitlines())
        if (replay_path != registry_path or checkpoints != replay_checkpoints
                or predictions != replay_predictions or resumed != 6 or "SAVED nn " in replay_text):
            raise ValueError("Resume changed artifacts or retrained completed months")
        report["resume_verified"] = {"checkpoint_bytes_and_mtimes_unchanged": True,
                                     "prediction_bytes_unchanged": True, "resumed_months": resumed}
        report["restored_state_checks"] = verify_restored_states(registry)
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(report, args.report)
        print(f"Verified pilot report: {args.report}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(report, args.report)
        raise


if __name__ == "__main__":
    main()
