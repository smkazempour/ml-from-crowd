"""Isolated, fixed-scope interim evaluation of completed F504/F252 NN3 fits.

Uses the original study's frozen evaluator and planned multiplicity families.
Never trains, changes the live study, or reads incomplete F756 predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (504, 252)
FEATURES = ("core", "all", "textcore", "textall")
TARGETS = ("raw", "dgtw")
ESTIMATORS = ("ols", "ridge", "lasso", "enet", "nn3")
PREPARED_FILES = ("manifest.json", "X.npy", "y.npy", "q.npy", "codes.npy",
                  "calendar.npy", "keys.pkl", "evaluation.pkl", "horizon_repair_audit.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def planned_models():
    return [dict(model_id=f"{est}_{feat}_{target}_fit{fit}_val126", estimator=est,
                 feature_set=feat, target=target, fit_days=fit)
            for est, feat, target, fit in itertools.product(
                ESTIMATORS, FEATURES, TARGETS, (504, 252, 756))]


def planned_family_counts(evaluator):
    counts = {}
    for target in TARGETS:
        counts[target] = dict(Counter(s["family"] for s in evaluator.contrast_specs(
            [m for m in planned_models() if m["target"] == target])))
    return counts


def retain_planned_multiplicity(frame, planned_counts):
    """Keep every original HAC statistic; enlarge only the correction family."""
    result = frame.copy()
    result["observed_family_size"] = result["family_size"]
    result["family_size"] = result["family"].map(planned_counts)
    if result.family_size.isna().any() or (result.family_size < result.observed_family_size).any():
        raise ValueError("Planned multiplicity must cover every evaluated contrast")
    result["p_bonferroni"] = np.minimum(1.0, result["p"] * result["family_size"])
    return result


def validate_included_matrix(models):
    expected = {(e, f, t, w) for e, f, t, w in
                itertools.product(ESTIMATORS, FEATURES, TARGETS, WINDOWS)}
    found = [(m["estimator"], m["feature_set"], m["target"], int(m["fit_days"]))
             for m in models]
    if len(found) != 80 or set(found) != expected or len({m["model_id"] for m in models}) != 80:
        raise ValueError("Preliminary registry must be the exact 80-model F504/F252 matrix")
    for m in models:
        if int(m["months"]) != 108 or m["validation_days"] != 126 or m.get("horizon", 1) != 1:
            raise ValueError("All included fits must cover108months, V126 and h1")
        if m["target_column"] != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[m["target"]]:
            raise ValueError("Target alias mismatch")


def freeze_sources(args, run, study, frozen):
    """Verify training identities and copy certified predictions before scoring."""
    runner = importlib.import_module("run_protocol_nn_study")
    data = importlib.import_module("protocol_data")
    checkpoint = importlib.import_module("nn_checkpoint")
    prepared = args.prepared.resolve()
    linear_path = Path(study["linear_registry"])
    manifest = read_json(prepared / "manifest.json")
    if (Path(study["prepared"]).resolve() != prepared
            or file_hash(prepared / "manifest.json") != study["prepared_manifest_sha256"]
            or file_hash(linear_path) != study["linear_registry_sha256"]):
        raise ValueError("Prepared manifest or full linear registry changed since study launch")
    reference = read_json(args.reference_bundle_manifest)
    if reference["prepared_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Independent prepared-data checksum snapshot is a different dataset")
    data_hashes = {}
    for name in PREPARED_FILES:
        digest = file_hash(prepared / name)
        expected = reference["files"][f"prepared/{name}"]
        if digest != expected["sha256"] or (prepared / name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Prepared data differ from the certified server snapshot: {name}")
        data_hashes[name] = digest
    print("Prepared arrays and evaluation data match independent snapshot checksums", flush=True)
    bundle = data.load_bundle(prepared)
    task_sets = {w: data.make_tasks(bundle, fit_days=w) for w in WINDOWS}
    linear = runner.read_registry(linear_path)
    runner.validate_matrix(linear, prepared, "linear")
    completion = read_json(linear_path.parent / "complete.json")
    configuration = read_json(linear_path.parent / "config.json")
    if (completion != {"fingerprint": linear["fingerprint"], "models": 96, "checkpoints": 648}
            or configuration != linear["config"]
            or configuration["data_manifest"] != manifest
            or checkpoint.fingerprint({k: v for k, v in configuration.items() if k != "fingerprint"})
            != linear["fingerprint"]):
        raise ValueError("Linear completion/configuration identity mismatch")
    for w, tasks in task_sets.items():
        if [t for t in configuration["tasks"] if t["fit_days"] == w] != tasks:
            raise ValueError(f"Linear task boundaries disagree at F{w}")
    models = [dict(m) for m in linear["models"] if m["fit_days"] in WINDOWS]
    sources = {"linear_registry": dict(path=str(linear_path), sha256=file_hash(linear_path)),
               "linear_config": dict(path=str(linear_path.parent / "config.json"),
                                     sha256=file_hash(linear_path.parent / "config.json")),
               "nn": {}}
    expected_config = dict(widths=[128, 64, 32], seeds=[7, 1007, 2007, 3007, 4007],
                           penalties=[1e-5, 1e-4, 1e-3], threads=2, max_epochs=100,
                           patience=5, learning_rate=.001, batch_size=10000, verbose=True,
                           objective="0.5 equal-date MSE + 0.5 lambda sum Linear weights squared; biases/BN excluded",
                           selection="mean daily validation IC of mean seed predictions")
    for fit, feature, target in itertools.product(WINDOWS, FEATURES, TARGETS):
        candidates = sorted((args.study / "nn").glob(f"nn3_{feature}_{target}_f{fit}_full_*/complete.json"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one completed NN3/{feature}/{target}/F{fit}")
        complete_path = candidates[0]
        spec_path = complete_path.with_name("run_spec.json")
        record, spec = read_json(complete_path), read_json(spec_path)
        expected_identity = dict(estimator="nn3", feature_set=feature, target=target,
                                 fit_days=fit, validation_days=126, horizon=1,
                                 coverage_scope="2014-2022", training_budget="standard", kind="full")
        if (any(record.get(k) != v or spec.get(k) != v for k, v in expected_identity.items())
                or spec["prepared"] != manifest or spec["tasks"] != task_sets[fit]
                or spec["config"] != expected_config
                or checkpoint.fingerprint(spec) != record["run_fingerprint"]
                or any(study["code"].get(k) != v for k, v in spec["code_sha256"].items())
                or len(spec["code_sha256"]) != 5):
            raise ValueError(f"NN trained specification mismatch: {spec_path}")
        sources["nn"][record["model_id"]] = dict(
            completion=str(complete_path), completion_sha256=file_hash(complete_path),
            specification=str(spec_path), specification_sha256=file_hash(spec_path),
            training_runtime=spec["versions"], run_fingerprint=record["run_fingerprint"])
        models.append(record)
    validate_included_matrix(models)
    del bundle
    gc.collect()
    # Completed source files can be republished identically by the resuming trainer.
    # Independent physical copies make the evaluation immune to those rewrites.
    for number, model in enumerate(models, 1):
        original = Path(model["predictions"])
        destination = run / "predictions" / (model["model_id"] + ".pkl")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, destination)
        if file_hash(destination) != model["sha256"]:
            raise ValueError(f"Prediction changed or copy is corrupt: {original}")
        model["source_predictions"] = str(original)
        model["predictions"] = str(destination.resolve())
        if number % 8 == 0:
            print(f"Copied and checksum-verified predictions {number}/80", flush=True)
    prediction_hashes = runner.verify_prediction_files(models, prepared)
    registry = dict(schema_version="protocol_v1_1_nn_preliminary", kind="preliminary",
                    prepared=str(prepared), models=models,
                    coverage_scope="2014-2022", included_windows=list(WINDOWS),
                    exclusion_rule="All F756 excluded before scoring because NN grid is incomplete; no performance selection",
                    full_planned_models=planned_models(), config={"linear": configuration, "nn": expected_config})
    registry["fingerprint"] = checkpoint.fingerprint(registry)
    registry_path = run / "registry.json"
    write_json(registry, registry_path)
    return registry_path, models, dict(data_sha256=data_hashes, prediction_sha256=prediction_hashes,
                                     training_sources=sources, prepared_fingerprint=manifest["fingerprint"],
                                     reference_bundle_manifest=str(args.reference_bundle_manifest.resolve()),
                                     reference_bundle_manifest_sha256=file_hash(args.reference_bundle_manifest))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=ROOT / ".runs/protocol_v1_1/nn_study/48b73c386ddd0dec")
    ap.add_argument("--prepared", type=Path, default=ROOT / ".runs/protocol_v1_1/prepared/8f3f4eb44b399771")
    ap.add_argument("--reference-bundle-manifest", type=Path,
                    default=ROOT / ".runs/server_nn_bundle_20260913/bundle_manifest.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/data/protocol_v1_1_nn_preliminary")
    ap.add_argument("--run-root", type=Path, default=ROOT / ".runs/protocol_v1_1/nn_preliminary")
    ap.add_argument("--threads", type=int, choices=[1, 2], default=1)
    ap.add_argument("--minimum-free-gib", type=float, default=14)
    args = ap.parse_args(argv)
    if args.out.with_suffix(".json").exists():
        raise ValueError("Preliminary output already exists; choose a new output prefix")
    if args.minimum_free_gib < 10:
        ap.error("Retain at least10 GiB of free-memory preflight headroom")
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    study = read_json(args.study / "study_spec.json")
    frozen = args.study / "code"
    for name, digest in study["code"].items():
        if file_hash(frozen / name) != digest:
            raise ValueError(f"Frozen study code hash mismatch: {name}")
    sys.path.insert(0, str(frozen.resolve()))
    evaluator = importlib.import_module("protocol_evaluate")
    counts = planned_family_counts(evaluator)
    if any(c != {"feature": 60, "estimator": 84, "history": 40} for c in counts.values()):
        raise ValueError("Unexpected full-study contrast families")
    run = args.run_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    shutil.copyfile(__file__, run / Path(__file__).name)
    started = time.monotonic()
    with threadpool_limits(args.threads):
        registry_path, models, audit = freeze_sources(args, run, study, frozen)
        gc.collect()
        free_gib = psutil.virtual_memory().available / 2**30
        if free_gib < args.minimum_free_gib:
            raise MemoryError(f"Only {free_gib:.1f}GiB available; needs{args.minimum_free_gib}GiB")
        original = evaluator.paired_contrasts

        def fixed_contrasts(daily, target_models, lags=(5, 21, 63)):
            targets = {m["target"] for m in target_models}
            if len(targets) != 1:
                raise ValueError("Paired contrasts require one target")
            return retain_planned_multiplicity(original(daily, target_models, lags), counts[targets.pop()])

        evaluator.paired_contrasts = fixed_contrasts
        try:
            metadata = evaluator.main(["--registry", str(registry_path), "--prepared", str(args.prepared),
                                       "--out", str(args.out), "--skip-horizons", "--skip-subgroups"])
            daily = pd.read_csv(args.out.with_name(args.out.name + "_daily.csv"), parse_dates=["date"])
            annual = []
            for target in TARGETS:
                target_models = [m for m in models if m["target"] == target]
                for year in range(2014, 2023):
                    subset = daily[(daily.target == target) & (daily.date.dt.year == year)]
                    annual.append(fixed_contrasts(subset, target_models).assign(target=target, year=year))
            annual_path = args.out.with_name(args.out.name + "_yearly_contrasts.csv")
            pd.concat(annual, ignore_index=True).to_csv(annual_path, index=False)
        finally:
            evaluator.paired_contrasts = original
    outputs = [p for p in args.out.parent.glob(args.out.name + "*.csv") if p.is_file()]
    # Main tables have been written using the frozen evaluator. Attach the explicit
    # interim audit without changing its reported raw statistics or observations.
    metadata.update(preliminary=True, included_models=80, included_nn_models=16,
                    included_linear_models=64, included_windows=list(WINDOWS), primary_window=504,
                    sensitivity_window=252, planned_models=120, excluded_models=40,
                    excluded_scope="All F756:8 NN3 and32 matched linear models; excluded before scoring",
                    pending_diagnostics="Subgroups and longer horizons not yet computed in this interim evaluation",
                    multiplicity="Bonferroni denominators retained from all120 planned models within target/family/metric/period; no interim-look correction",
                    planned_family_counts=counts, observed_family_counts={"feature": 40, "estimator": 56, "history": 20},
                    absent_contrasts_per_target_metric_period={"feature": 20, "estimator": 28, "history": 20},
                    annual_contrasts="Descriptive stability diagnostics; same planned family sizes within each year, not corrected across years or interim looks",
                    interpretation="Interim disclosure, not model or architecture selection; keep remaining experiment fixed",
                    frozen_code_sha256=study["code"], wrapper_sha256=file_hash(__file__),
                    source_audit=str((run / "source_audit.json").resolve()), numerical_threads=args.threads,
                    priority="below_normal" if os.name == "nt" else "nice10",
                    free_gib_before_scoring=free_gib, elapsed_seconds=time.monotonic()-started,
                    output_sha256={p.name: file_hash(p) for p in outputs})
    audit.update(snapshot_registry=str(registry_path.resolve()), registry_sha256=file_hash(registry_path),
                 frozen_code_sha256=study["code"], wrapper_sha256=file_hash(__file__),
                 evaluation_metadata=str(args.out.with_suffix(".json").resolve()))
    write_json(audit, run / "source_audit.json")
    write_json(metadata, args.out.with_suffix(".json"))
    print(f"PRELIMINARY COMPLETE: {args.out}; audit:{run / 'source_audit.json'}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
