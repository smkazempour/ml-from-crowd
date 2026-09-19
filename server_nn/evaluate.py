"""Evaluate complete portable NN studies with predeclared architecture contrasts.

Run as ``python -m server_nn.evaluate --bundle BUNDLE --output RESULTS --phase depth``.
The immutable vendor supplies all scoring rules. This wrapper validates complete
coverage, adds architecture contrasts, and optionally forms the validation-only
adaptive-depth model. It never trains or edits the running workstation study.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .common import (data_identity, ensure_vendor, hash_file, phase_models,
                     read_json, runtime_info, training_config, verify_bundle, write_json)


FEATURES = ("core", "all", "textcore", "textall")
LINEAR = ("ols", "ridge", "lasso", "enet")
DEPTH = ("nn1", "nn2", "nn3", "nn4")
WIDTH = ("nn3_narrow", "nn3", "nn3_wide")
MONTHS = tuple(str(x) for x in pd.period_range("2014-01", "2022-12", freq="M"))
POLICY = {
    "selection": "Maximum monthly ensemble validation IC across NN1/NN2/NN3/NN4; each architecture has already selected its penalty using validation",
    "ties": "Exact IC ties: fewer trainable parameters, then declared architecture order NN1, NN2, NN3, NN4",
    "test_use": "Test predictions are copied only after selection; test outcomes and test IC never enter selection",
    "architecture_family": "Fixed NN1/NN2/NN4 versus fixed NN3 only; 12 comparisons per target/metric/period",
    "adaptive_family": "Adaptive depth versus fixed NN3; 4 comparisons per target/metric/period",
}


def resolve_path(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base)/path).resolve()


def relative_path(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve()).replace("\\", "/")


def load_records(path, output=None):
    path = Path(path).resolve()
    doc = read_json(path)
    if doc.get("kind", "full") != "full":
        raise ValueError("Evaluation requires a complete full-budget registry, not a pilot")
    base = Path(output).resolve() if doc.get("path_base") == "output_root" else path.parent
    if doc.get("path_base") not in (None, "registry_directory", "output_root"):
        raise ValueError("Unknown registry path_base")
    records = copy.deepcopy(doc.get("models", []))
    for record in records:
        for field in ("predictions", "run_dir", "diagnostics"):
            if field in record:
                record[field] = str(resolve_path(record[field], base))
    return doc, records


def identity(record):
    return (record["estimator"], record["feature_set"], record["target"], int(record["fit_days"]))


def validate_matrix(records, phase, *, neural=True):
    if neural:
        specs = list(phase_models(phase))
        expected = {identity(x): x for x in specs}
    else:
        fits = (504, 252, 756) if phase == "nn3" else (504,)
        expected = {(e, f, t, n): None for e in LINEAR for f in FEATURES
                    for t in ("raw", "dgtw") for n in fits}
    if len(records) != len(expected) or {identity(x) for x in records} != set(expected):
        raise ValueError(f"Incomplete or unexpected {phase} {'NN' if neural else 'linear'} specification matrix")
    if len({x["model_id"] for x in records}) != len(records):
        raise ValueError("Duplicate model IDs")
    for record in records:
        if int(record.get("months", 0)) != 108 or int(record.get("validation_days", 0)) != 126:
            raise ValueError("Every model must have 108 months and 126 validation sessions")
        if int(record.get("horizon", 1)) != 1:
            raise ValueError("Only registered one-session training targets are supported")
        target = {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[record["target"]]
        if record.get("target_column") != target:
            raise ValueError("Incorrect target column")
        if neural:
            if (record.get("kind") != "full" or record.get("coverage_scope") != "2014-2022"
                    or record.get("training_budget") != "standard"):
                raise ValueError("NN evaluation requires full coverage and the standard training budget")
            if list(record.get("widths", [])) != list(expected[identity(record)]["widths"]):
                raise ValueError("Architecture widths differ from the registered phase")
    return records


def expected_keys(prepared):
    keys = pd.read_pickle(Path(prepared)/"keys.pkl").copy()
    keys["date"] = pd.to_datetime(keys["date"])
    if keys[["date", "permno"]].isna().any().any() or keys.duplicated(["date", "permno"]).any():
        raise ValueError("Prepared economic keys must be unique and nonmissing")
    selected = keys.date.between("2014-01-01", "2022-12-31").to_numpy()
    positions = np.flatnonzero(selected)
    expected = keys.iloc[positions]
    if tuple(sorted(expected.date.dt.to_period("M").astype(str).unique())) != MONTHS:
        raise ValueError("Prepared data lack the full 108-month study period")
    return keys, positions, expected


def validate_prediction(record, expected, *, neural=False):
    path = Path(record["predictions"])
    if hash_file(path) != record.get("sha256"):
        raise ValueError(f"Prediction checksum mismatch: {record['model_id']}")
    frame = pd.read_pickle(path)
    required = ["date", "permno", "prediction"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Prediction file lacks economic keys or prediction")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if (len(frame) != len(expected) or int(record.get("rows", -1)) != len(expected)
            or frame[["date", "permno"]].isna().any().any()
            or frame.duplicated(["date", "permno"]).any()
            or not np.isfinite(frame.prediction.to_numpy(dtype=float)).all()):
        raise ValueError("Prediction file does not contain complete unique finite predictions")
    columns = ["date", "permno"]
    a = frame[columns].sort_values(columns).reset_index(drop=True)
    b = expected[columns].sort_values(columns).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError("Prediction economic keys differ from the prepared study universe")
    if neural:
        complete = read_json(Path(record["run_dir"])/"complete.json")
        for field in ("model_id", "run_fingerprint", "sha256", "months", "rows"):
            if complete.get(field) != record.get(field):
                raise ValueError(f"NN completion marker differs from registry: {field}")
    return frame


def validate_nn_spec(record, expected_data):
    """Check the saved budget and model identity, not only registry assertions."""
    spec = read_json(Path(record["run_dir"])/"run_spec.json")
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if signature != record["run_fingerprint"]:
        raise ValueError("NN run specification fingerprint mismatch")
    if identity(spec["model"]) != identity(record) or list(spec["model"]["widths"]) != list(record["widths"]):
        raise ValueError("NN specification model differs from registry")
    if spec.get("data_identity") != expected_data:
        raise ValueError("NN specification uses different prepared data")
    config = spec["config"]
    if config != training_config(record["widths"], config["threads"]):
        raise ValueError("NN specification does not use the registered standard training budget")
    if tuple(task["month"] for task in spec["tasks"]) != MONTHS:
        raise ValueError("NN specification does not contain the full monthly task sequence")


def parameter_count(inputs, widths):
    """Trainable linear weights/biases and BatchNorm affine parameters."""
    total, previous = 0, int(inputs)
    for width in widths:
        total += previous*int(width) + 3*int(width)
        previous = int(width)
    return total + previous + 1


def choose_architecture(candidates):
    """Pure selection rule: no prediction or test-outcome values are accessed."""
    if len(candidates) != 4 or {x["estimator"] for x in candidates} != set(DEPTH):
        raise ValueError("Adaptive depth needs all four architectures for every month")
    if any(not np.isfinite(float(x["validation_ic"])) for x in candidates):
        raise ValueError("Adaptive selection requires finite validation IC for every architecture")
    return min(candidates, key=lambda x: (-float(x["validation_ic"]),
                                         int(x["parameter_count"]), DEPTH.index(x["estimator"])))


def extended_contrasts(models, phase, base_function, adaptive=False):
    specs = list(base_function(models))
    lookup = {identity(x): x["model_id"] for x in models}
    alternatives = ("nn1", "nn2", "nn4") if phase == "depth" else (
        ("nn3_narrow", "nn3_wide") if phase == "width" else ())
    for model in models:
        estimator, feature, target, fit = identity(model)
        family = "architecture" if estimator in alternatives else (
            "adaptive" if adaptive and estimator == "nn_adaptive" else None)
        if family:
            benchmark = lookup.get(("nn3", feature, target, fit))
            if benchmark is None:
                raise ValueError("Architecture comparison lacks its fixed NN3 benchmark")
            specs.append(dict(family=family, comparison="versus_nn3", model_id=model["model_id"],
                              benchmark_id=benchmark))
    return specs


def build_adaptive(records, prepared, output):
    """Combine verified checkpoint predictions chosen using validation only."""
    ensure_vendor()
    from nn_checkpoint import atomic_pickle, fingerprint
    from protocol_nn import validate_checkpoint
    keys, _, expected = expected_keys(prepared)
    key_months = keys.date.dt.to_period("M").astype(str).to_numpy()
    rows_by_month = {month: np.flatnonzero(key_months == month) for month in MONTHS}
    manifest = read_json(Path(prepared)/"manifest.json")
    records_by_id = {identity(x): x for x in records}
    generated, ledger = [], []
    for target in ("raw", "dgtw"):
        for feature in FEATURES:
            candidates_by_month = {month: [] for month in MONTHS}
            checkpoint_predictions = {}
            for estimator in DEPTH:
                record = records_by_id[(estimator, feature, target, 504)]
                run_dir = Path(record["run_dir"])
                spec = read_json(run_dir/"run_spec.json")
                tasks = {task["month"]: task for task in spec["tasks"]}
                if set(tasks) != set(MONTHS):
                    raise ValueError("Adaptive candidate specification lacks all months")
                frame = pd.read_pickle(record["predictions"]).copy()
                frame["date"] = pd.to_datetime(frame["date"])
                lookup = frame.set_index(["date", "permno"])["prediction"]
                count = parameter_count(len(manifest["feature_sets"][feature]), record["widths"])
                for month in MONTHS:
                    path = run_dir/"months"/f"{month}.pkl"
                    payload = pd.read_pickle(path)
                    if (payload.get("fingerprint") != record["run_fingerprint"]
                            or payload.get("task") != tasks[month]):
                        raise ValueError("Adaptive checkpoint identity or task mismatch")
                    rows = rows_by_month[month]
                    result = payload["result"]
                    validate_checkpoint(result, tasks[month], rows)
                    piece = keys.iloc[rows].copy()
                    piece["prediction"] = np.asarray(result["prediction"])
                    registered = lookup.reindex(pd.MultiIndex.from_frame(piece[["date", "permno"]]))
                    if not np.array_equal(registered.to_numpy(), piece.prediction.to_numpy()):
                        raise ValueError("Adaptive checkpoint predictions differ from registered predictions")
                    candidate = dict(target=target, feature_set=feature, month=month,
                        estimator=estimator, model_id=record["model_id"],
                        validation_ic=float(result["validation_ic"]), parameter_count=count,
                        chosen_penalty=float(result["chosen_penalty"]),
                        checkpoint=relative_path(path, output), checkpoint_sha256=hash_file(path),
                        run_fingerprint=record["run_fingerprint"])
                    candidates_by_month[month].append(candidate)
                    checkpoint_predictions[(month, estimator)] = piece
            pieces, chosen_records = [], []
            for month in MONTHS:
                selected = choose_architecture(candidates_by_month[month])
                pieces.append(checkpoint_predictions[(month, selected["estimator"])])
                chosen_records.append(selected)
                ledger.extend(item | {"selected": item["estimator"] == selected["estimator"]}
                              for item in candidates_by_month[month])
            frame = pd.concat(pieces).sort_values(["date", "permno"])
            model_id = f"nn_adaptive_{feature}_{target}_fit504_val126"
            signature = fingerprint({"policy": POLICY, "selected": chosen_records,
                                     "wrapper_sha256": hash_file(Path(__file__))})
            run_dir = Path(output)/"adaptive"/"depth"/model_id/signature[:16]
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir/"predictions.pkl"
            atomic_pickle(frame, path)
            record = dict(model_id=model_id, estimator="nn_adaptive", feature_set=feature,
                target=target, target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target],
                fit_days=504, validation_days=126, horizon=1, months=108, rows=len(frame),
                kind="full", coverage_scope="2014-2022", training_budget="standard",
                predictions=str(path.resolve()), run_dir=str(run_dir.resolve()),
                run_fingerprint=signature, sha256=hash_file(path), selection_policy=POLICY)
            write_json({"policy": POLICY, "months": chosen_records}, run_dir/"selection.json")
            write_json(record, run_dir/"complete.json")
            validate_prediction(record, expected)
            generated.append(record)
    ledger_path = Path(output)/"adaptive"/"depth"/"selection_ledger.csv"
    pd.DataFrame(ledger).to_csv(ledger_path, index=False)
    return generated, ledger_path


def portable_records(records, base):
    records = copy.deepcopy(records)
    for record in records:
        for field in ("predictions", "run_dir", "diagnostics"):
            if field in record:
                record[field] = relative_path(record[field], base)
    return records


def architecture_report(records, prefix, destination, phase, adaptive, provenance):
    ensure_vendor()
    from protocol_report import table, number, pvalue, link
    manifest = read_json(Path(provenance["prepared"])/"manifest.json")
    dimensions = {f: len(manifest["feature_sets"][f]) for f in FEATURES}
    architectures = []
    for record in records:
        if "widths" in record:
            architectures.append({"Architecture": record["estimator"], "Inputs": record["feature_set"],
                "Hidden widths": " → ".join(map(str, record["widths"])),
                "Trainable parameters": parameter_count(dimensions[record["feature_set"]], record["widths"])})
    sections = [f"# Server NN study: {phase}",
        "This report evaluates the complete registered phase over January 2014–December 2022. "
        "The architecture menu was fixed before inspecting these results. Validation selects penalties and seed checkpoints; "
        "the test period compares the registered models and is not an untouched historical holdout.",
        table(pd.DataFrame(architectures).drop_duplicates()),
        "Depth comparisons hold the first hidden width at 128; width comparisons hold depth at three. "
        "Parameter counts therefore vary, and these comparisons do not isolate depth from capacity."]
    contrasts = pd.read_csv(prefix.with_name(prefix.name+"_contrasts.csv"))
    lookup = {x["model_id"]: x for x in records}
    families = ["architecture"] + (["adaptive"] if adaptive else [])
    for family in families:
        if phase == "nn3" and family == "architecture":
            continue
        sections.append("## " + ("Fixed architectures versus NN3" if family == "architecture" else "Adaptive depth versus fixed NN3"))
        rows = contrasts[(contrasts.family == family) & (contrasts.period == "full")
                         & (contrasts.hac_lags == 5)]
        table_rows = []
        for row in rows.to_dict("records"):
            model = lookup[row["model_id"]]
            table_rows.append({"Target": row["target"], "Architecture": model["estimator"],
                "Inputs": model["feature_set"], "Metric": row["metric"], "Difference": number(row["mean"]),
                "95% CI": f"[{number(row['ci_low'])}, {number(row['ci_high'])}]",
                "Adjusted p": pvalue(row["p_bonferroni"]), "Family size": row["family_size"]})
        sections.append(table(pd.DataFrame(table_rows)))
    sections.append("Differences are model minus fixed NN3. Intervals are pointwise 95% Bartlett-HAC intervals "
        "with five lags; adjusted p-values use the entire declared family separately within target, metric and period. "
        "The full contrast artifact includes HAC21/HAC63 and subperiod sensitivities. "
        "IC is primary; equal-weighted and capitalization-weighted spreads in basis points are secondary gross diagnostics.")
    if adaptive:
        sections += ["## Adaptive selection", POLICY["selection"] + ". " + POLICY["ties"] + ". " + POLICY["test_use"] + ".",
            "The architecture family contains only the fixed-depth comparisons (12 per target/metric/period). "
            "Adaptive versus NN3 has a separate four-comparison family. Adaptive versus linear models remains "
            "inside the estimator family, and its feature additions remain inside the feature family.",
            "Selection ledger: " + link(provenance["adaptive_ledger"], destination) + "."]
    sections += ["## Full results and provenance",
        "Detailed matched linear/NN tables, yearly stability, portfolios and diagnostics: "
        + link(Path(destination).with_name("detailed_results.md"), destination) + ".",
        "Evaluation metadata: " + link(prefix.with_suffix(".json"), destination)
        + ". Run provenance: " + link(Path(destination).with_name("provenance.json"), destination) + ".",
        "Same-close execution is not certified: the source assigns messages to a nominal 16:00 close, "
        "including early-close sessions. No costs, turnover or factor alpha are estimated. Raw and DGTW "
        "outcome samples can differ. The cumulative-horizon diagnostics are not separately trained models."]
    Path(destination).write_text("\n\n".join(sections)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("nn3", "depth", "width"), required=True)
    parser.add_argument("--adaptive", action="store_true", help="Predeclared validation-selected depth model; depth phase only")
    args = parser.parse_args(argv)
    if args.adaptive and args.phase != "depth":
        parser.error("--adaptive is supported only for --phase depth")
    from filelock import FileLock
    lock_path = args.output.resolve()/"locks"/f"evaluation_{args.phase}_{int(args.adaptive)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path, timeout=0):
        return evaluate_study(args)


def evaluate_study(args):
    bundle, output = args.bundle.resolve(), args.output.resolve()
    prepared = bundle/"prepared"
    neural_registry, linear_registry = output/"registries"/f"{args.phase}.json", bundle/"linear"/"registry.json"
    verify_bundle(bundle)
    neural_doc, neural = load_records(neural_registry, output)
    if neural_doc.get("data_identity") != data_identity(bundle):
        raise ValueError("NN registry was trained from a different prepared data bundle")
    _, linear = load_records(linear_registry)
    if args.phase != "nn3":
        linear = [x for x in linear if int(x["fit_days"]) == 504]
    validate_matrix(neural, args.phase)
    validate_matrix(linear, args.phase, neural=False)
    _, _, expected = expected_keys(prepared)
    for record in linear + neural:
        validate_prediction(record, expected, neural=record["estimator"].startswith("nn"))
        if record["estimator"].startswith("nn"):
            validate_nn_spec(record, data_identity(bundle))
        print(f"VERIFIED {record['model_id']}", flush=True)
    records, ledger = linear + neural, None
    if args.adaptive:
        adaptive_records, ledger = build_adaptive(neural, prepared, output)
        records += adaptive_records
    report_dir = output/"reports"/(args.phase+"_adaptive" if args.adaptive else args.phase)
    report_dir.mkdir(parents=True, exist_ok=True)
    # A failed refresh must not leave a completion marker for overwritten files.
    (report_dir/"complete.json").unlink(missing_ok=True)
    registry_path = report_dir/"registry.json"
    write_json(dict(schema_version="server_nn_evaluation", kind="full", phase=args.phase,
        adaptive=args.adaptive, models=portable_records(records, report_dir),
        prepared=relative_path(prepared, report_dir), path_base="registry_directory"), registry_path)
    ensure_vendor()
    import protocol_evaluate as evaluator
    import protocol_report as reporter
    original = evaluator.contrast_specs
    evaluator.contrast_specs = lambda models: extended_contrasts(models, args.phase, original, args.adaptive)
    prefix = report_dir/"evaluation"
    try:
        metadata = evaluator.main(["--registry", str(registry_path), "--prepared", str(prepared), "--out", str(prefix)])
    finally:
        evaluator.contrast_specs = original
    wrapper_hash = hash_file(Path(__file__))
    package_dir = Path(__file__).resolve().parent
    code_hashes = {"server_nn/evaluate.py": wrapper_hash,
                   "server_nn/common.py": hash_file(package_dir/"common.py")}
    code_hashes.update({"server_nn/vendor/"+p.name: hash_file(p)
                       for p in (package_dir/"vendor").glob("*.py")})
    metadata["code_sha256"].update(code_hashes)
    metadata["server_phase"] = args.phase
    metadata["adaptive_depth"] = POLICY if args.adaptive else None
    family_counts = {}
    for target in ("raw", "dgtw"):
        specs = extended_contrasts([r for r in records if r["target"] == target], args.phase, original, args.adaptive)
        family_counts[target] = dict(Counter(s["family"] for s in specs))
    metadata["registered_family_sizes"] = family_counts
    metadata["contrast_families"] += " Architecture: fixed alternatives versus NN3 at identical inputs/history. Adaptive: validation-selected depth versus NN3 separately, if enabled."
    write_json(metadata, prefix.with_suffix(".json"))
    provenance = dict(phase=args.phase, adaptive=args.adaptive, prepared=str(prepared),
        prepared_manifest_sha256=hash_file(prepared/"manifest.json"),
        neural_registry_sha256=hash_file(neural_registry), linear_registry_sha256=hash_file(linear_registry),
        wrapper_sha256=wrapper_hash, code_sha256=code_hashes, runtime=runtime_info(),
        registered_family_sizes=family_counts,
        adaptive_ledger=str(ledger) if ledger else None)
    reporter.build_report(registry_path, prefix, report_dir/"detailed_results.md",
                          title=f"Server NN {args.phase}: complete matched results")
    architecture_report(records, prefix, report_dir/"results.md", args.phase, args.adaptive, provenance)
    provenance["prepared"] = relative_path(prepared, report_dir)
    if ledger:
        provenance["adaptive_ledger"] = relative_path(ledger, report_dir)
    write_json(provenance, report_dir/"provenance.json")
    source_dir = report_dir/"code"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source_dir/"evaluate.py")
    shutil.copy2(package_dir/"common.py", source_dir/"common.py")
    for source in (package_dir/"vendor").glob("*.py"):
        vendor_destination = source_dir/"vendor"/source.name
        vendor_destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, vendor_destination)
    write_json(dict(phase=args.phase, adaptive=args.adaptive,
        models=len(records), registry_sha256=hash_file(registry_path),
        evaluation_metadata_sha256=hash_file(prefix.with_suffix(".json")),
        results_sha256=hash_file(report_dir/"results.md"), wrapper_sha256=wrapper_hash), report_dir/"complete.json")
    print(f"COMPLETE evaluation: {report_dir/'results.md'}", flush=True)
    return report_dir


if __name__ == "__main__":
    main()
