"""Evaluate the fixed stock-characteristic conditioning experiment.

The scoring, economic-key alignment, common samples, HAC estimator and calendar
handling are reused from protocol_evaluate without modifying that source. Only
the declared comparison graph changes. Family sizes always refer to the planned
264-model study, including when a complete subset of fitting windows is scored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import protocol_evaluate as kernel


ESTIMATORS = ("ols", "ridge", "lasso", "enet")
TARGETS = ("raw", "dgtw")
WINDOWS = (504, 252, 756)
FEATURES = ("core", "all", "textcore", "textall", "characteristics",
            "characteristics_sentiment", "characteristics_attention",
            "characteristics_core", "characteristics_all",
            "characteristics_textcore", "characteristics_textall")
SOCIAL_EDGES = (
    ("sentiment_given_characteristics", "characteristics_sentiment", "characteristics"),
    ("attention_given_characteristics", "characteristics_attention", "characteristics"),
    ("core_given_characteristics", "characteristics_core", "characteristics"),
    ("attention_given_characteristics_sentiment", "characteristics_core", "characteristics_sentiment"),
    ("sentiment_given_characteristics_attention", "characteristics_core", "characteristics_attention"),
    ("engineered_given_characteristics_core", "characteristics_all", "characteristics_core"),
    ("text_given_characteristics_core", "characteristics_textcore", "characteristics_core"),
    ("text_given_characteristics_all", "characteristics_textall", "characteristics_all"),
    ("engineered_given_characteristics_textcore", "characteristics_textall", "characteristics_textcore"),
    ("all_given_characteristics", "characteristics_all", "characteristics"),
    ("textcore_given_characteristics", "characteristics_textcore", "characteristics"),
    ("textall_given_characteristics", "characteristics_textall", "characteristics"),
)
PLANNED_FAMILY_COUNTS = {"incremental_social": 144, "conditioning": 48,
                         "estimator": 99, "history": 88}
EXPECTED_ROWS = 3_034_035
KERNEL_FILES = ("protocol_evaluate.py", "prediction_metrics.py")
PREPARED_FILES = ("manifest.json", "X.npy", "y.npy", "q.npy", "codes.npy",
                  "calendar.npy", "keys.pkl", "evaluation.pkl")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_models():
    return [dict(model_id=f"{e}_{f}_{t}_fit{w}_val126", estimator=e,
                 feature_set=f, target=t, fit_days=w, validation_days=126)
            for e, f, t, w in itertools.product(ESTIMATORS, FEATURES, TARGETS, WINDOWS)]


def contrast_specs(models):
    """Return only registered directional edges, independent of observed scores."""
    targets = {m.get("target") for m in models}
    if len(targets) != 1:
        raise ValueError("Contrasts require one target at a time")
    lookup = {(m["estimator"], m["feature_set"], int(m["fit_days"])): m["model_id"]
              for m in models}
    if len(lookup) != len(models):
        raise ValueError("Duplicate estimator/features/history specification")
    specs = []

    def add(family, comparison, model, benchmark):
        if model in lookup and benchmark in lookup:
            specs.append(dict(family=family, comparison=comparison,
                              model_id=lookup[model], benchmark_id=lookup[benchmark]))

    for estimator, fit in itertools.product(ESTIMATORS, WINDOWS):
        for comparison, feature, reference in SOCIAL_EDGES:
            add("incremental_social", comparison, (estimator, feature, fit),
                (estimator, reference, fit))
        for social in ("core", "all", "textcore", "textall"):
            add("conditioning", "characteristics_given_" + social,
                (estimator, "characteristics_" + social, fit), (estimator, social, fit))
    for estimator, feature, fit in itertools.product(ESTIMATORS, FEATURES, WINDOWS):
        if estimator != "ols":
            add("estimator", "versus_ols", (estimator, feature, fit), ("ols", feature, fit))
        if fit != 504:
            add("history", "versus_fit504", (estimator, feature, fit), (estimator, feature, 504))
    return specs


def paired_contrasts(daily, models, lags=(5, 21, 63)):
    """Paired daily HAC with full-study denominators, even for missing statistics."""
    specs = contrast_specs(models)
    observed = Counter(s["family"] for s in specs)
    rows = []
    for metric in kernel.METRICS:
        wide = daily.pivot(index="date", columns="model_id", values=metric)
        for spec in specs:
            delta = wide[spec["model_id"]] - wide[spec["benchmark_id"]]
            count = PLANNED_FAMILY_COUNTS[spec["family"]]
            if observed[spec["family"]] > count:
                raise ValueError("Observed comparison count exceeds planned family")
            for lag in lags:
                stats = kernel.calendar_hac_mean(delta, lags=lag)
                p = stats["p"]
                rows.append(spec | dict(metric=metric, family_size=count,
                                        observed_family_size=observed[spec["family"]],
                                        p_bonferroni=min(1., p * count) if np.isfinite(p) else np.nan)
                            | stats)
    return pd.DataFrame(rows)


def validate_matrix(models, allow_pilot=False):
    """Production evaluation accepts complete windows, not convenient model subsets."""
    if not models or len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Nonempty unique model IDs are required")
    found = [(m["estimator"], m["feature_set"], m["target"], int(m["fit_days"])) for m in models]
    planned = {(m["estimator"], m["feature_set"], m["target"], m["fit_days"]) for m in planned_models()}
    if len(set(found)) != len(found) or not set(found) <= planned:
        raise ValueError("Registry contains duplicate or unplanned models")
    included_windows = [w for w in WINDOWS if any(m["fit_days"] == w for m in models)]
    expected = {x for x in planned if x[-1] in included_windows}
    if not allow_pilot and set(found) != expected:
        raise ValueError("Each evaluated window must contain all 88 models")
    for model in models:
        if (int(model.get("validation_days", -1)) != 126
                or model.get("horizon", 1) != 1
                or model.get("target_column") != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]]):
            raise ValueError("Models must retain V126, h1 and the declared target column")
        if not allow_pilot and (int(model.get("months", -1)) != 108
                                or int(model.get("rows", -1)) != EXPECTED_ROWS):
            raise ValueError("Production models must cover 108 months and the unchanged 3,034,035 keys")
    return included_windows


@contextmanager
def characteristic_contrasts():
    """Scoped adapter; the original module and its source files remain unchanged."""
    original = kernel.paired_contrasts
    kernel.paired_contrasts = paired_contrasts
    try:
        yield
    finally:
        kernel.paired_contrasts = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--skip-horizons", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.allow_pilot and (args.start, args.end) != ("2014-01-01", "2022-12-31"):
        parser.error("Changing the fixed evaluation period requires --allow-pilot")
    started = time.monotonic()
    registry, models = kernel.read_registry(args.registry)
    run_kind = registry.get("config", {}).get("run_kind", registry.get("kind"))
    if run_kind == "pilot" and not args.allow_pilot:
        raise ValueError("Pilot registry requires --allow-pilot")
    windows = validate_matrix(models, args.allow_pilot)
    for target in TARGETS:
        counts = Counter(s["family"] for s in contrast_specs([m for m in planned_models() if m["target"] == target]))
        if dict(counts) != PLANNED_FAMILY_COUNTS:
            raise AssertionError("Declared contrast graph does not match planned family sizes")
    prepared = args.prepared.resolve()
    if "prepared" in registry and Path(registry["prepared"]).resolve() != prepared:
        raise ValueError("Registry and requested prepared data differ")
    registry_hash = file_hash(args.registry)
    prepared_hashes = {name: file_hash(prepared / name) for name in PREPARED_FILES
                       if (prepared / name).exists()}
    kernel_hashes = {name: file_hash(Path(kernel.__file__).with_name(name)) for name in KERNEL_FILES}
    prediction_hashes, prediction_stats = {}, {}
    for model in models:
        path = Path(model["predictions"])
        digest = file_hash(path)
        if not args.allow_pilot and not model.get("sha256"):
            raise ValueError(f"Registry prediction digest missing: {model['model_id']}")
        if model.get("sha256", digest) != digest:
            raise ValueError(f"Prediction digest mismatch: {model['model_id']}")
        prediction_hashes[model["model_id"]] = digest
        prediction_stats[model["model_id"]] = kernel._file_info(path)
    forwarded = ["--registry", str(args.registry), "--prepared", str(prepared),
                 "--out", str(args.out), "--start", args.start, "--end", args.end]
    for flag in ("skip_horizons", "skip_subgroups"):
        if getattr(args, flag):
            forwarded.append("--" + flag.replace("_", "-"))
    with characteristic_contrasts(), threadpool_limits(args.threads):
        metadata = kernel.main(forwarded)
    if file_hash(args.registry) != registry_hash or file_hash(prepared / "manifest.json") != prepared_hashes["manifest.json"]:
        raise ValueError("Registry or prepared manifest changed during evaluation")
    for name, digest in kernel_hashes.items():
        if file_hash(Path(kernel.__file__).with_name(name)) != digest:
            raise ValueError(f"Reused evaluation kernel changed: {name}")
    for model in models:
        mid = model["model_id"]
        if kernel._file_info(model["predictions"]) != prediction_stats[mid]:
            raise ValueError(f"Prediction file changed during evaluation: {mid}")
    if not args.allow_pilot:
        for coverage in metadata["coverage"].values():
            if (coverage["eligible_stock_days"] != EXPECTED_ROWS
                    or coverage["common_prediction_stock_days"] != EXPECTED_ROWS):
                raise ValueError("Characteristic comparison changed the parent prediction universe")
    # Full-study daily output exceeds GitHub's single-file limit uncompressed.
    # Keep the raw local table, but publish a deterministic gzip artifact.
    daily_path = args.out.with_name(args.out.name + "_daily.csv")
    compressed = daily_path.with_suffix(".csv.gz")
    with daily_path.open("rb") as source, compressed.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as archive:
            shutil.copyfileobj(source, archive, length=8 * 1024 * 1024)
    metadata["output_files"]["_daily"] = kernel._file_info(compressed)
    output_hashes = {Path(info["path"]).name: file_hash(info["path"])
                     for info in metadata["output_files"].values()}
    metadata.update(schema_version="characteristic_conditioning_evaluation_v1",
                    planned_models=264, included_models=len(models), included_windows=windows,
                    preliminary=len(models) != 264, pilot=args.allow_pilot,
                    planned_family_counts=PLANNED_FAMILY_COUNTS,
                    observed_family_counts={target: dict(Counter(s["family"] for s in contrast_specs(
                        [m for m in models if m["target"] == target])))
                        for target in TARGETS if any(m["target"] == target for m in models)},
                    multiplicity="Bonferroni over the fixed 264-model plan, separately within target/family/metric/period; incomplete windows or missing statistics never reduce the denominators. HAC21/HAC63 are sensitivities. No correction across interim looks or the project's research history.",
                    contrast_families={"incremental_social": list(SOCIAL_EDGES),
                                       "conditioning": "C+S minus S for core/all/textcore/textall",
                                       "estimator": "Each penalized linear estimator minus OLS at matched inputs and history",
                                       "history": "F252 and F756 minus F504 at matched inputs and estimator"},
                    primary_target="raw", secondary_target="dgtw", primary_window=504,
                    optimization="Fitting histories are fixed sensitivities; no test-based selection, new hyperparameter optimization, or neural fits",
                    kernel_reuse="Unmodified protocol_evaluate score/alignment/calendar/HAC pipeline; only paired_contrasts replaced in a scoped adapter",
                    reused_kernel_sha256=kernel_hashes, wrapper_sha256=file_hash(__file__),
                    registry_sha256=registry_hash, prepared_sha256=prepared_hashes,
                    prediction_sha256=prediction_hashes, output_sha256=output_hashes,
                    daily_artifact="Deterministic gzip (mtime=0); uncompressed daily CSV retained locally",
                    numerical_threads=args.threads, elapsed_seconds=time.monotonic() - started)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"CHARACTERISTIC EVALUATION COMPLETE: {len(models)}/264 models; {args.out}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
