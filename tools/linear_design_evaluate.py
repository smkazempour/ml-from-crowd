"""Evaluate the prespecified 228-model linear-design development experiment.

Scoring, economic-key alignment, common samples and calendar-aware HAC reuse
the unchanged protocol evaluator. Scoped adapters replace only the comparison
graph and metadata copied into summaries. No 2023 outcomes are evaluated.
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
from threadpoolctl import threadpool_limits

import protocol_evaluate as kernel


ESTIMATORS = ("ols", "ridge", "lasso", "enet")
TARGETS = ("raw", "dgtw")
FEATURES = ("characteristics", "characteristics_core", "characteristics_textcore")
HISTORIES = ("fixed504", "select")
REFITS = ("retain", "recent", "union")
DESIGN_FIELDS = ("estimator", "feature_set", "representation", "history_policy", "refit_policy")
PLANNED_MODELS = 228
PLANNED_FAMILY_COUNTS = {"refit": 114, "history": 57, "group_penalty": 18,
                         "compression": 30, "incremental_social": 90, "estimator": 90}
EXPECTED_ROWS = 3_034_035
KERNEL_FILES = ("protocol_evaluate.py", "prediction_metrics.py")
PREPARED_FILES = ("manifest.json", "X.npy", "y.npy", "q.npy", "codes.npy",
                  "calendar.npy", "keys.pkl", "evaluation.pkl")
FAMILY_DESCRIPTIONS = {
    "refit": "Recent F minus retained coefficients; union F+126 minus retained; union minus recent, matched in every other design choice",
    "history": "Validation-selected F252/F504/F756 minus fixed F504, matched in every other design choice",
    "group_penalty": "Separate group penalties minus global ridge, at matched inputs, representation, history and refit",
    "compression": "Validation-selected text PCA dimension minus full text, at matched estimator, inputs, history and refit",
    "incremental_social": "C+core minus C; C+full-text+core minus C+core; C+PCA-text+core minus C+core. Group C+core uses global-ridge C as its characteristic-only reference",
    "estimator": "Ridge, lasso, elastic net and group ridge minus OLS, at matched inputs, representation, history and refit",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_models():
    """Independent menu enumeration; do not infer the plan from fitted results."""
    bases = [(e, f, "full") for e, f in itertools.product(ESTIMATORS, FEATURES)]
    bases += [("group_ridge", f, "full") for f in FEATURES[1:]]
    bases += [(e, "characteristics_textcore", "pca") for e in ESTIMATORS + ("group_ridge",)]
    models = []
    for (estimator, feature, representation), history, refit, target in itertools.product(
            bases, HISTORIES, REFITS, TARGETS):
        model = dict(estimator=estimator, feature_set=feature, representation=representation,
                     history_policy=history, refit_policy=refit, target=target,
                     fit_days=504 if history == "fixed504" else 0, validation_days=126,
                     horizon=1, target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target])
        model["model_id"] = "_".join(str(model[k]) for k in DESIGN_FIELDS + ("target",))
        models.append(model)
    return models


def design_key(model):
    return tuple(model[k] for k in DESIGN_FIELDS)


def contrast_specs(models):
    """Declared directional comparisons, independent of fitted scores and choices."""
    if len({m["target"] for m in models}) != 1:
        raise ValueError("Contrasts require one target at a time")
    lookup = {design_key(m): m["model_id"] for m in models}
    if len(lookup) != len(models):
        raise ValueError("Duplicate linear-design procedure")
    specs = []

    def add(model, family, comparison, **changes):
        benchmark = model | changes
        reference = lookup.get(design_key(benchmark))
        if reference is not None:
            specs.append(dict(family=family, comparison=comparison,
                              model_id=model["model_id"], benchmark_id=reference))

    for m in models:
        if m["refit_policy"] == "recent":
            add(m, "refit", "recent_minus_retain", refit_policy="retain")
        if m["refit_policy"] == "union":
            add(m, "refit", "union_minus_retain", refit_policy="retain")
            add(m, "refit", "union_minus_recent", refit_policy="recent")
        if m["history_policy"] == "select":
            add(m, "history", "selected_minus_fixed504", history_policy="fixed504")
        if m["estimator"] == "group_ridge":
            add(m, "group_penalty", "group_minus_global_ridge", estimator="ridge")
        if m["representation"] == "pca":
            add(m, "compression", "pca_minus_full_text", representation="full")
        if m["feature_set"] == "characteristics_core":
            # C has only one group. Its global ridge is the exact C-only
            # benchmark for the corresponding grouped C+core procedure.
            group = m["estimator"] == "group_ridge"
            add(m, "incremental_social", "group_core_minus_ridge_characteristics" if group else "core_given_characteristics",
                feature_set="characteristics", estimator="ridge" if group else m["estimator"])
        if m["feature_set"] == "characteristics_textcore":
            add(m, "incremental_social", "pca_text_given_core" if m["representation"] == "pca" else "full_text_given_core",
                feature_set="characteristics_core", representation="full")
        if m["estimator"] != "ols":
            add(m, "estimator", "versus_ols", estimator="ols")
    return specs


def paired_contrasts(daily, models, lags=(5, 21, 63)):
    import pandas as pd
    specs = contrast_specs(models)
    observed = Counter(s["family"] for s in specs)
    rows = []
    for metric in kernel.METRICS:
        wide = daily.pivot(index="date", columns="model_id", values=metric)
        for spec in specs:
            count = PLANNED_FAMILY_COUNTS[spec["family"]]
            if observed[spec["family"]] > count:
                raise ValueError("Observed comparisons exceed the registered family")
            delta = wide[spec["model_id"]] - wide[spec["benchmark_id"]]
            for lag in lags:
                stats = kernel.calendar_hac_mean(delta, lags=lag)
                p = stats["p"]
                rows.append(spec | dict(metric=metric, family_size=count,
                                        observed_family_size=observed[spec["family"]],
                                        p_bonferroni=min(1., p * count) if np.isfinite(p) else np.nan) | stats)
    return pd.DataFrame(rows)


def validate_matrix(models, allow_pilot=False):
    if not models or len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Nonempty unique model IDs are required")
    found = [design_key(m) + (m["target"],) for m in models]
    planned = {design_key(m) + (m["target"],) for m in planned_models()}
    if len(set(found)) != len(found) or not set(found) <= planned:
        raise ValueError("Registry contains duplicate or unplanned models")
    if not allow_pilot and set(found) != planned:
        raise ValueError("Production evaluation requires all 228 planned models")
    for m in models:
        if (int(m.get("validation_days", -1)) != 126 or m.get("horizon") != 1
                or m.get("target_column") != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[m["target"]]
                or int(m.get("fit_days", -1)) != (504 if m["history_policy"] == "fixed504" else 0)):
            raise ValueError("Models must retain declared history, V126, h1 and target column")
        if not allow_pilot and (int(m.get("months", -1)) != 108 or int(m.get("rows", -1)) != EXPECTED_ROWS):
            raise ValueError("Production models require 108 months and the unchanged 3,034,035 keys")
    for target in TARGETS:
        counts = Counter(s["family"] for s in contrast_specs([m for m in planned_models() if m["target"] == target]))
        if dict(counts) != PLANNED_FAMILY_COUNTS:
            raise AssertionError("Contrast graph disagrees with registered family sizes")


@contextmanager
def design_adapters():
    """Keep the original scientific kernels and module state intact after scoring."""
    original_contrasts = kernel.paired_contrasts
    original_summary = kernel.summarize_daily

    def summarize_with_design(daily, models, lags=5):
        import pandas as pd
        result = original_summary(daily, models, lags=lags)
        extra = ["model_id", "representation", "history_policy", "refit_policy"]
        return result.merge(pd.DataFrame(models)[extra], on="model_id", how="left", validate="many_to_one")

    kernel.paired_contrasts = paired_contrasts
    kernel.summarize_daily = summarize_with_design
    try:
        yield
    finally:
        kernel.paired_contrasts = original_contrasts
        kernel.summarize_daily = original_summary


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
        parser.error("Changing the development evaluation period requires --allow-pilot")
    if args.end >= "2023-01-01":
        parser.error("This development evaluator must not score 2023 or later outcomes")
    started = time.monotonic()
    registry, models = kernel.read_registry(args.registry)
    if registry.get("config", {}).get("run_kind", registry.get("kind")) == "pilot" and not args.allow_pilot:
        raise ValueError("Pilot registry requires --allow-pilot")
    validate_matrix(models, args.allow_pilot)
    prepared = args.prepared.resolve()
    if "prepared" in registry and Path(registry["prepared"]).resolve() != prepared:
        raise ValueError("Registry and requested prepared data differ")
    registry_hash = file_hash(args.registry)
    wrapper_hash = file_hash(__file__)
    prepared_hashes = {name: file_hash(prepared / name) for name in PREPARED_FILES if (prepared / name).exists()}
    kernel_hashes = {name: file_hash(Path(kernel.__file__).with_name(name)) for name in KERNEL_FILES}
    prediction_hashes, prediction_stats = {}, {}
    for m in models:
        path = Path(m["predictions"])
        digest = file_hash(path)
        if not args.allow_pilot and not m.get("sha256"):
            raise ValueError(f"Registry prediction digest missing: {m['model_id']}")
        if m.get("sha256", digest) != digest:
            raise ValueError(f"Prediction digest mismatch: {m['model_id']}")
        prediction_hashes[m["model_id"]] = digest
        prediction_stats[m["model_id"]] = kernel._file_info(path)
    forwarded = ["--registry", str(args.registry), "--prepared", str(prepared),
                 "--out", str(args.out), "--start", args.start, "--end", args.end]
    for flag in ("skip_horizons", "skip_subgroups"):
        if getattr(args, flag):
            forwarded.append("--" + flag.replace("_", "-"))
    with design_adapters(), threadpool_limits(args.threads):
        metadata = kernel.main(forwarded)
    if file_hash(args.registry) != registry_hash or file_hash(prepared / "manifest.json") != prepared_hashes["manifest.json"]:
        raise ValueError("Registry or prepared manifest changed during evaluation")
    if file_hash(__file__) != wrapper_hash:
        raise ValueError("Evaluation adapter changed during scoring")
    for name, digest in kernel_hashes.items():
        if file_hash(Path(kernel.__file__).with_name(name)) != digest:
            raise ValueError(f"Reused evaluation kernel changed: {name}")
    for m in models:
        if kernel._file_info(m["predictions"]) != prediction_stats[m["model_id"]]:
            raise ValueError(f"Prediction file changed during evaluation: {m['model_id']}")
    if not args.allow_pilot:
        for coverage in metadata["coverage"].values():
            if coverage["eligible_stock_days"] != EXPECTED_ROWS or coverage["common_prediction_stock_days"] != EXPECTED_ROWS:
                raise ValueError("Linear-design comparison changed the parent prediction universe")
    daily = args.out.with_name(args.out.name + "_daily.csv")
    compressed = daily.with_suffix(".csv.gz")
    with daily.open("rb") as source, compressed.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as archive:
            shutil.copyfileobj(source, archive, length=8 * 1024 * 1024)
    metadata["output_files"]["_daily"] = kernel._file_info(compressed)
    output_hashes = {Path(info["path"]).name: file_hash(info["path"]) for info in metadata["output_files"].values()}
    metadata.update(schema_version="linear_design_v2_evaluation", planned_models=PLANNED_MODELS,
                    included_models=len(models), pilot=args.allow_pilot, preliminary=args.allow_pilot,
                    planned_family_counts=PLANNED_FAMILY_COUNTS,
                    observed_family_counts={target: dict(Counter(s["family"] for s in contrast_specs(
                        [m for m in models if m["target"] == target]))) for target in TARGETS
                        if any(m["target"] == target for m in models)},
                    multiplicity="Bonferroni within target/family/metric/period over the fixed 228-model design. Missing statistics or pilot subsets do not reduce denominators. HAC21/HAC63 are sensitivities. No correction across the project's already-inspected development history.",
                    contrast_families=FAMILY_DESCRIPTIONS,
                    primary_target="raw", secondary_target="dgtw", primary_metric="rank_ic",
                    research_status="2014-2022 is exploratory development evidence; no globally best test-score procedure is selected. No 2023 outcomes scored. 2023 is not an established untouched holdout: available inputs cover only 178 of 250 expected dates and earlier project records mention predictions through 2023. A separate completeness and prior-use audit is required before any confirmation claim.",
                    coefficient_refits={"retain": "Keep fitting-block coefficients selected using the following 126 validation sessions",
                                        "recent": "After validation selection, refit on the latest F eligible matured sessions",
                                        "union": "After validation selection, refit on the exact original fitting and validation rows; F+126 sessions, excluding the original purge"},
                    window_selection="Select policy chooses F252/F504/F756 and hyperparameters using only preceding validation rank IC. fixed504 retains the original main history.",
                    text_compression="PCA only on 384 embedding coordinates; dimension selected from 16/32/64/128 within validation. The two agreement measures (embed_norm/embed_cos) remain separate inputs in the text penalty group. All transformations are rebuilt on each final refit's permitted inputs.",
                    kernel_reuse="Unmodified protocol_evaluate scoring, alignment, calendar and HAC; scoped comparison and summary-metadata adapters only",
                    reused_kernel_sha256=kernel_hashes, wrapper_sha256=wrapper_hash,
                    registry_sha256=registry_hash, prepared_sha256=prepared_hashes,
                    prediction_sha256=prediction_hashes, output_sha256=output_hashes,
                    daily_artifact="Deterministic gzip (mtime=0); raw CSV retained locally",
                    numerical_threads=args.threads, elapsed_seconds=time.monotonic() - started)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"LINEAR DESIGN EVALUATION COMPLETE: {len(models)}/{PLANNED_MODELS} models; {args.out}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
