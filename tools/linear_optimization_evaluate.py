"""Score the registered linear-optimization development experiment in bounded batches.

Economic-key alignment, portfolio tie handling and calendar-aware HAC are the
unchanged protocol kernels. Only the planned comparison graph and orchestration
are new. No matrix of every model's stock-day predictions is retained in RAM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import protocol_evaluate as kernel


TARGETS = ("raw", "dgtw")
FEATURES = ("characteristics", "characteristics_core", "characteristics_textcore")
ESTIMATORS = ("ols", "ridge", "lasso", "enet", "group_ridge")
DESIGN_FIELDS = ("estimator", "feature_set", "representation", "transform", "penalty_grid", "pca_grid",
                 "history_policy", "refit_policy")
EXPECTED_ROWS = 3_034_035
PLANNED_MODELS = 172
FAMILY_DESCRIPTIONS = {
    "penalty_expansion": "Expanded minus original penalty menu, with every other choice matched",
    "transformation": "Daily characteristic ranks minus historical standardization, with every other choice matched",
    "pca_expansion": "PCA menu including 4 and 8 components minus original PCA menu, with every other choice matched",
    "compression": "PCA text minus full text, matching estimator, transformation and penalty menu",
    "incremental_social": "Sentiment/attention given characteristics; full or PCA text given characteristics and sentiment/attention",
    "group_penalty": "Separate group penalties minus global ridge, with all input and search-menu choices matched",
    "estimator": "Each penalized procedure minus OLS; OLS has no penalty-menu axis",
}
PREPARED_FILES = ("manifest.json", "X.npy", "y.npy", "q.npy", "codes.npy", "calendar.npy", "keys.pkl", "evaluation.pkl")
KERNEL_FILES = ("protocol_evaluate.py", "prediction_metrics.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def design_key(model):
    return tuple(model[key] for key in DESIGN_FIELDS)


def planned_models():
    """Independent complete menu, used to reject accidental omissions or duplicates."""
    models = []
    for feature in FEATURES:
        representations = ("full", "pca") if feature == FEATURES[-1] else ("full",)
        estimators = ESTIMATORS if feature != FEATURES[0] else ESTIMATORS[:-1]
        for estimator, representation, transform, target in itertools.product(
                estimators, representations, ("standard", "daily_rank"), TARGETS):
            for penalty, pca in itertools.product(
                    ("original",) if estimator == "ols" else ("original", "expanded"),
                    ("original", "expanded") if representation == "pca" else ("none",)):
                model = dict(estimator=estimator, feature_set=feature, representation=representation,
                             transform=transform, penalty_grid=penalty, pca_grid=pca,
                             history_policy="fixed504", refit_policy="union", fit_days=504,
                             validation_days=126, horizon=1, target=target,
                             target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target])
                model["model_id"] = (f"ld3_{estimator}_{feature}_{representation}_{target}_"
                                     f"{transform}_{penalty}_{pca}_fixed504_union")
                models.append(model)
    return models


def contrast_specs(models):
    """Declare directional comparisons for one target, without inspecting scores."""
    if len({m["target"] for m in models}) > 1:
        raise ValueError("Contrast specification requires one target")
    lookup = {design_key(m): m["model_id"] for m in models}
    if len(lookup) != len(models):
        raise ValueError("Duplicate optimization procedure")
    specs = []

    def add(model, family, comparison, **changes):
        reference = lookup.get(design_key(model | changes))
        if reference is not None:
            specs.append(dict(family=family, comparison=comparison,
                              model_id=model["model_id"], benchmark_id=reference))

    for m in models:
        if m["penalty_grid"] == "expanded":
            add(m, "penalty_expansion", "expanded_minus_original_penalties", penalty_grid="original")
        if m["transform"] == "daily_rank":
            add(m, "transformation", "daily_rank_minus_standard", transform="standard")
        if m["pca_grid"] == "expanded":
            add(m, "pca_expansion", "expanded_minus_original_pca", pca_grid="original")
        if m["representation"] == "pca":
            add(m, "compression", "pca_minus_full_text", representation="full", pca_grid="none")
        if m["estimator"] == "group_ridge":
            add(m, "group_penalty", "group_minus_global_ridge", estimator="ridge")
        if m["feature_set"] == "characteristics_core":
            grouped = m["estimator"] == "group_ridge"
            add(m, "incremental_social", "group_core_minus_ridge_characteristics" if grouped else "core_given_characteristics",
                feature_set="characteristics", estimator="ridge" if grouped else m["estimator"])
        if m["feature_set"] == "characteristics_textcore":
            add(m, "incremental_social", "pca_text_given_core" if m["representation"] == "pca" else "full_text_given_core",
                feature_set="characteristics_core", representation="full", pca_grid="none")
        if m["estimator"] != "ols":
            add(m, "estimator", "versus_ols", estimator="ols", penalty_grid="original")
    return specs


def family_counts(models):
    """Per-target correction budgets; target-specific menus must agree."""
    counts = [dict(Counter(s["family"] for s in contrast_specs([m for m in models if m["target"] == target])))
              for target in dict.fromkeys(m["target"] for m in models)]
    if not counts or any(c != counts[0] for c in counts):
        raise ValueError("Targets must declare the same comparison families")
    return counts[0]


PLANNED_FAMILY_COUNTS = family_counts(planned_models())


def validate_matrix(models, allow_pilot=False):
    planned = {design_key(m) + (m["target"],) for m in planned_models()}
    found = [design_key(m) + (m["target"],) for m in models]
    if not found or len({m["model_id"] for m in models}) != len(found) or len(set(found)) != len(found):
        raise ValueError("Nonempty unique optimization procedures and model IDs required")
    if not set(found) <= planned:
        raise ValueError("Registry contains unplanned optimization procedures")
    if not allow_pilot and set(found) != planned:
        raise ValueError("Production requires all 172 registered optimization procedures")
    for m in models:
        if (m.get("fit_days") != 504 or m.get("validation_days") != 126 or m.get("horizon") != 1
                or m.get("target_column") != {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[m["target"]]):
            raise ValueError("Models must preserve F504, V126, union refit and h1 targets")
        if not allow_pilot and (m.get("rows") != EXPECTED_ROWS or m.get("months") != 108):
            raise ValueError("Production requires 108 months and the unchanged 3,034,035 economic keys")


def registered_family_counts(registry):
    """Refuse inferred/observed multiplicity budgets, including in a pilot."""
    declared = registry.get("planned_family_counts", registry.get("config", {}).get("planned_family_counts"))
    if declared != PLANNED_FAMILY_COUNTS:
        raise ValueError("Registry must predeclare the complete planned_family_counts")
    return declared


def paired_contrasts(daily, models, lags=(5, 21, 63), counts=None):
    specs = contrast_specs(models)
    observed = Counter(s["family"] for s in specs)
    counts = PLANNED_FAMILY_COUNTS if counts is None else counts
    rows = []
    for metric in kernel.METRICS:
        wide = daily.pivot(index="date", columns="model_id", values=metric)
        for spec in specs:
            count = counts[spec["family"]]
            if observed[spec["family"]] > count:
                raise ValueError("Observed comparisons exceed predeclared budget")
            delta = wide[spec["model_id"]] - wide[spec["benchmark_id"]]
            for lag in lags:
                stats = kernel.calendar_hac_mean(delta, lags=lag)
                p = stats["p"]
                rows.append(spec | stats | dict(metric=metric, family_size=count,
                            observed_family_size=observed[spec["family"]],
                            p_bonferroni=min(1., p * count) if np.isfinite(p) else np.nan))
    return pd.DataFrame(rows)


def summarize_daily(daily, models, lags=5):
    extra = [k for k in DESIGN_FIELDS if k not in ("estimator", "feature_set")]
    return kernel.summarize_daily(daily, models, lags).merge(
        pd.DataFrame(models)[["model_id"] + extra], on="model_id", how="left", validate="many_to_one")


def _bucket(values, codes):
    groups = np.full(len(values), -1, dtype=np.int8)
    order = np.argsort(codes, kind="stable")
    _, starts, sizes = np.unique(codes[order], return_index=True, return_counts=True)
    for first, size in zip(starts, sizes):
        positions = order[first:first + size]
        groups[positions] = kernel.terciles(values[positions])
    return groups


def score_registry(registry_path, prepared, out, models, counts, start, end, batch_size=8,
                   skip_subgroups=False, skip_horizons=False):
    """Two passes establish the global intersection before bounded-batch scoring."""
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    keys = pd.read_pickle(prepared / "keys.pkl")
    keys["date"] = pd.to_datetime(keys.date)
    if keys[["date", "permno"]].isna().any().any() or keys.duplicated(["date", "permno"]).any():
        raise ValueError("Prepared economic keys must be unique and nonmissing")
    evaluation = pd.read_pickle(prepared / "evaluation.pkl")
    if len(evaluation) != len(keys) or not evaluation.index.equals(keys.index):
        raise ValueError("Evaluation table must share prepared key row order and index")
    y = np.load(prepared / "y.npy", mmap_mode="r")
    full_calendar = pd.DatetimeIndex(np.load(prepared / "calendar.npy"), name="date")
    calendar = full_calendar[(full_calendar >= pd.Timestamp(start)) & (full_calendar <= pd.Timestamp(end))]
    if not len(calendar) or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Evaluation calendar must be nonempty, unique and increasing")
    positions = np.flatnonzero(keys.date.between(calendar.min(), calendar.max()).to_numpy())
    panel, ep = keys.iloc[positions].reset_index(drop=True), evaluation.iloc[positions].reset_index(drop=True)
    codes, cap = calendar.get_indexer(panel.date), ep["cap"].to_numpy(dtype=float)
    if (codes < 0).any():
        raise ValueError("Prepared keys fall outside the supplied exchange calendar")
    outputs = {name: [] for name in ("", "_yearly", "_periods", "_contrasts", "_coverage", "_deciles", "_subgroups", "_horizons")}
    out.parent.mkdir(parents=True, exist_ok=True)
    daily_path = out.with_name(out.name + "_daily.csv")
    daily_written = False
    sources, coverage_summary, skipped_horizons = {}, {}, []
    for target_name in dict.fromkeys(m["target"] for m in models):
        local = [m for m in models if m["target"] == target_name]
        target = np.asarray(y[positions, manifest["targets"].index(local[0]["target_column"])], dtype=float)
        common, coverage = np.ones(len(panel), dtype=bool), []
        for model in local:
            path = Path(model["predictions"])
            prediction = kernel.align_predictions(panel, pd.read_pickle(path)).to_numpy(dtype=np.float64)
            observed = np.isfinite(prediction)
            common &= observed
            sources[model["model_id"]] = kernel._file_info(path)
            coverage.append(dict(target=target_name, model_id=model["model_id"], eligible_stock_days=len(panel),
                                 prediction_stock_days=int(observed.sum()), evaluated_stock_days=int((observed & np.isfinite(target)).sum()),
                                 missing_prediction_stock_days=int((~observed).sum())))
        if not common.any():
            raise ValueError(f"No common predictions for {target_name}")
        coverage_summary[target_name] = dict(models=len(local), eligible_stock_days=len(panel),
                common_prediction_stock_days=int(common.sum()), common_scored_stock_days=int((common & np.isfinite(target)).sum()))
        for row in coverage:
            row.update({k: v for k, v in coverage_summary[target_name].items() if k.startswith("common_")})
        outputs["_coverage"].append(pd.DataFrame(coverage))
        t, c, d = target[common], cap[common], codes[common]
        buckets = {}
        if not skip_subgroups:
            if "log_volume" not in ep:
                raise ValueError("Activity diagnostics require log_volume")
            buckets = {"size": _bucket(np.where(c > 0, c, np.nan), d),
                       "activity": _bucket(ep.log_volume.to_numpy(dtype=float)[common], d)}
        horizons = []
        if not skip_horizons:
            for h in (3, 5, 10, 21, 42, 63):
                column = f"f_cumret{h}" if target_name == "raw" else f"ar_dgtw_{h}"
                reason = "column unavailable" if column not in ep else (
                    "Known source ar_dgtw_21 corruption; excluded pending repair" if target_name == "dgtw" and h == 21 else None)
                if reason:
                    skipped_horizons.append(dict(target=target_name, horizon=h, reason=reason))
                else:
                    horizons.append((h, column, ep[column].to_numpy(dtype=float)[common]))
        daily_parts = []
        for first in range(0, len(local), batch_size):
            batch = local[first:first + batch_size]
            p = np.empty((int(common.sum()), len(batch)), dtype=np.float64)
            for j, model in enumerate(batch):
                p[:, j] = kernel.align_predictions(panel, pd.read_pickle(model["predictions"])).to_numpy(dtype=np.float64)[common]
            daily, deciles = kernel.score_scope(p, t, c, d, calendar, batch)
            daily.assign(target=target_name).to_csv(daily_path, index=False, mode="a" if daily_written else "w", header=not daily_written)
            daily_written = True
            daily_parts.append(daily[["date", "model_id"] + kernel.METRICS])
            outputs[""].append(summarize_daily(daily, batch).assign(target=target_name))
            outputs["_deciles"].append(deciles.assign(target=target_name, scope="full", horizon=1))
            for year in sorted(daily.date.dt.year.unique()):
                outputs["_yearly"].append(summarize_daily(daily[daily.date.dt.year == year], batch).assign(target=target_name, year=year))
            for grouping, bucket in buckets.items():
                for number in range(3):
                    label = f"{grouping}_tercile_{number + 1}"
                    sd, sp = kernel.score_scope(p, t, c, d, calendar, batch, bucket == number)
                    outputs["_subgroups"].append(summarize_daily(sd, batch).assign(target=target_name, subgroup=label,
                                                    stock_days=int(((bucket == number) & np.isfinite(t)).sum())))
                    outputs["_deciles"].append(sp.assign(target=target_name, scope=label, horizon=1))
            for horizon, column, ht in horizons:
                hd, hp = kernel.score_scope(p, ht, c, d, calendar, batch)
                outputs["_horizons"].append(summarize_daily(hd, batch, lags=max(5, horizon - 1)).assign(
                    target=target_name, horizon=horizon, target_column=column, stock_days=int(np.isfinite(ht).sum())))
                outputs["_deciles"].append(hp.assign(target=target_name, scope="full", horizon=horizon))
            print(f"Scored {target_name}: {first + len(batch)}/{len(local)} models", flush=True)
        daily = pd.concat(daily_parts, ignore_index=True)
        for period, first, last in (("full", start, end), ("2014-2018", "2014-01-01", "2018-12-31"),
                                    ("2019-2022", "2019-01-01", "2022-12-31")):
            subset = daily[daily.date.between(first, last)]
            if len(subset):
                outputs["_periods"].append(summarize_daily(subset, local).assign(target=target_name, period=period))
                outputs["_contrasts"].append(paired_contrasts(subset, local, counts=counts).assign(target=target_name, period=period))
    written = {suffix or "summary": kernel._write_table(out, suffix, tables) for suffix, tables in outputs.items()}
    compressed = daily_path.with_suffix(".csv.gz")
    with daily_path.open("rb") as source, compressed.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as archive:
            shutil.copyfileobj(source, archive, length=8 * 1024 * 1024)
    written["_daily"] = kernel._file_info(compressed)
    return dict(prediction_sources=sources, coverage=coverage_summary, output_files=written,
                skipped_horizons=skipped_horizons, source_limitations=manifest.get("source_limitations", []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", "--outprefix", dest="out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--skip-horizons", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Threads and batch size must be positive")
    if args.end >= "2023-01-01":
        parser.error("No 2023 or later outcomes may be scored")
    if not args.allow_pilot and (args.start, args.end) != ("2014-01-01", "2022-12-31"):
        parser.error("A changed evaluation period requires --allow-pilot")
    started = time.monotonic()
    registry, models = kernel.read_registry(args.registry)
    if registry.get("config", {}).get("run_kind", registry.get("kind")) == "pilot" and not args.allow_pilot:
        raise ValueError("Pilot registry requires --allow-pilot")
    validate_matrix(models, args.allow_pilot)
    counts = registered_family_counts(registry)
    prepared = args.prepared.resolve()
    if "prepared" in registry and Path(registry["prepared"]).resolve() != prepared:
        raise ValueError("Registry and requested prepared data differ")
    registry_hash, wrapper_hash = file_hash(args.registry), file_hash(__file__)
    prepared_hashes = {name: file_hash(prepared / name) for name in PREPARED_FILES if (prepared / name).exists()}
    kernel_hashes = {name: file_hash(Path(kernel.__file__).with_name(name)) for name in KERNEL_FILES}
    prediction_hashes, prediction_stats = {}, {}
    for model in models:
        digest = file_hash(model["predictions"])
        if not args.allow_pilot and not model.get("sha256"):
            raise ValueError("Production registry requires prediction SHA-256 digests")
        if model.get("sha256", digest) != digest:
            raise ValueError(f"Prediction digest mismatch: {model['model_id']}")
        prediction_hashes[model["model_id"]], prediction_stats[model["model_id"]] = digest, kernel._file_info(model["predictions"])
    with threadpool_limits(args.threads):
        metadata = score_registry(args.registry, prepared, args.out, models, counts, args.start, args.end,
                                  args.batch_size, args.skip_subgroups, args.skip_horizons)
    if file_hash(args.registry) != registry_hash or file_hash(__file__) != wrapper_hash:
        raise ValueError("Registry or evaluation adapter changed during scoring")
    if file_hash(prepared / "manifest.json") != prepared_hashes["manifest.json"]:
        raise ValueError("Prepared manifest changed during scoring")
    for name, digest in kernel_hashes.items():
        if file_hash(Path(kernel.__file__).with_name(name)) != digest:
            raise ValueError("Reused scoring kernel changed during evaluation")
    for model in models:
        if kernel._file_info(model["predictions"]) != prediction_stats[model["model_id"]]:
            raise ValueError("Prediction file changed during scoring")
    if not args.allow_pilot:
        for coverage in metadata["coverage"].values():
            if coverage["eligible_stock_days"] != EXPECTED_ROWS or coverage["common_prediction_stock_days"] != EXPECTED_ROWS:
                raise ValueError("Production comparison changed the parent prediction universe")
    selection = args.registry.parent / "monthly_selection.csv"
    metadata.update(schema_version="linear_optimization_v3_evaluation", created=pd.Timestamp.now().isoformat(),
        registry=str(args.registry.resolve()), registry_sha256=registry_hash, prepared=str(prepared),
        prepared_sha256=prepared_hashes, prediction_sha256=prediction_hashes, reused_kernel_sha256=kernel_hashes,
        wrapper_sha256=wrapper_hash, planned_models=PLANNED_MODELS, included_models=len(models),
        pilot=args.allow_pilot, preliminary=args.allow_pilot, planned_family_counts=counts,
        observed_family_counts={target: dict(Counter(s["family"] for s in contrast_specs([m for m in models if m["target"] == target])))
                                for target in dict.fromkeys(m["target"] for m in models)},
        contrast_families=FAMILY_DESCRIPTIONS,
        multiplicity="Bonferroni within target/family/metric/period over the predeclared complete 172-model design. Missing statistics and pilot subsets never reduce denominators. HAC21/HAC63 are sensitivities. Prior development research is not covered.",
        primary_target="raw", secondary_target="dgtw", primary_metric="rank_ic", secondary_metrics=["ew_spread_bp", "cap_spread_bp"],
        hac_lags=[5, 21, 63], inference="Two-sided normal Bartlett HAC; complete supplied exchange calendar retains missing sessions",
        common_sample="Joint finite predictions across all registered models within each target, then observed outcomes; established before bounded-batch scoring",
        research_status="2014-2022 is already-inspected development evidence. No test-score winner is selected. No 2023 outcomes are scored; 2023 lacks a completeness and prior-use audit.",
        portfolio_limit="Gross descriptive decile sorts; no attainable same-close execution, turnover, costs or factor alpha estimated",
        kernel_reuse="Unchanged protocol_evaluate alignment, score_scope, score_day, terciles, summarize_daily and calendar_hac_mean; new bounded-batch orchestration and registered comparisons",
        daily_artifact="Deterministic gzip (mtime=0); uncompressed local daily table retained",
        monthly_selection={"path": str(selection.resolve()), "sha256": file_hash(selection)} if selection.exists() else None,
        numerical_threads=args.threads, prediction_batch_size=args.batch_size, elapsed_seconds=time.monotonic() - started,
        output_sha256={Path(info["path"]).name: file_hash(info["path"]) for info in metadata["output_files"].values()},
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        python=platform.python_version(), interpreter=sys.executable)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"LINEAR OPTIMIZATION EVALUATION COMPLETE: {len(models)}/{PLANNED_MODELS}; {args.out}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
