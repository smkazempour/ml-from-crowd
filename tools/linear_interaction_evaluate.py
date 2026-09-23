"""Evaluate Stage A's prespecified explicit-interaction forecasting procedures.

Bounded prediction batches preserve the global common sample using a local
adaptation of the v3 loop. Alignment, rank/portfolio scoring and calendar HAC
reuse the same protocol kernels. Imported module state is never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
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

import linear_optimization_evaluate as backend
import protocol_evaluate as kernel

TARGETS = ("raw", "dgtw")
ESTIMATORS = ("ols", "ridge", "enet")
CONTROL_NAMES = ("char_ret1", "char_ret5", "char_ret21", "char_ret63", "char_ret126", "char_ret252",
    "char_momentum252_skip21", "char_log_market_cap", "char_log_price", "char_turnover1", "char_turnover21",
    "char_log_dollar_volume1", "char_log_dollar_volume21", "char_volatility21", "char_volatility63",
    "char_max_return21", "char_amihud21")
BASIS_COLUMNS = {"additive_c": 34, "additive_cs": 36, "additive_cst_full": 422,
    "c_squares": 51, "c_quadratic": 187, "cq_s": 189, "cq_s_squares": 191, "cs_joint": 226,
    "text_main": 244, "text_squares": 262, "agreement_c": 296, "text_c": 568, "text_s": 604}
TEXT_BASES = ("text_main", "text_squares", "agreement_c", "text_c", "text_s")
DESIGN_FIELDS = ("estimator", "basis", "interaction_term", "history_policy", "refit_policy")
SUMMARY_FIELDS = ("basis", "interaction_term", "n_features", "representation", "transform", "penalty_grid", "pca_grid",
                  "history_policy", "refit_policy")
PLANNED_MODELS, EXPECTED_ROWS = 148, 3_034_035
FAMILY_DESCRIPTIONS = {
    "characteristic_shape": "Characteristic squares versus main effects; quadratic versus squares; quadratic versus main effects",
    "social_information": "Additive social main effects given additive C; social main effects, squares and joint products each given quadratic C",
    "social_shape": "Social squares versus social main effects; joint social products versus social squares; joint social model versus social main effects",
    "individual_social": "Each of all 35 named social products added separately to the same 191-column ridge benchmark",
    "text_information": "Additive full text given additive C+S; each compressed-text extension given the 226-column C+S interaction benchmark",
    "text_shape": "Successive text squares, agreement-by-C, embedding-PC-by-C, and text-by-S additions; combined text-by-C block versus text squares",
    "estimator": "Ridge and elastic net versus OLS, plus elastic net versus ridge, on each of the 13 joint/reference bases",
    "additive_reference": "Joint social versus additive C+S; each compressed-text interaction basis versus additive full-embedding C+S+text. These are compound procedure comparisons",
}
file_hash = backend.file_hash


def social_terms():
    return tuple(f"{social}__x__{control}" for social in ("net_sentiment", "log_volume") for control in CONTROL_NAMES) + ("net_sentiment__x__log_volume",)


def planned_models():
    """Independent menu to catch missing procedures, duplicate terms or changed bases."""
    bases = [(basis, None, count, estimator) for basis, count in BASIS_COLUMNS.items() for estimator in ESTIMATORS]
    bases += [("single_social192", term, 192, "ridge") for term in social_terms()]
    result = []
    for (basis, term, count, estimator), target in itertools.product(bases, TARGETS):
        feature = ("characteristics" if basis in ("additive_c", "c_squares", "c_quadratic") else
                   "characteristics_textcore" if basis in TEXT_BASES or basis == "additive_cst_full" else "characteristics_core")
        representation = "full" if basis.startswith("additive_") else ("pca_interactions" if basis in TEXT_BASES else "explicit")
        result.append(dict(model_id=f"li1_{estimator}_{basis}_{term or 'joint'}_{target}_fixed504_union",
            basis=basis, interaction_term=term, n_features=count, estimator=estimator, target=target,
            feature_set=feature, representation=representation, transform="standard", penalty_grid="original",
            pca_grid="fixed16" if basis in TEXT_BASES else "none", history_policy="fixed504", refit_policy="union",
            fit_days=504, validation_days=126, horizon=1, target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target]))
    return result


def design_key(model):
    return tuple(model[key] for key in DESIGN_FIELDS)


def contrast_specs(models):
    if len({m["target"] for m in models}) > 1:
        raise ValueError("Contrast specification requires one target")
    lookup = {design_key(m): m["model_id"] for m in models}
    if len(lookup) != len(models):
        raise ValueError("Duplicate interaction procedure")
    specs = []

    def add(model, family, comparison, basis, estimator=None):
        reference = lookup.get(design_key(model | dict(basis=basis, interaction_term=None,
            estimator=estimator or model["estimator"])))
        if reference is not None:
            specs.append(dict(family=family, comparison=comparison, model_id=model["model_id"], benchmark_id=reference))

    edges = {
        "c_squares": [("characteristic_shape", "c_squares_minus_c", "additive_c")],
        "c_quadratic": [("characteristic_shape", "c_pairs_given_squares", "c_squares"),
                        ("characteristic_shape", "c_quadratic_minus_c", "additive_c")],
        "additive_cs": [("social_information", "additive_social_given_c", "additive_c")],
        "cq_s": [("social_information", "social_main_given_quadratic_c", "c_quadratic")],
        "cq_s_squares": [("social_information", "social_curvature_given_quadratic_c", "c_quadratic"),
                         ("social_shape", "social_squares_given_main", "cq_s")],
        "cs_joint": [("social_information", "social_joint_given_quadratic_c", "c_quadratic"),
                     ("social_shape", "social_products_given_squares", "cq_s_squares"),
                     ("social_shape", "social_joint_minus_main", "cq_s"),
                     ("additive_reference", "social_joint_minus_additive_cs", "additive_cs")],
        "additive_cst_full": [("text_information", "full_text_given_additive_cs", "additive_cs")],
        "text_squares": [("text_shape", "text_squares_given_main", "text_main")],
        "agreement_c": [("text_shape", "agreement_c_given_text_squares", "text_squares")],
        "text_c": [("text_shape", "embedding_c_given_agreement_c", "agreement_c"),
                   ("text_shape", "all_text_c_given_text_squares", "text_squares")],
        "text_s": [("text_shape", "text_s_given_text_c", "text_c")],
        "single_social192": [("individual_social", "single_social_product_given_squares", "cq_s_squares")],
    }
    for model in models:
        for family, comparison, reference in edges.get(model["basis"], []):
            add(model, family, comparison, reference)
        if model["basis"] in TEXT_BASES:
            add(model, "text_information", model["basis"] + "_given_social_joint", "cs_joint")
            add(model, "additive_reference", model["basis"] + "_minus_additive_full_text", "additive_cst_full")
        if model["basis"] != "single_social192" and model["estimator"] in ("ridge", "enet"):
            add(model, "estimator", "versus_ols", model["basis"], "ols")
            if model["estimator"] == "enet":
                add(model, "estimator", "enet_versus_ridge", model["basis"], "ridge")
    return specs


def family_counts(models):
    counts = [dict(Counter(s["family"] for s in contrast_specs([m for m in models if m["target"] == target])))
              for target in dict.fromkeys(m["target"] for m in models)]
    if not counts or any(count != counts[0] for count in counts):
        raise ValueError("Targets must declare identical comparison families")
    return counts[0]


PLANNED_FAMILY_COUNTS = family_counts(planned_models())


def validate_matrix(models, allow_pilot=False):
    expected = {design_key(m) + (m["target"],): m for m in planned_models()}
    found = [design_key(m) + (m["target"],) for m in models]
    if not found or len(set(found)) != len(found) or len({m["model_id"] for m in models}) != len(found):
        raise ValueError("Unique nonempty interaction procedures and model IDs required")
    if not set(found) <= set(expected):
        raise ValueError("Registry contains an unplanned interaction procedure or term")
    if not allow_pilot and set(found) != set(expected):
        raise ValueError("Production requires all 148 registered interaction procedures")
    fields = ("n_features", "feature_set", "representation", "transform", "penalty_grid", "pca_grid",
              "fit_days", "validation_days", "horizon", "target_column")
    for key, model in zip(found, models):
        if any(model.get(field) != expected[key][field] for field in fields):
            raise ValueError("Procedure metadata differs from the declared basis, transformations or chronology")
        if not allow_pilot and (model.get("rows") != EXPECTED_ROWS or model.get("months") != 108):
            raise ValueError("Production requires 108 months and the unchanged 3,034,035 economic keys")


def registered_family_counts(registry):
    counts = registry.get("planned_family_counts", registry.get("config", {}).get("planned_family_counts"))
    if counts != PLANNED_FAMILY_COUNTS:
        raise ValueError("Registry must predeclare the complete planned_family_counts")
    return counts


def paired_contrasts(daily, models, lags=(5, 21, 63), counts=None):
    counts = PLANNED_FAMILY_COUNTS if counts is None else counts
    specs = contrast_specs(models)
    observed = Counter(s["family"] for s in specs)
    rows = []
    for metric in kernel.METRICS:
        wide = daily.pivot(index="date", columns="model_id", values=metric)
        for spec in specs:
            count = counts[spec["family"]]
            if observed[spec["family"]] > count:
                raise ValueError("Observed comparisons exceed their predeclared budget")
            delta = wide[spec["model_id"]] - wide[spec["benchmark_id"]]
            for lag in lags:
                stats = kernel.calendar_hac_mean(delta, lags=lag)
                p = stats["p"]
                rows.append(spec | stats | dict(metric=metric, family_size=count,
                    observed_family_size=observed[spec["family"]], p_bonferroni=min(1., p * count) if np.isfinite(p) else np.nan))
    return pd.DataFrame(rows)


def summarize_daily(daily, models, lags=5):
    return kernel.summarize_daily(daily, models, lags).merge(pd.DataFrame(models)[["model_id"] + list(SUMMARY_FIELDS)],
        on="model_id", how="left", validate="many_to_one")


_bucket = backend._bucket


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
        parser.error("Changed evaluation period requires --allow-pilot")
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
    prepared_hashes = {name: file_hash(prepared / name) for name in backend.PREPARED_FILES if (prepared / name).exists()}
    code_files = (*backend.KERNEL_FILES, "linear_optimization_evaluate.py")
    code_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in code_files}
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
    for name, digest in code_hashes.items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Reused scoring source changed during evaluation")
    for model in models:
        if kernel._file_info(model["predictions"]) != prediction_stats[model["model_id"]]:
            raise ValueError("Prediction file changed during scoring")
    if not args.allow_pilot:
        for coverage in metadata["coverage"].values():
            if coverage["eligible_stock_days"] != EXPECTED_ROWS or coverage["common_prediction_stock_days"] != EXPECTED_ROWS:
                raise ValueError("Production comparison changed the parent prediction universe")
    for name in ("summary", "_contrasts", "_coverage", "_yearly", "_periods"):
        path = metadata["output_files"][name]["path"]
        if pd.read_csv(path).empty:
            raise ValueError(f"Required evaluation artifact is empty: {name}")
    selection = args.registry.parent / "monthly_selection.csv"
    metadata.update(schema_version="linear_interactions_v1_evaluation", created=pd.Timestamp.now().isoformat(),
        registry=str(args.registry.resolve()), registry_sha256=registry_hash, prepared=str(prepared),
        prepared_sha256=prepared_hashes, prediction_sha256=prediction_hashes, reused_kernel_sha256=code_hashes,
        wrapper_sha256=wrapper_hash, planned_models=PLANNED_MODELS, included_models=len(models),
        pilot=args.allow_pilot, preliminary=args.allow_pilot, planned_family_counts=counts,
        observed_family_counts={target: dict(Counter(s["family"] for s in contrast_specs([m for m in models if m["target"] == target])))
                                for target in dict.fromkeys(m["target"] for m in models)},
        contrast_families=FAMILY_DESCRIPTIONS,
        multiplicity="Bonferroni within target/family/metric/period over the predeclared 148-model design. All 35 individual ridge terms share a 35-comparison budget. Missing statistics and pilots never reduce denominators. No across-family or development-history error guarantee.",
        primary_target="raw", secondary_target="dgtw", primary_metric="rank_ic", secondary_metrics=["ew_spread_bp", "cap_spread_bp"],
        primary_estimator="ridge", hac_lags=[5, 21, 63],
        inference="Two-sided normal Bartlett HAC; complete exchange calendar retains missing sessions; HAC21/63 are sensitivities",
        common_sample="Joint finite predictions across all models within each target before observed-outcome filtering, established before bounded-batch scoring",
        research_status="2014-2022 is already-inspected development evidence; no test-score winner is selected; no 2023 outcomes scored",
        transformation="Existing once-ranked continuous characteristics/social scalars; missing characteristic ranks neutral zero with separate binary flags. Products are not reranked. Fitting-only basis scaling and PCA; PCA16 rebuilt on final permitted fitting inputs.",
        hierarchy="Parent main-effect columns are retained; ordinary elastic net does not enforce nonzero parent coefficients",
        interpretation="Named single additions are conditional prediction tests, not causal effects or coefficient-sign tests. Monthly text PCs can rotate; interpret blocks, not stable PC topics.",
        portfolio_limit="Gross descriptive decile sorts; no attainable same-close execution, turnover, costs or factor alpha estimated",
        kernel_reuse="Local bounded-batch orchestration adapted from v3; unchanged protocol alignment, score_scope, score_day, terciles and calendar_hac_mean",
        daily_artifact="Deterministic gzip (mtime=0); uncompressed local table retained",
        monthly_selection={"path": str(selection.resolve()), "sha256": file_hash(selection)} if selection.exists() else None,
        numerical_threads=args.threads, prediction_batch_size=args.batch_size, elapsed_seconds=time.monotonic() - started,
        output_sha256={Path(info["path"]).name: file_hash(info["path"]) for info in metadata["output_files"].values()},
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        python=platform.python_version(), interpreter=sys.executable)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"LINEAR INTERACTION EVALUATION COMPLETE: {len(models)}/{PLANNED_MODELS}; {args.out}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
