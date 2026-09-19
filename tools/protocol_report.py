"""Build a reproducible Markdown report from registered protocol evaluations.

Example: python tools/protocol_report.py --registry RUN/registry.json
  --evaluation reports/data/protocol_v1_1_linear
  --out reports/07_protocol_linear_results.md

Pilot or incomplete evaluation is refused unless --allow-pilot is explicit. Such
reports are marked as plumbing checks and contain no automatic evidence claims.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["core", "all", "textcore", "textall"]
FEATURE_LABELS = {"core": "Core", "all": "All social", "textcore": "Text + core", "textall": "Text + all social"}
ESTIMATOR_LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
TARGET_LABELS = {"raw": "Raw", "dgtw": "DGTW"}


def number(value, digits=4):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "—"
    value = float(value)
    return f"{0.0 if abs(value) < .5*10**(-digits) else value:.{digits}f}"


def pvalue(value):
    if pd.isna(value):
        return "—"
    return "<0.001" if value < .001 else number(value, 3)


def table(frame):
    if frame.empty:
        return "No observations available."
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = [[cell(x) for x in frame.columns]]
    rows += [["---"]*len(frame.columns)]
    rows += [[cell(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def link(path, out, label=None):
    path = Path(path).resolve()
    relative = os.path.relpath(path, Path(out).resolve().parent).replace("\\", "/")
    return f"[{label or path.name}](<{relative}>)"


def read_csv(prefix, suffix=""):
    path = prefix.with_name(prefix.name+suffix+".csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def labels(frame):
    frame = frame.copy()
    if "estimator" in frame:
        frame["estimator"] = frame.estimator.map(lambda x: ESTIMATOR_LABELS.get(x, str(x).upper()))
    if "feature_set" in frame:
        frame["feature_set"] = frame.feature_set.map(lambda x: FEATURE_LABELS.get(x, x))
    if "target" in frame:
        frame["target"] = frame.target.map(lambda x: TARGET_LABELS.get(x, x))
    return frame


def matrix(frame, index, column, order=None, digits=4):
    if frame.empty:
        return "No observations available."
    frame = frame.copy()
    for name, base in [("target", list(TARGET_LABELS)), ("estimator", list(ESTIMATOR_LABELS))]:
        if name in frame:
            observed = frame[name].drop_duplicates().tolist()
            categories = [x for x in base if x in observed]+[x for x in observed if x not in base]
            frame[name] = pd.Categorical(frame[name], categories=categories, ordered=True)
    wide = frame.pivot(index=index, columns=column, values="mean")
    if order is not None:
        wide = wide.reindex(columns=[x for x in order if x in wide])
    wide = wide.map(lambda value: number(value, digits)).reset_index()
    return table(labels(wide).rename(columns={"estimator": "Estimator", "target": "Target",
                                             "year": "Year", "fit_days": "Fit dates"}
                                    | FEATURE_LABELS | ESTIMATOR_LABELS
                                    | {c: str(c).upper() for c in wide if str(c).startswith("nn")}))


def attach_model_fields(frame, models):
    fields = [c for c in ["estimator", "feature_set", "fit_days"] if c not in frame.columns]
    if frame.empty:
        return frame
    return frame.merge(models[["model_id"]+fields], on="model_id", how="left", validate="many_to_one")


def contrast_table(frame, include_feature=True, include_fit=False, include_benchmark=False):
    if frame.empty:
        return "No registered comparisons available."
    frame = labels(frame)
    result = pd.DataFrame({"Target": frame.target, "Estimator": frame.estimator})
    if include_benchmark:
        result["Benchmark"] = frame.benchmark_estimator.map(lambda x: ESTIMATOR_LABELS.get(x, str(x).upper()))
    if include_feature:
        result["Inputs"] = frame.feature_set
    if include_fit:
        result["Fit dates"] = frame.fit_days.astype(int)
    result["Δ IC"] = frame["mean"].map(number)
    result["95% CI"] = [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)]
    result["HAC t"] = frame.t.map(lambda x: number(x, 2))
    result["Adjusted p"] = frame.p_bonferroni.map(pvalue)
    return table(result)


def numerical_summary(models, out):
    sections = []
    directories = {Path(path).resolve().parent for path in models.predictions}
    selected = []
    for directory in sorted(directories):
        path = directory/"monthly_selection.csv"
        if path.exists():
            frame = pd.read_csv(path)
            selected.append(frame[frame.model_id.isin(models.model_id)])
    if selected:
        frame = pd.concat(selected, ignore_index=True).drop_duplicates(["model_id", "month"])
        rows = []
        for estimator, group in frame.groupby("estimator", sort=False):
            hit = group.get("coordinate_descent_limit", pd.Series(False, index=group.index))
            rows.append({"Estimator": ESTIMATOR_LABELS.get(estimator, estimator.upper()),
                         "Selected monthly fits": len(group),
                         "Max KKT residual": f"{group.kkt_violation.max():.2e}" if "kkt_violation" in group else "—",
                         "Max dual gap": f"{group.dual_gap.max():.2e}" if "dual_gap" in group and group.dual_gap.notna().any() else "—",
                         "Initial CD limit hits": int(hit.astype(str).str.lower().isin(["true", "1"]).sum())})
        sections.append(table(pd.DataFrame(rows)))
        sections.append("These statistics describe the validation-selected fits. Coordinate-descent limit hits refer to the initial solver; active-set refinement and final optimality checks determine acceptance. Complete candidate diagnostics remain in the monthly checkpoints.")
        selections = [directory/"monthly_selection.csv" for directory in sorted(directories)
                      if (directory/"monthly_selection.csv").exists()]
        sections.append("Selection records: " + ", ".join(link(path, out) for path in selections) + ".")
    neural_rows = []
    for model in models.to_dict("records"):
        if not str(model["estimator"]).startswith("nn"):
            continue
        directory = Path(model["predictions"]).resolve().parent
        path = directory/"diagnostics.json"
        if not path.exists():
            continue
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
        spec_path = directory/"run_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
        cap = spec.get("config", {}).get("max_epochs", 100)
        epochs, reached_cap, candidate_fits = [], 0, 0
        for month in diagnostics.get("month_diagnostics", []):
            for candidate in month.get("candidate_diagnostics", []):
                curves = candidate.get("curves", [])
                reached_cap += sum(len(curve) >= cap for curve in curves)
                candidate_fits += len(curves)
                if candidate["penalty"] == month.get("chosen_penalty"):
                    epochs.extend(candidate.get("best_epochs", []))
        neural_rows.append({"Estimator": model["estimator"].upper(),
                            "Target": TARGET_LABELS.get(model["target"], model["target"]),
                            "Inputs": FEATURE_LABELS.get(model["feature_set"], model["feature_set"]),
                            "Fit dates": model["fit_days"], "Months": len(diagnostics.get("month_diagnostics", [])),
                            "Median selected epoch": number(np.median(epochs), 1) if epochs else "—",
                            "Candidate fits reaching cap": f"{reached_cap}/{candidate_fits}"})
    if neural_rows:
        sections.append(table(pd.DataFrame(neural_rows)))
        sections.append("Epoch-cap counts cover every candidate seed fit; selected epochs describe the chosen penalty's seed checkpoints. Validation alone selected checkpoints and penalties.")
    return "\n\n".join(sections) or "No numerical-diagnostic files were available beside the registered predictions."


def repair_provenance(prepared, models, out):
    """Describe the immutable timing repair and any certified checkpoint reuse."""
    path = Path(prepared)/"horizon_repair_audit.json"
    if not path.exists():
        return []
    audit = json.loads(path.read_text(encoding="utf-8"))
    sections = ["### Timing repair and result provenance"]
    observations = audit.get("excluded_observations", [])
    gaps = audit.get("longer_horizon_exclusions", {}).get("gaps", [])
    descriptions = []
    for observation in observations:
        matching = [gap for gap in gaps if gap.get("permno") == observation["permno"]
                    and gap.get("signal_date") == observation["date"]]
        detail = (f"; its next observed stock row was {matching[0]['next_observed_stock_date']}, "
                  f"after {matching[0]['missing_sessions']} missing exchange sessions" if matching else "")
        descriptions.append(f"{observation.get('ticker', 'security')} (PERMNO {observation['permno']}), signal {observation['date']}{detail}")
    sections.append("The corrected immutable cache excludes the invalid one-day raw and DGTW labels for "
                    + "; ".join(descriptions) + ". The prediction row and its features remain present. "
                    "Daily target ranks were recomputed after the exclusion. Additional cumulative-horizon labels whose nominal intervals cross the identified gap were excluded from diagnostics. "
                    "The older uncorrected full run is retained solely as a provenance/reuse source and is not the current result.")
    primary = audit.get("coverage_after", {}).get("primary_2014_2022", {})
    if primary:
        sections.append(f"After repair, the 2014–2022 input universe contains {primary['rows']:,} stock-days: "
                        f"{primary['finite_raw_targets']:,} observed raw one-day labels and {primary['missing_raw_targets']:,} missing raw labels. "
                        f"All {primary['compared_next_session_raw_targets']:,} observed raw labels in this period reconcile with the next exchange-session return. "
                        "Finite labels at the end of the broader source calendar remain explicitly unverified and fall outside this main period.")
    sections.append("Repair details and per-horizon exclusions: " + link(path, out) + ".")
    directories = {Path(p).resolve().parent for p in models.predictions}
    for directory in sorted(directories):
        migration_path = directory/"migration_audit.json"
        if not migration_path.exists():
            continue
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        counts = migration.get("counts", {})
        sections.append(f"Checkpoint migration verified unchanged inputs and fitting/validation dependencies before reusing "
                        f"{counts.get('reused', 0):,} unaffected month-target checkpoints; "
                        f"{counts.get('retrain_dependency', 0):,} affected checkpoints required retraining. "
                        "Audit: " + link(migration_path, out) + ".")
    return sections


def build_report(registry_path, prefix, out, title=None, allow_pilot=False):
    registry_path, prefix, out = Path(registry_path), Path(prefix), Path(out)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    models = pd.DataFrame(registry["models"])
    if models.empty or models.model_id.duplicated().any():
        raise ValueError("Registry requires nonempty, unique model IDs")
    for j, path in models.predictions.items():
        candidate = Path(path)
        models.loc[j, "predictions"] = str(candidate if candidate.is_absolute() else registry_path.resolve().parent/candidate)
    summary = read_csv(prefix)
    if set(summary.model_id) != set(models.model_id):
        raise ValueError("Evaluation summary does not match the complete supplied registry")
    for model in models.to_dict("records"):
        source = metadata["prediction_sources"][model["model_id"]]
        if Path(model["predictions"]).resolve() != Path(source["path"]).resolve():
            raise ValueError("Evaluation prediction source differs from registry")
    arguments = metadata["arguments"]
    expected_months = len(pd.period_range(arguments["start"], arguments["end"], freq="M"))
    incomplete = (models.get("months", pd.Series(expected_months, index=models.index)).astype(int) < expected_months).any()
    pilot = incomplete or registry.get("kind") == "pilot" or "pilot" in registry_path.parent.name
    if pilot and not allow_pilot:
        raise ValueError("Pilot/incomplete report requires --allow-pilot and must not be presented as full results")
    prepared = Path(metadata["prepared"])
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    contrasts = attach_model_fields(read_csv(prefix, "_contrasts"), models)
    contrasts["benchmark_estimator"] = contrasts.benchmark_id.map(models.set_index("model_id").estimator)
    main_contrasts = contrasts[(contrasts.period == "full") & (contrasts.hac_lags == 5)]
    primary = summary[(summary.fit_days == 504) & (summary.metric == "rank_ic")]
    default_title = "Protocol v1.1: linear and neural-network comparisons" if models.estimator.str.startswith("nn").any() else "Protocol v1.1: revised linear models"
    sections = ["# " + (title or default_title),
                "Generated from the saved evaluation artifacts. " + ("**PLUMBING SMOKE ONLY: this partial/pilot run does not support scientific inference.**" if pilot else "The tables use the complete supplied registry; model settings were selected on chronological validation data.")]
    sections += ["## Design and coverage",
                 "The signal at close t summarizes messages assigned to that close; the main outcome is the return from close t to close t+1. Models refit monthly; this registry contains fitting histories of " + ", ".join(str(x) for x in sorted(models.fit_days.unique())) + " sessions, with a fixed 126-session validation block. F504 is the primary history. Horizon-specific maturity rules separate fitting, validation, and testing. The selected fitting-block model is retained without refitting on validation.",
                 "Scalar features use centered daily midranks; embeddings and agreement measures remain unranked. Fitting-only scaling and equal-date squared loss on centered return ranks are shared across estimators. Validation uses mean daily Spearman IC. Feature availability determines prediction eligibility before future outcome availability is considered."]
    coverage_rows = []
    for target, coverage in metadata["coverage"].items():
        scores = summary[(summary.target == target) & (summary.metric == "rank_ic")]
        coverage_rows.append({"Target": TARGET_LABELS.get(target, target), "Models": coverage["models"],
                              "Eligible stock-days": f"{coverage['eligible_stock_days']:,}",
                              "Common predictions": f"{coverage['common_prediction_stock_days']:,}",
                              "Observed outcomes": f"{coverage['common_scored_stock_days']:,}",
                              "IC dates/model": f"{int(scores.n.min()):,}–{int(scores.n.max()):,}"})
    sections.append(table(pd.DataFrame(coverage_rows)))
    counts = ", ".join(f"{FEATURE_LABELS.get(name, name)}: {len(columns)}" for name, columns in manifest.get("feature_sets", {}).items())
    sections += [f"Evaluation period: {arguments['start']} through {arguments['end']}. Feature counts, including declared missingness flags: {counts}. All models within each target share one joint prediction-coverage intersection. Missing outcomes remain unscored; raw and DGTW samples can differ.",
                 "Core is net sentiment plus attention (`log_volume`). `all` denotes the current 53 engineered social-media features, including core; these are not stock characteristics or return-history controls. The text addition comprises 384 embedding coordinates plus two agreement measures. Text + core therefore has 388 inputs, and text + all social has 439 inputs before any additional missingness flags."]
    sections += repair_provenance(prepared, models, out)
    sections += [
                 "## Primary F504 rank IC",
                 "Rank IC is the equally weighted mean of daily Spearman correlations. These rank predictions are not return forecasts in percent; no return-unit R² is calculated."]
    for target in models.target.drop_duplicates():
        sections.append("### " + TARGET_LABELS.get(target, target))
        sections.append(matrix(primary[primary.target == target], "estimator", "feature_set", FEATURES))
    sections += ["## Paired feature additions at F504",
                 "The following comparisons hold the estimator and fitting history fixed. Differences are model minus benchmark; intervals are pointwise 95% intervals. Adjusted p-values use the saved Bonferroni family, separately by target, metric, and period."]
    feature = main_contrasts[(main_contrasts.family == "feature") & (main_contrasts.fit_days == 504)
                             & (main_contrasts.metric == "rank_ic")]
    tc = feature[(feature.comparison == "versus_core") & (feature.feature_set == "textcore")]
    ta = feature[feature.comparison == "text_given_all"]
    sections += ["### Text + core minus core", contrast_table(tc, include_feature=False),
                 "### Text + all social minus all social", contrast_table(ta, include_feature=False)]
    if not pilot:
        for label, rows in [("text + core versus core", tc), ("text + all social versus all social", ta)]:
            valid = rows.p_bonferroni.notna()
            positive = (rows["mean"] > 0) & (rows.p_bonferroni < .05)
            negative = (rows["mean"] < 0) & (rows.p_bonferroni < .05)
            sections.append(f"For {label}, {int(positive.sum())} of {int(valid.sum())} available target/estimator contrasts are positive with adjusted p < 0.05; {int(negative.sum())} are negative with adjusted p < 0.05. This counts the stated comparisons and does not select a new benchmark.")
    families = main_contrasts[["target", "family", "family_size"]].drop_duplicates()
    sections.append("Family sizes in the saved evaluation:\n\n" + table(labels(families).rename(columns={"target": "Target", "family": "Family", "family_size": "Comparisons"})))
    sections.append("The feature family includes all noncore inputs versus core and text + all social versus all social. The experiment specification documents the pre-result clarification of this family. The complete contrast file also includes all social versus core and text + all social versus core, together with HAC21/HAC63 sensitivities.")
    sections += ["## Estimator changes with fixed inputs",
                 "Every non-OLS estimator is compared with OLS using the same inputs, dates, targets, and fitting history. These comparisons do not select the best model using test performance.",
                 contrast_table(main_contrasts[(main_contrasts.comparison == "versus_ols") & (main_contrasts.fit_days == 504)
                                               & (main_contrasts.metric == "rank_ic")])]
    neural_comparisons = main_contrasts[(main_contrasts.comparison == "nn_versus_regularized")
                                        & (main_contrasts.fit_days == 504) & (main_contrasts.metric == "rank_ic")]
    if not neural_comparisons.empty:
        sections += ["### Neural networks versus each regularized linear estimator",
                     "Each NN is paired with ridge, lasso, and elastic net separately using identical inputs and fitting history. All comparisons belong to the declared estimator family; no best linear benchmark is selected after observing test performance.",
                     contrast_table(neural_comparisons, include_benchmark=True)]
    sections += ["## Fitting-history sensitivity",
                 "Validation remains fixed at 126 sessions. The table shows text + core IC for all registered histories; the paired changes compare F252/F756 with F504 on the same evaluation sample."]
    history_levels = summary[(summary.metric == "rank_ic") & (summary.feature_set == "textcore")]
    sections.append(matrix(history_levels, ["target", "estimator"], "fit_days", [252, 504, 756]))
    history = main_contrasts[(main_contrasts.family == "history") & (main_contrasts.feature_set == "textcore")
                             & (main_contrasts.metric == "rank_ic")]
    sections.append(contrast_table(history, include_feature=False, include_fit=True))
    sections += ["## Gross portfolio diagnostics",
                 "Values are average daily top-minus-bottom decile returns in basis points. Equal-weighted and capitalization-weighted sorts require at least ten fractional security memberships per decile. Ties share their rank intervals; constant forecasts hold cash. Capitalization comes from `"+metadata["capitalization"]+"`. These are descriptive gross sorts, with no claim of attainable same-close execution, turnover-adjusted profitability, costs, or factor alpha."]
    for metric, title_metric in [("ew_spread_bp", "Equal-weighted"), ("cap_spread_bp", "Capitalization-weighted")]:
        sections.append("### " + title_metric)
        subset = summary[(summary.metric == metric) & (summary.fit_days == 504)]
        sections.append(matrix(subset, ["target", "estimator"], "feature_set", FEATURES, digits=2))
    sections.append("The decile artifact includes raw and date-demeaned legs separately. Demeaning uses the unweighted eligible-stock mean within that scope and day; cash remains zero. Paired spread confidence intervals and adjusted p-values are in the contrast artifact, rather than inferred from differences between individual-model t-statistics.")
    sections += ["## Stability and composition",
                 "The compact tables below use text + core at F504 without selecting it based on test results. All registered models' yearly and subperiod comparisons remain in the linked artifacts."]
    yearly = read_csv(prefix, "_yearly")
    periods = read_csv(prefix, "_periods")
    if not periods.empty:
        subset = periods[(periods.metric == "rank_ic") & (periods.fit_days == 504) & (periods.feature_set == "textcore")]
        sections.append(matrix(subset, ["target", "estimator"], "period", ["full", "2014-2018", "2019-2022"]))
    if not yearly.empty:
        subset = yearly[(yearly.metric == "rank_ic") & (yearly.fit_days == 504) & (yearly.feature_set == "textcore")]
        for target in models.target.drop_duplicates():
            sections.append("### Yearly text + core IC: " + TARGET_LABELS.get(target, target))
            sections.append(matrix(subset[subset.target == target], "year", "estimator"))
    subgroups = read_csv(prefix, "_subgroups")
    if not subgroups.empty:
        subset = subgroups[(subgroups.metric == "rank_ic") & (subgroups.feature_set == "textcore")]
        sections.append("### Size and activity gradients")
        sections.append(matrix(subset, ["target", "estimator"], "subgroup"))
        sections.append("Tercile 1 is low and tercile 3 is high. Groups use contemporaneous midranks before outcome filtering; identical values remain together. These subgroup means describe composition and do not establish a causal mechanism.")
    horizons = read_csv(prefix, "_horizons")
    if not horizons.empty:
        subset = horizons[(horizons.metric == "rank_ic") & (horizons.feature_set == "textcore")]
        sections.append("### Cumulative-horizon diagnostics")
        sections.append(matrix(subset, ["target", "estimator"], "horizon", [3, 5, 10, 21, 42, 63]))
        sections.append("The h=1-trained forecasts are evaluated against cumulative close-t-to-close-(t+h) returns. These are not models trained for each horizon or disjoint future return intervals; persistent cumulative IC does not show when a return accrues. DGTW h=21 is excluded because of the known source-data corruption. Longer-horizon summaries use at least h−1 HAC lags.")
    sections += ["## Numerical diagnostics", numerical_summary(models, out), "## Scope and limitations",
                 "This is historical development evidence in the tagged-message common-stock universe. Prior linear results and pilot periods were already inspected; the evaluation is not a newly untouched holdout. It does not establish social media's incremental information beyond stock characteristics, return history, or news, because those conditioning-information experiments remain separate.",
                 "The source routing assumes a nominal 16:00 close and does not repair early-close sessions. On those days some inputs can arrive after the actual close, so these results do not certify an executable forecast at that close. Label availability is assumed at the outcome's session endpoint; publication-delay metadata are unavailable. Longer-horizon imported labels have not received the same full certification as h=1. The source text panel has gaps across September-December 2023, including all October; these dates lie outside the main period.",
                 "Models share prediction and realized-outcome samples within each target family. Raw and DGTW results can use different realized-outcome samples, so differences across those two targets also reflect sample composition.",
                 "Multiplicity corrections cover the registered comparisons shown here, not the project's entire research history. Changing the fitting history changes only the designated historical fitting block; its interpretation should remain separate from input and estimator changes.",
                 "## Reproducibility and full artifacts"]
    artifacts = [("", "Main model statistics"), ("_coverage", "Prediction and outcome coverage"),
                 ("_contrasts", "All paired contrasts, intervals, adjusted p-values, and HAC sensitivities"),
                 ("_daily", "Calendar-aligned daily series"), ("_yearly", "Yearly statistics"),
                 ("_periods", "Subperiod statistics"), ("_deciles", "Raw and demeaned decile profiles"),
                 ("_subgroups", "Size and activity diagnostics"), ("_horizons", "Cumulative-horizon diagnostics")]
    sections.append("\n".join("- " + link(prefix.with_name(prefix.name+suffix+".csv"), out, label)
                               for suffix, label in artifacts if prefix.with_name(prefix.name+suffix+".csv").exists()))
    sections.append("Registry: " + link(registry_path, out) + ". Evaluation metadata: " + link(prefix.with_suffix(".json"), out)
                    + ". Prepared data manifest: " + link(prepared/"manifest.json", out) + ".")
    sections.append("`mean` is mean rank IC or mean spread in bp as named by `metric`; `n` is the usable number of trading dates; paired `mean` is model minus benchmark. Standard errors preserve missing sessions' calendar distances. Inputs, predictions, configuration identifiers, and code hashes are recorded in the manifests. Prediction caches remain under `.runs`.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    print(f"Saved {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True, help="Evaluation filename prefix")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    return build_report(args.registry, args.evaluation, args.out, args.title, args.allow_pilot)


if __name__ == "__main__":
    main()
