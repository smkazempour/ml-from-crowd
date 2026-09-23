"""Write the full linear-optimization audit and its plain-language companion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from linear_optimization_evaluate import (DESIGN_FIELDS, FAMILY_DESCRIPTIONS,
    PLANNED_FAMILY_COUNTS, PLANNED_MODELS, file_hash, registered_family_counts, validate_matrix)


FEATURE_LABELS = {"characteristics": "Characteristics", "characteristics_core": "Characteristics + sentiment/attention",
                  "characteristics_textcore": "Characteristics + sentiment/attention + text"}
ESTIMATOR_LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net", "group_ridge": "Group ridge"}
FAMILY_LABELS = {"penalty_expansion": "Broader penalty choices", "transformation": "Daily characteristic ranks",
    "pca_expansion": "Allowing smaller text summaries", "compression": "Compressing text",
    "incremental_social": "Adding social-media inputs", "group_penalty": "Separate penalties for input groups",
    "estimator": "Penalization versus OLS"}
COMPARISON_LABELS = {"expanded_minus_original_penalties": "Expanded minus original penalties",
    "daily_rank_minus_standard": "Daily ranks minus standardization", "expanded_minus_original_pca": "Expanded minus original PCA menu",
    "pca_minus_full_text": "PCA minus full text", "core_given_characteristics": "Sentiment/attention given characteristics",
    "group_core_minus_ridge_characteristics": "Grouped sentiment/attention + characteristics minus ridge characteristics",
    "full_text_given_core": "Full text given characteristics + sentiment/attention",
    "pca_text_given_core": "PCA text given characteristics + sentiment/attention",
    "group_minus_global_ridge": "Group penalties minus global ridge", "versus_ols": "Penalized minus OLS"}


def number(value, digits=4):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "--"
    return f"{0. if abs(value) < .5 * 10 ** (-digits) else float(value):.{digits}f}"


def pvalue(value):
    return "--" if pd.isna(value) else ("<0.001" if value < .001 else number(value, 3))


def table(frame):
    if frame.empty:
        return "No observations available."
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    rows = [[clean(c) for c in frame.columns], ["---"] * len(frame.columns)]
    rows += [[clean(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def link(path, output, label=None):
    relative = os.path.relpath(Path(path).resolve(), Path(output).resolve().parent).replace("\\", "/")
    return f"[{label or Path(path).name}](<{relative}>)"


def read_csv(prefix, suffix=""):
    try:
        return pd.read_csv(prefix.with_name(prefix.name + suffix + ".csv"))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def procedure_label(model):
    return (f"{ESTIMATOR_LABELS[model['estimator']]}: {FEATURE_LABELS[model['feature_set']]}"
            f"; {model['representation']}; {model['transform']}; penalties {model['penalty_grid']}"
            + (f"; PCA menu {model['pca_grid']}" if model['representation'] == "pca" else ""))


def attach_models(contrasts, models):
    if contrasts.empty:
        return contrasts
    result = contrasts.merge(models[["model_id"] + list(DESIGN_FIELDS)], on="model_id", how="left", validate="many_to_one")
    if result.estimator.isna().any() or not result.benchmark_id.isin(models.model_id).all():
        raise ValueError("Contrast refers to an unknown model")
    keys = ["model_id", "benchmark_id", "family", "comparison", "metric", "target", "period"]
    primary = result[result.hac_lags == 5].copy()
    for lag in (21, 63):
        sensitivity = result[result.hac_lags == lag][keys + ["p_bonferroni"]]
        primary = primary.merge(sensitivity.rename(columns={"p_bonferroni": f"p_bonferroni_hac{lag}"}),
                                on=keys, how="left", validate="one_to_one")
    return primary


def contrast_table(frame):
    if frame.empty:
        return "No registered comparisons available."
    result = pd.DataFrame({"Procedure": frame.apply(procedure_label, axis=1),
                           "Comparison": frame.comparison.map(COMPARISON_LABELS),
                           "Mean difference": frame["mean"].map(number),
                           "Pointwise 95% CI": [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)],
                           "Adjusted p, HAC5": frame.p_bonferroni.map(pvalue),
                           "Adjusted p, HAC21": frame.p_bonferroni_hac21.map(pvalue),
                           "Adjusted p, HAC63": frame.p_bonferroni_hac63.map(pvalue)})
    return table(result)


def family_summary(raw):
    rows = []
    for family in FAMILY_LABELS:
        frame = raw[raw.family == family] if not raw.empty else pd.DataFrame()
        finite = frame[np.isfinite(frame["mean"])] if len(frame) else frame
        significant = finite.p_bonferroni < .05 if len(finite) else pd.Series(dtype=bool)
        rows.append({"Change": FAMILY_LABELS[family], "Comparisons scored": len(finite),
                     "Positive mean difference": int((finite["mean"] > 0).sum()) if len(finite) else 0,
                     "Positive, adjusted p < .05": int((significant & (finite["mean"] > 0)).sum()) if len(finite) else 0,
                     "Negative, adjusted p < .05": int((significant & (finite["mean"] < 0)).sum()) if len(finite) else 0})
    return pd.DataFrame(rows)


def stability_summary(metadata):
    artifact = metadata.get("monthly_selection")
    if not artifact:
        return "The registry has no monthly selection artifact; validation-stability diagnostics are unavailable."
    path = Path(artifact["path"])
    if file_hash(path) != artifact["sha256"]:
        raise ValueError("Monthly selection artifact changed after evaluation")
    frame = pd.read_csv(path)
    candidates = [c for c in frame if any(token in c.lower() for token in ("loo_", "leave_one", "stability", "winner_gap", "top2_gap"))]
    if not candidates:
        return "The monthly selection artifact is recorded, but it contains no recognized leave-one-month-out diagnostics. No stability conclusion is inferred."
    numeric = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values[np.isfinite(values)]
        if len(valid):
            numeric.append({"Diagnostic": column, "Defined selections": len(valid),
                            "Median": number(valid.median()), "Minimum": number(valid.min()), "Maximum": number(valid.max())})
    return ("These diagnostics reuse fitted validation predictions while omitting one validation month from the scoring criterion at a time. They do not refit on omitted-month training samples, use test outcomes, or change the selected forecasting procedure. Identical or nearly tied candidates can produce unstable labels even when predictions barely change.\n\n"
            + (table(pd.DataFrame(numeric)) if numeric else "Diagnostics are present but no numeric summary is available."))


def verify_artifacts(registry_path, prefix, allow_pilot):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "linear_optimization_v3_evaluation":
        raise ValueError("Report requires the linear-optimization evaluator")
    if metadata.get("pilot") and not allow_pilot:
        raise ValueError("Pilot reports require --allow-pilot")
    validate_matrix(registry["models"], bool(metadata.get("pilot")))
    if file_hash(registry_path) != metadata.get("registry_sha256"):
        raise ValueError("Registry changed after evaluation")
    if registered_family_counts(registry) != metadata.get("planned_family_counts"):
        raise ValueError("Evaluation changed the predeclared comparison budgets")
    filenames = {Path(info["path"]).name for info in metadata["output_files"].values()}
    if filenames != set(metadata.get("output_sha256", {})):
        raise ValueError("Evaluation artifact digest inventory incomplete")
    for filename, digest in metadata["output_sha256"].items():
        if file_hash(prefix.parent / filename) != digest:
            raise ValueError(f"Evaluation artifact changed: {filename}")
    prepared = Path(metadata["prepared"])
    if file_hash(prepared / "manifest.json") != metadata["prepared_sha256"]["manifest.json"]:
        raise ValueError("Prepared manifest changed after evaluation")
    for model in registry["models"]:
        source = Path(model["predictions"])
        if not source.is_absolute():
            source = registry_path.resolve().parent / source
        if source.resolve() != Path(metadata["prediction_sources"][model["model_id"]]["path"]).resolve():
            raise ValueError("Registry prediction source differs from evaluated source")
    summary = read_csv(prefix)
    if set(summary.model_id) != {m["model_id"] for m in registry["models"]}:
        raise ValueError("Summary and registry model IDs differ")
    return registry, metadata, summary


def build_report(registry_path, prefix, output, allow_pilot=False, summary_output=None):
    registry_path, prefix, output = Path(registry_path), Path(prefix), Path(output)
    summary_output = Path(summary_output) if summary_output else output.with_name(output.stem + "_summary.md")
    registry, metadata, summary = verify_artifacts(registry_path, prefix, allow_pilot)
    models = pd.DataFrame(registry["models"])
    contrasts = attach_models(read_csv(prefix, "_contrasts"), models)
    if contrasts.empty:
        raw = contrasts
    else:
        raw = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic") & (contrasts.period == "full")]
    status = ("**PLUMBING PILOT: incomplete or artificial inputs; no scientific inference.**" if metadata.get("pilot")
              else f"All {PLANNED_MODELS} registered procedures completed evaluation: 86 for each target.")
    limits = ("The 2014–2022 period has already informed project decisions, so these are development results. Chronological training prevents direct future-label leakage but does not make this an untouched confirmation sample. The tables do not select a winner by its largest test score. "
              "No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim.")
    coverage = pd.DataFrame([{"Target": target, "Models": values["models"], "Eligible stock-days": values["eligible_stock_days"],
                              "Common predictions": values["common_prediction_stock_days"], "Observed outcomes": values["common_scored_stock_days"]}
                             for target, values in metadata["coverage"].items()])
    sections = ["# Linear optimization: penalties, characteristic ranks and text compression", status,
        "This is the complete audit record. " + link(summary_output, output, "The short companion") + " explains the main comparisons without ranking every model.",
        "## Fixed procedure and experiment menu",
        "Every model uses 504 training sessions followed by 126 validation sessions. Settings are selected by mean daily validation Spearman correlation; the final model is refitted on the exact original training and validation rows, preserving the purge. This is 630 fitting sessions, about 2.5 years. Forecasts are updated monthly. Close-t information predicts close-t to close-t+1 returns. The centered daily-rank target, equal-date squared loss and sample are unchanged.",
        "The input sets are the same 17 market/past-return characteristics plus 17 missingness flags (34 columns), those inputs plus sentiment/attention (36), and all of those plus 384 embedding coordinates and two agreement inputs, embed_norm and embed_cos (422). No accounting or news predictors are added.",
        "The complete crossed menu compares historical standardization with daily characteristic ranks; original with expanded penalty choices; full text with PCA text; and the original PCA menu with a menu that additionally permits 4 and 8 components. OLS, ridge, lasso and elastic net cover all input sets; group ridge covers the two social-augmented input sets. OLS has only one penalty-menu label because it has no penalty. Only text models have a PCA-menu choice. These deductions leave 86 distinct procedures per target, 172 total.",
        "Daily ranks use contemporaneously available characteristic values on the formation date. Binary missingness flags and embedding coordinates are not cross-sectionally ranked. Estimation transforms, scaling and PCA use only permitted fitting inputs and are rebuilt for the final refit. PCA compresses only the embedding coordinates; the two agreement inputs remain separate and in the text penalty group. Sparse refits preserve selected penalty fractions while recomputing their numerical scale from the final fitting sample.",
        "## Common sample", table(coverage),
        "Every procedure within a target uses the joint finite-prediction intersection before outcome filtering. The social-covered stock-day universe is unchanged; the characteristic-only comparator is not an all-stock benchmark. Prediction batches limit memory use without changing that global intersection or reducing forecast precision.",
        "## Summary of the primary comparisons", table(family_summary(raw)),
        "Positive counts describe paired procedure differences, not independent replications. Significance counts use two-sided, family-adjusted HAC5 p-values below 0.05. Undefined statistics are omitted from the scored-count column but never reduce the correction budget. No family is called a winner from its count.",
        "## Multiplicity and inference",
        "All comparisons were declared before scoring. Bonferroni correction uses the full family budget separately within target, metric and period. HAC5 is primary; HAC21 and HAC63 are sensitivities. Confidence intervals are pointwise 95% intervals, not simultaneous intervals. These corrections do not account for repeated historical research decisions.",
        table(pd.DataFrame([{"Family": FAMILY_LABELS[f], "Comparisons per target/metric/period": n}
                            for f, n in PLANNED_FAMILY_COUNTS.items()])),
        "The characteristic-only reference for grouped sentiment/attention is global ridge; that comparison combines the social addition with freedom to penalize groups separately. PCA text compares with the corresponding full-text procedure and with the corresponding smaller input set. Original and expanded penalized procedures share the same OLS comparator because the penalty axis is inapplicable to OLS.",
        "## Rank IC and gross portfolio levels",
        "Rows follow the registry order rather than realized performance. Rank IC is the mean daily Spearman correlation. Gross portfolio spreads are mean top-minus-bottom decile returns in basis points with fractional ties and the same daily stock sample. They exclude transaction costs, turnover, factor alpha and execution constraints; they are not attainable net strategy returns."]
    for target in ("raw", "dgtw"):
        local = summary[summary.target == target]
        if local.empty:
            continue
        wide = local.pivot(index="model_id", columns="metric", values="mean")
        order = models.loc[models.target == target, "model_id"]
        wide = wide.reindex(order)
        labels = models.set_index("model_id").loc[order].apply(procedure_label, axis=1)
        display = pd.DataFrame({"Procedure": labels, "Rank IC": wide.rank_ic.map(number),
                                "Equal-weight spread (bp)": wide.ew_spread_bp.map(lambda x: number(x, 2)),
                                "Cap-weight spread (bp)": wide.cap_spread_bp.map(lambda x: number(x, 2))})
        sections += ["### " + ("Raw returns (primary)" if target == "raw" else "DGTW returns (secondary)"), table(display)]
    sections += ["## All primary raw-return paired comparisons"]
    for family in FAMILY_LABELS:
        frame = raw[raw.family == family] if not raw.empty else raw
        sections += ["### " + FAMILY_LABELS[family], FAMILY_DESCRIPTIONS[family] + ".", contrast_table(frame)]
    sections += ["## Stability across the two fixed subperiods",
                 "These comparisons are descriptive checks in 2014–2018 and 2019–2022. Subperiods do not choose a model or a trading regime. Complete numerical results for every family, target, metric and HAC lag remain in the contrast artifact."]
    if not contrasts.empty:
        for period in ("2014-2018", "2019-2022"):
            local = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic") & (contrasts.period == period)]
            sections += ["### " + period, table(family_summary(local))]
    sections += ["## Validation selection stability", stability_summary(metadata),
                 "## Interpretation limits", limits,
                 "Market-based controls and past returns do not establish incremental information beyond accounting fundamentals or news. The inherited DGTW target remains secondary because upstream accounting availability has not been resolved. A failure to detect text gains does not establish that text contains no predictive information. No neural networks are fitted in this study.",
                 "## Reproducible artifacts",
                 "The same economic-key alignment, scoring, fractional tie handling and calendar-aware HAC functions used in earlier reports are reused unchanged. SHA-256 hashes cover the registry, prepared artifacts, predictions, kernels and evaluation outputs. The large daily table is published as deterministic gzip.",
                 "\n".join("- " + link(info["path"], output, name) for name, info in metadata["output_files"].items()),
                 "- " + link(prefix.with_suffix(".json"), output, "Evaluation metadata and hashes"),
                 "- " + link(registry_path, output, "Model registry")]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    digest = ["# Linear optimization: what the results tell us", status,
        "This study asks whether more suitable penalties, daily characteristic ranks, or stronger text compression improve linear prediction. The fitting schedule stays fixed at two years for training, six months for validation, then a refit on both. "
        "It separately asks whether sentiment/attention or text adds information beyond the stock characteristics.",
        "The following table summarizes all registered comparisons using the primary next-day raw-return ranking score. A positive difference means that the named change helped in the paired comparison. The final two columns require a family-adjusted p-value below 0.05; they do not count every small numerical difference as evidence.",
        table(family_summary(raw)),
        "These comparisons overlap substantially: many share inputs, forecasts and benchmarks. Their counts are useful for reading the pattern, not as independent votes for a design. The full report provides the size of each effect, uncertainty intervals and sensitivity to longer HAC lags."]
    if not raw.empty:
        for comparison, label in (("core_given_characteristics", "Sentiment and attention"),
                                  ("full_text_given_core", "Full text"), ("pca_text_given_core", "Compressed text")):
            local = raw[raw.comparison == comparison]
            if not len(local):
                continue
            positive = local[(local["mean"] > 0) & (local.p_bonferroni < .05)]
            robust = positive[(positive.p_bonferroni_hac21 < .05) & (positive.p_bonferroni_hac63 < .05)]
            digest.append(f"**{label}:** {len(positive)} of {len(local)} registered comparisons have a positive increment passing the primary adjusted test; {len(robust)} also pass both longer-lag sensitivity checks. "
                          + ("This is evidence for some specified procedures, not proof of a universally superior input set." if len(positive)
                             else "This study does not establish a positive increment under that primary criterion; it does not prove the inputs contain no information."))
    digest += ["Improving the overall score and demonstrating incremental social-media information are different outcomes. A better text procedure that only catches up with its no-text comparator establishes neither a unique text contribution nor a globally optimal model.",
               limits,
               "The same checks also report equal-weighted and capitalization-weighted gross portfolio spreads and the 2014–2018 versus 2019–2022 subperiods. A ranking improvement alone does not establish an improvement in portfolio returns after costs.",
               link(output, summary_output, "Full results, validation diagnostics and audit record")]
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text("\n\n".join(digest) + "\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--evaluation-prefix", "--evaluation", dest="evaluation", required=True, type=Path)
    parser.add_argument("--output", "--out", dest="output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(args.registry, args.evaluation, args.output, args.allow_pilot, args.summary_output)
    print(f"LINEAR OPTIMIZATION REPORT COMPLETE: {result}", flush=True)
    return result


if __name__ == "__main__":
    main()
