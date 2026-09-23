"""Write report 10 for the prespecified linear-design development experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from linear_design_evaluate import (DESIGN_FIELDS, FAMILY_DESCRIPTIONS, FEATURES,
                                    PLANNED_FAMILY_COUNTS, PLANNED_MODELS,
                                    file_hash, planned_models, validate_matrix)


FEATURE_LABELS = {"characteristics": "C", "characteristics_core": "C + sentiment/attention",
                  "characteristics_textcore": "C + sentiment/attention + text"}
ESTIMATOR_LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso",
                    "enet": "Elastic net", "group_ridge": "Group ridge"}
HISTORY_LABELS = {"fixed504": "Fixed 504", "select": "Validation-selected 252/504/756"}
REFIT_LABELS = {"retain": "Retain old coefficients", "recent": "Refit latest F", "union": "Refit F + 126"}
TARGET_LABELS = {"raw": "Raw returns (primary)", "dgtw": "DGTW returns (secondary)"}
COMPARISON_LABELS = {
    "recent_minus_retain": "Latest F minus retained",
    "union_minus_retain": "F + 126 minus retained",
    "union_minus_recent": "F + 126 minus latest F",
    "selected_minus_fixed504": "Selected window minus fixed 504",
    "group_minus_global_ridge": "Group penalties minus global ridge",
    "pca_minus_full_text": "PCA text minus full text",
    "core_given_characteristics": "Sentiment/attention given C",
    "group_core_minus_ridge_characteristics": "Grouped sentiment/attention + C minus ridge C",
    "full_text_given_core": "Full text given C + sentiment/attention",
    "pca_text_given_core": "PCA text given C + sentiment/attention",
    "versus_ols": "Penalized model minus OLS",
}


def number(value, digits=4):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "--"
    value = float(value)
    return f"{0.0 if abs(value) < .5 * 10 ** (-digits) else value:.{digits}f}"


def pvalue(value):
    return "--" if pd.isna(value) else ("<0.001" if value < .001 else number(value, 3))


def table(frame):
    if frame.empty:
        return "No observations available."
    clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    rows = [[clean(c) for c in frame.columns], ["---"] * len(frame.columns)]
    rows += [[clean(c) for c in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def link(path, out, label=None):
    relative = os.path.relpath(Path(path).resolve(), Path(out).resolve().parent).replace("\\", "/")
    return f"[{label or Path(path).name}](<{relative}>)"


def read_csv(prefix, suffix=""):
    try:
        return pd.read_csv(prefix.with_name(prefix.name + suffix + ".csv"))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def procedure_label(row):
    text = FEATURE_LABELS[row["feature_set"]]
    if row["representation"] == "pca":
        text += " (PCA)"
    return ESTIMATOR_LABELS[row["estimator"]] + ": " + text


def levels(frame, digits=4):
    if frame.empty:
        return "No observations available."
    local = frame.copy()
    local["Procedure"] = local.apply(procedure_label, axis=1)
    wide = local.pivot(index="Procedure", columns="refit_policy", values="mean")
    # Match the declared menu order instead of sorting by realized performance.
    order = list(dict.fromkeys(procedure_label(m) for m in planned_models()))
    wide = wide.reindex(index=[m for m in order if m in wide.index], columns=["retain", "recent", "union"])
    return table(wide.map(lambda x: number(x, digits)).reset_index().rename(columns=REFIT_LABELS))


def attach_models(contrasts, models):
    if contrasts.empty:
        raise ValueError("Evaluation contains no registered contrasts")
    result = contrasts.merge(models[["model_id"] + list(DESIGN_FIELDS)], on="model_id",
                             how="left", validate="many_to_one")
    if result.estimator.isna().any() or not result.benchmark_id.isin(models.model_id).all():
        raise ValueError("A contrast refers to an unknown model or benchmark")
    join = ["model_id", "benchmark_id", "family", "comparison", "metric", "target", "period"]
    main = result[result.hac_lags == 5].copy()
    for lag in (21, 63):
        sensitivity = result[result.hac_lags == lag][join + ["p_bonferroni"]]
        main = main.merge(sensitivity.rename(columns={"p_bonferroni": f"p_bonferroni_hac{lag}"}),
                          on=join, how="left", validate="one_to_one")
    return main


def contrast_table(frame, include_refit=True):
    if frame.empty:
        return "No registered comparisons available."
    result = pd.DataFrame(index=frame.index)
    result["Procedure"] = frame.apply(procedure_label, axis=1)
    if include_refit:
        result["Final fit"] = frame.refit_policy.map(REFIT_LABELS)
    result["Comparison"] = frame.comparison.map(COMPARISON_LABELS)
    result["Delta IC"] = frame["mean"].map(number)
    result["95% CI"] = [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)]
    result["Adjusted p, HAC5"] = frame.p_bonferroni.map(pvalue)
    result["Adjusted p, HAC21"] = frame.p_bonferroni_hac21.map(pvalue)
    result["Adjusted p, HAC63"] = frame.p_bonferroni_hac63.map(pvalue)
    return table(result)


def build_report(registry_path, prefix, out, allow_pilot=False):
    registry_path, prefix, out = Path(registry_path), Path(prefix), Path(out)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "linear_design_v2_evaluation":
        raise ValueError("Report requires the linear-design evaluator")
    if metadata.get("pilot") and not allow_pilot:
        raise ValueError("Pilot reports require --allow-pilot")
    records = registry["models"]
    validate_matrix(records, allow_pilot=bool(metadata.get("pilot")))
    if file_hash(registry_path) != metadata.get("registry_sha256"):
        raise ValueError("Registry changed after scoring")
    if metadata.get("planned_family_counts") != PLANNED_FAMILY_COUNTS:
        raise ValueError("Evaluation changed the registered multiplicity families")
    expected_artifacts = {Path(info["path"]).name for info in metadata["output_files"].values()}
    if set(metadata.get("output_sha256", {})) != expected_artifacts:
        raise ValueError("Evaluation artifact digest inventory is incomplete")
    for filename, digest in metadata["output_sha256"].items():
        if file_hash(prefix.parent / filename) != digest:
            raise ValueError(f"Evaluation artifact changed after scoring: {filename}")
    models = pd.DataFrame(records)
    summary = read_csv(prefix)
    if set(summary.model_id) != set(models.model_id):
        raise ValueError("Summary and registry model IDs differ")
    for record in records:
        source = Path(record["predictions"])
        source = source if source.is_absolute() else registry_path.resolve().parent / source
        if source.resolve() != Path(metadata["prediction_sources"][record["model_id"]]["path"]).resolve():
            raise ValueError("Evaluation prediction source differs from the registry")
    prepared = Path(metadata["prepared"])
    if file_hash(prepared / "manifest.json") != metadata["prepared_sha256"]["manifest.json"]:
        raise ValueError("Prepared manifest changed after scoring")
    contrasts = attach_models(read_csv(prefix, "_contrasts"), models)
    raw = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic") & (contrasts.period == "full")]
    fixed = raw[raw.history_policy == "fixed504"]
    status = ("**PLUMBING PILOT: artificial or incomplete inputs; no scientific inference.**"
              if metadata.get("pilot") else f"All {PLANNED_MODELS} registered linear procedures completed evaluation.")
    sections = ["# Linear design: refitting, group penalties, training history and text compression",
                status,
                "This experiment compares prespecified forecasting procedures on the already-inspected 2014–2022 development period. Its purpose is to diagnose whether the linear design obscures incremental social-media information. It does not identify a globally optimal model from the largest test score. All hyperparameter, window and PCA-dimension choices use earlier validation outcomes; test outcomes are used only for these reported comparisons.",
                "## What changed",
                "C is the same 17 market/past-return characteristics and 17 missingness indicators. The three input sets are C alone (34 columns), C plus sentiment and attention (36), and C plus sentiment, attention and text (422). The latter contains 384 embedding coordinates and two agreement measures, embed_norm and embed_cos. No accounting or news variables are added.",
                "There are 19 base procedures: OLS, ridge, lasso and elastic net on each full input set; group ridge on the two augmented input sets; and all five estimators with PCA text on the text input set. Each is evaluated under two training-history policies and three final-fit rules, separately for raw and DGTW returns: 19 × 2 × 3 × 2 = 228.",
                table(pd.DataFrame([
                    {"Choice": "Fixed history", "Definition": "504 fitting sessions, then 126 validation sessions; monthly forecasts."},
                    {"Choice": "Selected history", "Definition": "Validation jointly chooses 252, 504 or 756 fitting sessions and the estimator's settings."},
                    {"Choice": "Retain", "Definition": "Keep the coefficients estimated before validation, as in the previous study."},
                    {"Choice": "Refit latest F", "Definition": "Keep validation-selected settings, re-estimate using the latest F eligible sessions with matured outcomes."},
                    {"Choice": "Refit F + 126", "Definition": "Keep validation-selected settings, re-estimate on the exact original fitting and validation rows, excluding the original purge. At fixed F504 this is 630 sessions, about 2.5 years."},
                    {"Choice": "Group ridge", "Definition": "Allow different penalties for characteristics, sentiment/attention and text, including the registered exact block-exclusion candidates."},
                    {"Choice": "PCA text", "Definition": "Compress only the 384 embedding coordinates; validation chooses 16, 32, 64 or 128 components. The two agreement measures stay outside PCA and retain the text group's penalty."},
                ])),
                "Signals use information through close t to forecast close t to close t+1. Fitting uses the existing centered-rank target, equal-date squared loss and fitting-only scaling. Validation selects mean daily Spearman IC. Any refit uses only matured labels before the test month. Scaling and PCA are rebuilt on each refit's permitted input observations; the selected dimension and penalty settings stay fixed. For sparse regressions, the selected fraction of the maximum penalty is fixed while its numerical scale is recomputed on the final fitting sample. Test data never enter coefficient fitting or setting selection. Ties retain the declared selection order; no new rule treating close validation scores as equivalent is introduced.",
                "## Common sample and interpretation"]
    coverage = [{"Target": TARGET_LABELS[t], "Models": c["models"],
                 "Eligible stock-days": f"{c['eligible_stock_days']:,}",
                 "Common predictions": f"{c['common_prediction_stock_days']:,}",
                 "Observed outcomes": f"{c['common_scored_stock_days']:,}"}
                for t, c in metadata["coverage"].items()]
    sections += [table(pd.DataFrame(coverage)),
                 "Every procedure within a target is scored on the joint finite-prediction intersection and then observed outcomes. Rank IC is the mean daily Spearman correlation. Predicted rank scores are not percentage-return forecasts. The universe is the existing social-covered stock sample; C-only is not an all-stock benchmark.",
                 "## Rank IC by prescribed procedure",
                 "Rows follow the registered menu, not the ordering of realized test scores. Each row compares the three final-fit rules. In the selected-history tables, F can differ across forecast months and procedures because validation chooses it separately. The F + 126 rule therefore uses 378, 630 or 882 sessions as selected, while retaining the exact original training/validation row sets."]
    for target, target_label in TARGET_LABELS.items():
        for history, history_label in HISTORY_LABELS.items():
            section = summary[(summary.target == target) & (summary.history_policy == history) & (summary.metric == "rank_ic")]
            sections += ["### " + target_label + ": " + history_label, levels(section)]
    sections += ["## Paired raw-return comparisons",
                 "Positive delta IC means the named procedure exceeds its registered benchmark. Confidence intervals are pointwise 95% intervals. P-values compare daily paired differences with calendar-aware HAC, then Bonferroni adjustment over the complete family below, separately within target, metric and period. HAC5 is primary; HAC21 and HAC63 are sensitivities. These corrections do not cover earlier research decisions or repeated inspection of this development period.",
                 table(pd.DataFrame([{"Family": f, "Comparisons per target / metric / period": n}
                                     for f, n in PLANNED_FAMILY_COUNTS.items()])),
                 "The tables below show fixed-504 comparisons for refitting, penalties, compression, social additions and estimators; the history table compares selected against fixed history under every refit. The full contrast artifact also contains all selected-history comparisons, DGTW comparisons, gross-spread differences and the two fixed subperiods. No unfinished or undefined comparison reduces a multiplicity denominator."]
    family_titles = {"refit": "Does re-estimating coefficients help?",
                     "history": "Does validation-selected history help?",
                     "group_penalty": "Do separate penalties help?",
                     "compression": "Does text compression help?",
                     "incremental_social": "What does social media add beyond characteristics?",
                     "estimator": "Do penalized estimators improve on OLS?"}
    for family, title in family_titles.items():
        frame = (raw if family == "history" else fixed)
        frame = frame[frame.family == family]
        sections += ["### " + title, FAMILY_DESCRIPTIONS[family] + "."]
        if family == "incremental_social":
            sections += ["Global estimators compare each augmented input set with the separately validation-tuned smaller procedure. Group C + sentiment/attention versus ridge C combines the social addition with freedom to penalize feature groups separately; it is not a pure coefficient-controlled addition. The two grouped text comparisons use grouped C + sentiment/attention as their reference. Exact exclusion candidates allow the larger search to reproduce smaller input blocks, but validation-selection noise can still lower test performance."]
        sections.append(contrast_table(frame, include_refit=family != "refit"))
    sections += ["## Temporal stability of the incremental social signal",
                 "These prespecified ridge and group-ridge contrasts use fixed 504-session history and retained coefficients in each fixed subperiod. They are descriptive stability checks; no subperiod chooses a model or trading regime. Complete subperiod results for every procedure are in the contrast artifact."]
    for period in ("2014-2018", "2019-2022"):
        frame = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic")
                          & (contrasts.family == "incremental_social") & (contrasts.period == period)
                          & (contrasts.history_policy == "fixed504") & (contrasts.refit_policy == "retain")
                          & contrasts.estimator.isin(["ridge", "group_ridge"])]
        sections += ["### " + period, contrast_table(frame, include_refit=False)]
    sections += ["## Gross portfolio diagnostics",
                 "For raw returns and fixed 504-session history, the following tables show average daily top-minus-bottom decile returns in basis points. These use fractional tie handling and the same common samples. They are gross descriptive sorts, not attainable net trading returns. Turnover, transaction costs, factor alpha and execution constraints have not been estimated."]
    for metric, title in (("ew_spread_bp", "Equal-weighted spread"), ("cap_spread_bp", "Capitalization-weighted spread")):
        frame = summary[(summary.target == "raw") & (summary.history_policy == "fixed504") & (summary.metric == metric)]
        sections += ["### " + title, levels(frame, digits=2)]
    sections += ["## Development evidence and limits",
                 "The 2014–2022 years have already informed project decisions. Chronological fitting prevents direct future-label leakage, but it cannot undo research decisions made after viewing these historical results. These comparisons are development evidence rather than untouched confirmation, and failure to reject an incremental gain does not establish that the text contains no information.",
                 "2023 is not ready to serve as an untouched confirmation sample. The current input audit found 178 of 250 expected trading dates, with 72 missing dates in September–December, including all of October. Earlier project records also mention predictions through 2023. A separate audit of completeness and prior use is required, or additional data must be obtained. No 2023 model outcomes are scored in this experiment.",
                 "Only market-based characteristics and past returns are controlled for. Reliable accounting publication timing is unavailable, so accounting predictors remain excluded. The inherited DGTW target is secondary because its upstream characteristic sorts have their own accounting-availability caveat. No news controls or causal interpretation are supported by these comparisons.",
                 "The test sample does not choose one winning procedure. Any next-stage decision should use the registered paired comparisons, their temporal stability and computational costs, followed by a separately defensible confirmation design. No neural networks are fitted in this experiment.",
                 "## Reproducible artifacts",
                 "Evaluation uses the unchanged scoring, common-sample alignment, calendar and HAC kernels. New scoped adapters supply only the registered comparisons and procedure metadata. SHA-256 digests record the registry, prepared data, prediction files, reused kernels and every evaluation output. The daily table is published as deterministic gzip; the uncompressed copy remains local.",
                 "\n".join("- " + link(info["path"], out, name or "summary") for name, info in metadata["output_files"].items()),
                 "- " + link(prefix.with_suffix(".json"), out, "Evaluation metadata and hashes"),
                 "- " + link(registry_path, out, "Model registry (local run artifact)"),
                 "- " + link(Path(__file__).resolve().parents[1] / "reports" / "LINEAR_DESIGN_EXPERIMENT.md", out, "Frozen experiment specification")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path, help="Evaluation filename prefix")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(args.registry, args.evaluation, args.out, args.allow_pilot)
    print(f"LINEAR DESIGN REPORT COMPLETE: {result}", flush=True)
    return result


if __name__ == "__main__":
    main()
