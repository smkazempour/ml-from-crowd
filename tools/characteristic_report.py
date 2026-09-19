"""Generate report 09 from the stock-characteristic linear experiment artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from characteristic_evaluate import FEATURES, PLANNED_FAMILY_COUNTS, file_hash, validate_matrix


FEATURE_LABELS = {
    "core": "Sentiment + attention",
    "all": "All social",
    "textcore": "Text + sentiment + attention",
    "textall": "Text + all social",
    "characteristics": "C only",
    "characteristics_sentiment": "C + sentiment",
    "characteristics_attention": "C + attention",
    "characteristics_core": "C + sentiment + attention",
    "characteristics_all": "C + all social",
    "characteristics_textcore": "C + text + sentiment + attention",
    "characteristics_textall": "C + text + all social",
}
ESTIMATOR_LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
TARGET_LABELS = {"raw": "Raw (primary)", "dgtw": "DGTW (secondary)"}
COMPARISON_LABELS = {
    "sentiment_given_characteristics": "Sentiment given C",
    "attention_given_characteristics": "Attention given C",
    "core_given_characteristics": "Sentiment + attention given C",
    "attention_given_characteristics_sentiment": "Attention given C + sentiment",
    "sentiment_given_characteristics_attention": "Sentiment given C + attention",
    "engineered_given_characteristics_core": "Other social given C + sentiment + attention",
    "text_given_characteristics_core": "Text given C + sentiment + attention",
    "text_given_characteristics_all": "Text given C + all social",
    "engineered_given_characteristics_textcore": "Other social given C + text + sentiment + attention",
    "all_given_characteristics": "All social given C",
    "textcore_given_characteristics": "Text + sentiment + attention given C",
    "textall_given_characteristics": "Text + all social given C",
}


def number(value, digits=4):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "--"
    value = float(value)
    return f"{0.0 if abs(value) < .5 * 10 ** (-digits) else value:.{digits}f}"


def pvalue(value):
    if pd.isna(value):
        return "--"
    return "<0.001" if value < .001 else number(value, 3)


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
    path = prefix.with_name(prefix.name + suffix + ".csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def level_table(frame, digits=4):
    if frame.empty:
        return "No observations available."
    wide = frame.pivot(index="feature_set", columns="estimator", values="mean")
    wide = wide.reindex(index=[f for f in FEATURES if f in wide.index],
                        columns=[e for e in ESTIMATOR_LABELS if e in wide.columns])
    wide = wide.map(lambda x: number(x, digits)).reset_index()
    wide["feature_set"] = wide.feature_set.map(FEATURE_LABELS)
    return table(wide.rename(columns={"feature_set": "Inputs"} | ESTIMATOR_LABELS))


def contrast_table(frame, include_comparison=True):
    if frame.empty:
        return "No registered comparisons available."
    result = pd.DataFrame(index=frame.index)
    if include_comparison:
        result["Addition / comparison"] = frame.comparison.map(lambda x: COMPARISON_LABELS.get(x, x))
    result["Estimator"] = frame.estimator.map(ESTIMATOR_LABELS)
    if not include_comparison:
        result["Inputs"] = frame.feature_set.map(FEATURE_LABELS)
    result["Delta IC"] = frame["mean"].map(number)
    result["95% CI"] = [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)]
    result["HAC t"] = frame.t.map(lambda x: number(x, 2))
    result["Adjusted p (HAC5)"] = frame.p_bonferroni.map(pvalue)
    for lag in (21, 63):
        column = f"p_bonferroni_hac{lag}"
        if column in frame:
            result[f"Adjusted p (HAC{lag})"] = frame[column].map(pvalue)
    return table(result)


def attach_models(contrasts, models):
    if contrasts.empty:
        raise ValueError("Evaluation contains no registered contrasts")
    fields = ["model_id", "estimator", "feature_set", "fit_days"]
    result = contrasts.merge(models[fields], on="model_id", how="left", validate="many_to_one")
    if result.estimator.isna().any():
        raise ValueError("A contrast refers to an unknown model")
    if not result.benchmark_id.isin(models.model_id).all():
        raise ValueError("A contrast refers to an unknown benchmark")
    join = ["model_id", "benchmark_id", "family", "comparison", "metric", "target", "period"]
    main = result[result.hac_lags == 5].copy()
    for lag in (21, 63):
        sensitivity = result[result.hac_lags == lag][join + ["p_bonferroni"]]
        main = main.merge(sensitivity.rename(columns={"p_bonferroni": f"p_bonferroni_hac{lag}"}),
                          on=join, how="left", validate="one_to_one")
    return main


def evidence_summary(frame):
    """Summarize the declared core-versus-C question without selecting a winner."""
    rows = []
    for target in TARGET_LABELS:
        group = frame[(frame.target == target)
                      & (frame.comparison == "core_given_characteristics")]
        available = group.p_bonferroni.notna()
        positive = (group["mean"] > 0) & (group.p_bonferroni < .05)
        negative = (group["mean"] < 0) & (group.p_bonferroni < .05)
        stable = positive & (group.p_bonferroni_hac21 < .05) & (group.p_bonferroni_hac63 < .05)
        rows.append({"Target": TARGET_LABELS[target], "Available estimators": int(available.sum()),
                     "Positive, adjusted p < .05": int(positive.sum()),
                     "Negative, adjusted p < .05": int(negative.sum()),
                     "Positive at all three HAC lags": int(stable.sum())})
    return table(pd.DataFrame(rows))


def build_report(registry_path, prefix, out, allow_pilot=False):
    registry_path, prefix, out = Path(registry_path), Path(prefix), Path(out)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "characteristic_conditioning_evaluation_v1":
        raise ValueError("Report requires the explicit characteristic-conditioning evaluator")
    if metadata.get("pilot") and not allow_pilot:
        raise ValueError("Pilot reports require --allow-pilot")
    records = registry["models"]
    validate_matrix(records, allow_pilot=bool(metadata.get("pilot")))
    if file_hash(registry_path) != metadata.get("registry_sha256"):
        raise ValueError("Registry changed after scoring")
    if metadata.get("planned_family_counts") != PLANNED_FAMILY_COUNTS:
        raise ValueError("Evaluation does not preserve the planned multiplicity families")
    for filename, digest in metadata.get("output_sha256", {}).items():
        if file_hash(prefix.parent / filename) != digest:
            raise ValueError(f"Evaluation artifact changed after scoring: {filename}")
    models = pd.DataFrame(records)
    summary = read_csv(prefix)
    if set(summary.model_id) != set(models.model_id):
        raise ValueError("Summary and registry model IDs differ")
    for record in records:
        source = Path(record["predictions"])
        source = source if source.is_absolute() else registry_path.resolve().parent / source
        if str(source.resolve()) != str(Path(metadata["prediction_sources"][record["model_id"]]["path"]).resolve()):
            raise ValueError("Evaluation prediction source differs from the supplied registry")
    prepared = Path(metadata["prepared"])
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(prepared / "manifest.json") != metadata["prepared_sha256"]["manifest.json"]:
        raise ValueError("Prepared manifest changed after scoring")
    main = attach_models(read_csv(prefix, "_contrasts"), models)
    primary_contrasts = main[(main.period == "full") & (main.fit_days == 504)
                             & (main.metric == "rank_ic")]
    primary = summary[(summary.fit_days == 504) & (summary.metric == "rank_ic")]
    pilot = bool(metadata.get("pilot"))
    interim = bool(metadata.get("preliminary"))
    status = ("**PLUMBING PILOT: these artificial or incomplete inputs do not support scientific inference.**"
              if pilot else (f"**INTERIM: {len(models)} of 264 planned models are included.** All included histories contain the complete 88-model matrix; full-study multiplicity denominators are retained."
                             if interim else "All 264 registered linear models have been evaluated."))
    sections = ["# Stock characteristics and incremental social-media information: linear models",
                status,
                "This experiment asks whether sentiment, attention, engineered social features and text add predictive information after conditioning on observed stock characteristics and past returns. It also shows what adding those controls does to the original social-only specifications. Raw returns are primary; DGTW-adjusted returns are a secondary robustness check.",
                "## Design and common sample",
                "C denotes 17 market and past-return controls plus 17 always-present missingness indicators. Accounting characteristics are excluded because reliable publication/availability dates are unavailable. Missing controls are imputed under the declared preparation rules without dropping the corresponding social-covered stock-days. This is a market-information benchmark, not an exhaustive stock-characteristic or news benchmark.",
                "The signal at close t summarizes messages assigned to that close; h=1 returns run from close t to close t+1. OLS, ridge, lasso and elastic net use the same monthly fits, 126-session validation block, feature preparation, fitting-only scaling and equal-date squared loss on centered return ranks as the parent protocol. Penalties use the existing chronological validation procedure and mean daily Spearman IC. F504 is primary; F252/F756 are fixed sensitivities. Their test results do not select a training window. No new tuning search or neural-network run is part of this experiment."]
    coverage_rows = []
    for target, coverage in metadata["coverage"].items():
        coverage_rows.append({"Target": TARGET_LABELS.get(target, target), "Models": coverage["models"],
                              "Eligible stock-days": f"{coverage['eligible_stock_days']:,}",
                              "Common predictions": f"{coverage['common_prediction_stock_days']:,}",
                              "Observed outcomes": f"{coverage['common_scored_stock_days']:,}"})
    sections.append(table(pd.DataFrame(coverage_rows)))
    feature_counts = []
    for feature in FEATURES:
        columns = manifest.get("feature_sets", {}).get(feature, [])
        feature_counts.append({"Inputs": FEATURE_LABELS[feature], "Feature-set ID": feature,
                               "Columns, including missing flags": len(columns)})
    sections += [table(pd.DataFrame(feature_counts)),
                 "The original social-only controls and all characteristic-conditioned models use the same economic keys and prediction universe. Each target is scored on the joint finite-prediction intersection, then its observed outcomes. Raw and DGTW outcome coverage can differ. Rank IC is the mean daily Spearman correlation; the fitted rank scores are not percentage-return forecasts.",
                 "## Primary F504 rank IC"]
    for target, label in TARGET_LABELS.items():
        sections += ["### " + label, level_table(primary[primary.target == target])]
    incremental = primary_contrasts[primary_contrasts.family == "incremental_social"]
    sections += ["## Does sentiment and attention add information beyond C?",
                 "The prespecified comparison is C + sentiment + attention minus C, within each estimator and fitting history. A positive difference with a small adjusted p-value supports incremental out-of-sample ranking information relative to these observed controls. It does not establish causation or estimate the fraction of social information explained by stock characteristics."]
    if not pilot:
        sections.append(evidence_summary(incremental))
    sections += ["## All prespecified social additions at F504",
                 "Differences are model minus benchmark. Confidence intervals are pointwise 95% intervals. P-values use paired daily differences and full-study Bonferroni adjustment separately within target, metric, family and period. HAC5 is primary; HAC21/HAC63 expose sensitivity to persistence. Missing statistics and unfinished windows do not shrink a family."]
    for target, label in TARGET_LABELS.items():
        sections += ["### " + label, contrast_table(incremental[incremental.target == target])]
    sections.append(table(pd.DataFrame([{"Family": family, "Planned comparisons per target / metric / period": count}
                                        for family, count in PLANNED_FAMILY_COUNTS.items()])))
    sections += ["The 12 incremental-social edges distinguish sentiment, attention, their combination, other engineered social variables, and text. The conditioning family compares C + S with the corresponding original S. There are no blanket comparisons of every new feature set against bare sentiment and attention.",
                 "## What changes when C is added to the original social models?",
                 "These paired comparisons measure the gain from adding C to a fixed social input set. They complement C + S minus C; an improvement in C + S over S alone does not itself establish that social media adds information beyond C."]
    conditioning = primary_contrasts[primary_contrasts.family == "conditioning"]
    for target, label in TARGET_LABELS.items():
        sections += ["### " + label, contrast_table(conditioning[conditioning.target == target], False)]
    sections += ["## Penalized estimators versus OLS at fixed inputs",
                 "Penalization need not outperform OLS in every finite sample, particularly when inputs are few or the signal is weak. This stage changes the conditioning information while retaining the existing tuning procedure. Any later window or penalty-grid redesign must be chosen on chronological validation data rather than these test comparisons."]
    estimator = primary_contrasts[primary_contrasts.family == "estimator"]
    for target, label in TARGET_LABELS.items():
        sections += ["### " + label, contrast_table(estimator[estimator.target == target], False)]
    sections += ["## Fixed training-window sensitivity",
                 "These IC levels describe the three registered histories. F504 remains primary; no best-window result or optimized strategy is reported. The complete contrast artifact includes paired F252/F756 minus F504 differences with the fixed 88-comparison history family."]
    history = summary[summary.metric == "rank_ic"].copy()
    history["Target"] = history.target.map(TARGET_LABELS)
    history["Inputs"] = history.feature_set.map(FEATURE_LABELS)
    history["Estimator"] = history.estimator.map(ESTIMATOR_LABELS)
    wide = history.pivot(index=["Target", "Inputs", "Estimator"], columns="fit_days", values="mean")
    wide = wide.reindex(columns=[w for w in (252, 504, 756) if w in wide.columns])
    sections.append(table(wide.map(number).reset_index().rename(columns={252: "F252", 504: "F504 (primary)", 756: "F756"})))
    sections += ["## Gross portfolio diagnostics",
                 "Average daily top-minus-bottom decile returns are shown in basis points. All eleven input sets are retained. These are gross descriptive sorts with fractional tie handling and matched prediction samples, not implementable net returns. Paired spread differences, confidence intervals, and adjusted p-values are in the contrast artifact."]
    for metric, title in (("ew_spread_bp", "Equal-weighted spread"), ("cap_spread_bp", "Capitalization-weighted spread")):
        for target, label in TARGET_LABELS.items():
            sections += ["### " + title + ": " + label,
                         level_table(summary[(summary.metric == metric) & (summary.fit_days == 504)
                                             & (summary.target == target)], digits=2)]
    periods = main[(main.fit_days == 504) & (main.metric == "rank_ic")
                   & (main.comparison == "core_given_characteristics") & (main.period != "full")]
    sections += ["## Stability of sentiment and attention beyond C",
                 "The same C + sentiment + attention minus C comparison is reported in both fixed subperiods. These are descriptive stability checks, not a rule for selecting when to trade."]
    for target, label in TARGET_LABELS.items():
        for period in ("2014-2018", "2019-2022"):
            subset = periods[(periods.target == target) & (periods.period == period)]
            sections += ["### " + label + ", " + period, contrast_table(subset, False)]
    sections += ["## Scope and limitations",
                 "This remains historical development evidence: the parent models and evaluation years have already been inspected. Multiplicity adjustment covers the fixed comparisons in this experiment, not the project's full research history or repeated interim looks. Failure to reject a zero difference does not establish that social media contains no information.",
                 "The sample consists of the existing social-covered common-stock universe. A C-only benchmark here is not an all-stock benchmark, and the 17 market/past-return variables are not the full stock-characteristic set used in the broader literature. Controls can overlap economically with information conveyed in social media; these predictive comparisons do not identify a causal channel. No news controls are included.",
                 "The raw-return target is primary. Although the input block excludes accounting variables without availability dates, the inherited DGTW outcome uses upstream benchmark groups involving book-to-market information. The publication timing and point-in-time availability of those upstream accounting inputs are not certified here. Secondary DGTW comparisons therefore retain that uncertainty; adding current C inputs does not repair it.",
                 "Source message routing uses a nominal 16:00 market close and does not repair early-close sessions. Some inputs may arrive after the actual close on those dates. Close-t information also does not establish attainable execution at close t. Gross spreads exclude trading costs, turnover and factor-alpha analysis. Outcome timing is inherited from the corrected parent labels; longer-horizon diagnostics have weaker certification than h=1, and the known corrupted DGTW h=21 series remains excluded.",
                 "## Reproducibility and artifacts",
                 "The evaluator reuses the unchanged parent economic-key alignment, scoring, calendar handling and HAC code. Its isolated adapter changes only the registered contrast graph and retains the full planned denominators. The metadata records kernel, wrapper, registry, prepared-data and prediction hashes, together with output hashes. The new data preparation preserves the parent's keys and targets; the preparation manifest records feature provenance and missingness rules."]
    artifacts = [("", "Model statistics"), ("_contrasts", "Paired contrasts and HAC sensitivities"),
                 ("_coverage", "Common prediction and outcome coverage"), ("_daily", "Daily series"),
                 ("_yearly", "Yearly statistics"), ("_periods", "Fixed subperiod statistics"),
                 ("_deciles", "Decile profiles"), ("_subgroups", "Size and activity diagnostics"),
                 ("_horizons", "Cumulative-horizon diagnostics")]
    artifact_links = []
    for suffix, label in artifacts:
        artifact = prefix.with_name(prefix.name + suffix + (".csv.gz" if suffix == "_daily" else ".csv"))
        if artifact.exists():
            artifact_links.append("- " + link(artifact, out, label))
    sections.append("\n".join(artifact_links))
    sections.append("Registry: " + link(registry_path, out) + ". Evaluation metadata: "
                    + link(prefix.with_suffix(".json"), out) + ". Preparation manifest: "
                    + link(prepared / "manifest.json", out) + ".")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    return build_report(args.registry, args.evaluation, args.out, args.allow_pilot)


if __name__ == "__main__":
    main()
