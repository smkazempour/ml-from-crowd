"""Write Stage A's complete interaction register and short question-based digest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from linear_interaction_evaluate import (BASIS_COLUMNS, DESIGN_FIELDS, FAMILY_DESCRIPTIONS,
    PLANNED_FAMILY_COUNTS, PLANNED_MODELS, file_hash, registered_family_counts, social_terms, validate_matrix)

BASIS_LABELS = {
    "additive_c": "C main effects", "additive_cs": "Additive C + S", "additive_cst_full": "Additive C + S + full text",
    "c_squares": "C plus C squares", "c_quadratic": "C quadratic", "cq_s": "C quadratic + S main effects",
    "cq_s_squares": "C quadratic + S main effects/squares", "cs_joint": "Joint C + S interactions",
    "text_main": "Joint C + S plus text main effects", "text_squares": "Above plus text squares",
    "agreement_c": "Above plus agreement x C", "text_c": "Above plus embedding PCs x C",
    "text_s": "Above plus text x S", "single_social192": "One named social interaction"}
ESTIMATOR_LABELS = {"ols": "OLS", "ridge": "Ridge", "enet": "Elastic net"}
FAMILY_LABELS = {"characteristic_shape": "Nonlinearity in stock characteristics", "social_information": "Added sentiment/attention information",
    "social_shape": "Social curvature and joint interactions", "individual_social": "All 35 individual social interactions",
    "text_information": "Added text information", "text_shape": "Text curvature and interaction blocks",
    "estimator": "Estimator comparisons on the same basis", "additive_reference": "Comparison with additive reference procedures"}
COMPARISON_LABELS = {
    "c_squares_minus_c": "C squares beyond C main effects", "c_pairs_given_squares": "C pairs beyond C squares",
    "c_quadratic_minus_c": "Full C quadratic versus C main effects", "additive_social_given_c": "Social main effects given additive C",
    "social_main_given_quadratic_c": "Social main effects given quadratic C", "social_curvature_given_quadratic_c": "Social main effects/squares given quadratic C",
    "social_joint_given_quadratic_c": "Joint social model given quadratic C", "social_squares_given_main": "Social squares beyond social main effects",
    "social_products_given_squares": "All 35 social products beyond social squares", "social_joint_minus_main": "Social squares/products beyond social main effects",
    "social_joint_minus_additive_cs": "Joint social model versus additive C+S", "full_text_given_additive_cs": "Full text given additive C+S",
    "text_squares_given_main": "Text squares beyond text main effects", "agreement_c_given_text_squares": "Agreement x C beyond text squares",
    "embedding_c_given_agreement_c": "Embedding PCs x C beyond agreement x C", "all_text_c_given_text_squares": "All text x C beyond text squares",
    "text_s_given_text_c": "Text x S beyond text x C", "single_social_product_given_squares": "Single product beyond the 191-column benchmark",
    "versus_ols": "Penalized estimator versus OLS", "enet_versus_ridge": "Elastic net versus ridge"}
for _basis in ("text_main", "text_squares", "agreement_c", "text_c", "text_s"):
    COMPARISON_LABELS[_basis + "_given_social_joint"] = BASIS_LABELS[_basis] + " versus no text"
    COMPARISON_LABELS[_basis + "_minus_additive_full_text"] = BASIS_LABELS[_basis] + " versus additive full text"


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


def term_label(term):
    return str(term).replace("net_sentiment", "Sentiment").replace("log_volume", "Attention").replace("__x__", " x ").replace("char_", "")


def procedure_label(model):
    label = BASIS_LABELS[model["basis"]]
    if model["basis"] == "single_social192":
        label += ": " + term_label(model["interaction_term"])
    return ESTIMATOR_LABELS[model["estimator"]] + ": " + label


def attach_models(contrasts, models):
    if contrasts.empty:
        raise ValueError("No registered contrast results available")
    result = contrasts.merge(models[["model_id"] + list(DESIGN_FIELDS)], on="model_id", how="left", validate="many_to_one")
    if result.estimator.isna().any() or not result.benchmark_id.isin(models.model_id).all():
        raise ValueError("Contrast refers to an unknown procedure")
    keys = ["model_id", "benchmark_id", "family", "comparison", "metric", "target", "period"]
    main = result[result.hac_lags == 5].copy()
    for lag in (21, 63):
        sensitivity = result[result.hac_lags == lag][keys + ["p_bonferroni"]]
        main = main.merge(sensitivity.rename(columns={"p_bonferroni": f"p_bonferroni_hac{lag}"}),
                          on=keys, how="left", validate="one_to_one")
    return main


def contrast_table(frame):
    if frame.empty:
        return "No registered comparisons available."
    display = pd.DataFrame({"Procedure": frame.apply(procedure_label, axis=1),
        "Comparison": frame.comparison.map(COMPARISON_LABELS), "Delta IC": frame["mean"].map(number),
        "Pointwise 95% CI": [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)],
        "Adjusted p, HAC5": frame.p_bonferroni.map(pvalue), "Adjusted p, HAC21": frame.p_bonferroni_hac21.map(pvalue),
        "Adjusted p, HAC63": frame.p_bonferroni_hac63.map(pvalue)})
    return table(display)


def individual_results(contrasts, target="raw"):
    frame = contrasts[(contrasts.family == "individual_social") & (contrasts.metric == "rank_ic") & (contrasts.target == target)]
    result = pd.DataFrame({"interaction_term": list(social_terms())})
    for period, suffix in (("full", ""), ("2014-2018", "_early"), ("2019-2022", "_late")):
        columns = ["interaction_term", "mean", "p_bonferroni"]
        if period == "full":
            columns += ["ci_low", "ci_high", "p_bonferroni_hac21", "p_bonferroni_hac63"]
        local = frame[frame.period == period][columns]
        result = result.merge(local.rename(columns={c: c + suffix for c in columns if c != "interaction_term"}),
                              on="interaction_term", how="left", validate="one_to_one")
    return result


def individual_table(frame):
    return table(pd.DataFrame({"Term added to 191-column ridge": frame.interaction_term.map(term_label),
        "Delta IC": frame["mean"].map(number), "Pointwise 95% CI": [f"[{number(lo)}, {number(hi)}]" for lo, hi in zip(frame.ci_low, frame.ci_high)],
        "Adjusted p, HAC5": frame.p_bonferroni.map(pvalue), "Adjusted p, HAC21": frame.p_bonferroni_hac21.map(pvalue),
        "Adjusted p, HAC63": frame.p_bonferroni_hac63.map(pvalue),
        "2014-18 delta": frame.mean_early.map(number), "2014-18 adjusted p": frame.p_bonferroni_early.map(pvalue),
        "2019-22 delta": frame.mean_late.map(number), "2019-22 adjusted p": frame.p_bonferroni_late.map(pvalue)}))


def family_summary(frame):
    rows = []
    for family, label in FAMILY_LABELS.items():
        local = frame[(frame.family == family) & np.isfinite(frame["mean"])]
        rows.append({"Question": label, "Comparisons scored": len(local), "Positive mean difference": int((local["mean"] > 0).sum()),
                     "Positive, adjusted p < .05": int(((local["mean"] > 0) & (local.p_bonferroni < .05)).sum()),
                     "Negative, adjusted p < .05": int(((local["mean"] < 0) & (local.p_bonferroni < .05)).sum())})
    return pd.DataFrame(rows)


def verify_artifacts(registry_path, prefix, allow_pilot=False):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "linear_interactions_v1_evaluation":
        raise ValueError("Report requires the Stage A interaction evaluator")
    if metadata.get("pilot") and not allow_pilot:
        raise ValueError("Pilot reports require --allow-pilot")
    validate_matrix(registry["models"], bool(metadata.get("pilot")))
    if file_hash(registry_path) != metadata.get("registry_sha256"):
        raise ValueError("Registry changed after evaluation")
    if registered_family_counts(registry) != metadata.get("planned_family_counts"):
        raise ValueError("Evaluation changed the predeclared comparison budgets")
    expected_files = {Path(info["path"]).name for info in metadata["output_files"].values()}
    if expected_files != set(metadata.get("output_sha256", {})):
        raise ValueError("Evaluation artifact digest inventory is incomplete")
    for filename, digest in metadata["output_sha256"].items():
        if file_hash(prefix.parent / filename) != digest:
            raise ValueError(f"Evaluation artifact changed: {filename}")
    prepared = Path(metadata["prepared"])
    if file_hash(prepared / "manifest.json") != metadata["prepared_sha256"]["manifest.json"]:
        raise ValueError("Prepared manifest changed after evaluation")
    selection = metadata.get("monthly_selection")
    if selection and file_hash(selection["path"]) != selection["sha256"]:
        raise ValueError("Monthly selection artifact changed after evaluation")
    for model in registry["models"]:
        source = Path(model["predictions"])
        if not source.is_absolute():
            source = registry_path.resolve().parent / source
        if source.resolve() != Path(metadata["prediction_sources"][model["model_id"]]["path"]).resolve():
            raise ValueError("Registry prediction source differs from evaluated source")
    summary = read_csv(prefix)
    if summary.empty or set(summary.model_id) != {model["model_id"] for model in registry["models"]}:
        raise ValueError("Nonempty summary and registry model IDs must match")
    return registry, metadata, summary


def build_report(registry_path, prefix, output, allow_pilot=False, summary_output=None):
    registry_path, prefix, output = Path(registry_path), Path(prefix), Path(output)
    summary_output = Path(summary_output) if summary_output else output.with_name(output.stem + "_summary.md")
    registry, metadata, summary = verify_artifacts(registry_path, prefix, allow_pilot)
    models = pd.DataFrame(registry["models"])
    contrasts = attach_models(read_csv(prefix, "_contrasts"), models)
    raw = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic") & (contrasts.period == "full")]
    pilot = bool(metadata.get("pilot"))
    status = ("**PLUMBING PILOT: incomplete or artificial inputs; no scientific inference or model-menu selection.**" if pilot else
              f"All {PLANNED_MODELS} registered procedures completed evaluation: 74 per target.")
    limits = ("These are results for the already-inspected 2014-2022 development period, not untouched confirmation. No model is selected by its largest test score. "
        "No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim.")
    coverage = pd.DataFrame([{"Target": target, "Models": c["models"], "Eligible stock-days": c["eligible_stock_days"],
        "Common predictions": c["common_prediction_stock_days"], "Observed outcomes": c["common_scored_stock_days"]} for target, c in metadata["coverage"].items()])
    basis_table = pd.DataFrame([{"Basis": basis, "Meaning": BASIS_LABELS[basis], "Columns": count} for basis, count in BASIS_COLUMNS.items()]
        + [{"Basis": "single_social192", "Meaning": "191-column baseline plus one of 35 named products; ridge only", "Columns": 192}])
    sections = ["# Explicit interactions: linear prediction with nonlinear input terms", status,
        "This full register preserves all planned procedures and comparisons. " + link(summary_output, output, "The short companion") + " answers the main research questions.",
        "## What was fitted", table(basis_table),
        "C denotes 17 continuous market/past-return characteristics plus 17 binary missingness flags. S is net sentiment and attention (log message volume). T denotes 16 embedding principal components plus two agreement inputs (embed_norm and embed_cos). Additive full text instead uses all 384 embedding coordinates plus those agreement inputs. OLS, ridge and elastic net cover all 13 joint/reference bases. Another 35 ridge models each add one named social interaction to the same 191-column baseline: 39 + 35 = 74 procedures per target, 148 total.",
        "The 35 named terms are sentiment and attention each multiplied by every continuous characteristic (34 terms), plus sentiment times attention. They are fitted individually and jointly because these answer different conditional prediction questions. All parent main-effect columns remain included. Ordinary elastic net does not enforce nonzero-parent hierarchy.",
        "## Timing, transformations and sample", table(coverage),
        "Each monthly forecast uses 504 initial training sessions and 126 chronological validation sessions. Settings are chosen by mean daily validation Spearman IC. Final coefficients are refitted on the exact original training and validation rows, preserving the maturity purge: 630 sessions, about 2.5 years. Inputs through close t predict close t to close t+1. Daily return-rank targets, squared loss and equal total fitting weight per date are unchanged.",
        "The continuous characteristics and social scalars already use daily ranks from the certified original cache. This study does not compare raw characteristic levels with ranks and does not re-rank previously imputed zeros. Missing characteristic ranks retain neutral zero with separate binary flags. Squares/products are constructed from the declared representation and receive fitting-only scaling; they are not ranked again. Flags remain main effects, with no flag products or duplicate flag squares. PCA is fitted only on permitted fitting inputs, fixed at 16 components, and rebuilt for the union refit.",
        "All procedures within a target are scored on the joint finite-prediction intersection before filtering missing outcomes. The sample consists of stock-days with retained tagged StockTwits messages and embeddings, not the full stock universe. Characteristic-only forecasts use the same conditional sample. Market histories supply past-return controls without adding no-message days.",
        "## Interpretation and inference",
        "Ridge is the primary interaction estimator; OLS and elastic net are references/sensitivities. Rank IC is mean daily Spearman correlation. Paired daily differences use calendar-aware HAC5, with HAC21/HAC63 sensitivities. Confidence intervals are pointwise 95% intervals. Bonferroni correction uses the full declared family separately within each target, metric and period. Missing or undefined statistics never reduce a denominator. There is no across-family significance guarantee or correction for the project's previously inspected development history.",
        table(pd.DataFrame([{"Family": FAMILY_LABELS[f], "Comparisons per target/metric/period": count} for f, count in PLANNED_FAMILY_COUNTS.items()])),
        "Some aggregate and sequential comparisons overlap because they answer distinct conditional questions. Their counts are not independent replications. A predictive improvement or a coefficient sign does not establish a causal mechanism. All 35 individual terms are reported with the same 35-comparison adjustment, rather than displaying selected favorable terms.",
        "The agreement-by-characteristic step adds 34 products. The next step adds 272 embedding-PC-by-characteristic products, separating that block from agreement. The final step adds 36 text-by-social products. PCA coordinates may rotate across months; their coefficients do not identify stable topics. Additive-full-text reference comparisons change both text representation and the characteristic/social basis, so they cannot attribute any difference solely to interactions.",
        "## Full-period raw-return comparison overview", table(family_summary(raw)),
        "## Model levels",
        "Rows follow the registry, not realized performance. Portfolio diagnostics are gross daily top-minus-bottom decile returns in basis points with fractional tie handling. They do not include attainable execution, turnover, transaction costs or factor alpha."]
    for target in ("raw", "dgtw"):
        local = summary[summary.target == target]
        if local.empty:
            continue
        order = models.loc[models.target == target, "model_id"]
        wide = local.pivot(index="model_id", columns="metric", values="mean").reindex(order)
        labels = models.set_index("model_id").loc[order].apply(procedure_label, axis=1)
        display = pd.DataFrame({"Procedure": labels, "Rank IC": wide.rank_ic.map(number),
            "EW spread (bp)": wide.ew_spread_bp.map(lambda value: number(value, 2)),
            "Cap-weight spread (bp)": wide.cap_spread_bp.map(lambda value: number(value, 2))})
        sections += ["### " + ("Raw returns (primary)" if target == "raw" else "DGTW returns (secondary)"), table(display)]
    sections += ["## All paired raw-return block comparisons"]
    for family, title in FAMILY_LABELS.items():
        if family == "individual_social":
            continue
        sections += ["### " + title, FAMILY_DESCRIPTIONS[family] + ".", contrast_table(raw[raw.family == family])]
    sections += ["## Complete register of the 35 individual social terms",
        "Each term is tested against ridge on the same 191-column benchmark, with both procedures tuned separately by validation. The entire 35-term family is adjusted. Early/late differences and adjusted HAC5 p-values are descriptive persistence checks; no subperiod selects terms or a trading regime. Missing pilot comparisons are retained as blank entries."]
    for target in ("raw", "dgtw"):
        if target in set(models.target):
            sections += ["### " + ("Raw returns" if target == "raw" else "DGTW returns"), individual_table(individual_results(contrasts, target))]
    sections += ["## Early/late stability of the registered block comparisons",
                 "Complete metric-specific effects and sensitivities are in the contrast artifact. These subperiods are fixed at 2014-2018 and 2019-2022 and are not independent confirmation samples."]
    for period in ("2014-2018", "2019-2022"):
        local = contrasts[(contrasts.target == "raw") & (contrasts.metric == "rank_ic") & (contrasts.period == period)]
        sections += ["### " + period, table(family_summary(local))]
    sections += ["## Limits and reproducibility", limits,
        "The controls are market-based; no point-in-time accounting or news predictors are added. The inherited DGTW target remains secondary because upstream accounting availability is unresolved. Failure to detect a gain is not proof that the relevant information does not exist. This study fits explicit linear-basis models, not trees or neural networks.",
        "Bounded prediction batches preserve the global common sample and float64 forecasts. Existing alignment, rank/portfolio scoring and HAC kernels are reused unchanged. SHA-256 records cover the registry, predictions, prepared artifacts, reused sources and every evaluation output. The daily table is deterministic gzip.",
        "\n".join("- " + link(info["path"], output, name) for name, info in metadata["output_files"].items()),
        "- " + link(prefix.with_suffix(".json"), output, "Evaluation metadata and hashes"),
        "- " + link(registry_path, output, "Model registry")]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    ridge = raw[raw.estimator == "ridge"]
    questions = [
        ("Can richer stock characteristics help?", "c_quadratic_minus_c"),
        ("Does sentiment/attention add information after richer C controls?", "social_main_given_quadratic_c"),
        ("Do social squares add more?", "social_squares_given_main"),
        ("Do the 35 social interactions help jointly?", "social_products_given_squares"),
        ("Do compressed text main effects help?", "text_main_given_social_joint"),
        ("Do agreement x C interactions help?", "agreement_c_given_text_squares"),
        ("Do embedding-PC x C interactions add more?", "embedding_c_given_agreement_c"),
        ("Do text x sentiment/attention interactions add more?", "text_s_given_text_c")]
    question_rows = []
    for question, comparison in questions:
        local = ridge[ridge.comparison == comparison]
        row = local.iloc[0] if len(local) else {"mean": np.nan, "p_bonferroni": np.nan}
        question_rows.append({"Question (ridge)": question, "Delta rank IC": number(row["mean"]),
                              "Adjusted p": pvalue(row["p_bonferroni"])})
    singles = individual_results(contrasts)
    positive = (singles["mean"] > 0) & (singles.p_bonferroni < .05)
    negative = (singles["mean"] < 0) & (singles.p_bonferroni < .05)
    same_direction = positive & (singles.mean_early > 0) & (singles.mean_late > 0)
    positive_robust = positive & (singles.p_bonferroni_hac21 < .05) & (singles.p_bonferroni_hac63 < .05)
    digest = ["# Explicit interactions: a short reading of the results", status,
        "The study asks whether products such as sentiment times recent return improve prediction. First it allows stock characteristics to have their own curved and interacting relationships, so a social term is not credited merely for filling that gap. Every procedure uses the same message-covered stock-days, two-year training block, six-month validation block and final refit on both.",
        "**What do the joint models add?** The table uses ridge, declared in advance as the primary estimator. Each row compares the named addition with its appropriate smaller model. Positive differences mean better daily stock ranking. Adjusted p-values are two-sided and use the complete family budget; the full report gives effect intervals and longer-lag sensitivity checks.", table(pd.DataFrame(question_rows)),
        "**Do individual social interactions help?** " + ("This pilot checks the reporting interface only; it provides no scientific answer. " if pilot else
            f"Of the 35 prespecified individual additions, {int(positive.sum())} have positive differences passing the adjusted primary test and {int(negative.sum())} have negative differences passing it. "
            f"Among the positive results, {int(positive_robust.sum())} also pass both longer-lag checks; {int(same_direction.sum())} have positive mean differences in both early and late periods. " )
            + "All 35 terms, including unfavorable or undefined results, appear in the full register. Positive early/late signs are descriptive persistence, not separate confirmation. A term's fitted coefficient or nonzero selection alone does not establish predictive value.",
        "**How should text gains be interpreted?** The sequence separates agreement interactions, embedding-PC interactions with characteristics, and interactions with sentiment/attention. A gain in one block is conditional on everything already in its benchmark. Monthly PCs can change meaning; the results do not identify a stable PC topic. Comparisons with additive full text also change the representation, so they cannot isolate interactions alone.",
        "**What remains uncertain?** " + limits + " The characteristic inputs already use daily ranks; raw-versus-ranked inputs are not tested here. Gross portfolio spreads are separate diagnostics and do not establish returns after costs. These results do not test trees or neural networks.",
        link(output, summary_output, "Full comparisons, all 35 terms and the audit record")]
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
    print(f"LINEAR INTERACTION REPORT COMPLETE: {result}", flush=True)
    return result


if __name__ == "__main__":
    main()
