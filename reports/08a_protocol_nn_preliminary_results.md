# Preliminary NN3 results: completed 504- and 252-session windows

September 15, 2026. **Text features improve return ranking, while this NN3 specification has not established a full-period advantage over matched linear models.** The relative NN3 results vary substantially over time. These are interim findings over the complete test period; the 756-session NN3 comparison is still running.

## Scope and common procedure

This report covers **16 completed NN3 models and 64 matched linear models**, each with all 108 monthly test blocks in 2014-2022. The 504-session fitting window is primary; 252 sessions is a prespecified sensitivity. All models use 126 validation sessions, the same inputs, rank targets, equal-date loss and chronological selection. NN3 has hidden layers 128, 64 and 32 and averages five seeds after validation selects its penalty and checkpoints.

Every model predicts the same 3,034,035 stock-days. Scoring uses 3,033,080 observed raw returns or 2,748,978 DGTW-adjusted returns, across 2,266 dates. All 756-session models were excluded before scoring, including their matched linear controls. No partially completed NN prediction series enters these comparisons.

## Primary 504-session ranking results

IC is the mean daily Spearman correlation; higher is better. These are rank forecasts, so IC is not return-unit R-squared.

| Target | Inputs | OLS | Ridge | Lasso | Elastic net | NN3 |
| --- | --- | --- | --- | --- | --- | --- |
| Raw | Core | 0.0348 | 0.0346 | 0.0350 | 0.0350 | 0.0341 |
| Raw | All social | 0.0328 | 0.0345 | 0.0334 | 0.0335 | 0.0347 |
| Raw | Text + core | 0.0392 | 0.0404 | 0.0403 | 0.0403 | 0.0400 |
| Raw | Text + all social | 0.0390 | 0.0414 | 0.0401 | 0.0400 | 0.0417 |
| DGTW | Core | 0.0330 | 0.0330 | 0.0336 | 0.0336 | 0.0316 |
| DGTW | All social | 0.0296 | 0.0317 | 0.0307 | 0.0308 | 0.0314 |
| DGTW | Text + core | 0.0349 | 0.0363 | 0.0363 | 0.0363 | 0.0359 |
| DGTW | Text + all social | 0.0341 | 0.0372 | 0.0357 | 0.0357 | 0.0363 |

**NN3 versus linear models.** None of the 32 matched NN3-minus-linear IC comparisons has a positive gain significant under the primary HAC5/full-study Bonferroni rule. With text + all social, raw IC is 0.0417 for NN3 versus 0.0414 for ridge; DGTW IC is 0.0363 versus 0.0372. The raw NN3 gain over OLS for these inputs is +0.00275, but adjusted p = 0.053, above the prespecified 0.05 threshold. Core-only DGTW NN3 underperforms lasso and elastic net by about 0.00195 IC, with adjusted p about 0.0017. These findings do not establish that the models are equivalent, or that other nonlinear specifications cannot help.

**Text additions within NN3.** Both isolated text additions improve IC for both targets:

| NN3 feature addition | Raw IC gain | Raw adjusted p | DGTW IC gain | DGTW adjusted p |
| --- | --- | --- | --- | --- |
| Text + core minus core | +0.0059 | <0.001 | +0.0043 | <0.001 |
| Text + all social minus all social | +0.0070 | <0.001 | +0.0048 | <0.001 |

All four gains remain significant under HAC21 and HAC63. Text + all social beats all social in each of the nine individual years for both targets, a descriptive stability check. Moving from core to the 53 engineered social features without text gives no significant IC improvement. Here, "all social" contains social-media predictors, not stock characteristics; "text" adds 384 embedding coordinates and two agreement measures.

**Portfolio evidence is weaker.** None of the 64 primary NN3-versus-linear equal-weighted or capitalization-weighted spread comparisons shows an adjusted-significant improvement. Text + core actually has lower NN3 equal-weighted spreads than core by 3.30 bp/day for raw returns and 3.06 bp/day for DGTW, although those declines are also insignificant after adjustment. Ranking gains therefore do not establish portfolio gains. These are gross top-minus-bottom decile sorts; execution delays, turnover and costs have not been assessed.

## Time variation and the shorter window

The prespecified subperiods reveal a material difference: all 32 primary NN3-versus-linear IC differences are negative in 2014-2018. In 2019-2022, every noncore comparison turns positive; 10 raw-return and 9 DGTW comparisons pass their within-period adjustment. For text + all social versus ridge, the IC difference changes from -0.0039 to +0.0056 for raw returns and from -0.0047 to +0.0038 for DGTW. This motivates investigating when NN3 helps; it does not justify replacing the full-period test or selecting a new model using its test performance.

With 252 rather than 504 fitting sessions, NN3 IC falls by 0.0036-0.0046 for all three noncore input sets and both targets. All six declines survive adjustment at HAC5, 21 and 63; core-only differences are insignificant. This supports retaining the declared 504-session primary comparison. The 756-session results are needed before drawing a conclusion about the longer window.

## Inference, remaining work and reproducibility

Adjusted p-values count the **full planned 120-model study**: 60 feature, 84 estimator and 40 history contrasts separately by target, metric and period. HAC5 is primary; 21/63 are sensitivities. Report 07 used smaller linear-only families, so adjusted p-values can differ even though **all 192 included linear summary rows reproduce that report exactly**, including means, HAC statistics and confidence intervals. Subperiod/year results are diagnostics, not a new test-sample selection rule.

The evidence currently addresses richer social inputs and this NN3-versus-linear comparison. It does not establish incremental information beyond stock characteristics, return history or news. The sample remains historical development evidence, and nominal 16:00 tweet routing retains the known early-close limitation. Subgroup and longer-horizon diagnostics are deferred to the full report.

Training resumed on September 15 with all 1,752 saved monthly checkpoints reused. After the remaining fits finish, the existing controller will create `08_protocol_nn_results.md` with all 120 models. This dated preliminary report can then be updated or superseded using those full results.

Sources: [summary estimates](data/protocol_v1_1_nn_preliminary.csv), [paired contrasts](data/protocol_v1_1_nn_preliminary_contrasts.csv), [yearly estimates](data/protocol_v1_1_nn_preliminary_yearly.csv), [evaluation metadata](data/protocol_v1_1_nn_preliminary.json), [linear reconciliation](data/protocol_v1_1_nn_preliminary_reconciliation.json), [snapshot/source audit](../.runs/protocol_v1_1/nn_preliminary/20260915T155756561635Z/source_audit.json). Reproduction: [isolated evaluator](../tools/evaluate_protocol_nn_preliminary.py); four focused tests passed. The live training sources were not modified.
