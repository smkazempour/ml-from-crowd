# Linear design comparison

Authorized September 21, 2026. This experiment follows [Report 09](09_characteristic_linear_results.md)
and tests how estimation choices affect the contribution of social information.
The [machine-readable design](data/linear_design_v2_experiment.json) is fixed before
the new fits. Existing study sources and results remain archived unchanged.

## The decisions being tested

| Choice | Comparison |
| --- | --- |
| Final coefficient estimation | Keep the original fitting-block model; refit on the latest F sessions; refit on the original fitting plus validation samples |
| Training length | Fixed F=504 versus validation choosing F=252, 504, 756 jointly with penalty and representation settings |
| Penalties | Existing OLS/ridge/lasso/elastic net; additional ridge models with separate penalties for characteristics, sentiment/attention and text |
| Text representation | All 384 embedding coordinates versus validation selecting 16, 32, 64 or 128 principal components |

The combined-sample refit is expressly included at the user's request. At F=504,
it uses **504+126=630 sessions**, approximately 2.5 years. The recent-window refit
instead uses the latest 504 sessions. For validation-selected histories, the same
rules give F sessions or F+126 sessions. All final fitting labels must mature before
the test month. The union comprises the original fitting and validation rows,
excluding the original gap between them; it does not silently add gap observations.

Each outer test month still has one 126-session chronological validation block.
Highest validation mean daily Spearman IC selects settings. The proposed near-tie
preference and changes to validation folds are deferred. Fixed-window and
validation-selected-window procedures are evaluated as separate forecast series;
the observed test results do not choose the window for a past forecast.

## Inputs and model menu

Use the same prepared characteristic cache, stock-days, first-close forecast clock,
one-day labels and equal-date rank-target squared loss as Report 09. Raw returns are
primary and DGTW remains secondary with its previously disclosed accounting-timing
limitation. No accounting predictors or new stock filters are added.

The first design study uses three information sets:

- C: 17 market/past-return controls and 17 missingness flags.
- C+S: C plus sentiment and attention.
- C+S+T: C+S plus 384 embedding coordinates and 2 agreement measures.

Global OLS, ridge, lasso and elastic net use the existing penalty grids. Keeping
these grids unchanged isolates the estimation changes. Group ridge is added to
C+S and C+S+T: the base ridge alpha uses the existing grid; the social group receives
a multiplier of 1 or 10 and the text group a multiplier of 1, 10 or 100. Exact group-off
candidates let validation retain a smaller model. These are candidate choices,
not a guarantee that the selected larger procedure improves test performance.

For C+S+T, all five estimators also have a separate PCA procedure. It chooses among
16, 32, 64, 128 components on validation; the full-coordinate procedure remains separate.
PCA fits an equal-date covariance of fitting-scaled embedding coordinates, using
input rows without conditioning on future outcome availability. Component scores
are then scaled using fitting-only statistics. The two agreement measures remain
separate and share the text penalty. Final refits also refit scaling and PCA on
their declared fitting rows. No validation or test observations enter an earlier
fitted transformation.

Selected ridge penalties and group multipliers transfer to the final refit.
Lasso/elastic net transfer the selected alpha fraction and mixing ratio, recomputing
the training-dependent alpha maximum on the final fitting sample.

There are 19 base procedures: 4 for C, 5 for C+S, and 10 for C+S+T (full and PCA).
Two history policies and three refit policies produce 114 models per target,
**228 specifications overall**. One checkpoint shares work across all designs for
a month and target, giving 216 checkpoints over January 2014–December 2022.
Retained F 504 full-input forecasts from the four existing estimators must reproduce
the preceding characteristic study within the declared numerical tolerance.

## Evaluation and interpretation

Use the same keys and observed outcomes within each target, equal-date mean IC,
gross equal/capitalization-weighted decile spreads and paired calendar-aware HAC
inference. HAC 5 is primary; 21 and 63 are sensitivities. Full-period results precede
the existing 2014–2018 and 2019–2022 descriptive subperiods.

Registered Bonferroni families, per target/metric/period, contain 114 refit contrasts,
57 history contrasts, 18 group-penalty contrasts, 30 compression contrasts,
90 incremental-social contrasts and 90 estimator-versus-OLS contrasts. Full-study
denominators apply to any incomplete display. The group-ridge social comparisons
evaluate the addition of inputs together with their allowed group penalties.

All 2014–2022 results are **development evidence**, since earlier results informed
the new design. The goal is to compare prespecified forecasting procedures and
their conditional social increments, not select a test-period winner and treat its
score as fresh confirmation. Additional social feature groups, expanded grids,
alternative losses and nonlinear models remain subsequent experiments.

## Confirmation-data audit

No untouched confirmation sample is certified. A coverage-only check found
309,178 input rows on 178 of 250 exchange sessions in 2023, with 72 missing sessions
across September–December. The missing dates include all October. Earlier Report 01
also describes predictions through 2023. The protocol already documents these gaps
and prior-use concerns. No 2023 model performance was calculated in this audit.
That year requires data repair and a prior-use review before any confirmation claim.
The current run scores only 2014–2022.

## Execution

The controller runs a two-month implementation pilot, then all monthly fits,
prediction verification, evaluation and [Report 10](10_linear_design_results.md).
Large checkpoints and predictions stay under ignored `.runs/linear_design_v2/`.
Daily evaluation results are compressed for publication. See
[run status](LINEAR_DESIGN_RUN_STATUS.md) and the
[execution and recovery guide](../tools/LINEAR_DESIGN_STUDY.md).
