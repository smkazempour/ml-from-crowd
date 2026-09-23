# Linear optimization: transformations, penalties and text compression

This development study implements the follow-up to [Report 10](10_linear_design_results.md).
It compares every meaningful combination of the proposed changes instead of
choosing one configuration in advance. It does not use forecast-period results
to choose penalties or PCA dimensions within a procedure.

## What is fixed

- Forecast next-day returns using the existing close-to-close timing convention.
- Initial training: 504 trading sessions. Validation: the following 126 sessions,
  preserving the existing gaps that ensure labels have matured.
- Select settings using mean daily validation Spearman correlation. Refit selected
  settings on the union of the original training and validation inputs (630 sessions).
- Re-estimate fitted scaling and PCA on the final fitting inputs. Transfer sparse
  penalty fractions and recompute their maximum penalty on those final inputs.
- Monthly updates; daily return-rank targets; squared loss; equal total weight
  per trading day; unpenalized intercept.
- The same eligible stock-days, 2014-2022 forecast months, and raw/DGTW outcomes
  as the completed study. Raw-return ranking remains primary.

## What is crossed

1. **Characteristic transformation:** existing training-based standardization,
   or daily cross-sectional ranks for the 17 continuous market characteristics.
   Rank ties receive average ranks, mapped to [-1,1]. Ranks use all available
   input rows on the formation date, before filtering missing future outcomes.
   Missing values remain missing until fitting-based imputation. Missingness
   flags, sentiment, attention and embeddings are not cross-sectionally ranked.
   Subsequent fitting-based scaling remains common to both treatments.
2. **Penalty menu:** the original search or an expanded, nested search. Add
   intermediate ridge strengths within the original range, intermediate sparse
   penalty fractions, elastic-net mixtures closer to ridge, and text penalty
   multipliers up to 1,000. Exact text/social exclusion candidates remain available.
3. **Text representation:** full embeddings or training-estimated PCA. PCA
   procedures compare the original dimension menu (16,32,64,128) with the
   expanded menu (4,8,16,32,64,128). Two agreement variables remain outside PCA,
   in the text penalty group.
4. **Inputs and estimators:** characteristics alone; add sentiment/attention;
   add text. OLS, ridge, lasso and elastic net apply throughout. Separate group
   ridge also applies when social inputs are present.

No-op duplicates are omitted: OLS has no penalty-menu comparison; models without
PCA have no PCA-menu comparison. This gives **86 procedures per outcome, 172 in
total**, with **216 monthly outcome checkpoints**. These counts describe complete
forecasting procedures, not the larger number of candidate fits used for tuning.

The exact grids, procedure list and comparison-family sizes are recorded in
[the machine-readable declaration](data/linear_optimization_v3_experiment.json).
Original settings remain available in the expanded menus. The original numerical
tie tolerance is retained, with a deterministic candidate order declared in the JSON.

## Validation stability

For each candidate, retain validation IC sums and valid-day counts by calendar
month. Recalculate the winning setting with each validation calendar month
omitted in turn. Partial months count only their observed validation days.
Report how often the winner changes, the score gap to the runner-up, and the
omission-specific settings. These are sensitivity diagnostics: they do not
change the final selection rule or claim statistical equivalence.

## Comparisons and interpretation

Compare expanded versus original penalties, ranked versus standardized market
characteristics, and expanded versus original PCA menus while holding the other
choices fixed. Also compare full versus compressed text, estimators, and each
incremental social-data block. Use the original calendar-aware paired tests,
HAC lags 5/21/63, full/early/late periods, and declared multiple-comparison families.
Evaluate both ranking and gross equal-/capitalization-weighted portfolio spreads.

Distinguish improved estimation from additional information: repairing a text
model's shortfall does not establish that text beats its matched no-text model.
Results remain development evidence because 2014-2022 has already informed these
choices. No untouched confirmation sample is certified; incomplete, previously
examined 2023 data is excluded. No trading-cost claim is made.

## Integrity and outputs

Each monthly checkpoint must reproduce all 19 unchanged standard/original
procedures against the completed study's fixed504/union forecasts, including
group ridge and PCA. Forecast values, daily ranks, source hashes, prepared data,
selected settings and checkpoint coverage are certified. Completed earlier
studies are preserved. Interrupted runs resume validated monthly checkpoints.

The controller first runs an implementation pilot on January 2014 and December
2022 for both outcomes, then the complete study, evaluation, and two reports:

- `11_linear_optimization_results.md`: complete comparisons and audit trail.
- `11a_linear_optimization_summary.md`: concise reading guide.

An independent watcher records and sends a local Windows alert for completion,
failure, or an unexpected controller exit. See [execution instructions](../tools/LINEAR_OPTIMIZATION_STUDY.md).
