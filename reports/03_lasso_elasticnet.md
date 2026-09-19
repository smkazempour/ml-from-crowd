# Report 03 -- Lasso and elastic net on the text features

> Evaluation update (2026-09-12): these are historical results. See
> [Report 06](06_evaluation_and_nn_pilot.md) for fractional treatment of prediction ties
> and paired Newey-West comparisons with the core benchmark. Prediction files were preserved.

Date: 2026-09-12. Builds on Reports 01-02 (notation as in Report 02, Section 0). Data tables:
`reports/data/lasso_enet_f_cumret1.csv`, `reports/data/lasso_enet_ar_dgtw_1.csv` (each with a
`_yearly` companion). Common evaluation sample as before: tweeted stock-days 2012-2022 with
every listed model's prediction, target and predictions de-meaned by date; rank correlation =
mean daily Spearman, spread = equal-weighted top-minus-bottom prediction decile, bp/day.

## 0. Terms used in this report

- **L1 / L2 penalty.** Ridge, lasso and elastic net minimise the sum of squared errors plus a
  penalty on coefficient size. L2 = sum of squared coefficients (ridge): shrinks every
  coefficient toward zero, never exactly to zero. L1 = sum of absolute coefficients (lasso):
  shrinks and sets some coefficients exactly to zero, i.e. selects variables. Elastic net
  applies both. "Non-zero coefficients" counts the regressors a lasso kept.
- **Gram matrix.** X'X, the p x p cross-product matrix of the standardised regressors (388 x 388
  for text + core). All estimators here work from it and from X'y instead of the raw rows,
  which is why a 600k-row month fits in seconds.
- **Ridge-augmented Gram matrix.** X'X + l2 * n * I. Ridge's closed form is OLS with this
  replacement; running the lasso's L1 path on the same replaced matrix is exactly an elastic net
  with L2 penalty l2 * n. This puts the L2 penalty on the scale ridge needs here (l2 ~ 1-10)
  independently of the L1 penalty.
- **alpha_max.** For a given window, the smallest L1 penalty at which every coefficient is
  zero. L1 candidates are expressed as fractions of it so that they mean the same thing in
  every month.
- **Selection rule (`sse` / `rankcorr`) and the `_rc` tag.** Each month all candidate penalties
  are fitted; the one used for that month's predictions is chosen from the previous 12 months'
  out-of-sample record: `sse` = lowest squared error (the rule used for ridge in Reports 01-02),
  `rankcorr` = highest mean daily rank correlation. Models chosen with `rankcorr` carry `_rc`;
  e.g. `textcore_lasso_rank_rc` = lasso on text + core, trained on the daily rank of the
  return, penalty chosen by trailing rank correlation.

## 1. What was run

All models: monthly refit, 252-trading-day window, daily out-of-sample predictions on the
tweeted stock-days (`text_master.pkl`), **rank target** (daily percentile rank of the
next-day return), columns standardised inside each window. New in this report:

- **Lasso**: coordinate-descent L1 path on the window's precomputed Gram matrix (sklearn
  `enet_path` with `precompute`), so each month costs seconds even at 600k rows. Penalty
  candidates are fractions of `alpha_max` (the smallest penalty at which every coefficient is
  zero): 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001; fraction 1 is the null model.
- **Elastic net**: the same L1 path run on the ridge-augmented Gram matrix `G + l2 * n * I`
  (equivalent to data augmentation, so sklearn's duality-gap stopping rule stays exact), with
  `l2` in {1, 10} on the ridge scale and the L1 fraction in {0.3, 0.1, 0.03, 0.01, 0.003}: ten
  candidates. *Why not sklearn's `l1_ratio`*: it ties both penalties to one `alpha`; at the
  `alpha` sizes that make the L1 term bite here, the implied L2 term is about 1e-4 of the Gram
  diagonal, and a first implementation with `l1_ratio = 0.5` reproduced the lasso to four
  decimals in every run. The augmented-Gram form puts the two penalties on their natural scales.
- **Penalty selection**, walk-forward as before: for month *m* the candidate with the best record
  over the previous 12 months, default until any history exists. Two criteria are compared:
  `sse` (lowest out-of-sample squared error, both sides de-meaned by date -- the rule used for
  ridge in Reports 01-02) and `rankcorr` (highest mean daily rank correlation; models tagged
  `_rc`). Every run stores the per-month choice, the candidate-by-month error and
  rank-correlation tables, and the number of non-zero coefficients in a `.json` sidecar.
- Feature sets: `text` (386), `textcore` (388), `all` (53 non-text features). Ridge with the
  `rankcorr` rule and ridge on `textcore` (both new) are included for comparison.
- Evaluator change (`tools/evaluate_predictions.py`): a day on which a model's prediction is
  constant now counts as rank correlation 0 instead of being dropped. This matters for the
  `sse`-selected lasso, which chose the null model in 17 of 143 months.

## 2. Results, raw next-day return (3,171,329 stock-days)

| model | selection | rank corr (t) | D10-D1 bp (t) | D1 bp | non-zero coefficients (median) |
|---|---|---|---|---|---|
| all_rank (OLS, 53 features) | -- | 0.0306 (20.6) | 25.2 (9.2) | -23.2 | 53 |
| all_lasso_rank | sse | 0.0306 (19.5) | 24.1 (8.5) | -22.3 | 35 |
| all_lasso_rank_rc | rankcorr | 0.0315 (19.4) | 25.3 (8.8) | -22.4 | 16 |
| all_enet_rank | sse | 0.0318 (19.1) | 21.3 (7.4) | -19.0 | 53 |
| text_ridge_rank | sse | 0.0374 (21.3) | 13.4 (5.2) | -10.6 | 386 |
| text_lasso_rank | sse | 0.0347 (20.8) | 15.0 (5.3) | -13.1 | 159 |
| text_lasso_rank_rc | rankcorr | 0.0369 (21.2) | 16.6 (6.2) | -13.7 | 214 |
| text_enet_rank | sse | 0.0369 (20.8) | 13.2 (5.1) | -10.7 | 363 |
| textcore_ols_rank | -- | 0.0372 (22.7) | 22.6 (8.0) | -19.5 | 388 |
| textcore_ridge_rank | sse | **0.0397 (22.6)** | 22.7 (7.9) | -20.1 | 388 |
| textcore_ridge_rank_rc | rankcorr | 0.0393 (21.8) | 22.9 (8.0) | -20.0 | 388 |
| textcore_lasso_rank | sse | 0.0358 (21.3) | 25.3 (8.6) | -23.8 | 122 |
| textcore_lasso_rank_rc | rankcorr | 0.0382 (21.9) | **25.1 (8.9)** | -22.8 | 204 |
| textcore_enet_rank | sse | **0.0401 (21.7)** | 22.7 (7.5) | -22.0 | ~365 |
| textcore_enet_rank_rc | rankcorr | 0.0396 (21.4) | 22.9 (7.7) | -20.6 | ~365 |

## 3. Results, DGTW-adjusted next-day return (2,880,016 stock-days; models trained on the DGTW rank)

| model | rank corr (t) | D10-D1 bp (t) |
|---|---|---|
| lr_all_rank_dgtw (panel, 53) | 0.0267 (20.2) | 19.7 (8.0) |
| core_rank_dgtw | 0.0277 (18.8) | 18.2 (7.3) |
| text_ols_rank_dgtw | 0.0328 (26.6) | 13.6 (5.6) |
| textcore_ols_rank_dgtw | 0.0341 (27.6) | 20.1 (7.8) |
| textcore_ridge_rank_dgtw_rc | **0.0362 (26.4)** | 20.9 (8.0) |
| textcore_lasso_rank_dgtw_rc | 0.0348 (25.8) | **24.4 (9.2)** |
| textcore_enet_rank_dgtw_rc | **0.0362 (26.0)** | 21.9 (9.2) |

## 4. Reading

1. **Regularising the text + core model buys a modest, consistent gain in ordering**: from
   0.037 (OLS) to 0.040 (elastic net or ridge), on both targets (0.034 -> 0.036 on DGTW), with
   the decile spread unchanged at ~23 bp. The gain is of the same size as adding the core
   features to the text, and larger than anything the 53 non-text features contributed.
2. **Ridge and elastic net are the same model in practice.** The elastic net selects `l2 = 1`
   with the lightest L1 candidates in most months (median ~365 non-zero coefficients of 388):
   the data want shrinkage of all coefficients, not selection. Its rank correlation matches
   ridge's within 0.0005 everywhere.
3. **Lasso trades ordering for the tail.** With `rankcorr` selection it keeps ~200 of 388
   coefficients, has the lowest rank correlation of the three regularised `textcore` models
   (0.038) but the largest decile spread (25 bp, t 8.9-9.2 on both targets) -- the sparse
   model leans on the core features' bottom-decile effect. With `sse` selection it is clearly
   worse (0.036), because the squared-error rule cannot tell the candidates apart on a rank
   target and picks the null model in 17 months.
4. **The selection rule matters only for the lasso.** For ridge and elastic net `sse` and
   `rankcorr` give the same answer within noise; for the lasso the rank-correlation rule is
   the right one and is what the `_rc` files use.
5. **Regularisation does nothing for the 53 non-text features** (0.031-0.032 either way), so the
   OLS baselines are not handicapped by their estimator.
6. Yearly patterns are as before: every `textcore` model beats the 53-feature model in every
   year from 2013 on.

**Working model going forward**: `textcore` with elastic net or ridge and the `rankcorr` rule
(0.040 / 23 bp raw; 0.036 / 21-22 bp DGTW), or its lasso variant when the tail matters more
than the ordering.

## Files

- `03a/prediction_linear_regression_text_only.ipynb`: `ESTIMATOR` now `ols | ridge | lasso |
  enet`; `TEXTONLY_SELECT` = `sse | rankcorr` (output tag `_rc`); sidecar records the penalty
  path and non-zero counts.
- `tools/evaluate_predictions.py`: constant-prediction days count as zero rank correlation.
- Predictions (Data/): `predictions_{lasso,elasticnet,ridge}_{textonly,textcore,all}_rank[_dgtw][_rc]_input=*.pkl`.
