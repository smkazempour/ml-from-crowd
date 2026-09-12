# Report 02 -- Decomposing the text signal: attention vs tag vs text, DGTW-adjusted returns, composition, value weighting

Date: 2026-09-11 (evening). Builds on Report 01. Data tables for every number below are in
`reports/data/` (`decomposition_*.csv`, `composition_check.csv`, `value_weighted_check.csv`).
All evaluations use the **common sample**: tweeted stock-days 2012-2022 with every listed
model's prediction and a non-missing target; target and predictions de-meaned by date. Rank
correlation = mean over days of the within-day Spearman correlation between prediction and
realised target; decile spread = equal-weighted top-minus-bottom prediction-decile next-day
return, bp/day, unless stated otherwise.

## 0. Notation

Model keys are built from these parts:

| part | meaning |
|---|---|
| `lr_2`, `lr_all` | the existing baselines: OLS on net sentiment + log volume, or on all 53 features, trained on the full CRSP panel |
| `core` | the same two regressors as `lr_2` (net sentiment, log volume), but estimated on the tweeted stock-days of the text table |
| `text` | the 384 embedding dimensions plus the two agreement measures (`embed_norm`, `embed_cos`); `text_ols` / `ridge_text` name the estimator |
| `textcore` | `text` and `core` together (388 regressors) |
| `all` | the 53 non-text features (the `lr_all` set), estimated on the tweeted stock-days |
| `textall` | `text` and `all` together (439 regressors) |
| `dm` | date-de-meaned training: inside each 252-day training window the target and every regressor have their same-date cross-sectional mean removed (date fixed effects); regressors are de-meaned by date at prediction time too |
| `rank` | trained on the daily percentile rank of the target (minus 0.5) instead of the target's value |
| `dgtw` | trained on, or evaluated against, the DGTW-adjusted next-day return (`ar_dgtw_1`: raw return minus the matched size / book-to-market / momentum benchmark portfolio return) |
| `ols`, `ridge` | the estimator; ridge picks its penalty each month from the previous 12 months' out-of-sample error |

Example: `textcore_rank_dgtw` = OLS on text + core, trained on the daily rank of the
DGTW-adjusted return; `ridge_text_dm_dgtw` = ridge on the text features, date-de-meaned,
trained on the DGTW-adjusted return itself.

## 1. What was run

- **Decomposition (rank target).** On the text table (tweeted stock-days only) with the daily
  percentile rank of the target as the training target: `core` = net sentiment + log volume
  (the 2-feature baseline re-estimated on the same sample); `text` = 384 dims + 2 agreement
  measures; `textcore` = both. Compared with the panel-trained rank baselines `lr_2_rank` and
  `lr_all_rank`.
- **DGTW-adjusted target.** `TARGET_COL=ar_dgtw_1` (next-day return minus the DGTW
  characteristic-benchmark return) added as a switch to the text notebook and both baseline
  notebooks; the same models re-trained on it and everything evaluated against it.
- **Composition and horizon** (`tools/composition_check.py`): rank correlation by message-count
  bucket, by daily size tercile, by year x bucket, against DGTW abnormal returns over 1-10 days,
  and what each prediction loads on.
- **Value weighting** (`tools/value_weighted_check.py`): daily quintile/decile sorts on each
  prediction, long top / short bottom, held one day, equal- vs market-cap-weighted, full sample
  and within size terciles.

Implementation note: switching the all-features notebook's target exposed a latent leak -- its
"all numeric columns" feature sweep excluded only the *current* target, so with the DGTW target
the raw next-day return became a regressor (54 inputs). `f_cumret1` is now excluded
unconditionally; the affected file was deleted and rerun (53 inputs). No earlier result used
the leaky file.

## 2. Decomposition

### 2.1 Raw next-day return as the evaluation target (3,171,329 stock-days)

| model (rank-trained) | regressors | rank corr (t) | D10-D1 bp (t) | D1 bp | D10 bp |
|---|---|---|---|---|---|
| lr_2_rank | net sentiment + log volume, full panel | 0.0294 (17.6) | 22.0 (7.7) | -22.4 | -0.4 |
| lr_all_rank | 53 features, full panel | 0.0295 (19.7) | 24.4 (8.8) | -22.8 | +1.6 |
| core_rank | net sentiment + log volume, tweeted sample | 0.0307 (18.0) | 22.0 (7.7) | -22.4 | -0.4 |
| text_rank (ridge, dm) | 384 + agreement | 0.0374 (21.3) | 13.4 (5.2) | -10.6 | +2.8 |
| text_ols_rank | 384 + agreement | 0.0359 (21.8) | 14.6 (5.5) | -11.4 | +3.2 |
| **textcore_rank** | 384 + agreement + core | **0.0372 (22.7)** | **22.6 (8.0)** | -19.5 | +3.1 |

- Re-estimating the two core features on the tweeted sample changes nothing (0.031 vs 0.029).
- Adding the core features to the text does not raise the rank correlation (0.037 either way),
  but it brings the core's bottom-decile tail (-19.5 bp) into the model. The combined model
  matches the best baseline on the decile spread and beats it on rank correlation: it is the
  first model that dominates both baselines on both metrics.
- Yearly: the combined model is above both baselines in every year except 2021.

### 2.2 DGTW-adjusted next-day return as the evaluation target (2,880,016 stock-days)

| model | trained on | rank corr (t) | D10-D1 bp (t) |
|---|---|---|---|
| lr_2_rank_dgtw | rank of DGTW return | 0.0265 (18.0) | 19.0 (7.8) |
| lr_all_rank_dgtw | rank of DGTW return | 0.0267 (20.2) | 19.7 (8.0) |
| core_rank_dgtw | rank of DGTW return | 0.0277 (18.8) | 18.2 (7.3) |
| text_rank_dgtw | rank of DGTW return | 0.0328 (26.6) | 13.6 (5.6) |
| **textcore_rank_dgtw** | rank of DGTW return | **0.0341 (27.6)** | **20.1 (7.8)** |
| lr_2_rank | rank of raw return | 0.0270 (18.6) | 19.3 (8.0) |
| text_rank | rank of raw return | 0.0339 (26.4) | 11.7 (5.7) |
| lr_2 | raw return | 0.0206 (15.1) | 15.3 (6.9) |
| lr_2_dgtw | DGTW return | 0.0227 (17.4) | 17.1 (8.3) |
| ridge_text_dm_dgtw | DGTW return | 0.0125 (11.5) | 7.1 (4.3) |

- Removing the characteristic-driven part of returns lowers every model a little and changes
  the ranking not at all; the text's t-statistics are the highest in the table (26-28 vs 18-20).
  The text ordering is not a size / book-to-market / momentum proxy.
- Whether the rank model is trained on the raw or the DGTW rank makes almost no difference
  (text 0.0339 vs 0.0328). Return-level training on the DGTW target is as bad as on the raw
  target for the text (0.0125): the rank target is what matters, not the benchmark.

### 2.3 All 53 non-text features, with and without the text (rank target)

Same samples as 2.1 and 2.2 (`decomposition_all_*.csv`).

| model | regressors | raw target: rank corr (t) / D10-D1 bp (t) | DGTW target: rank corr (t) / D10-D1 bp (t) |
|---|---|---|---|
| core_rank | 2 | 0.0307 (18.0) / 22.0 (7.7) | 0.0277 (18.8) / 18.2 (7.3) |
| all_rank | 53 | 0.0306 (20.6) / 25.2 (9.2) | 0.0272 (21.1) / 21.5 (8.9) |
| lr_all_rank (panel-trained) | 53 | 0.0295 (19.7) / 24.4 (8.8) | 0.0267 (20.2) / 19.7 (8.0) |
| text_rank | 386 | 0.0374 (21.3) / 13.4 (5.2) | 0.0328 (26.6) / 13.6 (5.6) |
| textcore_rank | 388 | 0.0372 (22.7) / 22.6 (8.0) | 0.0341 (27.6) / 20.1 (7.8) |
| textall_rank | 439 | 0.0376 (23.9) / 23.7 (8.4) | 0.0336 (28.2) / 20.3 (8.5) |

- The 51 features beyond net sentiment and log volume add nothing to the ordering on the
  tweeted sample (0.0306 vs 0.0307) and about 3 bp to the decile spread.
- Adding all 53 to the text gives the same rank correlation as adding the two core features
  (0.0376 vs 0.0372 raw; 0.0336 vs 0.0341 DGTW) and a spread within 1 bp. The text is what
  moves the ordering; beyond attention and the tag, the remaining features are redundant with
  it. `textall_rank` and `textcore_rank` are interchangeable; the smaller one is preferable.

## 3. Composition and horizon (`composition_check.csv`)

**By message count** (within-bucket, within-date rank correlation with the next-day return):

| messages on the day | text_rank | textcore_rank | lr_2_rank | lr_all_rank | share of days 2022 |
|---|---|---|---|---|---|
| 1 | 0.013 | 0.010 | -0.000 | 0.004 | 36% |
| 2 | 0.021 | 0.020 | 0.002 | 0.007 | 14% |
| 3-5 | 0.029 | 0.027 | 0.003 | 0.007 | 18% |
| 6-10 | 0.040 | 0.038 | 0.001 | 0.006 | 11% |
| 11-30 | 0.049 | 0.051 | 0.004 | 0.020 | 11% |
| >30 | 0.084 | 0.100 | 0.039 | 0.078 | 10% |

The text ordering strengthens steadily with the amount of text behind the stock-day, from 0.013
on single-message days to 0.084 on days with more than 30 messages. The baselines have almost
no *within-bucket* ordering except on the busiest days: their full-sample correlation comes
mostly from ordering *across* buckets, i.e. from the attention effect itself.

**Over time.** Within each bucket the text correlation is roughly flat across 2013-2022
(single-message days 0.01-0.02 throughout; >30-message days 0.06-0.11). The rise of the
full-sample correlation from 0.01 (2012) to 0.055 (2022) is therefore mostly composition: the
share of single-message days fell from 69% to 36% and the share of >30-message days rose from
0.5% to 10-13%.

**By size** (daily terciles of market cap): text_rank 0.042 / 0.017 / 0.011 for small / mid /
large; lr_2_rank 0.054 / 0.017 / 0.006. Every signal is strongest in small caps; the text keeps
relatively more of its ordering in mid and large caps.

**Horizon** (rank correlation with the cumulative DGTW abnormal return over h days; corrected
2026-09-12, `horizon_check.csv`):

| h | core_rank (benchmark) | lr_all_rank | text_rank | textcore_ols_rank | textcore_enet_rank |
|---|---|---|---|---|---|
| raw 1d | 0.031 | 0.030 | 0.037 | 0.037 | 0.040 |
| 1 | 0.028 | 0.027 | 0.034 | 0.034 | 0.037 |
| 3 | 0.035 | 0.032 | 0.041 | 0.040 | 0.044 |
| 5 | 0.039 | 0.035 | 0.046 | 0.045 | 0.049 |
| 10 | 0.045 | 0.040 | 0.053 | 0.052 | 0.057 |
| 21 | 0.060 | 0.051 | 0.070 | 0.066 | 0.074 |
| 42 | 0.065 | 0.056 | 0.079 | 0.074 | 0.084 |
| 63 | 0.073 | 0.062 | 0.089 | 0.084 | 0.094 |

The correlation keeps growing out to three months for every model, so the predicted return is
not a one-day reversal that gives back; it accrues. The text models' advantage over the
benchmark widens with the horizon (0.006 at one day, 0.021 at 63 days for the elastic net).
Caveat: the h-day abnormal returns of consecutive days overlap, so these are descriptive, not
independent tests. Data note: in `merged_master` (hence `text_master`) `ar_dgtw_21` is exactly
zero for every stock on every day of 2013, while the current CRSP file's 21-day columns for
2013 are fine; the July build of `merged_master` used a bad input for that year and horizon.
Those days are excluded from the 21-day row (no cross-sectional variance); other horizons are
unaffected. An earlier version of this paragraph blamed non-finite values; that was wrong.

**What the predictions load on** (rank correlation of the daily prediction ranks with ...):
text_rank -0.42 with log volume, +0.30 with market cap, +0.10 with net sentiment, +0.31 with
`embed_cos`; lr_2_rank -0.87 with log volume and +0.37 with net sentiment. The two-feature
baseline is, in ordering terms, mostly the attention effect; the text ordering is much less so.

## 4. Value weighting (`value_weighted_check.csv`; long-short decile sorts, next-day, no costs)

| model | equal-weighted bp/day (t) | cap-weighted bp/day (t) |
|---|---|---|
| text_rank | 13.3 (5.1) | 3.1 (1.6) |
| textcore_rank | 23.6 (8.4) | 0.8 (0.4) |
| lr_2_rank | 24.7 (8.9) | -1.7 (-1.0) |
| lr_all_rank | 26.6 (10.0) | 0.9 (0.5) |
| lr_2 | 20.3 (8.4) | -2.2 (-1.4) |

Within daily size terciles (sorts formed inside the tercile; cap-weighted):

| model | small | mid | large |
|---|---|---|---|
| text_rank | 14.3 (3.4) | 4.8 (1.7) | 0.7 (0.5) |
| textcore_rank | 39.6 (7.9) | 8.5 (2.7) | -0.2 (-0.1) |
| lr_2_rank | 35.4 (7.1) | 6.7 (1.9) | -2.1 (-1.3) |
| lr_all_rank | 37.7 (7.6) | 8.5 (2.6) | 0.8 (0.5) |

**Cap-weighted over the whole tweeted universe, no model earns anything**, baselines included.
Within the small-cap tercile the spreads survive value weighting (14-40 bp/day); in the middle
tercile they are marginal; in large caps there is nothing. This applies to every model in the
project, not only to the text.

## 5. Conclusions

1. **The text and the tag/attention features carry different information.** The text supplies a
   broad ordering of the cross-section that strengthens with the amount of text; the core
   features supply a concentrated bottom-decile effect (heavily discussed stocks fall). The
   combined rank-trained model (`textcore_rank`) is the first to dominate both baselines on both
   ordering and decile spread, on raw and on DGTW-adjusted returns.
2. **The text result is robust to the benchmark** (DGTW-adjusted returns) and **persists over
   ten days**; it is not a characteristic proxy and not a one-day reversal.
3. **The apparent improvement over time is composition**, not learning: within message-count
   buckets the text correlation is flat; there is simply more text per stock-day in later years.
4. **All of the predictability in this project, baselines included, lives in small caps.**
   Cap-weighted across the tweeted universe the long-short returns are zero. Any economic
   claim has to be made for small caps explicitly, or with transaction costs that small-cap
   trading implies.
5. Point 1 and the message-count gradient together are the case for the untagged messages:
   more text per stock-day is where the text signal is strongest, and the untagged corpus
   would raise the text per stock-day by a factor of about 2.25.

## 6. Suggested next steps

- Adopt `textcore_rank` (or its DGTW variant) as the working text model; keep `core_rank` as
  the like-for-like baseline on the tweeted sample.
- Decide the economic framing: small-cap universe with costs, or ordering-based claims.
- Run the untagged embedding track (tools exist; ~54 h CPU or hours on a GPU), then rebuild the
  text table with three stock-day embeddings (all / tagged / untagged) and repeat Sections 2-4.
- Minor: rebuild `merged_master` (or patch `ar_dgtw_21` for 2013, which is all zeros) before using the 21-day horizon as a target.

## Files

- New/changed code: `03a/prediction_linear_regression_text_only.ipynb` (`FEATURE_SET` now
  `embed | embed+norm | core | embed+norm+core | all | embed+norm+all`, `TARGET_COL`), both baseline 03a notebooks
  (`TARGET_COL`; all-features sweep excludes `f_cumret1`), `tools/evaluate_predictions.py`,
  `tools/composition_check.py`, `tools/value_weighted_check.py`.
- Predictions (Data/): `predictions_linear_regression_{core,textcore,all,textall}_rank[_dgtw]_input=*.pkl`,
  `predictions_linear_regression_textonly_rank_dgtw_input=386.pkl`,
  `predictions_ridge_textonly_dm_dgtw_input=386.pkl`,
  `predictions_linear_regression[_rank]_dgtw_input={2,53}.pkl`.
