# Report 01 -- Can the text of StockTwits messages predict next-day returns? First results

Date: 2026-09-11. Code state: commit a0f23ab plus the figure notebook. Data on `D:\StockTwits\Data`.
Companion notes with measurements and code-level detail: `01 - feature extraction/features_08_integration_notes.md`, Sections 8-8c.

## 1. Question and setup

The project asks how much information StockTwits messages carry for next-day stock returns.
Until now every model used only the messages' self-reported Bullish/Bearish tags and their
timing and volume (53 stock-day features). This report covers the first attempt to use the
**text** of the messages, through sentence embeddings, and compares it with the existing
baselines on identical samples.

### Sample construction (relevant facts)

- Panel: CRSP common stocks (share codes 10-12, NYSE/AMEX/NASDAQ), about 4,300 stocks per day,
  2010-2023: 15,532,693 stock-days (`merged_master.pkl`). Target `f_cumret1` is the next
  trading day's CRSP return. Abnormal returns (DGTW, CAPM, FF3/5/6 at 1-63 days) are carried
  along but were not used as targets in this report.
- Messages: of ~501M raw StockTwits messages, the cleaning step keeps only the ~176M with a
  Bullish/Bearish tag. After cashtag explosion and matching to a CRSP stock on the message's
  trading date, 77.0M messages / 85.4M message-symbol pairs remain. Both the 53 features and
  the embeddings are built from this table, so "stock-days with messages" (22.6% of the panel)
  and "stock-days with text" coincide.
- Embeddings: every message encoded once with `all-MiniLM-L6-v2` (384-dim unit vectors,
  general-purpose sentence encoder, not finance-tuned), mean-pooled per (symbol, date); a
  message counts toward every ticker it names. 3,505,815 stock-days, 2010-06 to 2024-01.
  Full run: 43.6 h on CPU (2026-09-07 to 09-09).
- Text training table `text_master.pkl`: the embedding joined to the panel's keys, target,
  abnormal returns and 53 features: 3,514,785 tweeted stock-days x 480 columns. Two agreement
  measures are added from the stored mean vector: `embed_norm` (its length) and `embed_cos`
  (implied average pairwise cosine among the day's messages; 1 for single-message days, which
  are 38.9% of the table; mean 0.45 on multi-message days).
- **Common evaluation sample** for every comparison below: 3,171,329 tweeted stock-days,
  2012-01 to 2022-12, on which every model has a prediction; target and predictions are
  cross-sectionally de-meaned by date.

### Models

All models share the baseline design: monthly refit, 252-trading-day rolling window, daily
out-of-sample predictions 2012-2023.

| key | regressors | estimator | notes |
|---|---|---|---|
| lr_2 | net sentiment, log volume | OLS on the full panel | the paper's 2-feature baseline |
| lr_all | 53 features | OLS on the full panel | |
| lr_text | 384 embedding dims | OLS on tweeted stock-days | |
| lr_text_n / lr_text_n_dm | 384 + 2 agreement | OLS / OLS with date fixed effects in the window | |
| ridge_text_n / ridge_text_n_dm | 384 + 2 agreement | ridge, penalty chosen walk-forward on the previous 12 months' OOS error | |
| *_rank | as above | same | trained on the **daily percentile rank** of the target |

The message count is deliberately not a regressor in the text models (it is attention, not content).

Key suffixes: `_n` = the two agreement measures added; `_dm` = date-de-meaned training (target and regressors have their same-date cross-sectional mean removed inside each training window); `_rank` = trained on the daily percentile rank of the target.

## 2. Results

Metrics: mean daily Spearman rank correlation between prediction and realised de-meaned return;
equal-weighted top-minus-bottom prediction-decile spread (bp/day, t-statistic from the daily
series); pooled OLS slope of the return on the prediction (two-way clustered); OOS R2 (return
target only). Figures: `Figures/text_models/`.

### 2.1 Return target

| model | OOS R2 | slope (s.e.) | rank corr | D10-D1 bp/day (t) |
|---|---|---|---|---|
| lr_2 | +0.000214 | 1.02 (0.11) | 0.0228 | 17.4 (6.8) |
| lr_all | -0.000896 | 0.34 (0.08) | 0.0146 | 24.2 (10.8) |
| lr_text (384, OLS) | -0.001541 | 0.06 (0.02) | 0.0076 | 5.8 (3.4) |
| lr_text_n (386, OLS) | -0.001551 | 0.10 (0.02) | 0.0126 | 11.6 (6.1) |
| lr_text_n_dm | -0.001027 | 0.15 (0.03) | 0.0151 | 10.6 (5.7) |
| ridge_text_n | +0.000004 | 0.67 (0.26) | 0.0100 | 4.6 (2.4) |
| ridge_text_n_dm | +0.000018 | 0.87 (0.28) | 0.0161 | 9.8 (4.8) |

- Raw OLS on 384 dimensions orders stocks in the right direction but is over-dispersed by a
  factor of ~16 (slope 0.06), so its R2 is negative every year.
- The two agreement measures are the largest single improvement to the OLS text model
  (decile spread 5.8 -> 11.6 bp, rank corr 0.0076 -> 0.0126).
- Date-de-meaned training helps every metric; ridge repairs the calibration (slope 0.87, R2
  just positive) at the price of heavy shrinkage.
- **The two-feature baseline beats every richer model on rank correlation and R2**, and every
  text model is negative in 2020-2021. This was the central puzzle and motivated Section 3.

### 2.2 Rank target

| model | rank corr | D10-D1 bp/day (t) | pooled t |
|---|---|---|---|
| lr_2_rank | 0.0294 | 21.5 (7.5) | 7.3 |
| lr_all_rank | 0.0295 | 23.6 (8.6) | 5.6 |
| lr_text_n_rank | 0.0359 | 14.4 (5.5) | 4.9 |
| lr_text_n_dm_rank | 0.0361 | 14.5 (5.5) | 4.9 |
| ridge_text_n_rank | 0.0374 | 12.7 (4.9) | 3.7 |
| ridge_text_n_dm_rank | 0.0374 | 12.9 (5.0) | 3.7 |

- Training on the daily rank instead of the raw return more than doubles the text models'
  rank correlation (0.016 -> 0.037), puts them above both baselines (0.029), and makes them
  positive in every year 2012-2022 (`rank_corr_yearly.png`). Estimator and de-meaning stop
  mattering.
- The baselines keep the larger decile spread and pooled t: their information is concentrated
  in the bottom decile (about -20 bp), while the text's is a broad monotone ordering across the
  cross-section (`decile_profiles.png`). Squared-error training on raw returns was throwing
  that ordering away.
- Cumulative top-minus-bottom decile returns 2012-2022 (`cumulative_decile_spread.png`):
  baselines ~600-700%, rank-trained text ~400%, return-trained text ~300% (equal-weighted,
  no costs, de-meaned).

## 3. Diagnostics (Phase 0)

Same evaluation sample, no models.

- **Single features.** Daily Spearman with the next-day return: `log_volume` -0.030 (t -17),
  D10-D1 **-27 bp/day** (t -10); `unique_user_count` the same; `net_sentiment` +0.010 (t 8.7),
  +7.6 bp; `abnormal_sentiment_1d` +0.012; `disagreement_index` -0.017; `embed_norm` +0.025
  (t 15), +10.6 bp; `embed_cos` +0.013 with a flat decile spread (39% single-message ties).
  The dominant model-free signal among tweeted stock-days is a **negative attention effect**:
  yesterday's most-discussed stocks underperform today. That is most of what `lr_2` is.
  `embed_norm` inherits part of it mechanically (the norm falls with the message count).
- **Anatomy of the tag.** 65% of tweeted stock-days are unanimously bullish (net sentiment 1),
  mostly single messages, and earn +4.7 bp; the 14% that are mostly-but-not-unanimously
  bullish earn **-20.6 bp** (-40 bp with more than 30 messages); unanimously bearish days
  -1.9 bp. Heavy bullish chatter predicts negative returns; the bearish tail is small.
- **The tag inside the embedding.** Regressing net sentiment on the 386 embedding columns
  (fit on year t-1, applied to year t, no returns involved) gives OOS R2 0.16-0.34 and
  correlation ~0.5 with the tag. As a return predictor this "text stance" equals the tag
  (7.2 vs 7.6 bp decile spread) and adds nothing once orthogonalised to it (2.6 bp, t 1.7).
  The text's incremental information is not stance; it is the broad ordering of Section 2.2.
- **Untagged messages.** 65% of raw messages carry no tag and are excluded everywhere. A scan
  of the raw files (`tools/untagged_message_scan.py`) finds **96.5M untagged CRSP-matched
  messages / 111M message-symbol pairs** (2010-2023), 1.25x the tagged corpus; embedding them
  costs ~54 h of CPU or hours on a GPU. Since the text ordering does not depend on the tag,
  they are usable as they are.

## 4. What we conclude so far

1. The text carries information about next-day returns beyond what the tags and attention
   measures capture, but of a different kind: a broad ordering of the cross-section rather
   than a tail signal. It is only visible when the model is trained on ranks.
2. The attention effect is the strongest predictor in the tweeted sample and needs to be
   modelled explicitly rather than left inside a two-feature baseline.
3. The general-purpose encoder captures roughly half of the stance in a message; on tagged
   messages that adds nothing to the tag itself.
4. All effect sizes are small in R2 terms (best text R2 0.00002; best baseline 0.0002); the
   economically relevant quantities are ordering and decile spreads.

## 5. Open questions taken up next (in this order)

1. Decompose the signal on the same sample and target: attention + tag alone, text alone,
   both together (rank target). Also with DGTW-adjusted next-day returns as the target.
2. Stress the text ordering: value-weighted portfolios; horizon (3-21 day abnormal returns);
   composition (rank correlation within message-count buckets and size groups, since the
   text correlation rises steadily from 0.01 in 2012 to 0.055 in 2022 while coverage grows).
3. Only then the untagged-message embedding track.

## Files

- Notebooks: `02 - prepare training dataset/build_text_master.ipynb`,
  `03a - linear regression/prediction_linear_regression_text_only.ipynb` (switches
  `ESTIMATOR`, `TARGET_DEMEAN`, `FEATURE_SET`, `RANK_TARGET`), the two baseline 03a notebooks
  (`RANK_TARGET`), `04 - predictive regressions/{plot_oos_r2, predictive_regressions,
  rank_correlation, plot_model_performance}.ipynb`.
- Predictions: `Data/predictions_*textonly*.pkl`, `Data/predictions_linear_regression_rank_input={2,53}.pkl`,
  with `.json` sidecars holding the ridge penalty paths.
- Figures: `Figures/text_models/` (+ `model_performance_summary.csv`).
- Tools: `tools/run_notebook.py` (headless runner), `tools/untagged_message_scan.py`.
