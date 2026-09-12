# Project state -- StockTwits text and return prediction (as of 2026-09-12)

Read this first in a new session. Detailed results: Reports 01-03 in this folder; code-level
history: `01 - feature extraction/features_08_integration_notes.md`.

## 1. Question and benchmark convention

The project measures how much information StockTwits messages carry for next-day stock
returns. **Benchmark for every test: OLS on the two core features, net sentiment and log
volume (attention), estimated on the same sample and target as the model under test.** That
is how the literature currently measures the informativeness of social media; the question is
whether anything, in particular the message text, adds to it. In the files this benchmark is
`core_rank` (rank target) / `core` (return target) on the tweeted stock-days, or `lr_2` /
`lr_2_rank` when trained on the full panel. Every result table should carry it as the first
row and report gains relative to it.

Evaluation conventions: common sample (tweeted stock-days 2012-2022 on which every compared
model has a prediction), target and predictions de-meaned by date; metrics are the mean daily
Spearman rank correlation, the equal-weighted top-minus-bottom prediction-decile spread
(bp/day, with t-statistic), and, for return-level models only, OOS R2 and the pooled slope.
Targets: raw next-day return `f_cumret1` and the DGTW-adjusted `ar_dgtw_1`.

## 2. Data and where things live

- Everything is on `D:\StockTwits\` (`Code` = this repo, `Data`, `Figures`). The copy under
  `C:\Users\skazempour\Documents\StockTwits` is stale and should not be used.
- `Data/merged_master.pkl`: CRSP common-stock panel 2010-2023 with the 53 StockTwits features,
  15.5M stock-days; 22.6% have at least one tagged message. Target `f_cumret1` = next-day
  return; `ar_*` = abnormal returns (DGTW, CAPM, FF3/5/6; 1-63 days). `ar_dgtw_21` is exactly zero for every
  stock throughout 2013 (a bad input in the July 2026 build; the CRSP file itself is fine).
- `Data/v1/data/csv/text_embeddings_mlcrowd/text_embeddings_stock_day.pkl`: mean-pooled
  `all-MiniLM-L6-v2` embeddings (384 dims) of every tagged, CRSP-matched message, per
  (symbol, date); 3.5M stock-days; the full encode took 43.6 h CPU (2026-09-07..09). Only
  messages with a Bullish/Bearish tag are in the pipeline (35% of the raw 501M); message-level
  vectors were not kept.
- `Data/text_master.pkl` (built by `02 - prepare training dataset/build_text_master.ipynb`):
  one row per tweeted stock-day, 3,514,785 x 480: keys, `mm_index` (row label in
  merged_master), target, `ar_*`, the 53 features, `embed_n`, the two agreement measures
  `embed_norm` / `embed_cos`, and `embed_000..383`.
- Predictions: `Data/predictions_<model>_<featureset>[_dm][_rank][_dgtw][_rc]_input=N.pkl`
  (+ `.json` sidecar with the penalty path) for the text-track models;
  `Data/predictions_linear_regression[_rank][_dgtw]_input={2,53}.pkl` for the panel baselines.
- Reports and their CSVs: `reports/`, `reports/data/`. Figures: `Figures/text_models/`.
- Run logs of everything: `Data/v1/data/csv/text_embeddings_mlcrowd/_run/`.

## 3. Code you will use

- `03a - linear regression/prediction_linear_regression_text_only.ipynb`: the text-track model
  notebook. Switches (environment variables in brackets): `ESTIMATOR` [`TEXTONLY_ESTIMATOR`]
  = ols | ridge | lasso | enet; `FEATURE_SET` [`TEXTONLY_FEATURES`] = embed | embed+norm | core
  | embed+norm+core | all | embed+norm+all; `TARGET_DEMEAN` [`TEXTONLY_DEMEAN`]; `RANK_TARGET`;
  `TARGET_COL` (f_cumret1 | ar_dgtw_1); `TEXTONLY_SELECT` = sse | rankcorr. Monthly refit,
  252-day window, standardised columns, penalties chosen walk-forward from the previous 12
  months. Each variant runs in 2-5 minutes.
- Baseline notebooks `03a/prediction_linear_regression{,_all_features}.ipynb`: `RANK_TARGET`
  and `TARGET_COL` switches (the all-features sweep now excludes `f_cumret1` explicitly).
- `tools/run_notebook.py`: headless notebook runner (used for every run; logs per cell).
- `tools/evaluate_predictions.py --target ... --models "key=file,..."`: rank correlation,
  decile spreads, yearly table, CSV out. `tools/composition_check.py`: by message count, size,
  year, horizon, loadings. `tools/value_weighted_check.py`: equal- vs cap-weighted sorts.
  `tools/untagged_message_scan.py`: counts untagged CRSP-matched messages.
- 04 notebooks (`plot_oos_r2`, `predictive_regressions`, `rank_correlation`) have the text
  models registered and a `COMMON_SAMPLE` toggle; `04/plot_model_performance.ipynb` draws the
  figures. `05/form_portfolios.ipynb` has the text models registered but has not been run with
  them. The `TEXT_VARIANT` switches in the model and 04/05 notebooks are legacy and inert.
- The notebooks with a generator behind them (`build_text_master`, the text-only model
  notebook, `plot_model_performance`) were written by scripts; edit the notebooks directly
  from now on.

## 4. What we know (Reports 01-03)

1. Return-level training on the text loses: 384-dim OLS is over-dispersed (slope 0.06, R2 < 0);
   the two-feature benchmark beats every richer model on ordering.
2. **Rank-target training recovers the text's information.** Text alone reaches rank
   correlation 0.037 versus 0.031 for the benchmark (`core_rank`), positive in every year;
   its decile spread is smaller (13 vs 22 bp) because the benchmark's information is a
   bottom-decile tail effect while the text's is a broad ordering.
3. **Text + core** (`textcore`) keeps both: 0.037 / 22.6 bp with OLS, **0.040 / 23 bp with
   ridge or elastic net**, 0.038 / 25 bp with lasso. The other 51 features add nothing.
   Regularisation does nothing for the non-text features.
4. Robust to the DGTW benchmark (ordering unchanged, text t-stats highest), keeps accruing out
   to 63 days with the text's lead over the benchmark widening (Report 02, horizon table), strengthens with the amount of text (0.013 on single-message days to 0.084 on
   >30-message days); the rise over time is composition, not learning.
5. **Cap-weighted over the whole tweeted universe, every model earns ~0, benchmark included.**
   The predictability lives in small caps (within-small-cap cap-weighted spreads 14-40 bp/day).
6. The embedding contains about half of the tag's stance but nothing beyond it as stance; the
   text's incremental information is not sentiment.
7. Model-free: the strongest single predictor among tweeted stock-days is a *negative*
   attention effect (log volume, D10-D1 = -27 bp/day); heavy mostly-bullish chatter earns
   -21 bp next day.
8. Untagged messages: 96.5M CRSP-matched messages (1.25x the tagged corpus) are excluded by the
   cleaning filter; embedding them costs ~54 h CPU or hours on a GPU.

## 5. Suggested next steps

1. **Re-frame all tables against the core-OLS benchmark** (first row, increments reported) and
   regenerate `Figures/text_models/` with the benchmark and the working model
   (`textcore` ridge/elastic net, `rankcorr` rule) as the two headline series.
2. **Economic framing.** Decide between a small-cap universe with transaction costs and
   ordering-based claims; run `05/form_portfolios.ipynb` with the working model registered
   (`COMMON_SAMPLE = True`) and factor-adjust the long-short returns.
3. **Untagged messages** as a parallel embedding track (second cleaned table with the filter
   off and a `tagged` flag; embeddings for all / tagged-only / untagged-only messages; keep
   message-level vectors this time via `SAVE_MESSAGE_EMBEDDINGS`). The message-count
   gradient says this is where the text signal should be strongest.
4. **Text representation.** The encoder is general-purpose; a finance-tuned or fine-tuned
   encoder (needs GPU), or message-level rather than mean-pooled aggregation (variance of the
   day's vectors, share of messages near a learned direction), are the two obvious upgrades.
5. **Horizon and timing.** Multi-day targets as proper (non-overlapping) tests, after
   rebuilding `merged_master` or patching `ar_dgtw_21` for 2013; messages posted
   after the close versus during the day; the 21 bp reversal after heavy bullish chatter.
6. Housekeeping: delete the stale C: copy once confirmed; remove the inert `TEXT_VARIANT`
   plumbing; the `05` registry's LASSO entries point at files that do not exist.
