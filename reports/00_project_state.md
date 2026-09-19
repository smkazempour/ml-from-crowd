# Project state -- StockTwits text and return prediction (as of 2026-09-19)

**Completed study:** [Report 08](08_protocol_nn_results.md) contains the full
120-model comparison: 24 NN3 specifications and 96 matched linear specifications.
All 2,592 NN monthly checkpoints, evaluation and report generation completed on
September 17 at 17:07 Central. [NN_RUN_STATUS.md](NN_RUN_STATUS.md) records completion
and the two earlier interruptions. The sections below retain the historical
social-only design. The characteristic-conditioned linear experiment is now running,
before any further window optimization or neural-network runs. Its
[fixed design](CHARACTERISTIC_EXPERIMENT.md) covers 264 specifications, including a
characteristic-only benchmark, separate sentiment and attention additions, and the
original social-only models on exactly the same stock-days. See
[current run status](CHARACTERISTIC_RUN_STATUS.md) and
[reproduction instructions](../tools/CHARACTERISTIC_STUDY.md).
Market-based characteristics are available; accounting predictors are deferred
because the legacy book-to-market merge lacks publication-date timing.

**Current comparison convention:** Report 09 will measure social information relative
to a matched characteristic-only model, and the effect of adding characteristics
relative to each matched social-only model. The core-only OLS benchmark below remains
part of the historical social-only analyses; it is not the sole benchmark for this
new experiment.

**Research-design update (2026-09-12):** [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)
preserves Q1–Q6 and potential paper directions, including news, fundamentals and aggregate
markets. [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md) defines common timing,
transformation, split, loss and comparison rules, with task-specific extensions and an
explicit implementation-gap list. These are the adopted standards; historical model outputs
retain their historical specifications. Core OLS remains the reference for the current
stock-return track; future questions require the corresponding market/news/fundamental
benchmarks. The sections below describe the narrower historical pipeline.
Protocol v1.1 specifies 504 actual fitting dates plus 126 validation dates, with
252/756 fitting-date comparisons on the same validation/test periods. The code-backed
[timing convention](TIMING_CONVENTION.md) keeps t as the first assigned close and
targets close t to close t+h; it documents exact label-maturity gaps and current
calendar/target limitations. The window design is now implemented in shared
`tools/protocol_data.py`, with matched OLS/ridge/lasso/elastic-net and NN runners.
The revised full linear comparison is complete: 96 specifications, 648 month-target
jobs and 108 test months, evaluated on the corrected common samples in
[Report 07](07_protocol_linear_results.md). At the primary 504-session window, text
improves rank IC for ridge, lasso and elastic net on both targets after the registered
multiple-testing adjustment. Text + core has lower equal-weighted decile spreads than
core alone. These findings do not establish net trading profitability.
The [NN status note](NN_RUN_STATUS.md) tracks the subsequent matched NN3 study.
See `data/protocol_v1_1_experiment.json` for the frozen grids; 67 automated checks and
all 73 notebook schema checks passed. The original source data remain unchanged;
the prepared cache excludes one invalid one-day return and associated gap-crossing
diagnostic labels, with the correction and rerun provenance recorded in Report 07.

Read this first in a new session. Current linear results: Report 07. Historical results: Reports 01-03
(Report 04 = methodology review of Gu-Kelly-Xiu 2020 and Chen-Kelly-Xiu 2022); code-level
history: `01 - feature extraction/features_08_integration_notes.md`.

**Recovery update (2026-09-12):** Report 06 (`06_evaluation_and_nn_pilot.md`) gives the
corrected benchmark comparisons and the NN pilot validation. Reports 01-03 retain the
historical tables. The current evaluator uses fractional ties, cash on constant-prediction
days, economic-key joins and paired Newey-West tests. Its decile spreads supersede the
older row-order tie breaks. NN pilot artifacts and logs live under `Code/.runs/`, isolated
from the existing predictions in `Data/`.

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
   Here “51” means the additional columns in the current 53-feature table (groups 01-04).
   The cohort/conviction features in `features_05` have code but no output pickle in the
   current data, and were not tested by these comparisons. Report 06 finds significant
   incremental rank correlation, but no spread improvement surviving its Bonferroni tests.
4. Robust to the DGTW benchmark (ordering unchanged, text t-stats highest), with positive
   cumulative-horizon correlations out to 63 days (Report 02; descriptive overlapping
   outcomes, not a test of when the return is earned), strengthens with the amount of text (0.013 on single-message days to 0.084 on
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
7. **Nonlinear models on the text track** (added 2026-09-12): neural networks first, then
   gradient-boosted trees and random forests, all on `textcore` with the rank target and
   measured against the core-OLS benchmark. Follow the Gu-Kelly-Xiu recipe (1-3 hidden layers,
   ReLU, batch normalisation, Adam with early stopping on the last 20% of the window, weight
   penalty, 5-10 seed ensemble; shallow trees with depth tuned on the same split). The
   existing `03d` notebooks were built for the 2/53-feature panel (sklearn `MLPRegressor`,
   constant predictions on 2 features) and `03b`/`03c` are empty; new notebooks should read
   `text_master.pkl` and register in 04/05 like the linear text models. Give the trees a
   PLS/PCA-compressed text input as an optional comparison with the raw 388 columns.
   Report 04 describes strong raw-embedding forest results in the revised CKX paper.
   Details and priorities: `reports/04_methodology_gkx_ckx.md`
   Section 3, items 6-7 and 11.
8. **Methodology adoptions from Gu-Kelly-Xiu (2020) and Chen-Kelly-Xiu (2022)** (review
   written 2026-09-12, `reports/04_methodology_gkx_ckx.md`). In order: pairwise
   Diebold-Mariano-type test of every model against the benchmark (daily differences in rank
   correlation / spread, Newey-West); past-return controls in the benchmark; equal day
   weighting and a Fama-MacBeth ridge option; PLS/PCR on the text; Huber loss on the return
   target; portfolio reporting (predicted vs realised deciles, long/short legs, annualised
   Sharpe, turnover, drawdown, 10/20 bp costs, micro-cap exclusion); overnight vs intraday
   target split and skip-a-day test; corpus hygiene for the re-encode (single-cashtag,
   minimum length, near-duplicate flag); ensemble of the working models; tag-supervised vs
   market-reaction-supervised sentiment scores; 2022-2023 as the post-encoder-cutoff check.
