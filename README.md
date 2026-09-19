# Project Guide: Machine Learning from the Crowd

## What This Project Does

For the current research state, start with [reports/00_project_state.md](reports/00_project_state.md).
The revised 96-model linear comparison is complete:
[reports/07_protocol_linear_results.md](reports/07_protocol_linear_results.md).
Text improves return ranking, but equal-weighted portfolio gains are not established.
The [complete NN3 comparison](reports/08_protocol_nn_results.md) finished September 17, 2026:
24 NN3 specifications across all three fitting windows, evaluated alongside 96 linear models.
Completion evidence and run history are in [reports/NN_RUN_STATUS.md](reports/NN_RUN_STATUS.md).
The [preliminary NN3 report](reports/08a_protocol_nn_preliminary_results.md) remains available
as the earlier snapshot covering only the 504- and 252-session windows.
For a separate server run that finishes NN3 and compares network depth, use
[the server package and instructions](server_nn/README.md). It exports the corrected
data, completed linear baselines and compatible NN checkpoints into a portable snapshot.
The [server handoff](reports/SERVER_RUN_HANDOFF.md) identifies the prepared transfer folder and validation results.
The earlier evaluation recovery and historical NN pilot remain in
[Report 06](reports/06_evaluation_and_nn_pilot.md).
The adopted shared procedure is in [reports/EXPERIMENTAL_PROTOCOL.md](reports/EXPERIMENTAL_PROTOCOL.md).
Its first-close timing trace and concrete training-window design are in
[reports/TIMING_CONVENTION.md](reports/TIMING_CONVENTION.md).
The living agenda, including possible directions for multiple papers, is in
[reports/RESEARCH_QUESTIONS.md](reports/RESEARCH_QUESTIONS.md).
Older pipeline counts and model registries below are historical; the current text table
has 480 columns and its 53 non-text features exclude the unfinished cohort/conviction group.

This project predicts daily stock returns using StockTwits sentiment, attention, engineered
features and message embeddings. The revised experiment uses 504 fitting sessions plus
126 validation sessions, monthly retraining, and a common January 2014 to December 2022
test period. The 252- and 756-session fitting windows are prespecified comparisons.

Use the new [linear protocol notebook](03a%20-%20linear%20regression/prediction_linear_regression_protocol.ipynb)
and [NN protocol notebook](03d%20-%20neural%20network/prediction_neural_network_protocol.ipynb)
for these experiments. Both call the same input preparation and split code in
`tools/protocol_data.py`; `tools/protocol_evaluate.py` evaluates their prediction registries.
The frozen grids and comparison families are in
[the experiment specification](reports/data/protocol_v1_1_experiment.json).
Large prepared arrays, monthly checkpoints and predictions stay under ignored `.runs/`;
the historical notebooks and `Data/` predictions retain their original settings.

---

## Data Pipeline Overview

The project follows a linear pipeline. Each phase produces outputs consumed by the next:

```
Raw StockTwits (501M messages)
    |
    v
[00 - data cleaning]  data_cleaning.ipynb
    |                  Filters to Bullish/Bearish, maps to trading dates
    v
Cleaned StockTwits (176M messages, 15 yearly CSVs)
    |
    v
[00 - data cleaning]  merge_with_crsp.ipynb
    |                  Explodes symbol lists, merges with CRSP stock data
    v
merged_with_crsp_mlcrowd/ (85M rows, 15 yearly CSVs)
    |
    +--> [01 - feature extraction]  features_01  (basic sentiment)
    |                                features_02  (volume & attention)
    |                                features_04  (intraday sessions)
    |                                    |
    |                   features_01 --> features_03  (abnormal sentiment)
    |                                    |
    |                                    v
    |                              merge_all_feature_files.ipynb
    |                                    |
    |                                    v
    |                              features_master.pkl (28M rows, 33 columns)
    |                                    |
    v                                    v
    +------>  [02 - prepare training dataset]  perpare_training_data.ipynb
    |               |  Merges features with CRSP prices + abnormal returns
    |               v
    |         merged_master.pkl (16.8M rows, in Data/ folder)
    |               |
    |               |      [01 - feature extraction]  features_08_text_embeddings.ipynb
    |               |          (message text -> sentence embeddings, mean-pooled per stock-day;
    |               |           kept OUTSIDE features_mlcrowd/, never merged into features_master)
    |               |                          |
    |               |                          v
    |               |      text_embeddings_mlcrowd/text_embeddings_stock_day.pkl
    |               |                          |
    |               +--------------------------+
    |               v
    |         [02 - prepare training dataset]  build_text_master.ipynb   (text track)
    |               |  embeddings + panel keys/target/features -> Data/text_master.pkl
    |               |  (one row per tweeted stock-day, 3.5M rows x 478 columns)
    |               v
    |         [03a]  prediction_linear_regression_text_only.ipynb
    |               |  walk-forward OLS on the 384 embedding dims only
    |               |  -> Data/predictions_linear_regression_textonly_input=384.pkl
    |               v
    +------>  [03a through 03e]  16 ML model notebooks (+ the text-only model above)
                    |            Each produces a predictions_*.pkl file
                    v
              Data/predictions_*.pkl (14.5M rows each)
                    |
                    v
              [05 - trading]  form_portfolios.ipynb
                    |         Loads selected predictions, forms portfolios, calculates returns
                    v
              Data/trading_daily_results.pkl
                    |
                    +--> [05 - trading]  plot_cumulative_returns.ipynb
                    |                    Plots log cumulative returns by model group
                    |
                    +--> [05 - trading]  time_series_regressions.ipynb
                                         Fama-French factor regressions, LaTeX tables
```

---

## Folder Structure

```
42 - Machine learning from the crowd/
|
+-- Code/                     Git repository root
|   +-- 00 - data cleaning/
|   +-- 01 - feature extraction/
|   +-- 02 - prepare training dataset/
|   +-- 03a - linear regression/
|   +-- 03b - decision tree/           (empty, deprecated)
|   +-- 03c - random forest/           (empty, deprecated)
|   +-- 03d - neural network/
|   +-- 03e - penalized regressions/
|   +-- 04 - predictive regressions/
|   +-- 05 - trading/
|   +-- code_from_other_project/       (legacy reference code)
|
+-- Data/                     Model predictions and merged master dataset
+-- Figures/                  Charts and visualizations
```

---

## Detailed File Reference

### 00 - data cleaning/

#### data_cleaning.ipynb
- **Purpose**: Cleans raw StockTwits data
- **Input**: 248 raw CSV files from `Documents/StockTwits/dataset/v1/data/csv/feature_wo_messages/`
- **Processing**:
  - Filters to messages with Bullish or Bearish sentiment (35% of total)
  - Converts timestamps to US/Eastern timezone
  - Maps each message to a trading date (next market close after the message)
  - Derives temporal features: hour, session (HHMM int), is_after_hours, business_day, is_weekend, is_holiday
- **Output**: 15 yearly CSVs in `cleaned_by_year_mlcrowd/` (176M messages total)

#### merge_with_crsp.ipynb
- **Purpose**: Links StockTwits messages to CRSP stock market data
- **Input**: Cleaned yearly CSVs + CRSP daily files (`D:/CRSP/dsf_final_*.pkl`)
- **Processing**:
  - Step 1 (Explode): Each message mentions one or more symbols via `symbol_list`. This step creates one row per (message, symbol) pair. Output: `exploded_by_year_mlcrowd/` (133M rows)
  - Step 2 (CRSP Merge): Inner join on ticker symbol and date. Adds stock price, volume, shares outstanding, raw return, and abnormal returns (DGTW, CAPM, FF3, FF5, FF6) at horizons 1, 3, 5, 21, 63 days
- **Output**: 15 yearly CSVs in `merged_with_crsp_mlcrowd/` (85M rows, 43 columns)

> **Note**: The `merged_with_crsp_mlcrowd/` folder is the primary input for feature extraction. It contains message-level data (one row per tweet per symbol per trading date) with matched CRSP stock data.

---

### 01 - feature extraction/

All notebooks in this folder aggregate message-level data to **stock-day level** features.

#### features_01_basic_sentiment.ipynb
- **Purpose**: Compute sentiment ratios per stock-day
- **Input**: `merged_with_crsp_mlcrowd/` (message-level CSVs)
- **Features**: net_sentiment, bullish_ratio, bearish_ratio, extreme_bullish_80/90, extreme_bearish_80/90, disagreement_index
- **Output**: `features_mlcrowd/features_01_basic_sentiment.pkl` (3.5M stock-days)

#### features_02_volume_attention.ipynb
- **Purpose**: Compute volume, attention, and user engagement metrics
- **Input**: `merged_with_crsp_mlcrowd/` (message-level CSVs)
- **Features**: raw_volume, log_volume, unique_user_count, volume_diff, log_volume_change, abnormal volume/attention (5/21/63/250 day horizons), attention_surge, silence_gap_hours, relative_volume, attention_hhi
- **Output**: `features_mlcrowd/features_02_volume_attention.pkl` (14M rows -- includes zero-volume days from complete trading grid)

#### features_03_abnormal_sentiment.ipynb
- **Purpose**: Compute deviation of current sentiment from its moving average
- **Input**: `features_mlcrowd/features_01_basic_sentiment.pkl` (NOT the raw CSVs -- this is a second-order feature derived from features_01)
- **Features**: abnormal_sentiment at 1, 5, 21, 63, 250 day horizons
- **Output**: `features_mlcrowd/features_03_abnormal_sentiment.pkl` (28M rows -- complete trading grid with imputed neutral sentiment for missing days)

#### features_04_intraday_sessions.ipynb
- **Purpose**: Compute sentiment and volume broken down by time-of-day sessions, weekends, and holidays
- **Input**: `merged_with_crsp_mlcrowd/` (message-level CSVs). Uses `session` (int64 HHMM), `business_day`, `is_weekend`, `is_holiday` columns directly.
- **Features** (25 columns):
  - 8 business-day session pairs (volume + sentiment): midnight_to_morning (00:00-09:00), pre_market (09:00-09:30), market_open (09:30-10:00), late_morning (10:00-12:00), midday (12:00-13:00), early_afternoon (13:00-15:30), market_close (15:30-16:00), post_market (16:00-23:59)
  - Weekend and holiday volume + sentiment (2 pairs)
  - After-hours aggregate (midnight_to_morning + pre_market + post_market)
  - Market-hours aggregate (market_open + late_morning + midday + early_afternoon + market_close)
  - Intraday sentiment volatility (std dev across 8 business-day sessions)
- **Output**: `features_mlcrowd/features_04_intraday_sessions.pkl`
- **Status**: Code complete, pending validation and full run

#### features_06_full_text_exploration.ipynb
- **Purpose**: Exploration only. Confirms that the raw StockTwits export includes message text (`messages/`, 205 files, 52 GB) and per-message keywords (`msg_info/`), validates the `message_id` join against `feature_wo_messages/`, and prototypes cashtag / mention extraction as groundwork for Features 49-51. No pickle output.

#### features_08_text_embeddings.ipynb
- **Purpose**: Encode the text of every CRSP-matched message with a pretrained sentence-transformer (`all-MiniLM-L6-v2`, 384-dim, L2-normalised) and mean-pool to the stock-day. Raw material for the text track, not a model input by itself; no return label is used.
- **Input**: `merged_with_crsp_mlcrowd/` (message universe: message_id, symbol, trading date) + raw `messages/` text, joined on `message_id`. A message is encoded once and counted toward every symbol it mentions, as in features_01/02/04.
- **Processing**: universe build (cached) -> checkpointed join pass with a line-based record reader (the files are not year-chunked and break CSV tokenizers) -> chunked encoding with fan-out aggregation -> per-year checkpoints -> combine.
- **Output**: `text_embeddings_mlcrowd/text_embeddings_stock_day.pkl` (`symbol, date, embed_n, embed_000..embed_383`), kept **outside** `features_mlcrowd/` so the generic feature merge never picks it up.
- **Status**: Full 15-year encode completed 2026-09-09 (43.6 h on CPU; `text_embeddings_stock_day.pkl`, 3.5M stock-days, 5.2 GB). Requires `sentence-transformers` in the `py313` env. See `features_08_integration_notes.md`.

#### merge_all_feature_files.ipynb
- **Purpose**: Merge all individual feature files into one consolidated dataset
- **Input**: All `features_mlcrowd/features_*.pkl` files
- **Processing**: Outer merge on (symbol, date); drops redundant columns (n_bullish, n_bearish, bullish_ratio, bearish_ratio, total_labeled) to avoid multicollinearity
- **Output**: `features_mlcrowd/features_master.pkl` (28M rows, 33 feature columns)

> **Note**: features_01, features_02, and features_04 read raw message CSVs because they need individual tweet data (counting sentiments, classifying by timestamp). features_03 reads features_01's output because it only needs the already-aggregated net_sentiment per stock-day. features_06 and features_08 read the separate raw `messages/` text source, which no other notebook uses.

---

### 02 - prepare training dataset/

#### perpare_training_data.ipynb
- **Purpose**: Merge features with CRSP stock data and abnormal returns to create the final ML-ready dataset
- **Input**: `features_mlcrowd/features_master.pkl` + `D:/CRSP/dsf_final_*.pkl`
- **Processing**:
  - For each year: load features and CRSP data, compute abnormal returns (DGTW, CAPM, FF3, FF5, FF6) at horizons 1, 3, 5, 10, 21, 42, 63 days
  - Merge on ticker/date (left join from CRSP side)
  - Impute missing StockTwits features with 0 (no tweets = neutral/no signal)
  - Consolidate all years and delete intermediate files
- **Output**: `Data/merged_master.pkl` (16.8M rows). This is the single input file for all ML models.

#### build_text_master.ipynb (text track)
- **Purpose**: Join the stock-day embeddings from features_08 with the training panel into one compact table for the text models, without touching `merged_master.pkl`. Nothing is fitted, compressed or imputed.
- **Input**: `Data/merged_master.pkl` + `text_embeddings_mlcrowd/text_embeddings_stock_day.pkl`, inner-joined on (`ticker`, `date`) exactly as `perpare_training_data.ipynb` aligns features.
- **Output**: `Data/text_master.pkl` (3,514,785 rows x 478 columns, 7.6 GB): `mm_index` (the row label of the same stock-day in `merged_master`, carried by prediction files as `index`), keys, `f_cumret1`, the 35 `ar_*` columns, the 53 features, `embed_n` and `embed_000..embed_383`. One row per panel stock-day that has message text (22.6% of the panel); duplicate ticker-days (two PERMNOs) are kept as in the baselines.
- **Why a separate table**: only stock-days with messages can carry text, and attaching 384 float32 columns to the full 15.5M-row panel would take ~24 GB. The 53 baseline features are carried along so that "all features + text" can be estimated on the same tweeted stock-days.
- **History**: replaces the contributed `add_text_features.ipynb` (PCA / walk-forward-ridge builder), removed on 2026-09-11. The `TEXT_VARIANT` switches left in the model and 04/05 notebooks are legacy and inert.

---

### 03a - linear regression/

#### prediction_linear_regression.ipynb
- **Purpose**: OOS predictions using OLS linear regression with 2 features
- **Input**: `Data/merged_master.pkl`
- **Features**: net_sentiment, log_volume
- **Target**: f_cumret1 (next-day cumulative return)
- **Method**: Rolling 252-day window, retrain monthly, predict daily. 16 parallel cores.
- **Output**: `Data/predictions_linear_regression_input=2.pkl` (14.5M predictions)

#### prediction_linear_regression_all_features.ipynb
- **Purpose**: Same as above but with all 31 features (dynamically selected from merged_master)
- **Output**: `Data/predictions_linear_regression_input=31.pkl`

#### prediction_linear_regression_text_only.ipynb (text track)
- **Purpose**: OOS predictions from the **384 embedding dimensions only** -- no sentiment, volume or attention features and no message count -- with the same monthly-refit, 252-trading-day rolling-window OLS design as the other 03a notebooks.
- **Input**: `Data/text_master.pkl`
- **Output**: `Data/predictions_linear_regression_textonly_input=384.pkl` (3,480,379 predictions, 2012-2023, tweeted stock-days only; `index` = `merged_master` row label). The distinct model name keeps it out of the `find_all_features_file()` resolvers; it is registered explicitly as `lr_text` in the 04/05 notebooks, which also gained a `COMMON_SAMPLE` toggle to compare all models on the same stock-days.
- **Switches**: `ESTIMATOR` (`ols` | `ridge` | `lasso` | `enet`; penalties chosen walk-forward from the previous 12 months, criterion `TEXTONLY_SELECT` = `sse` | `rankcorr`), `TARGET_DEMEAN` (date fixed effects inside each window), `FEATURE_SET` (`embed` | `embed+norm` | `core` | `embed+norm+core` | `all` | `embed+norm+all`), `RANK_TARGET`, `TARGET_COL` (`f_cumret1` | `ar_dgtw_1`); environment variables (`TEXTONLY_*`, `RANK_TARGET`, `TARGET_COL`) override them for headless runs. The elastic net is an L1 path on the ridge-augmented Gram matrix (see `reports/03_lasso_elasticnet.md`). Output name encodes the choice: `predictions_{linear_regression|ridge|lasso|elasticnet}_{textonly|core|textcore|all|textall}[_dm][_rank][_dgtw][_rc]_input=N.pkl` plus a `.json` sidecar with the penalty path and non-zero counts. Results and the reasoning behind each step: `reports/01..03`.
- **Results (2026-09-11, common sample = 3,171,329 tweeted stock-days, 2012-2022, de-meaned by date)**, registered in 04 as `lr_text`, `lr_text_n`, `lr_text_n_dm`, `ridge_text_n`, `ridge_text_n_dm`:

| model | regressors | training target | OOS R2 | slope (s.e.) | rank corr | D10-D1 bp/day (t) |
|---|---|---|---|---|---|---|
| lr_2 | net sentiment + log volume | raw | +0.000214 | 1.02 (0.11) | 0.0228 | 17.4 (6.8) |
| lr_all | 53 features | raw | -0.000896 | 0.34 (0.08) | 0.0146 | 24.2 (10.8) |
| lr_text | 384 dims, OLS | raw | -0.001541 | 0.06 (0.02) | 0.0076 | 5.8 (3.4) |
| lr_text_n | 384 + norm/cos, OLS | raw | -0.001551 | 0.10 (0.02) | 0.0126 | 11.6 (6.1) |
| lr_text_n_dm | 384 + norm/cos, OLS | date-de-meaned | -0.001027 | 0.15 (0.03) | 0.0151 | 10.6 (5.7) |
| ridge_text_n | 384 + norm/cos, ridge | raw | +0.000004 | 0.67 (0.26) | 0.0100 | 4.6 (2.4) |
| ridge_text_n_dm | 384 + norm/cos, ridge | date-de-meaned | +0.000018 | 0.87 (0.28) | 0.0161 | 9.8 (4.8) |

  Reading. (1) The two agreement measures are the single biggest improvement: with OLS they
  double the decile spread (5.8 -> 11.6 bp/day) and lift the rank correlation from 0.0076 to
  0.0126 at no cost in calibration. (2) Date-de-meaned training helps every metric for both
  estimators. (3) Ridge does what it was meant to do -- the pooled slope goes from 0.10-0.15 to
  0.67-0.87 and the OOS R2 turns (barely) positive -- but the heavy shrinkage it selects
  (lambda 10-100 in most months, occasionally 10,000, on the standardised columns) costs some
  ordering on the raw target; combined with de-meaning it keeps the ordering (rank corr 0.016,
  above `lr_all`'s 0.015 on the same stock-days, below `lr_2`'s 0.023) and is the
  best-calibrated text model. (4) Every text model has negative rank correlation in 2020 and
  2021 and its best year in 2022; `lr_2` stays positive throughout. (5) All levels are tiny:
  the best text-only OOS R2 is 0.00002 versus 0.0002 for net sentiment + volume.
  Details: `01 - feature extraction/features_08_integration_notes.md` Section 8.
- **Rank target (`RANK_TARGET=1`, also in the two baseline 03a notebooks)**: training on the daily percentile rank of the target instead of the raw return. Text models: rank correlation 0.036-0.037 versus 0.029 for both baselines, positive in every year 2012-2022; the baselines keep the larger decile spread (tail information vs broad ordering). Registered as `*_rank` in `rank_correlation` / `predictive_regressions`. Phase 0 diagnostics (model-free benchmarks, tag anatomy, tag-in-embedding) in notes Section 8c.

---

### 03d - neural network/

All neural network notebooks follow the same structure as linear regression (rolling 252-day window, monthly retraining) but use MLPRegressor from scikit-learn.

#### prediction_neural_network.ipynb
- **Purpose**: Baseline NN with fixed architecture (8, 4, 2), 2 features
- **Output**: `Data/predictions_neural_network_input=2_layers=(8, 4, 2).pkl`
- **Note**: This model performs poorly -- tends to predict constants

#### prediction_neural_network_all_features.ipynb
- **Purpose**: Same fixed (8, 4, 2) architecture but with all 31 features
- **Output**: `Data/predictions_neural_network_input=31_layers=(8, 4, 2).pkl`

#### prediction_neural_network_tuned_1layer.ipynb
- **Purpose**: Tuned single-hidden-layer NN with 2 features
- **Tuning**: For each month, test neurons in {2, 4, 8, 16, 32, 64, 128}. Select best by validation MSE (temporal 80/20 split within rolling window). Retrain on full window with optimal architecture.
- **Output**: `Data/predictions_neural_network_tuned_1layer_input=2.pkl` + `Data/nn_tuning_results_1layer_input=2.pkl`

#### prediction_neural_network_tuned_2layer.ipynb
- **Purpose**: Tuned 2-layer NN with halving architecture (l, l/2)
- **Tuning**: l in {2, 4, 8, 16, 32, 64, 128}, minimum 2 neurons in last layer
- **Output**: `Data/predictions_neural_network_tuned_2layer_input=2.pkl`

#### prediction_neural_network_tuned_3layer.ipynb
- **Purpose**: Tuned 3-layer NN with halving architecture (l, l/2, l/4)
- **Tuning**: l in {8, 16, 32, 64, 128}, minimum 2 neurons in last layer
- **Output**: `Data/predictions_neural_network_tuned_3layer_input=2.pkl`

#### prediction_neural_network_tuned_4layer.ipynb
- **Purpose**: Tuned 4-layer NN with halving architecture (l, l/2, l/4, l/8)
- **Tuning**: l in {16, 32, 64, 128}, minimum 2 neurons in last layer
- **Output**: `Data/predictions_neural_network_tuned_4layer_input=2.pkl`

#### *_all_features.ipynb variants
Each of the above has an all-features counterpart that uses 31 dynamically selected features instead of the 2 hardcoded ones. Output files use `input=31` in the filename.

---

### 03e - penalized regressions/

#### prediction_lasso.ipynb / prediction_lasso_all_features.ipynb
- **Purpose**: LASSO regression (L1 penalty) with 2 or 31 features
- **Output**: `Data/predictions_lasso_input=2.pkl` / `predictions_lasso_input=31.pkl`

#### prediction_ridge.ipynb / prediction_ridge_all_features.ipynb
- **Purpose**: Ridge regression (L2 penalty) with 2 or 31 features
- **Output**: `Data/predictions_ridge_input=2.pkl` / `predictions_ridge_input=31.pkl`

#### prediction_elasticnet.ipynb / prediction_elasticnet_all_features.ipynb
- **Purpose**: Elastic Net (L1+L2 penalty) with 2 or 31 features
- **Output**: `Data/predictions_elasticnet_input=2.pkl` / `predictions_elasticnet_input=31.pkl`

---

### 04 - predictive regressions/

This folder contains analysis notebooks that operate on the prediction files produced by the model notebooks (`Data/predictions_*.pkl`). Predictive-power diagnostics live here; portfolio formation lives in `05 - trading/`.

#### predictive_regressions.ipynb
- **Purpose**: Run cross-sectional / pooled predictive regressions of realized returns on model predictions. Used to assess whether each model's predictions have explanatory power for next-day returns.

#### plot_oos_r2.ipynb
- **Purpose**: Compute and plot out-of-sample R² for each model over the OOS period.

#### rank_correlation.ipynb
- **Purpose**: Rank-correlation analysis (e.g., Spearman) between predicted and realized returns across models and over time.

#### portfolio_returns.ipynb
- **Purpose**: Exploratory portfolio-return analysis used alongside the regression diagnostics. Distinct from the production trading pipeline in `05 - trading/`.

#### debug_deciles.ipynb
- **Purpose**: Debugging notebook for inspecting decile sorts and detecting degenerate cases (e.g., dates with too few unique predictions).

#### plot_model_performance.ipynb
- **Purpose**: Figures comparing every registered model (baselines and text-only variants, return and rank targets) on the common sample: yearly and 12-month-rolling daily rank correlation, cumulative top-minus-bottom decile return, decile profiles per model, and a summary of full-sample rank correlation and decile spread with 95% bands.
- **Input**: `merged_master.pkl` + the prediction files listed in its `MODELS` registry
- **Output**: `Figures/text_models/*.png` and `model_performance_summary.csv` (the `Figures` folder sits next to `Code`)

---

### 05 - trading/

#### form_portfolios.ipynb
- **Purpose**: Portfolio construction and daily-return calculation
- **Input**: `Data/merged_master.pkl` + selected `Data/predictions_*.pkl` files + `D:/CRSP/dsf_final_*.pkl` + Fama-French factors (downloaded)
- **Processing**:
  - Loads the MODELS dictionary (central registry; entries can be commented in/out to control which models are evaluated)
  - Merges each model's predictions with actual returns
  - Replaces each model's most-frequent prediction per date with NaN to drop degenerate days where many stocks share the same predicted value
  - Forms decile-sorted long-short portfolios (long top 10%, short bottom 10%)
  - Applies minimum stock filters (4 or 10 stocks)
  - Calculates daily portfolio returns over 2012-2022
- **Output**: `Data/trading_daily_results.pkl` (portfolios dict, filtered portfolios, FF factors, model registry)

> **Important**: The MODELS dict in this notebook is the central registry for the trading pipeline. When adding a new model, register it here. The full registry of all 16 trained models is preserved as commented entries; uncomment to include a model in the pipeline.

#### plot_cumulative_returns.ipynb
- **Purpose**: Visualize log cumulative returns of the portfolios produced by `form_portfolios.ipynb`
- **Input**: `Data/trading_daily_results.pkl`
- **Output**: PNG charts in `Figures/`, grouped by 2-feature vs all-feature models

#### time_series_regressions.ipynb
- **Purpose**: Fama-French factor regressions of portfolio returns
- **Input**: `Data/trading_daily_results.pkl`
- **Processing**: CAPM / FF3 / FF5 / FF6 regressions with HAC standard errors; emits LaTeX tables via `latex_table.py`

#### testing_trading_algorithm.ipynb
- **Purpose**: Sandbox / sanity-check notebook for the portfolio formation logic. Used to validate the decile-sorting and long-short construction against simple cases before running the full pipeline.

#### trading_library.py
- **Purpose**: Shared utility functions for portfolio construction
- **Key functions**:
  - `form_portfolio_from_signals()`: Create long/short portfolios from signal values
  - `form_portfolio_from_signals_sort()`: Create decile portfolios from signal sorts

#### latex_table.py
- **Purpose**: Utility for formatting regression results as LaTeX tables

---

### Other Files in Code/

#### feature_proposal.md
- Feature design document listing all proposed features with definitions and formulas

#### FEATURE_IMPLEMENTATION_STATUS.md
- Tracks which features from the proposal have been implemented

#### AGENT_INSTRUCTIONS.md
- Instructions for AI coding assistants working on this project

#### code_from_other_project/
- Legacy reference notebooks from a related project (not part of the active pipeline)

---

## Data Locations

| Location | Contents |
|----------|----------|
| `D:/StockTwits/Data/v1/data/csv/feature_wo_messages/` | Raw StockTwits (248 CSVs, 501M messages) |
| `D:/StockTwits/Data/v1/data/csv/cleaned_by_year_mlcrowd/` | Cleaned StockTwits (15 CSVs, 176M messages) |
| `D:/StockTwits/Data/v1/data/csv/exploded_by_year_mlcrowd/` | Exploded by symbol (15 CSVs, 133M rows) |
| `D:/StockTwits/Data/v1/data/csv/merged_with_crsp_mlcrowd/` | Merged with CRSP (15 CSVs, 85M rows) |
| `D:/StockTwits/Data/v1/data/csv/features_mlcrowd/` | Feature pickle files (01-04 + master) |
| `D:/StockTwits/Data/v1/data/csv/messages/`, `msg_info/` | Raw message text (205 files, 52 GB) and keywords -- used only by features_06/08 |
| `D:/StockTwits/Data/v1/data/csv/text_embeddings_mlcrowd/` | features_08 output: stock-day sentence embeddings + per-year checkpoints |
| `D:/CRSP/` | CRSP daily stock data (`dsf_final_*.pkl`, 2008-2024) |
| `D:/StockTwits/Data/` | merged_master.pkl, all predictions_*.pkl, trading results |
| `D:/StockTwits/Figures/` | All charts and visualizations |

---

## Naming Conventions

**Prediction files**: `predictions_{model}_{input=N}[_text={mode}].pkl`
- model: linear_regression, lasso, elasticnet, neural_network, neural_network_tuned_Xlayer
- N: 2 (net_sentiment + log_volume) or the all-features count (31 originally; 53 with features_04; more once features_05 / text columns are added)
- `_text={mode}`: legacy tag of the removed `add_text_features` track; no such files exist. The text-only model uses its own model name: `predictions_linear_regression_textonly_input=384.pkl`

**Text-track files**: `text_embeddings_mlcrowd/text_embeddings_stock_day.pkl` (features_08 output), `Data/text_master.pkl` (build_text_master output), `Data/predictions_linear_regression_textonly_input=384.pkl` (text-only model)

**Tuning results**: `nn_tuning_results_{layers}_input={N}.pkl`

**Feature files**: `features_0X_{name}.pkl`

---

## Key Design Decisions

1. **Temporal validation**: Train/validation split is chronological (first 80% / last 20% of the rolling window), not random. This is standard in finance to prevent look-ahead bias.

2. **Rolling window**: 252 trading days (~1 calendar year), retrained monthly. This balances having enough training data with adapting to changing market conditions.

3. **Trading date assignment**: Messages posted after market close are assigned to the next trading day's close. Weekend/holiday messages roll forward to the next trading day.

4. **Missing data imputation**: Days with no tweets for a stock are treated as neutral (sentiment = 0, volume = 0). This ensures rolling windows are calculated over calendar time, not just days with data.

5. **Minimum 2 neurons in last layer**: For multi-layer networks, the smallest allowed layer has 2 neurons, preventing degenerate single-neuron bottlenecks.

6. **Two feature sets**: All models are run with both 2 features (net_sentiment, log_volume) and 31 features (all available) to compare the value of additional features.

---

## Current Model Registry (16 models)

| Key | Model | Features | Architecture |
|-----|-------|----------|-------------|
| lr | Linear Regression | 2 | - |
| lasso | LASSO | 2 | - |
| elasticnet | Elastic Net | 2 | - |
| nn | Neural Network (baseline) | 2 | Fixed (8,4,2) |
| nn_tuned_1layer | NN Tuned 1-Layer | 2 | Tuned: {2..128} neurons |
| nn_tuned_2layer | NN Tuned 2-Layer | 2 | Tuned: halving (l, l/2) |
| nn_tuned_3layer | NN Tuned 3-Layer | 2 | Tuned: halving (l, l/2, l/4) |
| nn_tuned_4layer | NN Tuned 4-Layer | 2 | Tuned: halving (l, l/2, l/4, l/8) |
| lr_all | Linear Regression | 31 | - |
| lasso_all | LASSO | 31 | - |
| elasticnet_all | Elastic Net | 31 | - |
| nn_all | Neural Network (baseline) | 31 | Fixed (8,4,2) |
| nn_tuned_1layer_all | NN Tuned 1-Layer | 31 | Tuned: {2..128} neurons |
| nn_tuned_2layer_all | NN Tuned 2-Layer | 31 | Tuned: halving (l, l/2) |
| nn_tuned_3layer_all | NN Tuned 3-Layer | 31 | Tuned: halving (l, l/2, l/4) |
| nn_tuned_4layer_all | NN Tuned 4-Layer | 31 | Tuned: halving (l, l/2, l/4, l/8) |
