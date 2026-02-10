# Project Guide: Machine Learning from the Crowd

## What This Project Does

This project predicts daily stock returns using social media sentiment from StockTwits. Machine learning models are trained on rolling 252-day (1 year) windows and retrained monthly. The out-of-sample evaluation period is 2012-2022.

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
                    |  Merges features with CRSP prices + abnormal returns
                    v
              merged_master.pkl (16.8M rows, in Data/ folder)
                    |
                    v
              [03a through 03e]  16 ML model notebooks
                    |            Each produces a predictions_*.pkl file
                    v
              Data/predictions_*.pkl (14.5M rows each)
                    |
                    v
              [05 - trading]  trading_daily.ipynb
                    |         Loads all predictions, forms portfolios, calculates returns
                    v
              Data/trading_daily_results.pkl
                    |
                    v
              [05 - trading]  trading_daily_analysis.ipynb
                              Plots cumulative returns, runs Fama-French regressions
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
  - Derives temporal features: hour, session, is_after_hours
- **Output**: 15 yearly CSVs in `cleaned_by_year_mlcrowd/` (176M messages total)

#### merge_with_crsp.ipynb
- **Purpose**: Links StockTwits messages to CRSP stock market data
- **Input**: Cleaned yearly CSVs + CRSP daily files (`D:/CRSP/dsf_final_*.pkl`)
- **Processing**:
  - Step 1 (Explode): Each message mentions one or more symbols via `symbol_list`. This step creates one row per (message, symbol) pair. Output: `exploded_by_year_mlcrowd/` (133M rows)
  - Step 2 (CRSP Merge): Inner join on ticker symbol and date. Adds stock price, volume, shares outstanding, raw return, and abnormal returns (DGTW, CAPM, FF3, FF5, FF6) at horizons 1, 3, 5, 21, 63 days
- **Output**: 15 yearly CSVs in `merged_with_crsp_mlcrowd/` (85M rows, 40 columns)

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
- **Purpose**: Compute sentiment and volume broken down by time-of-day sessions
- **Input**: `merged_with_crsp_mlcrowd/` (message-level CSVs)
- **Features**: 8 business-day session pairs (volume + sentiment for midnight_to_morning, pre_market, market_open, late_morning, midday, early_afternoon, market_close, post_market), weekend/holiday volume and sentiment, after_hours/market_hours aggregates, intraday_sentiment_volatility
- **Output**: `features_mlcrowd/features_04_intraday_sessions.pkl`

#### merge_all_feature_files.ipynb
- **Purpose**: Merge all individual feature files into one consolidated dataset
- **Input**: All `features_mlcrowd/features_*.pkl` files
- **Processing**: Outer merge on (symbol, date); drops redundant columns (n_bullish, n_bearish, bullish_ratio, bearish_ratio, total_labeled) to avoid multicollinearity
- **Output**: `features_mlcrowd/features_master.pkl` (28M rows, 33 feature columns)

> **Note**: features_01, features_02, and features_04 read raw message CSVs because they need individual tweet data (counting sentiments, classifying by timestamp). features_03 reads features_01's output because it only needs the already-aggregated net_sentiment per stock-day.

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

#### predictive_regressions.ipynb
- **Purpose**: Earlier analysis comparing model predictions. Likely superseded by the trading analysis workflow.

---

### 05 - trading/

#### trading_daily.ipynb
- **Purpose**: Portfolio construction and performance evaluation
- **Input**: `Data/merged_master.pkl` + all `Data/predictions_*.pkl` files + `D:/CRSP/dsf_final_*.pkl` + Fama-French factors (downloaded)
- **Processing**:
  - Loads the MODELS dictionary (central registry of all 16 models)
  - Merges each model's predictions with actual returns
  - Forms decile-sorted long-short portfolios (long top 10%, short bottom 10%)
  - Applies minimum stock filters (4 or 10 stocks)
  - Calculates daily portfolio returns over 2012-2022
- **Output**: `Data/trading_daily_results.pkl` (contains portfolios dict, filtered portfolios, FF factors, model registry)

> **Important**: The MODELS dict in this notebook is the central registry. When adding a new model, it must be registered here.

#### trading_daily_analysis.ipynb
- **Purpose**: Visualization and statistical analysis of trading results
- **Input**: `Data/trading_daily_results.pkl`
- **Processing**:
  - Plots log cumulative returns (grouped by 2-feature vs all-feature models)
  - Runs Fama-French factor regressions (CAPM, FF3, FF5, FF6) with HAC standard errors
  - Generates LaTeX regression tables
- **Output**: PNG charts in `Figures/`, LaTeX tables printed to console

> **Important**: The model_groups dict in this notebook controls which models appear in each chart. When adding a new model, update both this and trading_daily.ipynb.

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
| `Documents/StockTwits/.../feature_wo_messages/` | Raw StockTwits (248 CSVs, 501M messages) |
| `Documents/StockTwits/.../cleaned_by_year_mlcrowd/` | Cleaned StockTwits (15 CSVs, 176M messages) |
| `Documents/StockTwits/.../exploded_by_year_mlcrowd/` | Exploded by symbol (15 CSVs, 133M rows) |
| `Documents/StockTwits/.../merged_with_crsp_mlcrowd/` | Merged with CRSP (15 CSVs, 85M rows) |
| `Documents/StockTwits/.../features_mlcrowd/` | Feature pickle files (01-04 + master) |
| `D:/CRSP/` | CRSP daily stock data (`dsf_final_*.pkl`, 2008-2024) |
| `Dropbox/.../Data/` | merged_master.pkl, all predictions_*.pkl, trading results |
| `Dropbox/.../Figures/` | All charts and visualizations |

---

## Naming Conventions

**Prediction files**: `predictions_{model}_{input=N}.pkl`
- model: linear_regression, lasso, elasticnet, neural_network, neural_network_tuned_Xlayer
- N: 2 (net_sentiment + log_volume) or 31 (all features)

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
