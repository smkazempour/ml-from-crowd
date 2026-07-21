# Feature Implementation Status

**Last Updated**: July 21, 2026 (added Section 8 — full-text / text-embedding track)

This document tracks the implementation status of all 51 features from the feature proposal.

## Legend
- ✅ **Completed**: Feature implemented, tested, and saved to pickle
- 🚧 **In Progress**: Currently being implemented
- ⏸️ **Deprecated**: Feature removed from implementation plan
- ⬜ **Not Started**: Not yet implemented

---

## 1. Sentiment Dynamics (Features 1-16)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 1 | Bullish Ratio | ✅ | `features_01_basic_sentiment.pkl` | |
| 2 | Bearish Ratio | ✅ | `features_01_basic_sentiment.pkl` | |
| 3 | Net Sentiment | ✅ | `features_01_basic_sentiment.pkl` | |
| 4 | Sentiment Delta | ⏸️ | N/A | Removed - overlaps with Feature 11 |
| 5 | Bear-to-Bull Flips (Count) | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | H ∈ {5,21,63,250} td. Code complete, pending validation. |
| 6 | Bull-to-Bear Flips (Count) | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | H ∈ {5,21,63,250} td. Code complete, pending validation. |
| 7 | Net Flip Count | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 8 | Bear-to-Bull Flip Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 9 | Bull-to-Bear Flip Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 10 | Conviction Index | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | K ∈ {3,5,10}. Code complete, pending validation. |
| 11 | Abnormal Sentiment | ✅ | `features_03_abnormal_sentiment.pkl` | Horizons: [1,5,21,63,250] days (renamed from Reversal Magnitude) |
| 12 | Extreme Bullish Consensus (80%) | ✅ | `features_01_basic_sentiment.pkl` | Binary indicator |
| 13 | Extreme Bullish Consensus (90%) | ✅ | `features_01_basic_sentiment.pkl` | Binary indicator |
| 13b | Extreme Bearish Consensus (80%) | ✅ | `features_01_basic_sentiment.pkl` | Binary indicator |
| 13c | Extreme Bearish Consensus (90%) | ✅ | `features_01_basic_sentiment.pkl` | Binary indicator |
| 14 | After-Hours Sentiment/Volume | 🚧 | `features_04_intraday_sessions.pkl` | Aggregated from midnight_to_morning + pre_market + post_market sessions. Code complete, pending validation. |
| 15 | Market-Hours Sentiment/Volume | 🚧 | `features_04_intraday_sessions.pkl` | Aggregated from market_open + late_morning + midday + early_afternoon + market_close sessions. Code complete, pending validation. |
| 16 | First-Mover Sentiment | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | N ∈ {5,10,50}. Code complete, pending validation. |

---

## 2. User Cohort Composition (Features 17-27)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 17 | Fresh Blood Count | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | H ∈ {21,63}. Code complete, pending validation. |
| 18 | Fresh Blood Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 19 | Re-entry Volume | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 20 | Whale Dominance | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Global expanding cumcount. Code complete, pending validation. |
| 21 | Minnow Dominance | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | < 5 lifetime posts. Code complete, pending validation. |
| 22 | Night-Owl Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Expanding (not rolling) window. Code complete, pending validation. |
| 23 | Day-Trader Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Expanding window. Code complete, pending validation. |
| 24 | Retention Rate | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 25 | Crowding Index (Gini) | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | Code complete, pending validation. |
| 26 | Specialist Ratio | 🚧 | `features_05_sentiment_dynamics_cohorts.pkl` | ≤3 unique symbols (expanding). Code complete, pending validation. |
| 27 | Sector Expert Ratio | ⏸️ | N/A | Blocked — no sector mapping in current pipeline. |

---

## 3. Volume & Attention Dynamics (Features 28-38)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 28 | Raw Volume | ✅ | `features_02_volume_attention.pkl` | |
| 29 | Log Volume | ✅ | `features_02_volume_attention.pkl` | |
| 30 | Unique User Count | ✅ | `features_02_volume_attention.pkl` | |
| 31 | Volume Difference | ✅ | `features_02_volume_attention.pkl` | |
| 32 | Log Volume Change | ✅ | `features_02_volume_attention.pkl` | |
| 33 | Abnormal Volume (Simple) | ✅ | `features_02_volume_attention.pkl` | H=21 |
| 34 | Abnormal Attention (Standardized) | ✅ | `features_02_volume_attention.pkl` | Across horizons [5,21,63,250] |
| 35 | Silence Gap | ✅ | `features_02_volume_attention.pkl` | H ∈ {21, 252} |
| 36 | Relative Volume | ✅ | `features_02_volume_attention.pkl` | |
| 37 | Attention Surge | ✅ | `features_02_volume_attention.pkl` | Across horizons [5,21,63,250] |
| 38 | Attention Concentration (HHI) | ✅ | `features_02_volume_attention.pkl` | Herfindahl Index |

---

## 4. Network & Interaction Structure (Feature 39)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 39 | Disagreement Index | ✅ | `features_01_basic_sentiment.pkl` | Merged into basic sentiment notebook |

---

## 5. Intraday & Calendar Coverage (Features 40-48)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 40 | Midnight to Morning | 🚧 | `features_04_intraday_sessions.pkl` | 00:00-09:00 (business days). Code complete, pending validation. |
| 41 | Pre-Market | 🚧 | `features_04_intraday_sessions.pkl` | 09:00-09:30 (business days). Code complete, pending validation. |
| 42 | Market Open | 🚧 | `features_04_intraday_sessions.pkl` | 09:30-10:00 (business days). Code complete, pending validation. |
| 43 | Late Morning | 🚧 | `features_04_intraday_sessions.pkl` | 10:00-12:00 (business days). Code complete, pending validation. |
| 44 | Mid-Day | 🚧 | `features_04_intraday_sessions.pkl` | 12:00-13:00 (business days). Code complete, pending validation. |
| 45 | Early Afternoon | 🚧 | `features_04_intraday_sessions.pkl` | 13:00-15:30 (business days). Code complete, pending validation. |
| 46 | Market Close | 🚧 | `features_04_intraday_sessions.pkl` | 15:30-16:00 (business days). Code complete, pending validation. |
| 47 | Post-Market | 🚧 | `features_04_intraday_sessions.pkl` | 16:00-23:59 (business days). Code complete, pending validation. |
| 47b | Weekend Activity | 🚧 | `features_04_intraday_sessions.pkl` | Volume + sentiment on weekends. Code complete, pending validation. |
| 47c | Holiday Activity | 🚧 | `features_04_intraday_sessions.pkl` | Volume + sentiment on holidays. Code complete, pending validation. |
| 48 | Intraday Volatility (Sessions) | 🚧 | `features_04_intraday_sessions.pkl` | Std dev of sentiment across 8 business day sessions. Code complete, pending validation. |

---

## 6. Symbol-Specific & Contextual (Features 49-51)

| # | Feature Name | Status | Output File | Notes |
|---|--------------|--------|-------------|-------|
| 49 | Symbol Focus | ⏸️ | N/A | Blocked — requires message text (cashtag parsing). |
| 50 | Co-mention Count (Complexity) | ⏸️ | N/A | Blocked — requires message text. |
| 51 | Broadcasting/Spam Ratio | ⏸️ | N/A | Blocked — requires message text. |

---

## 7. Event-Category & Magnitude Features (Deferred)

**Status: ⏸️ BLOCKED — requires message body text**

The merged StockTwits-CRSP dataset (`merged_with_crsp_mlcrowd/*.csv`) was created from
`feature_wo_messages` source files that do not include message text.  Event classification
(litigation, sales/contract, guidance, etc.) and typed magnitude extraction (dollar/unit/
percent/time amounts) depend on regex over message bodies and cannot be computed from the
current pipeline inputs.  Once a raw message-text source is integrated, implement as
`features_06_event_categories.ipynb` using the reference code provided (see `event_classifier.py`
and `event_magnitudes.py` in the project notes).

---

## 8. Full-Text Track (New — beyond original 51)

**Status: 🚧 Code complete, not yet run on this machine**

This project was merged with a sibling project (`ml-from-crowd - full text`) that located a
raw message-text source (`messages/` + `msg_info/`, from the same S3 export as the rest of
the pipeline, but never previously wired in) and added three notebooks to
`01 - feature extraction/`:

- `features_06_full_text_exploration.ipynb` — validates the `messages/` join (by `message_id`)
  and prototypes cashtag/mention extraction; groundwork for Features 49-51. Exploration only,
  no pickle output.
- `features_07_user_influence_accuracy.ipynb` — scores users by historical accuracy
  (walk-forward, no look-ahead) and aggregates to `skill_wtd_net_sentiment` and related
  stock-day features.
- `features_08_text_embedding_signal.ipynb` — embeds each message body with a pretrained
  sentence-transformer (`all-MiniLM-L6-v2`) and trains an online (`partial_fit`) linear model,
  walk-forward by year, to predict next-day CAPM abnormal return directly from the embedding.
  Aggregates to `text_signal_mean` / `text_signal_std` / `text_signal_n` at the `(symbol, date)`
  level. This is the feature referenced elsewhere as "feature 08" or the text-embedding signal.

**Pipeline wiring for feature 08** (completed as part of this merge):
- `02 - prepare training dataset/merge_all_feature_files.ipynb` needed no changes — it already
  globs every `*.pkl` in `features_mlcrowd/` and outer-merges on `(symbol, date)`, so
  `features_08_text_embedding_signal.pkl` is picked up automatically once produced.
- `02 - prepare training dataset/perpare_training_data.ipynb` gained a new fill-value block
  (`text_signal_mean`/`std`/`n` → 0 on missing) so `03a`'s `dropna()` doesn't discard every row
  from the cold-start year or from messageless stock-days.
- `03a - linear regression/prediction_linear_regression_all_features.ipynb` auto-includes every
  numeric column not explicitly excluded, so `text_signal_*` reaches it with no further changes
  (a `ret`-prefix leakage-filter fix was also brought over here). The 2-feature curated baseline
  notebook (`net_sentiment`, `log_volume`) is intentionally left unchanged.
- `04 - predictive regressions/*` and `05 - trading/form_portfolios.ipynb` already resolve "the
  all-features prediction file" dynamically (`find_all_features_file()`, picks the largest
  `input=N` file on disk) and read the model list back out of `trading_daily_results.pkl`, so
  they require no changes at all — they pick up the new feature automatically once `03a`'s
  all-features notebook is re-run.

**Not yet done (external to this repo, requires the user's machine):**
`features_08`'s full 15-year run is CPU-only and estimated at 1-2 days of compute (see the
notebook's own Section 7 benchmark); it has not been executed. Until it has, and until
`merge_all_feature_files` → `perpare_training_data` → `03a_all_features` are re-run afterward,
`text_signal_*` will not actually appear in `merged_master.pkl` or downstream predictions.

---

## Implementation Summary

- **Total Features**: 53+ (some expanded with multiple horizons; 47b-c added for weekend/holiday), plus 3 text-embedding-signal variables from the full-text track (Section 8)
- **Completed**: 20 features (Features 1-3, 11, 12-13c, 28-38, 39)
- **Code Complete (pending validation)**: 31 features across features_04, features_05
- **Code complete, pipeline-wired, pending the full 15-year run**: `text_signal_mean/std/n` (features_08_text_embedding_signal.ipynb — see Section 8)
- **Blocked (missing data)**: Features 27, 49-51 (need sector mapping or message text)
- **Deprecated**: 2 features (Feature 4; Sector Expert Ratio 27 reclassified)

---

## Next Steps

1. ~~Complete Feature 39 (Disagreement Index)~~ ✅ Completed
2. ~~Implement Abnormal Sentiment (Feature 11)~~ ✅ Completed
3. ~~Implement Intraday Session features (Features 14-15, 40-48)~~ 🚧 Code complete in `features_04_intraday_sessions.ipynb`
4. ~~Implement Flip / Conviction / First-Mover / Cohort features (5-10, 16-26)~~ 🚧 Code complete in `features_05_sentiment_dynamics_cohorts.ipynb`
5. **Validate and run** features_04 on all 15 years
6. **Validate and run** features_05 on all 15 years
7. Acquire message-text source → implement features_06 (event categories + magnitudes)
8. Create final aggregation notebook combining all feature pickles
9. **Run** `features_08_text_embedding_signal.ipynb` on all 15 years (CPU-only; expect on the
   order of 1-2 days per the notebook's own runtime benchmark), then re-run
   `merge_all_feature_files.ipynb` → `perpare_training_data.ipynb` →
   `prediction_linear_regression_all_features.ipynb` → `04`/`05` notebooks in order to get
   `text_signal_*` flowing through predictions, portfolios, and trading results (see Section 8)
