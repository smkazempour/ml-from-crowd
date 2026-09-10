# Feature Implementation Status

**Last Updated**: September 7, 2026 (Section 8: text-embedding track integrated)

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

## 8. Text-Embedding Track (beyond the original 51; optional)

**Status: full 15-year encode completed 2026-09-09 (3,505,815 stock-days, 5.2 GB); downstream `add_text_features` / model runs not yet done**

The raw S3 export includes message text (`messages/`, 205 files, 52 GB) that was never wired
into the pipeline. Two notebooks in `01 - feature extraction/` and one in
`02 - prepare training dataset/` add an optional text track that leaves the existing pipeline
untouched:

- `features_06_full_text_exploration.ipynb` -- exploration only: validates the `message_id`
  join and prototypes cashtag / mention extraction (groundwork for Features 49-51, which
  remain unimplemented). No pickle output.
- `features_08_text_embeddings.ipynb` -- encodes every CRSP-matched message with
  `all-MiniLM-L6-v2` (384-dim, L2-normalised) and mean-pools to the stock-day; a message is
  counted toward every symbol it mentions. Output
  `text_embeddings_mlcrowd/text_embeddings_stock_day.pkl` (`symbol, date, embed_n,
  embed_000..embed_383`), deliberately **outside** `features_mlcrowd/`. No return label is
  used. Checkpointed join pass and per-year encoding; Section 7 prints the measured cost
  estimate before the multi-day full run.
- `add_text_features.ipynb` -- attaches the embeddings to the training data as
  `merged_master_text=<mode>.pkl` in one of three forms (`TEXT_MODE`): `raw` (384 columns),
  `pca` (top-K components, loadings fit pre-OOS), or `supervised` (walk-forward ridge
  `text_score`), always with `text_n`. Model notebooks select a variant with
  `TEXT_VARIANT`; 04/05 notebooks resolve the tagged prediction files with the same switch.

Review findings, corrections to the contributed code, and validation results are in
`01 - feature extraction/features_08_integration_notes.md`. The `features_07`
user-skill notebook from the same branch is not part of the project.

**Run order once the full encode is done**: `features_08` Sections 8-10 ->
`add_text_features` (per mode) -> any `*_all_features` model notebook with `TEXT_VARIANT`
set -> `04`/`05` notebooks with the same `TEXT_VARIANT`.

---

## Implementation Summary

- **Total Features**: 53+ (some expanded with multiple horizons; 47b-c added for weekend/holiday), plus the optional text-embedding track (Section 8)
- **Completed**: 20 features (Features 1-3, 11, 12-13c, 28-38, 39)
- **Code Complete (pending validation)**: 31 features across features_04, features_05
- **Text track**: features_08 + add_text_features code complete and validated on 2010-2011; full encode pending
- **Blocked (missing data)**: Features 27, 49-51 (need sector mapping or message text -- text is now available via features_06's join; 49-51 are still to be implemented)
- **Deprecated**: 2 features (Feature 4; Sector Expert Ratio 27 reclassified)

---

## Next Steps

1. ~~Complete Feature 39 (Disagreement Index)~~ ✅ Completed
2. ~~Implement Abnormal Sentiment (Feature 11)~~ ✅ Completed
3. ~~Implement Intraday Session features (Features 14-15, 40-48)~~ 🚧 Code complete in `features_04_intraday_sessions.ipynb`
4. ~~Implement Flip / Conviction / First-Mover / Cohort features (5-10, 16-26)~~ 🚧 Code complete in `features_05_sentiment_dynamics_cohorts.ipynb`
5. **Validate and run** features_04 on all 15 years
6. **Validate and run** features_05 on all 15 years
7. ~~Acquire message-text source~~ (found: `messages/`, see Section 8) → implement Features 49-51 and event categories from `features_06`'s join
8. Create final aggregation notebook combining all feature pickles
9. **Run** `features_08_text_embeddings.ipynb` on all 15 years (CPU-only here; check the
   Section 7 estimate first), then `add_text_features.ipynb` → model notebooks with
   `TEXT_VARIANT` → 04/05 with the same `TEXT_VARIANT` (Section 8)
