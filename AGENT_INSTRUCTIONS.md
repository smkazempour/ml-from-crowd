# Agent Instructions for Financial Social Media ML Project

## Project Overview
This project applies machine learning techniques to extract signals from financial social media data (StockTwits merged with CRSP). The goal is to engineer 51 novel features from social media conversations that can predict stock returns, going beyond traditional sentiment and attention measures.

**Data Specifications:**
- **Source**: StockTwits data merged with CRSP (2010-2024, 15 years)
- **Input Location**: `C:\Users\skazempour\Documents\StockTwits\dataset\v1\data\csv\merged_with_crsp_mlcrowd\`
- **Output Location**: `C:\Users\skazempour\Documents\StockTwits\dataset\v1\data\csv\features_mlcrowd\`
- **Format**: Annual CSV files → Consolidated pickle files (one per feature group)

## Current Project Status

### ✓ Completed Tasks

1. **features_01_basic_sentiment.ipynb** (Features 1-3)
   - Bullish Ratio, Bearish Ratio, Net Sentiment
   - Processes all 15 years (2010-2024)
   - Output: `features_01_basic_sentiment.pkl`
   - Status: Validated and pushed to git

2. **features_02_volume_attention.ipynb** (Features 28-38)
   - Raw Volume, Log Volume, Unique User Count
   - Volume Difference, Log Volume Change
   - Abnormal Volume & Attention (4 horizons: 5, 21, 63, 250 days)
   - Attention Surge indicators, Silence Gap, Relative Volume
   - Attention Concentration (HHI)
   - Output: `features_02_volume_attention.pkl`
   - Status: Validated and pushed to git
   - **Key Innovation**: Integrated Fama-French trading day calendar for accurate rolling windows

### 🎯 Pending Feature Groups (in priority order)

3. **features_03_sentiment_dynamics.ipynb** (Features 4-16)
   - Sentiment Delta (change over horizons)
   - User flip analysis (bear-to-bull, bull-to-bear)
   - Conviction Index, Reversal Magnitude
   - Extreme consensus indicators (>80%, >90%)
   - After-hours vs. market-hours sentiment/volume
   - First-mover sentiment

4. **features_04_user_cohorts.ipynb** (Features 17-27)
   - Fresh Blood (new users), Re-entry volume
   - Whale vs. Minnow dominance
   - Night-Owl vs. Day-Trader ratios
   - Retention Rate, Crowding Index (Gini)
   - Specialist Ratio, Sector Expert Ratio

5. **features_05_disagreement.ipynb** (Feature 39)
   - Disagreement Index (sentiment divergence)

6. **features_06_intraday_calendar.ipynb** (Features 40-48)
   - 8 daily session breakdowns (Pre-market, Market-open, Late morning, Mid-day, Early afternoon, Closing cross, Post-market, Overnight)
   - Intraday volatility across sessions

7. **features_07_contextual.ipynb** (Features 49-51)
   - Symbol Focus (single-stock attention)
   - Co-mention Count (complexity)
   - Broadcasting/Spam Ratio

8. **features_aggregate.ipynb** (Final integration)
   - Load all 7 pickle files
   - Merge on (symbol, date) key
   - Output: `features_complete.pkl` with all 51+ features
   - Quality checks and validation

## Core Technical Decisions

### 1. Modular Feature Architecture
- **One notebook per feature group** (not per individual feature)
- **One pickle file per feature group** (all years consolidated)
- **Incremental validation**: User validates each group before moving to next
- **Git workflow**: Commit and push after each validated group

### 2. Sparse Data Handling (CRITICAL)
**Problem**: Social media data is sparse - many stock-days have zero tweets, but rolling windows must count actual trading days.

**Solution**: Complete Trading Day Grid
- Download Fama-French daily factors for official CRSP trading calendar
- Create `(symbol, trading_day)` grid using `pd.MultiIndex.from_product()`
- Fill missing days with 0 volume/sentiment
- Ensures 250-day window = exactly 1 trading year (not distorted by gaps)

**Implementation Pattern**:
```python
# In each feature calculation function:
if trading_days is not None:
    data_start = df['date'].min()
    data_end = df['date'].max()
    all_trading_days = trading_days[(trading_days >= data_start) & (trading_days <= data_end)]
else:
    all_trading_days = sorted(df['date'].unique())

complete_grid = pd.MultiIndex.from_product(
    [all_symbols, all_trading_days],
    names=['symbol', 'date']
).to_frame(index=False)
```

### 3. Rolling Window Strategy
- **Load previous + current year** for accurate rolling calculations (especially 250-day horizon)
- **Filter trading days** to match the 2-year data window (prevents memory overflow)
- **Extract only current year** results after calculation
- **Standardized horizons**: `HORIZONS = [5, 21, 63, 250]` trading days

### 4. Data Management
- **Memory Efficiency**: Work with large datasets without unnecessary copies
- **In-place Operations**: Prefer modifications that don't duplicate data
- **Careful Loading**: Only load required columns when possible
- **No Redundancy**: Avoid intermediate saves unless explicitly requested

### 5. Code Organization
- **Structured Notebooks**: Consistent pattern across all feature notebooks
  1. Setup and Configuration (directories, parameters)
  2. Load Trading Day Calendar (Fama-French)
  3. Load Sample Data for Development
  4. Define Feature Calculation Functions
  5. Test on Sample Data
  6. Visualize Features
  7. Validation Checks
  8. Process All Years (with prev+current year loading)
  9. Final Data Inspection
  10. Save to Pickle

### 6. Naming Conventions
- **Global Variables**: Use CAPITAL_LETTERS
  - Directory paths: `DATA_DIR`, `INPUT_FOLDER`, `OUTPUT_FOLDER`, `CODE_DIR`
  - File names: `INPUT_FILE`, `OUTPUT_FILE`
  - Parameters: `HORIZONS`, `WINDOW_SIZE`, `THRESHOLD`
- **Local Variables**: Use lowercase_with_underscores

## Feature Engineering Guidelines

### Standardized Horizons
All rolling/lagged features use: **H ∈ {5, 21, 63, 250}** trading days
- 5 days ≈ 1 week
- 21 days ≈ 1 month
- 63 days ≈ 1 quarter
- 250 days ≈ 1 year

### No Look-Ahead Bias
- All calculations use data strictly available at time *t*
- Rolling windows look backward only
- User history computed up to *t-1* when determining cohorts

### Feature Validation
Each feature group must include:
- **Summary statistics**: describe(), value ranges
- **Distribution plots**: histograms with appropriate scales
- **Validation checks**: 
  - Ratios sum to 1.0 where applicable
  - Bounded features within expected ranges
  - No unexpected NaN patterns
- **Top stock analysis**: Inspect features for most active symbols

## Workflow Expectations

1. **Step-by-Step Validation**: Work incrementally with user approval between groups
2. **Test → Validate → Process**: 
   - Test on sample (most recent year)
   - User validates distributions and calculations
   - Process all 15 years only after approval
3. **Git Hygiene**: Commit and push after each validated feature group
4. **Memory Monitoring**: Print grid sizes and memory usage for large operations
5. **Efficiency**: Minimize unnecessary file I/O and data copies
6. **Reproducibility**: Ensure code can be re-run with consistent results

## Response Guidelines

### When to Be Creative
- User explicitly asks to "explore" or "investigate"
- User requests "new ways" to analyze features
- User asks for suggestions on feature engineering
- User wants to discover unexpected patterns

### When to Stay Focused
- User provides specific feature definitions (follow feature_proposal.md exactly)
- User asks for particular analysis or visualization
- User requests implementation of known method
- User specifies exact output format

### Output Expectations
- **Visualizations**: Clear, publication-quality figures with proper scales
- **Tables**: Well-formatted results with appropriate precision
- **Code**: Efficient, documented, reproducible
- **Validation**: Quantitative checks confirming correctness

## Technical Requirements

### Package Management
- Import all packages at the top of notebooks
- Standard stack: pandas, numpy, matplotlib, seaborn, tqdm
- Special: pandas_datareader (for Fama-French data)

### Analysis Approach
- **Feature Engineering**: Follow feature_proposal.md specifications exactly
- **Vectorized Operations**: Use pandas/numpy operations (avoid loops when possible)
- **Memory-Aware**: Monitor grid sizes, especially with complete trading day grids
- **Temporal Integrity**: Respect trading day calendar and no look-ahead constraints

### Documentation
- Comment complex operations (especially rolling windows)
- Explain feature calculations with references to feature_proposal.md
- Document any deviations from standard approach
- Provide interpretation of validation results

## Quality Standards
- Code should be production-quality, not experimental
- Visualizations should be publication-ready
- Features should match mathematical definitions in feature_proposal.md
- Analysis should be computationally efficient for 15 years × thousands of symbols
- Validation checks should confirm correctness before moving forward

## Key Files Reference
- **feature_proposal.md**: Source of truth for all 51 feature definitions
- **features_01_basic_sentiment.ipynb**: Template for modular feature notebooks
- **features_02_volume_attention.ipynb**: Reference for handling sparse data with FF calendar
- **AGENT_INSTRUCTIONS.md**: This file - project guidelines and current status

---

**Last Updated**: December 3, 2025
