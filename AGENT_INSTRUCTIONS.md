# Agent Instructions for Financial Social Media ML Project

## Project Overview
This project applies machine learning techniques to extract signals from financial social media data (StockTwits merged with CRSP). The goal is to engineer 51 novel features from social media conversations that can predict stock returns, going beyond traditional sentiment and attention measures.

**Data Specifications:**
- **Source**: StockTwits data merged with CRSP (2010-2024, 15 years)
- **Input Location**: `C:\Users\skazempour\Documents\StockTwits\dataset\v1\data\csv\merged_with_crsp_mlcrowd\`
- **Output Location**: `C:\Users\skazempour\Documents\StockTwits\dataset\v1\data\csv\features_mlcrowd\`
- **Format**: Annual CSV files → Consolidated pickle files (one per feature group)

## Current Project Status

**Location**: All feature extraction notebooks are in the `feature extraction/` folder.

**Status Tracking**: See `FEATURE_IMPLEMENTATION_STATUS.md` for detailed tracking of all 51 features.

### ✓ Completed Features (15 total)

1. **features_01_basic_sentiment.ipynb** (Features 1-3)
   - Bullish Ratio, Bearish Ratio, Net Sentiment
   - Output: `features_01_basic_sentiment.pkl`
   - Status: ✅ Completed and validated

2. **features_02_volume_attention.ipynb** (Features 28-38)
   - Raw Volume, Log Volume, Unique User Count, Volume metrics
   - Abnormal Attention (4 horizons: 5, 21, 63, 250 days)
   - Attention Surge, Silence Gap, Relative Volume, HHI
   - Output: `features_02_volume_attention.pkl`
   - Status: ✅ Completed and validated
   - **Key Innovation**: Integrated Fama-French trading day calendar

3. **feature_39_disagreement.ipynb** (Feature 39)
   - Disagreement Index (sentiment divergence)
   - Output: `feature_39_disagreement.pkl`
   - Status: ✅ Completed Dec 24, 2025

### 🎯 Next Features to Implement

See `FEATURE_IMPLEMENTATION_STATUS.md` for complete list. Priority order:
1. User Flip Analysis (Features 5-9)
2. Conviction Index (Feature 10)
3. Abnormal Sentiment (Feature 11)
4. Extreme Consensus (Features 12-13)
5. After-Hours vs Market-Hours (Features 14-15)
6. First-Mover Sentiment (Feature 16)
7. User Cohort features (Features 17-27)
8. Intraday/Calendar features (Features 40-48)
9. Contextual features (Features 49-51)
10. Final aggregation notebook

## Core Technical Decisions

### 1. Modular Feature Architecture (Updated Dec 24, 2025)
- **One notebook per feature** or closely related feature family (e.g., features that only differ by time horizon)
- **One pickle file per feature/family** (all years consolidated)
- **Descriptive naming**: Use descriptive names without feature numbers (e.g., `feature_disagreement.ipynb` not `feature_39_disagreement.ipynb`)
- **Feature numbers tracked separately** in `FEATURE_IMPLEMENTATION_STATUS.md`
- **Incremental validation**: User validates each feature before moving to next
- **Git workflow**: Commit and push after each validated feature
- **Organized folder structure**: All feature extraction notebooks in `feature extraction/` folder

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
  7. Feature Statistics
  8. Data Quality Checks
  9. Process All Years and Save Features
  10. Save to Pickle with Verification
  11. Summary Section (formula, interpretation, output file)

- **File Organization**: All feature extraction notebooks in `feature extraction/` folder

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
- **FEATURE_IMPLEMENTATION_STATUS.md**: Detailed tracking of implementation status for all 51 features
- **feature extraction/feature_39_disagreement.ipynb**: Simple template for single-feature notebooks
- **feature extraction/features_01_basic_sentiment.ipynb**: Template for basic feature calculations
- **feature extraction/features_02_volume_attention.ipynb**: Reference for handling sparse data with FF calendar
- **AGENT_INSTRUCTIONS.md**: This file - project guidelines and current status

---

**Last Updated**: December 24, 2025
