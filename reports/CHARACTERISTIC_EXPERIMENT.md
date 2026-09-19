# Stock characteristics and incremental social information

Specification frozen before fitting, September 19, 2026. Experiment ID:
`characteristics_v1`. Research questions: Q3 (information beyond stock characteristics
and return history), followed by Q1 (information beyond sentiment and attention).
The machine-readable design is [characteristics_v1_experiment.json](data/characteristics_v1_experiment.json).

The completed social-only study is archived in [Report 08](08_protocol_nn_results.md)
and Git commit `9ac568e`. This experiment first changes the information set, keeping
the existing linear estimation procedure fixed. Window optimization, expanded penalty
grids, alternative losses and neural-network architecture searches are subsequent work.

## Benchmark and information timing

The benchmark, `characteristics`, includes past returns and market-based stock
characteristics available at signal close t. Its forecast target remains the return
from close t to close t+1. It uses the existing social-data-covered stock-day universe;
this is not an all-stock experiment or a test of how social coverage itself is selected.

The controls are constructed from primitive daily CRSP observations, before matching
to the social-data rows. They include returns ending at t over 1, 5, 21, 63, 126 and
252 exchange sessions; 252-session momentum excluding the most recent 21 sessions;
security market capitalization; price; current and trailing 21-session turnover and
dollar volume; 21- and 63-session return volatility; the maximum daily return over
21 sessions; and 21-session Amihud illiquidity. There are 17 scalar controls, each
with an explicit missingness indicator. The prepared manifest records exact formulas.

All histories require the declared number of exchange sessions. A stock's missing
trading row must not turn a multi-session return into a one-day observation. Zero
volume is valid for activity measures; nonpositive dollar volume cannot be a denominator
for illiquidity. Calculations use no future returns. Current-day returns and trading
activity are known at close t under the existing forecasting convention; forming
an executable portfolio at that exact close remains a separate question.

Scalar controls use the same centered daily midranks as scalar social features,
neutral ranks for missing values, and fitting-only weighted scaling. Missingness
indicators retain their binary meanings. All models keep identical eligible stock-days,
even when a characteristic is unavailable. The parent social inputs, keys, labels and
evaluation panels are preserved exactly, allowing certified reuse of the 96 earlier
social-only linear predictions.

## Accounting-data limitation

The legacy daily `bm` field is joined from fiscal `datadate` months and filled forward
without a publication lag. It is excluded from predictors. The available CCM file
has no reporting/filing dates, includes some conflicting security mappings, and its
precomputed ratios were clipped using the full historical distribution. A point-in-time
accounting benchmark therefore needs a separate data audit and availability policy.
The current benchmark contains market-based characteristics, not a comprehensive set
of firm fundamentals. A remaining social increment cannot establish incremental
information beyond every accounting characteristic or news variable.

Raw returns are primary. DGTW-adjusted returns retain the earlier target definition
for a matched secondary comparison. Their upstream characteristic sorting inherits
the accounting-availability limitation; adding safe predictors does not certify the
historical DGTW benchmark as a point-in-time implementable strategy.

## Matched model grid

Use OLS, ridge, lasso and elastic net with the exact existing objectives, intercept,
candidate grids, numerical solver certificates and validation-IC selection rule.
Monthly fits use 504 fitting sessions as the primary history, with the already-declared
252/756-session sensitivity histories and a fixed 126-session validation block. The
selected fitting-block model is retained without refitting on validation. Label-end
purges, equal-date loss weights and test months (January 2014-December 2022) are unchanged.

The input sets are:

1. Existing social-only core, all social, text + core, and text + all social.
2. Characteristics alone.
3. Characteristics + sentiment, and characteristics + attention separately.
4. Characteristics + core (sentiment and attention jointly).
5. Characteristics + all social, characteristics + text + core, and characteristics + text + all social.

This is 11 input sets x 4 estimators x 2 targets x 3 fixed histories = **264 specifications**.
The 96 existing social-only specifications are reused only after exact parent-data and
prediction-source validation; 168 specifications are newly fitted. Each month-target-
history fit shares its sufficient statistics across the seven new input sets.

## Comparisons and reporting

The primary metric is equal-date mean Spearman IC. Equal- and capitalization-weighted
top-minus-bottom decile spreads are secondary, using the existing fractional-tie rules.
Compare models on the same dates and securities. Use paired daily HAC inference with
5 lags primary and 21/63 as sensitivities. Report the full 2014-2022 period first, with
2014-2018 and 2019-2022 as the existing descriptive subperiods.

Predeclared comparison families, separately within each target, metric and period:

| Family | Question | Full-study Bonferroni denominator |
| --- | --- | ---: |
| Incremental social information | Sentiment/attention separately and jointly; conditional contributions; other social features and text after characteristics | 144 |
| Adding characteristics | Characteristics + each social set versus that same social-only set | 48 |
| Estimator | Ridge, lasso and elastic net versus matched OLS | 99 |
| Fixed history | 252 and 756 sessions versus the primary 504 sessions | 88 |

The machine-readable specification names all 12 social-information edges. Full-study
denominators apply even if an interim report covers only the 504-session window.
Do not choose a new training window, estimator, feature list or penalty grid from the
test-period scores. Those choices belong to a subsequent validation-based design.

The new controller writes a separate Report 09 and separate result files. Completion
requires all requested monthly checkpoints, source/coverage checks, evaluation and
report generation; a successful training process alone is not a completed study.
