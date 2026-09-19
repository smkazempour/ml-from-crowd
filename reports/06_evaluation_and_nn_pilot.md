# Report 06 -- Evaluation recovery and a resumable NN3 pilot

Date: 2026-09-12. This implements the first recovery increment after the interrupted
nonlinear-model work: correct the benchmark comparison, protect long runs from interruption,
and validate the new NN execution path on real data. Existing production predictions and
the historical tables in Reports 01-03 were preserved. No commit was made.

## 1. Corrected benchmark comparisons

The core benchmark is OLS on net sentiment and log volume, estimated on the tweeted sample
and the same rank target as the other models. Comparisons cover 2012-01-03 to 2022-12-30:
3,171,329 stock-days for raw returns and 2,880,016 for DGTW-adjusted returns, 2,768 trading
dates each. Every listed model shares the same stock-day sample within its target.

| Target | Core rank IC | Text+core elastic-net IC | IC gain | Paired HAC t | Core spread, bp/day | Elastic-net spread | Spread gain | Paired HAC t | Spread adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw next-day return | 0.03067 | 0.03960 | 0.00893 | 7.69 | 21.29 | 23.00 | 1.70 | 0.81 | 1.000 |
| DGTW next-day return | 0.02766 | 0.03619 | 0.00853 | 8.24 | 17.76 | 21.67 | 3.91 | 1.89 | 0.295 |

Elastic net uses the trailing-rank-correlation penalty selection rule (`_rc`). Its rank IC
gain survives the multiplicity adjustment on both targets. Ridge and lasso also have
significant incremental rank IC. No model's decile-spread improvement survives the stated
Bonferroni adjustment: even the DGTW lasso's 6.31 bp gain has adjusted p = 0.094.
The evidence supports improved cross-sectional ranking; an incremental economic return
claim needs stronger evidence and implementation-cost analysis.

Inference: daily paired differences, Bartlett/Newey-West standard errors with five lags,
small-sample covariance correction, two-sided normal p-values and 95% intervals.
Bonferroni covers six comparisons for the raw target and five for DGTW, separately for
each metric and target. It does not account for the project's entire prior model-search
history. Individual-model t-statistics now also use HAC. The direct Bartlett formula is
checked independently against the implementation; the API is described in the
[statsmodels HAC documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

Data and reproducibility:

- `data/recovery_raw.csv`, `data/recovery_dgtw.csv`: full model comparisons, paired intervals
  and adjusted p-values. The benchmark is the first row.
- Corresponding `_daily.csv`, `_yearly.csv` and `.json` files retain the daily series,
  yearly IC, input file metadata, exact arguments and evaluation code hashes.
- A repeat execution reproduced the rank correlations and spreads within 1e-12.
- The CLI also supports explicitly named subperiods through `--periods label:start:end`.
  These runs retain the historical 2012-2022 evaluation period; no post-encoder-cutoff
  conclusion is asserted here.

## 2. Evaluation defects fixed

`tools/prediction_metrics.py` now supplies the shared rules used by the evaluator,
composition check, value-weighted check, NN scoring and the trading sort function.

1. **Tied predictions:** a tied block shares its rank interval fractionally across bins.
   Each tied security receives equal fractional membership; market-cap weights apply after
   membership allocation. This is independent of row order. The old `rank(method='first')`
   implementation could produce a nonzero spread from completely constant predictions.
   Fractional boundaries can also change results slightly when the universe size is not
   divisible by the number of bins; the new tables intentionally supersede those sorts.
2. **Constant signals:** zero rank IC on otherwise eligible days and cash (zero leg returns)
   for the portfolio sort. Missing signals remain missing. Constant-target days have
   undefined IC, and insufficient cross-sections are excluded. Minimum bin counts use
   fractional membership counts.
3. **Prediction joins:** attach by `(date, permno)`, validate key uniqueness, and retain
   missing coverage. A rebuilt panel's row numbering can no longer silently change the
   alignment in the three evaluation tools.
4. **Mode filter:** removed from the trading notebook. When predictions were all unique,
   the previous filter still removed the minimum; tied predictions are now handled by
   the shared sorter. The working trading registry contains core OLS plus text+core ridge,
   elastic net and lasso, with `COMMON_SAMPLE=True`; all four files exist. CRSP rows are
   explicitly sorted and checked before shifting the next-day return.
5. **Horizon configuration:** the composition tool now includes 42 and 63 days. The known
   corrupt 2013 `ar_dgtw_21` values remain a data repair item; this recovery did not rebuild
   or overwrite the panel.

Decile profiles in the evaluator use date-demeaned outcomes. Their individual long/short
Sharpe ratios are descriptive diagnostics, not standalone investable portfolio returns.
The long-short spread cancels that common daily mean. No transaction-cost or factor-alpha
results are claimed by this report, and the complete 05 trading notebook was not rerun.

## 3. NN recovery and validation

The existing untracked `prediction_neural_network_text.ipynb` was completed in place.
Its architecture and training recipe remain NN3 128-64-32, batch normalization, ReLU,
Adam, temporal 80/20 validation and validation-selected weight decay. All four feature
sets are supported: core (2), all (53), textcore (388), textall (439).

- Per-month checkpoints are published atomically by each worker. A later failure does
  not discard completed months. OS-backed file locks prevent simultaneous writes and
  release when a process dies. Reused checkpoints must match the run identity, month,
  row range and finite prediction shape.
- Run identity includes source-data size/mtime, ordered feature names, training settings,
  seed convention, library versions and code hashes. Changed configurations get distinct
  directories under `Code/.runs/text_nn/`. Pilots cannot overwrite full or historical
  prediction files. A completion marker follows atomic prediction and sidecar writes.
- Explicit pilot-month selection complements `MAX_MONTHS`. Arrays are retained within the
  run directory for reuse. There is no shared `_nn_tmp/X_<model>` filename.
- Prediction coverage and daily input ranks depend only on available features. Missing
  future labels no longer remove rows before preprocessing. Training uses finite labels;
  validation and realized test scoring handle missing labels explicitly. A regression test
  changes future-label availability and verifies identical feature ranks.
- Tests also perturb test-period returns and confirm unchanged selected settings and
  predictions; inject missing training labels; simulate failure after a completed month;
  reject mismatched/nonfinite checkpoints; and interrupt atomic publication.

Real-data pilot: January 2012 and December 2022, all four feature sets, raw-return rank
target, one seed, three epochs maximum, three weight-decay candidates, two worker processes
with two PyTorch threads each. This exercises early and recent window sizes and the actual
data/parallel-training/checkpoint/output path. These intentionally truncated fits do not
support choosing an architecture or claiming superiority over the linear models.

| Inputs | Features | January 2012 worker seconds | December 2022 worker seconds | Predictions | Resume |
|---|---:|---:|---:|---:|---|
| core | 2 | 2.8 | 5.3 | 35,562 | verified |
| all | 53 | 2.9 | 6.5 | 35,562 | verified |
| textcore | 388 | 3.1 | 18.8 | 35,562 | verified |
| textall | 439 | 3.3 | 19.0 | 35,562 | verified |

Each run predicts 35,562 stock-days, including 17 rows that the draft excluded because
future returns were missing. January uses 19,773 labeled training observations; December
uses 397,885. Worker times cover fitting all three penalties plus diagnostics, excluding
loading/ranking the full table. They must not be extrapolated into a promised full-sweep
runtime: full runs use five seeds and validation-dependent early stopping, up to 100 epochs.

The immediate repeat of each run loaded both saved monthly checkpoints, rewrote neither,
and produced byte-identical prediction pickles. See `data/nn_pilot_summary.csv` for per-month
counts/times and `data/nn_pilot_runs.json` for paths, fingerprints and prediction hashes.
The pilot runner writes a `resume_verified.json` record within each completed run directory.

Verification: 14 regression tests passed; all 71 notebooks passed schema validation and
all checked notebook/code Python parsed successfully. The full-data evaluation reproduced
its ICs and spreads within 1e-12. A 21/63-lag HAC sensitivity check gives elastic-net IC-gain
t-statistics 9.22/8.75 (raw) and 8.81/8.00 (DGTW), while its spread-gain t-statistics remain
below 1.89; see `data/recovery_hac_sensitivity.csv`.


Windows sandboxing initially prevented worker-process pipes. The same bounded run completed
with multiprocessing permissions; all new model artifacts remain inside the repository's
ignored `.runs/` directory. Existing files under `D:/StockTwits/Data/` were read only.

## 4. Reproduce and continue

**Protocol-v1.1 update:** The historical pilot in this report uses the old window and
loss settings. Matched linear and NN code now lives in `tools/protocol_linear.py` and
`tools/protocol_nn.py`, sharing `tools/protocol_data.py` and
`tools/protocol_evaluate.py`. New runs use 504 fitting sessions, 126 validation sessions,
the prescribed label gaps and equal-date weighting, with 252/756 fitting-session
comparisons. See `data/protocol_v1_1_experiment.json` for the registered specification.
The revised linear sweep and evaluation are complete in
[Report 07](07_protocol_linear_results.md). Current matched NN3 work is tracked in
[NN_RUN_STATUS.md](NN_RUN_STATUS.md); this report's older pilot metrics are not results
from the revised procedure.

Use `C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe`, not the Python 3.14
on PATH. Commands below assume that interpreter and the repository root:

```text
python -B -m unittest discover -s tools -p "test_*.py"
python -B tools/run_nn_pilot.py --features core all textcore textall --resume-check
python -B tools/evaluate_predictions.py --help
```

The next stage is fully trained NN3 comparisons, followed by trees, with matched
preprocessing/control models. The linear controls currently use different input transforms
and penalty selection; pilot ICs are not a clean estimator comparison. Preserve the four
feature-set design and the core OLS benchmark. Bring past-return controls and delayed
formation forward before interpreting the text as distinct from reversal; evaluate size,
turnover and costs before an economic claim. Corpus expansion and encoder upgrades remain
later work. The cohort/conviction `features_05` output is absent from the current 53-feature
table, so its usefulness remains untested.
