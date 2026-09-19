# Implementation plan -- matched linear models, then neural networks (revised 2026-09-13)

**Execution update:** The user has authorized the revised linear runs followed by neural
networks. Shared data preparation, explicit fitting/validation windows, rank targets,
equal-date scaling/loss, four linear estimators, NN3, and a common registry evaluator are
implemented. The corrected full linear sweep and all registered evaluations are complete;
[Report 07](07_protocol_linear_results.md) contains the 96-model comparison.
The subsequent NN3 execution is tracked in [NN_RUN_STATUS.md](NN_RUN_STATUS.md).
Full-period NN results remain pending. The run specification is
`data/protocol_v1_1_experiment.json`.
The new `prediction_linear_regression_protocol.ipynb` and
`prediction_neural_network_protocol.ipynb` are the execution entry points.
The historical phases below describe how the project reached this point.

**Design update:** The adopted [shared protocol](EXPERIMENTAL_PROTOCOL.md) and
[research agenda](RESEARCH_QUESTIONS.md) now guide the next experiment specification.
The historical phases below are not a frozen configuration: matching preprocessing,
observation weights and validation across estimators comes before the full comparison.
The new protocol lists which rules still require implementation.
Its v1.1 revision uses 504 actual fitting dates plus 126 validation dates as the
primary model, with 252/756 fitting-date checks, and January 2014–December 2022 as the
common one-day evaluation span subject to coverage checks. The
[timing trace](TIMING_CONVENTION.md) preserves the user-confirmed first-close target
and specifies horizon-dependent label gaps. The older 252-day/80–20 settings below
are historical, not the new full-run configuration.

## Recovery implementation (2026-09-12)

The first implementation increment is documented in `06_evaluation_and_nn_pilot.md`:
shared evaluation rules and paired HAC comparisons, keyed prediction joins, the working
05 registry, and a resumable NN3 pilot on the four feature sets. The original phase list
below remains the roadmap; a completed pilot does not mean the full Phase A sweep is done.
New NN runs write under `Code/.runs/text_nn/<configuration fingerprint>/`, with per-month
checkpoints and atomic output publication. Explicit month/epoch/seed limits mark a run as
a pilot; existing `Data/predictions_*.pkl` files are preserved. Feature ranks and prediction
coverage no longer depend on whether a future return label is observed. Source hashes,
input file size/mtime, ordered feature names and library versions identify each run.

Run the bounded validation with the py313 interpreter:
`python tools/run_nn_pilot.py --features core all textcore textall --resume-check`.
Windows worker-process pipes may require execution outside an application sandbox.
The pilot runner logs each run and verifies unchanged prediction bytes and checkpoint
modification times on resume. Full sweeps, matched preprocessing for linear controls,
return controls, timing and portfolio costs remain separate subsequent work.

Principle set by the user: develop the project as far as possible on the data that exists
(`text_master.pkl`, the prediction files, the CRSP yearly files already read by the 04/05
tools) before investing in data expansions. Phases A-C need no new data. Phase D is the small
CRSP join that the remaining items need; Phase E is the untagged-message re-encode. Every
phase ends with a report and a reviewable working tree; nothing is committed unless asked;
every result table starts with the core-OLS benchmark on the same sample and target.

## 0. Environment facts the plan depends on

- Notebook runs use the Anaconda `py313` environment
  (`C:\Users\skazempour\AppData\Local\anaconda3\envs\py313\python.exe`): scikit-learn 1.8,
  PyTorch 2.14 (CPU), statsmodels, joblib. The `python` on PATH is 3.14 **without**
  scikit-learn or joblib; every headless run must call the py313 interpreter explicitly:
  `<py313>\python.exe -u tools/run_notebook.py "<notebook>"`.
- No GPU; 32 CPU cores. LightGBM / XGBoost are not installed; the plan uses scikit-learn's
  `HistGradientBoostingRegressor` and `RandomForestRegressor` (both multithreaded). Installing
  LightGBM into `py313` is optional and would only shorten Phase A's tree sweeps.
- `text_master.pkl` (3.51M tweeted stock-days x 480) has keys, `f_cumret1`, 35 `ar_*`
  columns, the 53 features, `embed_n`, `embed_norm`, `embed_cos`, and the 384 dims. It has no
  lagged returns, open prices, market cap or single-day forward returns; those sit in the CRSP
  yearly files and are the subject of Phase D. Market cap for value weighting and size cuts is
  already pulled from CRSP at evaluation time by `tools/value_weighted_check.py` and
  `05/form_portfolios.ipynb`, so size-based tables do not wait for Phase D.

## Phase A -- Nonlinear models on the text track (3 days of work + 2-3 overnight sweeps)

Common design for the three new notebooks: read `text_master.pkl`; copy the text-only
notebook's `month_window`, walk-forward assembly, sidecar and output cells so files have the
same format (`index = mm_index`) and naming (`predictions_<model>_textcore_rank[_dgtw]_input=388.pkl`);
**four feature sets, the same for every model and the same names as the linear track**
[user, 2026-09-12]: `core` (net sentiment + log volume, 2), `all` (core + the other 51 = 53),
`textcore` (core + 384 dims + 2 agreement = 388), `textall` (53 + 386 = 439); the
non-embedding features are rank-transformed daily to [-1, 1] [GKX Section 2.1, footnote 29],
the embedding columns left untouched [own: they are already unit-scale]; rank target
[Report 01]; **inside each 252-day window the last 20% of dates form the validation block**
[GKX Section 1.1: training / validation / test blocks in temporal order; the 80/20 split is
the existing `03d` notebooks' convention]: hyperparameters and early stopping are chosen on
it.

Provenance tags used below: [GKX §x] = Gu, Kelly and Xiu 2020; [CKX §x] = Chen, Kelly and
Xiu, current PDF; [own] = my suggestion, not in either paper; [user] = user decision. Trees are
refit on the full window with the chosen setting; the validation-selected network predicts the
month directly. Sidecar records per-month choices, epochs, validation scores and seed spread.
Each model is registered in the 04 notebooks and in `05/form_portfolios.ipynb`.

**A1. Neural nets** -- new `03d - neural network/prediction_neural_network_text.ipynb`
(PyTorch CPU; months run in parallel worker processes, `torch.set_num_threads` per worker):

- `NN_ARCH` = `1 | 2 | 3`: 64 / 64-32 / 128-64-32 [CKX footnote 20: 128-64-32; GKX §1.7:
  geometric pyramid, 1-5 layers, NN3 best], batch norm [GKX §1.7; CKX fn. 20], ReLU [both],
  linear output; Adam with learning-rate decay [GKX §1.7 "learning rate shrinkage"; the
  plateau rule is own]; batch 10,000 [GKX Internet Appendix, from memory]; weight decay grid
  {1e-5, 1e-4, 1e-3} [own scale; GKX use an L1 penalty, CKX an L2 penalty in {0.1 ... 0.005}
  on a different parametrisation]; early stopping with patience 5 [GKX §1.7; CKX fn. 20];
  early stopping on validation *rank correlation* rather than loss [own: it is the metric];
  max 100 epochs [own]; **seed ensemble of 5** [GKX §1.7 use 10; 5 is a compute compromise,
  own]; standardised inputs [both].
- Cost: 500k x 388 rows, ~15 epochs, 5 seeds x 3 penalties = ~10-20 min per month on 4
  threads; 143 months over 8 workers = 4-6 h per architecture. NN3 first (GKX's best), NN1 and
  NN2 the following nights. The existing `03d` notebooks (panel, 2/53 features) stay as they
  are and are not reused.

**A2. Trees** -- new `03b - decision tree/prediction_gbrt_text.ipynb` and
`03c - random forest/prediction_random_forest_text.ipynb` (scikit-learn):

- GBRT: `HistGradientBoostingRegressor` [own choice of library; LightGBM not installed],
  depth {1, 2, 4} [GKX §1.6: boosted trees "with few leaves", depth 1-2 typical], learning
  rate 0.1 [GKX Internet Appendix, from memory], up to 500 iterations with early stopping on
  the validation block [GKX §1.6: number of trees tuned on validation], L2 leaf penalty 1.0
  and `min_samples_leaf` 100 [own; the leaf minimum mirrors CKX fn. 19]. Note CKX skip
  boosting deliberately [CKX §3.4, citing sparse learners losing when signals are weak]; we
  run it because GKX found it competitive on characteristics [GKX Table 1].
- Random forest, two configurations chosen per month on the validation block: shallow
  (depth 6, 300 trees, `sqrt` features per split) [GKX §1.6 and Figure 3: forests choose 1-5
  levels; 300 trees from the Internet Appendix, from memory] and deep (depth 20, 300 trees,
  `min_samples_leaf` 100, `max_samples` 0.3, features per split in {`sqrt`, 0.1}) [CKX fn.
  19: 10,000 trees, leaf 100, subsample {0.3, 0.5, 0.8}, depth {10, 20, 30}, features per
  split {sqrt, P/5, P/10, P/20}; 300 trees instead of 10,000 is a compute compromise, own].
  ~3-5 min per month on 32 cores; ~10 h for the sweep, overnight.
- Inputs: the four feature sets above [user], plus a fifth variant for the trees only,
  `all + 10 principal components of the embedding` computed on the window [own: neither
  paper ran trees on a compressed embedding; the suggestion combines CKX's finding that a
  forest on the raw embedding needs very heavy averaging (fn. 19, Table 7) with GKX's finding
  that PCR beat the elastic net on redundant noisy predictors (§1.4, Table 1: PCR 0.26% vs
  ENet 0.11%)]. Tag `allpca10`. Kept as a side comparison; the four sets are the design.

**A3. Interpretation** (final cells of each notebook, on the validation blocks):

- Group importance: drop in validation rank correlation when the 384 dims, the 2 agreement
  measures, the 2 core features (and, in the `all` variants, the 51 others) are zeroed in
  turn; averaged over months, normalised to sum to one [GKX §1.9 and §2.3: reduction in R2
  from setting a predictor to zero, normalised within model; grouping and the use of rank
  correlation instead of R2 are own].
- Placebo: five standard-normal noise columns appended; their importance rank relative to the
  median embedding dimension [GKX §2.3: five simulated placebo characteristics].
- Marginal-response curves for `log_volume` and `net_sentiment` (others at their median) and
  their interaction surface [GKX §2.3.1-2.3.2, Figures 6-7], from GBRT (`partial_dependence`)
  and NN3 (manual grid), saved to `Figures/text_models/marginal_*.png`.

**A4. Evaluation of the sweep** with the current evaluator plus the pairwise test from B1
(B1 is a few hours and should be written while the first sweep runs): NN1-3, GBRT, RF (both
configurations, four feature sets) against `core_rank` and the linear model on the same
feature set on the common sample, raw and DGTW targets, yearly, by message-count bucket and size tercile
(`composition_check.py`, `value_weighted_check.py` as they are).

**Deliverable**: Report 05 (nonlinear models). Registry entries for the best network and the
best tree in 04/05.

## Phase B -- Tests, ensemble and linear extensions on the same data (2 days)

**B1. Pairwise test** in `tools/evaluate_predictions.py`: `--benchmark <key>` reports, for
every other model, the daily difference in Spearman correlation and in decile spread against
the benchmark with a Newey-West standard error (statsmodels OLS on a constant, HAC, 5 lags)
and the Bonferroni critical value for the number of comparisons (GKX eq. 20 applied to the
ordering metrics; CKX Table 4). Also: annualised Sharpe of the daily D10-D1 series and of the
long and short legs, and a `--periods` option that adds 2022-2023 (post-encoder-cutoff) next to
the full sample and the yearly table. No new data.

**B2. Ensemble**: `tools/ensemble_predictions.py` rank-standardises the listed prediction
files by date, averages them, writes `predictions_ensemble_<name>_input=0.pkl` with a sidecar
listing the members. Member sets: elastic net + lasso; elastic net + best NN + best tree.

**B3. Linear extensions** in `03a/prediction_linear_regression_text_only.ipynb`, each a
switch with an output tag so existing files stay valid:

| switch | values | implementation | tag |
|---|---|---|---|
| `DAY_WEIGHT` | `0/1` | rows scaled by `sqrt(1/N_t)` before the Gram products so every day has equal total weight; weighted mean/sd for standardisation | `_dw` |
| `LAMBDA_GRID` (ridge) | extended | add `1e5, 1e6` to reach CKX's heavy-shrinkage region | none |
| `ESTIMATOR` | `huber` | IRLS on the window rows, `epsilon` in {1.35, 2.0, 3.0} on the standardised residual scale; return target only; ~1 min per month | `huber` |
| `ESTIMATOR` | `fmridge` | one `388 x 388` ridge solve per day of the window, coefficients averaged; lambda grid as ridge | `fmridge` |

Runs (2-5 min each, Huber ~2 h): `textcore` enet `_rc` `_dw`; `textcore` ridge wide grid;
`textcore` Huber (return target); `textcore` fmridge; raw and DGTW targets.

Parked until needed (user decision 2026-09-12): PCR / PLS on the embedding (how many
directions carry the signal). Both are cheap additions to the same notebook if wanted later.

**Deliverable**: Report 06: formal tests of every model against the benchmark, the ensemble,
Huber vs rank target, FM vs pooled ridge, day weighting. Update `00_project_state.md` Section 4.

## Phase C -- Portfolios, turnover and costs on the current models (2 days)

Uses `merged_master`, the prediction files and the CRSP cap/return columns the 05 notebook
already loads; no new data.

- `05/trading_library.py`: `form_portfolio_holdings()` returning daily weight vectors (long
  and short, EW or cap-weighted, deciles or quintiles); `smooth_weights(theta)` implementing
  CKX's exponentially weighted calendar-time rule (only a fraction `theta` of the target
  portfolio is traded per day); `turnover()` with the GKX/CKX formula on drifted weights;
  `net_returns()` with 10 bp (large) / 20 bp (small) per unit turnover, size by the daily NYSE
  20th percentile of cap (exchange code from CRSP).
- `05/form_portfolios.ipynb`: registry replaced by the working set -- `core_rank`
  (benchmark), `textcore` enet `_rc`, lasso `_rc`, ensemble, best NN, best tree -- with
  `COMMON_SAMPLE = True`; stale entries removed (`lr_text`, `ridge_text_n_dm`, the LASSO files
  that do not exist); EW and cap-weighted; deciles and quintiles; `theta` sweep {0.1, ...,
  1.0}; gross and net series, turnover, max drawdown, worst month, long/short legs; variants:
  all tweeted stocks, excluding below-NYSE-20th-percentile, small-cap tercile only. Results
  to `Data/trading_daily_results.pkl` and `reports/data/portfolios_*.csv`.
- `05/time_series_regressions.ipynb`: add the short- and long-term reversal factors
  (pandas-datareader) to the daily regressions; monthly-aggregated regression of the
  `theta = 0.1` portfolio on FF6 (CKX Section 4.4); HAC errors as now.

**Deliverable**: Report 07 (economic framing): gross and net Sharpe by `theta`, turnover,
break-even cost, alphas, drawdowns; the small-cap question answered with numbers.

## Phase D -- Small CRSP join, then the items that need it (2 days; deferred by decision)

The only data change short of a re-encode: a left join from the CRSP yearly files onto
`text_master` on (`permno`, `date`), columns prefixed `px_` so the model notebook's
53-feature assertion still holds (`px_` added to its exclusion list). ~10 min to rebuild.

| column | source | unlocks |
|---|---|---|
| `px_lret1, px_lret5, px_lret21` | `l_ret*` | past-return controls in the benchmark (`FEATURE_SET = core+ret`, `embed+norm+core+ret`); 5 x 5 sort on past 5-day return x prediction; reversal spread on all vs tweeted stocks (CKX Table 17 / Figure 9) |
| `px_fret1 .. px_fret6` | `f_ret1..6` | delayed-formation table Day+1 ... Day+6 by size tercile (CKX Table 18 / Figure 10); skip-a-day target |
| `px_ret_on`, `px_ret_id`, `px_ret_o2o` | `openprc, prc, cfacpr` | overnight / intraday split of the next-day return; CKX open-to-open target for after-close messages; split by after-hours message share from `features_04` |
| `px_cap, px_exchcd, px_size_dec` | `cap, exchcd, size_dec` | convenience only (the tools already fetch cap from CRSP) |

Runs: `core+ret` OLS and `textcore+ret` enet `_rc` (rank, raw and DGTW); `core` and
`textcore` enet on the four timing targets; the composition tool gains the delayed-formation
and reversal tables. **Deliverable**: Report 08 (past-return controls and timing): whether the
text lead survives reversal and momentum controls; where in the next day it is earned; what a
one-day skip leaves.

## Phase E -- Untagged-message track and corpus hygiene (deferred; compute-bound)

Decisions needed first: (a) the benchmark on stock-days with only untagged messages (proposal:
net sentiment 0 plus a "no tagged message" indicator, log volume over all messages); (b) keep
message-level vectors (proposal: yes, float16, ~75 GB). Steps: cleaning with the tag filter
off and a `tagged` flag; `merge_with_crsp`; `features_08` with `SAVE_MESSAGE_EMBEDDINGS`, a
single-cashtag flag, a minimum-token filter, and a post-encode near-duplicate flag (cosine
>= 0.8 against the same stock's messages over the prior five trading days, CKX's rule); three
stock-day embeddings (all / tagged / untagged); `build_text_master` v2; repeat Reports 02-03.
~54 h CPU; it contends with Phase A's sweeps, so not before they finish.

## Sequencing and effort

| phase | work | compute | new data |
|---|---|---|---|
| A nonlinear models | 3 d | 2-3 nights | none |
| B tests, ensemble, linear extensions | 2 d | ~3 h | none |
| C portfolios and costs | 2 d | ~1 h | none |
| D CRSP join + past-return / timing items | 2 d | ~1 h | small join, 10 min |
| E untagged track | 2 d + decisions | ~54 h CPU | re-encode |

Phases A-C are about seven working days and use only existing files. B1 (the pairwise test)
is written during A's first overnight sweep so that Report 05 already carries it. Each phase
ends with a report and an update to `00_project_state.md`; the tree is left for review before
anything is committed.

## Defaults assumed for Phase A (say so if any should change)

1. Trees use scikit-learn's histogram boosting and random forest; LightGBM not installed.
2. Neural nets in PyTorch on CPU, 5-seed ensemble; NN3 first, ~5 h per architecture.
3. Hyperparameters for NN/trees chosen on the last 20% of each window (GKX), not by the
   walk-forward rule used for the linear penalties; the sidecar records both scores so the
   two rules can be compared afterwards.
4. Rank target throughout; raw and DGTW; the four feature sets `core`, `all`, `textcore`,
   `textall` for every model, nothing compressed.
5. Overnight sweeps are launched detached with `tools/run_notebook.py` at low priority so
   the machine stays usable.
