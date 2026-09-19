# Shared experimental protocol

Version: v1.1, 2026-09-12. Adopted for the current matched one-day return experiments;
extensions to other targets and information sets remain specifications for future work.

Revision v1.1 makes the user-confirmed first-close timing explicit and replaces the
varying 80/20 history splits with 252/504/756 fitting dates plus a fixed 126-date
validation block. The primary fit length is 504 dates. The code trace and
exact session-index formulas are in [TIMING_CONVENTION.md](TIMING_CONVENTION.md).

This is the recommended procedure for new experiments, written in response to the
discussion about fair linear/nonlinear comparisons and the broader research agenda.
The shared data/split pipeline and matched linear and NN runners are implemented.
Corrected linear training and evaluation are complete: 96 model specifications,
648 month-target checkpoints and 108 test months; see
[Report 07](07_protocol_linear_results.md). The standard-budget NN study completed
on September 17, 2026: [Report 08](08_protocol_nn_results.md) evaluates all 24 NN3
specifications alongside 96 linear models; [NN_RUN_STATUS.md](NN_RUN_STATUS.md)
records completion and execution history. This is not a claim that every extension
or economic validation described below has been implemented or completed.
The [research questions](RESEARCH_QUESTIONS.md) define what each experiment is intended
to answer. Existing results and predictions retain their original specifications.

The common standard is the same information set, sample, timing, fitting opportunity,
and evaluation within a comparison. Different scientific tasks require different targets
and losses. In this document, **benchmark model** means a comparison estimator;
**conditioning variables** means additional predictors such as past returns or news.

## 1. What we take from GKX and CKX

| Source | Verified procedure | Decision for this project |
|---|---|---|
| GKX, published 2020 article, Sections 1.1–1.2 and 2.1 | Monthly excess-return levels; squared loss and selected Huber variants. Characteristics ranked cross-sectionally; missing characteristics median-imputed. Initial 18-year training and 12-year validation blocks, annual tests; training expands. Fitted parameters use training alone. | Adopt temporal validation and comparable transformations. Our daily frequency, shorter history and rank target require explicitly different settings. |
| GKX, Sections 1.2.1 and 1.8 | Pooled observation loss; inverse-cross-section-size and capitalization weights also considered. Return R-squared against zero and date-level forecast comparisons. | Equal-date weighting is our proposed primary choice because coverage changes strongly. Report actual-return accuracy separately from ranking accuracy. |
| CKX, February 2026 revision, Sections 2.2–2.3 | Frozen text representations with downstream return prediction; annual rolling six-year training, two-year validation and one-year test. News timing and open-to-open returns are aligned explicitly. | Keep representation and estimator comparisons separate. Specify a forecast clock for social messages, news and prices before interpreting a trading result. |
| CKX, Sections 3.4–3.5 and 4 | Ridge and nonlinear comparisons; turnover/cost analysis, delayed formation and information-content checks. Several implementation details, including an exact normalized ridge objective and final refit policy, are not explicit. | Define those details ourselves; do not infer equivalence from matching nominal penalty values. |

Sources: [GKX, RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276)
(including its Internet Appendix);
[CKX, author research page](https://dachxiu.chicagobooth.edu/)
and [dated SSRN paper record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687).
CKX section references were checked in the local copy,
[ssrn-4416687 (1).pdf](<../../Literature/ssrn-4416687 (1).pdf>).
The local CKX copy has 78 pages and corresponds to the February 2026 revision, rather
than the 2025 label in the original Report 04. The procedures below are our proposal,
not a replication claim or a prediction that either paper's performance will transfer.

## 2. Specify the question, information set and clock first

Every experiment receives an ID and records its research-question IDs, primary contrast,
target, forecast timestamp, earliest execution timestamp, universe, data version,
transformations, training schedule, loss, tuning grid, evaluation and planned robustness.
Write these before the full comparison. Record amendments and their reasons.

The default current task is next-day **cross-sectional ordering among covered common
stocks**. Its conclusions are conditional on that coverage. The broader questions need
additional designs; a result among tagged-message stock-days is not automatically a
result for all stocks, all social messages, news, earnings or the aggregate market.

### Information and price timing

- **Canonical signal clock:** assign each message to its first market close, called t.
  The stock-day feature row aggregates messages since the previous close; its h-day
  target runs from close t to close t+h. All joins preserve this assigned date. The
  code implements the first nominal close at or after the message, including exact
  equality; [the code trace](TIMING_CONVENTION.md) records early-close limitations.
- Keep original UTC timestamps and an exchange-time/calendar mapping, including daylight
  saving, holidays and early closes. Each feature has an availability timestamp, not just
  a nominal date. Accounting releases, revised news and labels use their actual availability.
- A predictor must be observable by the forecast cutoff. A fitted transformation, selected
  hyperparameter or trained model can use only information available at its fitting cutoff.
  Contemporaneous cross-sectional feature ranks are permitted when all their inputs are
  available at formation; future outcomes never determine that ranking universe.
- Specify the target's start, end and availability timestamps. Entry prices for an economic
  portfolio must be attainable after collecting inputs and computing the signal. A full-day
  message aggregate cannot simply assume an execution at that same closing price.
- Keep close t to close t+h as the main predictive target. After-close messages belong
  to the next assigned close t, without an additional lag after assignment. The interval
  from posting to that close is outside the target by design. Current cleaning uses a
  fixed 4 p.m. cutoff, so an exchange-calendar rebuild must address early closes.
  Pre-close input buffers and delayed execution belong in separately named economic
  sensitivities; they do not redefine the main prediction clock.
- Include same-day returns among conditioning variables only to the extent those returns
  have already accrued at the forecast cutoff. A future return, realized next-day attention,
  or post-announcement news cannot become a forecasting input or an ordinary causal control.

### Universe, absence and missingness

Use stable security IDs and one row per security and forecast time; join by economic keys,
not saved row numbers. Freeze coverage using contemporaneously observable rules. Preserve
delisted securities and audit corporate actions and missing realized returns; do not select
today's surviving tickers or silently discard difficult outcomes.

For the immediate replication, use the current tagged-message common-stock universe.
For each comparison family, build one feature table from the union of required columns,
then give every estimator the same eligible fitting, validation and prediction rows.
Use explicit imputation rules instead of estimator-specific complete-case selection.
Report any coverage restriction and its economic meaning. Different families may use
different universes, but their scores are not directly comparable without a common sample.

Distinguish three cases:

1. **No activity:** volume can be zero when the extraction feed is known to be complete.
   An undefined sentiment average gets a neutral placeholder plus a no-tagged-message flag.
2. **Feature unavailable:** use the declared imputation rule and an availability indicator.
   Never treat a missing data file as zero social activity.
3. **Outcome unavailable:** retain the prediction and record why evaluation is missing.
   Do not condition feature ranks or prediction eligibility on future label availability.

In an all-stock or social/news-union extension, use a zero embedding plus an explicit
no-text flag for genuine absence; a zero vector is not evidence of neutral content.
Give baseline models the corresponding coverage indicators so the comparison isolates
content from the presence of coverage. These extensions change the feature counts.

Keep genuine repetition/attention and text novelty distinguishable. Deduplicate ingestion
errors; define repost, near-duplicate and multi-cashtag policies before testing them.
Novelty filters must use only earlier documents. Preserve tagged/untagged and message-to-
security provenance. Do not silently change the corpus midway through an estimator comparison.

## 3. Transformations shared within a comparison

### Cross-sectional return track

For scalar social features and stock characteristics, use average ranks for ties within
the fixed eligible cross-section at date t:

    x_rank(i,t) = 2 * (midrank(x_i,t) - 0.5) / N_t - 1

This centers a constant observed cross-section at zero. For a missing continuous feature,
calculate ranks among the observed eligible values and assign the missing entry the neutral
rank zero, with a missingness indicator. Record the observed count used in that feature's
denominator. If the whole feature is missing, flag it and assign zero; report the occurrence.
Binary availability indicators and categorical encodings retain their meanings.

This rank convention is a proposed new version: existing notebooks use percentile rank
directly and have a small finite-sample centering difference. Apply the new formula to all
matched models together. Daily ranking can change the pooled ordering of observations,
so it can affect tree models as well as linear models and networks.

Keep the 384 embedding coordinates unranked. Preserve the current pooling convention
and retain embedding norm/cosine agreement as separate, unranked scalar features. Do not
renormalize a stock-day vector in a way that silently removes its agreement information.
Use ablations for coordinates and agreement separately when attributing a gain to text.

After these transformations, center and scale each continuous input using fitting-block
statistics only, with equal-date weights. Apply the same fitted transformation to validation
and test inputs for every estimator. Compute these feature statistics on eligible input
rows without filtering them by future outcome observability. Drop constant fitting columns
or use scale one consistently; record the mask. Date-local ranks at validation/test time
use that date's available inputs, while the fitted scaling parameters remain unchanged.

No full-sample standardization, PCA, vocabulary fitting, feature selection, clustering used
as predictors, or supervised embedding adaptation. Any learned representation or compression
is fit inside the training procedure and selected on validation. A frozen pretrained encoder
is versioned, including tokenizer, pooling, text truncation and revision.

Daily ranks discard common shifts and magnitudes. Retain a declared native-scale robustness
using domain transforms such as log(1 + count), trained scaling, and the same data splits
for all compared estimators. When market-wide activity is a predictor, preserve it separately.

### Target and loss

| Task | Target and primary fitting loss | Validation selection | Primary evaluation |
|---|---|---|---|
| Current cross-sectional ordering | Within-date centered midrank of next-day raw return; mean squared error on ranks | Mean daily Spearman IC | Daily IC and paired IC improvement |
| DGTW robustness | Same rank construction for next-day DGTW-adjusted return | Mean daily Spearman IC | IC and paired improvement on its common sample |
| Return-level forecasting | Actual forward return or specified excess/abnormal return; squared error | Equal-date mean squared error | Out-of-sample loss and return-level R-squared |
| Robust return-level companion | Same return target; Huber loss with declared residual scale and threshold grid | Same validation MSE as the squared-loss comparison | Same untouched return-level test outcomes |
| Fundamentals/earnings magnitude | Defined release-vintage quantity or surprise; squared error, with Huber robustness | Corresponding chronological forecast loss | Loss reduction against a fundamentals baseline |
| Event occurrence/direction | Predefined event or binary outcome; log loss | Validation log loss | Log loss, calibration and Brier score |
| Aggregate market return | Return level/excess return; squared error, with Huber robustness | Time-series forecast MSE | Time-series R-squared against a feasible benchmark |

For rank targets, define:

    q(i,t) = (midrank(y_i,t) - 0.5) / N_label,t - 0.5

Only labels available at the relevant fitting/selection cutoff enter that split's target
rank; missing or immature labels remain missing. Calculate once for the comparison family's
universe and availability cutoff, not separately for each estimator. Do not rank all
eventually realized outcomes and then drop immature rows: those outcomes could already
have affected the retained ranks. Days with fewer
than 10 observed targets are ineligible for primary IC training/selection/scoring; publish
their count. Constant-target dates have undefined IC and are excluded consistently.
Predictions for otherwise eligible rows are still retained.

Rank-target squared loss estimates expected relative rank, not expected return in percent.
Do not calculate return-unit R-squared from rank predictions. Portfolio returns always use
actual realized returns, never ranks. Raw-return ranking is the proposed primary task;
DGTW and return-level fits are declared companion tasks, not replacements chosen afterward.

For a fitting block with D eligible dates and N_t labeled observations per date:

    data_loss = (1 / D) * sum_t [ (1 / N_t) * sum_i loss(y_i,t, prediction_i,t) ]

Apply this **equal-date weighting to all estimator families**, including NN minibatches
and tree fitting. A stochastic implementation must target these weights in expectation;
do not independently normalize each minibatch in a way that changes the objective.
This aligns the objective with date-averaged evaluation and reduces the mechanical weight
of later, denser StockTwits days. Equal-row fitting is a planned robustness; cap weighting
of training is a separate estimand, distinct from cap-weighted portfolio evaluation.

Keep regularization outside the normalized data loss and document its exact scaling,
intercept treatment, penalized parameters and library mapping. For example, record whether
ridge uses mean loss plus lambda times squared coefficient norm or an unnormalized sum.
NN L2 penalties and optimizer weight decay also need an explicit definition. Equal numeric
lambdas across libraries, sample sizes or papers need not mean equal shrinkage.
Do not winsorize test outcomes to improve scores. Any training-only clipping or robust scale
is estimated inside the fitting block and reported as a separate specification.

## 4. Training, validation and test schedule

Use F for actual fitting dates and V for validation dates. The proposed main comparison
holds V fixed while varying F, so changing fitting history does not also change the
period used for model selection:

| Element | Rule |
|---|---|
| Re-estimation | Monthly; complete selection using outcomes realized by the preceding month's last trading close |
| Primary fitting history | F=504 trading dates |
| Validation | V=126 trading dates, ending at the latest signal date whose h-day label is available by the selection cutoff |
| Fitting-history comparisons | F=252 and F=756, with the identical V=126 validation dates at a given horizon |
| Historical reference | Existing 252 total dates = 202 fitting + 50 validation, explicitly labeled legacy |
| Test | Following calendar month's eligible trading dates; parameters fixed for that month |
| Evaluation span | January 2014–December 2022 common h=1 comparison, subject to feature/label coverage checks; 504/126 results from January 2013 as a supplemental history |
| Longer-validation robustness | A separate comparison holding F fixed; do not mix it with the primary fitting-history comparison |
| Expanding-history robustness | A separately registered later experiment with common start and validation rules |

This supersedes v1's 252/504/756 **total-history** 80/20 splits. Two years of actual fitting
is the proposed starting point, not an established optimum. Report all three windows.
The original 2012–2022 tables remain historical references, since the text-master input
starts on 2010-06-02 and the longer new windows cannot cover that entire test period.
The [calendar audit](data/window_calendar_audit.json) verifies initial warm-up from full
CRSP-panel dates: first h=1 months are December 2011, January 2013 and January 2014
for F=252, 504 and 756 respectively. It does not yet establish all test-period feature
or target coverage. An across-window comparison including h=63 starts in July 2014.

Use the exchange calendar rather than the dates that happen to contain messages. A month
is eligible only when the required history exists. For window comparisons, score only their
shared test span and report the loss of early years; do not improve a longer-window score
by silently changing its evaluation period.

**Exact close-to-close split:** let b be the last trading close before the test month,
and use exchange-session integer indices:

    validation_last  = b - h
    validation_first = b - h - V + 1
    fitting_last     = validation_first - h - 1
    fitting_first    = fitting_last - F + 1

Thus every fitting label ends before the first validation signal close, and the last
validation label ends at b. The h excluded signal dates at each boundary implement
the purge; do not add a second independent embargo. The preceding calendar span is
F + V + 2h sessions. For the primary h=1 case, this means 504 fitting dates, a one-date
gap, 126 validation dates and one final date with an immature label: 632 sessions.

If a label is published after its closing endpoint, use that later availability and move
the block accordingly. Irregular earnings/event outcomes require their actual publication
timestamps. Current code's month-start-minus-two-calendar-days heuristic is not equivalent
to this schedule; the trace gives a weekend-boundary example. Document actual dates,
counts and dropped labels for each monthly fit.

All stocks on a date belong to the same temporal block. No random row cross-validation
or random stock-day train/test splits. Earlier outer-test outcomes may enter a later fitting
window after realization under the declared walk-forward rule; that is distinct from using
a future month's results to choose its own model.

Select penalties, architecture candidates, training duration and any ensemble weights using
validation only. For the primary matched experiment, retain parameters fitted on the fitting
block; **do not refit on fitting plus validation** after selection. A final-refit experiment
would need a shared rule for every estimator, including how NN epochs are fixed without
reusing the validation data for early stopping.

The fixed 126-date validation block gives more time diversity than the old 50-date block,
but selection can still be noisy. Report penalty instability and convergence. Keep grids
modest, include zero/simple predictors, and assess the same fitting histories across every
estimator. A grid-boundary selection motivates a documented new experiment, not an
unreported full-test-driven grid expansion.

### NN and model-specific mechanics

For the immediate NN3 comparison, start from the implemented 128–64–32 network, ReLU,
batch normalization, Adam, five fixed seeds, learning rate 0.001, batch size 10,000,
100-epoch ceiling and patience five. The current weight-decay candidates are
0.00001, 0.0001 and 0.001. This is an adapted project recipe.

Use the same seed list for every comparable NN input set. For each penalty, save each
seed's checkpoint at its best validation IC. **Proposed selection is the mean daily
validation IC of the averaged seed predictions**, matching the ensemble ultimately used
for prediction. Resolve exact ties using a recorded candidate order. Average all declared
seeds; do not choose the luckiest seed. Record learning curves, failures and selected epochs.
NN1/NN2, other penalties and architectures are separate registered candidates.

OLS, ridge, lasso, elastic net and trees use the same rows, target, normalized observation
weights and validation metric; their model-specific grids need not be numerically identical.
Publish grids and comparable search budgets before the run. Tune portfolio smoothing and
trading thresholds on validation as well, rather than on the reported test Sharpe.

## 5. Comparisons, inference and economic interpretation

### Feature and estimator comparisons

For current returns, retain core OLS using net sentiment and log volume as the recognizable
historical reference. Add an OLS/ridge benchmark under the new matched protocol.
Compare richer inputs within each estimator and different estimators on identical inputs.
The current primary feature sets have 2, 53, 388 and 439 columns; the unfinished
cohort/conviction output is not part of the 53.

For incremental social information beyond market data, the relevant contrast is
market/characteristics/history versus those same inputs plus social features. For news,
compare market + news with market + news + social, including coverage/volume indicators
and an interaction-capable benchmark. See Q1–Q6 in the research agenda for the exact
contrasts. Core sentiment OLS is not a sufficient baseline for every future question.

### Shared scoring and statistical unit

- Save a prediction for every eligible stock-day, including those lacking a realized label.
  Report source-universe, fit, validation, prediction and scored counts separately.
  Failures and constant predictions are visible, not selectively removed.
- Primary tables use a fixed common evaluation sample within each comparison family.
  Report coverage changes alongside pairwise checks. Use fractional tie allocation in sorts,
  constant-signal IC zero on eligible dates, and cash for constant-signal portfolio days,
  following the corrected evaluator. Proposed primary portfolio minimum: 10 effective
  names per decile, hence at least 100 names per date; primary IC minimum: 10 names.
  Apply the same requirements to all models in the comparison.
- Report mean daily IC, daily paired differences against each declared benchmark, 95%
  confidence intervals and two-sided p-values. Use the daily series as the inference unit,
  retaining within-day cross-stock dependence rather than treating millions of rows as
  independent observations.
- For one-day targets, use Bartlett/Newey-West HAC with five trading-day lags and declared
  21/63-lag sensitivity. For overlapping H-day outcomes, predeclare at least H-1 lags
  (and a longer sensitivity); include non-overlapping or block-bootstrap checks where
  overlap or persistent event exposure is central. HAC bandwidth alone is not a cure for
  all dependence. Publish missing-date handling and retain trading-calendar spacing.
- Define comparison families before results. Use Bonferroni-adjusted p-values as the current
  simple convention, reporting raw values too. Count all planned primary contrasts, not just
  successful models. Clearly mark secondary, subgroup and exploratory analyses. These
  adjustments do not erase the project's earlier model-search history.

For return-level predictions, report actual forecast MSE and R-squared against both a
specified zero forecast (where economically appropriate) and a feasible historical-mean
or market-data benchmark. Align any averaging weights across numerator and denominator.
De-meaned cross-sectional R-squared may be a descriptive companion, but subtraction of
the realized future market/cross-sectional mean is not an ex ante forecast operation.
Report Pearson and Spearman correlations with explicit labels.

### Portfolios and interpretation

Report gross equal- and lagged-cap-weighted spreads, long and short legs, yearly performance,
size/liquidity splits and a declared microcap exclusion. Keep fractional sort mechanics
consistent. Disclose minimum effective names and days excluded for sparse portfolios.
Calculate investable leg performance from realizable return series; date-demeaned leg
Sharpes are descriptive and do not establish investability.

For economic claims, reconstruct holdings, drifted weights, turnover, execution prices,
short-side feasibility and transaction costs. State gross/net exposure and whether the
turnover convention is one-way or two-way before applying costs. Report a cost sensitivity
and break-even cost; paper-specific cost assumptions are scenarios, not estimates for our
StockTwits small-cap universe. Predetermined smoothing is allowed; test-selected smoothing
is a new experiment.

Distinguish cumulative H-day outcomes from returns earned on individual delayed days.
Use delayed formation to locate when predictability is realized. Feature-group ablation
with retraining tests incremental predictive usefulness; fixed-model masking measures
reliance and can create unrealistic inputs. Neither by itself establishes a causal
social-media effect, private information, or market inefficiency.

## 6. Task-specific extensions and honest test status

### News, earnings and fundamentals

Align news and social features to the same forecast cutoff. News documents cannot include
later revisions or information that arrived after the social forecast. A joint model's
incremental gain is different from textual novelty: interpretation, attention and diffusion
need their own timing, content and interaction tests.

For earnings, distinguish the probability/timing of an announcement, the released
fundamental or surprise, and the market's announcement-window return. Pre-announcement
forecasts must exclude the release and subsequent messages; post-release interpretation
uses a new forecast clock and return window. Use point-in-time expectations if defining
surprises against analyst consensus. Delayed reporting and restatements govern label
availability. Split by calendar time, keeping related event windows together, and use
firm/date or event clustering/block resampling appropriate to the dependence.

### Aggregate markets

Construct a separate index/ETF-message universe; SPX and tradable ETFs are not interchangeable
securities. Aggregate forecasts have one outcome per market/date. Many messages or many
ETF mentions do not create independent market-return observations.

Preserve common sentiment and attention levels. Use lagged time-series scaling, expanding
or rolling estimates fitted only on earlier dates, and economically relevant market/news
benchmarks. Daily cross-sectional target ranking or date demeaning would remove the
aggregate outcome of interest. Choose longer training/validation spans based on the much
smaller number of independent dates and label maturity, before evaluating performance.
The common chronological discipline applies; the 504/126-date stock-panel default does not
automatically transfer.

### Historical development versus untouched evidence

The 2012–2022 linear results and January 2012/December 2022 NN pilots have already been
examined. These are historical development comparisons, not a newly untouched holdout.
Do not relabel their best discovered specification as prospectively selected.

The source audit now confirms incomplete `text_master.pkl` coverage across
September–December 2023, beyond the previously noticed missing October predictions:
only 4/20, 0/22, 1/21 and 6/20 CRSP session dates, respectively, have any text rows.
Repair and audit this source coverage before reporting an extension. Do not automatically
designate 2023, 2024, or any other available year a
holdout without checking prior use and data completeness. Reserve genuinely unexamined
data for final confirmation when available, and record all accesses.

A pretrained encoder's exact training cutoff is a separate issue from downstream temporal
validation. Record the documented model revision and corpus dates where known; otherwise
mark the cutoff unknown. Neither a small parameter count nor a calendar year alone proves
absence of memorization. Use a clearly documented post-release evaluation, masking and/or
historically admissible representations as diagnostics, while recognizing their limitations.

## 7. Run record, implementation gaps and completion criteria

Each run should retain:

- Experiment/question IDs, protocol version, primary/secondary status and prior results seen.
- Data provenance and hashes or immutable snapshot IDs; code/config/library/encoder versions.
- Ordered feature names, transformations, imputation masks, target definitions and clocks.
- Actual month/date splits, purged rows, availability checks and coverage counts.
- Exact normalized loss, penalty mapping, grids, seed list, validation choices and fit logs.
- Atomic monthly checkpoints, prediction files keyed by security/time, evaluation arguments,
  daily paired series, uncertainty estimates and all planned comparison rows.

Use isolated run directories and resume only with matching identities. Check dataset
integrity and sample keys, perturb unavailable future outcomes to test leakage, and test
that different estimators receive identical transformed inputs and weights. Pilot runs
verify computation; they do not select winners using their realized test performance.
Document negative findings and failed runs as well as successful ones.

Implementation status for the authorized one-day comparison:

| Component | Current status and limits |
|---|---|
| [Shared data and splits](../tools/protocol_data.py) | Implemented: fixed input universe, centered ranks, fitting-only equal-date scaling, missingness rules, minimum target cross-section, full CRSP session calendar and explicit label-end purging. Source panels are unchanged. |
| [Matched linear models](../tools/protocol_linear.py) | Complete: 96 OLS/ridge/lasso/elastic-net specifications over 108 test months, with 648 month-target checkpoints. The corrected run combines 468 certified unaffected reuses and 180 completed reruns. Results are in [Report 07](07_protocol_linear_results.md). |
| [Matched neural networks](../tools/protocol_nn.py) | Implemented on the same prepared inputs and splits, with seed-ensemble validation selection and resumable runs. The standard-budget pilot passed training, restart and exact restored-state checks; the full study has started. Full NN results remain pending. See [live NN status](NN_RUN_STATUS.md). |
| [Protocol evaluation](../tools/protocol_evaluate.py) | Corrected full linear evaluation is complete and reported in [Report 07](07_protocol_linear_results.md). Additional economic validation and full NN comparisons remain outstanding. |
| Source timing and targets | The corrected cache excludes a known nonconsecutive stock/session h=1 target and the associated gap-crossing raw/DGTW h>1 diagnostic labels. All 3,033,080 finite primary-period raw targets now pass the next-session check. Actual early-close routing, delayed benchmark publication and broader longer-horizon label completeness remain limitations. |
| Other research tracks | Return-level/Huber fits, native-scale and expanding-window variants, added return-history/news/event datasets, aggregate-market forecasts and transaction-cost analysis remain future experiments. |

The source-input audit passes all required dates for all 108 primary test months,
January 2014–December 2022, at F=252/504/756 and V=126. A prescribed date window need
not contain the same number of eligible target dates: the January 2014 F=756 window
has all 756 input dates but only **740 eligible labeled fitting dates**, because 16
early dates fail the minimum-10-target rule for both raw and DGTW. Those dates are
excluded without replacing them with older history. The first F=252 and F=504 windows
have their full eligible fitting-date counts; each first validation block has 126.
Run split records retain both prescribed bounds and actual labeled-day/row counts.
See [the detailed timing and source audit](TIMING_CONVENTION.md#implemented-source-and-coverage-audit).

The active prepared version is
[`8f3f4eb44b399771`](../.runs/protocol_v1_1/prepared/8f3f4eb44b399771/manifest.json),
an immutable derivative of the preserved original `15426a6f29a923e3` cache. The
[repair audit](../.runs/protocol_v1_1/prepared/8f3f4eb44b399771/horizon_repair_audit.json)
records the FTNW/permno-17182 March 13, 2019 exception: the imported one-day return
actually matches the March 26 stock return, nine exchange sessions later. Both raw and
DGTW h=1 labels are now missing and both date-local target ranks are recomputed.
Input arrays, economic keys and calendar are unchanged, preserving all **3,034,035
primary prediction rows**. The primary sample now has **3,033,080 finite raw targets,
all verified against the next exchange session**, rather than the previous 3,033,081
which included the invalid observation.

The same known gap also invalidates FTNW raw/DGTW diagnostic outcomes whose nominal
3/5/10/21/42/63-session intervals cross March 14–25. These labels are conservatively
excluded from the corrected evaluation cache; exact counts and cell positions are in
the audit and timing note. This does not certify every other longer-horizon target.
The 752 finite raw labels at the December 29, 2023 cache boundary remain explicitly
unverified and outside the primary period. Original source data and audit artifacts
are retained. Checkpoint migration certified unchanged fitting inputs, targets, tuning
inputs and code semantics for 468 reused checkpoints; all 180 affected checkpoints were
rerun. Corrected training and evaluation are complete across all 648 checkpoints and
96 model specifications. [Report 07](07_protocol_linear_results.md) contains the resulting
108-month comparison and migration provenance.

Existing prediction files retain their original methods and results. The revised
runners write isolated run artifacts; their implementation and pilot checks do not
substitute for complete full-period estimates or the additional economic tests above.
