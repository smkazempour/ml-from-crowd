# Shared experimental protocol

Historical baseline: **v1.1, adopted 2026-09-12**. Current implementation and
method amendments updated **2026-09-22**. This is a living protocol, not a
replacement for the frozen specification of any completed experiment.

Revision v1.1 made the user-confirmed first-close timing explicit and replaced
varying 80/20 histories with 252/504/756 fitting dates plus 126 validation dates.
The original primary history was F504, and fitted coefficients were retained
without refitting on validation. Those rules describe the completed social-only
linear [Report 07](07_protocol_linear_results.md) and NN3
[Report 08](08_protocol_nn_results.md); their results are not relabeled.

**Current procedure for the characteristic-conditioned experiments:** 504 initial
training sessions, 126 chronological validation sessions, then a final monthly
refit on their exact union (630 input sessions), preserving the original maturity
exclusions. Validation selects settings before this refit. Fitting-only scaling
and PCA are rebuilt on the permitted combined inputs; no forecast-month outcomes
enter estimation or selection. The exact signal-session formulas remain in
[TIMING_CONVENTION.md](TIMING_CONVENTION.md).

Completed subsequent studies are the characteristic benchmarks
([09](09_characteristic_linear_results.md)), refitting/history/penalty/PCA design
([10](10_linear_design_results.md)), expanded linear search
([11](11_linear_optimization_results.md)) and explicit interactions
([13 summary](13a_linear_interaction_summary.md)). Stage A completed successfully
on September 22; no model run is active. Optimized trees (Stage B) and
characteristic-conditioned NN comparisons with a combined refit (Stage C) are
proposed in [Report 12](12_interactions_and_nonlinear_plan.md), but are not yet
implemented or running. The social-only NN3 runner does not supply that new NN
refit procedure.

**Transformation correction:** the characteristic cache already contained daily
ranks. Report 11's `standard` branch used those ranks plus fitted scaling;
`daily_rank` ranked the stored values again, including missing values previously
set to neutral zero. It did not test raw standardized characteristics against
ranks. The numerical results remain results for those actual procedures, and
Stage A uses the original once-ranked inputs. The intended raw-versus-ranked
comparison remains outstanding; see [audit 11b](11b_sample_and_transformation_audit.md).

All 2014-2022 comparisons are development evidence. No untouched confirmation
sample is certified; 2023 is incomplete and has prior recorded use. The
[research questions](RESEARCH_QUESTIONS.md) distinguish completed evidence from
future news, fundamentals, aggregate-market and corpus extensions. Each frozen
experiment specification controls the interpretation of its own outputs.

The common standard is the same information set, sample, timing, fitting opportunity,
and evaluation within a comparison. Different scientific tasks require different targets
and losses. In this document, **benchmark model** means a comparison estimator;
**conditioning variables** means additional predictors such as past returns or news.

## 1. What we take from GKX and CKX

| Source | Verified procedure | Decision for this project |
|---|---|---|
| GKX, published 2020 article, Sections 1.1–1.2 and 2.1 | Monthly excess-return levels; squared loss and selected Huber variants. Characteristics ranked cross-sectionally; missing characteristics median-imputed. Initial 18-year training and 12-year validation blocks, annual tests; training expands. Fitted parameters use training alone. | Adopt temporal validation and comparable transformations. Our daily frequency, shorter history and rank target require explicitly different settings. |
| GKX, Sections 1.2.1 and 1.8 | Pooled observation loss; inverse-cross-section-size and capitalization weights also considered. Return R-squared against zero and date-level forecast comparisons. | Equal-date weighting is our implemented primary choice because coverage changes strongly. Report actual-return accuracy separately from ranking accuracy. |
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

The completed comparisons use the current tagged-message common-stock universe.
For each comparison family, build one feature table from the union of required columns,
then give every estimator the same eligible fitting, validation and prediction rows.
Use explicit imputation rules instead of estimator-specific complete-case selection.
Report any coverage restriction and its economic meaning. Different families may use
different universes, but their scores are not directly comparable without a common sample.

The implemented text panel excludes stock-days without retained tagged messages and
embeddings. Every 2014-2022 prediction row has a positive embedding count; the
characteristic-only model receives exactly those same rows. Broader CRSP histories
are used to calculate controls, not to add no-message days. These experiments test
content conditional on coverage, not message presence versus absence across all
stocks. See [audit 11b](11b_sample_and_transformation_audit.md).

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

This rank convention is implemented in the certified protocol and characteristic
caches. Earlier legacy notebooks used percentile ranks directly and differ in
finite-sample centering. Daily ranking can change pooled observation ordering,
so it can affect trees as well as linear models and networks. Do not apply a
second ranking to prepared scalar inputs; that was the Report 11 comparison
error documented above.

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

Daily ranks discard common shifts and magnitudes. A genuine native-scale comparison
remains unimplemented: register a separate study using the saved unranked controls,
explicit missingness rules, fitting-only scaling and the same data splits. Report 11
does not supply this comparison. When market-wide activity is a predictor, preserve it separately.

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
actual realized returns, never ranks. Raw-return ranking is the implemented primary task;
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

Use F for the initial fitting dates and V for validation dates. The original
v1.1 comparison held V fixed while varying F. The table preserves those historical
window comparisons and identifies the current combined-refit default; varying F
is not required for every subsequent study.

| Element | Rule |
|---|---|
| Re-estimation | Monthly; complete selection using outcomes realized by the preceding month's last trading close |
| Current initial fitting history | F=504 trading dates |
| Current final refit | Original F504 fitting rows plus V126 validation rows, 630 input sessions; preserve the interior purge and exclude immature labels |
| Validation | V=126 trading dates, ending at the latest signal date whose h-day label is available by the selection cutoff |
| Historical fitting-history comparisons | Reports 07-08 compare F=252/504/756 at fixed V126; Report 10 also tests validation-selected history and three final-fit policies |
| Historical reference | Existing 252 total dates = 202 fitting + 50 validation, explicitly labeled legacy |
| Test | Following calendar month's eligible trading dates; parameters fixed for that month |
| Evaluation span | January 2014–December 2022 common h=1 comparison, subject to feature/label coverage checks; 504/126 results from January 2013 as a supplemental history |
| Longer-validation robustness | A separate comparison holding F fixed; do not mix it with the primary fitting-history comparison |
| Expanding-history robustness | A separately registered later experiment with common start and validation rules |

The historical v1.1 schedule superseded v1's 252/504/756 **total-history** 80/20
splits. Current studies keep two years of initial fitting plus six months of
validation and the combined refit; this is a working design, not an established
global optimum. Completed window comparisons remain in their original reports.
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
timestamps. The legacy notebook's month-start-minus-two-calendar-days heuristic
is not equivalent to the implemented exchange-session schedule; the trace gives
a weekend-boundary example. Document actual dates,
counts and dropped labels for each monthly fit.

All stocks on a date belong to the same temporal block. No random row cross-validation
or random stock-day train/test splits. Earlier outer-test outcomes may enter a later fitting
window after realization under the declared walk-forward rule; that is distinct from using
a future month's results to choose its own model.

Select penalties, architecture candidates, training duration and ensemble weights
using validation only. **Historical v1.1 / Reports 07-08:** retain the parameters
estimated on the initial fitting block; do not retrospectively describe those
forecasts as combined refits. **Current characteristic-conditioned procedure:**
freeze the validation-selected settings, then refit on the exact original fitting
and validation rows. This union excludes the original internal purge rather than
silently filling it with extra observations. Recompute permitted fitting-only
scaling/PCA and numerical sparse-penalty scale under the declared refit rule.

For future trees, freeze the validation-selected iteration/tree count before the
combined refit. For future NNs, fix stopping epochs and the learning-rate schedule
before that refit; do not use observations now in the fitting sample as an
independent early-stopping set. These characteristic-conditioned tree/NN refit
extensions remain planned, not implemented.

The fixed 126-date validation block gives more time diversity than the old 50-date block,
but selection can still be noisy. Report penalty instability and convergence. Keep grids
modest, include zero/simple predictors, and assess the same fitting histories across every
estimator. A grid-boundary selection motivates a documented new experiment, not an
unreported full-test-driven grid expansion.

### NN and model-specific mechanics

The completed historical NN3 comparison used the implemented 128–64–32 network, ReLU,
batch normalization, Adam, five fixed seeds, learning rate 0.001, batch size 10,000,
100-epoch ceiling and patience five. The current weight-decay candidates are
0.00001, 0.0001 and 0.001. This is an adapted project recipe.

Use the same seed list for every comparable NN input set. For each penalty, save each
seed's checkpoint at its best validation IC. **Implemented historical selection is the mean daily
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
The historical social-only feature sets have 2, 53, 388 and 439 columns; the
unfinished cohort/conviction output is not part of the 53. Current additive
characteristic-conditioned references have 34 columns (17 characteristics plus
17 flags), 36 with sentiment/attention, and 422 with full text. Stage A adds the
explicit bases declared in [its specification](LINEAR_INTERACTION_EXPERIMENT.md),
including 35 individual social products and joint models up to 604 columns.

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

The completed evaluators report gross equal-weighted and close-t capitalization-
weighted spreads and legs, yearly performance, and size/activity diagnostics. A
lagged-cap weighting variant, liquidity splits and a declared microcap exclusion
require their own explicitly recorded implementation. Keep fractional sort mechanics
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

The early 2012-2022 comparisons, NN pilots, completed social-only NN3 study and
all characteristic-conditioned Reports 09-13 have been examined. The common
2014-2022 period is development evidence. Do not relabel its best discovered
specification as prospectively selected or call the early/late subperiods
independent confirmation.

The source audit now confirms incomplete `text_master.pkl` coverage across
September–December 2023, beyond the previously noticed missing October predictions:
only 4/20, 0/22, 1/21 and 6/20 CRSP session dates, respectively, have any text rows.
Overall, 178 of 250 expected 2023 dates are available. Prior project records also
mention predictions through 2023. Repair coverage and audit prior use before
reporting an extension; completeness alone would not restore untouched status. Do not automatically
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

Implementation status as of September 22, 2026:

| Component | Current status and limits |
|---|---|
| [Shared data and splits](../tools/protocol_data.py) | Implemented: fixed input universe, centered ranks, fitting-only equal-date scaling, missingness rules, full exchange calendar and explicit label maturity. Source panels are unchanged. |
| Historical social-only linear and NN3 studies | Complete: 96 linear specifications ([07](07_protocol_linear_results.md)) and 24 NN3 specifications ([08](08_protocol_nn_results.md)); these retain pre-validation coefficient fits. They do not implement characteristic-conditioned combined-refit NNs. |
| Characteristic data and linear benchmarks | Complete: 17 market/past-return controls plus 17 missingness flags on the same tagged-message rows; [09](09_characteristic_linear_results.md). No point-in-time accounting or news controls. |
| Linear design and optimization | Complete: 228 procedures in [10](10_linear_design_results.md) and 172 in [11](11_linear_optimization_results.md). Report 11 requires [correction 11b](11b_sample_and_transformation_audit.md): raw-versus-ranked inputs were not tested. |
| Explicit interactions, Stage A | Complete: 148 procedures, 216 month-target jobs, 108 forecast months; [13 summary](13a_linear_interaction_summary.md). Ridge is primary; all 35 individual social terms are reported with the full correction budget. |
| Optimized trees / characteristic-conditioned NNs | Stage B / Stage C are planned in [12](12_interactions_and_nonlinear_plan.md), not implemented or running. No model run is active. |
| Evaluation | Completed studies retain common-sample daily ranking, paired HAC5/21/63, early/late periods and gross portfolio diagnostics. These are not net implementable returns; transaction costs, holdings and execution validation remain outstanding. |
| Source timing and targets | The corrected cache excludes the known nonconsecutive h=1 target and associated gap-crossing longer-horizon labels. All 3,033,080 finite primary-period raw h=1 targets pass the next-session check. Early-close routing and broader longer-horizon completeness remain limitations. |
| Other research tracks | Genuine raw-level/rank comparison, news/accounting/event datasets, aggregate-market forecasts, return-level/Huber models, expanding history and untagged-corpus expansion remain separate future experiments. |

Current evidence should not be read as "all interactions never help." In Stage A,
sentiment/attention added to quadratic characteristics improves primary raw-return
ridge IC by 0.000999 (HAC5 family-adjusted p = 5.64e-7). None of the 35 individual
social products passes the adjusted primary test, and their joint block has no
established gain. Characteristic-by-characteristic products help some comparisons,
including secondary DGTW specifications; these distinctions remain in
[Report 13](13_linear_interaction_results.md). No tested social/text interaction
finding establishes a causal mechanism or substitutes for independent confirmation.

The source-input audit passes all required dates for all 108 primary test months,
January 2014–December 2022, at F=252/504/756 and V=126. A prescribed date window need
not contain the same number of eligible target dates: the January 2014 F=756 window
has all 756 input dates but only **740 eligible labeled fitting dates**, because 16
early dates fail the minimum-10-target rule for both raw and DGTW. Those dates are
excluded without replacing them with older history. The first F=252 and F=504 windows
have their full eligible fitting-date counts; each first validation block has 126.
Run split records retain both prescribed bounds and actual labeled-day/row counts.
See [the detailed timing and source audit](TIMING_CONVENTION.md#implemented-source-and-coverage-audit).

The historical social-only prepared version is
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


### Current characteristic cache and amendment record

Characteristic-conditioned studies use
[`35ef3a5eb0cb5e42`](../.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/manifest.json),
an immutable derivative of the corrected social-only cache described above. It
preserves economic keys, targets and calendar, appends ranked market controls and
missingness flags, and separately retains unranked controls in
`characteristics_raw.npy`. The current primary prediction universe remains
3,034,035 stock-days. Individual frozen specifications and completion certificates
identify the actual code, configuration and artifacts for each study.

2026-09-22 maintenance update: separated the historical retained-fit v1.1 procedure
from the currently executed 504/126 union-refit design, recorded completion through
Stage A, corrected stale NN-pilot and characteristic-availability statements, and
linked the ranked-input audit. Historical studies, certified reports and frozen
sources retain their original provenance. Stages B-C are plans, not active runs.
