# Project state: StockTwits information and prediction

Updated September 22, 2026, after completion of Stage A. Start here in a new
session, then read the [latest short summary](13a_linear_interaction_summary.md)
and the [sample/transformation correction](11b_sample_and_transformation_audit.md).

The latest completed experiment is **Report 13: explicit interactions**, with
148 procedures, 216 monthly fitting jobs and complete evaluation over 2014-2022.
It finished September 22 at **4:38:43 p.m. Central**. All 13 final artifact hashes
were independently verified; Windows accepted the completion alert at 4:38:47.
No restart or remaining Stage A fit is needed. The
[status record](LINEAR_INTERACTION_RUN_STATUS.md) and
[execution evidence](data/linear_interactions_v1_execution.json) document the
30 passing tests, early/late pilot, exact additive-baseline checks and final run.

## Current conclusions

- **Market characteristics and past returns provide a strong benchmark.** In the
  current additive ridge comparison, raw-return rank IC is about 0.0694 with
  characteristics and 0.0704 after adding sentiment and attention.
- **Sentiment and attention add a small, detectable increment.** In Stage A,
  adding them after quadratic characteristic controls improves ridge rank IC
  by about **0.0010**, with adjusted p < 0.001. This concerns incremental
  information within the covered sample, not a causal effect of social media.
- **The tested social interactions do not establish an additional benefit.**
  The joint 35-term ridge addition changes rank IC by about -0.0001. None of the
  35 separately fitted additions has a positive gain passing the primary
  family-adjusted test. Some have positive point estimates; absence of a
  detectable increment does not establish that every interaction is useless.
- **Current text additions still lack a convincing conditional gain.** PCA and
  separate penalties can improve text procedures relative to poorly regularized
  full text, but Reports 10-13 do not establish a positive text increment under
  the primary adjusted raw-return ranking comparisons. Stage A separates
  agreement, embedding-PC and text-by-social interaction blocks.
- These are **development results**, with gross portfolio diagnostics reported
  separately. They do not establish net trading profitability or exhaust the
  information potentially available in social media.

The full [Report 13](13_linear_interaction_results.md) retains every registered
comparison, OLS/elastic-net sensitivities, DGTW results and portfolio diagnostics.
The [short summary](13a_linear_interaction_summary.md) is the reading guide.

## Completed studies and their scope

| Report | Completed evidence | How to use it now |
| --- | --- | --- |
| [07](07_protocol_linear_results.md) | 96 social-only OLS/ridge/lasso/elastic-net procedures | Historical matched linear study; selected training fits retained without validation refit |
| [08](08_protocol_nn_results.md) | 24 full-budget NN3 procedures alongside the 96 linear procedures | Completed social-only NN evidence, not merely a pilot; no characteristic-conditioned NN claim |
| [09](09_characteristic_linear_results.md) | 264 linear procedures, including characteristic-only and social additions | Establishes the market-information benchmark on the same stock-days |
| [10](10_linear_design_results.md), [summary](10a_linear_design_summary.md) | 228 procedures comparing histories, refits, global/group penalties and PCA | Supports the current working union-refit design; not proof of a universally optimal window |
| [11](11_linear_optimization_results.md), [summary](11a_linear_optimization_summary.md) | 172 procedures comparing penalty and PCA menus plus two implemented rank treatments | Read with [11b](11b_sample_and_transformation_audit.md): this was not raw levels versus ranks |
| [12](12_interactions_and_nonlinear_plan.md) | Research plan, not an estimated-results report | Stage A completed; Stage B trees and Stage C NN extension remain proposed |
| [13](13_linear_interaction_results.md), [summary](13a_linear_interaction_summary.md) | 148 explicit-interaction procedures: 13 joint/reference bases with three estimators, plus 35 individual ridge additions per target | Latest completed comparison and starting point for the next model study |

Earlier Reports 01-03 document exploratory text results; Report 04 reviews
GKX/CKX, Report 05 records implementation planning, and Report 06 records the
evaluation recovery and NN pilot. Their sample, benchmark and refit conventions
must not be substituted for those of later completed studies. The preliminary
NN report [08a](08a_protocol_nn_preliminary_results.md) is superseded by Report 08.

## Working procedure for the current return track

1. **Timing:** assign a message to its first market close, time t; predict the
   return from close t to close t+1. The inherited code uses nominal 16:00 closes
   and has an early-close limitation. Use the exact label-maturity exclusions
   in [TIMING_CONVENTION.md](TIMING_CONVENTION.md).
2. **Splits:** update monthly, using 504 initial training sessions and the next
   126 chronological validation sessions with the required maturity gaps.
   Choose settings on validation, then refit on the original training/validation
   union: 630 input sessions, preserving the interior purge. Test observations
   never enter selection or the final fit.
3. **Inputs:** continuous C and social scalars already use daily cross-sectional
   midranks in the certified cache. Missing characteristic values become neutral
   zero with 17 separate binary flags. Apply fitting-only scaling; retain flags
   as unscaled main effects. Embeddings and agreement measures are not ranked.
4. **Target and loss:** centered daily return ranks, squared loss, equal total
   fitting weight per day and an unpenalized intercept. Select penalties by mean
   daily validation Spearman IC, using the declared candidate/tie order.
5. **Interactions:** Stage A constructs squares/products from cached C/S ranks,
   16 fitting-estimated, centered/scaled embedding PCs and raw agreement inputs.
   Scale derived columns on fitting inputs; never rank the products again.
   PCA/scaling are rebuilt on the union for the final refit. Missingness flags
   do not receive product terms.
6. **Comparisons:** hold information sets fixed when comparing estimators; hold
   estimators and procedures fixed when adding information. Use C, C+S and
   C+S+text comparisons, and appropriate richer C controls for interaction tests.
   Core social-only OLS is a historical reference, not the sole current benchmark.
7. **Evaluation:** raw next-day returns are primary; DGTW is secondary. Use common
   economic keys within each target, then exclude unavailable outcomes from
   scoring. Report daily rank IC, paired calendar-aware HAC5 tests with HAC21/63
   sensitivities, declared comparison-family adjustments, early/late periods,
   and gross equal-/capitalization-weighted portfolio diagnostics.

[EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md) provides the common framework.
Each completed experiment retains its own declared specification; the later
union-refit extension does not retrospectively change Reports 07-09.

## Sample, data and known limits

The prediction universe consists of **stock-days with retained Bullish/Bearish-
tagged StockTwits messages and available embeddings**. The text panel is an inner
join, not the full CRSP stock-day panel. Attention counts those retained tagged
messages. Broader CRSP histories supply controls without adding no-message days.
For 2014-2022 there are **3,034,035 prediction rows**, 7,155 distinct PERMNOs and
2,266 formation dates; raw outcomes are observed for 3,033,080 rows and DGTW
outcomes for 2,748,978. All rows have a positive embedding count.

C comprises 17 market/past-return inputs plus 17 missingness flags: returns over
1/5/21/63/126/252 sessions; momentum excluding the last 21 sessions; size and
price; turnover and dollar volume; volatility; maximum daily return; and Amihud
illiquidity. Accounting characteristics and news controls are not included.
S denotes sentiment and attention. Full text comprises 384 mean-pooled
`all-MiniLM-L6-v2` coordinates plus `embed_norm` and `embed_cos`.

**Report 11 correction:** both implemented transformation branches read already
ranked `X.npy`. Its `daily_rank` branch ranks those values again, including
neutral-zero imputations. It mixes rank convention and missingness treatment;
it cannot answer raw levels versus ranks. The genuine comparison remains
outstanding; unranked controls are saved in `characteristics_raw.npy`. Preserve
the certified run and use [11b](11b_sample_and_transformation_audit.md) to interpret it.

The 2014-2022 period has already informed design choices. No untouched confirmation
sample is certified: 2023 inputs cover only 178 of 250 expected sessions, and
earlier work mentions predictions through 2023. Other retained limitations include
nominal-close timing, absent publication-date accounting controls, the tagged-only
corpus, and known source-label issues documented in the preparation/evaluation
audits. In particular, the corrupted source `ar_dgtw_21` diagnostic is excluded;
do not silently repair historical forecasts or infer longer-horizon training from
the descriptive horizon tables.

## Where to resume

Work in **`D:\StockTwits\Code`**, with source data in `D:\StockTwits\Data`.
The older copy under `C:\Users\skazempour\Documents\StockTwits` is stale.
Use `C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe` for the
existing local environment. Large caches, checkpoints and prediction files are
local artifacts; a Git checkout alone does not contain them.

- Certified characteristic cache:
  `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42`.
- Its preserved social-only parent:
  `.runs/protocol_v1_1/prepared/8f3f4eb44b399771`.
- Latest completed study, including `status.json`, logs, frozen source snapshot
  and notification receipts:
  `.runs/linear_interactions_v1/study/78b3dd65e0d5f8a6`.
- Latest full registry within that study:
  `linear/full_9f279d17e34278c6/registry.json`.
- Historical additive reference registry required for Stage A reproduction:
  `.runs/linear_design_v2/study/c11478141261534f/linear/full_9590a962af3e2fcb/registry.json`.
- [Stage A execution/resume instructions](../tools/LINEAR_INTERACTION_STUDY.md),
  [design](LINEAR_INTERACTION_EXPERIMENT.md) and
  [machine specification](data/linear_interactions_v1_experiment.json).
- Earlier operational guides:
  [characteristics](../tools/CHARACTERISTIC_STUDY.md),
  [linear design](../tools/LINEAR_DESIGN_STUDY.md),
  [linear optimization](../tools/LINEAR_OPTIMIZATION_STUDY.md), and
  [completion monitoring](../tools/STUDY_NOTIFICATIONS.md).

The Stage A numerical path is in `tools/linear_interaction_core.py`; its runner,
controller, evaluator and report generator have corresponding
`linear_interaction` filenames. Monthly checkpoints carry integrity certificates.
Resume instructions validate compatible checkpoints rather than overwriting them.
The independent watcher records success, failure or interruption and sends a
Windows desktop alert; it cannot send an unsolicited message in this chat or
continue while the machine is off.

**Frozen boundaries:** retain certified input caches, completed specifications,
source snapshots, checkpoints, evaluation tables and completed results reports as evidence
of what ran. Corrections belong in explicit companion notes such as 11b. A changed
model, data preparation or executable source needs a new identified experiment;
do not reuse incompatible checkpoints or relabel an old study. This living state
file and research/handoff notes may be updated without changing certified results.

Historical data entry points remain `Data/merged_master.pkl`,
`Data/text_master.pkl`, and
`Data/v1/data/csv/text_embeddings_mlcrowd/text_embeddings_stock_day.pkl`.
Upstream notebooks are under `00 - data cleaning`, `01 - feature extraction`, and
`02 - prepare training dataset`; [integration notes](../01%20-%20feature%20extraction/features_08_integration_notes.md)
explain the original text build. The old prediction/trading notebooks and their
`Data/predictions_*.pkl` outputs describe earlier designs, not the current runner.

## Proposed next work and broader agenda

The next model stage proposed in [Report 12](12_interactions_and_nonlinear_plan.md)
is **Stage B: histogram boosting and random forests**, followed by **Stage C:
characteristic-conditioned NN depth/width and training comparisons**. Their new
matched pipelines and runs have not been implemented or started. This handoff
does not authorize launching them. Completed social-only NN3 code/results remain
available, and the [existing server package](../server_nn/README.md) supports that
earlier design; it needs an explicitly updated bundle for the later stages.

Keep [research questions Q1-Q6](RESEARCH_QUESTIONS.md) as the broader agenda:
features beyond sentiment/attention; nonlinear models; information beyond market
characteristics; information relative to news; fundamentals and announcement
returns; and aggregate-market prediction. These can support more than one paper.
News, point-in-time fundamentals/events, market-index/ETF discussions, and
untagged-message expansion require separate data work and comparisons. The
current Stage A result does not settle those questions. A genuine raw-versus-
ranked characteristic experiment and independent confirmation also remain open.
