# Research questions: financial social media

Living agenda, created 2026-09-12. The project studies what financial social media tells us about firms and markets, how that information relates to other sources, and when it becomes reflected in prices. It may support several papers. The questions below define related research directions, not six promised papers or six established findings.

Keep the identifiers **Q1–Q6** stable when revising the wording; append Q7 and later identifiers for new questions. The proposed common estimation and evaluation rules are in [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md). This agenda does not imply that those rules, future datasets, or all proposed experiments have been implemented.

## Shared comparisons and interpretation

Use nested information sets to identify the source of a predictive gain:

- **Core social information:** net sentiment and log message volume, representing sentiment and attention.
- **Engineered social information:** additional measured features such as disagreement, abnormal sentiment, abnormal attention, and intraday activity. Distinguish the current 53-column table from proposed or unbuilt feature groups.
- **Text information:** message embeddings and related text summaries; keep message counts and agreement summaries explicit so a gain is not automatically attributed to semantic content.
- **Market information:** point-in-time stock characteristics and appropriately lagged return, volume, volatility, and liquidity histories.
- **News information:** timestamped news content and its sentiment, attention, and event summaries.

Compare additions within the same model and compare models within the same information set. Use identical forecast cutoffs, outcome definitions, eligible observations, and evaluation rules for each paired comparison. “Matched linear benchmark” means a linear model following the same preprocessing and temporal split; it does not mean adding regression control variables. Market characteristics and news are additional information sets and can also serve as regression controls.

Out-of-sample prediction can establish incremental forecast information under the tested design. It does not by itself establish that social media causes returns, that posters possess private information, or that a feature-importance measure identifies an economic mechanism. Lack of a statistically detectable gain does not establish that no information exists. Report effect sizes, uncertainty, coverage, and sensitivity to the representation and model class.

## Q1. Do features beyond sentiment and attention improve stock-return prediction?

**Main question.** Does adding text embeddings or other features extracted from social media improve forecasts relative to the two core measures?

**Comparisons.** Within each model family, compare core; core plus engineered features; core plus text; and core plus engineered features plus text. Separate embeddings from their agreement summaries through ablations. Distinguish improvements in stock ranking, return-level accuracy, and implementable portfolio performance. Examine whether results differ with stock size, message activity, or market conditions using prespecified subgroup comparisons.

**Current evidence.** On the current tagged-message sample, the corrected linear comparisons show higher daily rank IC from text plus core than from core alone. For elastic net, raw-return IC rises from 0.03067 to 0.03960; the corresponding DGTW comparison also improves. No decile-spread increment survives the stated multiple-comparison adjustment. These are findings for the tested sample, targets, representations, and historical model search—not evidence that every social feature helps. See [Report 06](06_evaluation_and_nn_pilot.md).

**What remains.** Complete the matched-model comparisons; test engineered feature groups separately; and later compare tagged, untagged, and combined corpora while holding coverage and attention information explicit. The cohort/conviction group has code but no output in the current 53-column table, so its contribution remains untested. No user-skill or influence feature expansion is included in this agenda.

## Q2. Do nonlinear models improve return prediction from social media?

**Main question.** Do neural networks, boosted trees, or random forests extract predictive relationships that linear models miss?

**Comparisons.** For each fixed information set, compare OLS and regularized linear models with nonlinear models using matched transformations, fitting observations, validation dates, and outcomes. Include the original core OLS benchmark as a historical reference. Compare the *increment from adding text* within linear and nonlinear families; a better nonlinear core model alone does not show that nonlinearity unlocks text information.

**Current evidence.** The NN execution path supports several architectures, but only bounded NN3 pilots have been validated on real data: four feature sets, two months, one seed, and three epochs. They establish working training, prediction, and restart behavior. They do not establish nonlinear superiority. Full NN performance comparisons and new tree comparisons remain outstanding; see [Report 06](06_evaluation_and_nn_pilot.md).

**What remains.** Finish the full-budget NN3 study under the common procedure, then evaluate architecture and tree alternatives under a declared comparison family. Inspect convergence and seed variability. Use held-out ablations to study interactions, recognizing that correlated features can make individual importance rankings unstable. A useful conclusion may be that richer information matters more than model complexity.

## Q3. Does social media add information beyond stock characteristics and return history?

**Main question.** Does social media predict returns beyond information already available from prices, trading activity, and firm characteristics?

**Comparisons.** Estimate market-only; market plus core social information; and market plus richer social information models, within both linear and nonlinear families. Include short-term reversal, momentum, recent volatility, volume, liquidity, and size among the candidate market predictors, with the final list and lookbacks specified before the main comparison. Compare the social increment conditional on market information, not only the standalone social model against the market-only model.

**Timing and interpretation.** Social activity may react to the same price movement that predicts a subsequent reversal. Separate posting windows and return windows; use delayed formation and overnight/intraday outcomes when timestamps and prices permit. DGTW-adjusting the target does **not** replace conditioning on lagged returns and stock characteristics in the predictive model.

**Needed data and status.** Build and audit the point-in-time market-feature join, preserving delisted stocks and avoiding publication-date leakage in accounting characteristics. The current evidence does not establish an incremental social signal after these controls. Any claim of distinct information must survive this comparison; an explanation based on attention or delayed incorporation remains a hypothesis without additional identification.

## Q4. How does social-media information relate to news?

**Main question.** Does social media contain information beyond measured news coverage? Does it repeat or spread news, help interpret news, or contribute complementary information?

**Comparisons.** With the same market-characteristic and return-history baseline in every specification, add news; social information; news plus core social information; and news plus richer social information. Compare additive specifications with joint nonlinear models or prespecified news–social interactions. Hold the information cutoff and stock-day coverage fixed. A joint gain can arise from complementary interpretation even when social media has weak standalone predictive power.

**Distinguishing explanations.** Study social messages before and after news publication, links or semantic overlap with prior news, the lag between news and discussion, and whether social disagreement or stance changes forecast the subsequent response conditional on news content. Compare announcement/event categories where a defensible classification exists. These tests can separate patterns consistent with novel content, repetition, attention/diffusion, or interpretation; none alone proves the corresponding mechanism.

**Needed data and status.** Obtain a timestamped news corpus with publication/update times, entity mappings, deduplication, and documented historical coverage. Measure information available at the forecast time rather than using later article revisions. Apparent absence of news is only meaningful when source coverage is adequate. Social predictability after controlling for an incomplete news measure is not proof of private information; failure to improve forecasts is not proof that social media merely repeats news. This question has not yet been tested in the current pipeline.

## Q5. Does social media predict fundamentals and returns around their release?

**Main question.** Can social media forecast subsequent firm fundamentals, particularly earnings, and can it forecast the market's return response to their announcement?

Treat three distinct outcomes separately:

1. **Fundamental realization:** earnings levels, earnings changes, or surprises relative to a consensus or another benchmark that was available at the forecast cutoff.
2. **Announcement return:** the return around a precisely timed earnings release, with the forecast formed strictly before the chosen response window.
3. **Event occurrence or timing:** whether an unscheduled event occurs or a release date changes. This is a separate possible extension; predicting a known scheduled earnings date is not predicting earnings content.

**Comparisons.** Start with historical fundamentals, point-in-time analyst expectations where available, market information, and news; measure the incremental contribution of core and richer social information. Pre-release forecasts of announcement returns and post-release forecasts of delayed reaction answer different questions and need different cutoffs. A fundamental forecast need not translate into a return forecast because prices may already reflect it.

**Needed data and status.** Link actual announcement timestamps, point-in-time accounting vintages, analyst forecasts, and event-window returns. State treatment of premarket, intraday, and after-hours releases and overlapping events. Use chronological evaluation with label-availability gaps appropriate to each outcome; repeated posts about the same event must not cross a training/test boundary. These outcomes and joins are not implemented in the current return-prediction pipeline.

## Q6. Does social media predict market-level returns?

**Main question.** Do messages about broad-market instruments, including SPX and large ETFs, contain information about subsequent aggregate returns? Can aggregation of stock-specific messages provide additional market information?

**Comparisons.** Compare lagged market returns and other available market predictors; those predictors plus market-level sentiment and attention; and those predictors plus richer social representations. Keep direct index/ETF discussions distinct from aggregates of constituent-stock discussions. If aggregate news is available, test the social increment conditional on it as a later extension.

**Universe and timing.** Recover the relevant posts currently excluded by the common-stock universe. Map actual platform cashtags to the intended indices or funds, record changes in mappings, and handle posts mentioning multiple instruments. An index discussion series, an ETF discussion series, the predicted benchmark, and a tradable implementation vehicle are distinct objects. Specify market hours, quote timestamps, and forecast/return windows for each; SPX itself is not a directly traded security.

**Needed data and status.** Build a separate instrument universe and target series, audit coverage, and preserve the current stock panel. A single market-return series has far fewer independent forecast dates than the stock-day panel. Cross-sectional daily ranks cannot be transferred mechanically to it: use time-series forecasts, train-only transformations, suitable forecast benchmarks, and time-series loss comparisons. No market-level results are claimed by the current project.

## Possible papers and ordering

These are overlapping bundles, to be revised as evidence and data availability develop:

| Potential paper | Central contribution | Questions and dependencies |
|---|---|---|
| Social information and cross-sectional return prediction | What richer social data and nonlinear models add after conventional market predictors | Q1–Q3; closest to the current pipeline; timing and economic implementation remain essential |
| News, social interpretation, and information diffusion | When social discussion complements news and how their timing relates to prices | Q4 with selected Q1–Q3 analyses; requires the news join and sharper mechanism tests |
| Social expectations, fundamentals, and earnings reactions | Whether social information forecasts fundamentals, market expectations, or reactions | Q5, potentially Q4; requires point-in-time event and expectations data |
| Aggregate beliefs and market predictability | Information in direct market discussions and constituent-level aggregation | Q6; requires the excluded instrument corpus and a separate time-series design |

The immediate sequence is to complete Q1–Q2 with matched benchmarks, bring Q3 and timing checks forward, and then choose data expansion according to the intended paper. Portfolio costs and liquidity constrain economic claims wherever return prediction is central. A coherent negative or boundary result can be informative; paper boundaries should follow the question and evidence rather than favorable model outcomes.

## Maintaining the agenda

For each question, retain its identifier and update: current evidence and its exact scope; next discriminating comparison; required data; unresolved design issues; and links to experiment specifications and results. Label ideas, implemented methods, pilot checks, and completed evidence separately. Log material changes in a short dated note instead of silently rewriting a previously evaluated hypothesis. New questions should specify the outcome, information increment, forecast cutoff, and comparison that could distinguish competing explanations.
