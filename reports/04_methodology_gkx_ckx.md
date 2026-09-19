# Report 04 -- Methodology review: Gu, Kelly and Xiu (2020) and Chen, Kelly and Xiu (February 2026 revision), and what to adopt here

Verification addendum, 2026-09-12: this report records the earlier literature review and
project recommendations. The [shared experimental protocol](EXPERIMENTAL_PROTOCOL.md)
and [research agenda](RESEARCH_QUESTIONS.md) supersede its proposed workflow and narrow
scope; aggregate-market prediction is now explicitly part of the agenda. "Here today"
entries describe the earlier review, not a fresh implementation inventory; see
[Report 06](06_evaluation_and_nn_pilot.md) for the completed recovery and NN pilots.
Material source claims have been corrected below. CKX's local 78-page PDF was created
on February 23, 2026 and matches the February 24 revision date in the
[SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687) and the February
2026 listing on [Xiu's research page](https://dachxiu.chicagobooth.edu/).

Date: 2026-09-12. Purpose: extract the parts of the two papers' designs that bear on this
project, compare each with what the pipeline does today, and rank what to adopt. No code was
changed; the resulting to-do items are Section 5, items 7-8 of `reports/00_project_state.md`.

Sources (both in `D:\StockTwits\Literature\`, outside the repo): Gu, Kelly and Xiu,
"Empirical Asset Pricing via Machine Learning", RFS 33(5), 2020, pp. 2223-2273
(`hhaa009.pdf`; GKX below). Its Internet Appendix is now available locally as
`Gu Kelly Xiu - Internet Appendix.pdf`; grid references are marked [IA]. Chen, Kelly and Xiu, "Expected Returns and
Large Language Models", SSRN 4416687 (`ssrn-4416687 (1).pdf`; CKX below). The downloaded
version is the February 2026 revision (LLaMA3, transaction-cost analysis, topic decomposition,
AI summaries), which differs in places from the 2022-23 drafts and the 2024 conference
slides; where the versions differ this report follows the PDF.

## 0. One-paragraph summary of each paper

**GKX** compare thirteen return-prediction methods (OLS, OLS-3, PLS, PCR, elastic net, group-lasso
GLM, random forest, gradient-boosted trees, neural networks with 1-5 hidden layers) on the
monthly cross-section of nearly 30,000 US stocks, 1957-2016, using 94 characteristics x (8
macro variables + 1) + 74 industry dummies = 920 predictors. The lasting contribution is the
*protocol*: temporal train / validation / test split, hyperparameters tuned on the validation
block, an out-of-sample R2 measured against a zero forecast, a Diebold-Mariano test adapted to
a panel, two variable-importance measures, and prediction-sorted decile portfolios with
drawdown, turnover and factor alphas. Trees and shallow nets win; the same handful of
price-trend, liquidity and volatility variables drive every model.

**CKX** embed 3.0M single-stock Refinitiv news articles and 2.9M alerts (US, 1996-2019; 15
other markets) with BERT, RoBERTa, LLaMA 1/2/3 and OpenAI's `text-embedding-3-large`, average
the token vectors per article, and regress the next open-to-open return on the embedding with
ridge in an annually refit rolling window (6 years training, 2 years validation, 1 year test;
out-of-sample 2004-2019). Predictions sort news-day stocks into quintile long-short
portfolios. Equal-weighted Sharpe ratios run from 2.3 (Loughran-McDonald dictionary) to 4.8
(OpenAI embedding); value-weighted from 0.4 to 1.3. A random forest or a three-layer network
on the same embedding beats ridge (5.5 / 5.3 vs 4.6). The signal is incremental to five-day
reversal, persists three to four days in small stocks and one to two in large ones, survives
costs with turnover control, survives their look-ahead diagnostics, and is strongest in
analyst- and earnings-related news. It is a close working-paper analogue of our text track.

## 1. GKX: elements and how they map to this project

### 1.1 Sample splitting and tuning

| GKX | Here today | Adopt? |
|---|---|---|
| Three disjoint, temporally ordered blocks: training (18 years, expanding), validation (12 years, rolling), test (1 year). Refit annually. No cross-validation, "to maintain the temporal ordering". Hyperparameters chosen on the validation block; parameters re-estimated on training only. | 252-day rolling window, monthly refit; penalties chosen from the previous 12 months' *realised* out-of-sample record of every candidate (`sse` or `rankcorr` rule). The NN notebooks use a temporal 80/20 split inside the window. | Our walk-forward rule is a legitimate validation scheme (closer to a rolling-origin evaluation than to a fixed block) and cheaper. Keep it for the linear path. For neural nets and trees adopt the explicit split: fit on the first part of the window, tune and early-stop on the last part, then predict. |
| Long training histories (18-48 years of monthly data). | One year of daily data, 600k+ rows per window. | Many stock-days do not replace independent dates. Test 2- and 3-year windows with the same history opportunities for all compared estimators, as specified in the shared protocol. |

### 1.2 Feature preparation

| GKX | Here today | Adopt? |
|---|---|---|
| Every characteristic is cross-sectionally *ranked* each period and mapped to [-1, 1]; missing values set to the cross-sectional median. Characteristics lagged to avoid look-ahead. | Columns standardised within the training window; `dm` variants remove date means. Message assignment uses a fixed 4 p.m. Eastern cutoff before the forward close-to-close target. | Compare daily ranks across all matched estimators. A different transformation on each date can change pooled ordering, so trees are not invariant to this operation. Keep the 384 embedding coordinates and two agreement measures unranked, with fitting-only scaling under the shared protocol. |
| Macro x characteristic interactions (`z = x_t (x) c_it`). | None. | Deferred in the original daily-ranking scope. Removing a common return component does not eliminate the possibility of state-dependent cross-sectional relationships; test such interactions only under an explicit hypothesis and common information set. |

### 1.3 Objective functions

| GKX | Here today | Adopt? |
|---|---|---|
| Squared loss; **Huber loss** `H(x; xi) = x^2 if |x| <= xi, 2 xi |x| - xi^2 otherwise`, `xi` tuned; the "+H" variants beat squared loss for OLS, ENet, GLM, GBRT. | Squared loss on the return (over-dispersed, Report 01) or on the daily rank of the return (what works). | Adopt as a *test*: Huber on the raw return target for `textcore`. If it recovers most of the rank target's ordering, the rank result has a conventional explanation (heavy tails), which is easier to defend than a change of target. |
| The main objective pools observations; inverse-cross-section-size and market-value weights are also considered. `w_it = 1/N_t` gives each period equal weight. | Unweighted pooled fit. The number of tweeted stocks per day grows about tenfold over the sample (Report 02). | Equal-date loss is our proposed primary adaptation, to apply consistently across all estimator families. Value weighting of the loss is a separate specification from value weighting of portfolios. |

### 1.4 Models and regularisation

| GKX | Here today | Adopt? |
|---|---|---|
| **Elastic net** `lambda (1-rho) sum|theta| + 0.5 lambda rho sum theta^2`, both tuned. | Ridge, lasso, elastic net on the precomputed Gram (Report 03). | Done. |
| **PCR / PLS**: `K` components tuned; PCR from the SVD of `Z`, PLS by SIMPLS. Both beat the elastic net on the 920 predictors ("characteristics are partially redundant and fundamentally noisy signals"). | Not run on the text. | Adopt. Both are trivial on the 386-column Gram (PCR = the eigendecomposition we already compute for ridge). PLS is the interesting one: it asks how many *return-relevant* directions the embedding has. If 2-5 components carry the whole 0.040, the text signal is low-dimensional and can be named. |
| **Generalised linear model**: spline expansion per predictor with a group lasso; fails to beat linear because it has no interactions. | None. | Skip for the embedding. For the *core* features it is cheap and relevant: the attention effect is a tail effect (Report 01), so a spline in log volume may beat the linear term. Add to the benchmark check rather than as a new model. |
| **Random forest** (depth, features per split, trees tuned; forests choose 1-5 levels) and **GBRT** (depth typically 1-2, shrinkage, number of trees). [IA: RF depth up to 6, 300 trees; GBRT depth 1-2, up to 1,000 trees, learning rate 0.01 or 0.1.] | `03b`/`03c` are empty. | Adopt for `textcore` with the rank target (Section 3). CKX's forest on raw embeddings needed 10,000 trees and a 100-observation leaf minimum to work (Section 2.3), so also give the trees the 53 features plus a PLS-compressed text input. Note CKX skip boosting on purpose (sparse learners lose when signals are weak); run GBRT anyway, since GKX found it competitive on characteristics. |
| **Neural nets NN1-NN5**: 32 / 32-16 / 32-16-8 / 32-16-8-4 / 32-16-8-4-2 neurons ("geometric pyramid"), ReLU, linear output. Regularisation: L1 penalty on weights, Adam with learning-rate shrinkage, early stopping on the validation block, batch normalisation, ensemble of 10 random seeds. [IA: L1 in (1e-5, 1e-3), learning rate 0.001 or 0.01, batch 10,000, up to 100 epochs, patience 5.] NN3 best; NN4-5 no better: "shallow learning outperforms deeper learning". | `03d` has sklearn `MLPRegressor` notebooks (fixed (8,4,2) and tuned halving architectures, 1-4 layers) for 2 and 53 features on the full panel; the 2-feature versions predict constants. Not run on the text. | Adopt the recipe for the text track: NN1-NN3 with pyramid widths scaled to the 388-column input (CKX used 128-64-32), ReLU, batch norm, Adam, early stopping on the last 20% of the window, weight penalty, **seed ensemble of 5-10**. The seed ensemble is the most important item: one MLP run at this signal-to-noise ratio is noise. Rank target, squared loss. CPU is adequate for 600k x 388 and three thin layers. |

### 1.5 Evaluation

| GKX | Here today | Adopt? |
|---|---|---|
| `R2_oos = 1 - sum (r - r_hat)^2 / sum r^2` over the test block, **without demeaning**: "predicting future excess stock returns with historical averages typically underperforms a naive forecast of zero". Reported for all stocks, top-1,000 and bottom-1,000 by market cap. | OOS R2 on date-de-meaned target and predictions (Report 01). | Date-demeaned R2 is a descriptive cross-sectional diagnostic, not the same feasible forecast benchmark as zero: the future cross-sectional mean is unknown at formation. Report return-unit R2 only for return-level forecasts, with explicit zero/historical benchmarks and size splits. |
| **Diebold-Mariano test adapted to a panel**: `d_12,t+1 = (1/n_t+1) sum_i [ (e1_i,t+1)^2 - (e2_i,t+1)^2 ]`, `DM_12 = mean(d) / NW s.e.(d)`; Bonferroni for the 12-way comparison (one-sided 5% critical value 2.64). CKX report pairwise DM tests, but their exact test differential is not clearly specified; Table 4's note refers to top/bottom quintile predictions and its comparison count differs from the text. | The original review found only own-model t-statistics; paired HAC comparisons were subsequently implemented in Report 06. | Use daily paired loss differences for return-level models and explicitly defined paired IC/spread differences for ranking models. The latter are our adaptation, not an exact CKX replication. Count the actual planned comparison family rather than copying either paper's critical value. |
| Variable importance: (a) drop in panel R2 from setting predictor `j` to zero in the training sample, model fixed; (b) SSD = sum of squared partial derivatives; mean impurity decrease for trees. Normalised to sum to one; ranks summed across models. Placebo: five simulated noise characteristics appended to check that importance is not spurious. | Feature-set decomposition (`core` / `text` / `textcore` / `all` / `textall`); prediction loadings (Report 02). | Adopt (a) at the *group* level for the nonlinear models: zero out the 384 dims, the 2 agreement measures, the 2 core features in turn; report the drop in validation rank correlation. Adopt the placebo cheaply: append 5 noise columns to `textcore`; a model that ranks them above the median embedding dimension is overfitting. |
| Marginal-association plots (expected return vs one characteristic over [-1, 1], others at median) and pairwise interaction plots. | None. | Adopt for the two core features once an NN or GBRT exists: the shape of the log-volume response is the attention effect made visible, and its interaction with net sentiment is the "heavy mostly-bullish chatter earns -21 bp" finding (Report 01) in one picture. |

### 1.6 Portfolios

| GKX | Here today | Adopt? |
|---|---|---|
| Prediction-sorted deciles, long D10 / short D1, monthly, value- and equal-weighted; the table shows **predicted** and **realised** mean return per decile side by side, with SD and Sharpe. Equal-weight results re-run excluding stocks below the 20th NYSE size percentile. | Daily deciles, EW and cap-weighted spreads (Report 02), realised only. | Adopt the Pred / Avg decile table for return-level models (the calibration diagnostic that exposed the over-dispersion in Report 01) and the NYSE-20th-percentile exclusion as the standard micro-cap check, next to the size terciles. |
| Max drawdown `max_{t1<=t2} (Y_t1 - Y_t2)` on cumulative log return; max one-month loss; **turnover** `(1/T) sum_t sum_i | w_it - w_it (1+r_it+1) / (1 + sum_j w_jt r_jt+1) |`; FF5 + momentum alphas, R2, information ratio. Reference: a short-term-reversal decile spread turns over 172.6% per month, size 22.9%. | `05/time_series_regressions.ipynb` runs CAPM / FF3 / FF5 / FF6 with HAC errors; no turnover or drawdown. | Adopt turnover and drawdown in `05`. Turnover decides whether a daily-rebalanced small-cap strategy survives costs (Report 00 next step 2; CKX Section 2.4 shows how badly). |
| Meta-strategies: (i) equal-weighted average of all models' decile portfolios (Sharpe 2.49 EW, beats every single model); (ii) pick the model with the best validation R2 each year. | None. | Adopt (i) cheaply: average of the daily rank-standardised predictions of `textcore` elastic net, lasso and later NN/GBRT. |
| Bottom-up factor-portfolio forecasts; Campbell-Thompson timing gains `SR* = sqrt((SR^2 + R^2)/(1 - R^2))`. | None. | Deferred from the original daily stock-ranking experiment. Aggregate forecasting is now a separate research question, requiring a time-series target and benchmark. |

## 2. CKX: elements and how they map to this project

### 2.1 Corpus construction (Section 3.1, Table 2)

| CKX | Here today | Adopt? |
|---|---|---|
| Refinitiv RTRS + third-party news, January 1996 to June 2019 (US extension to May 2022 for the look-ahead test). Keep only documents **tagged to a single stock** ("the information in such an article may be positive for one stock and negative for another"); require non-missing open-to-open returns; drop articles under 100 or over 100,000 characters; drop near-duplicates, defined as bag-of-words cosine >= 0.8 with any article on the same stock in the prior five business days. US: 11.2M raw -> 7.0M single-stock -> 4.8M matched -> 4.1M length-filtered -> 3.0M novel. Alerts (headline only) kept separately; the first alert in a "take sequence" carries most of the signal. | Messages with a Bullish/Bearish tag (35% of raw); cashtag explosion so a message counts toward every ticker it names; no length filter; no duplicate filter. | (a) **Single-cashtag**: build the stock-day embedding from single-cashtag messages only (or carry the single-cashtag share as a feature) and compare. (b) **Near-duplicates**: StockTwits has bots and copy-paste promoters; their rule (cosine >= 0.8 vs the same stock's messages over five days) is directly usable once message-level vectors are kept in the re-encode; a cheap proxy today is messages per unique user. (c) **Length**: drop messages that are only a cashtag plus an emoji before pooling. All three belong in the untagged-track re-encode (Report 00 next step 3). |
| Article timing: predictions target the **open-to-open** return; news arriving on day t sets positions at the open of t+1, held to the open of t+2; articles between 9:00 and 9:30 am are pushed to the next day's input set so that a trader has time to refit before the opening auction (Figure 2). Appendix D.3: with the same open-to-open predictions, executing close-to-close cuts the EW Sharpe from 4.59 to 2.19 and the VW Sharpe from 1.10 to 0.14; for large stocks "most of the predictive content dissipat[es] within a single trading day". | Cleaning uses a fixed 4 p.m. Eastern cutoff. Messages strictly after it on trading days map to the next trading close and then predict the following close-to-close return, omitting the initial overnight/session reaction. Early closes and exactly-at-close messages require an execution audit. | CKX's delayed C2C execution is not an exact equivalent of our different input window and C2C training target. Their findings motivate a timing experiment, not a forecast of our results. Audit feature cutoff, label maturity and executable entry together; then construct separate overnight, intraday and next-open targets where data support them. A full-day aggregate cannot assume a trade at that same close. |

### 2.2 Embedding construction (Sections 2.1, 2.4, Appendix D.4-D.5)

| CKX | Here today | Adopt? |
|---|---|---|
| Pretrained models, no fine-tuning: BERT-large (1,024 dims), RoBERTa-large, LLaMA 13B (5,120), LLaMA2 13B, LLaMA3 8B (4,096), OpenAI `text-embedding-3-large` (3,072). Article vector = equal-weighted **mean of the token vectors** over the first 512 tokens; longer context (1k-8k tokens) changes nothing (Table IA10). Model scale: gains are "modest and non-monotonic" (Table IA8: LLaMA2-70B 4.89 vs LLaMA3-8B 4.59 vs LLaMA-7B 3.99). Pairwise DM tests among LLaMA 1/2/3 and ChatGPT are insignificant. | `all-MiniLM-L6-v2` (384 dims, 22M parameters), sentence-level mean pooling, then mean over the day's messages. | Our two-stage mean (tokens -> message -> day) is the same construction one level up, which is reassuring. The scale evidence is *less* alarming than the 2022 drafts suggested: the big step is dictionary -> transformer, not small -> large transformer. A 335M-parameter encoder at 1,024 dims costs roughly 10x MiniLM's 44 h on CPU, so it needs a GPU; keep it as the representation upgrade (Report 00 next step 4) but do not expect a jump. |
| Ensembling across language models (2024 slides: Sharpe 5.1 vs 4.6 for the best single model; the PDF shows LLM portfolios correlated 0.6-0.8 with each other, Figure 7). | One encoder. | Only after a second encoder exists. |
| **Headline vs body vs AI summary** (Section 4.2): summaries generated by LLaMA3-Instruct beat both (LLaMA3 EW Sharpe 3.59 headline, 4.59 body, 5.42 summary); word-based models do *worse* on longer text. | Messages are already short. | The analogue is *which messages* to pool, not summarisation. Note only. |
| **Negation / context examples** (Section 3.3): "concerns have been raised" read as positive by word models; "erase all 2018 losses" read as negative. | Report 01: the embedding's stance adds nothing to the tag. | Cheap analogue: stock-days where the embedding-implied stance disagrees with the tag; is the text ordering stronger or weaker there? One table. |
| **Look-ahead** (Section 4.1): performance *after* the encoder's knowledge cutoff (BERT Oct 2018, RoBERTa Jul 2019) is higher than before (EW Sharpe 4.97 / 4.91 vs 2.90 / 3.72); anonymising firm names, people and dates in the text (86% of articles masked) leaves Sharpe ratios essentially unchanged (Table 15). | Our best reported year is 2022 and 2023 data exist, but the exact cutoff of the MiniLM checkpoint and its sentence-embedding training inputs has not been verified. | Record checkpoint and training-corpus provenance; mark an unknown cutoff as unknown. Small parameter count does not establish absence of memorization. Report later subperiods descriptively, and consider verified post-release data, masking and historically admissible representations as distinct diagnostics. Do not label 2022-2023 proven post-cutoff evidence. |

### 2.3 From embedding to prediction (Sections 2.2, 3.4)

| CKX | Here today | Adopt? |
|---|---|---|
| `E(r_i,t+1 | x_i,t) = x_i,t' theta`, pooled regression with **ridge**. Footnote 18 gives a log grid 1e-1 to 1e8 "scaled by sample size and feature dimension" without the normalization formula. Section 2.2 specifies annual 6 + 2 + 1-year temporal splits; Section 3.4 calls penalty selection cross-validation but does not specify the score or final refit rule. Target: next open-to-open return; footnote 10 replaces the older three-day-sign target with next-day returns. | Pooled ridge / lasso / elastic net over a 252-day window (Report 03); date-de-meaned variants; rank target. | Our ridge estimator is related, but the preprocessing, target and timing differ. Numeric penalty magnitudes cannot establish stronger shrinkage without matching objective and feature normalization. Specify our normalized loss and tune a declared grid. Return-level versus rank targets, window length and forecast clock require matched experiments; the paper's successful results do not resolve why our earlier return-level fits were weak. |
| **Nonlinear downstream models on LLaMA3 embeddings** (Table 7, EW long-short Sharpe): lasso 3.09, ridge 4.59, **random forest 5.51**, **NN 5.26**; VW: NN 1.58 best. RF: 10,000 trees, min leaf 100, subsample {0.3, 0.5, 0.8}, depth {10, 20, 30}, features per split {sqrt(P), P/5, P/10, P/20}. NN: 128-64-32, batch norm, ReLU, dropout, L2 in {0.1, 0.05, 0.01, 0.005}, learning rate 0.05/{100, ..., 1000}, early stopping with patience 5. No boosting ("sparsity-inducing methods are dominated by dense models when predictive signals are weak"). Net of costs (Table 10) RF and NN keep their lead over ridge at every turnover level. | Ridge ~ elastic net > lasso on ordering (Report 03), same ranking as theirs among linear models. | This is the concrete expectation for the NN/trees item: +10-20% over ridge, with a forest that is deep (10-30 levels), heavily averaged (10,000 trees) and leaf-constrained rather than GKX's shallow forest. Run both configurations. Their finding that a *better embedding with ridge* (ChatGPT) beats *RF/NN on a weaker embedding* (LLaMA3) is the argument for ordering the to-do list as representation before estimator, once a GPU exists. |
| Residual-return targets (CAPM / FF3 / FF6 residuals from intraday factors) improve the EW Sharpe slightly (4.58 / 4.85 / 5.03 vs 4.59). | DGTW-adjusted target (Report 02), same conclusion: ordering unchanged. | Done. |
| Earnings-announcement exclusion (t-1 to t+1): EW 4.30 vs 4.59; "a substantial portion of the predictive content in news is not mechanically driven by earnings". | None. | Adopt as a robustness row once earnings dates are in the panel (Compustat/IBES dates needed); cheap otherwise. |

### 2.4 Portfolio evaluation and costs (Sections 2.3, 3.3, 3.5, 4.4)

| CKX | Here today | Adopt? |
|---|---|---|
| Daily **quintile** long-short among stocks with news that day (deciles as robustness: EW 4.28 vs 4.59, VW 1.25 vs 1.10), EW and VW, one-day holding. Reported as annualised mean, SD and Sharpe of long, short and long-short legs; long leg carries most of the return, short leg roughly flat (LLaMA3 EW: long 2.08, short -0.33). Prediction accuracy reported as the mean daily cross-sectional **correlation** in percent (ChatGPT 2.68, LLaMA3 2.45, BERT 1.67, dictionary 0.98) with a large-cap (above NYSE median) subsample and a yearly table. | Tweeted stock-days sorted into deciles; EW and cap-weighted spreads in bp/day with t-statistics (Report 02); rank correlation ~0.03-0.04. | Same design. Adopt their reporting: annualised Sharpe of the daily long-short series next to bp/day, the long and short legs separately, and the above-NYSE-median subsample. Our Spearman 0.037-0.040 on tweeted days and their Pearson 0.024-0.027 on news days are the same order of magnitude; worth stating in the paper. |
| **Transaction costs** (Section 3.5): 10 bp per 100% turnover for large stocks, 20 bp for small (NYSE 20th percentile); turnover `(1/2T) sum_t sum_i | w_i,t+1 - w_i,t (1+y_i,t+1) / (1 + sum_j w_j,t y_j,t+1) |`; **exponentially weighted calendar-time (EWCT)** signals rebalance only a fraction `theta` of the portfolio per day. Articles: gross Sharpe 4.59 at full turnover but net *negative* above theta = 0.5; net Sharpe peaks at 0.77 at theta = 0.1 (turnover 17%/day). Alerts: net 1.68 at theta = 0.4. Dictionary and SESTM never earn a positive net Sharpe on articles. | None; Report 02's decile spreads are gross of costs and daily rebalanced. | **Adopt the whole block** in `05`: two-tier costs, the turnover formula, and EWCT smoothing over theta in {0.1, ..., 0.9}. Their article result is a warning: a daily text signal in small stocks at full turnover is a net loser at 20 bp. The turnover-controlled version is the only honest economic claim available to us, and it is what Report 00 next step 2 should produce. |
| Factor alphas: daily DMRS + q-factors + momentum + short- and long-term reversal (EW alpha 0.52 annualised for ChatGPT vs mean 0.53; VW 0.16 vs 0.16, all significant); monthly JKP 13-theme regressions on the theta = 0.1 portfolio (EW alpha ~6%/yr of ~10%; VW insignificant). | FF5 / FF6 with HAC in `05`. | Adopt short-term reversal as a factor in the daily regressions (it is the one factor that loads consistently in their Table 20) and run the monthly-aggregated JKP or FF version on the turnover-controlled portfolio. |
| **Reversal and double sorts** (Section 4.3, Table 17, Figure 9): five-day reversal is strong in the full universe (EW Sharpe -2.22) and *attenuated* among stocks with news (-1.48); independent 5 x 5 sorts on past five-day return and the news signal show the text signal monotone within every reversal quintile. | Report 00 open item 1 (past-return controls in the benchmark). | Adopt exactly this: (a) reversal spread on all stocks vs tweeted stocks; (b) 5 x 5 independent sort on past 5-day return x text prediction. It is the cleanest way to show the text is not reversal, and it is the same check as adding lagged returns to the core benchmark. |
| **Delayed formation** (Table 18, Figure 10): Day+1 EW Sharpe 4.58, Day+2 1.35, Day+3 0.98, Day+4 0.91, then ~0.5; large stocks absorbed within 1-2 days, small stocks 3-4 days. Alerts: first take (TS1) 4.79 > second 3.79 > rest 3.02. | Horizon table to 63 days (cumulative), size terciles, no delayed-formation or size x horizon table. | Adopt the **skip-a-day** table (form on t+2, t+3, ... with day-t messages) by size tercile. Our cumulative horizon table cannot separate "accrues slowly" from "earned on day 1 then flat"; theirs can. It also answers the implementability question at realistic latency. |
| Topic decomposition (Section 4.5): t-SNE + 200 K-means + agglomerative clustering into 10 meta-topics on the embeddings of traded articles; analyst-related news most profitable. | None. | Adopt in a light form later: cluster the day-level embeddings of top/bottom-decile stock-days and report the rank correlation by cluster. It is the only interpretability tool that works on an embedding without labels. |

### 2.5 Discussant's suggestions (Li, 2024 slides) that still apply

1. Labels from the immediate market reaction rather than multi-day windows: the current PDF already moved to next-day open-to-open; for us the overnight/intraday split is the feasible version.
2. Benchmark against news momentum (Jiang, Li and Wang 2021): the return earned while messages arrive as a predictor. Same conclusion as Report 00 item 1: the core benchmark needs the message-day return.
3. Front-loaded predictability (is the effect in the first minutes after the open?): CKX's D.3 open-vs-close comparison is the answer at daily resolution; ours would be the overnight leg.
4. News clustering: does today's signal predict tomorrow's *messages*, and is the return effect associated with continued attention? Tomorrow's message count can be a separate prediction outcome. Conditioning on tomorrow's realized attention is a retrospective association, can introduce post-outcome or selection bias, and does not by itself identify a mechanism. It cannot enter an ex ante forecasting benchmark.

## 3. Recommended adoptions, in order

Costs are for this machine (CPU only). "Where" names the notebook or tool to change.

| # | Item | Why | Where | Cost |
|---|---|---|---|---|
| 1 | **Pairwise DM-type test** on daily rank-correlation and decile-spread differences vs the core benchmark, Newey-West s.e., Bonferroni note (GKX eq. 20; CKX Table 4). | Turns every "0.040 vs 0.031" into a test statistic. | `tools/evaluate_predictions.py` | hours |
| 2 | **Past-return controls**: same-day, 5-day and 21-day returns in the benchmark; reversal spread on all vs tweeted stocks; 5 x 5 sort on past 5-day return x text prediction (CKX Table 17 / Figure 9). | GKX: reversal is the top predictor in every model; CKX: the text must be shown incremental to it. Report 00 open item 1. | text-only notebook `FEATURE_SET`; `build_text_master` if lags are missing; evaluator | half a day |
| 3 | **Equal day weighting** (`1/N_t`) and a declared normalized ridge grid. | Composition drift; numeric penalty ranges across papers do not establish comparable shrinkage. | shared fitting path for matched estimators | hours |
| 4 | **PLS / PCR** on the 386 text columns, `K` chosen walk-forward. | Is the text signal low-dimensional? GKX: dimension reduction beat the elastic net on redundant noisy predictors. | text-only notebook (`ESTIMATOR = pls | pcr`) | hours |
| 5 | **Huber loss** on the raw return target for `textcore`. | Conventional fix for heavy tails; explains or replaces the rank target. | text-only notebook | hours |
| 6 | **Neural nets NN1-NN3** on `textcore`, rank target, GKX recipe with CKX widths (128-64-32), batch norm, Adam, early stopping on the last 20% of the window, weight penalty, 5-10 seed ensemble. | User's request; CKX: NN beats ridge by 15% in EW Sharpe and 40% in VW Sharpe on the same embedding. | new `03d/prediction_neural_network_text.ipynb` on `text_master.pkl`, `index = mm_index`, registered in 04/05 | 1-2 days incl. runs (143 refits x seeds) |
| 7 | **Random forest and GBRT** on `textcore` (raw 388, and 53 + PLS-compressed text), rank target; forest in both the GKX shallow and the CKX deep-and-leaf-constrained configuration. | User's request; CKX: RF is their best EW model on embeddings. | new `03b`/`03c` notebooks, LightGBM / sklearn | 1 day |
| 8 | **Portfolio reporting and costs**: Pred vs realised per decile, long and short legs, annualised Sharpe, GKX turnover and drawdown, CKX two-tier costs with EWCT smoothing over theta, NYSE-20th-percentile exclusion, short-term reversal added to the factor regressions. | Economic framing (Report 00 next step 2); CKX shows a full-turnover daily small-cap text strategy loses net of 20 bp. | `05/form_portfolios.ipynb`, `05/time_series_regressions.ipynb` | 1-2 days |
| 9 | **Timing**: audit fixed-close message assignment and execution; construct overnight/intraday or next-open targets separately; delayed-formation table by size (Day+1 ... Day+6). | CKX D.3 and Table 18 show that timing matters in their sample. Their C2C execution is not an exact equivalent of our current specification. | `perpare_training_data` (open prices), `build_text_master`, evaluator | 1-2 days |
| 10 | **Corpus hygiene** for the re-encode: single-cashtag-only pooling, minimum length, near-duplicate rule (cosine >= 0.8 vs the same stock's prior five days) on message vectors. | CKX filters; StockTwits has more repetition than Reuters. | `features_08` re-encode (untagged track) | with the re-encode |
| 11 | **Group-level importance, placebo columns**, marginal-response and interaction plots for the core features from the NN/GBRT. | GKX interpretation tools; makes the attention effect visible. | after 6-7 | hours |
| 12 | **Ensemble** of the rank-standardised predictions of the working models. | GKX and CKX both report the ensemble beating the best single model. | evaluator | hours |
| 13 | **Encoder provenance and later-period diagnostics**, with cutoff claims only when verified. | CKX Section 4.1 motivates checks; 2022-2023 is not automatically an untouched or proven post-cutoff sample. | metadata and reports; masking requires re-encoding | scope dependent |
| 14 | **Earnings-window exclusion** (t-1 to t+1) and **topic clustering** of traded stock-days. | CKX robustness and interpretation; both need extra inputs (earnings dates; a clustering pass). | later | 1 day each |

Items 1-5 fit inside the current linear framework and should precede the nonlinear models, so
that NN and trees are compared against a benchmark that already carries past returns and a
formal test. Items 6-7 are the user's new to-do item. Items 8-9 are the economic framing, and
item 9 is the one most likely to change a headline number. Item 10 and the encoder upgrade
wait for the re-encode / GPU.

## 4. What not to adopt, and why

- **The papers' literal window lengths and refit frequency.** GKX forecast monthly returns
  using predictors with different update frequencies; CKX forecast daily news returns with
  six training and two validation years. Our shorter, changing corpus motivates a different
  baseline. Compare longer histories across all matched estimators, not only nonlinear ones.
- **Macro interactions and bottom-up factor-portfolio forecasts.** Deferred from the original
  stock-ranking scope. Aggregate forecasting is now a separate question in the research
  agenda. Cross-sectional rank targets remove its outcome of interest, so that track requires
  return levels and a separate time-series design. State interactions can also matter within
  cross-sectional forecasting and are not ruled out by demeaning.
- **OOS R2 without demeaning as the headline metric.** Correct for a return-level monthly
  problem; for a daily cross-sectional ordering on a tweeted subsample, rank correlation and
  the decile spread are the objects, with R2 for return-level models only (the convention in
  Reports 01-03). CKX themselves report correlations, not R2.
- **CKX's earlier three-day-return sentiment classifier.** The current PDF dropped it in
  favour of the direct next-day regression (footnote 10), and we have something they do not:
  the poster's own label. If a market-reaction label is ever used here, it is as a contrast to
  the tag ("does the crowd label what the market rewards?"), not as the primary model.
- **Deep networks and very wide grids.** GKX's deeper nets do not improve on their shallower
  alternatives; CKX report a three-layer network without establishing an optimal depth.
  Beginning with NN3 is a compute/design choice, not evidence that deeper models cannot help.
- **Model-scale chasing.** CKX's Appendix D.4: gains from larger LLaMA variants are "modest
  and non-monotonic"; the step that matters is dictionary -> transformer, which we have made.

## Files

- This report only. No code changed. To-do items: `reports/00_project_state.md` Section 5,
  items 7-8.
- Papers: `D:\StockTwits\Literature\hhaa009.pdf` (GKX), `D:\StockTwits\Literature\ssrn-4416687 (1).pdf` (CKX).
- Verified appendix: `D:\StockTwits\Literature\Gu Kelly Xiu - Internet Appendix.pdf`.
