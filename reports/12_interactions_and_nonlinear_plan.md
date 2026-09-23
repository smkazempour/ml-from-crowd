# Next study: explicit interactions and nonlinear prediction

**Status update, September 22, 2026:** Stage A is complete; see
[the short interaction results](13a_linear_interaction_summary.md),
[the complete register](13_linear_interaction_results.md), and
[the next-session handoff](NEXT_SESSION_HANDOFF.md). Stages B and C remain proposed
and have not been implemented or launched. The planning text below records the
original sequence; its Stage A declaration is now
[LINEAR_INTERACTION_EXPERIMENT.md](LINEAR_INTERACTION_EXPERIMENT.md).

Prepared September 22, 2026 following the user's request to study interaction
terms and optimize nonlinear models, particularly trees. This is a proposed
research and implementation plan, not a completed experiment or an executable
frozen specification. No new model fits were launched while preparing this plan.

## Questions and sequence

1. Can named interactions improve the existing linear forecasts?
2. Do those gains come from stock-characteristic relationships, social-media
   relationships, or the interaction of the two?
3. Do optimized tree ensembles or neural networks improve on these explicit
   interaction benchmarks, and do they reveal additional social information?

Implement the explicit interactions and tree pipeline first, with a runtime and
integrity pilot. Develop the NN extension next using the same data and evaluation
interfaces. The full research menu should be declared before forecast-period
performance is used to judge it. Pilot forecast outcomes do not select the menu.

## Shared procedure

Keep 504 initial training sessions, 126 chronological validation sessions, the
original maturity exclusions, and a final refit on their union. Use monthly
updates, daily return-rank targets, squared loss, and equal total weight per day.
Select settings by mean daily validation Spearman IC. Raw next-day returns are
primary; DGTW is secondary. All comparisons use the same existing message-covered
stock-days and correctly labeled transformations.

The baseline stock characteristics and social scalars already use daily ranks;
see [the transformation correction](11b_sample_and_transformation_audit.md).
Use that certified original representation, not the second-ranking branch, in
this study. The genuine raw-versus-ranked comparison remains a separate limited
follow-up using `characteristics_raw.npy`.

Recompute fitting-only scaling and PCA on the final fitting inputs. Construct
products and squares from the specified input representation, then scale the
derived columns using fitting data only. Do not rank the resulting products
again. Missingness flags remain main effects rather than automatically generating
products with every flag. Missing inputs follow the existing neutral-rank plus
flag convention, which must be explicit in the experiment manifest.

For trees, use training sample weights with equal total weight per date. Normalize
them to mean one and record that normalization explicitly: histogram boosting's
leaf L2 penalty depends on the absolute weight scale, unlike the unpenalized
weighted split criterion. Do not
use random validation splits or out-of-bag performance to select settings.
Choose boosting iteration counts on the chronological validation block and
refit for that fixed count on the combined sample. For NNs, choose stopping
epochs and the learning-rate schedule before the combined-sample refit; replay
the declared schedule rather than validating on observations now used to fit.

Retain the current calendar-aware paired evaluation, HAC5 with HAC21/HAC63
sensitivities, early/late periods, and gross equal-/capitalization-weighted
portfolio diagnostics. Register comparison families and their adjustment budgets
before scoring. These remain 2014-2022 development results; no untouched
confirmation period is currently certified.

## Stage A: explicit interactions

Let C be the 17 continuous market/past-return controls and their 17 missingness
flags; S be sentiment and attention. Start with ridge as the main interaction
estimator, with elastic net as a joint-model sensitivity. Keep OLS as a reference
where the design matrix is manageable. Include all parent main-effect columns
when adding their products. Including parent columns is not a claim that ordinary
elastic net enforces nonzero-parent hierarchy.

Use a sequence that distinguishes curvature within a feature from interactions
between features:

| Basis | Total columns, excluding intercept | Purpose |
| --- | ---: | --- |
| C main effects | 34 | Existing characteristic-only reference |
| C plus continuous C squares | 51 | Univariate curvature in characteristics |
| C plus squares and all distinct C-by-C products | 187 | Characteristic-only quadratic benchmark |
| Quadratic C plus S main effects | 189 | Social addition given richer characteristic controls |
| Above plus the two S squares | 191 | Social curvature without social interaction products |
| Above plus all C-by-S and sentiment-by-attention products | 226 | Joint social interaction model |

There are **35 named social interaction terms**: sentiment and attention each
times the 17 continuous characteristics (34), plus sentiment times attention (1).
Compare each of these individually with the same 191-column benchmark, then fit
all 35 together. Individual additions and the joint block answer different
questions, so neither replaces the other. Evaluate predictive increments rather
than declaring a term useful because its fitted coefficient is nonzero. Adjust
inference for the whole declared search; do not report only favorable terms.

Examples with economic interpretations include sentiment by recent return,
attention by volatility, sentiment by liquidity, and sentiment by attention.
The full characteristic-by-characteristic benchmark matters: otherwise a social
product could compensate for an omitted relationship among market predictors.

For a bounded text-interaction extension, use **16 training-estimated embedding
PCs plus the two agreement variables**, T=18. Holding that dimension fixed makes
the first interaction comparison easier to interpret:

- Add T main effects to the 226-column basis: 244 columns.
- Add T squares: 262 columns.
- Add the two agreement inputs by 17 characteristics: 296 columns.
- Add the 16 embedding PCs by 17 characteristics: 568 columns in total.
- Add the 18-by-2 T-by-S block as a separate extension: 604 columns.

The agreement-only intermediate step separates those 34 products from the 272
embedding-PC products. A combined text-block gain must not automatically be
attributed to semantic content. The market and social interaction extensions
likewise test different conditional relationships.
Keep additive full-embedding models as references so a compressed basis is not
the only representation tested. Avoid all pairwise products among the 384 raw
embedding coordinates in the initial study. Monthly PCA coordinates can rotate
and change meaning; interpret their interaction blocks, not a pooled coefficient
on a purportedly stable "PC1" topic.

## Stage B: optimize trees and isolate interaction flexibility

Use histogram gradient boosting and random forests. A single shallow decision
tree can illustrate a fitted rule, but it is not the main forecasting benchmark.
For each family, cover characteristic-only, characteristics plus S, and those
inputs plus full or compressed text. The characteristic-only model must receive
the same opportunity to model nonlinear relationships.

Histogram boosting supports three scientifically useful procedures:

1. **Additive:** each tree branch may use only one feature; its effect can be
   nonlinear, but features cannot interact within a branch.
2. **Pairwise:** a branch may involve at most two distinct features.
3. **Unrestricted interactions:** ordinary trees subject to the declared size limits.

The installed scikit-learn 1.8.0 exposes `interaction_cst` with
`no_interactions` and `pairwise` options. The three procedures should share the
same structural candidate menu and chronological selection rule. A difference
measures the benefit of the fitted procedure's permitted flexibility; capacity
and regularization still change, so it is not a causal decomposition.

Proposed search ranges to finalize after profiling, without consulting pilot
forecast scores:

| Model | Candidate settings |
| --- | --- |
| Histogram boosting | Depth 2,4,8 with max_leaf_nodes=None; minimum leaf observations 100,500,2000; learning rate .02,.05,.10; leaf L2 penalty 0,1,10 with mean-one weights; evaluate iteration checkpoints through 1000 |
| Random forest | Depth 4,8,16 or unrestricted; minimum leaf observations 100,500,2000; feature fraction .3,.7,1; bootstrap row fraction .5 or 1; 500 and 1000 trees |

Use a reproducible, declared sample of approximately 24 structural configurations
per search procedure, including specified conservative and flexible anchors.
Iteration/tree-count checkpoints share a fitted path where supported. Finalize
the exact configuration list, any tree-count convergence check, and seed schedule
before the full run. This explores multiple dimensions without fitting every
combination of every possible setting. Expand a range only under a separately
recorded validation-based rule or a new experiment declaration.

Tree early stopping must use the project's daily IC, not silently substitute
pooled R-squared or validation MSE. The installed histogram-boosting interface
accepts explicit validation inputs and weights; a project scorer or external
iteration-checkpoint evaluator must enforce the existing daily-IC convention.

## Stage C: NN architecture and training optimization

The completed NN3 study used social inputs without market-characteristic controls
and retained the original training fit. It cannot substitute for the proposed
characteristic-conditioned, combined-sample-refit comparison.

Compare depth at a fixed first-layer width: **128**, **128-64**, and
**128-64-32**. Then, as a separately declared width comparison, use
**64-32-16**, **128-64-32**, and **256-128-64**. The shared NN3 configuration is
fitted once. These five structures explore both depth and width; parameter counts
must be reported rather than attributing every difference to depth alone.

Optimize a bounded menu of L2 penalties and learning rates using validation; keep
the existing ReLU/BatchNorm recipe and five-seed prediction average initially.
Record epoch ceilings, convergence behavior, learning curves and seed variation.
The same nested information sets, PCA construction, monthly dates and final fitting
sample apply to NN and tree comparisons. Runtime pilots determine safe server
worker allocation, not which architectures appear to win on forecast outcomes.

## Interpretation and confirmation

For a named interaction, report its paired out-of-sample predictive increment,
its stability across periods and activity groups, and a two-variable plot over
regions with actual joint data support. Keep forecast-period plots descriptive.
For tree findings, supplement plots with group removal and refitting under matched
tuning budgets; individual split importance or a coefficient sign is insufficient.
Correlated inputs can share predictive information and obscure individual-feature
attribution. If a tree suggests a previously unlisted interaction, treat the
subsequent explicit-term model as a new development experiment, with separate
confirmation still required.

Every results report must distinguish:

- Improved characteristic-only forecasts.
- Added sentiment/attention information conditional on comparable market controls.
- Added text information conditional on both market controls and simpler social measures.
- Better estimator performance for a fixed information set.

Keep the tagged-message corpus fixed during these model comparisons. Audit and
add untagged messages in a separate data extension using the same encoder and
initially the same stock-days; add their counts before their text so additional
attention and content are distinguishable.

## Implementation deliverables

- An isolated interaction feature builder with exact counts, source-column names,
  missingness rules, and fitting-only transformation tests.
- Resumable tree and NN adapters with chronological daily-IC selection, combined
  refits, candidate and seed diagnostics, and certified common prediction keys.
- A runtime/memory pilot spanning early and late sample months, then a frozen
  machine-readable candidate and comparison specification.
- An updated server bundle containing the characteristic cache and matched
  baselines. The existing social-only server bundle cannot simply be relabeled.
- Automatic completion/error monitoring, a detailed results record, and a short
  report organized around interaction and information-set questions.

## Technical references

- [Histogram gradient boosting, installed-version documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
- [Random forest regression, installed-version documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- [Permutation importance with correlated features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html).
