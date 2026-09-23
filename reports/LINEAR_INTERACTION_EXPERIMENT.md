# Explicit interaction regressions

Stage A of [the interaction and nonlinear-model plan](12_interactions_and_nonlinear_plan.md),
authorized September 22, 2026. This study asks whether explicit products of inputs
improve forecasts, and whether the improvement comes from stock characteristics,
sentiment/attention, or text. Trees and neural networks are later stages.

## Fixed procedure

Use the existing message-covered stock-days and original cached daily ranks.
There are 504 initial fitting sessions and 126 chronological validation sessions.
Monthly penalty selection maximizes mean daily validation Spearman IC, with the
existing deterministic tie rule. Refit on the union of those inputs, preserving
the original label-maturity exclusions. Re-estimate scaling and PCA on this union.
Daily return ranks, squared loss, equal total weight per day, and an unpenalized
intercept remain unchanged. Raw next-day returns are primary; DGTW is secondary.
The forecast interval is January 2014 through December 2022.

These years have already informed research decisions: results are development
evidence, not an untouched confirmation test. The sample includes retained,
sentiment-tagged StockTwits messages with embeddings; it is not all CRSP stock-days.
See [the sample and transformation audit](11b_sample_and_transformation_audit.md).

## Model menu

C contains 17 continuous market/past-return characteristics plus their 17
missingness flags. S contains sentiment and attention. T contains 16 fitting-only
embedding PCs and the two agreement measures, `embed_norm` and `embed_cos`.

| Design | Columns, excluding intercept |
| --- | ---: |
| Additive C reference | 34 |
| Additive C + S reference | 36 |
| Additive C + S + full text reference | 422 |
| C plus continuous C squares | 51 |
| Above plus all distinct C-by-C products | 187 |
| Quadratic C plus S | 189 |
| Above plus S squares | 191 |
| Above plus all 35 social interaction terms | 226 |
| Above plus T main effects | 244 |
| Above plus T squares | 262 |
| Above plus agreement-by-C products | 296 |
| Above plus PC-by-C products | 568 |
| Above plus T-by-S products | 604 |

Fit OLS, ridge, and elastic net on all 13 designs. Ridge is the primary estimator;
elastic net and OLS are matched comparisons. Separately fit each of the 35 named
social products, one at a time, on top of the same 191-column design, using ridge.
The products are sentiment times each continuous characteristic, attention times
each continuous characteristic, and sentiment times attention.

This gives **74 procedures per outcome, 148 total**, fitted in **216 monthly
outcome jobs**. A job includes validation candidates and all final procedures.
The largest OLS reference has 604 columns. The earlier lasso study remains an
additive reference; this new study uses the estimators specified for Stage A.

## Feature construction and tuning

Retain all parent main effects. Missingness flags enter as main effects only.
Construct characteristic/social squares and products from their cached ranks,
where an originally missing characteristic has neutral rank zero and a separate
flag. Do not rank products again. Construct text products from centered,
component-SD-scaled fitting-only PCs and the two raw agreement variables. Missing
agreement values receive their weighted fitting-input mean before products are
formed. PCA uses the legacy fitting-only standardized embedding coordinates and
equal-date covariance, including fitting inputs without observed outcomes.
Fit these transformations without future outcomes. Apply fitting-only weighted scaling to
the derived columns. Rebuild these transformations on the final fitting union.
The executable declaration records the exact imputation and PCA conventions.

Use the existing penalty menus: ridge strengths from 1,000 down to 0.000001;
elastic-net fractions 1, .3, .1, .03, .01, .003, .001, .0003, .0001 with L1
ratios .1, .5, .9. Transfer the selected fraction into the refit and recompute
its maximum penalty on the refitting inputs. PCA dimension is fixed at 16 for
this study; its basis is re-estimated each month. A monthly PC is not a stable
topic whose coefficient can be pooled over time.

The nine additive references must reproduce the certified earlier monthly
forecasts, including daily-rank agreement. The original numerical path is
preserved for these references, rather than changing archived fits.

## Evaluation and reporting

Use common prediction keys and the existing daily IC, gross equal-weighted and
capitalization-weighted decile-spread kernels. Compare paired daily differences
with calendar-aware HAC5 inference, HAC21/HAC63 sensitivity, and full/early/late
periods. Family sizes are fixed in the machine-readable declaration before fits.
Every individual term is compared against the same 191-column ridge benchmark;
inference accounts for all 35 trials. A nonzero coefficient does not establish a
useful interaction. The joint block answers a separate question.

Distinguish characteristic curvature, characteristic interactions, added social
information, and added text information. Agreement-by-characteristic gains must
not automatically be attributed to embedding content. Comparisons against the
full additive text model also change compression and feature shape; label them
as combined design comparisons.

The controller runs an early/late-month integrity pilot, then fitting,
verification, evaluation, a detailed report, and a short readable summary.
The pilot does not select the research menu by forecast performance. Each phase
is logged. An independent watcher records success, errors, or interruption and
sends the configured Windows desktop notification.

- [Exact machine-readable declaration](data/linear_interactions_v1_experiment.json)
- [Run status](LINEAR_INTERACTION_RUN_STATUS.md)
- [Start/resume instructions](../tools/LINEAR_INTERACTION_STUDY.md)
