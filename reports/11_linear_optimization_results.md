# Linear optimization: penalties, characteristic ranks and text compression

All 172 registered procedures completed evaluation: 86 for each target.

This is the complete audit record. [The short companion](<11a_linear_optimization_summary.md>) explains the main comparisons without ranking every model.

## Fixed procedure and experiment menu

Every model uses 504 training sessions followed by 126 validation sessions. Settings are selected by mean daily validation Spearman correlation; the final model is refitted on the exact original training and validation rows, preserving the purge. This is 630 fitting sessions, about 2.5 years. Forecasts are updated monthly. Close-t information predicts close-t to close-t+1 returns. The centered daily-rank target, equal-date squared loss and sample are unchanged.

The input sets are the same 17 market/past-return characteristics plus 17 missingness flags (34 columns), those inputs plus sentiment/attention (36), and all of those plus 384 embedding coordinates and two agreement inputs, embed_norm and embed_cos (422). No accounting or news predictors are added.

The complete crossed menu compares historical standardization with daily characteristic ranks; original with expanded penalty choices; full text with PCA text; and the original PCA menu with a menu that additionally permits 4 and 8 components. OLS, ridge, lasso and elastic net cover all input sets; group ridge covers the two social-augmented input sets. OLS has only one penalty-menu label because it has no penalty. Only text models have a PCA-menu choice. These deductions leave 86 distinct procedures per target, 172 total.

Daily ranks use contemporaneously available characteristic values on the formation date. Binary missingness flags and embedding coordinates are not cross-sectionally ranked. Estimation transforms, scaling and PCA use only permitted fitting inputs and are rebuilt for the final refit. PCA compresses only the embedding coordinates; the two agreement inputs remain separate and in the text penalty group. Sparse refits preserve selected penalty fractions while recomputing their numerical scale from the final fitting sample.

## Common sample

| Target | Models | Eligible stock-days | Common predictions | Observed outcomes |
| --- | --- | --- | --- | --- |
| raw | 86 | 3034035 | 3034035 | 3033080 |
| dgtw | 86 | 3034035 | 3034035 | 2748978 |

Every procedure within a target uses the joint finite-prediction intersection before outcome filtering. The social-covered stock-day universe is unchanged; the characteristic-only comparator is not an all-stock benchmark. Prediction batches limit memory use without changing that global intersection or reducing forecast precision.

## Summary of the primary comparisons

| Change | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Broader penalty choices | 38 | 6 | 0 | 0 |
| Daily characteristic ranks | 43 | 22 | 0 | 0 |
| Allowing smaller text summaries | 18 | 14 | 0 | 0 |
| Compressing text | 36 | 35 | 12 | 0 |
| Adding social-media inputs | 72 | 38 | 18 | 6 |
| Separate penalties for input groups | 16 | 11 | 4 | 0 |
| Penalization versus OLS | 76 | 75 | 16 | 0 |

Positive counts describe paired procedure differences, not independent replications. Significance counts use two-sided, family-adjusted HAC5 p-values below 0.05. Undefined statistics are omitted from the scored-count column but never reduce the correction budget. No family is called a winner from its count.

## Multiplicity and inference

All comparisons were declared before scoring. Bonferroni correction uses the full family budget separately within target, metric and period. HAC5 is primary; HAC21 and HAC63 are sensitivities. Confidence intervals are pointwise 95% intervals, not simultaneous intervals. These corrections do not account for repeated historical research decisions.

| Family | Comparisons per target/metric/period |
| --- | --- |
| Daily characteristic ranks | 43 |
| Penalization versus OLS | 76 |
| Broader penalty choices | 38 |
| Adding social-media inputs | 72 |
| Separate penalties for input groups | 16 |
| Compressing text | 36 |
| Allowing smaller text summaries | 18 |

The characteristic-only reference for grouped sentiment/attention is global ridge; that comparison combines the social addition with freedom to penalize groups separately. PCA text compares with the corresponding full-text procedure and with the corresponding smaller input set. Original and expanded penalized procedures share the same OLS comparator because the penalty axis is inapplicable to OLS.

## Rank IC and gross portfolio levels

Rows follow the registry order rather than realized performance. Rank IC is the mean daily Spearman correlation. Gross portfolio spreads are mean top-minus-bottom decile returns in basis points with fractional ties and the same daily stock sample. They exclude transaction costs, turnover, factor alpha and execution constraints; they are not attainable net strategy returns.

### Raw returns (primary)

| Procedure | Rank IC | Equal-weight spread (bp) | Cap-weight spread (bp) |
| --- | --- | --- | --- |
| OLS: Characteristics; full; standard; penalties original | 0.0684 | 23.12 | 12.51 |
| Ridge: Characteristics; full; standard; penalties original | 0.0694 | 20.22 | 15.59 |
| Ridge: Characteristics; full; standard; penalties expanded | 0.0692 | 20.13 | 14.84 |
| Lasso: Characteristics; full; standard; penalties original | 0.0688 | 20.27 | 15.49 |
| Lasso: Characteristics; full; standard; penalties expanded | 0.0686 | 19.22 | 12.12 |
| Elastic net: Characteristics; full; standard; penalties original | 0.0687 | 20.35 | 15.33 |
| Elastic net: Characteristics; full; standard; penalties expanded | 0.0684 | 19.44 | 13.47 |
| OLS: Characteristics + sentiment/attention; full; standard; penalties original | 0.0694 | 25.10 | 10.00 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties original | 0.0704 | 24.47 | 14.70 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0703 | 23.92 | 12.92 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties original | 0.0699 | 24.18 | 16.86 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0697 | 22.94 | 14.32 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties original | 0.0698 | 23.86 | 15.96 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0698 | 24.05 | 14.73 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties original | 0.0704 | 24.17 | 13.82 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0703 | 23.51 | 12.96 |
| OLS: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0652 | 25.54 | 7.21 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0683 | 23.10 | 11.42 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0685 | 22.99 | 13.74 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0698 | 23.97 | 13.05 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0696 | 24.37 | 13.74 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0695 | 23.48 | 10.77 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0695 | 24.67 | 11.71 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0702 | 23.99 | 12.69 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0700 | 24.08 | 12.70 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0692 | 27.08 | 10.41 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0694 | 27.04 | 11.02 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0703 | 25.41 | 13.53 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0704 | 25.74 | 13.67 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0702 | 25.18 | 14.65 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0703 | 25.44 | 13.61 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0699 | 25.84 | 16.26 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0701 | 26.05 | 16.32 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0695 | 24.41 | 12.72 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0698 | 24.93 | 14.19 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0700 | 25.64 | 14.40 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0701 | 25.71 | 14.26 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0697 | 25.07 | 12.28 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0698 | 25.03 | 11.92 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0704 | 24.89 | 14.28 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0704 | 25.11 | 15.14 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0704 | 24.43 | 13.75 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0703 | 24.66 | 13.58 |
| OLS: Characteristics; full; daily_rank; penalties original | 0.0684 | 23.26 | 11.90 |
| Ridge: Characteristics; full; daily_rank; penalties original | 0.0694 | 20.47 | 15.70 |
| Ridge: Characteristics; full; daily_rank; penalties expanded | 0.0692 | 20.09 | 14.17 |
| Lasso: Characteristics; full; daily_rank; penalties original | 0.0689 | 20.34 | 16.25 |
| Lasso: Characteristics; full; daily_rank; penalties expanded | 0.0686 | 18.75 | 11.81 |
| Elastic net: Characteristics; full; daily_rank; penalties original | 0.0687 | 20.46 | 15.12 |
| Elastic net: Characteristics; full; daily_rank; penalties expanded | 0.0683 | 19.27 | 13.57 |
| OLS: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0694 | 25.41 | 10.41 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0704 | 24.36 | 14.80 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0703 | 24.14 | 13.16 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0698 | 23.88 | 16.80 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0697 | 22.66 | 14.95 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0698 | 24.03 | 16.20 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0698 | 24.38 | 14.41 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0704 | 24.61 | 15.04 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0703 | 24.37 | 13.67 |
| OLS: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0652 | 25.52 | 7.28 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0683 | 23.17 | 11.54 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0685 | 23.00 | 13.06 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0697 | 24.07 | 12.91 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0696 | 24.10 | 13.41 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0695 | 23.52 | 10.98 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0695 | 24.62 | 10.92 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0703 | 24.59 | 14.82 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0700 | 23.84 | 12.25 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0692 | 27.35 | 9.74 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0694 | 27.55 | 10.72 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0703 | 24.99 | 13.01 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0704 | 25.43 | 13.33 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0702 | 24.93 | 14.54 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0703 | 25.43 | 14.00 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0699 | 25.39 | 15.77 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0703 | 25.76 | 16.20 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0696 | 24.32 | 13.04 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0701 | 25.06 | 15.25 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0700 | 25.39 | 14.59 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0701 | 25.63 | 14.89 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0696 | 25.03 | 12.63 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0698 | 25.03 | 13.37 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0704 | 24.99 | 14.95 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0704 | 25.03 | 14.70 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0704 | 24.04 | 11.97 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0704 | 24.22 | 12.43 |

### DGTW returns (secondary)

| Procedure | Rank IC | Equal-weight spread (bp) | Cap-weight spread (bp) |
| --- | --- | --- | --- |
| OLS: Characteristics; full; standard; penalties original | 0.0602 | 18.25 | 10.35 |
| Ridge: Characteristics; full; standard; penalties original | 0.0610 | 14.92 | 11.27 |
| Ridge: Characteristics; full; standard; penalties expanded | 0.0612 | 15.32 | 11.16 |
| Lasso: Characteristics; full; standard; penalties original | 0.0608 | 15.71 | 8.88 |
| Lasso: Characteristics; full; standard; penalties expanded | 0.0608 | 15.14 | 8.61 |
| Elastic net: Characteristics; full; standard; penalties original | 0.0609 | 16.20 | 10.37 |
| Elastic net: Characteristics; full; standard; penalties expanded | 0.0608 | 15.20 | 11.34 |
| OLS: Characteristics + sentiment/attention; full; standard; penalties original | 0.0615 | 20.62 | 13.88 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties original | 0.0627 | 18.66 | 11.41 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0627 | 18.38 | 11.69 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties original | 0.0624 | 18.88 | 9.96 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0623 | 17.50 | 8.72 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties original | 0.0625 | 19.01 | 10.58 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0624 | 18.18 | 11.32 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties original | 0.0627 | 18.27 | 10.55 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | 0.0627 | 18.12 | 11.45 |
| OLS: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0574 | 22.28 | 9.19 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0608 | 20.99 | 12.06 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0608 | 20.66 | 11.36 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0620 | 18.49 | 8.76 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0620 | 18.18 | 8.96 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0621 | 18.71 | 9.79 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0621 | 18.17 | 11.38 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | 0.0625 | 17.95 | 10.61 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | 0.0626 | 18.41 | 11.49 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0615 | 21.33 | 11.43 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0617 | 21.54 | 11.89 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0626 | 20.18 | 9.40 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0628 | 20.27 | 8.13 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0627 | 19.97 | 9.78 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0629 | 20.08 | 9.45 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0625 | 19.83 | 10.19 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0625 | 20.49 | 8.29 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0623 | 18.40 | 7.39 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0623 | 19.29 | 7.08 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0624 | 19.94 | 9.49 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0625 | 20.56 | 9.06 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0623 | 18.73 | 9.27 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0623 | 19.67 | 9.18 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | 0.0626 | 19.54 | 11.32 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | 0.0627 | 19.71 | 9.58 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | 0.0627 | 19.53 | 11.50 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | 0.0627 | 19.95 | 10.48 |
| OLS: Characteristics; full; daily_rank; penalties original | 0.0602 | 18.10 | 10.17 |
| Ridge: Characteristics; full; daily_rank; penalties original | 0.0610 | 15.33 | 12.10 |
| Ridge: Characteristics; full; daily_rank; penalties expanded | 0.0612 | 15.48 | 10.58 |
| Lasso: Characteristics; full; daily_rank; penalties original | 0.0608 | 15.42 | 8.95 |
| Lasso: Characteristics; full; daily_rank; penalties expanded | 0.0608 | 14.81 | 7.88 |
| Elastic net: Characteristics; full; daily_rank; penalties original | 0.0608 | 15.72 | 10.87 |
| Elastic net: Characteristics; full; daily_rank; penalties expanded | 0.0609 | 15.18 | 10.92 |
| OLS: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0615 | 20.67 | 13.50 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0626 | 18.51 | 11.64 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0627 | 18.49 | 12.02 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0624 | 18.59 | 9.92 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0623 | 17.18 | 8.55 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0625 | 18.65 | 10.04 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0624 | 18.23 | 11.12 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | 0.0626 | 18.47 | 11.13 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | 0.0627 | 18.48 | 11.90 |
| OLS: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0574 | 22.27 | 8.83 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0608 | 21.12 | 12.50 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0608 | 20.68 | 11.94 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0621 | 18.40 | 8.52 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0621 | 17.89 | 8.20 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0621 | 18.20 | 9.34 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0621 | 17.98 | 11.75 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | 0.0625 | 18.16 | 11.52 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | 0.0626 | 18.35 | 11.94 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0615 | 21.22 | 10.90 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0617 | 21.44 | 12.17 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0627 | 20.39 | 9.06 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0629 | 20.04 | 9.00 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0627 | 19.94 | 9.79 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0629 | 20.10 | 8.39 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0625 | 19.50 | 9.31 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0625 | 20.33 | 7.58 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0624 | 18.48 | 7.70 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0624 | 19.14 | 6.75 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0624 | 19.26 | 9.03 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0625 | 19.80 | 8.85 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0623 | 18.61 | 9.86 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0623 | 19.77 | 8.93 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | 0.0626 | 19.52 | 11.29 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | 0.0627 | 19.45 | 9.91 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | 0.0627 | 19.41 | 11.89 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | 0.0628 | 19.50 | 8.95 |

## All primary raw-return paired comparisons

### Broader penalty choices

Expanded minus original penalty menu, with every other choice matched.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge: Characteristics; full; standard; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0003, 0.0000] | 0.343 | 0.561 | 1.000 |
| Lasso: Characteristics; full; standard; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; standard; penalties expanded | Expanded minus original penalties | -0.0003 | [-0.0007, 0.0002] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Expanded minus original penalties | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties expanded | Expanded minus original penalties | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Expanded minus original penalties | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Expanded minus original penalties | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0006, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Expanded minus original penalties | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Expanded minus original penalties | -0.0001 | [-0.0003, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0004 | [-0.0007, -0.0001] | 0.760 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0003 | [-0.0006, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0003 | [-0.0008, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0003 | [-0.0007, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original penalties | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0003, -0.0001] | 0.079 | 0.162 | 0.269 |
| Lasso: Characteristics; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0004 | [-0.0008, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Expanded minus original penalties | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Expanded minus original penalties | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Expanded minus original penalties | 0.0002 | [-0.0001, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Expanded minus original penalties | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Expanded minus original penalties | -0.0003 | [-0.0005, 0.0000] | 0.910 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0001 | [-0.0003, 0.0000] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0003 | [-0.0007, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0002 | [-0.0005, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Expanded minus original penalties | -0.0003 | [-0.0008, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original penalties | -0.0003 | [-0.0007, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Expanded minus original penalties | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original penalties | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 |

### Daily characteristic ranks

Daily characteristic ranks minus historical standardization, with every other choice matched.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Characteristics; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; daily_rank; penalties expanded | Daily ranks minus standardization | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties original | Daily ranks minus standardization | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Daily ranks minus standardization | -0.0001 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Daily ranks minus standardization | 0.0001 | [0.0000, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Daily ranks minus standardization | 0.0000 | [0.0000, 0.0000] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Daily ranks minus standardization | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Daily ranks minus standardization | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Daily ranks minus standardization | 0.0003 | [0.0000, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Daily ranks minus standardization | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 |

### Allowing smaller text summaries

PCA menu including 4 and 8 components minus original PCA menu, with every other choice matched.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0002 | [0.0000, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0003 | [0.0001, 0.0006] | 0.196 | 0.531 | 0.683 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0000 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0004 | [0.0001, 0.0006] | 0.069 | 0.248 | 0.587 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0005 | [0.0002, 0.0008] | 0.053 | 0.195 | 0.193 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Expanded minus original PCA menu | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0002 | [-0.0001, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Expanded minus original PCA menu | -0.0001 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Expanded minus original PCA menu | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 |

### Compressing text

PCA text minus full text, matching estimator, transformation and penalty menu.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA minus full text | 0.0040 | [0.0029, 0.0052] | <0.001 | <0.001 | <0.001 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA minus full text | 0.0042 | [0.0030, 0.0054] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA minus full text | 0.0020 | [0.0013, 0.0026] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA minus full text | 0.0021 | [0.0014, 0.0028] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA minus full text | 0.0017 | [0.0011, 0.0023] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA minus full text | 0.0018 | [0.0011, 0.0024] | <0.001 | <0.001 | <0.001 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA minus full text | 0.0001 | [-0.0003, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA minus full text | 0.0003 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA minus full text | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA minus full text | 0.0003 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA minus full text | 0.0005 | [0.0000, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA minus full text | 0.0006 | [0.0001, 0.0011] | 0.401 | 0.362 | 0.761 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA minus full text | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA minus full text | 0.0003 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA minus full text | 0.0003 | [0.0000, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA minus full text | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA minus full text | 0.0003 | [0.0000, 0.0006] | 0.896 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA minus full text | 0.0003 | [0.0000, 0.0006] | 0.956 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA minus full text | 0.0040 | [0.0029, 0.0051] | <0.001 | <0.001 | <0.001 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA minus full text | 0.0042 | [0.0030, 0.0054] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA minus full text | 0.0020 | [0.0013, 0.0026] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA minus full text | 0.0021 | [0.0014, 0.0028] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA minus full text | 0.0017 | [0.0012, 0.0023] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA minus full text | 0.0018 | [0.0011, 0.0025] | <0.001 | <0.001 | <0.001 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA minus full text | 0.0002 | [-0.0002, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA minus full text | 0.0005 | [0.0001, 0.0010] | 0.765 | 0.742 | 0.721 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA minus full text | 0.0000 | [-0.0004, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA minus full text | 0.0005 | [0.0000, 0.0010] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA minus full text | 0.0005 | [0.0000, 0.0010] | 1.000 | 0.927 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA minus full text | 0.0006 | [0.0002, 0.0011] | 0.348 | 0.289 | 0.644 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA minus full text | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA minus full text | 0.0003 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA minus full text | 0.0001 | [-0.0001, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA minus full text | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA minus full text | 0.0004 | [0.0001, 0.0006] | 0.392 | 0.897 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA minus full text | 0.0003 | [0.0001, 0.0006] | 0.332 | 0.429 | 0.893 |

### Adding social-media inputs

Sentiment/attention given characteristics; full or PCA text given characteristics and sentiment/attention.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Characteristics + sentiment/attention; full; standard; penalties original | Sentiment/attention given characteristics | 0.0010 | [0.0006, 0.0014] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties original | Sentiment/attention given characteristics | 0.0010 | [0.0006, 0.0015] | 0.001 | 0.002 | <0.001 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Sentiment/attention given characteristics | 0.0011 | [0.0006, 0.0016] | <0.001 | 0.002 | <0.001 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties original | Sentiment/attention given characteristics | 0.0011 | [0.0005, 0.0016] | 0.008 | 0.007 | 0.029 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties expanded | Sentiment/attention given characteristics | 0.0011 | [0.0005, 0.0016] | 0.004 | 0.004 | 0.015 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties original | Sentiment/attention given characteristics | 0.0011 | [0.0006, 0.0017] | 0.003 | 0.002 | 0.009 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties expanded | Sentiment/attention given characteristics | 0.0014 | [0.0009, 0.0018] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties original | Grouped sentiment/attention + characteristics minus ridge characteristics | 0.0010 | [0.0005, 0.0014] | <0.001 | 0.002 | <0.001 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Grouped sentiment/attention + characteristics minus ridge characteristics | 0.0011 | [0.0006, 0.0015] | <0.001 | 0.003 | 0.002 |
| OLS: Characteristics + sentiment/attention + text; full; standard; penalties original | Full text given characteristics + sentiment/attention | -0.0041 | [-0.0054, -0.0028] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | Full text given characteristics + sentiment/attention | -0.0021 | [-0.0029, -0.0013] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Full text given characteristics + sentiment/attention | -0.0018 | [-0.0026, -0.0011] | <0.001 | <0.001 | <0.001 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties original | Full text given characteristics + sentiment/attention | -0.0001 | [-0.0008, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Full text given characteristics + sentiment/attention | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties original | Full text given characteristics + sentiment/attention | -0.0003 | [-0.0008, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Full text given characteristics + sentiment/attention | -0.0003 | [-0.0007, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | Full text given characteristics + sentiment/attention | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Full text given characteristics + sentiment/attention | -0.0002 | [-0.0005, 0.0000] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0005, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0002 | [-0.0003, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0002 | [-0.0007, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0002 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention; full; daily_rank; penalties original | Sentiment/attention given characteristics | 0.0010 | [0.0006, 0.0014] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Sentiment/attention given characteristics | 0.0010 | [0.0005, 0.0014] | 0.003 | 0.004 | 0.001 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Sentiment/attention given characteristics | 0.0011 | [0.0006, 0.0016] | <0.001 | 0.002 | <0.001 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties original | Sentiment/attention given characteristics | 0.0010 | [0.0004, 0.0015] | 0.028 | 0.025 | 0.066 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Sentiment/attention given characteristics | 0.0011 | [0.0005, 0.0016] | 0.006 | 0.004 | 0.014 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties original | Sentiment/attention given characteristics | 0.0011 | [0.0005, 0.0016] | 0.008 | 0.007 | 0.025 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Sentiment/attention given characteristics | 0.0014 | [0.0010, 0.0019] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Grouped sentiment/attention + characteristics minus ridge characteristics | 0.0009 | [0.0005, 0.0013] | 0.001 | 0.003 | 0.001 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Grouped sentiment/attention + characteristics minus ridge characteristics | 0.0011 | [0.0006, 0.0015] | <0.001 | <0.001 | <0.001 |
| OLS: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Full text given characteristics + sentiment/attention | -0.0041 | [-0.0054, -0.0028] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Full text given characteristics + sentiment/attention | -0.0021 | [-0.0029, -0.0013] | <0.001 | <0.001 | <0.001 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Full text given characteristics + sentiment/attention | -0.0019 | [-0.0026, -0.0011] | <0.001 | <0.001 | <0.001 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Full text given characteristics + sentiment/attention | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Full text given characteristics + sentiment/attention | -0.0001 | [-0.0007, 0.0004] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Full text given characteristics + sentiment/attention | -0.0003 | [-0.0008, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Full text given characteristics + sentiment/attention | -0.0003 | [-0.0007, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Full text given characteristics + sentiment/attention | -0.0001 | [-0.0004, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Full text given characteristics + sentiment/attention | -0.0003 | [-0.0006, 0.0000] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0004, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0001 | [-0.0006, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0002 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | -0.0002 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | PCA text given characteristics + sentiment/attention | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | PCA text given characteristics + sentiment/attention | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 |

### Separate penalties for input groups

Separate group penalties minus global ridge, with all input and search-menu choices matched.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties original | Group penalties minus global ridge | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Group penalties minus global ridge | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | Group penalties minus global ridge | 0.0019 | [0.0011, 0.0026] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Group penalties minus global ridge | 0.0015 | [0.0008, 0.0022] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | Group penalties minus global ridge | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Group penalties minus global ridge | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Group penalties minus global ridge | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Group penalties minus global ridge | 0.0001 | [-0.0002, 0.0003] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Group penalties minus global ridge | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Group penalties minus global ridge | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Group penalties minus global ridge | 0.0020 | [0.0012, 0.0027] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Group penalties minus global ridge | 0.0015 | [0.0008, 0.0022] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Group penalties minus global ridge | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Group penalties minus global ridge | -0.0001 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Group penalties minus global ridge | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Group penalties minus global ridge | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 |

### Penalization versus OLS

Each penalized procedure minus OLS; OLS has no penalty-menu axis.

| Procedure | Comparison | Mean difference | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge: Characteristics; full; standard; penalties original | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; standard; penalties expanded | Penalized minus OLS | 0.0008 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; standard; penalties original | Penalized minus OLS | 0.0005 | [-0.0003, 0.0012] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; standard; penalties expanded | Penalized minus OLS | 0.0003 | [-0.0006, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; standard; penalties original | Penalized minus OLS | 0.0003 | [-0.0006, 0.0011] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; standard; penalties expanded | Penalized minus OLS | 0.0000 | [-0.0009, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties original | Penalized minus OLS | 0.0011 | [0.0002, 0.0020] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties original | Penalized minus OLS | 0.0005 | [-0.0003, 0.0014] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; standard; penalties expanded | Penalized minus OLS | 0.0003 | [-0.0006, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties original | Penalized minus OLS | 0.0004 | [-0.0005, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; standard; penalties expanded | Penalized minus OLS | 0.0004 | [-0.0006, 0.0014] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties original | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; standard; penalties expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | Penalized minus OLS | 0.0031 | [0.0016, 0.0046] | 0.003 | 0.007 | 0.092 |
| Ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Penalized minus OLS | 0.0033 | [0.0018, 0.0048] | 0.001 | 0.004 | 0.057 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties original | Penalized minus OLS | 0.0045 | [0.0028, 0.0063] | <0.001 | <0.001 | 0.002 |
| Lasso: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Penalized minus OLS | 0.0044 | [0.0026, 0.0061] | <0.001 | <0.001 | 0.002 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties original | Penalized minus OLS | 0.0043 | [0.0025, 0.0060] | <0.001 | <0.001 | 0.009 |
| Elastic net: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Penalized minus OLS | 0.0043 | [0.0025, 0.0060] | <0.001 | <0.001 | 0.004 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties original | Penalized minus OLS | 0.0049 | [0.0032, 0.0067] | <0.001 | <0.001 | <0.001 |
| Group ridge: Characteristics + sentiment/attention + text; full; standard; penalties expanded | Penalized minus OLS | 0.0048 | [0.0031, 0.0065] | <0.001 | <0.001 | 0.001 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | Penalized minus OLS | 0.0011 | [0.0001, 0.0020] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Penalized minus OLS | 0.0010 | [0.0001, 0.0018] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Penalized minus OLS | 0.0010 | [0.0000, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | Penalized minus OLS | 0.0007 | [-0.0002, 0.0016] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Penalized minus OLS | 0.0007 | [-0.0002, 0.0015] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Penalized minus OLS | 0.0003 | [-0.0007, 0.0013] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0004 | [-0.0005, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | Penalized minus OLS | 0.0007 | [-0.0002, 0.0017] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Penalized minus OLS | 0.0007 | [-0.0002, 0.0016] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Penalized minus OLS | 0.0004 | [-0.0006, 0.0015] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0004 | [-0.0006, 0.0014] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu original | Penalized minus OLS | 0.0012 | [0.0002, 0.0022] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties original; PCA menu expanded | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu original | Penalized minus OLS | 0.0011 | [0.0002, 0.0021] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; standard; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; daily_rank; penalties original | Penalized minus OLS | 0.0010 | [0.0002, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; daily_rank; penalties original | Penalized minus OLS | 0.0005 | [-0.0003, 0.0013] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0003 | [-0.0006, 0.0011] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; daily_rank; penalties original | Penalized minus OLS | 0.0003 | [-0.0005, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0000 | [-0.0010, 0.0009] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties original | Penalized minus OLS | 0.0005 | [-0.0004, 0.0013] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0003 | [-0.0006, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties original | Penalized minus OLS | 0.0004 | [-0.0005, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0004 | [-0.0005, 0.0014] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties original | Penalized minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Penalized minus OLS | 0.0031 | [0.0016, 0.0045] | 0.003 | 0.007 | 0.100 |
| Ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0032 | [0.0017, 0.0047] | 0.002 | 0.005 | 0.072 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Penalized minus OLS | 0.0045 | [0.0028, 0.0062] | <0.001 | <0.001 | 0.002 |
| Lasso: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0043 | [0.0025, 0.0061] | <0.001 | <0.001 | 0.002 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Penalized minus OLS | 0.0042 | [0.0025, 0.0060] | <0.001 | <0.001 | 0.009 |
| Elastic net: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0043 | [0.0026, 0.0061] | <0.001 | <0.001 | 0.003 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties original | Penalized minus OLS | 0.0050 | [0.0033, 0.0068] | <0.001 | <0.001 | 0.001 |
| Group ridge: Characteristics + sentiment/attention + text; full; daily_rank; penalties expanded | Penalized minus OLS | 0.0048 | [0.0031, 0.0065] | <0.001 | <0.001 | 0.002 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Penalized minus OLS | 0.0010 | [0.0001, 0.0020] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Penalized minus OLS | 0.0010 | [0.0001, 0.0018] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Penalized minus OLS | 0.0010 | [0.0000, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0008 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Penalized minus OLS | 0.0006 | [-0.0003, 0.0015] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Penalized minus OLS | 0.0008 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Penalized minus OLS | 0.0004 | [-0.0007, 0.0014] | 1.000 | 1.000 | 1.000 |
| Lasso: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0006 | [-0.0003, 0.0016] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Penalized minus OLS | 0.0007 | [-0.0002, 0.0017] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Penalized minus OLS | 0.0007 | [-0.0002, 0.0015] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Penalized minus OLS | 0.0004 | [-0.0006, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0004 | [-0.0006, 0.0013] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu original | Penalized minus OLS | 0.0012 | [0.0002, 0.0021] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties original; PCA menu expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu original | Penalized minus OLS | 0.0011 | [0.0002, 0.0021] | 1.000 | 1.000 | 1.000 |
| Group ridge: Characteristics + sentiment/attention + text; pca; daily_rank; penalties expanded; PCA menu expanded | Penalized minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |

## Stability across the two fixed subperiods

These comparisons are descriptive checks in 2014–2018 and 2019–2022. Subperiods do not choose a model or a trading regime. Complete numerical results for every family, target, metric and HAC lag remain in the contrast artifact.

### 2014-2018

| Change | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Broader penalty choices | 38 | 8 | 0 | 0 |
| Daily characteristic ranks | 43 | 25 | 0 | 0 |
| Allowing smaller text summaries | 18 | 14 | 0 | 0 |
| Compressing text | 36 | 33 | 12 | 0 |
| Adding social-media inputs | 72 | 35 | 2 | 6 |
| Separate penalties for input groups | 16 | 11 | 2 | 0 |
| Penalization versus OLS | 76 | 61 | 16 | 0 |

### 2019-2022

| Change | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Broader penalty choices | 38 | 8 | 0 | 0 |
| Daily characteristic ranks | 43 | 19 | 0 | 0 |
| Allowing smaller text summaries | 18 | 18 | 0 | 0 |
| Compressing text | 36 | 32 | 12 | 0 |
| Adding social-media inputs | 72 | 43 | 12 | 5 |
| Separate penalties for input groups | 16 | 10 | 4 | 0 |
| Penalization versus OLS | 76 | 76 | 0 | 0 |

## Validation selection stability

These diagnostics reuse fitted validation predictions while omitting one validation month from the scoring criterion at a time. They do not refit on omitted-month training samples, use test outcomes, or change the selected forecasting procedure. Identical or nearly tied candidates can produce unstable labels even when predictions barely change.

| Diagnostic | Defined selections | Median | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| stability_diagnostic_only | 18576 | 1.0000 | 1.0000 | 1.0000 |
| stability_full_candidate_id | 38 | 71452359834370383872.0000 | 71452359834370383872.0000 | 71452359834370383872.0000 |
| stability_month_count | 18576 | 7.0000 | 6.0000 | 7.0000 |
| stability_runner_up_gap | 17280 | 0.0001 | 0.0000 | 0.0060 |
| stability_same_winner_fraction | 18576 | 0.6667 | 0.0000 | 1.0000 |
| stability_distinct_winners | 18576 | 2.0000 | 1.0000 | 7.0000 |

## Interpretation limits

The 2014–2022 period has already informed project decisions, so these are development results. Chronological training prevents direct future-label leakage but does not make this an untouched confirmation sample. The tables do not select a winner by its largest test score. No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim.

Market-based controls and past returns do not establish incremental information beyond accounting fundamentals or news. The inherited DGTW target remains secondary because upstream accounting availability has not been resolved. A failure to detect text gains does not establish that text contains no predictive information. No neural networks are fitted in this study.

## Reproducible artifacts

The same economic-key alignment, scoring, fractional tie handling and calendar-aware HAC functions used in earlier reports are reused unchanged. SHA-256 hashes cover the registry, prepared artifacts, predictions, kernels and evaluation outputs. The large daily table is published as deterministic gzip.

- [summary](<data/linear_optimization_v3.csv>)
- [_yearly](<data/linear_optimization_v3_yearly.csv>)
- [_periods](<data/linear_optimization_v3_periods.csv>)
- [_contrasts](<data/linear_optimization_v3_contrasts.csv>)
- [_coverage](<data/linear_optimization_v3_coverage.csv>)
- [_deciles](<data/linear_optimization_v3_deciles.csv>)
- [_subgroups](<data/linear_optimization_v3_subgroups.csv>)
- [_horizons](<data/linear_optimization_v3_horizons.csv>)
- [_daily](<data/linear_optimization_v3_daily.csv.gz>)

- [Evaluation metadata and hashes](<data/linear_optimization_v3.json>)

- [Model registry](<../.runs/linear_optimization_v3/study/ed854a1f5bf3da39/linear/full_b217f4648f5fe89a/registry.json>)
