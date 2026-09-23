# Explicit interactions: linear prediction with nonlinear input terms

All 148 registered procedures completed evaluation: 74 per target.

This full register preserves all planned procedures and comparisons. [The short companion](<13a_linear_interaction_summary.md>) answers the main research questions.

## What was fitted

| Basis | Meaning | Columns |
| --- | --- | --- |
| additive_c | C main effects | 34 |
| additive_cs | Additive C + S | 36 |
| additive_cst_full | Additive C + S + full text | 422 |
| c_squares | C plus C squares | 51 |
| c_quadratic | C quadratic | 187 |
| cq_s | C quadratic + S main effects | 189 |
| cq_s_squares | C quadratic + S main effects/squares | 191 |
| cs_joint | Joint C + S interactions | 226 |
| text_main | Joint C + S plus text main effects | 244 |
| text_squares | Above plus text squares | 262 |
| agreement_c | Above plus agreement x C | 296 |
| text_c | Above plus embedding PCs x C | 568 |
| text_s | Above plus text x S | 604 |
| single_social192 | 191-column baseline plus one of 35 named products; ridge only | 192 |

C denotes 17 continuous market/past-return characteristics plus 17 binary missingness flags. S is net sentiment and attention (log message volume). T denotes 16 embedding principal components plus two agreement inputs (embed_norm and embed_cos). Additive full text instead uses all 384 embedding coordinates plus those agreement inputs. OLS, ridge and elastic net cover all 13 joint/reference bases. Another 35 ridge models each add one named social interaction to the same 191-column baseline: 39 + 35 = 74 procedures per target, 148 total.

The 35 named terms are sentiment and attention each multiplied by every continuous characteristic (34 terms), plus sentiment times attention. They are fitted individually and jointly because these answer different conditional prediction questions. All parent main-effect columns remain included. Ordinary elastic net does not enforce nonzero-parent hierarchy.

## Timing, transformations and sample

| Target | Models | Eligible stock-days | Common predictions | Observed outcomes |
| --- | --- | --- | --- | --- |
| raw | 74 | 3034035 | 3034035 | 3033080 |
| dgtw | 74 | 3034035 | 3034035 | 2748978 |

Each monthly forecast uses 504 initial training sessions and 126 chronological validation sessions. Settings are chosen by mean daily validation Spearman IC. Final coefficients are refitted on the exact original training and validation rows, preserving the maturity purge: 630 sessions, about 2.5 years. Inputs through close t predict close t to close t+1. Daily return-rank targets, squared loss and equal total fitting weight per date are unchanged.

The continuous characteristics and social scalars already use daily ranks from the certified original cache. This study does not compare raw characteristic levels with ranks and does not re-rank previously imputed zeros. Missing characteristic ranks retain neutral zero with separate binary flags. Squares/products are constructed from the declared representation and receive fitting-only scaling; they are not ranked again. Flags remain main effects, with no flag products or duplicate flag squares. PCA is fitted only on permitted fitting inputs, fixed at 16 components, and rebuilt for the union refit.

All procedures within a target are scored on the joint finite-prediction intersection before filtering missing outcomes. The sample consists of stock-days with retained tagged StockTwits messages and embeddings, not the full stock universe. Characteristic-only forecasts use the same conditional sample. Market histories supply past-return controls without adding no-message days.

## Interpretation and inference

Ridge is the primary interaction estimator; OLS and elastic net are references/sensitivities. Rank IC is mean daily Spearman correlation. Paired daily differences use calendar-aware HAC5, with HAC21/HAC63 sensitivities. Confidence intervals are pointwise 95% intervals. Bonferroni correction uses the full declared family separately within each target, metric and period. Missing or undefined statistics never reduce a denominator. There is no across-family significance guarantee or correction for the project's previously inspected development history.

| Family | Comparisons per target/metric/period |
| --- | --- |
| Estimator comparisons on the same basis | 39 |
| Added sentiment/attention information | 12 |
| Added text information | 18 |
| Nonlinearity in stock characteristics | 9 |
| Social curvature and joint interactions | 9 |
| Comparison with additive reference procedures | 18 |
| Text curvature and interaction blocks | 15 |
| All 35 individual social interactions | 35 |

Some aggregate and sequential comparisons overlap because they answer distinct conditional questions. Their counts are not independent replications. A predictive improvement or a coefficient sign does not establish a causal mechanism. All 35 individual terms are reported with the same 35-comparison adjustment, rather than displaying selected favorable terms.

The agreement-by-characteristic step adds 34 products. The next step adds 272 embedding-PC-by-characteristic products, separating that block from agreement. The final step adds 36 text-by-social products. PCA coordinates may rotate across months; their coefficients do not identify stable topics. Additive-full-text reference comparisons change both text representation and the characteristic/social basis, so they cannot attribute any difference solely to interactions.

## Full-period raw-return comparison overview

| Question | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Nonlinearity in stock characteristics | 9 | 5 | 2 | 1 |
| Added sentiment/attention information | 12 | 12 | 12 | 0 |
| Social curvature and joint interactions | 9 | 7 | 0 | 0 |
| All 35 individual social interactions | 35 | 18 | 0 | 0 |
| Added text information | 18 | 0 | 0 | 5 |
| Text curvature and interaction blocks | 15 | 3 | 0 | 3 |
| Estimator comparisons on the same basis | 39 | 27 | 2 | 0 |
| Comparison with additive reference procedures | 18 | 17 | 3 | 0 |

## Model levels

Rows follow the registry, not realized performance. Portfolio diagnostics are gross daily top-minus-bottom decile returns in basis points with fractional tie handling. They do not include attainable execution, turnover, transaction costs or factor alpha.

### Raw returns (primary)

| Procedure | Rank IC | EW spread (bp) | Cap-weight spread (bp) |
| --- | --- | --- | --- |
| OLS: C main effects | 0.0684 | 23.12 | 12.51 |
| Ridge: C main effects | 0.0694 | 20.22 | 15.59 |
| Elastic net: C main effects | 0.0687 | 20.35 | 15.33 |
| OLS: Additive C + S | 0.0694 | 25.10 | 10.00 |
| Ridge: Additive C + S | 0.0704 | 24.47 | 14.70 |
| Elastic net: Additive C + S | 0.0698 | 23.86 | 15.96 |
| OLS: Additive C + S + full text | 0.0652 | 25.54 | 7.21 |
| Ridge: Additive C + S + full text | 0.0683 | 23.10 | 11.42 |
| Elastic net: Additive C + S + full text | 0.0695 | 23.48 | 10.77 |
| OLS: C plus C squares | 0.0671 | 25.76 | 20.38 |
| Ridge: C plus C squares | 0.0672 | 20.46 | 15.37 |
| Elastic net: C plus C squares | 0.0672 | 21.13 | 18.78 |
| OLS: C quadratic | 0.0694 | 35.18 | 29.70 |
| Ridge: C quadratic | 0.0691 | 32.36 | 32.18 |
| Elastic net: C quadratic | 0.0692 | 29.13 | 30.53 |
| OLS: C quadratic + S main effects | 0.0701 | 36.88 | 28.65 |
| Ridge: C quadratic + S main effects | 0.0701 | 33.38 | 32.47 |
| Elastic net: C quadratic + S main effects | 0.0700 | 30.46 | 30.66 |
| OLS: C quadratic + S main effects/squares | 0.0703 | 37.17 | 29.61 |
| Ridge: C quadratic + S main effects/squares | 0.0703 | 34.54 | 32.79 |
| Elastic net: C quadratic + S main effects/squares | 0.0701 | 30.67 | 32.77 |
| OLS: Joint C + S interactions | 0.0702 | 39.22 | 28.54 |
| Ridge: Joint C + S interactions | 0.0702 | 36.97 | 33.68 |
| Elastic net: Joint C + S interactions | 0.0703 | 32.20 | 29.78 |
| OLS: Joint C + S plus text main effects | 0.0701 | 40.08 | 30.99 |
| Ridge: Joint C + S plus text main effects | 0.0700 | 36.53 | 35.15 |
| Elastic net: Joint C + S plus text main effects | 0.0701 | 32.18 | 30.34 |
| OLS: Above plus text squares | 0.0698 | 40.30 | 29.74 |
| Ridge: Above plus text squares | 0.0697 | 36.20 | 34.75 |
| Elastic net: Above plus text squares | 0.0699 | 31.87 | 31.60 |
| OLS: Above plus agreement x C | 0.0693 | 41.29 | 28.46 |
| Ridge: Above plus agreement x C | 0.0701 | 34.66 | 30.05 |
| Elastic net: Above plus agreement x C | 0.0698 | 32.57 | 31.97 |
| OLS: Above plus embedding PCs x C | 0.0674 | 39.62 | 26.42 |
| Ridge: Above plus embedding PCs x C | 0.0694 | 32.37 | 30.76 |
| Elastic net: Above plus embedding PCs x C | 0.0699 | 30.71 | 30.19 |
| OLS: Above plus text x S | 0.0673 | 39.68 | 26.59 |
| Ridge: Above plus text x S | 0.0693 | 32.23 | 29.09 |
| Elastic net: Above plus text x S | 0.0698 | 31.26 | 29.47 |
| Ridge: One named social interaction: Sentiment x ret1 | 0.0703 | 34.38 | 33.43 |
| Ridge: One named social interaction: Sentiment x ret5 | 0.0703 | 34.23 | 31.41 |
| Ridge: One named social interaction: Sentiment x ret21 | 0.0704 | 34.42 | 32.37 |
| Ridge: One named social interaction: Sentiment x ret63 | 0.0703 | 34.20 | 33.14 |
| Ridge: One named social interaction: Sentiment x ret126 | 0.0703 | 34.34 | 32.38 |
| Ridge: One named social interaction: Sentiment x ret252 | 0.0702 | 34.28 | 33.35 |
| Ridge: One named social interaction: Sentiment x momentum252_skip21 | 0.0703 | 34.24 | 32.87 |
| Ridge: One named social interaction: Sentiment x log_market_cap | 0.0704 | 34.86 | 32.81 |
| Ridge: One named social interaction: Sentiment x log_price | 0.0704 | 34.99 | 31.36 |
| Ridge: One named social interaction: Sentiment x turnover1 | 0.0702 | 34.23 | 32.49 |
| Ridge: One named social interaction: Sentiment x turnover21 | 0.0703 | 34.57 | 32.77 |
| Ridge: One named social interaction: Sentiment x log_dollar_volume1 | 0.0706 | 35.00 | 33.31 |
| Ridge: One named social interaction: Sentiment x log_dollar_volume21 | 0.0705 | 35.60 | 34.60 |
| Ridge: One named social interaction: Sentiment x volatility21 | 0.0702 | 34.45 | 32.40 |
| Ridge: One named social interaction: Sentiment x volatility63 | 0.0702 | 34.23 | 33.17 |
| Ridge: One named social interaction: Sentiment x max_return21 | 0.0702 | 34.38 | 32.47 |
| Ridge: One named social interaction: Sentiment x amihud21 | 0.0705 | 35.15 | 31.70 |
| Ridge: One named social interaction: Attention x ret1 | 0.0703 | 34.10 | 33.26 |
| Ridge: One named social interaction: Attention x ret5 | 0.0702 | 34.49 | 34.06 |
| Ridge: One named social interaction: Attention x ret21 | 0.0702 | 34.46 | 31.59 |
| Ridge: One named social interaction: Attention x ret63 | 0.0702 | 33.84 | 34.13 |
| Ridge: One named social interaction: Attention x ret126 | 0.0703 | 34.11 | 32.88 |
| Ridge: One named social interaction: Attention x ret252 | 0.0702 | 34.44 | 33.84 |
| Ridge: One named social interaction: Attention x momentum252_skip21 | 0.0703 | 34.59 | 32.63 |
| Ridge: One named social interaction: Attention x log_market_cap | 0.0702 | 34.62 | 34.04 |
| Ridge: One named social interaction: Attention x log_price | 0.0703 | 34.83 | 31.73 |
| Ridge: One named social interaction: Attention x turnover1 | 0.0703 | 33.98 | 31.66 |
| Ridge: One named social interaction: Attention x turnover21 | 0.0703 | 34.08 | 32.40 |
| Ridge: One named social interaction: Attention x log_dollar_volume1 | 0.0703 | 34.38 | 33.55 |
| Ridge: One named social interaction: Attention x log_dollar_volume21 | 0.0703 | 34.74 | 32.36 |
| Ridge: One named social interaction: Attention x volatility21 | 0.0703 | 34.72 | 33.60 |
| Ridge: One named social interaction: Attention x volatility63 | 0.0703 | 34.34 | 32.91 |
| Ridge: One named social interaction: Attention x max_return21 | 0.0703 | 35.05 | 33.51 |
| Ridge: One named social interaction: Attention x amihud21 | 0.0703 | 34.92 | 31.18 |
| Ridge: One named social interaction: Sentiment x Attention | 0.0703 | 34.43 | 32.35 |

### DGTW returns (secondary)

| Procedure | Rank IC | EW spread (bp) | Cap-weight spread (bp) |
| --- | --- | --- | --- |
| OLS: C main effects | 0.0602 | 18.25 | 10.35 |
| Ridge: C main effects | 0.0610 | 14.92 | 11.27 |
| Elastic net: C main effects | 0.0609 | 16.20 | 10.37 |
| OLS: Additive C + S | 0.0615 | 20.62 | 13.88 |
| Ridge: Additive C + S | 0.0627 | 18.66 | 11.41 |
| Elastic net: Additive C + S | 0.0625 | 19.01 | 10.58 |
| OLS: Additive C + S + full text | 0.0574 | 22.28 | 9.19 |
| Ridge: Additive C + S + full text | 0.0608 | 20.99 | 12.06 |
| Elastic net: Additive C + S + full text | 0.0621 | 18.71 | 9.79 |
| OLS: C plus C squares | 0.0592 | 20.10 | 19.41 |
| Ridge: C plus C squares | 0.0600 | 17.21 | 14.18 |
| Elastic net: C plus C squares | 0.0605 | 17.21 | 13.41 |
| OLS: C quadratic | 0.0624 | 33.35 | 27.16 |
| Ridge: C quadratic | 0.0626 | 32.13 | 32.29 |
| Elastic net: C quadratic | 0.0626 | 26.59 | 26.66 |
| OLS: C quadratic + S main effects | 0.0634 | 34.56 | 27.89 |
| Ridge: C quadratic + S main effects | 0.0637 | 32.40 | 29.46 |
| Elastic net: C quadratic + S main effects | 0.0638 | 28.52 | 27.47 |
| OLS: C quadratic + S main effects/squares | 0.0637 | 35.61 | 28.73 |
| Ridge: C quadratic + S main effects/squares | 0.0639 | 33.46 | 30.20 |
| Elastic net: C quadratic + S main effects/squares | 0.0642 | 29.23 | 24.80 |
| OLS: Joint C + S interactions | 0.0635 | 37.43 | 27.03 |
| Ridge: Joint C + S interactions | 0.0639 | 35.50 | 25.54 |
| Elastic net: Joint C + S interactions | 0.0641 | 31.14 | 22.51 |
| OLS: Joint C + S plus text main effects | 0.0635 | 37.94 | 26.76 |
| Ridge: Joint C + S plus text main effects | 0.0639 | 35.76 | 25.99 |
| Elastic net: Joint C + S plus text main effects | 0.0642 | 32.25 | 24.37 |
| OLS: Above plus text squares | 0.0631 | 38.34 | 29.62 |
| Ridge: Above plus text squares | 0.0636 | 35.23 | 26.18 |
| Elastic net: Above plus text squares | 0.0639 | 31.36 | 25.13 |
| OLS: Above plus agreement x C | 0.0632 | 39.05 | 27.52 |
| Ridge: Above plus agreement x C | 0.0636 | 34.23 | 24.58 |
| Elastic net: Above plus agreement x C | 0.0639 | 32.20 | 22.37 |
| OLS: Above plus embedding PCs x C | 0.0611 | 37.54 | 20.07 |
| Ridge: Above plus embedding PCs x C | 0.0620 | 28.91 | 19.44 |
| Elastic net: Above plus embedding PCs x C | 0.0632 | 30.38 | 20.18 |
| OLS: Above plus text x S | 0.0610 | 37.42 | 19.54 |
| Ridge: Above plus text x S | 0.0622 | 29.00 | 21.24 |
| Elastic net: Above plus text x S | 0.0633 | 29.50 | 20.18 |
| Ridge: One named social interaction: Sentiment x ret1 | 0.0639 | 33.34 | 29.90 |
| Ridge: One named social interaction: Sentiment x ret5 | 0.0641 | 34.15 | 29.85 |
| Ridge: One named social interaction: Sentiment x ret21 | 0.0640 | 34.20 | 30.34 |
| Ridge: One named social interaction: Sentiment x ret63 | 0.0640 | 33.89 | 30.40 |
| Ridge: One named social interaction: Sentiment x ret126 | 0.0639 | 33.90 | 30.28 |
| Ridge: One named social interaction: Sentiment x ret252 | 0.0640 | 34.05 | 30.42 |
| Ridge: One named social interaction: Sentiment x momentum252_skip21 | 0.0640 | 33.51 | 30.14 |
| Ridge: One named social interaction: Sentiment x log_market_cap | 0.0642 | 34.89 | 30.39 |
| Ridge: One named social interaction: Sentiment x log_price | 0.0641 | 34.91 | 31.78 |
| Ridge: One named social interaction: Sentiment x turnover1 | 0.0639 | 33.50 | 29.31 |
| Ridge: One named social interaction: Sentiment x turnover21 | 0.0639 | 33.47 | 30.38 |
| Ridge: One named social interaction: Sentiment x log_dollar_volume1 | 0.0641 | 34.29 | 30.56 |
| Ridge: One named social interaction: Sentiment x log_dollar_volume21 | 0.0641 | 34.38 | 30.49 |
| Ridge: One named social interaction: Sentiment x volatility21 | 0.0639 | 33.62 | 29.59 |
| Ridge: One named social interaction: Sentiment x volatility63 | 0.0639 | 34.10 | 29.78 |
| Ridge: One named social interaction: Sentiment x max_return21 | 0.0639 | 33.58 | 29.66 |
| Ridge: One named social interaction: Sentiment x amihud21 | 0.0641 | 34.73 | 29.47 |
| Ridge: One named social interaction: Attention x ret1 | 0.0640 | 34.13 | 31.73 |
| Ridge: One named social interaction: Attention x ret5 | 0.0639 | 33.42 | 29.42 |
| Ridge: One named social interaction: Attention x ret21 | 0.0639 | 33.64 | 30.46 |
| Ridge: One named social interaction: Attention x ret63 | 0.0639 | 33.92 | 30.10 |
| Ridge: One named social interaction: Attention x ret126 | 0.0639 | 33.77 | 30.21 |
| Ridge: One named social interaction: Attention x ret252 | 0.0639 | 33.86 | 29.55 |
| Ridge: One named social interaction: Attention x momentum252_skip21 | 0.0639 | 33.76 | 29.70 |
| Ridge: One named social interaction: Attention x log_market_cap | 0.0638 | 34.24 | 29.46 |
| Ridge: One named social interaction: Attention x log_price | 0.0638 | 33.95 | 30.63 |
| Ridge: One named social interaction: Attention x turnover1 | 0.0640 | 33.18 | 29.30 |
| Ridge: One named social interaction: Attention x turnover21 | 0.0640 | 33.69 | 29.34 |
| Ridge: One named social interaction: Attention x log_dollar_volume1 | 0.0638 | 33.83 | 29.54 |
| Ridge: One named social interaction: Attention x log_dollar_volume21 | 0.0638 | 34.09 | 30.65 |
| Ridge: One named social interaction: Attention x volatility21 | 0.0640 | 33.79 | 29.05 |
| Ridge: One named social interaction: Attention x volatility63 | 0.0640 | 33.77 | 29.26 |
| Ridge: One named social interaction: Attention x max_return21 | 0.0639 | 33.97 | 30.15 |
| Ridge: One named social interaction: Attention x amihud21 | 0.0639 | 34.37 | 29.73 |
| Ridge: One named social interaction: Sentiment x Attention | 0.0639 | 32.97 | 28.38 |

## All paired raw-return block comparisons

### Nonlinearity in stock characteristics

Characteristic squares versus main effects; quadratic versus squares; quadratic versus main effects.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: C plus C squares | C squares beyond C main effects | -0.0013 | [-0.0027, 0.0002] | 0.727 | 0.820 | 0.673 |
| Ridge: C plus C squares | C squares beyond C main effects | -0.0022 | [-0.0037, -0.0007] | 0.029 | 0.028 | 0.023 |
| Elastic net: C plus C squares | C squares beyond C main effects | -0.0015 | [-0.0028, -0.0001] | 0.269 | 0.289 | 0.403 |
| OLS: C quadratic | C pairs beyond C squares | 0.0023 | [0.0009, 0.0036] | 0.010 | 0.009 | 0.005 |
| OLS: C quadratic | Full C quadratic versus C main effects | 0.0010 | [-0.0010, 0.0030] | 1.000 | 1.000 | 1.000 |
| Ridge: C quadratic | C pairs beyond C squares | 0.0019 | [0.0005, 0.0034] | 0.089 | 0.104 | 0.111 |
| Ridge: C quadratic | Full C quadratic versus C main effects | -0.0003 | [-0.0024, 0.0018] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic | C pairs beyond C squares | 0.0020 | [0.0006, 0.0033] | 0.034 | 0.031 | 0.008 |
| Elastic net: C quadratic | Full C quadratic versus C main effects | 0.0005 | [-0.0013, 0.0023] | 1.000 | 1.000 | 1.000 |

### Added sentiment/attention information

Additive social main effects given additive C; social main effects, squares and joint products each given quadratic C.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Additive C + S | Social main effects given additive C | 0.0010 | [0.0006, 0.0014] | <0.001 | <0.001 | <0.001 |
| Ridge: Additive C + S | Social main effects given additive C | 0.0010 | [0.0006, 0.0015] | <0.001 | <0.001 | <0.001 |
| Elastic net: Additive C + S | Social main effects given additive C | 0.0011 | [0.0006, 0.0017] | <0.001 | <0.001 | 0.001 |
| OLS: C quadratic + S main effects | Social main effects given quadratic C | 0.0007 | [0.0004, 0.0010] | <0.001 | <0.001 | <0.001 |
| Ridge: C quadratic + S main effects | Social main effects given quadratic C | 0.0010 | [0.0006, 0.0014] | <0.001 | <0.001 | <0.001 |
| Elastic net: C quadratic + S main effects | Social main effects given quadratic C | 0.0009 | [0.0005, 0.0012] | <0.001 | <0.001 | <0.001 |
| OLS: C quadratic + S main effects/squares | Social main effects/squares given quadratic C | 0.0009 | [0.0005, 0.0013] | <0.001 | <0.001 | <0.001 |
| Ridge: C quadratic + S main effects/squares | Social main effects/squares given quadratic C | 0.0012 | [0.0007, 0.0016] | <0.001 | <0.001 | <0.001 |
| Elastic net: C quadratic + S main effects/squares | Social main effects/squares given quadratic C | 0.0010 | [0.0004, 0.0015] | 0.006 | 0.009 | 0.010 |
| OLS: Joint C + S interactions | Joint social model given quadratic C | 0.0008 | [0.0003, 0.0013] | 0.016 | 0.050 | 0.072 |
| Ridge: Joint C + S interactions | Joint social model given quadratic C | 0.0011 | [0.0006, 0.0016] | <0.001 | 0.002 | 0.006 |
| Elastic net: Joint C + S interactions | Joint social model given quadratic C | 0.0012 | [0.0006, 0.0018] | 0.002 | 0.003 | 0.005 |

### Social curvature and joint interactions

Social squares versus social main effects; joint social products versus social squares; joint social model versus social main effects.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: C quadratic + S main effects/squares | Social squares beyond social main effects | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: C quadratic + S main effects/squares | Social squares beyond social main effects | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic + S main effects/squares | Social squares beyond social main effects | 0.0001 | [-0.0003, 0.0005] | 1.000 | 1.000 | 1.000 |
| OLS: Joint C + S interactions | All 35 social products beyond social squares | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: Joint C + S interactions | Social squares/products beyond social main effects | 0.0001 | [-0.0004, 0.0005] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S interactions | All 35 social products beyond social squares | -0.0001 | [-0.0005, 0.0004] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S interactions | Social squares/products beyond social main effects | 0.0001 | [-0.0004, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S interactions | All 35 social products beyond social squares | 0.0002 | [-0.0003, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S interactions | Social squares/products beyond social main effects | 0.0003 | [-0.0002, 0.0008] | 1.000 | 1.000 | 1.000 |

### Added text information

Additive full text given additive C+S; each compressed-text extension given the 226-column C+S interaction benchmark.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Additive C + S + full text | Full text given additive C+S | -0.0041 | [-0.0054, -0.0028] | <0.001 | <0.001 | <0.001 |
| Ridge: Additive C + S + full text | Full text given additive C+S | -0.0021 | [-0.0029, -0.0013] | <0.001 | <0.001 | <0.001 |
| Elastic net: Additive C + S + full text | Full text given additive C+S | -0.0003 | [-0.0008, 0.0002] | 1.000 | 1.000 | 1.000 |
| OLS: Joint C + S plus text main effects | Joint C + S plus text main effects versus no text | -0.0001 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S plus text main effects | Joint C + S plus text main effects versus no text | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S plus text main effects | Joint C + S plus text main effects versus no text | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus text squares | Above plus text squares versus no text | -0.0004 | [-0.0007, -0.0001] | 0.345 | 0.294 | 0.219 |
| Ridge: Above plus text squares | Above plus text squares versus no text | -0.0005 | [-0.0009, -0.0001] | 0.220 | 0.259 | 0.310 |
| Elastic net: Above plus text squares | Above plus text squares versus no text | -0.0005 | [-0.0009, 0.0000] | 0.589 | 1.000 | 1.000 |
| OLS: Above plus agreement x C | Above plus agreement x C versus no text | -0.0009 | [-0.0014, -0.0003] | 0.019 | 0.040 | 0.102 |
| Ridge: Above plus agreement x C | Above plus agreement x C versus no text | -0.0001 | [-0.0008, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus agreement x C | Above plus agreement x C versus no text | -0.0005 | [-0.0010, 0.0000] | 0.576 | 0.854 | 1.000 |
| OLS: Above plus embedding PCs x C | Above plus embedding PCs x C versus no text | -0.0027 | [-0.0036, -0.0018] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus embedding PCs x C | Above plus embedding PCs x C versus no text | -0.0008 | [-0.0017, 0.0001] | 1.000 | 1.000 | 0.762 |
| Elastic net: Above plus embedding PCs x C | Above plus embedding PCs x C versus no text | -0.0004 | [-0.0012, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus text x S | Above plus text x S versus no text | -0.0028 | [-0.0037, -0.0019] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus text x S | Above plus text x S versus no text | -0.0009 | [-0.0019, 0.0000] | 0.982 | 0.970 | 0.609 |
| Elastic net: Above plus text x S | Above plus text x S versus no text | -0.0005 | [-0.0013, 0.0002] | 1.000 | 1.000 | 1.000 |

### Text curvature and interaction blocks

Successive text squares, agreement-by-C, embedding-PC-by-C, and text-by-S additions; combined text-by-C block versus text squares.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Above plus text squares | Text squares beyond text main effects | -0.0003 | [-0.0005, -0.0002] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus text squares | Text squares beyond text main effects | -0.0003 | [-0.0006, 0.0000] | 0.517 | 0.308 | 0.184 |
| Elastic net: Above plus text squares | Text squares beyond text main effects | -0.0003 | [-0.0006, 0.0000] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus agreement x C | Agreement x C beyond text squares | -0.0005 | [-0.0009, -0.0001] | 0.318 | 0.502 | 0.951 |
| Ridge: Above plus agreement x C | Agreement x C beyond text squares | 0.0004 | [-0.0003, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus agreement x C | Agreement x C beyond text squares | -0.0001 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus embedding PCs x C | Embedding PCs x C beyond agreement x C | -0.0019 | [-0.0025, -0.0012] | <0.001 | <0.001 | <0.001 |
| OLS: Above plus embedding PCs x C | All text x C beyond text squares | -0.0023 | [-0.0031, -0.0015] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus embedding PCs x C | Embedding PCs x C beyond agreement x C | -0.0008 | [-0.0013, -0.0002] | 0.120 | 0.081 | 0.060 |
| Ridge: Above plus embedding PCs x C | All text x C beyond text squares | -0.0003 | [-0.0012, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus embedding PCs x C | Embedding PCs x C beyond agreement x C | 0.0001 | [-0.0005, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus embedding PCs x C | All text x C beyond text squares | 0.0000 | [-0.0006, 0.0006] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus text x S | Text x S beyond text x C | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Above plus text x S | Text x S beyond text x C | -0.0001 | [-0.0004, 0.0002] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus text x S | Text x S beyond text x C | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 |

### Estimator comparisons on the same basis

Ridge and elastic net versus OLS, plus elastic net versus ridge, on each of the 13 joint/reference bases.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge: C main effects | Penalized estimator versus OLS | 0.0010 | [0.0001, 0.0019] | 0.853 | 1.000 | 1.000 |
| Elastic net: C main effects | Penalized estimator versus OLS | 0.0003 | [-0.0006, 0.0011] | 1.000 | 1.000 | 1.000 |
| Elastic net: C main effects | Elastic net versus ridge | -0.0007 | [-0.0014, -0.0001] | 1.000 | 1.000 | 1.000 |
| Ridge: Additive C + S | Penalized estimator versus OLS | 0.0011 | [0.0002, 0.0020] | 0.780 | 1.000 | 1.000 |
| Elastic net: Additive C + S | Penalized estimator versus OLS | 0.0004 | [-0.0005, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: Additive C + S | Elastic net versus ridge | -0.0006 | [-0.0013, 0.0000] | 1.000 | 1.000 | 1.000 |
| Ridge: Additive C + S + full text | Penalized estimator versus OLS | 0.0031 | [0.0016, 0.0046] | 0.001 | 0.003 | 0.047 |
| Elastic net: Additive C + S + full text | Penalized estimator versus OLS | 0.0043 | [0.0025, 0.0060] | <0.001 | <0.001 | 0.004 |
| Elastic net: Additive C + S + full text | Elastic net versus ridge | 0.0012 | [0.0003, 0.0020] | 0.209 | 0.172 | 0.255 |
| Ridge: C plus C squares | Penalized estimator versus OLS | 0.0001 | [-0.0012, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: C plus C squares | Penalized estimator versus OLS | 0.0001 | [-0.0013, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: C plus C squares | Elastic net versus ridge | 0.0000 | [-0.0009, 0.0009] | 1.000 | 1.000 | 1.000 |
| Ridge: C quadratic | Penalized estimator versus OLS | -0.0003 | [-0.0011, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic | Penalized estimator versus OLS | -0.0002 | [-0.0016, 0.0011] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic | Elastic net versus ridge | 0.0001 | [-0.0010, 0.0011] | 1.000 | 1.000 | 1.000 |
| Ridge: C quadratic + S main effects | Penalized estimator versus OLS | 0.0000 | [-0.0009, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic + S main effects | Penalized estimator versus OLS | -0.0001 | [-0.0015, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic + S main effects | Elastic net versus ridge | -0.0001 | [-0.0012, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: C quadratic + S main effects/squares | Penalized estimator versus OLS | 0.0000 | [-0.0009, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic + S main effects/squares | Penalized estimator versus OLS | -0.0001 | [-0.0016, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: C quadratic + S main effects/squares | Elastic net versus ridge | -0.0002 | [-0.0013, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S interactions | Penalized estimator versus OLS | 0.0000 | [-0.0009, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S interactions | Penalized estimator versus OLS | 0.0001 | [-0.0014, 0.0017] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S interactions | Elastic net versus ridge | 0.0001 | [-0.0010, 0.0013] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S plus text main effects | Penalized estimator versus OLS | -0.0001 | [-0.0010, 0.0008] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S plus text main effects | Penalized estimator versus OLS | 0.0000 | [-0.0015, 0.0015] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S plus text main effects | Elastic net versus ridge | 0.0001 | [-0.0011, 0.0013] | 1.000 | 1.000 | 1.000 |
| Ridge: Above plus text squares | Penalized estimator versus OLS | -0.0001 | [-0.0010, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus text squares | Penalized estimator versus OLS | 0.0001 | [-0.0015, 0.0017] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus text squares | Elastic net versus ridge | 0.0002 | [-0.0010, 0.0014] | 1.000 | 1.000 | 1.000 |
| Ridge: Above plus agreement x C | Penalized estimator versus OLS | 0.0008 | [-0.0005, 0.0022] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus agreement x C | Penalized estimator versus OLS | 0.0005 | [-0.0012, 0.0022] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus agreement x C | Elastic net versus ridge | -0.0003 | [-0.0013, 0.0006] | 1.000 | 1.000 | 1.000 |
| Ridge: Above plus embedding PCs x C | Penalized estimator versus OLS | 0.0019 | [0.0004, 0.0035] | 0.569 | 1.000 | 1.000 |
| Elastic net: Above plus embedding PCs x C | Penalized estimator versus OLS | 0.0024 | [0.0005, 0.0044] | 0.620 | 1.000 | 1.000 |
| Elastic net: Above plus embedding PCs x C | Elastic net versus ridge | 0.0005 | [-0.0004, 0.0015] | 1.000 | 1.000 | 1.000 |
| Ridge: Above plus text x S | Penalized estimator versus OLS | 0.0019 | [0.0003, 0.0035] | 0.764 | 1.000 | 1.000 |
| Elastic net: Above plus text x S | Penalized estimator versus OLS | 0.0024 | [0.0004, 0.0044] | 0.681 | 1.000 | 1.000 |
| Elastic net: Above plus text x S | Elastic net versus ridge | 0.0005 | [-0.0005, 0.0015] | 1.000 | 1.000 | 1.000 |

### Comparison with additive reference procedures

Joint social versus additive C+S; each compressed-text interaction basis versus additive full-embedding C+S+text. These are compound procedure comparisons.

| Procedure | Comparison | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: Joint C + S interactions | Joint social model versus additive C+S | 0.0008 | [-0.0012, 0.0028] | 1.000 | 1.000 | 1.000 |
| Ridge: Joint C + S interactions | Joint social model versus additive C+S | -0.0002 | [-0.0023, 0.0019] | 1.000 | 1.000 | 1.000 |
| Elastic net: Joint C + S interactions | Joint social model versus additive C+S | 0.0005 | [-0.0012, 0.0023] | 1.000 | 1.000 | 1.000 |
| OLS: Joint C + S plus text main effects | Joint C + S plus text main effects versus additive full text | 0.0049 | [0.0034, 0.0064] | <0.001 | <0.001 | <0.001 |
| Ridge: Joint C + S plus text main effects | Joint C + S plus text main effects versus additive full text | 0.0017 | [-0.0002, 0.0035] | 1.000 | 0.964 | 0.555 |
| Elastic net: Joint C + S plus text main effects | Joint C + S plus text main effects versus additive full text | 0.0006 | [-0.0010, 0.0023] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus text squares | Above plus text squares versus additive full text | 0.0045 | [0.0031, 0.0060] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus text squares | Above plus text squares versus additive full text | 0.0014 | [-0.0005, 0.0032] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus text squares | Above plus text squares versus additive full text | 0.0004 | [-0.0012, 0.0020] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus agreement x C | Above plus agreement x C versus additive full text | 0.0041 | [0.0025, 0.0056] | <0.001 | <0.001 | <0.001 |
| Ridge: Above plus agreement x C | Above plus agreement x C versus additive full text | 0.0018 | [0.0002, 0.0035] | 0.573 | 0.338 | 0.188 |
| Elastic net: Above plus agreement x C | Above plus agreement x C versus additive full text | 0.0003 | [-0.0014, 0.0020] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus embedding PCs x C | Above plus embedding PCs x C versus additive full text | 0.0022 | [0.0005, 0.0040] | 0.232 | 0.235 | 0.133 |
| Ridge: Above plus embedding PCs x C | Above plus embedding PCs x C versus additive full text | 0.0011 | [-0.0006, 0.0028] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus embedding PCs x C | Above plus embedding PCs x C versus additive full text | 0.0004 | [-0.0012, 0.0020] | 1.000 | 1.000 | 1.000 |
| OLS: Above plus text x S | Above plus text x S versus additive full text | 0.0021 | [0.0004, 0.0039] | 0.331 | 0.338 | 0.224 |
| Ridge: Above plus text x S | Above plus text x S versus additive full text | 0.0010 | [-0.0007, 0.0026] | 1.000 | 1.000 | 1.000 |
| Elastic net: Above plus text x S | Above plus text x S versus additive full text | 0.0003 | [-0.0013, 0.0019] | 1.000 | 1.000 | 1.000 |

## Complete register of the 35 individual social terms

Each term is tested against ridge on the same 191-column benchmark, with both procedures tuned separately by validation. The entire 35-term family is adjusted. Early/late differences and adjusted HAC5 p-values are descriptive persistence checks; no subperiod selects terms or a trading regime. Missing pilot comparisons are retained as blank entries.

### Raw returns

| Term added to 191-column ridge | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 | 2014-18 delta | 2014-18 adjusted p | 2019-22 delta | 2019-22 adjusted p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentiment x ret1 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x ret5 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0001 | 1.000 |
| Sentiment x ret21 | 0.0001 | [0.0000, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret63 | 0.0000 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret126 | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret252 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Sentiment x momentum252_skip21 | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x log_market_cap | 0.0001 | [-0.0001, 0.0004] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0001 | 1.000 |
| Sentiment x log_price | 0.0001 | [-0.0002, 0.0004] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0000 | 1.000 |
| Sentiment x turnover1 | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x turnover21 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Sentiment x log_dollar_volume1 | 0.0003 | [0.0000, 0.0005] | 1.000 | 0.694 | 0.513 | 0.0004 | 1.000 | 0.0001 | 1.000 |
| Sentiment x log_dollar_volume21 | 0.0002 | [0.0000, 0.0005] | 0.971 | 0.929 | 0.745 | 0.0003 | 1.000 | 0.0002 | 1.000 |
| Sentiment x volatility21 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x volatility63 | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 | -0.0002 | 1.000 | 0.0000 | 1.000 |
| Sentiment x max_return21 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | -0.0001 | 1.000 |
| Sentiment x amihud21 | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0002 | 1.000 |
| Attention x ret1 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | -0.0001 | 1.000 |
| Attention x ret5 | -0.0001 | [-0.0002, 0.0000] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x ret21 | -0.0001 | [-0.0002, 0.0000] | 1.000 | 1.000 | 1.000 | -0.0002 | 1.000 | 0.0000 | 1.000 |
| Attention x ret63 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x ret126 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x ret252 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x momentum252_skip21 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x log_market_cap | -0.0001 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x log_price | 0.0001 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x turnover1 | 0.0000 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0001 | 1.000 |
| Attention x turnover21 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x log_dollar_volume1 | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0001 | 1.000 |
| Attention x log_dollar_volume21 | 0.0000 | [-0.0003, 0.0003] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Attention x volatility21 | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | -0.0002 | 1.000 |
| Attention x volatility63 | 0.0000 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x max_return21 | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Attention x amihud21 | 0.0000 | [-0.0003, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x Attention | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |

### DGTW returns

| Term added to 191-column ridge | Delta IC | Pointwise 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 | 2014-18 delta | 2014-18 adjusted p | 2019-22 delta | 2019-22 adjusted p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentiment x ret1 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret5 | 0.0002 | [0.0000, 0.0003] | 0.974 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0003 | 1.000 |
| Sentiment x ret21 | 0.0001 | [0.0000, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0001 | 1.000 |
| Sentiment x ret63 | 0.0001 | [0.0000, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret126 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x ret252 | 0.0001 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0002 | 1.000 |
| Sentiment x momentum252_skip21 | 0.0001 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0002 | 1.000 |
| Sentiment x log_market_cap | 0.0003 | [0.0001, 0.0005] | 0.510 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0004 | 1.000 |
| Sentiment x log_price | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0001 | 1.000 |
| Sentiment x turnover1 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x turnover21 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Sentiment x log_dollar_volume1 | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0002 | 1.000 |
| Sentiment x log_dollar_volume21 | 0.0002 | [0.0000, 0.0004] | 0.985 | 0.749 | 0.607 | 0.0003 | 1.000 | 0.0002 | 1.000 |
| Sentiment x volatility21 | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x volatility63 | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0001 | 1.000 |
| Sentiment x max_return21 | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x amihud21 | 0.0002 | [0.0000, 0.0004] | 1.000 | 1.000 | 1.000 | 0.0002 | 1.000 | 0.0001 | 1.000 |
| Attention x ret1 | 0.0001 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0001 | 1.000 |
| Attention x ret5 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x ret21 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Attention x ret63 | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x ret126 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |
| Attention x ret252 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x momentum252_skip21 | 0.0000 | [-0.0001, 0.0001] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0000 | 1.000 |
| Attention x log_market_cap | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0002 | 1.000 |
| Attention x log_price | -0.0001 | [-0.0002, 0.0000] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | -0.0001 | 1.000 |
| Attention x turnover1 | 0.0001 | [0.0000, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0001 | 1.000 | 0.0001 | 1.000 |
| Attention x turnover21 | 0.0001 | [0.0000, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0002 | 1.000 |
| Attention x log_dollar_volume1 | -0.0001 | [-0.0002, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Attention x log_dollar_volume21 | -0.0001 | [-0.0003, 0.0000] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | -0.0002 | 1.000 |
| Attention x volatility21 | 0.0000 | [-0.0001, 0.0002] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0001 | 1.000 |
| Attention x volatility63 | 0.0001 | [-0.0001, 0.0003] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0002 | 1.000 |
| Attention x max_return21 | 0.0000 | [-0.0002, 0.0002] | 1.000 | 1.000 | 1.000 | -0.0001 | 1.000 | 0.0001 | 1.000 |
| Attention x amihud21 | -0.0001 | [-0.0003, 0.0001] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | -0.0001 | 1.000 |
| Sentiment x Attention | 0.0000 | [-0.0001, 0.0000] | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.0000 | 1.000 |

## Early/late stability of the registered block comparisons

Complete metric-specific effects and sensitivities are in the contrast artifact. These subperiods are fixed at 2014-2018 and 2019-2022 and are not independent confirmation samples.

### 2014-2018

| Question | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Nonlinearity in stock characteristics | 9 | 5 | 0 | 0 |
| Added sentiment/attention information | 12 | 12 | 4 | 0 |
| Social curvature and joint interactions | 9 | 5 | 0 | 0 |
| All 35 individual social interactions | 35 | 16 | 0 | 0 |
| Added text information | 18 | 2 | 0 | 4 |
| Text curvature and interaction blocks | 15 | 3 | 0 | 3 |
| Estimator comparisons on the same basis | 39 | 29 | 2 | 0 |
| Comparison with additive reference procedures | 18 | 15 | 3 | 0 |

### 2019-2022

| Question | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Nonlinearity in stock characteristics | 9 | 5 | 1 | 0 |
| Added sentiment/attention information | 12 | 12 | 10 | 0 |
| Social curvature and joint interactions | 9 | 8 | 0 | 0 |
| All 35 individual social interactions | 35 | 13 | 0 | 0 |
| Added text information | 18 | 0 | 0 | 4 |
| Text curvature and interaction blocks | 15 | 5 | 0 | 1 |
| Estimator comparisons on the same basis | 39 | 18 | 0 | 0 |
| Comparison with additive reference procedures | 18 | 17 | 2 | 0 |

## Limits and reproducibility

These are results for the already-inspected 2014-2022 development period, not untouched confirmation. No model is selected by its largest test score. No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim.

The controls are market-based; no point-in-time accounting or news predictors are added. The inherited DGTW target remains secondary because upstream accounting availability is unresolved. Failure to detect a gain is not proof that the relevant information does not exist. This study fits explicit linear-basis models, not trees or neural networks.

Bounded prediction batches preserve the global common sample and float64 forecasts. Existing alignment, rank/portfolio scoring and HAC kernels are reused unchanged. SHA-256 records cover the registry, predictions, prepared artifacts, reused sources and every evaluation output. The daily table is deterministic gzip.

- [summary](<data/linear_interactions_v1.csv>)
- [_yearly](<data/linear_interactions_v1_yearly.csv>)
- [_periods](<data/linear_interactions_v1_periods.csv>)
- [_contrasts](<data/linear_interactions_v1_contrasts.csv>)
- [_coverage](<data/linear_interactions_v1_coverage.csv>)
- [_deciles](<data/linear_interactions_v1_deciles.csv>)
- [_subgroups](<data/linear_interactions_v1_subgroups.csv>)
- [_horizons](<data/linear_interactions_v1_horizons.csv>)
- [_daily](<data/linear_interactions_v1_daily.csv.gz>)

- [Evaluation metadata and hashes](<data/linear_interactions_v1.json>)

- [Model registry](<../.runs/linear_interactions_v1/study/78b3dd65e0d5f8a6/linear/full_9f279d17e34278c6/registry.json>)
