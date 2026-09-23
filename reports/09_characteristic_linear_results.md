# Stock characteristics and incremental social-media information: linear models

All 264 registered linear models have been evaluated.

This experiment asks whether sentiment, attention, engineered social features and text add predictive information after conditioning on observed stock characteristics and past returns. It also shows what adding those controls does to the original social-only specifications. Raw returns are primary; DGTW-adjusted returns are a secondary robustness check.

## Design and common sample

C denotes 17 market and past-return controls plus 17 always-present missingness indicators. Accounting characteristics are excluded because reliable publication/availability dates are unavailable. Missing controls are imputed under the declared preparation rules without dropping the corresponding social-covered stock-days. This is a market-information benchmark, not an exhaustive stock-characteristic or news benchmark.

The signal at close t summarizes messages assigned to that close; h=1 returns run from close t to close t+1. OLS, ridge, lasso and elastic net use the same monthly fits, 126-session validation block, feature preparation, fitting-only scaling and equal-date squared loss on centered return ranks as the parent protocol. Penalties use the existing chronological validation procedure and mean daily Spearman IC. F504 is primary; F252/F756 are fixed sensitivities. Their test results do not select a training window. No new tuning search or neural-network run is part of this experiment.

| Target | Models | Eligible stock-days | Common predictions | Observed outcomes |
| --- | --- | --- | --- | --- |
| Raw (primary) | 132 | 3,034,035 | 3,034,035 | 3,033,080 |
| DGTW (secondary) | 132 | 3,034,035 | 3,034,035 | 2,748,978 |

| Inputs | Feature-set ID | Columns, including missing flags |
| --- | --- | --- |
| Sentiment + attention | core | 2 |
| All social | all | 53 |
| Text + sentiment + attention | textcore | 388 |
| Text + all social | textall | 439 |
| C only | characteristics | 34 |
| C + sentiment | characteristics_sentiment | 35 |
| C + attention | characteristics_attention | 35 |
| C + sentiment + attention | characteristics_core | 36 |
| C + all social | characteristics_all | 87 |
| C + text + sentiment + attention | characteristics_textcore | 422 |
| C + text + all social | characteristics_textall | 473 |

The original social-only controls and all characteristic-conditioned models use the same economic keys and prediction universe. Each target is scored on the joint finite-prediction intersection, then its observed outcomes. Raw and DGTW outcome coverage can differ. Rank IC is the mean daily Spearman correlation; the fitted rank scores are not percentage-return forecasts.

## Primary F504 rank IC

### Raw (primary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | 0.0348 | 0.0346 | 0.0350 | 0.0350 |
| All social | 0.0328 | 0.0345 | 0.0334 | 0.0335 |
| Text + sentiment + attention | 0.0392 | 0.0404 | 0.0403 | 0.0403 |
| Text + all social | 0.0390 | 0.0414 | 0.0401 | 0.0400 |
| C only | 0.0675 | 0.0690 | 0.0685 | 0.0684 |
| C + sentiment | 0.0681 | 0.0695 | 0.0690 | 0.0690 |
| C + attention | 0.0681 | 0.0699 | 0.0690 | 0.0692 |
| C + sentiment + attention | 0.0685 | 0.0700 | 0.0695 | 0.0695 |
| C + all social | 0.0675 | 0.0698 | 0.0698 | 0.0698 |
| C + text + sentiment + attention | 0.0628 | 0.0675 | 0.0693 | 0.0690 |
| C + text + all social | 0.0624 | 0.0676 | 0.0692 | 0.0693 |

### DGTW (secondary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | 0.0330 | 0.0330 | 0.0336 | 0.0336 |
| All social | 0.0296 | 0.0317 | 0.0307 | 0.0308 |
| Text + sentiment + attention | 0.0349 | 0.0363 | 0.0363 | 0.0363 |
| Text + all social | 0.0341 | 0.0372 | 0.0357 | 0.0357 |
| C only | 0.0593 | 0.0607 | 0.0604 | 0.0604 |
| C + sentiment | 0.0600 | 0.0616 | 0.0612 | 0.0613 |
| C + attention | 0.0603 | 0.0618 | 0.0617 | 0.0617 |
| C + sentiment + attention | 0.0606 | 0.0623 | 0.0619 | 0.0620 |
| C + all social | 0.0600 | 0.0623 | 0.0622 | 0.0623 |
| C + text + sentiment + attention | 0.0550 | 0.0599 | 0.0614 | 0.0615 |
| C + text + all social | 0.0549 | 0.0597 | 0.0616 | 0.0616 |

## Does sentiment and attention add information beyond C?

The prespecified comparison is C + sentiment + attention minus C, within each estimator and fitting history. A positive difference with a small adjusted p-value supports incremental out-of-sample ranking information relative to these observed controls. It does not establish causation or estimate the fraction of social information explained by stock characteristics.

| Target | Available estimators | Positive, adjusted p < .05 | Negative, adjusted p < .05 | Positive at all three HAC lags |
| --- | --- | --- | --- | --- |
| Raw (primary) | 4 | 4 | 0 | 3 |
| DGTW (secondary) | 4 | 4 | 0 | 4 |

## All prespecified social additions at F504

Differences are model minus benchmark. Confidence intervals are pointwise 95% intervals. P-values use paired daily differences and full-study Bonferroni adjustment separately within target, metric, family and period. HAC5 is primary; HAC21/HAC63 expose sensitivity to persistence. Missing statistics and unfinished windows do not shrink a family.

### Raw (primary)

| Addition / comparison | Estimator | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sentiment given C | OLS | 0.0006 | [0.0004, 0.0009] | 4.85 | <0.001 | <0.001 | <0.001 |
| Attention given C | OLS | 0.0006 | [0.0002, 0.0010] | 2.95 | 0.456 | 0.418 | 0.762 |
| Sentiment + attention given C | OLS | 0.0010 | [0.0005, 0.0014] | 4.50 | <0.001 | <0.001 | 0.001 |
| Attention given C + sentiment | OLS | 0.0003 | [0.0000, 0.0007] | 1.99 | 1.000 | 1.000 | 1.000 |
| Sentiment given C + attention | OLS | 0.0004 | [0.0002, 0.0006] | 3.47 | 0.074 | 0.153 | 0.268 |
| Other social given C + sentiment + attention | OLS | -0.0010 | [-0.0017, -0.0003] | -2.73 | 0.920 | 1.000 | 1.000 |
| Text given C + sentiment + attention | OLS | -0.0057 | [-0.0073, -0.0041] | -6.96 | <0.001 | <0.001 | <0.001 |
| Text given C + all social | OLS | -0.0050 | [-0.0064, -0.0037] | -7.14 | <0.001 | <0.001 | <0.001 |
| Other social given C + text + sentiment + attention | OLS | -0.0003 | [-0.0008, 0.0001] | -1.35 | 1.000 | 1.000 | 1.000 |
| All social given C | OLS | 0.0000 | [-0.0009, 0.0009] | -0.08 | 1.000 | 1.000 | 1.000 |
| Text + sentiment + attention given C | OLS | -0.0048 | [-0.0065, -0.0030] | -5.44 | <0.001 | <0.001 | 0.026 |
| Text + all social given C | OLS | -0.0051 | [-0.0070, -0.0031] | -5.16 | <0.001 | 0.001 | 0.103 |
| Sentiment given C | Ridge | 0.0005 | [0.0002, 0.0009] | 2.92 | 0.506 | 1.000 | 1.000 |
| Attention given C | Ridge | 0.0009 | [0.0005, 0.0013] | 4.23 | 0.003 | 0.003 | <0.001 |
| Sentiment + attention given C | Ridge | 0.0011 | [0.0006, 0.0016] | 4.32 | 0.002 | 0.004 | 0.002 |
| Attention given C + sentiment | Ridge | 0.0005 | [0.0002, 0.0009] | 3.08 | 0.301 | 0.202 | 0.026 |
| Sentiment given C + attention | Ridge | 0.0002 | [-0.0001, 0.0004] | 1.19 | 1.000 | 1.000 | 1.000 |
| Other social given C + sentiment + attention | Ridge | -0.0002 | [-0.0010, 0.0005] | -0.64 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Ridge | -0.0026 | [-0.0034, -0.0017] | -5.70 | <0.001 | <0.001 | <0.001 |
| Text given C + all social | Ridge | -0.0022 | [-0.0030, -0.0014] | -5.63 | <0.001 | <0.001 | <0.001 |
| Other social given C + text + sentiment + attention | Ridge | 0.0001 | [-0.0005, 0.0007] | 0.33 | 1.000 | 1.000 | 1.000 |
| All social given C | Ridge | 0.0008 | [-0.0001, 0.0018] | 1.79 | 1.000 | 1.000 | 1.000 |
| Text + sentiment + attention given C | Ridge | -0.0015 | [-0.0024, -0.0005] | -2.96 | 0.450 | 0.412 | 0.506 |
| Text + all social given C | Ridge | -0.0014 | [-0.0026, -0.0001] | -2.20 | 1.000 | 1.000 | 1.000 |
| Sentiment given C | Lasso | 0.0005 | [0.0002, 0.0008] | 3.45 | 0.081 | 0.075 | 0.094 |
| Attention given C | Lasso | 0.0005 | [0.0000, 0.0011] | 1.80 | 1.000 | 1.000 | 1.000 |
| Sentiment + attention given C | Lasso | 0.0010 | [0.0005, 0.0016] | 3.64 | 0.040 | 0.044 | 0.094 |
| Attention given C + sentiment | Lasso | 0.0005 | [0.0000, 0.0010] | 1.80 | 1.000 | 1.000 | 1.000 |
| Sentiment given C + attention | Lasso | 0.0005 | [0.0002, 0.0008] | 3.20 | 0.201 | 0.759 | 0.789 |
| Other social given C + sentiment + attention | Lasso | 0.0003 | [-0.0004, 0.0009] | 0.79 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Lasso | -0.0002 | [-0.0009, 0.0004] | -0.69 | 1.000 | 1.000 | 1.000 |
| Text given C + all social | Lasso | -0.0005 | [-0.0009, -0.0001] | -2.59 | 1.000 | 1.000 | 1.000 |
| Other social given C + text + sentiment + attention | Lasso | 0.0000 | [-0.0005, 0.0005] | -0.11 | 1.000 | 1.000 | 1.000 |
| All social given C | Lasso | 0.0013 | [0.0004, 0.0022] | 2.86 | 0.603 | 0.343 | 0.231 |
| Text + sentiment + attention given C | Lasso | 0.0008 | [0.0000, 0.0015] | 2.06 | 1.000 | 1.000 | 1.000 |
| Text + all social given C | Lasso | 0.0008 | [-0.0002, 0.0017] | 1.61 | 1.000 | 1.000 | 1.000 |
| Sentiment given C | Elastic net | 0.0006 | [0.0003, 0.0009] | 3.61 | 0.043 | 0.031 | 0.042 |
| Attention given C | Elastic net | 0.0008 | [0.0003, 0.0014] | 2.82 | 0.684 | 0.718 | 0.930 |
| Sentiment + attention given C | Elastic net | 0.0011 | [0.0005, 0.0016] | 3.90 | 0.014 | 0.011 | 0.022 |
| Attention given C + sentiment | Elastic net | 0.0005 | [0.0000, 0.0010] | 1.83 | 1.000 | 1.000 | 1.000 |
| Sentiment given C + attention | Elastic net | 0.0002 | [-0.0001, 0.0006] | 1.52 | 1.000 | 1.000 | 1.000 |
| Other social given C + sentiment + attention | Elastic net | 0.0004 | [-0.0002, 0.0010] | 1.16 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Elastic net | -0.0005 | [-0.0010, 0.0001] | -1.56 | 1.000 | 1.000 | 1.000 |
| Text given C + all social | Elastic net | -0.0005 | [-0.0009, -0.0001] | -2.55 | 1.000 | 1.000 | 1.000 |
| Other social given C + text + sentiment + attention | Elastic net | 0.0003 | [-0.0001, 0.0007] | 1.48 | 1.000 | 1.000 | 1.000 |
| All social given C | Elastic net | 0.0014 | [0.0006, 0.0023] | 3.36 | 0.114 | 0.036 | 0.048 |
| Text + sentiment + attention given C | Elastic net | 0.0006 | [-0.0001, 0.0014] | 1.65 | 1.000 | 1.000 | 1.000 |
| Text + all social given C | Elastic net | 0.0009 | [0.0000, 0.0018] | 2.04 | 1.000 | 1.000 | 1.000 |

### DGTW (secondary)

| Addition / comparison | Estimator | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sentiment given C | OLS | 0.0007 | [0.0004, 0.0010] | 5.13 | <0.001 | <0.001 | <0.001 |
| Attention given C | OLS | 0.0009 | [0.0006, 0.0013] | 5.15 | <0.001 | <0.001 | <0.001 |
| Sentiment + attention given C | OLS | 0.0013 | [0.0009, 0.0017] | 6.67 | <0.001 | <0.001 | <0.001 |
| Attention given C + sentiment | OLS | 0.0006 | [0.0003, 0.0009] | 4.01 | 0.009 | 0.009 | 0.030 |
| Sentiment given C + attention | OLS | 0.0004 | [0.0002, 0.0006] | 3.52 | 0.061 | 0.065 | 0.180 |
| Other social given C + sentiment + attention | OLS | -0.0006 | [-0.0012, -0.0001] | -2.15 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | OLS | -0.0056 | [-0.0069, -0.0044] | -9.09 | <0.001 | <0.001 | <0.001 |
| Text given C + all social | OLS | -0.0052 | [-0.0063, -0.0041] | -9.11 | <0.001 | <0.001 | <0.001 |
| Other social given C + text + sentiment + attention | OLS | -0.0001 | [-0.0005, 0.0003] | -0.69 | 1.000 | 1.000 | 1.000 |
| All social given C | OLS | 0.0007 | [0.0000, 0.0014] | 1.98 | 1.000 | 1.000 | 1.000 |
| Text + sentiment + attention given C | OLS | -0.0043 | [-0.0056, -0.0030] | -6.59 | <0.001 | <0.001 | 0.017 |
| Text + all social given C | OLS | -0.0045 | [-0.0059, -0.0031] | -6.26 | <0.001 | <0.001 | 0.053 |
| Sentiment given C | Ridge | 0.0009 | [0.0006, 0.0012] | 6.19 | <0.001 | <0.001 | <0.001 |
| Attention given C | Ridge | 0.0011 | [0.0007, 0.0015] | 5.56 | <0.001 | <0.001 | <0.001 |
| Sentiment + attention given C | Ridge | 0.0016 | [0.0012, 0.0021] | 7.10 | <0.001 | <0.001 | <0.001 |
| Attention given C + sentiment | Ridge | 0.0007 | [0.0003, 0.0011] | 3.74 | 0.027 | 0.023 | 0.003 |
| Sentiment given C + attention | Ridge | 0.0005 | [0.0003, 0.0007] | 3.99 | 0.009 | 0.017 | 0.036 |
| Other social given C + sentiment + attention | Ridge | 0.0000 | [-0.0006, 0.0005] | -0.16 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Ridge | -0.0024 | [-0.0032, -0.0016] | -5.95 | <0.001 | <0.001 | <0.001 |
| Text given C + all social | Ridge | -0.0025 | [-0.0033, -0.0018] | -6.62 | <0.001 | <0.001 | <0.001 |
| Other social given C + text + sentiment + attention | Ridge | -0.0002 | [-0.0007, 0.0003] | -0.66 | 1.000 | 1.000 | 1.000 |
| All social given C | Ridge | 0.0016 | [0.0009, 0.0023] | 4.33 | 0.002 | 0.004 | 0.004 |
| Text + sentiment + attention given C | Ridge | -0.0008 | [-0.0017, 0.0001] | -1.74 | 1.000 | 1.000 | 1.000 |
| Text + all social given C | Ridge | -0.0010 | [-0.0020, 0.0001] | -1.81 | 1.000 | 1.000 | 1.000 |
| Sentiment given C | Lasso | 0.0009 | [0.0005, 0.0012] | 4.69 | <0.001 | <0.001 | 0.002 |
| Attention given C | Lasso | 0.0013 | [0.0008, 0.0018] | 5.19 | <0.001 | <0.001 | <0.001 |
| Sentiment + attention given C | Lasso | 0.0015 | [0.0010, 0.0020] | 5.88 | <0.001 | <0.001 | <0.001 |
| Attention given C + sentiment | Lasso | 0.0007 | [0.0002, 0.0012] | 2.74 | 0.875 | 0.828 | 1.000 |
| Sentiment given C + attention | Lasso | 0.0002 | [0.0000, 0.0005] | 1.87 | 1.000 | 1.000 | 1.000 |
| Other social given C + sentiment + attention | Lasso | 0.0004 | [0.0000, 0.0007] | 1.82 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Lasso | -0.0005 | [-0.0009, 0.0000] | -2.02 | 1.000 | 1.000 | 1.000 |
| Text given C + all social | Lasso | -0.0007 | [-0.0010, -0.0003] | -3.36 | 0.114 | 0.204 | 0.532 |
| Other social given C + text + sentiment + attention | Lasso | 0.0002 | [-0.0002, 0.0005] | 0.86 | 1.000 | 1.000 | 1.000 |
| All social given C | Lasso | 0.0019 | [0.0013, 0.0025] | 6.02 | <0.001 | <0.001 | <0.001 |
| Text + sentiment + attention given C | Lasso | 0.0011 | [0.0005, 0.0017] | 3.44 | 0.083 | 0.074 | 0.148 |
| Text + all social given C | Lasso | 0.0012 | [0.0005, 0.0019] | 3.54 | 0.058 | 0.035 | 0.081 |
| Sentiment given C | Elastic net | 0.0009 | [0.0006, 0.0013] | 4.95 | <0.001 | <0.001 | <0.001 |
| Attention given C | Elastic net | 0.0013 | [0.0008, 0.0018] | 5.06 | <0.001 | <0.001 | <0.001 |
| Sentiment + attention given C | Elastic net | 0.0016 | [0.0011, 0.0021] | 6.50 | <0.001 | <0.001 | <0.001 |
| Attention given C + sentiment | Elastic net | 0.0007 | [0.0002, 0.0012] | 2.74 | 0.894 | 0.964 | 1.000 |
| Sentiment given C + attention | Elastic net | 0.0003 | [0.0001, 0.0006] | 2.36 | 1.000 | 1.000 | 1.000 |
| Other social given C + sentiment + attention | Elastic net | 0.0003 | [-0.0001, 0.0007] | 1.51 | 1.000 | 1.000 | 1.000 |
| Text given C + sentiment + attention | Elastic net | -0.0005 | [-0.0009, -0.0001] | -2.22 | 1.000 | 1.000 | 1.000 |
| Text given C + all social | Elastic net | -0.0007 | [-0.0011, -0.0003] | -3.43 | 0.087 | 0.092 | 0.278 |
| Other social given C + text + sentiment + attention | Elastic net | 0.0001 | [-0.0002, 0.0004] | 0.47 | 1.000 | 1.000 | 1.000 |
| All social given C | Elastic net | 0.0019 | [0.0013, 0.0025] | 6.12 | <0.001 | <0.001 | <0.001 |
| Text + sentiment + attention given C | Elastic net | 0.0011 | [0.0005, 0.0017] | 3.71 | 0.030 | 0.025 | 0.050 |
| Text + all social given C | Elastic net | 0.0012 | [0.0005, 0.0019] | 3.53 | 0.059 | 0.037 | 0.062 |

| Family | Planned comparisons per target / metric / period |
| --- | --- |
| incremental_social | 144 |
| conditioning | 48 |
| estimator | 99 |
| history | 88 |

The 12 incremental-social edges distinguish sentiment, attention, their combination, other engineered social variables, and text. The conditioning family compares C + S with the corresponding original S. There are no blanket comparisons of every new feature set against bare sentiment and attention.

## What changes when C is added to the original social models?

These paired comparisons measure the gain from adding C to a fixed social input set. They complement C + S minus C; an improvement in C + S over S alone does not itself establish that social media adds information beyond C.

### Raw (primary)

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0337 | [0.0289, 0.0385] | 13.83 | <0.001 | <0.001 | <0.001 |
| OLS | C + all social | 0.0347 | [0.0299, 0.0394] | 14.35 | <0.001 | <0.001 | <0.001 |
| OLS | C + text + sentiment + attention | 0.0235 | [0.0204, 0.0266] | 14.91 | <0.001 | <0.001 | <0.001 |
| OLS | C + text + all social | 0.0235 | [0.0204, 0.0265] | 14.99 | <0.001 | <0.001 | <0.001 |
| Ridge | C + sentiment + attention | 0.0354 | [0.0303, 0.0405] | 13.67 | <0.001 | <0.001 | <0.001 |
| Ridge | C + all social | 0.0353 | [0.0304, 0.0402] | 14.20 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + sentiment + attention | 0.0271 | [0.0234, 0.0307] | 14.41 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + all social | 0.0262 | [0.0226, 0.0297] | 14.38 | <0.001 | <0.001 | <0.001 |
| Lasso | C + sentiment + attention | 0.0345 | [0.0296, 0.0394] | 13.74 | <0.001 | <0.001 | <0.001 |
| Lasso | C + all social | 0.0364 | [0.0313, 0.0414] | 14.04 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + sentiment + attention | 0.0289 | [0.0247, 0.0331] | 13.57 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + all social | 0.0292 | [0.0250, 0.0334] | 13.66 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + sentiment + attention | 0.0345 | [0.0295, 0.0394] | 13.69 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + all social | 0.0364 | [0.0313, 0.0415] | 14.02 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + sentiment + attention | 0.0287 | [0.0245, 0.0329] | 13.42 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + all social | 0.0293 | [0.0251, 0.0335] | 13.66 | <0.001 | <0.001 | <0.001 |

### DGTW (secondary)

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0277 | [0.0253, 0.0301] | 22.67 | <0.001 | <0.001 | <0.001 |
| OLS | C + all social | 0.0304 | [0.0278, 0.0330] | 23.15 | <0.001 | <0.001 | <0.001 |
| OLS | C + text + sentiment + attention | 0.0201 | [0.0184, 0.0218] | 23.13 | <0.001 | <0.001 | <0.001 |
| OLS | C + text + all social | 0.0207 | [0.0191, 0.0224] | 24.49 | <0.001 | <0.001 | <0.001 |
| Ridge | C + sentiment + attention | 0.0293 | [0.0271, 0.0315] | 25.60 | <0.001 | <0.001 | <0.001 |
| Ridge | C + all social | 0.0306 | [0.0282, 0.0329] | 25.77 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + sentiment + attention | 0.0236 | [0.0218, 0.0253] | 26.40 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + all social | 0.0225 | [0.0208, 0.0243] | 25.25 | <0.001 | <0.001 | <0.001 |
| Lasso | C + sentiment + attention | 0.0283 | [0.0261, 0.0306] | 24.55 | <0.001 | <0.001 | <0.001 |
| Lasso | C + all social | 0.0315 | [0.0290, 0.0340] | 24.26 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + sentiment + attention | 0.0252 | [0.0230, 0.0274] | 22.31 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + all social | 0.0259 | [0.0237, 0.0281] | 23.28 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + sentiment + attention | 0.0284 | [0.0262, 0.0307] | 24.77 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + all social | 0.0315 | [0.0290, 0.0341] | 24.42 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + sentiment + attention | 0.0252 | [0.0230, 0.0274] | 22.49 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + all social | 0.0259 | [0.0237, 0.0281] | 23.31 | <0.001 | <0.001 | <0.001 |

## Penalized estimators versus OLS at fixed inputs

Penalization need not outperform OLS in every finite sample, particularly when inputs are few or the signal is weak. This stage changes the conditioning information while retaining the existing tuning procedure. Any later window or penalty-grid redesign must be chosen on chronological validation data rather than these test comparisons.

### Raw (primary)

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | Sentiment + attention | -0.0001 | [-0.0003, 0.0001] | -1.04 | 1.000 | 1.000 | 1.000 |
| Ridge | All social | 0.0017 | [0.0006, 0.0028] | 3.00 | 0.266 | 0.233 | 0.263 |
| Ridge | Text + sentiment + attention | 0.0012 | [0.0002, 0.0022] | 2.43 | 1.000 | 0.288 | 0.411 |
| Ridge | Text + all social | 0.0025 | [0.0011, 0.0038] | 3.50 | 0.046 | 0.011 | 0.066 |
| Ridge | C only | 0.0015 | [0.0005, 0.0024] | 3.08 | 0.202 | 0.366 | 1.000 |
| Ridge | C + sentiment | 0.0014 | [0.0004, 0.0023] | 2.88 | 0.397 | 0.688 | 1.000 |
| Ridge | C + attention | 0.0018 | [0.0008, 0.0028] | 3.51 | 0.044 | 0.080 | 0.287 |
| Ridge | C + sentiment + attention | 0.0016 | [0.0006, 0.0026] | 3.14 | 0.168 | 0.271 | 0.658 |
| Ridge | C + all social | 0.0023 | [0.0013, 0.0034] | 4.45 | <0.001 | 0.004 | 0.099 |
| Ridge | C + text + sentiment + attention | 0.0047 | [0.0030, 0.0065] | 5.35 | <0.001 | <0.001 | 0.006 |
| Ridge | C + text + all social | 0.0052 | [0.0034, 0.0069] | 5.76 | <0.001 | <0.001 | 0.011 |
| Lasso | Sentiment + attention | 0.0002 | [-0.0002, 0.0006] | 1.05 | 1.000 | 1.000 | 1.000 |
| Lasso | All social | 0.0006 | [-0.0003, 0.0014] | 1.28 | 1.000 | 1.000 | 1.000 |
| Lasso | Text + sentiment + attention | 0.0011 | [0.0003, 0.0020] | 2.68 | 0.731 | 0.516 | 0.472 |
| Lasso | Text + all social | 0.0011 | [0.0003, 0.0019] | 2.57 | 0.994 | 1.000 | 1.000 |
| Lasso | C only | 0.0010 | [0.0002, 0.0018] | 2.33 | 1.000 | 1.000 | 1.000 |
| Lasso | C + sentiment | 0.0009 | [0.0001, 0.0017] | 2.10 | 1.000 | 1.000 | 1.000 |
| Lasso | C + attention | 0.0009 | [0.0001, 0.0018] | 2.12 | 1.000 | 1.000 | 1.000 |
| Lasso | C + sentiment + attention | 0.0010 | [0.0002, 0.0019] | 2.40 | 1.000 | 1.000 | 1.000 |
| Lasso | C + all social | 0.0023 | [0.0010, 0.0036] | 3.45 | 0.056 | 0.113 | 0.431 |
| Lasso | C + text + sentiment + attention | 0.0065 | [0.0045, 0.0085] | 6.35 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + all social | 0.0068 | [0.0046, 0.0090] | 6.12 | <0.001 | <0.001 | 0.001 |
| Elastic net | Sentiment + attention | 0.0002 | [-0.0002, 0.0006] | 1.07 | 1.000 | 1.000 | 1.000 |
| Elastic net | All social | 0.0006 | [-0.0002, 0.0015] | 1.42 | 1.000 | 1.000 | 1.000 |
| Elastic net | Text + sentiment + attention | 0.0011 | [0.0003, 0.0020] | 2.65 | 0.786 | 0.563 | 0.517 |
| Elastic net | Text + all social | 0.0011 | [0.0003, 0.0019] | 2.60 | 0.926 | 1.000 | 1.000 |
| Elastic net | C only | 0.0009 | [0.0000, 0.0017] | 1.90 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + sentiment | 0.0008 | [0.0000, 0.0017] | 1.86 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + attention | 0.0011 | [0.0002, 0.0020] | 2.36 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + sentiment + attention | 0.0010 | [0.0001, 0.0019] | 2.14 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + all social | 0.0023 | [0.0010, 0.0036] | 3.54 | 0.040 | 0.083 | 0.336 |
| Elastic net | C + text + sentiment + attention | 0.0062 | [0.0042, 0.0083] | 6.02 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + all social | 0.0069 | [0.0047, 0.0090] | 6.18 | <0.001 | <0.001 | <0.001 |

### DGTW (secondary)

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | Sentiment + attention | 0.0000 | [-0.0002, 0.0003] | 0.24 | 1.000 | 1.000 | 1.000 |
| Ridge | All social | 0.0021 | [0.0010, 0.0032] | 3.79 | 0.015 | 0.051 | 0.216 |
| Ridge | Text + sentiment + attention | 0.0014 | [0.0006, 0.0023] | 3.29 | 0.100 | 0.065 | 0.210 |
| Ridge | Text + all social | 0.0030 | [0.0017, 0.0044] | 4.38 | 0.001 | 0.002 | 0.005 |
| Ridge | C only | 0.0013 | [0.0007, 0.0020] | 3.87 | 0.011 | 0.037 | 0.441 |
| Ridge | C + sentiment | 0.0016 | [0.0010, 0.0022] | 5.08 | <0.001 | <0.001 | 0.009 |
| Ridge | C + attention | 0.0016 | [0.0009, 0.0022] | 4.59 | <0.001 | 0.003 | 0.062 |
| Ridge | C + sentiment + attention | 0.0017 | [0.0010, 0.0023] | 5.18 | <0.001 | <0.001 | 0.004 |
| Ridge | C + all social | 0.0022 | [0.0016, 0.0029] | 6.45 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + sentiment + attention | 0.0049 | [0.0037, 0.0061] | 8.04 | <0.001 | <0.001 | <0.001 |
| Ridge | C + text + all social | 0.0049 | [0.0036, 0.0061] | 7.80 | <0.001 | <0.001 | <0.001 |
| Lasso | Sentiment + attention | 0.0006 | [0.0003, 0.0009] | 4.20 | 0.003 | 0.010 | 0.264 |
| Lasso | All social | 0.0011 | [0.0002, 0.0021] | 2.38 | 1.000 | 1.000 | 1.000 |
| Lasso | Text + sentiment + attention | 0.0014 | [0.0004, 0.0023] | 2.89 | 0.377 | 0.565 | 0.384 |
| Lasso | Text + all social | 0.0016 | [0.0005, 0.0027] | 2.78 | 0.543 | 0.613 | 0.704 |
| Lasso | C only | 0.0010 | [0.0004, 0.0016] | 3.18 | 0.147 | 0.097 | 0.062 |
| Lasso | C + sentiment | 0.0012 | [0.0006, 0.0017] | 4.15 | 0.003 | <0.001 | <0.001 |
| Lasso | C + attention | 0.0014 | [0.0008, 0.0020] | 4.48 | <0.001 | <0.001 | <0.001 |
| Lasso | C + sentiment + attention | 0.0012 | [0.0006, 0.0018] | 4.12 | 0.004 | 0.002 | 0.002 |
| Lasso | C + all social | 0.0022 | [0.0015, 0.0029] | 5.91 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + sentiment + attention | 0.0064 | [0.0050, 0.0078] | 9.13 | <0.001 | <0.001 | <0.001 |
| Lasso | C + text + all social | 0.0067 | [0.0052, 0.0082] | 8.97 | <0.001 | <0.001 | <0.001 |
| Elastic net | Sentiment + attention | 0.0006 | [0.0003, 0.0009] | 4.20 | 0.003 | 0.010 | 0.268 |
| Elastic net | All social | 0.0011 | [0.0002, 0.0021] | 2.40 | 1.000 | 1.000 | 1.000 |
| Elastic net | Text + sentiment + attention | 0.0014 | [0.0005, 0.0024] | 3.06 | 0.221 | 0.335 | 0.193 |
| Elastic net | Text + all social | 0.0016 | [0.0005, 0.0027] | 2.77 | 0.550 | 0.623 | 0.706 |
| Elastic net | C only | 0.0011 | [0.0004, 0.0017] | 3.30 | 0.094 | 0.084 | 0.065 |
| Elastic net | C + sentiment | 0.0013 | [0.0007, 0.0019] | 4.54 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + attention | 0.0014 | [0.0008, 0.0021] | 4.41 | 0.001 | 0.001 | 0.003 |
| Elastic net | C + sentiment + attention | 0.0014 | [0.0008, 0.0020] | 4.35 | 0.001 | <0.001 | <0.001 |
| Elastic net | C + all social | 0.0023 | [0.0015, 0.0030] | 5.87 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + sentiment + attention | 0.0065 | [0.0051, 0.0079] | 9.07 | <0.001 | <0.001 | <0.001 |
| Elastic net | C + text + all social | 0.0068 | [0.0053, 0.0082] | 8.93 | <0.001 | <0.001 | <0.001 |

## Fixed training-window sensitivity

These IC levels describe the three registered histories. F504 remains primary; no best-window result or optimized strategy is reported. The complete contrast artifact includes paired F252/F756 minus F504 differences with the fixed 88-comparison history family.

| Target | Inputs | Estimator | F252 | F504 (primary) | F756 |
| --- | --- | --- | --- | --- | --- |
| DGTW (secondary) | All social | Elastic net | 0.0309 | 0.0308 | 0.0307 |
| DGTW (secondary) | All social | Lasso | 0.0309 | 0.0307 | 0.0307 |
| DGTW (secondary) | All social | OLS | 0.0293 | 0.0296 | 0.0300 |
| DGTW (secondary) | All social | Ridge | 0.0314 | 0.0317 | 0.0312 |
| DGTW (secondary) | C + all social | Elastic net | 0.0614 | 0.0623 | 0.0625 |
| DGTW (secondary) | C + all social | Lasso | 0.0613 | 0.0622 | 0.0625 |
| DGTW (secondary) | C + all social | OLS | 0.0587 | 0.0600 | 0.0607 |
| DGTW (secondary) | C + all social | Ridge | 0.0611 | 0.0623 | 0.0627 |
| DGTW (secondary) | C + attention | Elastic net | 0.0608 | 0.0617 | 0.0620 |
| DGTW (secondary) | C + attention | Lasso | 0.0607 | 0.0617 | 0.0620 |
| DGTW (secondary) | C + attention | OLS | 0.0592 | 0.0603 | 0.0609 |
| DGTW (secondary) | C + attention | Ridge | 0.0613 | 0.0618 | 0.0621 |
| DGTW (secondary) | C + sentiment | Elastic net | 0.0604 | 0.0613 | 0.0619 |
| DGTW (secondary) | C + sentiment | Lasso | 0.0604 | 0.0612 | 0.0618 |
| DGTW (secondary) | C + sentiment | OLS | 0.0588 | 0.0600 | 0.0608 |
| DGTW (secondary) | C + sentiment | Ridge | 0.0608 | 0.0616 | 0.0620 |
| DGTW (secondary) | C + sentiment + attention | Elastic net | 0.0610 | 0.0620 | 0.0624 |
| DGTW (secondary) | C + sentiment + attention | Lasso | 0.0610 | 0.0619 | 0.0624 |
| DGTW (secondary) | C + sentiment + attention | OLS | 0.0595 | 0.0606 | 0.0613 |
| DGTW (secondary) | C + sentiment + attention | Ridge | 0.0616 | 0.0623 | 0.0626 |
| DGTW (secondary) | C + text + all social | Elastic net | 0.0608 | 0.0616 | 0.0620 |
| DGTW (secondary) | C + text + all social | Lasso | 0.0607 | 0.0616 | 0.0620 |
| DGTW (secondary) | C + text + all social | OLS | 0.0528 | 0.0549 | 0.0565 |
| DGTW (secondary) | C + text + all social | Ridge | 0.0585 | 0.0597 | 0.0607 |
| DGTW (secondary) | C + text + sentiment + attention | Elastic net | 0.0610 | 0.0615 | 0.0619 |
| DGTW (secondary) | C + text + sentiment + attention | Lasso | 0.0608 | 0.0614 | 0.0619 |
| DGTW (secondary) | C + text + sentiment + attention | OLS | 0.0531 | 0.0550 | 0.0566 |
| DGTW (secondary) | C + text + sentiment + attention | Ridge | 0.0587 | 0.0599 | 0.0606 |
| DGTW (secondary) | C only | Elastic net | 0.0596 | 0.0604 | 0.0610 |
| DGTW (secondary) | C only | Lasso | 0.0595 | 0.0604 | 0.0609 |
| DGTW (secondary) | C only | OLS | 0.0582 | 0.0593 | 0.0601 |
| DGTW (secondary) | C only | Ridge | 0.0600 | 0.0607 | 0.0613 |
| DGTW (secondary) | Sentiment + attention | Elastic net | 0.0334 | 0.0336 | 0.0335 |
| DGTW (secondary) | Sentiment + attention | Lasso | 0.0334 | 0.0336 | 0.0335 |
| DGTW (secondary) | Sentiment + attention | OLS | 0.0325 | 0.0330 | 0.0327 |
| DGTW (secondary) | Sentiment + attention | Ridge | 0.0328 | 0.0330 | 0.0328 |
| DGTW (secondary) | Text + all social | Elastic net | 0.0346 | 0.0357 | 0.0358 |
| DGTW (secondary) | Text + all social | Lasso | 0.0345 | 0.0357 | 0.0359 |
| DGTW (secondary) | Text + all social | OLS | 0.0330 | 0.0341 | 0.0357 |
| DGTW (secondary) | Text + all social | Ridge | 0.0368 | 0.0372 | 0.0374 |
| DGTW (secondary) | Text + sentiment + attention | Elastic net | 0.0358 | 0.0363 | 0.0368 |
| DGTW (secondary) | Text + sentiment + attention | Lasso | 0.0358 | 0.0363 | 0.0368 |
| DGTW (secondary) | Text + sentiment + attention | OLS | 0.0339 | 0.0349 | 0.0363 |
| DGTW (secondary) | Text + sentiment + attention | Ridge | 0.0356 | 0.0363 | 0.0365 |
| Raw (primary) | All social | Elastic net | 0.0339 | 0.0335 | 0.0331 |
| Raw (primary) | All social | Lasso | 0.0339 | 0.0334 | 0.0331 |
| Raw (primary) | All social | OLS | 0.0326 | 0.0328 | 0.0328 |
| Raw (primary) | All social | Ridge | 0.0346 | 0.0345 | 0.0336 |
| Raw (primary) | C + all social | Elastic net | 0.0683 | 0.0698 | 0.0704 |
| Raw (primary) | C + all social | Lasso | 0.0682 | 0.0698 | 0.0704 |
| Raw (primary) | C + all social | OLS | 0.0653 | 0.0675 | 0.0682 |
| Raw (primary) | C + all social | Ridge | 0.0682 | 0.0698 | 0.0703 |
| Raw (primary) | C + attention | Elastic net | 0.0680 | 0.0692 | 0.0700 |
| Raw (primary) | C + attention | Lasso | 0.0679 | 0.0690 | 0.0699 |
| Raw (primary) | C + attention | OLS | 0.0660 | 0.0681 | 0.0688 |
| Raw (primary) | C + attention | Ridge | 0.0688 | 0.0699 | 0.0701 |
| Raw (primary) | C + sentiment | Elastic net | 0.0676 | 0.0690 | 0.0697 |
| Raw (primary) | C + sentiment | Lasso | 0.0676 | 0.0690 | 0.0697 |
| Raw (primary) | C + sentiment | OLS | 0.0660 | 0.0681 | 0.0688 |
| Raw (primary) | C + sentiment | Ridge | 0.0685 | 0.0695 | 0.0698 |
| Raw (primary) | C + sentiment + attention | Elastic net | 0.0681 | 0.0695 | 0.0701 |
| Raw (primary) | C + sentiment + attention | Lasso | 0.0681 | 0.0695 | 0.0701 |
| Raw (primary) | C + sentiment + attention | OLS | 0.0664 | 0.0685 | 0.0692 |
| Raw (primary) | C + sentiment + attention | Ridge | 0.0691 | 0.0700 | 0.0703 |
| Raw (primary) | C + text + all social | Elastic net | 0.0679 | 0.0693 | 0.0700 |
| Raw (primary) | C + text + all social | Lasso | 0.0678 | 0.0692 | 0.0700 |
| Raw (primary) | C + text + all social | OLS | 0.0592 | 0.0624 | 0.0642 |
| Raw (primary) | C + text + all social | Ridge | 0.0650 | 0.0676 | 0.0684 |
| Raw (primary) | C + text + sentiment + attention | Elastic net | 0.0680 | 0.0690 | 0.0701 |
| Raw (primary) | C + text + sentiment + attention | Lasso | 0.0678 | 0.0693 | 0.0699 |
| Raw (primary) | C + text + sentiment + attention | OLS | 0.0596 | 0.0628 | 0.0645 |
| Raw (primary) | C + text + sentiment + attention | Ridge | 0.0658 | 0.0675 | 0.0685 |
| Raw (primary) | C only | Elastic net | 0.0669 | 0.0684 | 0.0690 |
| Raw (primary) | C only | Lasso | 0.0669 | 0.0685 | 0.0691 |
| Raw (primary) | C only | OLS | 0.0654 | 0.0675 | 0.0683 |
| Raw (primary) | C only | Ridge | 0.0682 | 0.0690 | 0.0692 |
| Raw (primary) | Sentiment + attention | Elastic net | 0.0357 | 0.0350 | 0.0343 |
| Raw (primary) | Sentiment + attention | Lasso | 0.0357 | 0.0350 | 0.0343 |
| Raw (primary) | Sentiment + attention | OLS | 0.0352 | 0.0348 | 0.0339 |
| Raw (primary) | Sentiment + attention | Ridge | 0.0351 | 0.0346 | 0.0338 |
| Raw (primary) | Text + all social | Elastic net | 0.0386 | 0.0400 | 0.0408 |
| Raw (primary) | Text + all social | Lasso | 0.0387 | 0.0401 | 0.0408 |
| Raw (primary) | Text + all social | OLS | 0.0375 | 0.0390 | 0.0404 |
| Raw (primary) | Text + all social | Ridge | 0.0408 | 0.0414 | 0.0413 |
| Raw (primary) | Text + sentiment + attention | Elastic net | 0.0399 | 0.0403 | 0.0405 |
| Raw (primary) | Text + sentiment + attention | Lasso | 0.0399 | 0.0403 | 0.0405 |
| Raw (primary) | Text + sentiment + attention | OLS | 0.0379 | 0.0392 | 0.0405 |
| Raw (primary) | Text + sentiment + attention | Ridge | 0.0399 | 0.0404 | 0.0406 |

## Gross portfolio diagnostics

Average daily top-minus-bottom decile returns are shown in basis points. All eleven input sets are retained. These are gross descriptive sorts with fractional tie handling and matched prediction samples, not implementable net returns. Paired spread differences, confidence intervals, and adjusted p-values are in the contrast artifact.

### Equal-weighted spread: Raw (primary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | 26.98 | 27.42 | 27.79 | 27.79 |
| All social | 27.45 | 25.90 | 25.88 | 26.07 |
| Text + sentiment + attention | 21.05 | 20.02 | 23.34 | 23.14 |
| Text + all social | 26.58 | 26.96 | 27.46 | 27.31 |
| C only | 22.19 | 19.87 | 20.55 | 20.46 |
| C + sentiment | 24.08 | 22.61 | 22.82 | 22.82 |
| C + attention | 23.24 | 21.74 | 21.03 | 21.95 |
| C + sentiment + attention | 25.59 | 23.73 | 23.13 | 23.42 |
| C + all social | 29.25 | 29.87 | 26.18 | 26.53 |
| C + text + sentiment + attention | 25.52 | 22.69 | 23.12 | 23.15 |
| C + text + all social | 27.47 | 29.10 | 25.51 | 25.84 |

### Equal-weighted spread: DGTW (secondary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | 23.54 | 24.99 | 25.33 | 25.31 |
| All social | 23.06 | 22.41 | 22.44 | 22.69 |
| Text + sentiment + attention | 17.04 | 16.59 | 19.61 | 19.48 |
| Text + all social | 22.47 | 22.73 | 23.05 | 23.15 |
| C only | 17.93 | 14.50 | 14.68 | 14.96 |
| C + sentiment | 19.28 | 16.63 | 16.88 | 17.33 |
| C + attention | 19.20 | 17.89 | 16.75 | 16.59 |
| C + sentiment + attention | 20.47 | 19.14 | 17.93 | 18.03 |
| C + all social | 24.72 | 25.30 | 22.40 | 22.01 |
| C + text + sentiment + attention | 20.02 | 18.90 | 17.12 | 17.33 |
| C + text + all social | 23.36 | 24.81 | 21.22 | 21.33 |

### Capitalization-weighted spread: Raw (primary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | -2.12 | -2.21 | -2.40 | -2.40 |
| All social | -1.06 | -0.84 | -1.17 | -1.08 |
| Text + sentiment + attention | 3.25 | 1.96 | 3.23 | 3.13 |
| Text + all social | 5.39 | 3.86 | 4.50 | 4.57 |
| C only | 11.50 | 14.69 | 16.48 | 16.55 |
| C + sentiment | 9.79 | 16.44 | 16.36 | 15.97 |
| C + attention | 13.79 | 14.87 | 10.94 | 11.99 |
| C + sentiment + attention | 13.43 | 15.42 | 15.18 | 15.33 |
| C + all social | 15.02 | 15.64 | 11.91 | 12.71 |
| C + text + sentiment + attention | 9.74 | 11.68 | 8.83 | 8.95 |
| C + text + all social | 13.87 | 18.34 | 10.49 | 10.52 |

### Capitalization-weighted spread: DGTW (secondary)

| Inputs | OLS | Ridge | Lasso | Elastic net |
| --- | --- | --- | --- | --- |
| Sentiment + attention | -1.33 | -0.77 | -1.21 | -1.21 |
| All social | 1.61 | 0.14 | 0.67 | 0.73 |
| Text + sentiment + attention | 2.13 | 0.73 | 1.66 | 1.86 |
| Text + all social | 4.04 | 2.20 | 2.45 | 2.50 |
| C only | 9.39 | 10.70 | 9.48 | 10.54 |
| C + sentiment | 10.01 | 9.86 | 8.44 | 9.43 |
| C + attention | 11.24 | 11.08 | 9.44 | 7.82 |
| C + sentiment + attention | 10.72 | 11.64 | 9.39 | 8.77 |
| C + all social | 5.61 | 9.57 | 6.83 | 8.21 |
| C + text + sentiment + attention | 4.07 | 9.55 | 9.27 | 8.68 |
| C + text + all social | 7.40 | 10.61 | 9.52 | 9.75 |

## Stability of sentiment and attention beyond C

The same C + sentiment + attention minus C comparison is reported in both fixed subperiods. These are descriptive stability checks, not a rule for selecting when to trade.

### Raw (primary), 2014-2018

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0007 | [0.0002, 0.0012] | 2.73 | 0.899 | 1.000 | 1.000 |
| Ridge | C + sentiment + attention | 0.0006 | [0.0000, 0.0012] | 2.03 | 1.000 | 1.000 | 1.000 |
| Lasso | C + sentiment + attention | 0.0008 | [0.0002, 0.0014] | 2.57 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + sentiment + attention | 0.0008 | [0.0002, 0.0014] | 2.75 | 0.867 | 1.000 | 1.000 |

### Raw (primary), 2019-2022

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0013 | [0.0006, 0.0020] | 3.60 | 0.046 | 0.023 | 0.008 |
| Ridge | C + sentiment + attention | 0.0016 | [0.0009, 0.0024] | 4.08 | 0.006 | 0.005 | <0.001 |
| Lasso | C + sentiment + attention | 0.0013 | [0.0003, 0.0023] | 2.62 | 1.000 | 1.000 | 1.000 |
| Elastic net | C + sentiment + attention | 0.0014 | [0.0004, 0.0024] | 2.81 | 0.719 | 0.521 | 0.649 |

### DGTW (secondary), 2014-2018

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0012 | [0.0006, 0.0017] | 4.20 | 0.004 | 0.005 | 0.018 |
| Ridge | C + sentiment + attention | 0.0014 | [0.0008, 0.0020] | 4.58 | <0.001 | <0.001 | <0.001 |
| Lasso | C + sentiment + attention | 0.0014 | [0.0008, 0.0020] | 4.45 | 0.001 | 0.001 | 0.003 |
| Elastic net | C + sentiment + attention | 0.0014 | [0.0008, 0.0020] | 4.48 | 0.001 | 0.001 | 0.003 |

### DGTW (secondary), 2019-2022

| Estimator | Inputs | Delta IC | 95% CI | HAC t | Adjusted p (HAC5) | Adjusted p (HAC21) | Adjusted p (HAC63) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | C + sentiment + attention | 0.0015 | [0.0010, 0.0021] | 5.40 | <0.001 | <0.001 | <0.001 |
| Ridge | C + sentiment + attention | 0.0019 | [0.0012, 0.0026] | 5.55 | <0.001 | <0.001 | <0.001 |
| Lasso | C + sentiment + attention | 0.0017 | [0.0009, 0.0026] | 3.93 | 0.012 | 0.015 | 0.024 |
| Elastic net | C + sentiment + attention | 0.0019 | [0.0011, 0.0027] | 4.72 | <0.001 | <0.001 | <0.001 |

## Scope and limitations

This remains historical development evidence: the parent models and evaluation years have already been inspected. Multiplicity adjustment covers the fixed comparisons in this experiment, not the project's full research history or repeated interim looks. Failure to reject a zero difference does not establish that social media contains no information.

The sample consists of the existing social-covered common-stock universe. A C-only benchmark here is not an all-stock benchmark, and the 17 market/past-return variables are not the full stock-characteristic set used in the broader literature. Controls can overlap economically with information conveyed in social media; these predictive comparisons do not identify a causal channel. No news controls are included.

The raw-return target is primary. Although the input block excludes accounting variables without availability dates, the inherited DGTW outcome uses upstream benchmark groups involving book-to-market information. The publication timing and point-in-time availability of those upstream accounting inputs are not certified here. Secondary DGTW comparisons therefore retain that uncertainty; adding current C inputs does not repair it.

Source message routing uses a nominal 16:00 market close and does not repair early-close sessions. Some inputs may arrive after the actual close on those dates. Close-t information also does not establish attainable execution at close t. Gross spreads exclude trading costs, turnover and factor-alpha analysis. Outcome timing is inherited from the corrected parent labels; longer-horizon diagnostics have weaker certification than h=1, and the known corrupted DGTW h=21 series remains excluded.

## Reproducibility and artifacts

The evaluator reuses the unchanged parent economic-key alignment, scoring, calendar handling and HAC code. Its isolated adapter changes only the registered contrast graph and retains the full planned denominators. The metadata records kernel, wrapper, registry, prepared-data and prediction hashes, together with output hashes. The new data preparation preserves the parent's keys and targets; the preparation manifest records feature provenance and missingness rules.

- [Model statistics](<data/characteristics_v1_linear.csv>)
- [Paired contrasts and HAC sensitivities](<data/characteristics_v1_linear_contrasts.csv>)
- [Common prediction and outcome coverage](<data/characteristics_v1_linear_coverage.csv>)
- [Daily series](<data/characteristics_v1_linear_daily.csv.gz>)
- [Yearly statistics](<data/characteristics_v1_linear_yearly.csv>)
- [Fixed subperiod statistics](<data/characteristics_v1_linear_periods.csv>)
- [Decile profiles](<data/characteristics_v1_linear_deciles.csv>)
- [Size and activity diagnostics](<data/characteristics_v1_linear_subgroups.csv>)
- [Cumulative-horizon diagnostics](<data/characteristics_v1_linear_horizons.csv>)

Registry: [registry.json](<../.runs/characteristics_v1/study/d20631f4712e3eb5/linear/full_fc16bf210fe45a57/registry.json>). Evaluation metadata: [characteristics_v1_linear.json](<data/characteristics_v1_linear.json>). Preparation manifest: [manifest.json](<../.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/manifest.json>).
