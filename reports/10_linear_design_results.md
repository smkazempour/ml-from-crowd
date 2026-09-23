# Linear design: refitting, group penalties, training history and text compression

All 228 registered linear procedures completed evaluation.

This experiment compares prespecified forecasting procedures on the already-inspected 2014–2022 development period. Its purpose is to diagnose whether the linear design obscures incremental social-media information. It does not identify a globally optimal model from the largest test score. All hyperparameter, window and PCA-dimension choices use earlier validation outcomes; test outcomes are used only for these reported comparisons.

## What changed

C is the same 17 market/past-return characteristics and 17 missingness indicators. The three input sets are C alone (34 columns), C plus sentiment and attention (36), and C plus sentiment, attention and text (422). The latter contains 384 embedding coordinates and two agreement measures, embed_norm and embed_cos. No accounting or news variables are added.

There are 19 base procedures: OLS, ridge, lasso and elastic net on each full input set; group ridge on the two augmented input sets; and all five estimators with PCA text on the text input set. Each is evaluated under two training-history policies and three final-fit rules, separately for raw and DGTW returns: 19 × 2 × 3 × 2 = 228.

| Choice | Definition |
| --- | --- |
| Fixed history | 504 fitting sessions, then 126 validation sessions; monthly forecasts. |
| Selected history | Validation jointly chooses 252, 504 or 756 fitting sessions and the estimator's settings. |
| Retain | Keep the coefficients estimated before validation, as in the previous study. |
| Refit latest F | Keep validation-selected settings, re-estimate using the latest F eligible sessions with matured outcomes. |
| Refit F + 126 | Keep validation-selected settings, re-estimate on the exact original fitting and validation rows, excluding the original purge. At fixed F504 this is 630 sessions, about 2.5 years. |
| Group ridge | Allow different penalties for characteristics, sentiment/attention and text, including the registered exact block-exclusion candidates. |
| PCA text | Compress only the 384 embedding coordinates; validation chooses 16, 32, 64 or 128 components. The two agreement measures stay outside PCA and retain the text group's penalty. |

Signals use information through close t to forecast close t to close t+1. Fitting uses the existing centered-rank target, equal-date squared loss and fitting-only scaling. Validation selects mean daily Spearman IC. Any refit uses only matured labels before the test month. Scaling and PCA are rebuilt on each refit's permitted input observations; the selected dimension and penalty settings stay fixed. For sparse regressions, the selected fraction of the maximum penalty is fixed while its numerical scale is recomputed on the final fitting sample. Test data never enter coefficient fitting or setting selection. Ties retain the declared selection order; no new rule treating close validation scores as equivalent is introduced.

## Common sample and interpretation

| Target | Models | Eligible stock-days | Common predictions | Observed outcomes |
| --- | --- | --- | --- | --- |
| Raw returns (primary) | 114 | 3,034,035 | 3,034,035 | 3,033,080 |
| DGTW returns (secondary) | 114 | 3,034,035 | 3,034,035 | 2,748,978 |

Every procedure within a target is scored on the joint finite-prediction intersection and then observed outcomes. Rank IC is the mean daily Spearman correlation. Predicted rank scores are not percentage-return forecasts. The universe is the existing social-covered stock sample; C-only is not an all-stock benchmark.

## Rank IC by prescribed procedure

Rows follow the registered menu, not the ordering of realized test scores. Each row compares the three final-fit rules. In the selected-history tables, F can differ across forecast months and procedures because validation chooses it separately. The F + 126 rule therefore uses 378, 630 or 882 sessions as selected, while retaining the exact original training/validation row sets.

### Raw returns (primary): Fixed 504

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 0.0675 | 0.0679 | 0.0684 |
| OLS: C + sentiment/attention | 0.0685 | 0.0689 | 0.0694 |
| OLS: C + sentiment/attention + text | 0.0628 | 0.0647 | 0.0652 |
| Ridge: C | 0.0690 | 0.0690 | 0.0694 |
| Ridge: C + sentiment/attention | 0.0700 | 0.0701 | 0.0704 |
| Ridge: C + sentiment/attention + text | 0.0675 | 0.0678 | 0.0683 |
| Lasso: C | 0.0685 | 0.0684 | 0.0688 |
| Lasso: C + sentiment/attention | 0.0695 | 0.0694 | 0.0699 |
| Lasso: C + sentiment/attention + text | 0.0693 | 0.0694 | 0.0698 |
| Elastic net: C | 0.0684 | 0.0683 | 0.0687 |
| Elastic net: C + sentiment/attention | 0.0695 | 0.0693 | 0.0698 |
| Elastic net: C + sentiment/attention + text | 0.0690 | 0.0691 | 0.0695 |
| Group ridge: C + sentiment/attention | 0.0700 | 0.0701 | 0.0704 |
| Group ridge: C + sentiment/attention + text | 0.0698 | 0.0698 | 0.0702 |
| OLS: C + sentiment/attention + text (PCA) | 0.0683 | 0.0687 | 0.0692 |
| Ridge: C + sentiment/attention + text (PCA) | 0.0699 | 0.0699 | 0.0703 |
| Lasso: C + sentiment/attention + text (PCA) | 0.0695 | 0.0695 | 0.0699 |
| Elastic net: C + sentiment/attention + text (PCA) | 0.0697 | 0.0696 | 0.0700 |
| Group ridge: C + sentiment/attention + text (PCA) | 0.0701 | 0.0701 | 0.0704 |

### Raw returns (primary): Validation-selected 252/504/756

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 0.0682 | 0.0685 | 0.0687 |
| OLS: C + sentiment/attention | 0.0692 | 0.0697 | 0.0697 |
| OLS: C + sentiment/attention + text | 0.0644 | 0.0658 | 0.0662 |
| Ridge: C | 0.0692 | 0.0692 | 0.0694 |
| Ridge: C + sentiment/attention | 0.0701 | 0.0702 | 0.0704 |
| Ridge: C + sentiment/attention + text | 0.0681 | 0.0684 | 0.0686 |
| Lasso: C | 0.0689 | 0.0688 | 0.0689 |
| Lasso: C + sentiment/attention | 0.0697 | 0.0697 | 0.0699 |
| Lasso: C + sentiment/attention + text | 0.0696 | 0.0696 | 0.0698 |
| Elastic net: C | 0.0689 | 0.0689 | 0.0690 |
| Elastic net: C + sentiment/attention | 0.0699 | 0.0700 | 0.0701 |
| Elastic net: C + sentiment/attention + text | 0.0699 | 0.0698 | 0.0700 |
| Group ridge: C + sentiment/attention | 0.0702 | 0.0703 | 0.0704 |
| Group ridge: C + sentiment/attention + text | 0.0700 | 0.0700 | 0.0702 |
| OLS: C + sentiment/attention + text (PCA) | 0.0688 | 0.0694 | 0.0695 |
| Ridge: C + sentiment/attention + text (PCA) | 0.0702 | 0.0703 | 0.0705 |
| Lasso: C + sentiment/attention + text (PCA) | 0.0701 | 0.0700 | 0.0703 |
| Elastic net: C + sentiment/attention + text (PCA) | 0.0703 | 0.0701 | 0.0705 |
| Group ridge: C + sentiment/attention + text (PCA) | 0.0702 | 0.0702 | 0.0705 |

### DGTW returns (secondary): Fixed 504

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 0.0593 | 0.0598 | 0.0602 |
| OLS: C + sentiment/attention | 0.0606 | 0.0612 | 0.0615 |
| OLS: C + sentiment/attention + text | 0.0550 | 0.0572 | 0.0574 |
| Ridge: C | 0.0607 | 0.0607 | 0.0610 |
| Ridge: C + sentiment/attention | 0.0623 | 0.0624 | 0.0627 |
| Ridge: C + sentiment/attention + text | 0.0599 | 0.0604 | 0.0608 |
| Lasso: C | 0.0604 | 0.0604 | 0.0608 |
| Lasso: C + sentiment/attention | 0.0619 | 0.0620 | 0.0624 |
| Lasso: C + sentiment/attention + text | 0.0614 | 0.0618 | 0.0620 |
| Elastic net: C | 0.0604 | 0.0605 | 0.0609 |
| Elastic net: C + sentiment/attention | 0.0620 | 0.0621 | 0.0625 |
| Elastic net: C + sentiment/attention + text | 0.0615 | 0.0619 | 0.0621 |
| Group ridge: C + sentiment/attention | 0.0623 | 0.0623 | 0.0627 |
| Group ridge: C + sentiment/attention + text | 0.0621 | 0.0622 | 0.0625 |
| OLS: C + sentiment/attention + text (PCA) | 0.0607 | 0.0611 | 0.0615 |
| Ridge: C + sentiment/attention + text (PCA) | 0.0621 | 0.0622 | 0.0626 |
| Lasso: C + sentiment/attention + text (PCA) | 0.0619 | 0.0621 | 0.0625 |
| Elastic net: C + sentiment/attention + text (PCA) | 0.0619 | 0.0620 | 0.0624 |
| Group ridge: C + sentiment/attention + text (PCA) | 0.0623 | 0.0623 | 0.0626 |

### DGTW returns (secondary): Validation-selected 252/504/756

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 0.0600 | 0.0603 | 0.0605 |
| OLS: C + sentiment/attention | 0.0612 | 0.0616 | 0.0619 |
| OLS: C + sentiment/attention + text | 0.0571 | 0.0582 | 0.0588 |
| Ridge: C | 0.0610 | 0.0611 | 0.0613 |
| Ridge: C + sentiment/attention | 0.0626 | 0.0627 | 0.0629 |
| Ridge: C + sentiment/attention + text | 0.0605 | 0.0609 | 0.0612 |
| Lasso: C | 0.0607 | 0.0609 | 0.0609 |
| Lasso: C + sentiment/attention | 0.0623 | 0.0626 | 0.0626 |
| Lasso: C + sentiment/attention + text | 0.0618 | 0.0623 | 0.0622 |
| Elastic net: C | 0.0607 | 0.0609 | 0.0610 |
| Elastic net: C + sentiment/attention | 0.0623 | 0.0626 | 0.0626 |
| Elastic net: C + sentiment/attention + text | 0.0618 | 0.0624 | 0.0623 |
| Group ridge: C + sentiment/attention | 0.0625 | 0.0626 | 0.0628 |
| Group ridge: C + sentiment/attention + text | 0.0624 | 0.0624 | 0.0627 |
| OLS: C + sentiment/attention + text (PCA) | 0.0609 | 0.0615 | 0.0617 |
| Ridge: C + sentiment/attention + text (PCA) | 0.0624 | 0.0626 | 0.0628 |
| Lasso: C + sentiment/attention + text (PCA) | 0.0622 | 0.0626 | 0.0627 |
| Elastic net: C + sentiment/attention + text (PCA) | 0.0622 | 0.0626 | 0.0627 |
| Group ridge: C + sentiment/attention + text (PCA) | 0.0625 | 0.0625 | 0.0629 |

## Paired raw-return comparisons

Positive delta IC means the named procedure exceeds its registered benchmark. Confidence intervals are pointwise 95% intervals. P-values compare daily paired differences with calendar-aware HAC, then Bonferroni adjustment over the complete family below, separately within target, metric and period. HAC5 is primary; HAC21 and HAC63 are sensitivities. These corrections do not cover earlier research decisions or repeated inspection of this development period.

| Family | Comparisons per target / metric / period |
| --- | --- |
| refit | 114 |
| history | 57 |
| group_penalty | 18 |
| compression | 30 |
| incremental_social | 90 |
| estimator | 90 |

The tables below show fixed-504 comparisons for refitting, penalties, compression, social additions and estimators; the history table compares selected against fixed history under every refit. The full contrast artifact also contains all selected-history comparisons, DGTW comparisons, gross-spread differences and the two fixed subperiods. No unfinished or undefined comparison reduces a multiplicity denominator.

### Does re-estimating coefficients help?

Recent F minus retained coefficients; union F+126 minus retained; union minus recent, matched in every other design choice.

| Procedure | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| OLS: C | Latest F minus retained | 0.0004 | [-0.0002, 0.0011] | 1.000 | 1.000 | 1.000 |
| OLS: C | F + 126 minus retained | 0.0009 | [0.0004, 0.0013] | 0.023 | 0.006 | 0.008 |
| OLS: C | F + 126 minus latest F | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Ridge: C | Latest F minus retained | 0.0001 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Ridge: C | F + 126 minus retained | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Ridge: C | F + 126 minus latest F | 0.0003 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Latest F minus retained | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: C | F + 126 minus retained | 0.0004 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: C | F + 126 minus latest F | 0.0004 | [0.0000, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Latest F minus retained | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | F + 126 minus retained | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | F + 126 minus latest F | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention | Latest F minus retained | 0.0004 | [-0.0002, 0.0010] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention | F + 126 minus retained | 0.0009 | [0.0004, 0.0013] | 0.013 | 0.002 | 0.002 |
| OLS: C + sentiment/attention | F + 126 minus latest F | 0.0005 | [0.0001, 0.0009] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Latest F minus retained | 0.0000 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | F + 126 minus retained | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | F + 126 minus latest F | 0.0003 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Latest F minus retained | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | F + 126 minus retained | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | F + 126 minus latest F | 0.0005 | [0.0001, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Latest F minus retained | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | F + 126 minus retained | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | F + 126 minus latest F | 0.0005 | [0.0001, 0.0009] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Latest F minus retained | 0.0001 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | F + 126 minus retained | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | F + 126 minus latest F | 0.0003 | [0.0000, 0.0006] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text | Latest F minus retained | 0.0019 | [0.0010, 0.0029] | 0.008 | 0.004 | 0.005 |
| OLS: C + sentiment/attention + text | F + 126 minus retained | 0.0025 | [0.0017, 0.0032] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text | F + 126 minus latest F | 0.0005 | [0.0000, 0.0011] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | Latest F minus retained | 0.0003 | [-0.0003, 0.0009] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | F + 126 minus retained | 0.0008 | [0.0004, 0.0012] | 0.024 | 0.014 | 0.005 |
| Ridge: C + sentiment/attention + text | F + 126 minus latest F | 0.0005 | [0.0002, 0.0009] | 0.483 | 0.886 | 1.000 |
| Lasso: C + sentiment/attention + text | Latest F minus retained | 0.0002 | [-0.0005, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | F + 126 minus retained | 0.0005 | [0.0001, 0.0009] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | F + 126 minus latest F | 0.0004 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Latest F minus retained | 0.0001 | [-0.0005, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | F + 126 minus retained | 0.0005 | [0.0001, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | F + 126 minus latest F | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Latest F minus retained | 0.0000 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | F + 126 minus retained | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | F + 126 minus latest F | 0.0003 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Latest F minus retained | 0.0004 | [-0.0003, 0.0011] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | F + 126 minus retained | 0.0009 | [0.0004, 0.0014] | 0.027 | 0.006 | 0.002 |
| OLS: C + sentiment/attention + text (PCA) | F + 126 minus latest F | 0.0005 | [0.0001, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Latest F minus retained | 0.0000 | [-0.0006, 0.0005] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | F + 126 minus retained | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | F + 126 minus latest F | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Latest F minus retained | -0.0001 | [-0.0007, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | F + 126 minus retained | 0.0003 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | F + 126 minus latest F | 0.0004 | [0.0000, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Latest F minus retained | -0.0001 | [-0.0007, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | F + 126 minus retained | 0.0003 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | F + 126 minus latest F | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Latest F minus retained | -0.0001 | [-0.0006, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | F + 126 minus retained | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | F + 126 minus latest F | 0.0004 | [0.0000, 0.0007] | 1.000 | 1.000 | 1.000 |

### Does validation-selected history help?

Validation-selected F252/F504/F756 minus fixed F504, matched in every other design choice.

| Procedure | Final fit | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS: C | Retain old coefficients | Selected window minus fixed 504 | 0.0007 | [0.0001, 0.0012] | 0.906 | 0.598 | 0.456 |
| OLS: C | Refit latest F | Selected window minus fixed 504 | 0.0006 | [0.0000, 0.0011] | 1.000 | 1.000 | 1.000 |
| OLS: C | Refit F + 126 | Selected window minus fixed 504 | 0.0003 | [-0.0001, 0.0008] | 1.000 | 1.000 | 1.000 |
| Ridge: C | Retain old coefficients | Selected window minus fixed 504 | 0.0002 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C | Refit latest F | Selected window minus fixed 504 | 0.0002 | [-0.0004, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C | Refit F + 126 | Selected window minus fixed 504 | 0.0000 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Retain old coefficients | Selected window minus fixed 504 | 0.0004 | [-0.0002, 0.0010] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Refit latest F | Selected window minus fixed 504 | 0.0004 | [-0.0003, 0.0010] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Refit F + 126 | Selected window minus fixed 504 | 0.0001 | [-0.0005, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Retain old coefficients | Selected window minus fixed 504 | 0.0005 | [-0.0001, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Refit latest F | Selected window minus fixed 504 | 0.0007 | [0.0000, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Refit F + 126 | Selected window minus fixed 504 | 0.0003 | [-0.0003, 0.0009] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention | Retain old coefficients | Selected window minus fixed 504 | 0.0007 | [0.0001, 0.0012] | 0.741 | 0.459 | 0.379 |
| OLS: C + sentiment/attention | Refit latest F | Selected window minus fixed 504 | 0.0008 | [0.0002, 0.0014] | 0.343 | 0.097 | 0.132 |
| OLS: C + sentiment/attention | Refit F + 126 | Selected window minus fixed 504 | 0.0004 | [0.0000, 0.0008] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Retain old coefficients | Selected window minus fixed 504 | 0.0001 | [-0.0004, 0.0005] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Refit latest F | Selected window minus fixed 504 | 0.0001 | [-0.0004, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Refit F + 126 | Selected window minus fixed 504 | 0.0000 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Retain old coefficients | Selected window minus fixed 504 | 0.0002 | [-0.0004, 0.0008] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Refit latest F | Selected window minus fixed 504 | 0.0003 | [-0.0004, 0.0010] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Refit F + 126 | Selected window minus fixed 504 | 0.0000 | [-0.0006, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Retain old coefficients | Selected window minus fixed 504 | 0.0005 | [-0.0002, 0.0011] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Refit latest F | Selected window minus fixed 504 | 0.0007 | [0.0000, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Refit F + 126 | Selected window minus fixed 504 | 0.0003 | [-0.0003, 0.0008] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Retain old coefficients | Selected window minus fixed 504 | 0.0002 | [-0.0002, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Refit latest F | Selected window minus fixed 504 | 0.0002 | [-0.0003, 0.0008] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Refit F + 126 | Selected window minus fixed 504 | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text | Retain old coefficients | Selected window minus fixed 504 | 0.0016 | [0.0009, 0.0022] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text | Refit latest F | Selected window minus fixed 504 | 0.0012 | [0.0005, 0.0018] | 0.047 | 0.020 | 0.069 |
| OLS: C + sentiment/attention + text | Refit F + 126 | Selected window minus fixed 504 | 0.0009 | [0.0004, 0.0015] | 0.030 | 0.006 | 0.003 |
| Ridge: C + sentiment/attention + text | Retain old coefficients | Selected window minus fixed 504 | 0.0006 | [0.0001, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | Refit latest F | Selected window minus fixed 504 | 0.0006 | [0.0001, 0.0012] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | Refit F + 126 | Selected window minus fixed 504 | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | Retain old coefficients | Selected window minus fixed 504 | 0.0003 | [-0.0003, 0.0009] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | Refit latest F | Selected window minus fixed 504 | 0.0002 | [-0.0005, 0.0009] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | Refit F + 126 | Selected window minus fixed 504 | 0.0001 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Retain old coefficients | Selected window minus fixed 504 | 0.0009 | [0.0002, 0.0015] | 0.538 | 0.245 | 0.166 |
| Elastic net: C + sentiment/attention + text | Refit latest F | Selected window minus fixed 504 | 0.0007 | [-0.0001, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Refit F + 126 | Selected window minus fixed 504 | 0.0005 | [0.0000, 0.0011] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Retain old coefficients | Selected window minus fixed 504 | 0.0001 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Refit latest F | Selected window minus fixed 504 | 0.0002 | [-0.0004, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Refit F + 126 | Selected window minus fixed 504 | 0.0000 | [-0.0004, 0.0004] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Retain old coefficients | Selected window minus fixed 504 | 0.0005 | [-0.0001, 0.0011] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Refit latest F | Selected window minus fixed 504 | 0.0007 | [0.0002, 0.0013] | 0.547 | 0.366 | 0.557 |
| OLS: C + sentiment/attention + text (PCA) | Refit F + 126 | Selected window minus fixed 504 | 0.0002 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | Selected window minus fixed 504 | 0.0003 | [-0.0002, 0.0008] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Refit latest F | Selected window minus fixed 504 | 0.0004 | [-0.0001, 0.0010] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | Selected window minus fixed 504 | 0.0002 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Retain old coefficients | Selected window minus fixed 504 | 0.0006 | [-0.0001, 0.0012] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit latest F | Selected window minus fixed 504 | 0.0005 | [-0.0002, 0.0012] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit F + 126 | Selected window minus fixed 504 | 0.0005 | [-0.0001, 0.0010] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Retain old coefficients | Selected window minus fixed 504 | 0.0006 | [0.0000, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit latest F | Selected window minus fixed 504 | 0.0005 | [-0.0002, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit F + 126 | Selected window minus fixed 504 | 0.0005 | [-0.0001, 0.0010] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | Selected window minus fixed 504 | 0.0001 | [-0.0003, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit latest F | Selected window minus fixed 504 | 0.0001 | [-0.0004, 0.0007] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | Selected window minus fixed 504 | 0.0000 | [-0.0004, 0.0004] | 1.000 | 1.000 | 1.000 |

### Do separate penalties help?

Separate group penalties minus global ridge, at matched inputs, representation, history and refit.

| Procedure | Final fit | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Group ridge: C + sentiment/attention | Retain old coefficients | Group penalties minus global ridge | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Refit latest F | Group penalties minus global ridge | 0.0000 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Refit F + 126 | Group penalties minus global ridge | -0.0001 | [-0.0002, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Retain old coefficients | Group penalties minus global ridge | 0.0023 | [0.0015, 0.0031] | <0.001 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text | Refit latest F | Group penalties minus global ridge | 0.0021 | [0.0013, 0.0029] | <0.001 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text | Refit F + 126 | Group penalties minus global ridge | 0.0019 | [0.0011, 0.0026] | <0.001 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | Group penalties minus global ridge | 0.0002 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit latest F | Group penalties minus global ridge | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | Group penalties minus global ridge | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |

### Does text compression help?

Validation-selected text PCA dimension minus full text, at matched estimator, inputs, history and refit.

| Procedure | Final fit | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text minus full text | 0.0056 | [0.0041, 0.0070] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text (PCA) | Refit latest F | PCA text minus full text | 0.0040 | [0.0029, 0.0052] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text minus full text | 0.0040 | [0.0029, 0.0052] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text minus full text | 0.0024 | [0.0017, 0.0031] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text (PCA) | Refit latest F | PCA text minus full text | 0.0022 | [0.0015, 0.0029] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text minus full text | 0.0020 | [0.0013, 0.0026] | <0.001 | <0.001 | <0.001 |
| Lasso: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text minus full text | 0.0003 | [-0.0001, 0.0007] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit latest F | PCA text minus full text | 0.0001 | [-0.0003, 0.0005] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text minus full text | 0.0001 | [-0.0003, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text minus full text | 0.0007 | [0.0002, 0.0011] | 0.104 | 0.126 | 0.403 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit latest F | PCA text minus full text | 0.0005 | [0.0000, 0.0009] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text minus full text | 0.0005 | [0.0000, 0.0009] | 0.965 | 0.856 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text minus full text | 0.0003 | [0.0000, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit latest F | PCA text minus full text | 0.0002 | [-0.0001, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text minus full text | 0.0003 | [0.0000, 0.0006] | 1.000 | 1.000 | 1.000 |

### What does social media add beyond characteristics?

C+core minus C; C+full-text+core minus C+core; C+PCA-text+core minus C+core. Group C+core uses global-ridge C as its characteristic-only reference.

Global estimators compare each augmented input set with the separately validation-tuned smaller procedure. Group C + sentiment/attention versus ridge C combines the social addition with freedom to penalize feature groups separately; it is not a pure coefficient-controlled addition. The two grouped text comparisons use grouped C + sentiment/attention as their reference. Exact exclusion candidates allow the larger search to reproduce smaller input blocks, but validation-selection noise can still lower test performance.

| Procedure | Final fit | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS: C + sentiment/attention | Retain old coefficients | Sentiment/attention given C | 0.0010 | [0.0005, 0.0014] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention | Refit latest F | Sentiment/attention given C | 0.0009 | [0.0005, 0.0013] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention | Refit F + 126 | Sentiment/attention given C | 0.0010 | [0.0006, 0.0014] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention | Retain old coefficients | Sentiment/attention given C | 0.0011 | [0.0006, 0.0016] | 0.001 | 0.003 | 0.001 |
| Ridge: C + sentiment/attention | Refit latest F | Sentiment/attention given C | 0.0011 | [0.0006, 0.0015] | 0.003 | 0.004 | <0.001 |
| Ridge: C + sentiment/attention | Refit F + 126 | Sentiment/attention given C | 0.0010 | [0.0006, 0.0015] | 0.002 | 0.003 | <0.001 |
| Lasso: C + sentiment/attention | Retain old coefficients | Sentiment/attention given C | 0.0010 | [0.0005, 0.0016] | 0.025 | 0.028 | 0.059 |
| Lasso: C + sentiment/attention | Refit latest F | Sentiment/attention given C | 0.0010 | [0.0005, 0.0016] | 0.027 | 0.032 | 0.127 |
| Lasso: C + sentiment/attention | Refit F + 126 | Sentiment/attention given C | 0.0011 | [0.0005, 0.0016] | 0.010 | 0.009 | 0.037 |
| Elastic net: C + sentiment/attention | Retain old coefficients | Sentiment/attention given C | 0.0011 | [0.0005, 0.0016] | 0.009 | 0.007 | 0.014 |
| Elastic net: C + sentiment/attention | Refit latest F | Sentiment/attention given C | 0.0011 | [0.0005, 0.0016] | 0.015 | 0.015 | 0.058 |
| Elastic net: C + sentiment/attention | Refit F + 126 | Sentiment/attention given C | 0.0011 | [0.0006, 0.0017] | 0.004 | 0.003 | 0.011 |
| Group ridge: C + sentiment/attention | Retain old coefficients | Grouped sentiment/attention + C minus ridge C | 0.0010 | [0.0006, 0.0015] | <0.001 | 0.002 | <0.001 |
| Group ridge: C + sentiment/attention | Refit latest F | Grouped sentiment/attention + C minus ridge C | 0.0010 | [0.0006, 0.0015] | <0.001 | 0.002 | <0.001 |
| Group ridge: C + sentiment/attention | Refit F + 126 | Grouped sentiment/attention + C minus ridge C | 0.0010 | [0.0005, 0.0014] | <0.001 | 0.002 | <0.001 |
| OLS: C + sentiment/attention + text | Retain old coefficients | Full text given C + sentiment/attention | -0.0057 | [-0.0073, -0.0041] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text | Refit latest F | Full text given C + sentiment/attention | -0.0042 | [-0.0055, -0.0029] | <0.001 | <0.001 | <0.001 |
| OLS: C + sentiment/attention + text | Refit F + 126 | Full text given C + sentiment/attention | -0.0041 | [-0.0054, -0.0028] | <0.001 | <0.001 | 0.001 |
| Ridge: C + sentiment/attention + text | Retain old coefficients | Full text given C + sentiment/attention | -0.0026 | [-0.0034, -0.0017] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text | Refit latest F | Full text given C + sentiment/attention | -0.0023 | [-0.0032, -0.0015] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text | Refit F + 126 | Full text given C + sentiment/attention | -0.0021 | [-0.0029, -0.0013] | <0.001 | <0.001 | <0.001 |
| Lasso: C + sentiment/attention + text | Retain old coefficients | Full text given C + sentiment/attention | -0.0002 | [-0.0009, 0.0004] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | Refit latest F | Full text given C + sentiment/attention | 0.0000 | [-0.0006, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text | Refit F + 126 | Full text given C + sentiment/attention | -0.0001 | [-0.0008, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Retain old coefficients | Full text given C + sentiment/attention | -0.0005 | [-0.0010, 0.0001] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Refit latest F | Full text given C + sentiment/attention | -0.0002 | [-0.0008, 0.0003] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text | Refit F + 126 | Full text given C + sentiment/attention | -0.0003 | [-0.0008, 0.0002] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Retain old coefficients | Full text given C + sentiment/attention | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Refit latest F | Full text given C + sentiment/attention | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text | Refit F + 126 | Full text given C + sentiment/attention | -0.0002 | [-0.0005, 0.0001] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text given C + sentiment/attention | -0.0002 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Refit latest F | PCA text given C + sentiment/attention | -0.0002 | [-0.0006, 0.0002] | 1.000 | 1.000 | 1.000 |
| OLS: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text given C + sentiment/attention | -0.0001 | [-0.0005, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text given C + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Refit latest F | PCA text given C + sentiment/attention | -0.0002 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text given C + sentiment/attention | -0.0001 | [-0.0006, 0.0003] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text given C + sentiment/attention | 0.0000 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit latest F | PCA text given C + sentiment/attention | 0.0001 | [-0.0005, 0.0006] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text given C + sentiment/attention | 0.0000 | [-0.0005, 0.0005] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text given C + sentiment/attention | 0.0002 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit latest F | PCA text given C + sentiment/attention | 0.0002 | [-0.0002, 0.0007] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text given C + sentiment/attention | 0.0002 | [-0.0003, 0.0006] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | PCA text given C + sentiment/attention | 0.0001 | [-0.0002, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit latest F | PCA text given C + sentiment/attention | 0.0000 | [-0.0004, 0.0004] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | PCA text given C + sentiment/attention | 0.0001 | [-0.0003, 0.0004] | 1.000 | 1.000 | 1.000 |

### Do penalized estimators improve on OLS?

Ridge, lasso, elastic net and group ridge minus OLS, at matched inputs, representation, history and refit.

| Procedure | Final fit | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge: C | Retain old coefficients | Penalized model minus OLS | 0.0015 | [0.0005, 0.0024] | 0.183 | 0.333 | 1.000 |
| Ridge: C | Refit latest F | Penalized model minus OLS | 0.0011 | [0.0002, 0.0020] | 1.000 | 1.000 | 1.000 |
| Ridge: C | Refit F + 126 | Penalized model minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Retain old coefficients | Penalized model minus OLS | 0.0010 | [0.0002, 0.0018] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Refit latest F | Penalized model minus OLS | 0.0004 | [-0.0003, 0.0012] | 1.000 | 1.000 | 1.000 |
| Lasso: C | Refit F + 126 | Penalized model minus OLS | 0.0005 | [-0.0003, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Retain old coefficients | Penalized model minus OLS | 0.0009 | [0.0000, 0.0017] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Refit latest F | Penalized model minus OLS | 0.0003 | [-0.0005, 0.0012] | 1.000 | 1.000 | 1.000 |
| Elastic net: C | Refit F + 126 | Penalized model minus OLS | 0.0003 | [-0.0006, 0.0011] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Retain old coefficients | Penalized model minus OLS | 0.0016 | [0.0006, 0.0026] | 0.153 | 0.246 | 0.599 |
| Ridge: C + sentiment/attention | Refit latest F | Penalized model minus OLS | 0.0012 | [0.0003, 0.0021] | 0.810 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention | Refit F + 126 | Penalized model minus OLS | 0.0011 | [0.0002, 0.0020] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Retain old coefficients | Penalized model minus OLS | 0.0010 | [0.0002, 0.0019] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Refit latest F | Penalized model minus OLS | 0.0005 | [-0.0003, 0.0014] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention | Refit F + 126 | Penalized model minus OLS | 0.0005 | [-0.0003, 0.0014] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Retain old coefficients | Penalized model minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Refit latest F | Penalized model minus OLS | 0.0004 | [-0.0004, 0.0013] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention | Refit F + 126 | Penalized model minus OLS | 0.0004 | [-0.0005, 0.0013] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Retain old coefficients | Penalized model minus OLS | 0.0015 | [0.0005, 0.0025] | 0.239 | 0.369 | 0.856 |
| Group ridge: C + sentiment/attention | Refit latest F | Penalized model minus OLS | 0.0012 | [0.0003, 0.0021] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Refit F + 126 | Penalized model minus OLS | 0.0010 | [0.0001, 0.0019] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | Retain old coefficients | Penalized model minus OLS | 0.0047 | [0.0030, 0.0065] | <0.001 | <0.001 | 0.005 |
| Ridge: C + sentiment/attention + text | Refit latest F | Penalized model minus OLS | 0.0031 | [0.0016, 0.0046] | 0.005 | 0.009 | 0.086 |
| Ridge: C + sentiment/attention + text | Refit F + 126 | Penalized model minus OLS | 0.0031 | [0.0016, 0.0046] | 0.003 | 0.008 | 0.109 |
| Lasso: C + sentiment/attention + text | Retain old coefficients | Penalized model minus OLS | 0.0065 | [0.0045, 0.0085] | <0.001 | <0.001 | <0.001 |
| Lasso: C + sentiment/attention + text | Refit latest F | Penalized model minus OLS | 0.0047 | [0.0030, 0.0064] | <0.001 | <0.001 | <0.001 |
| Lasso: C + sentiment/attention + text | Refit F + 126 | Penalized model minus OLS | 0.0045 | [0.0028, 0.0063] | <0.001 | <0.001 | 0.002 |
| Elastic net: C + sentiment/attention + text | Retain old coefficients | Penalized model minus OLS | 0.0062 | [0.0042, 0.0083] | <0.001 | <0.001 | <0.001 |
| Elastic net: C + sentiment/attention + text | Refit latest F | Penalized model minus OLS | 0.0044 | [0.0027, 0.0062] | <0.001 | <0.001 | 0.005 |
| Elastic net: C + sentiment/attention + text | Refit F + 126 | Penalized model minus OLS | 0.0043 | [0.0025, 0.0060] | <0.001 | <0.001 | 0.010 |
| Group ridge: C + sentiment/attention + text | Retain old coefficients | Penalized model minus OLS | 0.0071 | [0.0050, 0.0091] | <0.001 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text | Refit latest F | Penalized model minus OLS | 0.0052 | [0.0034, 0.0069] | <0.001 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text | Refit F + 126 | Penalized model minus OLS | 0.0049 | [0.0032, 0.0067] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | Penalized model minus OLS | 0.0016 | [0.0006, 0.0027] | 0.212 | 0.332 | 0.654 |
| Ridge: C + sentiment/attention + text (PCA) | Refit latest F | Penalized model minus OLS | 0.0012 | [0.0003, 0.0022] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | Penalized model minus OLS | 0.0011 | [0.0001, 0.0020] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Retain old coefficients | Penalized model minus OLS | 0.0012 | [0.0003, 0.0022] | 1.000 | 1.000 | 0.878 |
| Lasso: C + sentiment/attention + text (PCA) | Refit latest F | Penalized model minus OLS | 0.0008 | [-0.0001, 0.0016] | 1.000 | 1.000 | 1.000 |
| Lasso: C + sentiment/attention + text (PCA) | Refit F + 126 | Penalized model minus OLS | 0.0007 | [-0.0002, 0.0016] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Retain old coefficients | Penalized model minus OLS | 0.0013 | [0.0004, 0.0023] | 0.627 | 0.841 | 0.439 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit latest F | Penalized model minus OLS | 0.0009 | [0.0000, 0.0018] | 1.000 | 1.000 | 1.000 |
| Elastic net: C + sentiment/attention + text (PCA) | Refit F + 126 | Penalized model minus OLS | 0.0007 | [-0.0002, 0.0017] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Retain old coefficients | Penalized model minus OLS | 0.0018 | [0.0007, 0.0029] | 0.075 | 0.154 | 0.372 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit latest F | Penalized model minus OLS | 0.0014 | [0.0004, 0.0024] | 0.623 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | Refit F + 126 | Penalized model minus OLS | 0.0012 | [0.0002, 0.0022] | 1.000 | 1.000 | 1.000 |

## Temporal stability of the incremental social signal

These prespecified ridge and group-ridge contrasts use fixed 504-session history and retained coefficients in each fixed subperiod. They are descriptive stability checks; no subperiod chooses a model or trading regime. Complete subperiod results for every procedure are in the contrast artifact.

### 2014-2018

| Procedure | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge: C + sentiment/attention | Sentiment/attention given C | 0.0006 | [0.0000, 0.0012] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention | Grouped sentiment/attention + C minus ridge C | 0.0006 | [0.0000, 0.0011] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text | Full text given C + sentiment/attention | -0.0028 | [-0.0041, -0.0015] | 0.003 | <0.001 | <0.001 |
| Group ridge: C + sentiment/attention + text | Full text given C + sentiment/attention | -0.0002 | [-0.0006, 0.0002] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | PCA text given C + sentiment/attention | -0.0002 | [-0.0009, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | PCA text given C + sentiment/attention | 0.0003 | [-0.0002, 0.0008] | 1.000 | 1.000 | 1.000 |

### 2019-2022

| Procedure | Comparison | Delta IC | 95% CI | Adjusted p, HAC5 | Adjusted p, HAC21 | Adjusted p, HAC63 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge: C + sentiment/attention | Sentiment/attention given C | 0.0016 | [0.0009, 0.0024] | 0.004 | 0.003 | <0.001 |
| Group ridge: C + sentiment/attention | Grouped sentiment/attention + C minus ridge C | 0.0016 | [0.0009, 0.0023] | <0.001 | <0.001 | <0.001 |
| Ridge: C + sentiment/attention + text | Full text given C + sentiment/attention | -0.0023 | [-0.0034, -0.0011] | 0.007 | 0.047 | 0.246 |
| Group ridge: C + sentiment/attention + text | Full text given C + sentiment/attention | -0.0001 | [-0.0005, 0.0002] | 1.000 | 1.000 | 1.000 |
| Ridge: C + sentiment/attention + text (PCA) | PCA text given C + sentiment/attention | 0.0000 | [-0.0005, 0.0005] | 1.000 | 1.000 | 1.000 |
| Group ridge: C + sentiment/attention + text (PCA) | PCA text given C + sentiment/attention | 0.0000 | [-0.0004, 0.0003] | 1.000 | 1.000 | 1.000 |

## Gross portfolio diagnostics

For raw returns and fixed 504-session history, the following tables show average daily top-minus-bottom decile returns in basis points. These use fractional tie handling and the same common samples. They are gross descriptive sorts, not attainable net trading returns. Turnover, transaction costs, factor alpha and execution constraints have not been estimated.

### Equal-weighted spread

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 22.19 | 22.84 | 23.12 |
| OLS: C + sentiment/attention | 25.59 | 24.73 | 25.10 |
| OLS: C + sentiment/attention + text | 25.52 | 26.12 | 25.54 |
| Ridge: C | 19.87 | 19.67 | 20.22 |
| Ridge: C + sentiment/attention | 23.73 | 23.52 | 24.47 |
| Ridge: C + sentiment/attention + text | 22.69 | 23.73 | 23.10 |
| Lasso: C | 20.55 | 20.52 | 20.27 |
| Lasso: C + sentiment/attention | 23.13 | 22.98 | 24.18 |
| Lasso: C + sentiment/attention + text | 23.12 | 24.08 | 23.97 |
| Elastic net: C | 20.46 | 20.61 | 20.35 |
| Elastic net: C + sentiment/attention | 23.42 | 23.45 | 23.86 |
| Elastic net: C + sentiment/attention + text | 23.15 | 23.33 | 23.48 |
| Group ridge: C + sentiment/attention | 23.59 | 23.86 | 24.17 |
| Group ridge: C + sentiment/attention + text | 23.11 | 23.51 | 23.99 |
| OLS: C + sentiment/attention + text (PCA) | 26.47 | 26.80 | 27.08 |
| Ridge: C + sentiment/attention + text (PCA) | 24.75 | 25.13 | 25.41 |
| Lasso: C + sentiment/attention + text (PCA) | 24.68 | 25.03 | 25.84 |
| Elastic net: C + sentiment/attention + text (PCA) | 24.88 | 24.72 | 25.64 |
| Group ridge: C + sentiment/attention + text (PCA) | 24.01 | 24.61 | 24.89 |

### Capitalization-weighted spread

| Procedure | Retain old coefficients | Refit latest F | Refit F + 126 |
| --- | --- | --- | --- |
| OLS: C | 11.50 | 11.75 | 12.51 |
| OLS: C + sentiment/attention | 13.43 | 12.25 | 10.00 |
| OLS: C + sentiment/attention + text | 9.74 | 9.31 | 7.21 |
| Ridge: C | 14.69 | 14.99 | 15.59 |
| Ridge: C + sentiment/attention | 15.42 | 14.32 | 14.70 |
| Ridge: C + sentiment/attention + text | 11.68 | 11.32 | 11.42 |
| Lasso: C | 16.48 | 13.75 | 15.49 |
| Lasso: C + sentiment/attention | 15.18 | 14.51 | 16.86 |
| Lasso: C + sentiment/attention + text | 8.83 | 13.24 | 13.05 |
| Elastic net: C | 16.55 | 14.77 | 15.33 |
| Elastic net: C + sentiment/attention | 15.33 | 15.10 | 15.96 |
| Elastic net: C + sentiment/attention + text | 8.95 | 12.52 | 10.77 |
| Group ridge: C + sentiment/attention | 13.41 | 14.91 | 13.82 |
| Group ridge: C + sentiment/attention + text | 13.11 | 13.63 | 12.69 |
| OLS: C + sentiment/attention + text (PCA) | 12.44 | 11.71 | 10.41 |
| Ridge: C + sentiment/attention + text (PCA) | 14.54 | 12.24 | 13.53 |
| Lasso: C + sentiment/attention + text (PCA) | 11.95 | 14.61 | 16.26 |
| Elastic net: C + sentiment/attention + text (PCA) | 12.84 | 15.00 | 14.40 |
| Group ridge: C + sentiment/attention + text (PCA) | 13.97 | 14.99 | 14.28 |

## Development evidence and limits

The 2014–2022 years have already informed project decisions. Chronological fitting prevents direct future-label leakage, but it cannot undo research decisions made after viewing these historical results. These comparisons are development evidence rather than untouched confirmation, and failure to reject an incremental gain does not establish that the text contains no information.

2023 is not ready to serve as an untouched confirmation sample. The current input audit found 178 of 250 expected trading dates, with 72 missing dates in September–December, including all of October. Earlier project records also mention predictions through 2023. A separate audit of completeness and prior use is required, or additional data must be obtained. No 2023 model outcomes are scored in this experiment.

Only market-based characteristics and past returns are controlled for. Reliable accounting publication timing is unavailable, so accounting predictors remain excluded. The inherited DGTW target is secondary because its upstream characteristic sorts have their own accounting-availability caveat. No news controls or causal interpretation are supported by these comparisons.

The test sample does not choose one winning procedure. Any next-stage decision should use the registered paired comparisons, their temporal stability and computational costs, followed by a separately defensible confirmation design. No neural networks are fitted in this experiment.

## Reproducible artifacts

Evaluation uses the unchanged scoring, common-sample alignment, calendar and HAC kernels. New scoped adapters supply only the registered comparisons and procedure metadata. SHA-256 digests record the registry, prepared data, prediction files, reused kernels and every evaluation output. The daily table is published as deterministic gzip; the uncompressed copy remains local.

- [summary](<data/linear_design_v2.csv>)
- [_daily](<data/linear_design_v2_daily.csv.gz>)
- [_yearly](<data/linear_design_v2_yearly.csv>)
- [_periods](<data/linear_design_v2_periods.csv>)
- [_contrasts](<data/linear_design_v2_contrasts.csv>)
- [_coverage](<data/linear_design_v2_coverage.csv>)
- [_deciles](<data/linear_design_v2_deciles.csv>)
- [_subgroups](<data/linear_design_v2_subgroups.csv>)
- [_horizons](<data/linear_design_v2_horizons.csv>)

- [Evaluation metadata and hashes](<data/linear_design_v2.json>)

- [Model registry (local run artifact)](<../.runs/linear_design_v2/study/c11478141261534f/linear/full_9590a962af3e2fcb/registry.json>)

- [Frozen experiment specification](<LINEAR_DESIGN_EXPERIMENT.md>)
