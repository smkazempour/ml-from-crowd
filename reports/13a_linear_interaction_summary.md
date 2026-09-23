# Explicit interactions: a short reading of the results

All 148 registered procedures completed evaluation: 74 per target.

The study asks whether products such as sentiment times recent return improve prediction. First it allows stock characteristics to have their own curved and interacting relationships, so a social term is not credited merely for filling that gap. Every procedure uses the same message-covered stock-days, two-year training block, six-month validation block and final refit on both.

**What do the joint models add?** The table uses ridge, declared in advance as the primary estimator. Each row compares the named addition with its appropriate smaller model. Positive differences mean better daily stock ranking. Adjusted p-values are two-sided and use the complete family budget; the full report gives effect intervals and longer-lag sensitivity checks.

| Question (ridge) | Delta rank IC | Adjusted p |
| --- | --- | --- |
| Can richer stock characteristics help? | -0.0003 | 1.000 |
| Does sentiment/attention add information after richer C controls? | 0.0010 | <0.001 |
| Do social squares add more? | 0.0002 | 1.000 |
| Do the 35 social interactions help jointly? | -0.0001 | 1.000 |
| Do compressed text main effects help? | -0.0002 | 1.000 |
| Do agreement x C interactions help? | 0.0004 | 1.000 |
| Do embedding-PC x C interactions add more? | -0.0008 | 0.120 |
| Do text x sentiment/attention interactions add more? | -0.0001 | 1.000 |

**Do individual social interactions help?** Of the 35 prespecified individual additions, 0 have positive differences passing the adjusted primary test and 0 have negative differences passing it. Among the positive results, 0 also pass both longer-lag checks; 0 have positive mean differences in both early and late periods. All 35 terms, including unfavorable or undefined results, appear in the full register. Positive early/late signs are descriptive persistence, not separate confirmation. A term's fitted coefficient or nonzero selection alone does not establish predictive value.

**How should text gains be interpreted?** The sequence separates agreement interactions, embedding-PC interactions with characteristics, and interactions with sentiment/attention. A gain in one block is conditional on everything already in its benchmark. Monthly PCs can change meaning; the results do not identify a stable PC topic. Comparisons with additive full text also change the representation, so they cannot isolate interactions alone.

**What remains uncertain?** These are results for the already-inspected 2014-2022 development period, not untouched confirmation. No model is selected by its largest test score. No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim. The characteristic inputs already use daily ranks; raw-versus-ranked inputs are not tested here. Gross portfolio spreads are separate diagnostics and do not establish returns after costs. These results do not test trees or neural networks.

[Full comparisons, all 35 terms and the audit record](<13_linear_interaction_results.md>)
