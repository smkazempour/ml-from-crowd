# Linear optimization: what the results tell us

All 172 registered procedures completed evaluation: 86 for each target.

This study asks whether more suitable penalties, daily characteristic ranks, or stronger text compression improve linear prediction. The fitting schedule stays fixed at two years for training, six months for validation, then a refit on both. It separately asks whether sentiment/attention or text adds information beyond the stock characteristics.

The following table summarizes all registered comparisons using the primary next-day raw-return ranking score. A positive difference means that the named change helped in the paired comparison. The final two columns require a family-adjusted p-value below 0.05; they do not count every small numerical difference as evidence.

| Change | Comparisons scored | Positive mean difference | Positive, adjusted p < .05 | Negative, adjusted p < .05 |
| --- | --- | --- | --- | --- |
| Broader penalty choices | 38 | 6 | 0 | 0 |
| Daily characteristic ranks | 43 | 22 | 0 | 0 |
| Allowing smaller text summaries | 18 | 14 | 0 | 0 |
| Compressing text | 36 | 35 | 12 | 0 |
| Adding social-media inputs | 72 | 38 | 18 | 6 |
| Separate penalties for input groups | 16 | 11 | 4 | 0 |
| Penalization versus OLS | 76 | 75 | 16 | 0 |

These comparisons overlap substantially: many share inputs, forecasts and benchmarks. Their counts are useful for reading the pattern, not as independent votes for a design. The full report provides the size of each effect, uncertainty intervals and sensitivity to longer HAC lags.

**Sentiment and attention:** 14 of 14 registered comparisons have a positive increment passing the primary adjusted test; 13 also pass both longer-lag sensitivity checks. This is evidence for some specified procedures, not proof of a universally superior input set.

**Full text:** 0 of 18 registered comparisons have a positive increment passing the primary adjusted test; 0 also pass both longer-lag sensitivity checks. This study does not establish a positive increment under that primary criterion; it does not prove the inputs contain no information.

**Compressed text:** 0 of 36 registered comparisons have a positive increment passing the primary adjusted test; 0 also pass both longer-lag sensitivity checks. This study does not establish a positive increment under that primary criterion; it does not prove the inputs contain no information.

Improving the overall score and demonstrating incremental social-media information are different outcomes. A better text procedure that only catches up with its no-text comparator establishes neither a unique text contribution nor a globally optimal model.

The 2014–2022 period has already informed project decisions, so these are development results. Chronological training prevents direct future-label leakage but does not make this an untouched confirmation sample. The tables do not select a winner by its largest test score. No 2023 outcomes are scored. Existing 2023 inputs cover only 178 of 250 expected sessions, and prior project records mention predictions through 2023; completeness and prior use must be audited before any confirmation claim.

The same checks also report equal-weighted and capitalization-weighted gross portfolio spreads and the 2014–2018 versus 2019–2022 subperiods. A ranking improvement alone does not establish an improvement in portfolio returns after costs.

[Full results, validation diagnostics and audit record](<11_linear_optimization_results.md>)
