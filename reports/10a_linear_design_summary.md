# What we learned from the linear design study

The revised models perform better, but the main conclusion remains: **stock
characteristics and past returns provide a strong starting point; sentiment and
attention add a small amount; the current text features do not yet show a clear
additional benefit.** Better estimation reduces the harm from adding many text
features, without establishing that those features improve prediction beyond
the simpler social-media measures.

This is a reading guide to [Report 10](10_linear_design_results.md), which retains
the full results and statistical tests. We compared 228 procedures over 2014–2022.
The discussion below focuses on the primary outcome: ranking stocks by their
next-day raw returns.

**First, sentiment and attention still help after controlling for characteristics.**
Across the four ordinary estimators with the fixed training window, adding these
two variables raises the average daily rank correlation by about 0.001. These
gains pass the primary statistical tests after adjustment for the many comparisons.
They are small gains over an already informative characteristic-only benchmark.
The evidence is stronger in 2019–2022 than in the earlier years.

Here is one compact illustration. These models use two years for initial training,
six months for validation, and then refit on both samples. The score is the average
correlation between predicted and realized stock rankings; higher is better.
It is not a percentage return or prediction-accuracy rate.

| Inputs and treatment | Ranking score |
| --- | ---: |
| Characteristics and past returns, ridge | 0.0694 |
| Add sentiment and attention, ridge | 0.0704 |
| Also add all text features, ridge | 0.0683 |
| Allow separate penalties for text and other inputs | 0.0702 |
| Compress the text features with PCA, ridge | 0.0703 |

The last two rows are alternative treatments of text, not cumulative changes.

**Second, the way we handle text matters—but improving a text model does not prove
that text adds information.** Adding hundreds of text features can make predictions
worse. Allowing stronger penalties on text, or compressing it into fewer summary
variables, largely repairs that loss. This is consistent with noisy coefficient
estimation being part of the problem. Some improved text procedures slightly
exceed their simpler benchmarks, but none of the positive text increments passes
the adjusted full-period tests for raw-return ranking. That is an absence of
convincing evidence in this setup, not proof that social-media text is uninformative.
Suppressing unhelpful inputs can improve a model without extracting a new signal.

**Third, refitting on training plus validation was useful.** For OLS with
characteristics, sentiment and attention, the score moves from **0.0685** with the
old coefficients to **0.0689** after refitting on the latest two years, and to
**0.0694** after the requested 2.5-year refit. The last improvement over the old
coefficients survives the adjusted tests and the longer statistical dependence
checks. More generally, the combined-sample refit raises average ranking scores
across the matched procedures. However, its advantage over the latest-window
refit is small: none of those differences passes the multiple-comparison adjustment.
It is a sensible working choice, not a proven universally superior window.

**Fourth, penalized regressions do help, particularly with all the text features.**
With the same 2.5-year refit and full text inputs, OLS scores **0.0652**, while lasso
scores **0.0698**. That gain is statistically clear, including under the longer
dependence checks. Penalized methods also have higher average scores than matched
OLS in the smaller models, but those differences are not statistically clear after
adjustment. The earlier concern that penalties never help is therefore resolved
most clearly in the setting with many text inputs. Beating OLS on those inputs
still does not establish that text beats the model without text.

**Fifth, choosing the training-window length adds relatively little.** Letting
validation choose one, two or three years usually improves the average score
slightly. The statistically clearest improvements occur in full-text OLS, which
remains weaker than the simpler penalized benchmarks. This change does not
materially alter the conclusion about the contribution of text.

**Better ranking does not automatically mean a better trading strategy.** Sentiment
and attention improve some equal-weighted portfolio spreads, but the evidence is
less consistent when larger companies receive more weight. No positive incremental
text gain passes the adjusted full-period raw-return portfolio tests. These are
gross returns; trading costs and execution constraints have not been incorporated.

My suggested working baseline is a fixed two-year training window, six-month
validation, and a final refit on the combined sample, with ridge models for
characteristics alone and characteristics plus sentiment/attention. Keep OLS and
sparse regressions as comparisons, and retain compressed text as a candidate for
further study. The results support improving estimation; they do not yet justify
an expensive search over neural-network designs. A small, matched nonlinear
experiment could separately test whether nonlinear relationships reveal a text
increment that these linear models miss.

These are development results: we have already used 2014–2022 to guide project
decisions. They need separate confirmation before being treated as final evidence.
The study also does not answer whether social media adds information beyond news
or reliably timed accounting fundamentals, neither of which is included here.
