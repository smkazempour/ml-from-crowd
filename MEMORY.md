# Project memory

Updated September 22, 2026, at the user's request before moving to a new conversation.
This is versioned project memory. Detailed state and file locations are in
[the project state](reports/00_project_state.md) and
[the next-session handoff](reports/NEXT_SESSION_HANDOFF.md).

## Where we stopped

- Explicit interaction regressions (Stage A) are **complete**. All 216 monthly
  outcome jobs and 148 procedures finished, including evaluation and reports,
  September 22 at 4:38:43 p.m. Central. Thirteen final artifact hashes were verified.
  Windows accepted the completion alert at 4:38:47. No study needs resuming.
- Read [Report 13a](reports/13a_linear_interaction_summary.md) for the digest and
  [Report 13](reports/13_linear_interaction_results.md) for the complete register.
- In primary raw-return ridge comparisons, sentiment/attention adds about .001
  daily rank IC beyond quadratic market controls (adjusted p below .001).
  None of the 35 single social products has an adjusted-significant gain. Their
  joint block and the text-interaction blocks show no established positive gain.
  Do not turn this into "all interactions never help": some characteristic-only
  comparisons improve, and some added text interactions significantly hurt.
- Reports 09, 10/10a and 11/11a cover the completed characteristic, linear-design
  and linear-optimization studies. The social-only NN3 study is complete in
  Report 08; it did not condition on these market characteristics or use this
  combined-sample refit. The old server NN package is also social-only.
- Proposed next work is Stage B, tuned trees, then Stage C, matched NN design
  comparisons, in [Report 12](reports/12_interactions_and_nonlinear_plan.md).
  These extensions have not been implemented or launched. The user's latest
  instruction was to save the handoff and push the repo, not start another study.

## Decisions and corrections to retain

- Evaluate social information against matched characteristic-only models and
  added text against matched characteristics-plus-sentiment/attention models.
  Better text estimation is different from evidence that text adds information.
- Current procedure: 504 initial fitting sessions, 126 chronological validation
  sessions, monthly updates, maximum mean daily validation Spearman IC, then
  refit on the original 630-session union with maturity exclusions preserved.
  Refit scalers/PCA and sparse alpha maxima on that union. Do not train on test labels.
- Use daily return-rank targets, squared loss, equal total weight per day, and
  common prediction keys. Primary inference is paired calendar-aware HAC5 with
  HAC21/63 checks and declared comparison-family adjustment. Raw return is primary;
  DGTW is secondary. Gross decile spreads are not net trading returns.
- Tweets map to the first assigned market close after posting, called t; the
  target is close t to close t+1 (or t+h). Preserve the documented early-close caveat.
- The sample already contains only retained, sentiment-tagged message-covered
  stock-days with embeddings, not all CRSP stock-days. Broader CRSP histories
  calculate controls without adding no-message rows. Untagged messages are a
  separate future data extension.
- **The characteristics and social scalars were already daily-ranked in X.npy.**
  Report 11's additional rank branch re-ranked existing ranks and imputed zeros;
  it was not a raw-versus-ranked experiment. Read [the correction](reports/11b_sample_and_transformation_audit.md).
  Do not silently edit the certified historical reports to conceal that correction.
- 2014-2022 has repeatedly informed design decisions and is development evidence.
  2023 has only 178/250 expected sessions and prior predictions; no untouched
  confirmation sample is certified. Accounting publication timing and the known
  `ar_dgtw_21` 2013 corruption remain unresolved limitations.

## Working preferences

The user wants straightforward explanations, concise result summaries alongside
full tables, autonomous execution of agreed experiments, and automatic reporting
of completion or errors. They asked to explore meaningful model alternatives
and to study market controls before expensive nonlinear designs. Their agenda
may support multiple papers: preserve Q1-Q6 in [RESEARCH_QUESTIONS.md](reports/RESEARCH_QUESTIONS.md).

Use `D:/StockTwits/Code`; the old C: project copy is stale. The working interpreter
is `C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe`.
Large data and `.runs/` remain local; GitHub contains code, declarations, reports
and compressed evaluation tables. The 128-CPU/200-GB server remains available as
a proposed execution option, but its existing bundle needs market controls added
before it can run the next matched nonlinear study.
