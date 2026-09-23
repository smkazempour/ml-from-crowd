# Handoff for the next conversation

Prepared September 22, 2026. The user requested this handoff and a complete GitHub
commit before continuing elsewhere. No new experiment was launched during handoff.

## Start here

Read [project memory](../MEMORY.md), [current state](00_project_state.md),
[the interaction digest](13a_linear_interaction_summary.md), and
[the transformation correction](11b_sample_and_transformation_audit.md).
The research agenda is [Q1-Q6](RESEARCH_QUESTIONS.md); it can lead to several papers.

All studies through explicit interaction Stage A are complete. There is no
unfinished fit to resume. Completion means fitting, prediction verification,
evaluation, reporting, and artifact checks all passed. Local controllers retain
status records rather than relying on a previously recorded process ID.

## Latest result and next decision

Report 13 includes OLS/ridge/elastic net on 13 joint/reference bases and 35 separate
ridge social-product additions: 74 procedures per target, 148 total. Each single
term is added to the same 191-column characteristic-quadratic/social-square base.
The menu includes characteristic-only quadratic controls and PCA16 text extensions.

The primary raw-return ridge results support a small sentiment/attention increment
beyond market controls. They do not establish positive gains from the 35 individual
social products, their joint block, or text-interaction extensions after the
declared corrections. Some characteristic-only shape comparisons help, so do not
generalize this to all nonlinear relationships. This is a development result,
not a causal interpretation or untouched confirmation test.

The proposed next stage in [Report 12](12_interactions_and_nonlinear_plan.md) is
**trees**: histogram boosting and random forests on matched characteristic-only,
plus sentiment/attention, and plus text information sets. The proposal compares
additive, pairwise and unrestricted boosting interactions. Exact tree candidate
lists, iteration checks, seeds and resource allocations still need implementation,
declaration, and an integrity/runtime pilot. Preserve chronological validation and
daily IC selection. Do not silently use random splits, validation MSE, or OOB scores.

Later NN work would use the same market-conditioned information sets and final
combined-sample refit. The completed social-only NN3 run and the old server export
do not establish this comparison. Tagged-versus-untagged messages, genuine raw
versus ranked characteristics, news, earnings, and market-level targets remain
separate extensions. Follow the user's new instructions about which to pursue.

## Local execution map

Workspace: `D:/StockTwits/Code`. Python:
`C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe`.
The prepared characteristic cache is
`.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/`.

| Completed study | Local study directory under `.runs/` | Full fitting directory within it |
| --- | --- | --- |
| Characteristics, Report 09 | `characteristics_v1/study/d20631f4712e3eb5` | `linear/full_fc16bf210fe45a57` |
| Linear design, Report 10 | `linear_design_v2/study/c11478141261534f` | `linear/full_9590a962af3e2fcb` |
| Linear optimization, Report 11 | `linear_optimization_v3/study/ed854a1f5bf3da39` | `linear/full_b217f4648f5fe89a` |
| Explicit interactions, Report 13 | `linear_interactions_v1/study/78b3dd65e0d5f8a6` | `linear/full_9f279d17e34278c6` |

Each fitting directory contains `registry.json` and certified monthly checkpoints.
Each study has `status.json`, phase logs and a source snapshot. Report 13's
completion/error record is [LINEAR_INTERACTION_COMPLETION.md](LINEAR_INTERACTION_COMPLETION.md);
the execution audit is [linear_interactions_v1_execution.json](data/linear_interactions_v1_execution.json).
Completion was September 22 at 16:38:43 Central, with Windows notification accepted
four seconds later. All 13 final output hashes were independently verified.

Implementation families are `tools/linear_design_*`, `tools/linear_optimization_*`,
and `tools/linear_interaction_*`, with corresponding `run_*_study.py` controllers.
Use their study guides for reproduction, not the older notebooks' default settings.
The shared notification watcher is documented in [STUDY_NOTIFICATIONS.md](../tools/STUDY_NOTIFICATIONS.md).

The interaction implementation passed 30 focused tests before launch, then a full
January 2014/December 2022, two-target pilot including evaluation and reports.
The nine additive references reproduced archived forecasts and daily ranks exactly
in all four pilot jobs. Full publication certifies the declared baseline tolerances
for every monthly job. No methods changed during the completed run.

## Reproducibility and storage

Git does not contain the multi-GB prepared cache, monthly predictions/checkpoints,
or the upstream StockTwits/CRSP source data. A fresh clone alone cannot refit the
study; copy or rebuild those separately using the documented manifests. Compressed
daily evaluation tables are committed; duplicate uncompressed daily exports stay
ignored. Other aggregate result tables and machine-readable declarations are included.

Completed source and output identities use byte-level SHA-256. Preserve their
line endings and frozen numerical versions when validating an existing study;
Git attributes preserve the certified files' bytes. Implement changed methods
under a new declaration/run identity and retain the old records. Do not edit a
completed report or numerical source and then reuse its old completion certificate.
Put later corrections in an explicit companion or current-state document.

On Windows, run background controllers hidden with redirected logs. Attach the
completion/error watcher; an alert is local to Windows and is not a scheduled chat
message. After a blackout, inspect status and verified checkpoints before resuming.
The original 128-CPU/200-GB server package is social-only and needs a new export for
the proposed characteristic-conditioned tree/NN experiments.
