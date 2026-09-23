# Linear design study status

Restarted **September 21, 2026 at 1:19 p.m. Central** (18:19 UTC).
The `linear_design_v2` study follows the completed
[characteristic comparison](09_characteristic_linear_results.md).

**Complete:** September 21, 2026 at **5:31:46 p.m. Central** (22:31:46 UTC).
All **216 monthly jobs** and **228 model specifications** completed, followed by
prediction verification, evaluation and [Report 10](10_linear_design_results.md).
Every model has 108 months and 3,034,035 eligible prediction rows. All four
supervised phases returned exit code zero. The repaired controller has exited.
The 12 saved output artifacts passed independent SHA-256 revalidation; see the
[completion record](data/linear_design_v2_completion.json).

**Automatic alert delivered:** the watcher recorded Windows acceptance of the
completion notification at **5:31:56 p.m. Central**, ten seconds after completion.
This confirms delivery to Windows, not that the user viewed the banner.
The watcher wrote the [outcome report](LINEAR_DESIGN_COMPLETION.md) and then exited.
The [notification guide](../tools/STUDY_NOTIFICATIONS.md) contains monitoring and
recovery details. The watcher did not change the experiment's sources.

## First launch and repair

The initial launch was at 10:57 a.m. Central. It stopped at **11:10:06 a.m.**,
after saving four of 216 full jobs (January and February 2014, both targets).
A March 2014 retained OLS forecast differed from its historical baseline by
1.72e-9 and failed the strict numerical reproduction check. The narrower input
matrix changed a scaler's floating-point reduction; one standardized value then
rounded to an adjacent float32 value. Daily IC was unchanged.

The repair preserves the historical transformation and moment calculations before
selecting this study's inputs. **The experiment settings and certificate tolerances
are unchanged.** The repaired March job's largest discrepancy across all 12
baselines is 2.08e-16, with zero daily IC difference. All 31 focused design tests
passed, including a new regression test, and an independent review found no blocker.

The failed run's logs, source snapshot and four checkpoints remain under
`.runs/linear_design_v2/study/93abd4871c37fe50/`. A changed source hash gives the
repaired study a new identity, so its jobs are recomputed. See the
[failure record](data/linear_design_v2_failure_20260921.json) and
[recovery record](data/linear_design_v2_recovery.json).

## Comparisons

The [fixed design](LINEAR_DESIGN_EXPERIMENT.md) includes:

- Original training coefficients; refitting on the latest F sessions; and refitting
  on the original training plus validation samples. With F=504, the last option
  uses **630 sessions, approximately 2.5 years**, as requested.
- A fixed 504-session training history versus validation choosing 252, 504 or 756.
- OLS, ridge, lasso and elastic net, plus ridge with separate feature-group penalties.
- Full text embeddings versus validation-selected PCA compression.
- Characteristics alone; characteristics plus sentiment/attention; and those
  inputs plus text, on the same prediction rows.

There are **228 specifications**, two targets and 108 test months (2014–2022).
Full fitting requires **216 month-target checkpoints**, each containing all 114
procedures for its target. Four pilot checkpoints are separate from this count.
Raw returns are primary; DGTW remains secondary. No neural networks are started.

## Verification and execution

Before the first launch, the tools suite and subsequent checks validated 140 tests.
The repair adds one regression test; all 31 focused design tests passed afterward.
The checks
cover fitting dates, label maturity, fitting-only PCA, penalty transfer, exact
group exclusion, baseline reproduction, checkpoint integrity and output coverage.
Independent reviews of the model menu, evaluation contrasts and controller found
no blocker before launch.

Controller PID: **20204** (now exited). Full fitting used four workers with two
numerical threads each; the pilot used two workers. Prepared data and historical
checkpoints were reused without changing the earlier studies.

Original pilot job times were 90.1/82.5 seconds for January 2014 (raw/DGTW) and 434.0/404.5
seconds for December 2022. The latter month's 504-session training block has
1,056,018 input rows versus 88,388 in January 2014. Each job scores 2,877 candidate
fits across the three histories before producing its final refits, so runtime
varies substantially by month. Actual end-to-end time for the repaired study was
**4 hours 12 minutes 26 seconds**. Full fitting/publication took 3 hours 43 minutes
12 seconds; evaluation took 21 minutes 3 seconds, followed by report generation.

Study directory:
`.runs/linear_design_v2/study/c11478141261534f/`

Final controller status:
`.runs/linear_design_v2/study/c11478141261534f/status.json`

Launch logs:
`.runs/linear_design_v2/launch_20260921T181919Z.log` and
`.runs/linear_design_v2/launch_20260921T181919Z.err`.

The study directory contains per-phase logs, source snapshots and the complete
experiment identity. The original [execution snapshot](data/linear_design_v2_execution.json)
and current [recovery snapshot](data/linear_design_v2_recovery.json) record provenance.
Keep the numerical source files, experiment JSON and
certified inputs with the archived study for reproducibility.

## Recovery and completion

From the repository root, in the existing `py313` environment:

```powershell
python -B tools/run_linear_design_study.py --workers 4 --threads 2
```

No restart is currently needed. Use the same command after an interruption.
Validated completed checkpoints are
reused; a file lock prevents duplicate controllers for the same study. See the
[operational guide](../tools/LINEAR_DESIGN_STUDY.md) for prerequisites and details.

The controller writes `reports/10_linear_design_results.md` and
`reports/data/linear_design_v2*`. Completion requires all fitting checkpoints,
a verified 228-model registry, evaluation, report generation and output hashes.
Training ending alone does not mean the study has completed.

## Interpretation

These are development comparisons: the 2014–2022 results have already informed
the design. The 2023 coverage audit found 72 missing exchange sessions and evidence
of earlier predictions, so that year is not certified as untouched confirmation.
No 2023 performance is scored. We retain the existing validation block and defer
the proposed near-tie rule and new validation-fold arrangements.
