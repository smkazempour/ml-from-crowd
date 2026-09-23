# Characteristic-conditioned linear study

Launched September 19, 2026 at 12:32 p.m. Central (17:32 UTC). This is the
`characteristics_v1` experiment, following the completed social-only linear/NN3
study committed and pushed as `9ac568e`.

**Complete:** September 19, 2026 at **1:49:53 p.m. Central** (18:49:53 UTC).
The full [results report](09_characteristic_linear_results.md) is available.
All **648 monthly checkpoints** and **264 model specifications** completed, with
108 test months and 3,034,035 eligible prediction rows per model. All 96 reused
social-only models passed their source and prediction checks. Preparation/pilot,
training, evaluation and report generation succeeded; all four supervised phases
returned exit code zero. The controller is no longer running.

The saved registry, report, evaluation metadata and all nine result tables passed
independent SHA-256 revalidation on September 19. The completion record is
[data/characteristics_v1_completion.json](data/characteristics_v1_completion.json).
The full fitting directory is
`.runs/characteristics_v1/study/d20631f4712e3eb5/linear/full_fc16bf210fe45a57`.

## Design and prepared data

- [Fixed experiment specification](CHARACTERISTIC_EXPERIMENT.md): 11 input sets,
  four linear estimators, two targets and three existing fitting windows, totaling
  264 specifications. Of these, 96 social-only specifications are reused after
  exact input/prediction checks and 168 specifications are newly fitted.
- The new benchmark contains 17 market-based characteristics and 17 missingness
  flags. Sentiment, attention, other social features and text are added separately
  and jointly. The 504-session history is primary; the existing 252/756-session
  histories remain prespecified sensitivity comparisons.
- Prepared cache: `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42`.
  It preserves all 3,514,785 parent rows and all 7,974 stocks. Every parent key
  exists in the primitive CRSP history. The original 439 social/text columns are
  byte-identical; the largest augmented input has 473 columns.
- Missing characteristic values do not remove observations. The full preparation
  audit is `characteristic_audit.json` inside the prepared cache. See also the
  [independent real-data audit](CHARACTERISTIC_DATA_AUDIT.md).
- Accounting predictors are deferred because the available fiscal-date merge
  lacks publication timing. Raw returns are primary; the inherited DGTW target
  remains secondary with this upstream accounting-timing limitation disclosed.
- All 110 tools tests pass, including 32 new tests. The unchanged server package
  also passed its 32 tests before the original-study commit.

## Process and recovery

Controller PID at launch: **9532** (now exited). Six fitting workers used two numerical
threads each; the four-job implementation pilot used two workers. Its two test months were
January 2014 and December 2022, for both targets. Pilot outcomes do not select or
alter the scientific design.

Pilot fit times were 12.7/10.8 seconds for January 2014 (raw/DGTW) and 33.5/30.2
seconds for December 2022. Actual end-to-end controller time was **1 hour 17 minutes
24 seconds**. The full training/publication phase took 54 minutes 32 seconds;
evaluation took 19 minutes 46 seconds, followed by 5 seconds for report generation.

Live status:
`.runs/characteristics_v1/study/d20631f4712e3eb5/status.json`.

Controller output:
`.runs/characteristics_v1/launch_20260919T173229Z.log` and
`.runs/characteristics_v1/launch_20260919T173229Z.err`.

Phase logs, frozen source snapshots, the complete study identity and checkpoints
live under `.runs/characteristics_v1/study/d20631f4712e3eb5/`.
The historical launch snapshot is [data/characteristics_v1_execution.json](data/characteristics_v1_execution.json);
its running status records the launch and is superseded by the completion record.
See [the reproduction and recovery guide](../tools/CHARACTERISTIC_STUDY.md) for
verification or reproduction. No resume is needed. File locks, checkpoint hashes
and deterministic run identities protect restart integrity.

Completion requires all 648 monthly fitting checkpoints, a certified 264-model
registry with 108 test months per model, evaluation, report generation and output
hash verification. The controller records `status: complete` only after all stages
succeed. Merely observing that the training process stopped is insufficient.

## Results and next decision

The controller wrote [Report 09](09_characteristic_linear_results.md) and
`reports/data/characteristics_v1_linear*`. The daily table is published as a gzip
file; its uncompressed local copy is ignored by Git. Prepared arrays and model
predictions remain in ignored `.runs/`.

The first question is how much sentiment and attention improve a matched
characteristic-only model, followed by the conditional contribution of text and
other social features. Training-window optimization, expanded penalty grids and
further neural networks remain deferred until these results are reviewed.
