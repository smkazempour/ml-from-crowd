# Neural-network execution under protocol v1.1

**Complete: September 17, 2026 at 17:07 Central (22:07 UTC).**
All **24 NN3 specifications and 2,592 monthly checkpoints** are finished. The full
comparison evaluates **120 models: 24 neural networks and 96 matched linear models**,
covering all 108 test months in 2014-2022 and fitting histories of 252, 504 and 756 sessions.

The [final report](08_protocol_nn_results.md) and
[evaluation metadata](data/protocol_v1_1_combined.json) are available. Training,
evaluation and report generation all exited successfully. The recorded SHA-256
hashes of the combined registry, evaluation metadata and final report were verified
against the controller's completion record. The attempt 3 launcher error log is empty.
The controller has exited; no NN training process remains active.

The [execution record](data/protocol_v1_1_execution.json) contains completion evidence,
artifact paths and historical progress snapshots. The
[completed study status](../.runs/protocol_v1_1/nn_study/48b73c386ddd0dec/status.json)
records attempt 3 and its successful pipeline commands. Its last update is the
completion timestamp, not a stale running heartbeat.

Training finished at approximately 16:59 Central; verification, evaluation and report
generation followed, completing at 17:07. Attempt 3 started September 16 at 09:55
Central and completed in 31.2 hours, reusing 1,926 existing checkpoints. The earlier
interruptions and recovery are preserved below as historical records.

The [training log](../.runs/protocol_v1_1/nn_study/48b73c386ddd0dec/nn_attempt_003.log),
[evaluation log](../.runs/protocol_v1_1/nn_study/48b73c386ddd0dec/evaluation_attempt_003.log),
[report log](../.runs/protocol_v1_1/nn_study/48b73c386ddd0dec/report_attempt_003.log) and
[study specification](../.runs/protocol_v1_1/nn_study/48b73c386ddd0dec/study_spec.json)
provide the detailed execution record. [Report 07](07_protocol_linear_results.md)
remains the completed linear-only comparison; Report 06 describes the older pilot.

## Status-file failure and recovery: 2026-09-16

Attempt 2's last controller heartbeat was September 15 at 18:32 Central. Its
launcher error log records `PermissionError: [WinError 5]` while atomically replacing
`status.json`. The exception handler then terminated the training parent, leaving
orphan workers; the final checkpoint was saved at 19:17 Central. The machine had
not rebooted since September 15 at 02:25 Central. This was a controller status-write
failure, not evidence of a failed model fit. The exact source of the access denial
has not been identified; the same atomic replacement succeeded during recovery.

The original status was archived, the eight idle orphan workers and their resource
tracker were stopped, and all eight frozen source hashes, the prepared manifest,
linear registry and Python identity were verified unchanged. The
[attempt 2 audit](data/protocol_v1_1_interruption_attempt_002.json) records the error,
process cleanup and inventory of all 1,926 saved checkpoints. The restart reuses
the existing study and its normal checkpoint verification.

The isolated [recovery launcher](../tools/resume_protocol_nn_study.py) wraps only
the controller's status updates. It retries Windows access/sharing errors for two
seconds and skips a blocked running-child heartbeat after logging the issue.
Startup and terminal writes remain strict, as do prediction, checkpoint, registry
and evaluation writes. The eight frozen source files and scientific recipe were
not edited. Seven recovery tests and five existing controller tests passed.
The [attempt 3 launch log](../.runs/protocol_v1_1/nn_resume_20260916T145510Z.log)
records the launcher's pinned SHA-256 digest and the original study identity.

The September 16 planning estimate allowed 35-45 hours from restart. The actual
remaining runtime was 31.2 hours, including evaluation and report generation;
completion was September 17 at 17:07 Central. Historical forecasts are superseded
by the verified completion timestamp.

## Interruption snapshot and recovery: 2026-09-15

At interruption, **1,752 of 2,592 monthly jobs were saved (67.6% by job count)**, and 16 of the 24
specifications have published complete 108-month prediction series. At that point,
840 monthly jobs remained, followed by evaluation and report generation.

| Fitting window | Saved monthly jobs | Complete specifications | Status |
|---|---:|---:|---|
| 504 sessions (primary) | 864 / 864 | 8 / 8 | All feature sets and both targets complete |
| 252 sessions | 864 / 864 | 8 / 8 | All feature sets and both targets complete |
| 756 sessions | 24 / 864 | 0 / 8 | Interrupted during raw/core; saved through December 2015 |

All **1,752 saved checkpoints passed integrity checks**: each file loaded,
matched its specification and monthly task, contained finite predictions for the
exact expected rows, and included all three candidate penalties and five saved
seed states per candidate. All **16 published prediction files matched their
recorded SHA-256 hashes**. No empty checkpoints or temporary checkpoint files were
found. The eight frozen source hashes, prepared manifest, linear registry and
controller Python identity remain unchanged.

The log contains 26,354 completed candidate-seed fits, with epochs ranging from
6 to 52. No errors, warnings or 100-epoch cap hits were found before the abrupt stop.
Windows Event 1074 attributes the first restart to `MoUsoCoreWorker.exe` for a
planned operating-system service pack, followed by a planned upgrade restart from
`TrustedInstaller.exe`. This is evidence of a Windows update restart; the shutdown
cause is not being inferred from a stale process ID alone.

At interruption, eight workers were fitting January-August 2016 for raw returns,
core inputs and the 756-session window. Those unfinished month jobs had 74 completed
seed fits in memory, plus any fits still in progress. That within-month work must
be repeated because durable checkpoints are written only after the whole month
finishes. The 1,752 completed months can be reused.

The existing controller supports restart with the original command and environment;
it verifies saved checkpoints and resumes missing months under a new attempt log.
The original stopped heartbeat and first-attempt training log have been preserved.
The authorized restart on September 15 uses `nn_attempt_002.log`; all 1,752 old monthly
checkpoints were confirmed as reused before the missing-month fits resumed.
Most of the longer-window block remains.

At that historical interruption snapshot, full-period predictions were available for
two fitting windows and the full comparison was pending. All three windows and
Report 08 are now complete, as recorded above.

## Preliminary evaluation

The [preliminary report](08a_protocol_nn_preliminary_results.md) preserves the earlier F504/F252 snapshot.
Use [Report 08](08_protocol_nn_results.md) for the complete three-window study. The preliminary snapshot contains:
16 NN3 models and 64 matched linear models over all 108 test months in 2014-2022.
The remaining F756 models are excluded before scoring. The full planned Bonferroni
families are retained: 60 feature, 84 estimator and 40 history contrasts per target,
metric and period. This analysis uses copied prediction files and does not change
the training code or the final controller's Report 08 destination.

Text additions improve NN3 ranking, but no positive primary full-period NN3-versus-linear
IC or portfolio-spread difference passes the planned adjustment. Results vary by subperiod:
NN3 with richer inputs performs better relative to linears in 2019-2022. The shorter
252-session fitting history lowers NN3 IC for all three noncore input sets and both targets.
All 192 included linear summary rows reproduce Report 07 exactly. See the preliminary
report for effect sizes, inference details and the limits of these interim findings.

## Training specification and pilot checks

The neural network uses the same prepared rows, transformations, fitting/validation
windows, rank targets and equal-date loss weights as the revised linear models. NN3 has
hidden layers of 128, 64 and 32 units, BatchNorm and ReLU, Adam with learning rate 0.001,
and batches of 10,000. Every candidate averages five fixed seeds. Validation alone selects
each seed's checkpoint and then selects among L2 penalties 0.00001, 0.0001 and 0.001.
Training stops after five epochs without validation improvement, with a 100-epoch ceiling.
The fitted model is not refitted on the validation period.

The pilot uses the complete training budget for core and text + core inputs in January
2014, March 2015 and December 2022. It checks training behavior, saved-state reconstruction
and restart integrity. These six tasks do not establish full-period predictive performance.
The checks require unchanged checkpoint bytes and modification times, unchanged prediction
files on resume, and exact restored predictions for two representative tasks.

All 90 candidate-seed fits passed, with zero epoch-cap hits. Selected checkpoints ranged
from epochs 1 to 18. The first pass took 1,249.9 seconds with three workers; replay took
8.3 seconds and resumed all six tasks without altering checkpoint bytes/modification
times or prediction bytes. Restored five-seed ensembles reproduced all 13,159 January
2014 core predictions and 33,260 December 2022 text + core predictions exactly.

| Inputs | January 2014 | March 2015 | December 2022 |
|---|---:|---:|---:|
| Core | 24.1 s | 46.9 s | 174.0 s |
| Text + core | 67.2 s | 260.1 s | 1,068.3 s |

These per-task times include block construction and fitting, but exclude initial bundle
loading. They overlap across workers and should not be summed as wall time. Task sizes
and stopping epochs vary substantially; historical runtime estimates in Report 05 do not
budget this larger revised study.

The authorized full study covers all 108 test months in 2014–2022, four feature sets,
raw and DGTW targets, and fitting windows of 252, 504 and 756 sessions with 126 validation
sessions. This is 24 NN specifications, 2,592 monthly jobs and 38,880 candidate-seed fits.
The full comparison includes all 96 linear specifications, including separately registered
NN-versus-ridge/lasso/elastic-net contrasts. The primary fitting window remains 504 sessions.

[run_protocol_nn_study.py](../tools/run_protocol_nn_study.py) supervises training, checks
full-period prediction coverage and hashes, then evaluates the combined 120-model registry
and writes `08_protocol_nn_results.md`. Monthly checkpoints support interruption and resume.
The controller publishes a live `status.json` with its phase, process IDs, logs and any
failure; it marks completion only after evaluation and report generation succeed.

The completed study used eight workers and two threads per worker. Exact launch and
restart arguments are in the execution record. Its eight source files were copied
into the study's `code/` directory and checked before final publication. A changed
experimental implementation needs a new run identity.

The [frozen experiment specification](data/protocol_v1_1_experiment.json),
[shared protocol](EXPERIMENTAL_PROTOCOL.md) and [timing audit](TIMING_CONVENTION.md) apply.
In particular, this is historical development evidence, with known nominal-close routing
limitations. Gross portfolio sorts do not establish attainable returns after trading costs.

## Independent server continuation and architecture comparisons

The [server handoff](SERVER_RUN_HANDOFF.md) identifies the ready transfer snapshot and its validation checks.

The [server package](../server_nn/README.md) exports a self-contained snapshot of the
corrected prepared inputs, all 96 completed linear prediction files, and finished local
NN3 month checkpoints. Its training/evaluation kernels are frozen copies of this run's
sources; preparing the package does not stop or change the active local study.

The default server sequence completes NN3 at all three fitting windows, then compares
NN1 `[128]`, NN2 `[128, 64]`, NN3 `[128, 64, 32]` and NN4 `[128, 64, 32, 16]` at 504
fitting sessions. Width comparisons and validation-selected adaptive depth are optional.
All retain the same inputs, targets, 126-session validation period and full tuning budget.
Server results have their own output registries and reports; they do not overwrite Report 08.

Before local checkpoints can be reused, the server must pass bundle checksums, runtime
compatibility and restored-prediction bridge checks. The export is a point-in-time copy;
subsequent local progress is not automatically synchronized. Follow the server README
for installation, resource benchmarking, launch, restart and results retrieval.
