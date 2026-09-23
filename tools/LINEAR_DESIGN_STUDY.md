# Running the linear design study

This experiment compares grouped ridge penalties, text compression, training-window selection, and final coefficient refitting. Its frozen design is in [linear_design_v2_experiment.json](../reports/data/linear_design_v2_experiment.json). The main outputs are [Report 10](../reports/10_linear_design_results.md) and `reports/data/linear_design_v2*`.

## Environment and prerequisites

Run from the repository root using the same `py313` environment as the completed characteristic study. On this machine its Python executable is:

```text
C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe
```

The pipeline uses the existing NumPy, pandas, SciPy, scikit-learn, joblib, threadpoolctl, filelock and psutil installation. It verifies the recorded Python and numerical-library versions against the earlier study. Changing environments can prevent baseline certification even when the package names are the same.

The following local artifacts must exist; Git does not contain these large ignored data files:

- Prepared data: `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/`, including its manifest, feature matrix, keys, evaluation panel, targets, calendar and other certified artifacts.
- Historical registry: `.runs/characteristics_v1/study/d20631f4712e3eb5/linear/full_fc16bf210fe45a57/registry.json`.
- Historical monthly checkpoints beside that registry: `checkpoints/fit504_{raw|dgtw}_{YYYY-MM}.pkl` for January 2014 through December 2022.

Preserve the certified files and paths. A fresh Git clone alone cannot run this experiment. The prepared matrix and evaluation panel occupy several gigabytes, and the new checkpoints and prediction files also require substantial local disk space.

## Start and resume

With `py313` activated, use:

```powershell
python -B tools/run_linear_design_study.py --workers 4 --threads 2
```

The controller sets numerical-library thread limits for its child processes. Four workers each use two numerical threads during full fitting; the pilot uses two workers. These are independent monthly jobs, not four copies of the study.

After an interruption, use **the same command**, with the same environment, sources, specification and inputs. Completed monthly checkpoints are validated and reused. Incomplete atomic writes are not accepted as finished checkpoints. The controller retries its stages, so the pilot may print `RESUME` lines before full fitting resumes. Evaluation or reporting may be repeated after an interruption during those stages.

A study lock prevents two controllers from owning the same run. Check the recorded processes before attempting a restart. The controller and each worker also use separate checkpoint locks.

## Stages and progress

The controller prints `STUDY <directory>`. If it was launched in the background, find this line in its launch log. For the initial September 21 launch:

```text
.runs/linear_design_v2/launch_20260921T155709Z.log
.runs/linear_design_v2/study/93abd4871c37fe50/status.json
```

That initial run stopped on a numerical reproduction check. The repaired study was launched at 1:19 p.m. Central on September 21; its current paths are:

```text
.runs/linear_design_v2/launch_20260921T181919Z.log
.runs/linear_design_v2/study/c11478141261534f/status.json
```

The repair preserves historical scaling before subsetting inputs, with no tolerance or experiment-setting changes. The old run remains archived; its checkpoints are not reused under the new source identity. See the [status and repair history](../reports/LINEAR_DESIGN_RUN_STATUS.md).

`status.json` records the phase, controller and child process IDs, active log, timestamps, exit codes, and final output paths. Each attempt has separate pilot, fitting, evaluation and report logs. A heartbeat indicates the controller is active; a `SAVED` line in the fitting log indicates a completed monthly checkpoint.

Automatic Windows desktop alerts and a durable outcome report are provided by the
separate [study watcher](STUDY_NOTIFICATIONS.md). It observes the entire controller,
including evaluation and reporting, and handles both success and failure.

The stages are:

1. Verify prepared data, the declared experiment and historical baseline provenance.
2. Run an implementation pilot for January 2014 and December 2022, on both targets, using the full candidate menu.
3. Run **216 month-target checkpoints**: 108 months times two targets. Each checkpoint produces all 114 procedures for its target. The complete registry therefore contains **228 model specifications**.
4. Verify every model's actual prediction keys, finite values and file hashes.
5. Evaluate the complete registry and generate Report 10.

The pilot and full study have separate directories. Pilot checkpoints do not count toward the full study's 216 checkpoints. Completion requires `status: "complete"` and successful evaluation/report generation; finishing the fitting stage alone is insufficient.

## What is certified

All procedures forecast the same eligible stock-days. Missing outcomes exclude rows from supervised fitting or scoring, while predictions are retained for those rows. Missing predictor values follow the existing fitting-only transformation rules. The complete 2014–2022 prediction panel contains 3,034,035 rows per model.

Each month, the 12 global full-input procedures using the fixed 504-session window and retained coefficients must reproduce the historical characteristic-study forecasts. Certification checks prediction differences and daily rank-IC differences. Checkpoints also certify prediction values, row indices and selection diagnostics; a changed saved penalty or feature choice is detected even if the predictions were untouched.

The registry records all model metadata and prediction hashes. `monthly_selection.csv` records selected histories, penalties, PCA dimensions and scalar fitting diagnostics. More detailed candidate and transformation diagnostics remain in the monthly checkpoints. Evaluation tables include compressed daily results to keep published files manageable.

## Keep the running study unchanged

Do not edit the experiment JSON, certified inputs or source files listed in the controller's `SOURCES` while this run is active. Their hashes identify the study and are checked again before completion. The controller saves a source snapshot for audit; its subprocesses still execute the working-tree files.

Changes to numerical code, the menu, thread count or certified inputs can create a different run identity or invalidate completion. Investigate a failed certificate rather than deleting checkpoints or weakening its checks. Documentation and unrelated project work can continue while the frozen study runs.
