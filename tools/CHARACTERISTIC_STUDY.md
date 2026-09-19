# Reproducing the characteristic-controlled linear study

The frozen design is [characteristics_v1_experiment.json](../reports/data/characteristics_v1_experiment.json). This phase measures the incremental information in sentiment, attention, engineered social features and text after conditioning on market characteristics and past returns. Window optimization, new penalty grids and neural networks are deferred.

The experiment retains the original stock-day universe, close-t information convention, one-day labels, monthly refits, equal-date squared loss on return ranks, fitting-only scaling and validation rank IC. F504 is primary; F252 and F756 are fixed sensitivities, with 126 validation sessions and no refit on validation. Four estimators and eleven feature sets across two targets and three histories produce 264 specifications. Seven characteristic feature sets require 648 new month/target/history checkpoints; 96 existing social-only prediction series are verified and reused. The 17 market controls each have an explicit missingness indicator. Accounting characteristics remain excluded pending a point-in-time availability audit; raw returns are primary and inherited DGTW returns are secondary.

## Environment and required local artifacts

Run from `D:\StockTwits\Code` using the existing interpreter:

```powershell
$studyPython = 'C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe'
```

The completed parent study and current run use Python 3.13.11, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.5.0, filelock 3.20.0 and psutil 7.0.0. Reuse explicitly requires matching Python and numerical-library versions. The controller limits BLAS/OpenMP threads in child processes. This linear study does not train PyTorch models. The default PATH Python is not the configured project environment.

Required inputs, which are local and excluded from Git:

- Parent prepared cache: `.runs/protocol_v1_1/prepared/8f3f4eb44b399771`, including its original feature matrix, keys, labels, calendar and evaluation panel.
- Parent linear registry: `.runs/protocol_v1_1/linear/full_3314fb5ca2fe170a/registry.json`, together with all referenced prediction files. A registry JSON alone is insufficient.
- Primitive CRSP source: `D:/StockTwits/Data/CRSP/dsf.pkl`, even when supplying an already prepared cache: the controller checks its identity.
- Prepared characteristic cache: `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42` for the September 19, 2026 launch.

The manifest identifies source hashes, feature recipes, versions and preserved parent files. Model reuse verifies exact parent data and feature-prefix bytes, source code, grids, chronological tasks, prediction hashes and economic-key coverage. Retain the existing directory layout and files when resuming this local study.

## Launch and resume

Use the verified prepared cache and the current six-worker allocation:

```powershell
& $studyPython -B tools/run_characteristic_study.py `
  --prepared .runs/characteristics_v1/prepared/35ef3a5eb0cb5e42 `
  --workers 6 --threads 2
```

This command runs in the foreground. The September 19 study was launched separately in the background; do not start a second copy while that controller is active. The study lock also prevents simultaneous controllers for the same identity.

For a fresh launch that prepares the characteristic cache from the raw source first, omit `--prepared`:

```powershell
& $studyPython -B tools/run_characteristic_study.py `
  --parent .runs/protocol_v1_1/prepared/8f3f4eb44b399771 `
  --linear-registry .runs/protocol_v1_1/linear/full_3314fb5ca2fe170a/registry.json `
  --source D:/StockTwits/Data/CRSP/dsf.pkl `
  --workers 6 --threads 2
```

The builder verifies and reuses an existing cache with the same identity. These two launch forms have different controller identities, so resume with the **same command originally used**, including its prepared-cache choice. After an interruption, completed matching monthly checkpoints are validated and reused; an unfinished month/target job is recomputed. Previous attempt logs and status are retained. Preparation and the implementation pilot may be reverified before training resumes.

Do not edit the frozen experiment JSON, controller, characteristic modules or reused scientific kernels while the study runs. Source changes invalidate its identity or publication checks. Work on unrelated project files can continue. Dependency upgrades, alternative grids or new timing conventions belong in a separately declared experiment.

## Progress and completion

The controller prints `STUDY <directory>` and writes `.runs/characteristics_v1/study/<identity>/status.json`. The September 19 prepared-cache launch is `d20631f4712e3eb5`:

```powershell
$studyStatus = '.runs/characteristics_v1/study/d20631f4712e3eb5/status.json'
$studyState = Get-Content -LiteralPath $studyStatus -Raw | ConvertFrom-Json
$studyState | Select-Object status, phase, pid, child_pid, updated_at, log, error
if ($studyState.log) { Get-Content -LiteralPath $studyState.log -Tail 15 }
```

Check both the controller/child processes and timestamp when investigating an interruption; a stale `running` status alone does not prove the process is alive. Status timestamps use UTC. Heartbeats normally update every ten seconds during supervised child phases; hashing and final controller-side verification can take longer between updates.

The phases are preparation/verification, a two-month implementation pilot (January 2014 and December 2022), full linear fitting, prediction verification, evaluation and reporting. Pilot test performance does not select the design. Attempt logs are `prepare_attempt_*.log`, `pilot_attempt_*.log`, `linear_attempt_*.log`, `evaluation_attempt_*.log` and `report_attempt_*.log`; the preparation log is absent when `--prepared` is supplied.

Completion requires `status: complete`, 264 valid model specifications, complete evaluation and a nonempty report. The controller records output hashes and checks them if the completed command is rerun.

Outputs:

- `reports/09_characteristic_linear_results.md`: full scientific report.
- `reports/data/characteristics_v1_linear*.csv`, compressed `_daily.csv.gz`, and `.json`: comparisons, diagnostics and provenance. The uncompressed daily table remains local.
- Study `linear/full_<fingerprint>/registry.json`, `reuse_audit.json`, `monthly_selection.csv`, prediction files, `checkpoints/` and `complete.json`: model and recovery artifacts. These stay under `.runs/`.
- Study `study_spec.json`, `code/`, attempt logs and `status.json`: frozen launch identity and execution history.

## Resource planning

Six workers with two numerical threads each were chosen to leave room for other work. The previous largest linear job used about 5.44 GB private memory; the new union grows from 439 to 473 inputs, so this is a baseline rather than a new peak-memory measurement. Preserve headroom for shared input pages, the controller, evaluation and the operating system. Keep substantial free disk space: the 168 new full prediction series alone are roughly 20 GB, in addition to checkpoints, preparation and evaluation artifacts.

The old 648 linear jobs sum to 4.56 worker-hours, an idealized 46 minutes with six perfectly balanced workers. Their mean job times were 26.2 seconds for F504, 18.8 for F252 and 30.9 for F756. The new inputs and additional feature sets increase fitting work; hashing, publication and evaluation add time. Treat this as a baseline for updating an estimate from the new pilot and saved-job timings, not a promised finish time. These are linear Gram-matrix fits and do not inherit the multi-day neural-network runtime.
