# Running the linear optimization study

The design is in [LINEAR_OPTIMIZATION_EXPERIMENT.md](../reports/LINEAR_OPTIMIZATION_EXPERIMENT.md).
The exact declaration is `reports/data/linear_optimization_v3_experiment.json`.
This is a separate, resumable experiment; Report 10 and its checkpoints remain unchanged.

From the repository root, use the existing numerical environment:

```powershell
& 'C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe' -B tools/run_linear_optimization_study.py --workers 4 --threads 2
```

The controller runs the two-month/two-outcome implementation pilot, all 216
monthly outcome jobs, forecast certification, evaluation, and detailed/readable
reports. It attaches the independent completion/error watcher automatically.
The process prints `STUDY <path>`; inspect `<path>/status.json` and the phase log
named there. `SAVED` lines mark completed monthly checkpoints. A heartbeat alone
does not prove that fitting has advanced.

Repeat the same command after an interruption. Identity-checked checkpoints are
reused. Do not start a second controller while the existing controller is alive.
The study and checkpoint locks prevent simultaneous ownership. A changed source,
specification, input or numerical thread setting creates a different run identity.

Required local inputs (large artifacts are not included in Git):

- `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/`
- `.runs/linear_design_v2/study/c11478141261534f/linear/full_9590a962af3e2fcb/registry.json`
- All completed v2 monthly checkpoints beside that registry.

The new full registry includes 172 model specifications with certified prediction
files and `monthly_selection.csv`. Monthly checkpoints also retain complete
validation candidate scores, omitted-month sensitivity results and final-fit
transformation metadata. No new near-tie decision rule is applied.

Outputs use `reports/data/linear_optimization_v3*`, Report 11 and its short
companion. Only a `complete` controller status with matching output hashes
certifies completion; fitting alone is insufficient.

The watcher's durable outcome is `reports/LINEAR_OPTIMIZATION_COMPLETION.md`.
Its receipt and logs live beside the study status. Alerts are local Windows
notifications; a machine shutdown also stops the watcher. Restarting the study
starts monitoring the new attempt. `--no-notify` disables alerts for manual
testing. Details of the watcher are in [STUDY_NOTIFICATIONS.md](STUDY_NOTIFICATIONS.md).

Do not edit sources listed in the controller's `SOURCES`, the declaration, or
certified inputs during a run. Other project work can continue independently.
