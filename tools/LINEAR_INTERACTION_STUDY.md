# Run the explicit interaction study

Run from the repository root using the existing `py313` environment:

```powershell
& 'C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe' -B tools/run_linear_interaction_study.py --workers 4 --threads 2
```

The command runs the declared early/late-month pilot and checks the evaluation
and report pipeline before starting the full 216 monthly outcome jobs. It then
verifies every prediction file, evaluates the 148 procedures, and writes both
the complete results register and a readable summary. Settings are declared in
`reports/data/linear_interactions_v1_experiment.json`.

The prepared cache and the certified additive comparison are required locally:

- `.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42`
- `.runs/linear_design_v2/study/c11478141261534f/linear/full_9590a962af3e2fcb/registry.json`

The controller records the exact source, data, specification and numerical
environment under `.runs/linear_interactions_v1/study/<identity>/`. Monthly
checkpoints include predictions, selection diagnostics, feature transformations,
and integrity certificates. Rerun the same command after an interruption to
validate and reuse completed monthly checkpoints. A changed experiment or source
gets a different identity; incompatible checkpoints are not silently reused.

Use a hidden `Start-Process` with separate stdout/stderr logs for an unattended
Windows run. The controller automatically starts the existing independent
completion watcher. `reports/LINEAR_INTERACTION_COMPLETION.md` records the final
outcome, including failures; the watcher also sends a Windows desktop alert.
A machine shutdown stops both processes, so resume the controller after reboot.

The authoritative live record is the study's `status.json`; the current path and
launch details are in `reports/LINEAR_INTERACTION_RUN_STATUS.md`. Desktop alerts
do not imply an automatic message in this chat.

Tests:

```powershell
& 'C:/Users/skazempour/AppData/Local/anaconda3/envs/py313/python.exe' -B -m unittest discover -s tools -p 'test_*linear_interaction*.py' -v
```

The earlier additive studies, prepared cache, and reports stay unchanged.
