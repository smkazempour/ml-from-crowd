# Server run handoff — September 13, 2026

The transfer bundle is ready at
[`D:\StockTwits\Code\.runs\server_nn_bundle_20260913`](../.runs/server_nn_bundle_20260913).
It contains **496 certified files, 20.20 GB**, including the corrected prepared
data, all 96 linear prediction files, and **349 finished NN3 monthly checkpoints**.
Copy this entire folder to the server; the original raw-data folders are not needed.

Start with the **transfer command near the end of step 1** in
[the server instructions](../server_nn/README.md), then continue to step 2.
The export command has already been completed for this snapshot. The same instructions are
inside the transferred folder at `server_nn/README.md`. They cover Linux over SSH,
Slurm, and Windows Server. Install the pinned environment, run verification and
the portability bridge on the server, then benchmark before the full launch.
The verification receipt produced here does not replace verification on the server.

The default launcher completes the existing NN3 study at all three fitting windows
and then compares NN1 `[128]`, NN2 `[128, 64]`, NN3 `[128, 64, 32]`, and
NN4 `[128, 64, 32, 16]` at the primary 504-session fitting window. It uses
24 workers with two threads each, subject to the server resource check.
Compatible NN3 checkpoints are shared across phases. Width sensitivity and
validation-selected adaptive depth are optional and have separate instructions.
The [frozen study plan](../server_nn/study_plan.json) records the settings and comparisons.

After environment setup and benchmarking, a Linux launch from the bundle directory is:

```bash
PYTHON="$PWD/.venv/bin/python" WORKERS=24 THREADS=2 OUTPUT=runs \
  nohup bash server_nn/run_local.sh > server-study.log 2>&1 &
```

Use the Slurm template instead if the server requires scheduled jobs. On Windows,
use `server_nn/run_windows.ps1` as described in the instructions. Results appear
in `runs/reports/nn3/results.md` and `runs/reports/depth/results.md`; checkpoints,
registries and validation diagnostics stay under `runs/`. Preserve that directory
and the bundle for restart and reproducibility.

## Validation completed here

- 37 unit/fixture tests passed, including real small-network fitting, restart,
  complete 108-month synthetic evaluation, model-matrix checks and corrupt-transfer rejection.
- All 496 exported files passed SHA-256 verification from the copied package.
- Saved core and text-plus-core ensembles reproduced 13,159 and 33,260 predictions
  exactly. The declared server portability tolerance is absolute `1e-6`.
- Actual checkpoint import, partial publication, and restart passed. A separate
  two-worker dispatch check imported and resumed both references without changing
  checkpoint bytes or timestamps. These checks did not fit new production models.
- Bash syntax and PowerShell launcher checks passed. No remote server job has been
  launched; actual Linux/Slurm execution and server performance remain to be checked there.

The [machine-readable handoff record](data/server_nn_handoff.json) contains the
bundle hash and detailed verification results. The local controller and training
child were still running at handoff, and all eight protected source hashes were
unchanged. Develop other parts of the project here while keeping those active
sources fixed. The server uses its own copied code and outputs.

This export captures only completed checkpoints available when it was made.
Later local progress is not synchronized automatically; make a new export into a
new folder if another snapshot is needed. Do not merge checkpoints into either
active run by hand.
