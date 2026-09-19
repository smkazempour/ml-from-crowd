# Run the neural-network study on the server

This package completes the current NN3 study and compares network depth, with a
separate optional width study. It uses a frozen copy of the existing training and
evaluation code. The active Windows run can continue while you export and use this
package; server outputs belong in their own directory.

The machine-readable [study plan](study_plan.json) records the architecture menu,
training rules, phase sizes and comparison families before server training. The
bundle manifest certifies that plan together with the code and input files.

The primary instructions assume a Linux server accessed through SSH. A Slurm
template and a Windows PowerShell launcher are also included. These are prepared
instructions, not evidence that the code has already run on your server.

## What the server will run

| Phase | Hidden-layer widths | Fitting windows | Full specifications | Additional specifications after NN3 |
|---|---|---|---:|---:|
| `nn3` | 128, 64, 32 | 504, 252, 756 sessions | 24 | Completes the current study |
| `depth` | 128; 128, 64; 128, 64, 32; 128, 64, 32, 16 | 504 sessions | 32 | 24 |
| `width` (optional) | 64, 32, 16; 128, 64, 32; 256, 128, 64 | 504 sessions | 24 | 16 |

Each phase crosses its architectures/windows with four feature sets (`core`,
`textcore`, `all`, `textall`) and raw/DGTW targets. Each specification has 108 monthly
test jobs, January 2014 through December 2022. Every month retains all three L2
penalties and five seeds, with a 100-epoch ceiling and validation early stopping.
The same rank transformations, equal-date loss, 126-session validation window,
timing exclusions, targets and evaluation procedure apply.

NN3 has the same identity in all three phases, so compatible checkpoints already
in the server output are reused. The default `nn3` + `depth` sequence contains 48
unique specifications: 5,184 monthly jobs / 77,760 candidate-seed fits before
subtracting reusable local checkpoints. All three phases contain 64 unique
specifications: 6,912 monthly jobs / 103,680 candidate-seed fits. Job counts are not
runtime estimates: input size, window length and stopping epochs matter.

The first-layer width is fixed at 128 in the depth study. This differs from the old
unused NN1/NN2 presets in the local script. Parameter counts still change with
depth; the reports include them. The width study changes capacity at fixed depth.

## 1. Export a self-contained transfer folder on this machine

Run this in PowerShell from the repository. Choose a new destination for each
snapshot; the exporter does not alter the active study or source data.

```powershell
Set-Location -LiteralPath 'D:\StockTwits\Code'
$studyPython = 'C:\Users\skazempour\AppData\Local\anaconda3\envs\py313\python.exe'
& $studyPython -B -m server_nn.export --output '.runs\server_nn_bundle'
if ($LASTEXITCODE -ne 0) { throw 'Export failed' }
```

The resulting directory contains:

- `server_nn/`: runnable package, launchers, pinned requirements and frozen kernels.
- `prepared/`: the corrected immutable feature/target cache and evaluation data.
- `linear/`: all 96 completed linear prediction series and their registry.
- `legacy_nn/`: completed local NN3 checkpoints available at export time and the
  saved pilot material used to check portability.
- `legacy_index.json`: import-candidate locations and any excluded local runs.
- `reference_reports/`: available project methodology, linear results and research
  context copied by the exporter.
- `bundle_manifest.json`: file sizes, SHA-256 hashes and provenance.

The prepared inputs occupy approximately 8.15 GB and linear predictions 11.2 GB,
plus NN checkpoints and code. The exporter reports the actual total. Plan for at
least 100 GB of free server disk for the bundle, environment, new checkpoints,
predictions and reports; optional width runs and repeated benchmarks need more.
Use a persistent data volume with local SSD performance when available.

Copy the **entire exported folder**, including its manifest. For example, after
replacing the account, host and destination:

```powershell
scp -r '.runs\server_nn_bundle' 'YOUR_USER@YOUR_SERVER:/YOUR/PERSISTENT/PATH/'
```

An interrupted transfer can be resumed with your normal transfer tool. The
verification step below checks every exported file, so it also detects truncated
copies. Raw tweet files and the original `D:\StockTwits\Data` tree are not needed.

The export is a point-in-time snapshot. Jobs completed on this machine afterward
are not automatically synchronized. To include newer completed checkpoints, make
a new export and transfer it into a new bundle directory. Keep the previous
bundle intact for provenance; do not overwrite files while a server job reads
them. Reusing an existing server output with a newer compatible snapshot can
retain server checkpoints and import additional compatible local checkpoints.

If the default repository pointers are unavailable, provide `--prepared` and
`--linear-registry` explicitly. Repeated `--legacy-root` arguments select local
checkpoint roots; include the local pilot root to supply the two bridge cases.
`--skip-legacy` exports no local NN checkpoints and therefore requires fresh NN
training. It still includes the prepared inputs and completed linear baselines.

## 2. Create the server Python environment

Use Python **3.13.11** for the checkpoint-reuse path. Obtain that interpreter
through the server's supported module, Conda or Python installation; do not copy
the Windows virtual environment to Linux.

```bash
cd /YOUR/PERSISTENT/PATH/server_nn_bundle
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r server_nn/requirements.txt
python -m pip check
```

The CPU index provides PyTorch 2.14.0 wheels for Python 3.13 on Linux x86-64 and
Windows. These runs use CPU training; no CUDA installation is required.
[Official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/),
[PyTorch installation guidance](https://pytorch.org/get-started/locally/).

The requirements pin the versions measured in the active local environment. A
`+cpu` build suffix does not change the public release comparison, but the
portability check below must still pass. If an exact wheel/interpreter is
unavailable for your server, do not silently substitute versions and import old
checkpoints. Use the explicit fresh-run procedure later in this guide.

## 3. Check the allocation, transferred files and checkpoint portability

Run the resource check **inside the intended compute allocation**, not on a
cluster login node. It uses the process CPU affinity, Slurm allocation and
available/cgroup-limited memory where available.

```bash
python -B -m server_nn.check_environment --workers 24 --threads 2
python -B -m server_nn.verify --bundle .
python -B -m server_nn.verify --bundle . --bridge --threads 2
```

The initial 24-worker / 2-thread setting uses 48 compute threads. The conservative
planning allowance is 6 GiB per worker plus 24 GiB shared/headroom: **168 GiB**.
200 GB of physical RAM is about 186 GiB, and the operating system or scheduler
may make less available. This allowance comes from previous large-block
profiling; the NN benchmark establishes the actual peak. It is not a guarantee
that every 24-worker configuration fits.

The bridge restores saved pilot networks on the server, rebuilds their inputs,
and compares their predictions and scalers with the exported references. Exact
software releases alone do not guarantee identical arithmetic across operating
systems and libraries. A successful check permits supported NN3 checkpoint
imports; a failed check stops reuse. Saved prediction reconstruction is a
portability check, not a guarantee of bitwise-identical future training across
machines. Runtime and checkpoint provenance remain recorded.

The two reference cases are raw-target, F504 core in January 2014 and text + core
in December 2022. Feature order and row positions must match exactly; scalers and
restored ensemble predictions must agree within absolute tolerance **0.000001**
with zero relative tolerance. This is an explicit numerical portability tolerance,
not a claim of exact cross-platform prediction equality.

Verification writes `verification.json` in the bundle. A hash-only verification
overwrites any previous bridge approval with `not_requested`. Therefore run
`verify --bridge --threads 2` **last** before importing local checkpoints. The
launchers already do this. After changing the environment, bundle or thread
setting, rerun the bridge. With no exported local checkpoints, it reports
`not_needed_no_legacy` and no checkpoint import is attempted.

## 4. Benchmark before committing the server for days

Start with the same complete-budget, six-job pilot used locally:

```bash
python -B -u -m server_nn.benchmark --bundle . --output benchmark_runs \
  --profile core_textcore --workers 3 --threads 2
```

This fits core/text + core for January 2014, March 2015 and December 2022 with all
15 candidate-seed fits per month. It measures full-budget training time, active
concurrency and process memory and writes checkpoints. Then measure the larger
data block:

```bash
python -B -u -m server_nn.benchmark --bundle . --output benchmark_runs \
  --profile largest --workers 3 --threads 2
```

Three or six tasks cannot establish throughput at 24 workers. The optional
resource sweep creates enough distinct tasks to keep the candidate worker counts
busy and repeats the same workload for each configuration. It can itself take
hours. Begin with 16 and 24 workers; consider 32 only after the measured memory
budget supports it.

```bash
python -B -u -m server_nn.benchmark --bundle . --output benchmark_sweep \
  --profile largest --resource-sweep --workers 16 24 --threads 2
```

Compare completed jobs per hour and peak memory, not just CPU utilization. The
sum of process RSS can count shared mappings more than once; use the recorded
memory measures together with the scheduler's job/cgroup memory accounting.
Higher thread counts are an optional fresh-runtime benchmark, but changing the
two-thread setting prevents direct reuse of the original local checkpoints and
creates a different server run identity. Keep two threads unless the measured
benefit justifies doing that work again.

Both training and benchmarking enforce the same resource preflight. If profiling
supports a different memory allowance, pass `--memory-per-worker-gib` and
`--reserve-gib` explicitly. The launchers accept `MEMORY_PER_WORKER_GIB` and
`RESERVE_GIB` on Linux, or `-MemoryPerWorkerGiB` and `-ReserveGiB` on Windows.
Keep headroom for shared data, the parent process and evaluation; do not lower
these estimates merely to bypass the allocation check.

Before the full depth comparison, its full-budget pilot can test all candidate
architectures for a selected month:

```bash
python -B -u -m server_nn.train --bundle . --output runs --phase depth \
  --months 2014-01 --workers 24 --threads 2
```

This has 32 monthly jobs and is explicitly labeled a pilot. Those compatible
checkpoints are reused by the subsequent full run. The architecture menu and
training budget are fixed before examining test-period results. Do not select
architectures from these pilot test returns.

Repeat that exact pilot command to verify real-data resume before the long run.
It should report `resume` actions for its completed months, with no new training.
The separate portability bridge checks restored reference predictions; the
benchmark itself does not perform a checkpoint-reconstruction or replay test.

## 5. Launch the study

### Dedicated Linux server or interactive compute allocation

The launcher checks resources and compatibility, finishes NN3, evaluates it,
then runs and evaluates the depth study. It exits on the first failed command.

```bash
cd /YOUR/PERSISTENT/PATH/server_nn_bundle
export PYTHON="$PWD/.venv/bin/python"
export WORKERS=24 THREADS=2 OUTPUT=runs
nohup bash server_nn/run_local.sh > server-study.log 2>&1 < /dev/null &
echo "$!" > server-study.pid
```

This detaches from the SSH terminal. On a managed cluster, use the scheduler
instead of `nohup` on a login node. A `tmux` session is another option for a
dedicated server. Run only one study launcher per output directory.

### Slurm

The template requests one node, one task, 64 CPUs and 180 GiB, with a three-day
wall-time limit. Adjust the resources, wall time and site-required account or
partition before submitting. The code parallelizes within one node; requesting
multiple nodes does not distribute these workers. Slurm does not copy the bundle
to the compute node for you. [Slurm `sbatch` documentation](https://slurm.schedmd.com/sbatch.html).

```bash
cd /YOUR/PERSISTENT/PATH/server_nn_bundle
export BUNDLE="$PWD"
export PYTHON="$PWD/.venv/bin/python"
export WORKERS=24 THREADS=2 OUTPUT=runs
sbatch server_nn/run_slurm.sh
```

Add your site's required `--account=...` or `--partition=...` options to `sbatch`.
The log is `stocktwits-nn-JOBID.log` in the submission directory. If the scheduler
wall time expires, resubmit the same command and output directory; completed
monthly checkpoints resume. The incomplete current monthly jobs are retrained.
The script does not automatically submit additional scheduler jobs.

### Windows server

Create a Python 3.13.11 environment on that server and install the same packages:

```powershell
Set-Location -LiteralPath 'D:\YOUR_PATH\server_nn_bundle'
py -3.13 --version
py -3.13 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r server_nn/requirements.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\server_nn\run_windows.ps1' -Python '.\.venv\Scripts\python.exe' -Workers 24 -Threads 2
```

Confirm the first command prints `Python 3.13.11`; the `-3.13` launcher selector
alone does not guarantee the patch release. If needed, invoke the full path to
the installed 3.13.11 interpreter when creating the environment.

That last command runs in the current session. To keep it separate from an
interactive PowerShell window, use a hidden background process after setup:

```powershell
$bundlePath = (Get-Location).Path
$launcherPath = Join-Path $bundlePath 'server_nn\run_windows.ps1'
$pythonPath = Join-Path $bundlePath '.venv\Scripts\python.exe'
$launchArguments = @('-NoProfile', '-File', ('"' + $launcherPath + '"'),
                     '-Bundle', ('"' + $bundlePath + '"'),
                     '-Python', ('"' + $pythonPath + '"'), '-Workers', '24', '-Threads', '2')
$studyProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList $launchArguments `
    -WorkingDirectory $bundlePath -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $bundlePath 'server-study.stdout.log') `
    -RedirectStandardError (Join-Path $bundlePath 'server-study.stderr.log')
$studyProcess.Id
```

A server policy can still terminate processes at logout or reboot. Use the
server's supported persistent-job mechanism if required, and resume from the
same output directory after an interruption.

## Progress, results and restart

- The training heartbeat is `runs/status.json`. It reports the current phase,
  model, completed monthly jobs, process ID, and any failure. It is a training
  status; evaluation has its own completion marker.
- Full prediction registries are `runs/registries/nn3.json`, `depth.json` and,
  when requested, `width.json`. A pilot cannot publish one of these full aliases.
- The readable results are `runs/reports/PHASE/results.md` and
  `detailed_results.md`. Evaluation CSVs, provenance, source copies and
  `complete.json` sit alongside them.
- Monthly model directories hold the saved seed states, validation choices,
  scalers and predictions. Resuming checks identity and coverage before reuse.

For a Linux log/status check:

```bash
tail -n 30 server-study.log
cat runs/status.json
```

If a training process stops, restart the same launcher with the same bundle,
output, software environment and thread count. Worker count may be reduced
without changing the mathematical specification. If only evaluation failed,
rerun it directly after addressing the reported error:

```bash
python -B -u -m server_nn.evaluate --bundle . --output runs --phase nn3
python -B -u -m server_nn.evaluate --bundle . --output runs --phase depth
```

Do not edit the transferred package while it is running. Keep development on this
machine separate from the server's frozen bundle. Do not point both machines at
the same mutable output directory, and do not copy server checkpoints into the
active local study's folders by hand. Returning the complete `runs/` directory
preserves saved models and provenance. For a compact first review, copy the
`runs/reports/` tree and its CSVs; keep the full bundle and output on the server
because report registries reference the prediction files.

## Optional extensions

To run only depth, or add the width study after the defaults complete:

```bash
PHASES=depth bash server_nn/run_local.sh
PHASES=width bash server_nn/run_local.sh
```

PowerShell equivalents are `-Phases depth` and `-Phases width`. Width is never
launched by the default scripts.

The optional adaptive-depth comparison chooses the architecture each month by
validation IC and uses that architecture's already-selected penalty. It does not
choose from full-period test performance:

```bash
python -B -u -m server_nn.evaluate --bundle . --output runs --phase depth --adaptive
```

It writes a separate `runs/reports/depth_adaptive/` evaluation, including the
monthly selection ledger and separately declared comparison family. Keep the
fixed-architecture report as the primary depth comparison.

If matching the source runtime or passing the portability bridge is impossible,
use a separate output directory and explicitly disable importing local models.
Record and pin the chosen installed environment before the full study. The bundle
hash checks remain mandatory; a fresh run does not excuse corrupted inputs.

```bash
FRESH=1 OUTPUT=runs_fresh bash server_nn/run_local.sh
```

On Windows use `-Fresh -Output runs_fresh`. This permits installed release
differences at preflight and passes `--no-reuse-local` to training. Missing
dependencies still fail. New server checkpoints resume only under their recorded
identity. Report the runtime difference when comparing server results with the
existing linear/local NN results; do not call the mixed environments bitwise
equivalent.

The economic interpretation remains subject to the shared protocol: the sample
is historical development evidence, tweet routing uses the nominal 16:00 Eastern
close even on early-close days, and gross decile spreads do not establish returns
after execution delays or trading costs.
