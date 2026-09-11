"""Headless notebook runner.

Executes a notebook's code cells in order, in one namespace, exactly as the kernel would, so a
long job can run detached from Jupyter and be logged line by line. Two shims cover the only
notebook-specific names the pipeline uses: `display()` prints, and `tqdm.notebook` becomes a
one-line-per-item progress printer. The notebook file itself is never modified.

Usage:
    python -u tools/run_notebook.py "<path to .ipynb>" [--dry-run] [--stop-at "<section title prefix>"]

    --dry-run       list the code cells (with the section each belongs to) and exit
    --stop-at X     stop before the first cell whose "## " section title starts with X

Example (detached, logged, low priority) from PowerShell:
    $p = Start-Process python -ArgumentList '-u','tools/run_notebook.py','"02 - prepare training dataset/build_text_master.ipynb"' `
         -RedirectStandardOutput run.log -RedirectStandardError run.err -PassThru; $p.PriorityClass='BelowNormal'
"""
import datetime as dt
import os
import sys
import time
import traceback
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + MKL both bundle OpenMP on this machine

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    print(__doc__)
    sys.exit(2)
NOTEBOOK = Path(args[0]).resolve()
DRY_RUN = "--dry-run" in sys.argv
STOP_AT = sys.argv[sys.argv.index("--stop-at") + 1] if "--stop-at" in sys.argv else None


def log(msg=""):
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ---- shims ------------------------------------------------------------------------------------
import tqdm as _tqdm_pkg  # noqa: E402


def _line_tqdm(iterable=None, desc="", total=None, **kw):
    if iterable is None:
        return _tqdm_pkg.std.tqdm(iterable, desc=desc, total=total, disable=True, **kw)
    items = list(iterable)
    t0 = time.time()
    for i, it in enumerate(items, 1):
        log(f"{desc}: {i}/{len(items)} -> {getattr(it, 'name', it)}  (elapsed {(time.time() - t0) / 60:.1f} min)")
        yield it


_shim = types.ModuleType("tqdm.notebook")
_shim.tqdm = _line_tqdm
sys.modules["tqdm.notebook"] = _shim


def display(obj):
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            print(f"[DataFrame {obj.shape[0]} x {obj.shape[1]}]")
            print(obj.to_string(max_cols=14, max_rows=20))
            return
    except Exception:
        pass
    print(obj)


# ---- load and run -----------------------------------------------------------------------------
import nbformat  # noqa: E402

nb = nbformat.read(NOTEBOOK, as_version=4)
section = "preamble"
cells = []
for i, c in enumerate(nb.cells):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        first = src.strip().splitlines()[0] if src.strip() else ""
        if first.startswith("## "):
            section = first[3:].strip()
    elif c["cell_type"] == "code":
        cells.append((i, section, src))

log(f"Runner start. notebook={NOTEBOOK}  pid={os.getpid()}  python={sys.executable}")
log(f"{len(cells)} code cells: " + ", ".join(str(i) for i, _, _ in cells))
if DRY_RUN:
    for i, sec, src in cells:
        print(f"--- cell {i}  [{sec}]  {len(src.splitlines())} lines")
    sys.exit(0)

os.chdir(NOTEBOOK.parent)
sys.path.insert(0, str(NOTEBOOK.parent))   # like a kernel: the notebook's folder is importable (latex_table, trading_library)
ns = {"__name__": "__main__", "display": display}
t_start = time.time()
for i, sec, src in cells:
    if STOP_AT and sec.startswith(STOP_AT):
        log(f"--stop-at '{STOP_AT}': stopping before cell {i} [{sec}].")
        break
    log(f"===== cell {i}  [{sec}] =====")
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception:
        log(f"!!!!! cell {i} FAILED after {(time.time() - t0) / 60:.1f} min")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    log(f"----- cell {i} done in {(time.time() - t0) / 60:.1f} min (total {(time.time() - t_start) / 3600:.2f} h)")
log("Runner finished successfully.")
