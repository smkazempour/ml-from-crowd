"""Run isolated early/recent NN3 pilots and optionally verify checkpoint resume.

Use the py313 interpreter. Example:
  <py313>/python.exe tools/run_nn_pilot.py --features core textcore --resume-check
This deliberately caps training. It is an execution/runtime test, not a full sweep.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {"core": "core", "all": "all", "textcore": "embed+norm+core", "textall": "embed+norm+all"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", choices=list(FEATURES), default=["core", "textcore"])
    ap.add_argument("--months", default="2012-01,2022-12")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume-check", action="store_true")
    a = ap.parse_args()
    if min(a.epochs, a.seeds, a.workers, a.threads) < 1 or not a.months.strip():
        ap.error("Positive training settings and explicit pilot months are required")
    run_root = ROOT/".runs"/"text_nn"
    run_root.mkdir(parents=True, exist_ok=True)
    for feature in a.features:
        original = None
        env = os.environ.copy()
        env.update(TEXTNN_ARCH="3", TEXTONLY_FEATURES=FEATURES[feature], RANK_TARGET="1",
                   TARGET_COL="f_cumret1", TEXTNN_RANK_FEATURES="1", TEXTNN_PLACEBO="0",
                   TEXTNN_MONTHS=a.months, TEXTNN_MAX_MONTHS="0", TEXTNN_SEEDS=str(a.seeds),
                   TEXTNN_MAX_EPOCHS=str(a.epochs), TEXTNN_WORKERS=str(a.workers),
                   TEXTNN_THREADS=str(a.threads), TEXTNN_RUN_ROOT=str(run_root),
                   TEMP=str(ROOT/".runs"), TMP=str(ROOT/".runs"), PYTHONIOENCODING="utf-8")
        for attempt in range(1+int(a.resume_check)):
            log = ROOT/".runs"/f"nn3_{feature}_{'resume' if attempt else 'pilot'}.log"
            cmd = [sys.executable, "-B", "-u", str(ROOT/"tools/run_notebook.py"),
                   str(ROOT/"03d - neural network/prediction_neural_network_text.ipynb")]
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
            print(f"Running {feature} ({'resume verification' if attempt else 'pilot'}); log {log}", flush=True)
            with log.open("w", encoding="utf-8") as fh:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags)
            if completed.returncode:
                raise RuntimeError(f"Pilot failed; inspect {log}")
            log_text = log.read_text(encoding="utf-8")
            prediction_path = Path(next(line.removeprefix("Predictions: ").strip()
                                        for line in log_text.splitlines() if line.startswith("Predictions: ")))
            signature = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            checkpoints = {str(p): p.stat().st_mtime_ns for p in prediction_path.parent.glob("months/*.pkl")}
            current = {"prediction": str(prediction_path), "sha256": signature, "checkpoints": checkpoints}
            if attempt:
                if current != original or "SAVED " in log_text:
                    raise AssertionError("Resume changed predictions or rewrote a completed checkpoint")
                if log_text.count("RESUME ") != len(checkpoints):
                    raise AssertionError("Resume did not verify every completed month")
                verification = {**current, "status": "verified", "resume_log": str(log)}
                (prediction_path.parent/"resume_verified.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
                print(f"Verified identical prediction bytes and {len(checkpoints)} untouched checkpoints", flush=True)
            original = current
            print(f"Completed {feature}; {log}", flush=True)
    for marker in sorted(run_root.glob("*/complete.json")):
        completed = json.loads(marker.read_text())
        print(completed["kind"], completed["rows"], completed["prediction_file"])


if __name__ == "__main__":
    main()
