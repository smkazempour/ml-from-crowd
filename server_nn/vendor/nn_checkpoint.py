"""Atomic per-month NN checkpoints, keyed by data, code and training configuration."""
import hashlib
import json
import os
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_pickle(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_result(result, task):
    expected = task["t1"]-task["t0"]
    if (result["month"] != task["month"] or tuple(result["rows"]) != (task["t0"], task["t1"])
            or np.asarray(result["pred"]).shape != (expected,)
            or not np.isfinite(result["pred"]).all()):
        raise ValueError(f"Invalid month result for {task['month']}")


def fit_month_checkpoint(task, y_all, codes_all, cfg, fit):
    path = Path(cfg["checkpoint_dir"])/f"{task['month']}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+".lock", timeout=0):
        if path.exists():
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            if payload["fingerprint"] != cfg["fingerprint"] or payload["task"] != task:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            validate_result(payload["result"], task)
            print(f"RESUME {task['month']}: verified checkpoint", flush=True)
            return payload["result"]
        start = time.monotonic()
        result = fit(task, y_all, codes_all, cfg)
        validate_result(result, task)
        result["elapsed_seconds"] = time.monotonic()-start
        atomic_pickle({"fingerprint": cfg["fingerprint"], "task": task, "result": result}, path)
        print(f"SAVED {task['month']}: {result['elapsed_seconds']:.1f}s", flush=True)
        return result
