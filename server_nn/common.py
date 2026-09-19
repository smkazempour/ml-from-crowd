"""Portable identities and frozen experiment definitions; no source-data writes."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import uuid

PACKAGE = Path(__file__).resolve().parent
VENDOR = PACKAGE / "vendor"
FEATURES = ("core", "textcore", "all", "textall")
TARGETS = ("raw", "dgtw")
ARCHITECTURES = {"nn1": [128], "nn2": [128, 64], "nn3": [128, 64, 32],
                 "nn4": [128, 64, 32, 16], "nn3_narrow": [64, 32, 16],
                 "nn3_wide": [256, 128, 64]}
PHASES = {"nn3": (["nn3"], [504, 252, 756]),
          "depth": (["nn1", "nn2", "nn3", "nn4"], [504]),
          "width": (["nn3_narrow", "nn3", "nn3_wide"], [504])}
TRAINING_VENDOR = ("protocol_nn.py", "protocol_nn_metrics.py", "protocol_data.py",
                   "prediction_metrics.py", "nn_checkpoint.py")
PREPARED_FILES = ("X.npy", "y.npy", "q.npy", "codes.npy", "calendar.npy", "keys.pkl",
                  "manifest.json", "evaluation.pkl", "horizon_repair_audit.json")


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_vendor():
    snapshot = read_json(VENDOR / "snapshot.json")["sha256"]
    for name, digest in snapshot.items():
        if hash_file(VENDOR / name) != digest:
            raise ValueError(f"Frozen vendor source changed: {name}")
        loaded = sys.modules.get(Path(name).stem)
        if loaded is not None and Path(loaded.__file__).resolve() != (VENDOR / name).resolve():
            raise RuntimeError(f"{name} was imported from another checkout; start a fresh Python process")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return VENDOR


def runtime_info():
    packages = {}
    for name in ("numpy", "pandas", "torch", "scipy", "scikit-learn", "joblib",
                 "threadpoolctl", "filelock", "psutil"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": packages}


def critical_runtime(info):
    """Public package releases; platform differences are checked by the bridge."""
    return {"python": info["python"], **{name: str(info["packages"].get(name)).split("+")[0]
            for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}}


def training_hashes():
    ensure_vendor()
    result = {"vendor/" + name: hash_file(VENDOR / name) for name in TRAINING_VENDOR}
    for name in ("common.py", "train.py", "migration.py"):
        result[name] = hash_file(PACKAGE / name)
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = str(relative)
    if "\\" in relative or ":" in relative:
        raise ValueError(f"Nonportable path in bundle: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes bundle: {relative}")
    return path


def verify_bundle(root, full_hash=True):
    root = Path(root).resolve()
    manifest = read_json(root / "bundle_manifest.json")
    if manifest.get("schema") != "stocktwits_server_bundle_v1":
        raise ValueError("Unsupported or incomplete server bundle")
    for relative, record in manifest["files"].items():
        path = safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or truncated bundle file: {relative}")
        if full_hash and hash_file(path) != record["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {relative}")
    for name in PREPARED_FILES:
        if "prepared/" + name not in manifest["files"]:
            raise ValueError(f"Required prepared input is not certified: {name}")
    ensure_vendor()
    for name in TRAINING_VENDOR:
        if manifest["vendor_sha256"][name] != hash_file(VENDOR / name):
            raise ValueError(f"Bundle and executing training kernel differ: {name}")
    return manifest


def data_identity(root):
    manifest = read_json(Path(root) / "bundle_manifest.json")
    return {"prepared_fingerprint": manifest["prepared_fingerprint"],
            "files": {name: record["sha256"] for name, record in sorted(manifest["files"].items())
                      if name.startswith("prepared/")}}


def training_config(widths, threads):
    if int(threads) < 1:
        raise ValueError("Threads must be positive")
    return {"widths": list(widths), "seeds": [7, 1007, 2007, 3007, 4007],
            "penalties": [1e-5, 1e-4, 1e-3], "threads": int(threads), "max_epochs": 100,
            "patience": 5, "learning_rate": .001, "batch_size": 10000, "verbose": True,
            "objective": "0.5 equal-date MSE + 0.5 lambda sum Linear weights squared; biases/BN excluded",
            "selection": "mean daily validation IC of mean seed predictions"}


def phase_models(phase):
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    architectures, windows = PHASES[phase]
    return [{"model_id": f"{arch}_{feature}_{target}_fit{fit}_val126", "estimator": arch,
             "feature_set": feature, "target": target, "fit_days": fit,
             "widths": ARCHITECTURES[arch]}
            for fit in windows for target in TARGETS for feature in FEATURES for arch in architectures]
