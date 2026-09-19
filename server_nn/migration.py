"""Audited import of immutable, full-budget local NN3 month checkpoints."""
from pathlib import Path

from .common import (critical_runtime, data_identity, ensure_vendor, hash_file,
                     read_json, runtime_info, safe_path, TRAINING_VENDOR, VENDOR)

ensure_vendor()
import numpy as np
import pandas as pd
from nn_checkpoint import atomic_pickle, fingerprint
from protocol_nn import validate_checkpoint


def legacy_key(model, month):
    return f"{model['target']}|{model['feature_set']}|{model['fit_days']}|{month}"


def certified_path(root, relative):
    manifest = read_json(Path(root) / "bundle_manifest.json")
    record = manifest["files"].get(relative)
    path = safe_path(root, relative)
    if record is None or not path.is_file() or hash_file(path) != record["sha256"]:
        raise ValueError(f"Uncertified or changed checkpoint input: {relative}")
    return path


def legacy_entries(root):
    path = Path(root) / "legacy_index.json"
    if not path.exists():
        return {}
    return read_json(certified_path(root, "legacy_index.json"))["entries"]


def require_bridge(root, threads):
    root = Path(root)
    if not legacy_entries(root):
        return None
    receipt_path = root / "verification.json"
    error = "Local checkpoint reuse requires: python -m server_nn.verify --bundle BUNDLE --bridge --threads " + str(threads) + "; or explicitly use --no-reuse-local to train fresh."
    if not receipt_path.exists():
        raise ValueError(error)
    receipt = read_json(receipt_path)
    if (receipt.get("bundle_manifest_sha256") != hash_file(root / "bundle_manifest.json")
            or receipt.get("critical_runtime") != critical_runtime(runtime_info())
            or receipt.get("runtime") != runtime_info()
            or receipt.get("data_identity") != data_identity(root)
            or receipt.get("bridge", {}).get("status") != "passed"
            or receipt.get("bridge", {}).get("threads") != threads
            or receipt.get("vendor_sha256") != {n: hash_file(VENDOR/n) for n in TRAINING_VENDOR}):
        raise ValueError("Verification receipt is stale or belongs to another runtime. " + error)
    return receipt


def compatible_spec(spec, model, task, config, prepared_manifest):
    if any(spec.get(k) != model[k] for k in ("estimator", "feature_set", "target", "fit_days")):
        return False
    if (spec.get("protocol") != "v1.1" or spec.get("training_budget") != "standard"
            or spec.get("validation_days") != 126 or spec.get("horizon") != 1
            or spec.get("prepared") != prepared_manifest or spec.get("config") != config
            or task not in spec.get("tasks", [])
            or spec.get("code_sha256") != {n: hash_file(VENDOR/n) for n in TRAINING_VENDOR}):
        return False
    current = critical_runtime(runtime_info())
    return all(str(v).split("+")[0] == current.get(k) for k, v in spec.get("versions", {}).items()) and set(spec.get("versions", {})) == {"python", "torch", "numpy", "pandas"}


def load_legacy(root, entry, task, bundle):
    spec = read_json(certified_path(root, entry["specification"]))
    path = certified_path(root, entry["checkpoint"])
    payload = pd.read_pickle(path)
    if payload.get("fingerprint") != fingerprint(spec) or payload.get("task") != task:
        raise ValueError(f"Legacy checkpoint specification mismatch: {path}")
    result = payload["result"]
    rows = np.flatnonzero((bundle["codes"] >= task["test_first"]) & (bundle["codes"] <= task["test_last"]))
    validate_checkpoint(result, task, rows)
    states = result.get("model_states", [])
    if [c["penalty"] for c in states] != spec["config"]["penalties"]:
        raise ValueError("Legacy checkpoint lacks the complete candidate states")
    for candidate in states:
        if [s["seed"] for s in candidate["seeds"]] != spec["config"]["seeds"]:
            raise ValueError("Legacy checkpoint lacks all five saved seed states")
        if any(not np.isfinite(v).all() for s in candidate["seeds"] for v in s["state_dict"].values()):
            raise ValueError("Nonfinite saved network state")
    return spec, payload


def try_import_checkpoint(bundle_root, model, task, config, destination, signature, bundle):
    if model["estimator"] != "nn3":
        return None
    entry = legacy_entries(bundle_root).get(legacy_key(model, task["month"]))
    if entry is None:
        return None
    receipt = require_bridge(bundle_root, config["threads"])
    spec, payload = load_legacy(bundle_root, entry, task, bundle)
    if not compatible_spec(spec, model, task, config, bundle["manifest"]):
        print(f"FRESH {model['model_id']} {task['month']}: indexed local checkpoint has different training settings or provenance", flush=True)
        return None
    provenance = {"source_checkpoint": entry["checkpoint"],
                  "source_sha256": hash_file(safe_path(bundle_root, entry["checkpoint"])),
                  "source_fingerprint": payload["fingerprint"], "source_versions": spec["versions"],
                  "destination_runtime": runtime_info(), "bridge": receipt["bridge"],
                  "bundle_manifest_sha256": receipt["bundle_manifest_sha256"],
                  "note": "Saved local predictions preserved; bridge checks inference portability, not identical future training across platforms."}
    atomic_pickle({"fingerprint": signature, "task": task, "result": payload["result"],
                   "import_provenance": provenance}, destination)
    return payload["result"]
