"""Verify transfer hashes and optionally restore two local NN3 reference ensembles."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import (critical_runtime, data_identity, ensure_vendor, hash_file,
                     runtime_info, TRAINING_VENDOR, VENDOR, verify_bundle, write_json)


def compare(reference, actual, name, *, atol=1e-6):
    import numpy as np
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or not np.allclose(reference, actual, atol=atol, rtol=0, equal_nan=False):
        raise ValueError(f"Bridge {name} mismatch (absolute tolerance {atol})")
    return float(np.max(np.abs(reference.astype(float)-actual.astype(float)))) if reference.size else 0.


def bridge(root, manifest, threads):
    ensure_vendor()
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from protocol_data import load_bundle, make_block
    from protocol_nn import predict, restore_network
    from .migration import legacy_entries, load_legacy, certified_path
    from .common import read_json
    entries = legacy_entries(root)
    if not entries:
        return {"status": "not_needed_no_legacy", "threads": threads, "cases": []}
    if critical_runtime(manifest["source_runtime"]) != critical_runtime(runtime_info()):
        raise ValueError("Runtime releases differ from the exported source environment. Install the pinned environment or train fresh with --no-reuse-local; checkpoint reuse is disabled.")
    references = ("raw|core|504|2014-01", "raw|textcore|504|2022-12")
    if any(key not in entries for key in references):
        raise ValueError("The export lacks the two registered bridge cases; re-export with local pilot checkpoints or train fresh.")
    bundle = load_bundle(Path(root)/"prepared")
    cases = []
    torch.set_num_threads(threads)
    for key in references:
        entry = entries[key]
        spec = read_json(certified_path(root, entry["specification"]))
        task = next(t for t in spec["tasks"] if t["month"] == key.split("|")[-1])
        spec, payload = load_legacy(root, entry, task, bundle)
        saved = payload["result"]
        with threadpool_limits(limits=threads):
            block = make_block(bundle, task, spec["feature_set"], spec["target"])
            if block["feature_names"] != saved["feature_names"] or not np.array_equal(block["test_indices"], saved["test_indices"]):
                raise ValueError("Bridge feature ordering or row identity mismatch")
            errors = {name: compare(saved[name], block[name], name) for name in ("mean", "scale", "constant")}
            chosen = next(s for s in saved["model_states"] if s["penalty"] == saved["chosen_penalty"])
            xtest = torch.from_numpy(np.asarray(block["X_test"], dtype=np.float32))
            predictions = [predict(restore_network(s["state_dict"], xtest.shape[1], spec["config"]["widths"]), xtest) for s in chosen["seeds"]]
            errors["prediction"] = compare(saved["prediction"], np.stack(predictions).mean(axis=0), "prediction")
            cases.append({"key": key, "rows": len(xtest), "source_checkpoint": entry["checkpoint"],
                          "max_absolute_errors": errors, "absolute_tolerance": 1e-6})
        print(f"BRIDGE {key}: passed; maximum prediction difference {errors['prediction']:.3g}", flush=True)
        del block, xtest, predictions, payload, saved
    return {"status": "passed", "threads": threads, "cases": cases,
            "scope": "Input/scaler and restored-ensemble inference only; no test outcomes or new training."}


def verify(root, *, run_bridge=False, threads=2):
    if threads < 1:
        raise ValueError("Threads must be positive")
    root = Path(root).resolve()
    manifest = verify_bundle(root, full_hash=True)
    receipt = {"verified_at": datetime.now(timezone.utc).isoformat(),
               "bundle_manifest_sha256": hash_file(root/"bundle_manifest.json"),
               "data_identity": data_identity(root), "runtime": runtime_info(),
               "critical_runtime": critical_runtime(runtime_info()),
               "vendor_sha256": {n: hash_file(VENDOR/n) for n in TRAINING_VENDOR},
               "file_count": len(manifest["files"]), "bridge": {"status": "not_requested"}}
    # Invalidate previous runtime approval before attempting a new bridge.
    write_json(receipt, root/"verification.json")
    if run_bridge:
        receipt["bridge"] = bridge(root, manifest, threads)
        write_json(receipt, root/"verification.json")
    print(f"VERIFIED {receipt['file_count']} files; bridge={receipt['bridge']['status']}", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    verify(args.bundle, run_bridge=args.bridge, threads=args.threads)


if __name__ == "__main__":
    main()
