"""Create a self-contained snapshot without modifying the live local NN study."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .common import (PACKAGE, PREPARED_FILES, TRAINING_VENDOR, VENDOR, ensure_vendor,
                     hash_file, read_json, runtime_info, training_config, write_json)


def copy_certified(source, destination):
    """Hash while copying; reject a source replaced or edited during the read."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Source changed during snapshot; use a new export directory: {source}")
    if destination.stat().st_size != before.st_size:
        raise ValueError(f"Incomplete copy: {source}")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def export(output, prepared, linear_registry, legacy_roots):
    ensure_vendor()
    import numpy as np
    import pandas as pd
    from nn_checkpoint import fingerprint
    from protocol_data import load_bundle, make_tasks
    from protocol_nn import validate_checkpoint
    from .migration import compatible_spec, legacy_key
    output, prepared, linear_registry = Path(output).resolve(), Path(prepared).resolve(), Path(linear_registry).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export destination must be new or empty; existing bundles are never overwritten")
    if output == PACKAGE or output.is_relative_to(PACKAGE) or output == prepared or output.is_relative_to(prepared):
        raise ValueError("Export must be outside package and prepared inputs")
    output.mkdir(parents=True, exist_ok=True)
    files = {}

    def copy(source, relative, expected_sha=None):
        record = copy_certified(source, output/relative)
        if expected_sha is not None and record["sha256"] != expected_sha:
            raise ValueError(f"Source checksum differs from its registry: {source}")
        files[relative] = record

    def register(relative, value):
        write_json(value, output/relative)
        files[relative] = {"bytes": (output/relative).stat().st_size, "sha256": hash_file(output/relative)}

    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            copy(path, "server_nn/" + path.relative_to(PACKAGE).as_posix())
    for name in PREPARED_FILES:
        print(f"COPY prepared/{name}", flush=True)
        copy(prepared/name, "prepared/"+name)
    prepared_manifest = read_json(prepared/"manifest.json")
    linear = read_json(linear_registry)
    from .evaluate import validate_matrix
    validate_matrix(linear["models"], "nn3", neural=False)
    if len(linear["models"]) != 96 or len({m["model_id"] for m in linear["models"]}) != 96:
        raise ValueError("Expected the corrected complete 96-model linear registry")
    source_prepared = Path(linear["prepared"])
    if read_json(source_prepared/"manifest.json") != prepared_manifest:
        raise ValueError("Linear baseline uses a different prepared dataset")
    records = []
    for model in linear["models"]:
        if model["months"] != 108:
            raise ValueError("Linear baseline is incomplete")
        relative = f"linear/predictions_{model['model_id']}.pkl"
        source = Path(model["predictions"])
        if not source.is_absolute():
            source = linear_registry.parent/source
        copy(source, relative, expected_sha=model["sha256"])
        records.append({**model, "predictions": Path(relative).name})
    register("linear/registry.json", {**linear, "prepared": "../prepared", "path_base": "registry_directory",
             "source_registry": str(linear_registry), "source_registry_sha256": hash_file(linear_registry),
             "models": records, "portability_note": "Prediction bytes unchanged; source fingerprint retained as provenance."})
    for name in ("monthly_selection.csv", "migration_audit.json", "config.json"):
        if (linear_registry.parent/name).is_file():
            copy(linear_registry.parent/name, "linear/"+name)
    print("COPY completed local checkpoints (snapshot only)", flush=True)
    bundle = load_bundle(prepared)
    task_sets = {fit: {t["month"]: t for t in make_tasks(bundle, fit_days=fit)} for fit in (504, 252, 756)}
    entries, excluded = {}, []
    # Ordered roots prefer the active full study; pilot provides missing bridge cases.
    for root in map(Path, legacy_roots):
        if not root.exists():
            continue
        for spec_path in sorted(root.rglob("run_spec.json")):
            spec = read_json(spec_path)
            fit = spec.get("fit_days")
            if fit not in task_sets or spec.get("estimator") != "nn3":
                continue
            model = {k: spec.get(k) for k in ("estimator", "target", "feature_set", "fit_days")}
            if not spec.get("tasks") or not compatible_spec(spec, model, spec["tasks"][0], training_config([128,64,32], 2), prepared_manifest):
                excluded.append({"source": str(spec_path), "reason": "Different data, kernel, runtime or full-budget configuration"})
                continue
            signature = fingerprint(spec)
            source_id = spec_path.parent.name + "_" + signature[:12]
            target_spec = f"legacy_nn/{source_id}/run_spec.json"
            copied_spec = False
            for checkpoint in sorted((spec_path.parent/"months").glob("????-??.pkl")):
                task = task_sets[fit].get(checkpoint.stem)
                key = legacy_key(model, checkpoint.stem)
                if key in entries or task is None or task not in spec["tasks"]:
                    continue
                payload = pd.read_pickle(checkpoint)
                if payload.get("fingerprint") != signature or payload.get("task") != task:
                    raise ValueError(f"Completed local checkpoint has invalid identity: {checkpoint}")
                rows = np.flatnonzero((bundle["codes"] >= task["test_first"]) & (bundle["codes"] <= task["test_last"]))
                validate_checkpoint(payload["result"], task, rows)
                del payload
                if not copied_spec:
                    copy(spec_path, target_spec)
                    copied_spec = True
                target_checkpoint = f"legacy_nn/{source_id}/months/{checkpoint.name}"
                copy(checkpoint, target_checkpoint)
                entries[key] = {"specification": target_spec, "checkpoint": target_checkpoint,
                                "source_path": str(checkpoint), "source_fingerprint": signature}
    register("legacy_index.json", {"schema": "server_nn_legacy_index_v1", "entries": entries, "excluded": excluded})
    for name in ("04_methodology_gkx_ckx.md", "07_protocol_linear_results.md", "EXPERIMENTAL_PROTOCOL.md", "TIMING_CONVENTION.md", "RESEARCH_QUESTIONS.md"):
        source = PACKAGE.parent/"reports"/name
        if source.is_file():
            copy(source, "reference_reports/"+name)
    manifest = {"schema": "stocktwits_server_bundle_v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "prepared_fingerprint": prepared_manifest["fingerprint"], "source_runtime": runtime_info(),
                "vendor_sha256": {name: hash_file(VENDOR/name) for name in TRAINING_VENDOR},
                "source_prepared": str(prepared), "source_linear_registry": str(linear_registry),
                "legacy_checkpoints": len(entries), "total_bytes": sum(x["bytes"] for x in files.values()),
                "files": files, "note": "Read-only point-in-time copies. No raw-data source paths are needed at execution."}
    write_json(manifest, output/"bundle_manifest.json")
    print(f"EXPORTED {len(files)} files, {manifest['total_bytes']/1e9:.2f} GB, {len(entries)} reusable NN3 months to {output}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--linear-registry", type=Path)
    parser.add_argument("--legacy-root", type=Path, action="append")
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args(argv)
    pointers = PACKAGE.parent/"reports/data/protocol_v1_1_execution.json"
    default = read_json(pointers) if pointers.exists() else {}
    prepared = args.prepared or default.get("prepared")
    linear = args.linear_registry or default.get("linear", {}).get("registry")
    if not prepared or not linear:
        parser.error("Provide --prepared and --linear-registry when exporting outside the original repository")
    roots = args.legacy_root
    if roots is None:
        study = default.get("neural_full", {}).get("run_directory")
        roots = ([Path(study)/"nn"] if study else []) + [PACKAGE.parent/".runs/protocol_v1_1/nn_pilot"]
    export(args.output, prepared, linear, [] if args.skip_legacy else roots)


if __name__ == "__main__":
    main()
