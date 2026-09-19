"""Small fixtures for transfer corruption, relocation, and checkpoint import gates."""
import copy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

from server_nn import common, export, migration, verify
common.ensure_vendor()
from nn_checkpoint import atomic_pickle, fingerprint


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd()/".runs"/("server_transfer_test_"+uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.assertEqual(self.root.resolve().parent, (Path.cwd()/".runs").resolve())
        shutil.rmtree(self.root)

    def make_export(self):
        prepared = self.root/"source_prepared"
        prepared.mkdir()
        for name in common.PREPARED_FILES:
            (prepared/name).write_bytes(b"small certified fixture\n")
        common.write_json({"fingerprint": "fixture"}, prepared/"manifest.json")
        prediction = self.root/"predictions.pkl"
        pd.DataFrame({"prediction": [0., 1.]}).to_pickle(prediction)
        models = [{"model_id": f"{e}_{f}_{t}_{n}", "estimator": e, "feature_set": f,
                   "target": t, "fit_days": n, "validation_days": 126, "months": 108,
                   "target_column": {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[t],
                   "predictions": str(prediction), "sha256": common.hash_file(prediction)}
                  for e in ("ols", "ridge", "lasso", "enet") for f in common.FEATURES
                  for t in common.TARGETS for n in (504, 252, 756)]
        registry = self.root/"registry.json"
        common.write_json({"prepared": str(prepared), "models": models}, registry)
        output = self.root/"exported"
        with patch("protocol_data.load_bundle", return_value={}), patch("protocol_data.make_tasks", return_value=[]):
            export.export(output, prepared, registry, [])
        return output

    def test_export_relocation_and_tampering(self):
        output = self.make_export()
        identity = common.data_identity(output)
        relocated = self.root/"relocated"
        shutil.copytree(output, relocated)
        self.assertEqual(identity, common.data_identity(relocated))
        manifest = common.verify_bundle(relocated)
        self.assertEqual(manifest["legacy_checkpoints"], 0)
        portable = common.read_json(relocated/"linear/registry.json")
        self.assertTrue(all(not Path(m["predictions"]).is_absolute() for m in portable["models"]))
        self.assertEqual(verify.verify(relocated, run_bridge=True)["bridge"]["status"], "not_needed_no_legacy")
        (relocated/"prepared/X.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "truncated"):
            common.verify_bundle(relocated)

    def test_export_will_not_overwrite(self):
        destination = self.root/"existing"
        destination.mkdir()
        (destination/"keep").write_text("mine")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            export.export(destination, self.root, self.root/"anything", [])
        self.assertEqual((destination/"keep").read_text(), "mine")

    def test_safe_paths(self):
        for value in ("../outside", "D:/data", "a\\b", "."):
            with self.assertRaises(ValueError):
                common.safe_path(self.root, value)

    def test_bridge_numeric_tolerance(self):
        self.assertLess(verify.compare([1.], [1.+1e-7], "prediction"), 1e-6)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify.compare([1.], [1.+2e-6], "prediction")
        with self.assertRaises(ValueError):
            verify.compare([float("nan")], [float("nan")], "prediction")

    def legacy_fixture(self):
        model = next(m for m in common.phase_models("nn3") if m["feature_set"] == "core" and m["target"] == "raw" and m["fit_days"] == 504)
        config = common.training_config(model["widths"], 2)
        task = {"month": "2014-01", "test_first": 1, "test_last": 2}
        bundle = {"manifest": {"fingerprint": "fixture"}, "codes": np.array([0, 1, 2])}
        runtime = common.runtime_info()
        spec = {k: model[k] for k in ("estimator", "feature_set", "target", "fit_days")}
        spec.update(protocol="v1.1", training_budget="standard", validation_days=126, horizon=1,
                    prepared=bundle["manifest"], config=config, tasks=[task],
                    code_sha256={n: common.hash_file(common.VENDOR/n) for n in common.TRAINING_VENDOR},
                    versions={k: common.critical_runtime(runtime)[k] for k in ("python", "numpy", "pandas", "torch")})
        result = {"prediction": np.array([.1, .2]), "test_indices": np.array([1, 2]), "month": "2014-01",
                  "model_states": [{"penalty": p, "seeds": [{"seed": s, "state_dict": {"w": np.array([1.])}} for s in config["seeds"]]} for p in config["penalties"]]}
        common.write_json(spec, self.root/"legacy_spec.json")
        atomic_pickle({"fingerprint": fingerprint(spec), "task": task, "result": result}, self.root/"legacy.pkl")
        entry = {"checkpoint": "legacy.pkl", "specification": "legacy_spec.json"}
        common.write_json({"entries": {migration.legacy_key(model, task["month"]): entry}}, self.root/"legacy_index.json")
        files = {name: {"bytes": (self.root/name).stat().st_size, "sha256": common.hash_file(self.root/name)} for name in ("legacy_spec.json", "legacy.pkl", "legacy_index.json")}
        common.write_json({"files": files, "prepared_fingerprint": "fixture"}, self.root/"bundle_manifest.json")
        receipt = {"bundle_manifest_sha256": common.hash_file(self.root/"bundle_manifest.json"),
                   "critical_runtime": common.critical_runtime(runtime), "runtime": runtime,
                   "data_identity": common.data_identity(self.root), "bridge": {"status": "passed", "threads": 2},
                   "vendor_sha256": spec["code_sha256"]}
        common.write_json(receipt, self.root/"verification.json")
        return model, task, config, bundle

    def test_import_preserves_source_and_provenance(self):
        model, task, config, bundle = self.legacy_fixture()
        source_hash = common.hash_file(self.root/"legacy.pkl")
        destination = self.root/"imported.pkl"
        result = migration.try_import_checkpoint(self.root, model, task, config, destination, "new", bundle)
        self.assertIsNotNone(result)
        payload = pd.read_pickle(destination)
        self.assertEqual(payload["fingerprint"], "new")
        self.assertEqual(payload["import_provenance"]["source_sha256"], source_hash)
        self.assertEqual(common.hash_file(self.root/"legacy.pkl"), source_hash)
        changed = copy.deepcopy(config)
        changed["penalties"] = [1e-4]
        self.assertIsNone(migration.try_import_checkpoint(self.root, model, task, changed, self.root/"incompatible.pkl", "new", bundle))

    def test_receipt_and_checkpoint_rejection(self):
        model, task, config, bundle = self.legacy_fixture()
        with self.assertRaisesRegex(ValueError, "stale"):
            migration.require_bridge(self.root, 4)
        (self.root/"verification.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires"):
            migration.require_bridge(self.root, 2)
        (self.root/"legacy.pkl").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Uncertified"):
            migration.certified_path(self.root, "legacy.pkl")


if __name__ == "__main__":
    unittest.main()
