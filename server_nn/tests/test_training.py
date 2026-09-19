"""Bounded fixtures for portable training identities, restarts, and publication."""
import contextlib
import copy
import io
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from server_nn import train
from server_nn.common import phase_models, training_config
from nn_checkpoint import atomic_pickle, fingerprint
from protocol_nn import predict, restore_network
import torch


def fixture():
    rng = np.random.default_rng(831)
    xf = rng.normal(size=(40, 3)).astype(np.float32)
    xv = rng.normal(size=(20, 3)).astype(np.float32)
    xt = rng.normal(size=(20, 3)).astype(np.float32)
    block = {"X_fit": xf, "X_valid": xv, "X_test": xt,
             "y_fit": xf[:, 0] * .1, "y_valid": xv[:, 0] * .1,
             "raw_test": xt[:, 0] * .1, "w_fit": np.full(40, 1 / 40),
             "codes_valid": np.repeat([3, 4], 10), "codes_test": np.repeat([5, 6], 10),
             "test_indices": np.arange(40, 60), "feature_names": ["a", "b", "c"],
             "mean": np.zeros(3), "scale": np.ones(3), "constant": np.zeros(3, bool)}
    config = {"threads": 1, "widths": [6], "penalties": [1e-5], "seeds": [7, 1007],
              "learning_rate": .001, "max_epochs": 2, "batch_size": 20, "patience": 2}
    task = {"month": "2014-01", "test_first": 5, "test_last": 6, "fit_days": 504}
    model = {**phase_models("depth")[0], "widths": [6]}
    bundle = {"codes": np.r_[np.zeros(40), block["codes_test"]],
              "keys": pd.DataFrame({"date": np.repeat(pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-04"]), 20),
                                     "permno": np.tile(np.arange(20), 3)}),
              "manifest": {"feature_sets": {"core": [0, 1, 2]}}}
    return block, config, task, model, bundle


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[2] / ".runs" / ("server_training_test_" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)

    def tearDown(self):
        expected = Path(__file__).resolve().parents[2] / ".runs"
        self.assertEqual(self.scratch.resolve().parent, expected.resolve())
        shutil.rmtree(self.scratch)

    def test_architectures_and_exact_phase_matrices(self):
        self.assertEqual(len(phase_models("nn3")), 24)
        self.assertEqual(len(phase_models("depth")), 32)
        self.assertEqual(len(phase_models("width")), 24)
        self.assertEqual({tuple(m["widths"]) for m in phase_models("depth")},
                         {(128,), (128, 64), (128, 64, 32), (128, 64, 32, 16)})
        for phase in ("nn3", "depth", "width"):
            self.assertEqual(len(phase_models(phase)), len({m["model_id"] for m in phase_models(phase)}))

    def test_shared_nn3_has_same_identity_across_phases(self):
        models = [next(m for m in phase_models(p) if m["estimator"] == "nn3" and
                       m["feature_set"] == "textall" and m["target"] == "dgtw" and m["fit_days"] == 504)
                  for p in ("nn3", "depth", "width")]
        ids = [fingerprint(train.model_spec(m, [{"month": "2014-01"}], training_config(m["widths"], 2),
                 {"immutable": "abc"}, runtime={"torch": "test"}, code={"sha256": "test"})) for m in models]
        self.assertEqual(len(set(ids)), 1)
        spec = train.model_spec(models[0], [], training_config(models[0]["widths"], 2), {}, runtime={}, code={})
        changed = copy.deepcopy(spec)
        changed["config"]["threads"] = 4
        self.assertNotEqual(fingerprint(spec), fingerprint(changed))
        self.assertNotIn("workers", spec)
        self.assertNotIn("phase", spec)

    def test_standard_budget_cannot_drift_with_architecture(self):
        for model in phase_models("depth") + phase_models("width"):
            cfg = training_config(model["widths"], 2)
            self.assertEqual(cfg["seeds"], [7, 1007, 2007, 3007, 4007])
            self.assertEqual(cfg["penalties"], [1e-5, 1e-4, 1e-3])
            self.assertEqual((cfg["max_epochs"], cfg["patience"], cfg["batch_size"]), (100, 5, 10000))

    def test_pilot_selection_keeps_canonical_full_tasks(self):
        tasks = [{"month": month} for month in train.expected_months()]
        selected = train.select_tasks(tasks, ["2022-12", "2014-01"])
        self.assertEqual([t["month"] for t in selected], ["2014-01", "2022-12"])
        self.assertEqual(len(tasks), 108)
        for months in (["2014-01", "2014-01"], ["2023-01"], []):
            with self.assertRaises(ValueError):
                train.select_tasks(tasks, months)
        with self.assertRaises(ValueError):
            train.select_tasks(tasks[:-1], None)

    def test_real_fixture_checkpoint_resume_and_exact_reconstruction(self):
        block, config, task, model, bundle = fixture()
        run_dir = self.scratch / "model"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(train, "load_bundle", return_value=bundle), \
             patch.object(train, "make_block", return_value=block):
            receipt = train.month_job(self.scratch, model, task, config, run_dir, "sig", import_legacy=False)
            self.assertNotIn("model_states", receipt)
            path = Path(receipt["checkpoint"])
            before, modified = path.read_bytes(), path.stat().st_mtime_ns
            with patch.object(train, "fit_block", side_effect=AssertionError("Resume attempted training")):
                resumed = train.month_job(self.scratch, model, task, config, run_dir, "sig", import_legacy=False)
            self.assertEqual(resumed["action"], "resume")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, modified)
            result = pd.read_pickle(path)["result"]
            seed_states = result["model_states"][0]["seeds"]
            models = [restore_network(seed["state_dict"], 3, config["widths"]) for seed in seed_states]
            restored = np.mean([predict(m, torch.from_numpy(block["X_test"])) for m in models], axis=0)
            np.testing.assert_array_equal(restored, result["prediction"])
            with self.assertRaises(ValueError):
                train.month_job(self.scratch, model, task, config, run_dir, "wrong", import_legacy=False)

    def test_incomplete_wrong_month_checkpoint_is_rejected(self):
        _, _, task, _, bundle = fixture()
        checkpoint = self.scratch / "wrong.pkl"
        result = {"test_indices": np.arange(40, 59), "prediction": np.zeros(19), "month": task["month"]}
        atomic_pickle({"fingerprint": "sig", "task": task, "result": result}, checkpoint)
        with self.assertRaisesRegex(ValueError, "rows"):
            train.load_checkpoint(checkpoint, task, "sig", train.expected_rows(bundle, task))

    def test_publication_has_portable_paths_and_excludes_network_states(self):
        _, config, task, model, bundle = fixture()
        run_dir = self.scratch / "models" / "model" / "sig"
        result = {"test_indices": np.arange(40, 60), "prediction": np.linspace(-1, 1, 20),
                  "month": task["month"], "candidate_diagnostics": [],
                  "model_states": [{"big": np.ones(10)}]}
        atomic_pickle({"fingerprint": "sig", "task": task, "result": result,
                       "import_provenance": {"source_runtime": "fixture"}}, run_dir / "months" / "2014-01.pkl")
        record = train.publish_model(bundle, model, [task], config, run_dir, "sig", self.scratch, pilot=True)
        self.assertEqual(record["kind"], "pilot")
        self.assertFalse(Path(record["predictions"]).is_absolute())
        self.assertNotIn("\\", record["predictions"])
        self.assertFalse((run_dir / "complete.json").exists())
        diags = train.read_json(self.scratch / record["diagnostics"])
        self.assertNotIn("model_states", diags["month_diagnostics"][0])
        self.assertEqual(diags["month_diagnostics"][0]["import_provenance"], {"source_runtime": "fixture"})
        self.assertEqual(record["imported_checkpoint_months"], 1)
        before = (self.scratch / record["predictions"]).stat().st_mtime_ns
        train.publish_model(bundle, model, [task], config, run_dir, "sig", self.scratch, pilot=True)
        self.assertEqual(before, (self.scratch / record["predictions"]).stat().st_mtime_ns)
        self.assertEqual(record["parameter_count"], (3 + 1) * 6 + (6 + 1) + 12)

    def test_failed_heartbeat_records_error(self):
        path = self.scratch / "status.json"
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with train.Heartbeat(path, {"status": "running"}, interval=.01):
                raise RuntimeError("fixture failure")
        self.assertEqual(train.read_json(path)["status"], "failed")
        self.assertIn("fixture failure", train.read_json(path)["error"])

    def test_pilot_cannot_replace_full_phase_alias_and_reuses_full_model_identity(self):
        model = phase_models("nn3")[0]
        tasks = [{"month": month, "test_first": j, "test_last": j, "fit_days": 504}
                 for j, month in enumerate(train.expected_months())]
        bundle = {"codes": np.arange(108), "keys": pd.DataFrame({
            "date": pd.date_range("2014-01-01", periods=108, freq="MS"), "permno": np.ones(108, dtype=int)}),
            "manifest": {"feature_sets": {"core": [0, 1]}}}
        alias = self.scratch / "registries" / "nn3.json"
        train.write_json({"earlier_full_run": True}, alias)
        signatures = []

        def fake_job(bundle_root, requested_model, task, config, run_dir, signature, **kwargs):
            signatures.append(signature)
            result = {"prediction": np.array([.01]), "test_indices": np.array([task["test_first"]]),
                      "month": task["month"], "candidate_diagnostics": []}
            path = run_dir / "months" / f"{task['month']}.pkl"
            atomic_pickle({"fingerprint": signature, "task": task, "result": result}, path)
            return {"checkpoint": str(path), "month": task["month"], "action": "fit"}

        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(train, "verify_bundle", return_value={}), \
             patch.object(train, "load_bundle", return_value=bundle), \
             patch.object(train, "data_identity", return_value={"data": "immutable"}), \
             patch.object(train, "runtime_info", return_value={"runtime": "fixture"}), \
             patch.object(train, "training_hashes", return_value={"code": "fixture"}), \
             patch.object(train, "phase_models", return_value=[model]), \
             patch.object(train, "make_tasks", return_value=tasks), \
             patch.object(train, "month_job", side_effect=fake_job):
            pilot_path = train.run_study("unused", self.scratch, "nn3", 1, 2, ["2014-01"], reuse_local=False)
            self.assertEqual(train.read_json(alias), {"earlier_full_run": True})
            pilot = train.read_json(pilot_path)
            self.assertEqual(pilot["kind"], "pilot")
            self.assertEqual(pilot["models"][0]["months"], 1)
            full_path = train.run_study("unused", self.scratch, "nn3", 1, 2, reuse_local=False)
        full = train.read_json(full_path)
        self.assertEqual(train.read_json(alias), full)
        self.assertEqual(full["kind"], "full")
        self.assertEqual(full["models"][0]["months"], 108)
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(train.read_json(self.scratch / "status.json")["status"], "complete")

    def test_run_lock_rejects_overlapping_writer(self):
        with train.FileLock(str(self.scratch / "training.lock"), timeout=0):
            from filelock import Timeout
            with self.assertRaises(Timeout), patch.object(train, "verify_bundle", side_effect=AssertionError("Must lock before reading")):
                train.run_study("unused", self.scratch, "nn3", 1, 2)


if __name__ == "__main__":
    unittest.main()
