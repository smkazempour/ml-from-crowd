"""Completion cannot be certified from process exit codes or metadata alone."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch, Mock
import uuid

import pandas as pd

import run_linear_optimization_study as study


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1]/".runs"
        self.root = self.base/f"linear_design_controller_test_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

        def cleanup():
            assert self.root.resolve().parent == self.base.resolve()
            assert self.root.name.startswith("linear_design_controller_test_")
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)

    def fixture(self):
        keys = pd.DataFrame({"date": pd.date_range("2014-01-01", periods=108, freq="MS"), "permno": 1})
        keys.to_pickle(self.root/"keys.pkl")
        prediction = self.root/"pred.pkl"
        keys.assign(prediction=0.1).to_pickle(prediction)
        spec = study.core.procedure_specs()[0]
        model = dict(model_id=study.core.model_name(spec, "raw"), months=108, rows=108,
                     validation_days=126, horizon=1, target="raw", target_column="f_cumret1",
                     predictions=str(prediction), sha256=study.sha256(prediction))
        registry = dict(prepared=str(self.root), models=[model])
        path = self.root/"registry.json"
        path.write_text(json.dumps(registry))
        return path, registry, spec

    def test_valid_grid_and_actual_forecasts(self):
        path, registry, spec = self.fixture()
        with patch.object(study.core, "procedure_specs", return_value=[spec]):
            self.assertEqual(study.validate_registry(path, self.root, ["raw"]), registry)
            registry["models"].append(copy.deepcopy(registry["models"][0]))
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "model grid"):
                study.validate_registry(path, self.root, ["raw"])

    def test_changed_keys_rejected_even_with_updated_hash(self):
        path, registry, spec = self.fixture()
        pred = Path(registry["models"][0]["predictions"])
        frame = pd.read_pickle(pred)
        frame.loc[0, "permno"] = 99
        frame.to_pickle(pred)
        registry["models"][0]["sha256"] = study.sha256(pred)
        path.write_text(json.dumps(registry))
        with patch.object(study.core, "procedure_specs", return_value=[spec]):
            with self.assertRaisesRegex(ValueError, "coverage"):
                study.validate_registry(path, self.root, ["raw"])

    def test_specification_matches_actual_menu(self):
        path = Path(__file__).resolve().parents[1]/"reports/data/linear_optimization_v3_experiment.json"
        spec = json.loads(path.read_text())
        study.validate_specification(spec)
        spec["grid"] = {"invalid": True}
        with self.assertRaisesRegex(ValueError, "executable design"):
            study.validate_specification(spec)

    def test_completed_table_hashes_and_presence(self):
        prefix = self.root/"evaluation"
        registry, report = self.root/"registry.json", self.root/"report.md"
        registry.write_text("{}")
        report.write_text("report")
        outputs, hashes = {}, {}
        for key in ["summary", "_daily", "_contrasts", "_coverage", "_yearly", "_periods"]:
            path = self.root/f"{key}.csv"
            path.write_text("value\n1\n")
            outputs[key] = {"path": str(path)}
            hashes[path.name] = study.sha256(path)
        metadata = {"output_files": outputs, "output_sha256": hashes}
        prefix.with_suffix(".json").write_text(json.dumps(metadata))
        self.assertEqual(len(study.evaluation_artifacts(prefix, report, registry)), 9)
        Path(outputs["_daily"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            study.evaluation_artifacts(prefix, report, registry)
        outputs.pop("_daily")
        prefix.with_suffix(".json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "required evaluation"):
            study.evaluation_artifacts(prefix, report, registry)

    def test_watcher_retries_previous_attempt_lock_and_confirms_current_identity(self):
        status = self.root/"status.json"
        identity = dict(fingerprint="new", started_at="2026-09-22T00:00:00+00:00", pid=100)
        status.write_text(json.dumps(identity))
        (self.root/"watcher_status.json").write_text(json.dumps(dict(
            monitored_identity={**identity, "pid": 99}, watcher_pid=10, terminal=None)))
        old = Mock(pid=10)
        old.poll.return_value = 3
        fresh = Mock(pid=11)
        fresh.poll.return_value = None

        def wait(_):
            (self.root/"watcher_status.json").write_text(json.dumps(dict(
                monitored_identity=identity, watcher_pid=11, terminal=None)))

        with patch.object(study.subprocess, "Popen", side_effect=[old, fresh]) as launch, \
                patch.object(study.time, "sleep", side_effect=wait):
            self.assertIs(study.start_watcher(self.root, status), fresh)
            self.assertEqual(launch.call_count, 2)

    def test_watcher_unexpected_exit_is_not_silently_accepted(self):
        status = self.root/"status.json"
        status.write_text(json.dumps(dict(fingerprint="x", started_at="x", pid=100)))
        failed = Mock()
        failed.poll.return_value = 2
        with patch.object(study.subprocess, "Popen", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "watcher exited 2"):
                study.start_watcher(self.root, status)


if __name__ == "__main__":
    unittest.main()
