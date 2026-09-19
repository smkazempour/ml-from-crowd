"""Controller checks reject incorrect publication even after a successful child."""
import json
from pathlib import Path
import shutil
import unittest
import uuid

import pandas as pd

import run_characteristic_study as study


class CharacteristicControllerTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]/".runs"
        root.mkdir(exist_ok=True)
        self.root = root/f"characteristic_controller_test_{uuid.uuid4().hex}"
        self.root.mkdir()

        def cleanup():
            resolved = self.root.resolve()
            assert resolved.parent == root.resolve() and resolved.name.startswith("characteristic_controller_test_")
            shutil.rmtree(resolved)

        self.addCleanup(cleanup)

    def fixture(self):
        keys = pd.DataFrame({"date": pd.date_range("2014-01-01", periods=108, freq="MS"), "permno": 1})
        keys.to_pickle(self.root/"keys.pkl")
        predictions = keys.assign(prediction=0.1)
        predictions.to_pickle(self.root/"predictions.pkl")
        model = dict(model_id="fixture", estimator="ols", feature_set="characteristics", target="raw",
                     target_column="f_cumret1", fit_days=504, validation_days=126, months=108, rows=108,
                     predictions=str(self.root/"predictions.pkl"), sha256=study.sha256(self.root/"predictions.pkl"))
        registry = dict(prepared=str(self.root), models=[model])
        spec = dict(estimators=["ols"], feature_sets=["characteristics"], targets=["raw"], fit_days=[504])
        path = self.root/"registry.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        return path, registry, spec

    def test_output_marker_requires_existing_path(self):
        log = self.root/"child.log"
        destination = self.root/"prepared"
        destination.mkdir()
        log.write_text(f"noise\nPrepared: {destination}\n", encoding="utf-8")
        self.assertEqual(study.output_path(log, "prepared"), destination)
        log.write_text("REGISTRY nonexistent.json\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            study.output_path(log, "registry")

    def test_complete_registry_validates_actual_keyed_predictions(self):
        path, _, spec = self.fixture()
        self.assertEqual(len(study.validate_registry(path, self.root, spec)["models"]), 1)
        frame = pd.read_pickle(self.root/"predictions.pkl")
        frame.loc[0, "permno"] = 2
        frame.to_pickle(self.root/"predictions.pkl")
        registry = json.loads(path.read_text())
        registry["models"][0]["sha256"] = study.sha256(self.root/"predictions.pkl")
        path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coverage"):
            study.validate_registry(path, self.root, spec)

    def test_missing_month_or_duplicate_model_cannot_publish(self):
        path, registry, spec = self.fixture()
        registry["models"].append(registry["models"][0])
        path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "grid"):
            study.validate_registry(path, self.root, spec)
        registry["models"].pop()
        path.write_text(json.dumps(registry), encoding="utf-8")
        keys = pd.read_pickle(self.root/"keys.pkl").iloc[1:]
        keys.to_pickle(self.root/"keys.pkl")
        with self.assertRaisesRegex(ValueError, "108 months"):
            study.validate_registry(path, self.root, spec)

    def test_wrong_target_column_cannot_publish(self):
        path, registry, spec = self.fixture()
        registry["models"][0]["target_column"] = "ar_dgtw_1"
        path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "certificate"):
            study.validate_registry(path, self.root, spec)


if __name__ == "__main__":
    unittest.main()
