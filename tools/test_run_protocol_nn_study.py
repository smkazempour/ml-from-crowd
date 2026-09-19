"""Reject incomplete/mismatched experiment grids before starting costly work."""
from contextlib import contextmanager
import copy
import itertools
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

import run_protocol_nn_study as study


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parents[1]/".runs"
    folder = root/f"nn_study_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        assert folder.resolve().parent == root.resolve() and folder.name.startswith("nn_study_test_")
        shutil.rmtree(folder)


def registry(prepared, family):
    estimators = study.LINEAR if family == "linear" else study.NN
    models = []
    for estimator, features, target, fit in itertools.product(estimators, study.FEATURES, study.TARGETS, study.WINDOWS):
        models.append({"model_id": f"{estimator}_{features}_{target}_fit{fit}_val126",
                       "estimator": estimator, "feature_set": features, "target": target,
                       "target_column": {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target],
                       "fit_days": fit, "validation_days": 126, "horizon": 1,
                       "months": 108, "rows": 216, "predictions": "fixture.pkl", "sha256": "fixture",
                       "kind": "full", "coverage_scope": "2014-2022", "training_budget": "standard"})
    return {"prepared": str(Path(prepared).resolve()), "models": models, "fingerprint": family,
            "kind": "full", "monthly_selection": "linear/monthly_selection.csv",
            "config": {"run_kind": "full", "max_epochs": 100,
                       "seeds": [7, 1007, 2007, 3007, 4007], "widths": [128, 64, 32]}}


class NNStudyTests(unittest.TestCase):
    def test_complete_matrix_merges_and_preserves_linear_references(self):
        linear, neural = registry("prepared", "linear"), registry("prepared", "nn")
        combined = study.combine_registries(linear, neural, "prepared", "linear.json", "nn.json")
        self.assertEqual(len(combined["models"]), 120)
        self.assertEqual(combined["monthly_selection"], linear["monthly_selection"])
        self.assertEqual(combined["config"]["linear"], linear["config"])
        self.assertEqual(combined["source_registries"][0]["fingerprint"], "linear")
        self.assertEqual(combined["kind"], "full")

    def test_incomplete_grid_pilot_and_changed_recipe_are_rejected(self):
        for family in ["linear", "nn"]:
            baseline = registry("prepared", family)
            changes = [lambda d: d["models"].pop(),
                       lambda d: d["models"].append(copy.deepcopy(d["models"][0])),
                       lambda d: d.update(prepared="different"),
                       lambda d: d["models"][0].update(months=107),
                       lambda d: d["models"][0].update(kind="pilot"),
                       lambda d: d["models"][0].update(target_column="ar_dgtw_1"),
                       lambda d: d["models"][0].update(validation_days=50)]
            if family == "nn":
                changes += [lambda d: d["config"].update(seeds=[7]),
                            lambda d: d["config"].update(max_epochs=3),
                            lambda d: d["models"][0].update(coverage_scope="partial")]
            for change in changes:
                document = copy.deepcopy(baseline)
                change(document)
                with self.assertRaises(ValueError):
                    study.validate_matrix(document, "prepared", family)

    def test_actual_prediction_coverage_and_hashes_are_checked(self):
        with fixture_directory() as folder:
            keys = pd.DataFrame({"date": np.repeat(pd.date_range("2014-01-01", periods=108, freq="MS"), 2),
                                 "permno": np.tile([100, 101], 108)})
            keys.to_pickle(folder/"keys.pkl")
            prediction = keys.assign(prediction=np.arange(216)/216)
            path = folder/"predictions.pkl"
            prediction.to_pickle(path)
            record = {"predictions": str(path), "sha256": study.file_hash(path), "rows": len(keys)}
            self.assertEqual(study.verify_prediction_files([record], folder)[str(path)], record["sha256"])
            prediction.loc[0, "date"] = pd.Timestamp("2013-12-31")
            prediction.to_pickle(path)
            record["sha256"] = study.file_hash(path)
            with self.assertRaisesRegex(ValueError, "coverage"):
                study.verify_prediction_files([record], folder)
            prediction = keys.assign(prediction=np.arange(216)/216)
            prediction.to_pickle(path)
            record["sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "digest"):
                study.verify_prediction_files([record], folder)

    def test_invalid_linear_input_records_failure_without_launching_nn(self):
        with fixture_directory() as folder:
            prepared = folder/"prepared"
            prepared.mkdir()
            (prepared/"manifest.json").write_text("{}", encoding="utf-8")
            document = registry(prepared, "linear")
            document["models"].pop()
            path = folder/"linear.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(study, "run_child", side_effect=AssertionError("must not launch")):
                with self.assertRaises(ValueError):
                    study.main(["--linear-registry", str(path), "--prepared", str(prepared),
                                "--out-root", str(folder/"study")])
            statuses = list((folder/"study").glob("*/status.json"))
            self.assertEqual(len(statuses), 1)
            status = json.loads(statuses[0].read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "verify_linear")
            self.assertIn("96-model matrix", status["error"])

    def test_last_registry_uses_final_printed_path(self):
        with fixture_directory() as folder:
            first, last = folder/"first.json", folder/"last.json"
            first.write_text("{}")
            last.write_text("{}")
            log = folder/"nn.log"
            log.write_text(f"Registry: {first}\ntraining\nRegistry: {last}\n", encoding="utf-8")
            self.assertEqual(study.last_registry(log), last.resolve())


if __name__ == "__main__":
    unittest.main()
