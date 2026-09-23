"""Execution, provenance and resumability tests without expensive candidate fits."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import pickle
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

import linear_interaction_runner as runner
import characteristic_linear as previous
from nn_checkpoint import atomic_json, atomic_pickle
from test_characteristic_linear import fixture as old_fixture, write_parent_fixture, memory_bundle


@contextmanager
def directory():
    root = Path(__file__).resolve().parents[1] / ".runs"
    folder = root / f"linear_interaction_runner_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        assert folder.resolve().parent == root.resolve()
        assert folder.name.startswith("linear_interaction_runner_test_")
        shutil.rmtree(folder)


def task_fixture():
    _, original = old_fixture()
    return [{**original, "fit_days": fit, "validation_days": 126, "cutoff": 43}
            for fit in runner.FIT_DAYS]


def result_fixture(bundle, tasks, target="raw"):
    indices = runner.expected_indices(bundle, tasks)
    names = [runner.core.model_name(spec, target) for spec in runner.core.procedure_specs()]
    values = np.arange(len(indices), dtype=float)[:, None] * .001 + np.arange(len(names))[None, :] * .01
    diagnostics = {}
    for spec, name in zip(runner.core.procedure_specs(), names):
        diagnostics[name] = dict(selected=dict(selected_fit_days=504, pca_dim=16 if spec["representation"] == "pca" else 0,
            alpha=.1, validation_ic=.04), final=dict(refit=spec["refit"], fit_rows=500, fit_dates=504,
            coefficient=np.ones(4), mean=np.zeros(4), scale=np.ones(4)))
    return dict(month=tasks[0]["month"], target=target, test_indices=indices, predictions=values,
                model_names=names, diagnostics=diagnostics, window_diagnostics={"504": {"candidate_ic": [0.01, float("nan")]}})


def write_baseline(folder, bundle, tasks, target="raw"):
    folder.mkdir(parents=True, exist_ok=True)
    result = result_fixture(bundle, tasks, target)
    old_names, columns = [], []
    for spec in runner.core.procedure_specs():
        old_spec = runner.core.legacy_spec(spec)
        if old_spec is not None:
            old_names.append(runner.legacy_core.model_name(old_spec, target))
            columns.append(result["model_names"].index(runner.core.model_name(spec, target)))
    old_values = result["predictions"][:, columns]
    old_result = dict(month=result["month"], target=target, test_indices=result["test_indices"],
                      predictions=old_values, model_names=old_names,
                      prediction_sha256=runner.hashlib.sha256(old_values.tobytes()).hexdigest())
    path = folder / "checkpoints" / f"{target}_{tasks[0]['month']}.pkl"
    atomic_pickle(dict(fingerprint="baseline-fixture", tasks=tasks, result=old_result,
                      result_certificate=runner.legacy.result_certificate(old_result)), path)
    config = dict(data_manifest=bundle["manifest"], prepared_manifest_sha256=runner.sha256_file(Path(bundle["path"]) / "manifest.json"),
                  validation_days=126, horizon=1, grid=runner.legacy_core.grid_specification(), code=runner.legacy.code_identity(),
                  versions=runner.versions(), tasks=[tasks])
    registry = dict(schema_version="linear_design_v2", prepared=bundle["path"],
                    fingerprint="baseline-fixture", config=config,
                    checkpoint_sha256={path.name: runner.sha256_file(path)})
    atomic_json(registry, folder / "registry.json")
    return folder / "registry.json", dict(path=str(path.resolve()), sha256=runner.sha256_file(path)), result


class LinearInteractionRunnerTests(unittest.TestCase):
    def test_planned_matrix_is_complete_and_uses_numeric_history_sentinel(self):
        models = runner.planned_models()
        self.assertEqual(len(models), 148)
        self.assertEqual(len({model["model_id"] for model in models}), 148)
        for target in ("raw", "dgtw"):
            self.assertEqual(sum(model["target"] == target for model in models), 74)
        for model in models:
            self.assertEqual(model["fit_days"], 504 if model["history_policy"] == "fixed504" else 0)

    def test_prepared_hashes_detect_changed_matrix(self):
        with directory() as folder:
            bundle, _ = write_parent_fixture(folder)
            runner.verify_prepared(bundle)
            path = Path(bundle["path"]) / "X.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "Prepared artifact changed: X.npy"):
                runner.verify_prepared(bundle)

    def test_baseline_provenance_and_direct_prediction_comparison(self):
        with directory() as folder:
            bundle, _ = write_parent_fixture(folder)
            tasks = task_fixture()
            registry, record, result = write_baseline(folder / "baseline", bundle, tasks)
            proof = runner.verify_baseline(bundle, registry, [tasks], ["raw"])
            self.assertEqual(len(proof["checkpoints"]), 1)
            self.assertEqual(len(runner.verify_baseline_predictions(result, record, bundle)), 9)
            result["predictions"][0, 0] += .1
            with self.assertRaisesRegex(ValueError, "not reproduced"):
                runner.verify_baseline_predictions(result, record, bundle)
            payload = json.loads(registry.read_text())
            payload["config"]["versions"]["numpy"] = "0.0-invalid"
            atomic_json(payload, registry)
            with self.assertRaisesRegex(ValueError, "numerical version"):
                runner.verify_baseline(bundle, registry, [tasks], ["raw"])

    def test_checkpoint_exact_resume_prediction_and_selection_tampering(self):
        with directory() as folder:
            bundle, _ = write_parent_fixture(folder)
            tasks = task_fixture()
            _, record, result = write_baseline(folder / "baseline", bundle, tasks)
            with patch.object(runner.data, "load_bundle", return_value=bundle), patch.object(runner.core, "fit_month", return_value=result):
                path = runner.checkpoint_month("fixture", tasks, "raw", folder / "run", "id", record, 1)
            with patch.object(runner.data, "load_bundle", return_value=bundle), patch.object(runner.core, "fit_month", side_effect=AssertionError("must resume")):
                self.assertEqual(path, runner.checkpoint_month("fixture", tasks, "raw", folder / "run", "id", record, 1))
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    runner.checkpoint_month("fixture", tasks, "raw", folder / "run", "new-id", record, 1)
                with Path(path).open("rb") as source:
                    saved = pickle.load(source)
                for kind in ("prediction", "selection"):
                    changed = copy.deepcopy(saved)
                    if kind == "prediction":
                        changed["result"]["predictions"][0, 1] += .1
                    else:
                        name = changed["result"]["model_names"][0]
                        changed["result"]["diagnostics"][name]["selected"]["alpha"] = 999
                    atomic_pickle(changed, path)
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        runner.checkpoint_month("fixture", tasks, "raw", folder / "run", "id", record, 1)

    def test_tiny_prediction_difference_cannot_hide_changed_daily_ranks(self):
        with directory() as folder:
            bundle, _ = write_parent_fixture(folder)
            tasks = task_fixture()
            _, record, result = write_baseline(folder / "baseline", bundle, tasks)
            with Path(record["path"]).open("rb") as source:
                previous = pickle.load(source)
            near_tie = .01 + np.arange(len(result["test_indices"]), dtype=float) * 1e-12
            previous["result"]["predictions"][:, 0] = near_tie
            previous["result"]["prediction_sha256"] = runner.hashlib.sha256(previous["result"]["predictions"].tobytes()).hexdigest()
            atomic_pickle(previous, record["path"])
            record["sha256"] = runner.sha256_file(record["path"])
            result["predictions"][:, 0] = near_tie
            result["predictions"][[0, 1], 0] = result["predictions"][[1, 0], 0]
            self.assertTrue(np.allclose(result["predictions"][:, 0], near_tie,
                                       rtol=runner.BASELINE_RTOL, atol=runner.BASELINE_ATOL))
            with self.assertRaisesRegex(ValueError, "daily IC differs"):
                runner.verify_baseline_predictions(result, record, bundle)

    def test_publication_complete_coverage_and_saved_metadata(self):
        with directory() as folder:
            bundle, _ = write_parent_fixture(folder)
            tasks = task_fixture()
            baseline, record, result = write_baseline(folder / "baseline", bundle, tasks)
            proof = runner.verify_baseline(bundle, baseline, [tasks], ["raw"])
            run_dir = folder / "run"
            with patch.object(runner.data, "load_bundle", return_value=bundle), patch.object(runner.core, "fit_month", return_value=result):
                path = runner.checkpoint_month("fixture", tasks, "raw", run_dir, "id", record, 1)
            config = dict(prepared=bundle["path"], fingerprint="id", targets=["raw"], tasks=[tasks],
                          baseline_proof=proof, code=runner.code_identity(), versions=runner.versions(),
                          prepared_manifest_sha256=runner.sha256_file(Path(bundle["path"]) / "manifest.json"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.publish_run(bundle, run_dir, config, [])
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                runner.publish_run(bundle, run_dir, config, [path, path])
            published = runner.publish_run(bundle, run_dir, config, [path])
            self.assertEqual(len(published["models"]), 74)
            for model in published["models"]:
                self.assertEqual(model["rows"], len(result["test_indices"]))
                predictions = pd.read_pickle(model["predictions"])
                self.assertTrue(predictions.index.is_unique)
                self.assertEqual(model["sha256"], runner.sha256_file(model["predictions"]))
            selections = pd.read_csv(published["monthly_selection"])
            self.assertEqual(len(selections), 74)
            self.assertIn("final_fit_dates", selections)
            self.assertNotIn("final_mean", selections)
            self.assertNotIn("final_coefficient", selections)

    def test_invalid_month_rows_or_selected_history_are_rejected(self):
        bundle, _ = old_fixture()
        tasks = task_fixture()
        original = result_fixture(bundle, tasks)
        for mutation in ("rows", "history", "nonfinite"):
            result = copy.deepcopy(original)
            if mutation == "rows":
                result["test_indices"][0] += 1
            elif mutation == "history":
                result["diagnostics"][result["model_names"][0]]["selected"]["selected_fit_days"] = 252
            else:
                result["predictions"][0, 0] = float("nan")
            with self.assertRaises(ValueError):
                runner.validate_month(result, tasks, "raw", bundle)


if __name__ == "__main__":
    unittest.main()

