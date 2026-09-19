"""Meaningful integration checks for isolated characteristic linear fitting."""
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
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

import characteristic_linear as study
import protocol_data as data
import protocol_linear as linear
from nn_checkpoint import atomic_json, atomic_pickle


@contextmanager
def directory():
    root = Path(__file__).resolve().parents[1] / ".runs"
    folder = root / f"characteristic_linear_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        assert folder.resolve().parent == root.resolve()
        assert folder.name.startswith("characteristic_linear_test_")
        shutil.rmtree(folder)


def fixture():
    rng = np.random.default_rng(874)
    calendar = pd.bdate_range("2020-01-01", periods=65).to_numpy()
    counts = np.resize([13, 17, 21], len(calendar))
    codes = np.repeat(np.arange(len(calendar)), counts)
    X = rng.normal(size=(len(codes), 9)).astype(np.float32)
    X[:, 8] = (np.arange(len(X)) % 9 == 0).astype(np.float32)
    X[X[:, 8] == 1, 6] = 0
    y = .04 * X[:, 0] + .09 * X[:, 6] + rng.normal(scale=.12, size=len(X))
    targets = np.column_stack([y, y + rng.normal(scale=.03, size=len(y))])
    q = np.column_stack([data.centered_rank(targets[:, j], codes, target=True) for j in range(2)])
    keys = pd.DataFrame({"date": calendar[codes],
                         "permno": np.concatenate([np.arange(n) + 100 for n in counts]),
                         "ticker": "FIXTURE"}, index=pd.Index(np.arange(len(X)) + 123, name="mm_index"))
    social = {"core": [0, 1], "all": [0, 1, 2, 3], "textcore": [0, 1, 4, 5], "textall": list(range(6))}
    features = dict(social)
    features.update(characteristics=[6, 7, 8], characteristics_sentiment=[0, 6, 7, 8],
                    characteristics_attention=[1, 6, 7, 8], characteristics_core=[0, 1, 6, 7, 8],
                    characteristics_all=[0, 1, 2, 3, 6, 7, 8],
                    characteristics_textcore=[0, 1, 4, 5, 6, 7, 8],
                    characteristics_textall=[8, 5, 0, 7, 2, 6, 1, 4, 3])
    bundle = {"X": X, "y": targets, "q": q, "codes": codes, "calendar": calendar,
              "keys": keys, "manifest": {"fingerprint": "fixture-characteristics",
                  "feature_names": [f"x{j}" for j in range(8)] + ["missing__x6"],
                  "feature_sets": features, "binary_features": ["missing__x6"],
                  "scalar_rank_features": [f"x{j}" for j in [0, 1, 2, 3, 6, 7]],
                  "unranked_features": ["x4", "x5"], "targets": ["f_cumret1", "ar_dgtw_1"]}}
    task = {"month": "2020-03", "fit_days": 30, "validation_days": 12, "horizon": 1,
            "fit_first": 0, "fit_last": 29, "valid_first": 31, "valid_last": 42,
            "test_first": 44, "test_last": 46}
    return bundle, task


def write_parent_fixture(folder):
    bundle, _ = fixture()
    parent = copy.deepcopy(bundle)
    parent["X"] = parent["X"][:, :6]
    parent["manifest"].update(fingerprint="fixture-parent", feature_names=[f"x{j}" for j in range(6)],
                              feature_sets={f: parent["manifest"]["feature_sets"][f] for f in study.SOCIAL_FEATURE_SETS},
                              binary_features=[], scalar_rank_features=[f"x{j}" for j in range(4)])
    parent_path, current_path = folder / "parent", folder / "current"
    for path, item in ((parent_path, parent), (current_path, bundle)):
        path.mkdir()
        for name in ("X", "y", "q", "codes", "calendar"):
            np.save(path / f"{name}.npy", item[name])
        atomic_pickle(item["keys"], path / "keys.pkl")
        atomic_pickle(item["keys"], path / "evaluation.pkl")
        atomic_json(item["manifest"], path / "manifest.json")
    for name in study.PRESERVED_FILES:
        shutil.copyfile(parent_path / name, current_path / name)
    bundle["manifest"]["derivation"] = dict(
        kind="append_market_characteristics", parent_path=str(parent_path.resolve()),
        parent_fingerprint=parent["manifest"]["fingerprint"], parent_feature_count=6,
        parent_manifest_sha256=study.sha256_file(parent_path / "manifest.json"),
        parent_X_sha256=study.sha256_file(parent_path / "X.npy"))
    bundle["manifest"]["artifact_records"] = {name: dict(sha256=study.sha256_file(current_path / name),
        size=(current_path / name).stat().st_size) for name in (*study.PRESERVED_FILES, "X.npy")}
    atomic_json(bundle["manifest"], current_path / "manifest.json")
    registry = dict(prepared=str(parent_path.resolve()), config=dict(
        data_manifest=parent["manifest"], code=linear.code_identity(), grid=linear.grid_specification(),
        validation_days=126, horizon=1, estimators=list(study.ESTIMATORS),
        feature_sets=list(study.SOCIAL_FEATURE_SETS), versions={"python": study.platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": study.scipy.__version__,
            "sklearn": study.sklearn.__version__}))
    return memory_bundle(current_path), registry


def memory_bundle(path):
    result = data.load_bundle(path)
    for name in ("X", "y", "q", "codes", "calendar"):
        result[name] = np.array(result[name])
    return result


class CharacteristicLinearTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.limit.restore_original_limits()

    def test_permuted_union_ols_matches_direct_weighted_regression(self):
        bundle, task = fixture()
        result = study.fit_month(bundle, task, "raw")
        study.validate_month(result, task, "raw", bundle)
        self.assertEqual(result["predictions"].shape[1], 28)
        for feature in study.FEATURE_SETS:
            block = data.make_block(bundle, task, feature, "raw")
            reference = LinearRegression().fit(block["X_fit"], block["y_fit"], sample_weight=block["w_fit"])
            name = linear.model_name("ols", feature, "raw", 30, 12)
            actual = result["predictions"][:, result["model_names"].index(name)]
            np.testing.assert_allclose(actual, reference.predict(block["X_test"]), rtol=1e-5, atol=1e-7)
            self.assertEqual(result["diagnostics"][name]["features"], block["feature_names"])

    def test_union_order_changes_do_not_change_any_estimator_predictions(self):
        bundle, task = fixture()
        first = study.fit_month(bundle, task, "dgtw")
        bundle["manifest"]["feature_sets"][study.UNION_FEATURE_SET] = list(range(9))
        second = study.fit_month(bundle, task, "dgtw")
        np.testing.assert_allclose(first["predictions"], second["predictions"], rtol=1e-5, atol=1e-8)

    def test_test_labels_do_not_change_predictions_or_coverage(self):
        bundle, task = fixture()
        first = study.fit_month(bundle, task, "raw")
        bundle["q"][first["test_indices"]] = np.nan
        bundle["y"][first["test_indices"]] = np.nan
        second = study.fit_month(bundle, task, "raw")
        np.testing.assert_array_equal(first["predictions"], second["predictions"])
        np.testing.assert_array_equal(first["test_indices"], second["test_indices"])

    def test_checkpoint_resume_and_corruption_detection(self):
        bundle, task = fixture()
        with directory() as folder, patch.object(study.data, "load_bundle", return_value=bundle):
            path = study.checkpoint_month("fixture", task, "raw", folder, "run-id", threads=1)
            with patch.object(study, "fit_month", side_effect=AssertionError("Should resume")):
                self.assertEqual(path, study.checkpoint_month("fixture", task, "raw", folder, "run-id", threads=1))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                study.checkpoint_month("fixture", task, "raw", folder, "different-id", threads=1)
            with Path(path).open("rb") as source:
                saved = pickle.load(source)
            saved["result"]["predictions"][0, 0] += .01
            atomic_pickle(saved, path)
            with self.assertRaisesRegex(ValueError, "Invalid characteristic checkpoint"):
                study.checkpoint_month("fixture", task, "raw", folder, "run-id", threads=1)

    def test_feature_set_outside_union_is_rejected(self):
        bundle, _ = fixture()
        bundle["manifest"]["feature_sets"][study.UNION_FEATURE_SET] = list(range(8))
        with self.assertRaisesRegex(ValueError, "not contained"):
            study.local_feature_columns(bundle, "characteristics")

    def test_exact_parent_certifies_and_changed_targets_reject(self):
        with directory() as folder:
            bundle, registry = write_parent_fixture(folder)
            proof = study.validate_parent_identity(bundle, registry)
            self.assertTrue(proof["parent_feature_prefix_verified"])
            changed = np.array(bundle["y"])
            changed[0, 0] += .1
            # Windows refuses to overwrite an open memory-mapped file. Write to
            # a new prepared directory with the altered target instead.
            altered = folder / "altered"
            shutil.copytree(folder / "current", altered)
            np.save(altered / "y.npy", changed)
            broken = memory_bundle(altered)
            with self.assertRaisesRegex(ValueError, "data identity mismatch: y.npy"):
                study.validate_parent_identity(broken, registry)
            del broken, bundle

    def test_changed_feature_prefix_and_numerical_source_reject(self):
        with directory() as folder:
            bundle, registry = write_parent_fixture(folder)
            bad_source = copy.deepcopy(registry)
            bad_source["config"]["code"]["protocol_linear.py"] = "not-the-source"
            with self.assertRaisesRegex(ValueError, "numerical source identity"):
                study.validate_parent_identity(bundle, bad_source)
            changed = np.array(bundle["X"])
            changed[0, 0] += .1
            altered = folder / "altered"
            shutil.copytree(folder / "current", altered)
            np.save(altered / "X.npy", changed)
            broken = memory_bundle(altered)
            with self.assertRaisesRegex(ValueError, "feature values changed"):
                study.validate_parent_identity(broken, registry)
            del broken, bundle

    def test_changed_characteristics_or_environment_reject(self):
        with directory() as folder:
            bundle, registry = write_parent_fixture(folder)
            changed_environment = copy.deepcopy(registry)
            changed_environment["config"]["versions"]["numpy"] = "0.0.0"
            with self.assertRaisesRegex(ValueError, "numerical version differs"):
                study.validate_parent_identity(bundle, changed_environment)
            changed = np.array(bundle["X"])
            changed[0, 7] += .1
            altered = folder / "altered"
            shutil.copytree(folder / "current", altered)
            np.save(altered / "X.npy", changed)
            broken = memory_bundle(altered)
            with self.assertRaisesRegex(ValueError, "prepared artifact changed: X.npy"):
                study.validate_parent_identity(broken, registry)
            del broken, bundle

    def test_publication_requires_every_task(self):
        bundle, task = fixture()
        config = dict(fingerprint="run-id", prepared="fixture", tasks=[task], targets=["raw"],
                      fit_days=[30], validation_days=12, code=study.code_identity())
        with directory() as folder:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study.publish_run(bundle, folder, config, [], [])

    def test_verified_reuse_and_complete_publication_preserve_source_files(self):
        with directory() as folder:
            bundle, old_registry = write_parent_fixture(folder)
            _, task = fixture()
            task["validation_days"] = 126
            old_registry["config"]["tasks"] = [task]
            expected = bundle["keys"].iloc[study._expected_indices(bundle, [task])].copy()
            expected["prediction"] = np.linspace(-.1, .1, len(expected))
            old_models = []
            for feature in study.SOCIAL_FEATURE_SETS:
                for estimator in study.ESTIMATORS:
                    name = linear.model_name(estimator, feature, "raw", 30)
                    path = folder / f"old_{name}.pkl"
                    atomic_pickle(expected, path)
                    old_models.append(dict(model_id=name, estimator=estimator, feature_set=feature,
                        target="raw", target_column="f_cumret1", fit_days=30, validation_days=126,
                        predictions=str(path), sha256=study.sha256_file(path), rows=len(expected), months=1))
            old_registry["models"] = old_models
            registry_path = folder / "old_registry.json"
            atomic_json(old_registry, registry_path)
            reused, proof = study.verify_reused_models(bundle, registry_path, [task], ["raw"])
            self.assertEqual(len(reused), 16)
            self.assertEqual(len(proof["models"]), 16)
            run_dir = folder / "run"
            run_dir.mkdir()
            config = dict(fingerprint="run-id", prepared=bundle["path"], tasks=[task], targets=["raw"],
                          fit_days=[30], validation_days=126, code=study.code_identity())
            with patch.object(study.data, "load_bundle", return_value=bundle):
                checkpoint = study.checkpoint_month("fixture", task, "raw", run_dir, "run-id", threads=1)
            registry = study.publish_run(bundle, run_dir, config, [checkpoint], reused)
            self.assertEqual(len(registry["models"]), 44)
            self.assertEqual(sum(m["reused"] for m in registry["models"]), 16)
            for model in registry["models"]:
                self.assertEqual(study.sha256_file(model["predictions"]), model["sha256"])
            self.assertTrue((run_dir / "complete.json").exists())
            # Finite tampering must be rejected by the original registered hash.
            changed = pd.read_pickle(old_models[0]["predictions"])
            changed.iloc[0, changed.columns.get_loc("prediction")] += .01
            atomic_pickle(changed, old_models[0]["predictions"])
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                study.verify_reused_models(bundle, registry_path, [task], ["raw"])


if __name__ == "__main__":
    unittest.main()
