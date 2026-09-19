"""Independent objective checks and recovery/label-isolation tests for new fits."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from threadpoolctl import threadpool_limits

import protocol_data as data
import protocol_linear as linear
from prediction_metrics import rank_correlation


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parents[1]/".runs"
    folder = root/f"protocol_linear_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        assert folder.resolve().parent == root.resolve()
        assert folder.name.startswith("protocol_linear_test_")
        shutil.rmtree(folder)


def fixture_bundle():
    rng = np.random.default_rng(723)
    calendar = pd.bdate_range("2020-01-01", periods=60).to_numpy()
    counts = np.resize([12, 20, 15], len(calendar))
    codes = np.repeat(np.arange(len(calendar)), counts)
    X = rng.normal(size=(len(codes), 6)).astype(np.float32)
    y = X[:, 0]*.04-X[:, 3]*.02+rng.normal(scale=.06, size=len(X))
    targets = np.column_stack([y, y+rng.normal(scale=.02, size=len(y))])
    q = np.column_stack([data.centered_rank(targets[:, j], codes, target=True) for j in range(2)])
    keys = pd.DataFrame({"date": calendar[codes],
                         "permno": np.concatenate([np.arange(n)+100 for n in counts]),
                         "ticker": "FIXTURE"}, index=pd.Index(np.arange(len(X))+123, name="mm_index"))
    bundle = {"X": X, "y": targets, "q": q, "codes": codes, "calendar": calendar,
              "keys": keys, "manifest": {"feature_names": [f"x{j}" for j in range(6)],
                  "feature_sets": {"core": [0, 1], "all": [0, 1, 2, 3],
                                   "textcore": [0, 1, 4, 5], "textall": list(range(6))},
                  "targets": ["f_cumret1", "ar_dgtw_1"]}}
    task = {"month": "2020-03", "fit_days": 30, "validation_days": 12, "horizon": 1,
            "fit_first": 0, "fit_last": 29, "valid_first": 31, "valid_last": 42,
            "test_first": 44, "test_last": 46}
    return bundle, task


class LinearProtocolTests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.thread_limit.restore_original_limits()

    def test_gram_estimators_match_independent_weighted_sklearn(self):
        rng = np.random.default_rng(910)
        codes = np.repeat(np.arange(8), [11, 23, 12, 17, 15, 30, 10, 19])
        X = rng.normal(size=(len(codes), 7))
        X[:, 5] = X[:, 0]+rng.normal(scale=.3, size=len(X))
        X[:, 6] = 1
        y = .4+X[:, 0]*.11-X[:, 2]*.2+rng.normal(scale=.2, size=len(X))
        weights = data.equal_date_weights(codes)
        moments = linear.weighted_moments(X, y, weights, chunk_rows=19)
        center = X-weights@X
        np.testing.assert_allclose(moments["gram"], center.T@(center*weights[:, None]), atol=1e-14)
        np.testing.assert_allclose(moments["cross"], center.T@(weights*(y-weights@y)), atol=1e-14)
        candidates = linear.fit_candidates(moments, list(range(X.shape[1])))
        self.assertEqual(len(candidates), 47)
        for candidate in candidates:
            estimator = candidate["estimator"]
            if estimator == "ols":
                reference = LinearRegression().fit(X, y, sample_weight=weights)
            elif estimator == "ridge":
                # Ridge's unnormalized weighted SSE uses alpha directly when sum(weights)=1.
                reference = Ridge(alpha=candidate["alpha"], solver="svd").fit(X, y, sample_weight=weights)
            else:
                reference = ElasticNet(alpha=candidate["alpha"], l1_ratio=candidate["l1_ratio"],
                                       tol=1e-11, max_iter=200_000).fit(X, y, sample_weight=weights)
            actual = X@candidate["coefficient"]+candidate["intercept"]
            np.testing.assert_allclose(actual, reference.predict(X), rtol=2e-5, atol=3e-7,
                                       err_msg=str({k: candidate[k] for k in ("estimator", "alpha", "l1_ratio")}))

    def test_exact_collinearity_preserves_ols_predictions(self):
        rng = np.random.default_rng(444)
        X = rng.normal(size=(90, 4))
        X[:, 3] = X[:, 0]
        y = .3*X[:, 0]+rng.normal(scale=.1, size=len(X))
        weights = np.ones(len(X))/len(X)
        candidates = linear.fit_candidates(linear.weighted_moments(X, y, weights), [0, 1, 2, 3])
        actual = X@candidates[0]["coefficient"]+candidates[0]["intercept"]
        reference = LinearRegression().fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(actual, reference.predict(X), atol=1e-12)
        for candidate in candidates:
            if candidate["estimator"] == "enet":
                self.assertAlmostEqual(candidate["coefficient"][0], candidate["coefficient"][3], places=8)

    def test_newton_polish_resolves_flat_duplicate_coordinate_direction(self):
        gram, cross = np.ones((2, 2)), np.full(2, .2)
        alpha, ratio = 1e-7, .9
        beta, steps = linear.polish_solution(gram, cross, np.array([.1999999, 0]), alpha, ratio)
        expected = (.2-alpha*ratio)/(2+alpha*(1-ratio))
        np.testing.assert_allclose(beta, np.full(2, expected), atol=1e-9)
        self.assertGreater(steps, 1)
        self.assertLess(linear.kkt_violation(gram, cross, beta, alpha, ratio), 1e-12)
        self.assertLess(linear.objective_gap(gram, cross, beta, alpha, ratio, .1), 1e-12)

    def test_validation_matches_shared_metric_including_ties_and_constants(self):
        rng = np.random.default_rng(98)
        codes = np.repeat(np.arange(5), [12, 18, 9, 20, 15])
        targets = rng.integers(0, 4, len(codes)).astype(float)
        targets[codes == 3] = 1
        targets[0] = np.nan
        predictions = np.column_stack([rng.integers(0, 5, len(codes)), np.zeros(len(codes))])
        scores, days = linear.validation_scores(predictions, targets, codes)
        frame = pd.DataFrame({"date": codes, "y": targets})
        for j in range(2):
            frame["prediction"] = predictions[:, j]
            reference = rank_correlation(frame, "prediction", "y")
            self.assertAlmostEqual(scores[j], reference.mean(), places=14)
            self.assertEqual(days, reference.notna().sum())
        self.assertEqual(scores[1], 0)

    def test_zero_information_and_deterministic_tie_selection(self):
        X = np.ones((30, 3))
        y = np.tile(np.arange(10)/10, 3)
        candidates = linear.fit_candidates(linear.weighted_moments(X, y, np.ones(30)/30), [0, 1, 2])
        selected = linear.select_candidates(candidates, X, y, np.repeat(np.arange(3), 10))
        self.assertEqual(selected["ridge"]["alpha"], linear.RIDGE_ALPHAS[0])
        self.assertEqual(selected["lasso"]["alpha_fraction"], 1)
        self.assertEqual(selected["enet"]["l1_ratio"], .1)
        for result in selected.values():
            np.testing.assert_array_equal(result["coefficient"], np.zeros(3))
            self.assertAlmostEqual(result["intercept"], y.mean())
            self.assertEqual(result["validation_ic"], 0)

    def test_future_test_labels_never_change_training_selection_or_predictions(self):
        bundle, task = fixture_bundle()
        baseline = linear.fit_month(bundle, task, "raw")
        test_rows = (bundle["codes"] >= task["test_first"]) & (bundle["codes"] <= task["test_last"])
        bundle["q"][test_rows, :] = np.nan
        bundle["y"][test_rows, :] = 999
        perturbed = linear.fit_month(bundle, task, "raw")
        np.testing.assert_array_equal(baseline["predictions"], perturbed["predictions"])
        self.assertEqual(len(perturbed["test_indices"]), test_rows.sum())
        for name in baseline["model_names"]:
            before = baseline["diagnostics"][name]["selected"]
            after = perturbed["diagnostics"][name]["selected"]
            for key in ("alpha", "l1_ratio", "validation_ic", "intercept"):
                self.assertEqual(before[key], after[key])

    def test_checkpoint_resume_and_publication_preserve_keys(self):
        bundle, task = fixture_bundle()
        with fixture_directory() as folder:
            with patch.object(data, "load_bundle", return_value=bundle):
                path = Path(linear.checkpoint_month("fixture", task, "raw", folder, "fixture-id", threads=1))
                before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                before_mtime = path.stat().st_mtime_ns
                with patch.object(linear, "fit_month", side_effect=AssertionError("must resume")):
                    linear.checkpoint_month("fixture", task, "raw", folder, "fixture-id", threads=1)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before_hash)
                self.assertEqual(path.stat().st_mtime_ns, before_mtime)
                with self.assertRaises(ValueError):
                    linear.checkpoint_month("fixture", task, "raw", folder, "different-id", threads=1)
            config = {"fingerprint": "fixture-id", "prepared": "fixture", "fit_days": [30], "targets": ["raw"]}
            registry = linear.publish_run(bundle, folder, config, [str(path)])
            self.assertEqual(len(registry["models"]), 16)
            for entry in registry["models"]:
                frame = pd.read_pickle(entry["predictions"])
                self.assertTrue(frame.index.is_unique)
                self.assertFalse(frame.duplicated(["date", "permno"]).any())
                self.assertEqual(frame.index.name, "mm_index")
                self.assertTrue(np.isfinite(frame.prediction).all())
            self.assertTrue((folder/"complete.json").exists())
            self.assertEqual(json.loads((folder/"registry.json").read_text())["fingerprint"], "fixture-id")

    def test_invalid_weights_and_nonfinite_results_fail_explicitly(self):
        with self.assertRaises(ValueError):
            linear.weighted_moments(np.ones((10, 2)), np.ones(10), np.ones(10))
        with self.assertRaises(ValueError):
            linear.validation_scores(np.full(20, np.nan), np.arange(20), np.zeros(20))
        bundle, task = fixture_bundle()
        result = linear.fit_month(bundle, task, "raw")
        result["predictions"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            linear.validate_month(result, task, "raw", bundle)


if __name__ == "__main__":
    unittest.main()
