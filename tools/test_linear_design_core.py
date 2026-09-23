"""Independent numerical and information-boundary checks for linear redesign."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from threadpoolctl import threadpool_limits

import linear_design_core as core
import protocol_data as data
import protocol_linear as linear


def fixture():
    rng = np.random.default_rng(8256)
    calendar = pd.bdate_range("2020-01-01", periods=95).to_numpy()
    counts = np.resize([13, 15, 18], len(calendar))
    codes = np.repeat(np.arange(len(calendar)), counts)
    X = rng.normal(size=(len(codes), 13)).astype(np.float32)
    X[:, 9] = (rng.uniform(size=len(codes)) < .15).astype(np.float32)
    X[:, 12] = (rng.uniform(size=len(codes)) < .12).astype(np.float32)
    X[codes == 6, 3] = np.nan
    y = .1 * np.nan_to_num(X[:, 3]) - .3 * X[:, 10] + rng.normal(size=len(codes))
    q = data.centered_rank(y, codes, target=True)
    q[codes == 8] = np.nan
    bundle = {"X": X, "y": np.column_stack([y, y]), "q": np.column_stack([q, q]),
        "codes": codes, "calendar": calendar,
        "manifest": {"feature_names": ["net_sentiment", "log_volume"] +
            [f"embed_{j:03d}" for j in range(6)] + ["embed_norm", "embed_cos", "char_a", "char_b", "missing_char"],
            "feature_sets": {"characteristics": [10, 11, 12], "characteristics_core": [0, 1, 10, 11, 12],
                "characteristics_textcore": [1, 10, 5, 9, 0, 11, 3, 7, 12, 4, 2, 8, 6],
                "characteristics_textall": list(range(13))},
            "binary_features": ["embed_cos", "missing_char"],
            "targets": ["f_cumret1", "ar_dgtw_1"]}}
    task = dict(month="2020-04", fit_days=30, validation_days=12, horizon=1,
                fit_first=0, fit_last=29, valid_first=31, valid_last=42,
                cutoff=43, test_first=44, test_last=49)
    return bundle, task


class LinearDesignCoreTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.limit.restore_original_limits()

    def test_spec_count_and_distinct_names(self):
        specs = core.procedure_specs()
        self.assertEqual(len(specs), 114)
        self.assertEqual(len({core.model_name(item, "raw") for item in specs}), 114)
        self.assertEqual(len(core.base_specs()), 19)

    def test_full_menu_recovers_old_global_selection(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw")
        selected, _ = core.fit_menu(bundle, block, pca_dimensions=(2, 4))
        for feature in core.FEATURE_SETS:
            old = data.make_block(bundle, task, feature, "raw")
            moment = linear.weighted_moments(old["X_fit"], old["y_fit"], old["w_fit"])
            menu = linear.fit_candidates(moment, np.arange(old["X_fit"].shape[1]))
            chosen = linear.select_candidates(menu, old["X_valid"], old["y_valid"], old["codes_valid"])
            for estimator in linear.ESTIMATORS:
                actual = selected[(feature, estimator, "full")]
                np.testing.assert_allclose(block["X_test"] @ actual["coefficient"] + actual["intercept"],
                    old["X_test"] @ chosen[estimator]["coefficient"] + chosen[estimator]["intercept"], atol=1e-10, rtol=1e-8)
                self.assertAlmostEqual(actual["validation_ic"], chosen[estimator]["validation_ic"], places=12)

    def test_retained_scaling_and_moments_preserve_historical_matrix_width(self):
        bundle, task = fixture()
        # Extra historical-only columns and a different union order reproduce
        # the narrower matrix/layout that triggered the production discrepancy.
        rng = np.random.default_rng(498)
        bundle["X"] = np.column_stack([bundle["X"], rng.normal(size=(len(bundle["X"]), 7)).astype(np.float32)])
        bundle["manifest"]["feature_names"] += [f"unused_social_{j}" for j in range(7)]
        bundle["manifest"]["feature_sets"][core.HISTORICAL_UNION] = list(range(20))
        historical = data.make_block(bundle, task, core.HISTORICAL_UNION, "raw")
        moments = linear.weighted_moments(historical["X_fit"], historical["y_fit"], historical["w_fit"])
        actual = core.build_block(bundle, task, "raw")
        columns = data.feature_columns(bundle, core.UNION)
        for field in ("mean", "scale", "constant", "all_missing"):
            np.testing.assert_array_equal(actual[field], historical[field][columns])
        for part in ("fit", "valid", "test"):
            np.testing.assert_array_equal(actual[f"X_{part}"], historical[f"X_{part}"][:, columns])
        np.testing.assert_array_equal(actual["shared_moments"]["gram"], moments["gram"][np.ix_(columns, columns)])
        for field in ("cross", "x_mean"):
            np.testing.assert_array_equal(actual["shared_moments"][field], moments[field][columns])

    def test_group_ridge_global_collapse_and_independent_weighted_solution(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw")
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        groups = core.group_labels(bundle)
        menu = core.group_ridge_candidates(moments, groups)
        same = next(c for c in menu if c["alpha"] == .1 and c["drop_block"] == "none"
                    and c["group_social_multiplier"] == 1 and c["group_text_multiplier"] == 1)
        reference = Ridge(alpha=.1).fit(block["X_fit"].astype(float), block["y_fit"], sample_weight=block["w_fit"])
        np.testing.assert_allclose(same["coefficient"], reference.coef_, atol=1e-10)
        asymmetric = next(c for c in menu if c["alpha"] == .1 and c["drop_block"] == "none"
                          and c["group_social_multiplier"] == 10 and c["group_text_multiplier"] == 100)
        D = np.where(groups == "social", 10., np.where(groups == "text", 100., 1.))
        reference = Ridge(alpha=.1).fit(block["X_fit"].astype(float) / np.sqrt(D), block["y_fit"], sample_weight=block["w_fit"])
        np.testing.assert_allclose(asymmetric["coefficient"], reference.coef_ / np.sqrt(D), atol=1e-10)

    def test_drop_text_is_exact_smaller_feature_benchmark(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw")
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        groups = core.group_labels(bundle)
        menu = core.group_ridge_candidates(moments, groups)
        candidate = next(c for c in menu if c["drop_block"] == "text" and c["alpha"] == .1
                         and c["group_social_multiplier"] == 1)
        columns = core.local_columns(bundle, "characteristics_core")
        reference = Ridge(alpha=.1).fit(block["X_fit"][:, columns].astype(float), block["y_fit"], sample_weight=block["w_fit"])
        np.testing.assert_allclose(block["X_test"] @ candidate["coefficient"] + candidate["intercept"],
                                   reference.predict(block["X_test"][:, columns]), atol=1e-10)
        self.assertTrue(np.all(candidate["coefficient"][groups == "text"] == 0))

    def test_pca_uses_all_fit_inputs_and_never_validation_test_or_labels(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw")
        first = core.pca_state(bundle, block)
        self.assertEqual(len(first["embedding_columns"]), 6)
        changed = copy.deepcopy(bundle)
        changed["X"][changed["codes"] >= task["valid_first"]] *= 1000
        changed["q"][:] = np.random.default_rng(1).normal(size=changed["q"].shape)
        changed_block = core.build_block(changed, task, "raw")
        second = core.pca_state(changed, changed_block)
        np.testing.assert_array_equal(first["components"], second["components"])
        np.testing.assert_array_equal(first["eigenvalues"], second["eigenvalues"])
        self.assertEqual(first["input_dates"], 30)
        self.assertGreater(first["input_rows"], len(block["X_fit"]))
        matrix, _, _ = core.representation_map(bundle, core.UNION, first, 4)
        x = data._transform(bundle["X"][np.ix_(block["input_indices"], data.feature_columns(bundle, core.UNION))], block["mean"], block["scale"])
        represented = x @ matrix
        weights = data.equal_date_weights(bundle["codes"][block["input_indices"]])
        pcs = represented[:, -4:]
        centered = pcs - weights @ pcs
        np.testing.assert_allclose(centered.T @ (weights[:, None] * centered), np.eye(4), atol=1e-12)

    def test_recent_and_union_dates_and_fit_only_scalers(self):
        bundle, task = fixture()
        recent = core.input_indices(bundle, task, "recent")
        union = core.input_indices(bundle, task, "union")
        self.assertEqual(list(np.unique(bundle["codes"][recent])), list(range(13, 43)))
        self.assertEqual(list(np.unique(bundle["codes"][union])), list(range(30)) + list(range(31, 43)))
        self.assertEqual(len(np.unique(bundle["codes"][union])), 42)
        for policy in ("recent", "union"):
            first = core.build_block(bundle, task, "raw", policy)
            changed = copy.deepcopy(bundle)
            changed["X"][changed["codes"] > 42] += 1e6
            changed["q"][changed["codes"] > 42] = np.nan
            second = core.build_block(changed, task, "raw", policy)
            np.testing.assert_array_equal(first["mean"], second["mean"])
            np.testing.assert_array_equal(first["scale"], second["scale"])
            np.testing.assert_array_equal(first["X_fit"], second["X_fit"])
            self.assertEqual(len(first["test_indices"]), len(second["test_indices"]))

    def test_sparse_refit_recomputes_alpha_and_matches_weighted_sklearn(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw", "union")
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        for estimator, ratio in (("lasso", 1.), ("enet", .5)):
            candidate = dict(estimator=estimator, alpha=999., alpha_fraction=.03, l1_ratio=ratio)
            fitted = core.fit_selected(moments, candidate, core.group_labels(bundle))
            expected_alpha = .03 * np.abs(moments["cross"]).max() / ratio
            self.assertAlmostEqual(fitted["alpha"], expected_alpha)
            reference = ElasticNet(alpha=expected_alpha, l1_ratio=ratio, tol=1e-12, max_iter=100000).fit(
                block["X_fit"].astype(float), block["y_fit"], sample_weight=block["w_fit"])
            np.testing.assert_allclose(fitted["coefficient"], reference.coef_, atol=1e-9)
            self.assertLess(fitted["kkt_violation"], linear.KKT_TOL)

    def test_batched_validation_matches_full_scores_and_tie_order(self):
        rng = np.random.default_rng(20)
        X = rng.normal(size=(100, 3))
        y = rng.normal(size=100)
        codes = np.repeat(np.arange(5), 20)
        candidates = [dict(coefficient=rng.normal(size=3), intercept=float(i)) for i in range(23)]
        actual = core.score_candidates(candidates, X, y, codes, batch_size=4)
        predicted = X @ np.column_stack([c["coefficient"] for c in candidates]) + np.arange(23)
        expected, _ = linear.validation_scores(predicted, y, codes)
        np.testing.assert_allclose([c["validation_ic"] for c in actual], expected, atol=1e-15)
        first, second = dict(validation_ic=.04), dict(validation_ic=.04 + 1e-13)
        self.assertIs(core.best_candidate([first, second]), first)

    def test_month_end_to_end_all_procedures_and_no_test_label_dependency(self):
        bundle, _ = fixture()
        # Small dates and PCA menu keep this integration fixture inexpensive;
        # production history identifiers still exercise fixed504 and select.
        tasks = [dict(month="2020-04", fit_days=days, validation_days=12, horizon=1,
            fit_first=first, fit_last=29, valid_first=31, valid_last=42,
            cutoff=43, test_first=44, test_last=49)
            for days, first in ((504, 0), (252, 10), (756, 0))]
        # Dates must be coherent with each declared history. Instead mock only
        # the indices helper to provide a controlled small analog of the windows.
        original = core.input_indices
        def short_indices(item, task, policy="retain"):
            mapped = {**task, "fit_days": 30 if task["fit_days"] != 252 else 20}
            return original(item, mapped, policy)
        with patch.object(core, "input_indices", side_effect=short_indices):
            first = core.fit_month(bundle, tasks, "raw", pca_dimensions=(2, 4))
            bundle["q"][bundle["codes"] >= 44] = np.nan
            bundle["y"][bundle["codes"] >= 44] = np.nan
            second = core.fit_month(bundle, tasks, "raw", pca_dimensions=(2, 4))
        self.assertEqual(first["predictions"].shape, (len(first["test_indices"]), 114))
        np.testing.assert_array_equal(first["predictions"], second["predictions"])
        for name, diagnostic in first["diagnostics"].items():
            self.assertEqual(len(diagnostic["final"]["coefficient_features"]),
                             len(diagnostic["final"]["coefficient"]))
            self.assertEqual(diagnostic["final"]["coefficient_features"],
                [bundle["manifest"]["feature_names"][i] for i in data.feature_columns(bundle, core.UNION)])
            self.assertIn(diagnostic["selected"]["selected_fit_days"], core.FIT_WINDOWS)
            if diagnostic["procedure"]["history"] == "fixed504":
                self.assertEqual(diagnostic["selected"]["selected_fit_days"], 504)
            if diagnostic["procedure"]["refit"] == "union":
                expected = 42 if diagnostic["selected"]["selected_fit_days"] != 252 else 32
                self.assertEqual(diagnostic["final"]["input_dates"], expected)


if __name__ == "__main__":
    unittest.main()
