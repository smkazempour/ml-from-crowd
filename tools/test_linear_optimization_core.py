"""Numerical identity, information boundaries and diagnostic checks for v3."""
import copy
import unittest

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from threadpoolctl import threadpool_limits

import linear_optimization_core as core
import linear_design_core as legacy
import protocol_data as data
import protocol_linear as linear
from test_linear_design_core import fixture


class LinearOptimizationCoreTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.limit.restore_original_limits()

    def test_identity_and_nested_grids(self):
        specs = core.procedure_specs()
        self.assertEqual(len(specs), 86)
        self.assertEqual(len({core.model_name(spec, "raw") for spec in specs}), 86)
        baseline = [spec for spec in specs if spec["transform"] == "standard"
                    and spec["penalty_grid"] == "original" and spec["pca_grid"] != "expanded"]
        self.assertEqual(len(baseline), 19)
        self.assertTrue(set(linear.RIDGE_ALPHAS) < set(core.RIDGE_EXPANDED))
        self.assertTrue(set(linear.ALPHA_FRACTIONS) < set(core.FRACTIONS_EXPANDED))
        self.assertTrue(set(linear.L1_RATIOS) < set(core.RATIOS_EXPANDED))
        self.assertTrue(set(legacy.TEXT_MULTIPLIERS) < set(core.TEXT_EXPANDED))
        self.assertTrue(set(core.PCA_MENUS["original"]) < set(core.PCA_MENUS["expanded"]))
        self.assertTrue(all(spec["penalty_grid"] == "original" for spec in specs if spec["estimator"] == "ols"))
        self.assertTrue(all(spec["pca_grid"] == "none" for spec in specs if spec["representation"] == "full"))

    def test_daily_ranks_ties_singletons_missing_and_preserved_columns(self):
        X = np.array([[10., 0., 9.], [10., 1., 8.], [20., 1., 7.], [np.nan, 0., 6.],
                      [3., 1., 5.], [np.nan, 0., 4.]])
        codes = np.array([0, 0, 0, 0, 1, 1])
        actual = core.daily_rank_characteristics(X, codes, [0])
        np.testing.assert_allclose(actual[:, 0], [-.5, -.5, 1., np.nan, 0., np.nan], equal_nan=True)
        np.testing.assert_array_equal(actual[:, 1:], X[:, 1:])
        self.assertEqual(X[0, 0], 10.)
        changed = X.copy()
        changed[codes == 1, 0] = [999., 1000.]
        np.testing.assert_array_equal(actual[codes == 0], core.daily_rank_characteristics(changed, codes, [0])[codes == 0])

    def test_rank_block_uses_unlabelled_inputs_and_only_fitting_scaler(self):
        bundle, task = fixture()
        bundle["q"][0] = np.nan
        block = core.build_block(bundle, task, "raw", transform="daily_rank")
        columns = data.feature_columns(bundle, core.UNION)
        raw = core.daily_rank_characteristics(bundle["X"][np.ix_(block["input_indices"], columns)],
            bundle["codes"][block["input_indices"]],
            [columns.index(column) for column in core.continuous_characteristic_columns(bundle)])
        expected = data._weighted_scaler(raw, bundle["codes"][block["input_indices"]],
            [name in bundle["manifest"]["binary_features"] for name in block["feature_names"]])
        np.testing.assert_array_equal(block["mean"], expected[0])
        np.testing.assert_array_equal(block["scale"], expected[1])
        self.assertNotIn(0, block["fit_indices"])
        changed = copy.deepcopy(bundle)
        changed["X"][changed["codes"] >= task["valid_first"]] *= 10000.
        second = core.build_block(changed, task, "raw", transform="daily_rank")
        np.testing.assert_array_equal(block["mean"], second["mean"])
        np.testing.assert_array_equal(block["scale"], second["scale"])
        np.testing.assert_array_equal(block["X_fit"], second["X_fit"])
        binary = block["feature_names"].index("missing_char")
        np.testing.assert_array_equal(block["X_fit"][:, binary], bundle["X"][block["fit_indices"], 12])

    def test_parameterized_solvers_match_independent_weighted_fits(self):
        bundle, task = fixture()
        block = core.build_block(bundle, task, "raw", transform="daily_rank")
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        menu = core.fit_candidates(moments, (.3,), (.07,), (.01, .03))
        for candidate in menu:
            self.assertLess(candidate["kkt_violation"], linear.KKT_TOL)
            if candidate["estimator"] in ("lasso", "enet"):
                reference = ElasticNet(alpha=candidate["alpha"], l1_ratio=candidate["l1_ratio"],
                    tol=1e-12, max_iter=100000).fit(block["X_fit"].astype(float), block["y_fit"], sample_weight=block["w_fit"])
                np.testing.assert_allclose(candidate["coefficient"], reference.coef_, atol=1e-9)
        groups = legacy.group_labels(bundle)
        grouped = core.group_ridge_candidates(moments, groups, (.3,), (300.,))
        chosen = next(item for item in grouped if item["drop_block"] == "none" and item["group_social_multiplier"] == 10.)
        multipliers = np.where(groups == "social", 10., np.where(groups == "text", 300., 1.))
        reference = Ridge(alpha=.3).fit(block["X_fit"].astype(float) / np.sqrt(multipliers),
                                      block["y_fit"], sample_weight=block["w_fit"])
        np.testing.assert_allclose(chosen["coefficient"], reference.coef_ / np.sqrt(multipliers), atol=1e-10)
        dropped = next(item for item in grouped if item["drop_block"] == "text_social")
        np.testing.assert_array_equal(dropped["coefficient"][groups != "characteristic"], 0.)

    def test_calendar_month_summaries_and_lomo_match_direct_rescoring(self):
        bundle, task = fixture()
        task.update(valid_first=18, valid_last=64)
        block = core.build_block(bundle, task, "raw")
        rng = np.random.default_rng(563)
        candidates = [dict(coefficient=rng.normal(size=13), intercept=float(i), candidate_id=str(i)) for i in range(5)]
        core.score_candidates(candidates, block["X_valid"], block["y_valid"], block["codes_valid"], bundle["calendar"], batch_size=2)
        predictions = block["X_valid"] @ np.column_stack([item["coefficient"] for item in candidates]) + np.arange(5)
        scores, days = linear.validation_scores(predictions, block["y_valid"], block["codes_valid"])
        np.testing.assert_allclose([item["validation_ic"] for item in candidates], scores, atol=1e-15)
        self.assertEqual(candidates[0]["validation_days"], days)
        stability = core.selection_stability(candidates)
        months = bundle["calendar"][block["codes_valid"]].astype("datetime64[M]").astype(str)
        for row in stability["omissions"]:
            keep = months != row["omitted_month"]
            score, count = linear.validation_scores(predictions[keep], block["y_valid"][keep], block["codes_valid"][keep])
            winner = next(i for i, value in enumerate(score) if value >= max(score) - linear.TIE_TOL)
            self.assertEqual(row["winner_candidate_id"], str(winner))
            self.assertAlmostEqual(row["remaining_ic"], score[winner], places=14)
            self.assertEqual(row["remaining_days"], count)
        tied = copy.deepcopy(candidates[:2])
        tied[1].update({key: copy.deepcopy(tied[0][key]) for key in ("validation_ic", "validation_sum", "validation_days", "month_sum", "month_count")})
        self.assertEqual(core.selection_stability(tied)["full_candidate_id"], "0")

    def test_month_identity_all19_and_no_holdout_label_dependency(self):
        bundle, task = fixture()
        task["fit_days"] = 504
        menus = {"original": (2, 4), "expanded": (1, 2, 4)}
        result = core.fit_month(bundle, [task], "raw", pca_menus=menus)
        self.assertEqual(result["predictions"].shape[1], 86)
        old_block = legacy.build_block(bundle, task, "raw")
        old_selected, _ = legacy.fit_menu(bundle, old_block, pca_dimensions=menus["original"])
        final = legacy.build_block(bundle, task, "raw", "union")
        moments = linear.weighted_moments(final["X_fit"], final["y_fit"], final["w_fit"])
        pca = legacy.pca_state(bundle, final)
        baseline = [spec for spec in core.procedure_specs() if spec["transform"] == "standard"
                    and spec["penalty_grid"] == "original" and spec["pca_grid"] != "expanded"]
        for spec in baseline:
            key = tuple(spec[field] for field in ("feature_set", "estimator", "representation"))
            chosen = old_selected[key]
            matrix, groups, _ = legacy.representation_map(bundle, spec["feature_set"], pca, chosen["pca_dim"])
            fitted = legacy.fit_selected(legacy.mapped_moments(moments, matrix), chosen, groups)
            expected = final["X_test"] @ (matrix @ fitted["coefficient"]) + fitted["intercept"]
            name = core.model_name(spec, "raw")
            np.testing.assert_array_equal(result["predictions"][:, result["model_names"].index(name)], expected)
            self.assertEqual(core.setting_key(result["diagnostics"][name]["selected"]), core.setting_key(chosen))
        changed = copy.deepcopy(bundle)
        changed["q"][changed["codes"] >= task["test_first"]] = np.nan
        changed["y"][changed["codes"] >= task["test_first"]] = np.nan
        second = core.fit_month(changed, [task], "raw", pca_menus=menus)
        np.testing.assert_array_equal(result["predictions"], second["predictions"])
        for name in result["model_names"]:
            self.assertEqual(result["diagnostics"][name]["selected"]["candidate_id"],
                             second["diagnostics"][name]["selected"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
