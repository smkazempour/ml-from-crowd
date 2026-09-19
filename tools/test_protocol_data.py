"""Protocol tests cover temporal boundaries, information sets and weighting."""
import unittest

import numpy as np
import pandas as pd

from protocol_data import (centered_rank, equal_date_weights, make_tasks,
                           make_block, feature_columns, _weighted_scaler,
                           repair_horizon_labels, rerank_repaired_dates,
                           exclude_known_gap_horizons)


def synthetic_bundle():
    calendar = pd.bdate_range("2010-01-01", "2014-12-31").to_numpy().astype("datetime64[D]")
    codes = np.repeat(np.arange(len(calendar), dtype=np.int32), 12)
    stock = np.tile(np.arange(12), len(calendar))
    X = np.column_stack([stock / 12, codes / 100, np.sin(stock), stock == 0]).astype(np.float32)
    y = np.column_stack([stock + np.sin(codes), 12 - stock + np.cos(codes)])
    q = np.column_stack([centered_rank(y[:, i], codes, target=True) for i in range(2)])
    keys = pd.DataFrame({"date": pd.to_datetime(calendar[codes]), "permno": stock, "ticker": stock.astype(str)})
    return {"X": X, "y": y, "q": q, "codes": codes, "calendar": calendar,
            "keys": keys, "manifest": {"feature_names": ["x", "z", "embed", "missing__x"],
                "feature_sets": {"core": [0, 3], "all": [0, 1, 3],
                                 "textcore": [0, 2, 3], "textall": [0, 1, 2, 3]},
                "binary_features": ["missing__x"],
                "targets": ["f_cumret1", "ar_dgtw_1"]}}


class TransformationTests(unittest.TestCase):
    def test_midrank_centering_ties_and_missingness(self):
        values = [3, 3, 3, np.nan, 1, 1, 3, 5]
        codes = [0] * 4 + [1] * 4
        ranks = centered_rank(values, codes)
        np.testing.assert_allclose(ranks[:4], 0)
        np.testing.assert_allclose(ranks[4:], [-.5, -.5, .25, .75])
        # Permutation cannot break ties or change ranks.
        order = [2, 0, 3, 1, 6, 7, 4, 5]
        permuted = centered_rank(np.array(values)[order], np.array(codes)[order])
        np.testing.assert_allclose(permuted, ranks[order])

    def test_target_minimum_and_constant_days(self):
        codes = np.repeat([0, 1, 2], [9, 10, 10])
        values = np.r_[np.arange(9), np.ones(10), np.arange(10)]
        q = centered_rank(values, codes, target=True)
        self.assertTrue(np.isnan(q[:19]).all())
        np.testing.assert_allclose(q[19:], (np.arange(1, 11) - .5) / 10 - .5)
        self.assertAlmostEqual(float(q[19:].mean()), 0, places=7)

    def test_equal_date_weighting_and_scaling(self):
        codes = np.array([0, 0, 1])
        weights = equal_date_weights(codes)
        np.testing.assert_allclose(weights, [.25, .25, .5])
        mean, scale, constant, absent = _weighted_scaler(np.array([[0.], [2.], [100.]]), codes)
        np.testing.assert_allclose(mean, [50.5])
        np.testing.assert_allclose(scale ** 2, [2450.75])
        self.assertFalse(constant.any())
        self.assertFalse(absent.any())

    def test_binary_identity_and_missing_continuous_statistics(self):
        values = np.array([[1, 2, np.nan], [0, np.nan, np.nan], [0, 8, np.nan]])
        mean, scale, constant, absent = _weighted_scaler(values, [0, 0, 1], [True, False, False])
        self.assertEqual(mean[0], 0)
        self.assertEqual(scale[0], 1)
        self.assertAlmostEqual(mean[1], 6)
        self.assertTrue(absent[2])
        self.assertTrue(constant[2])
        self.assertEqual(scale[2], 1)


class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = synthetic_bundle()

    def test_boundaries_counts_nesting_and_same_validation(self):
        for horizon in (1, 3, 5, 10, 21, 42, 63):
            tasks = [make_tasks(self.bundle, fit_days=fit, horizon=horizon, months=["2014-03"])[0]
                     for fit in (252, 504, 756)]
            for task in tasks:
                self.assertEqual(task["fit_last"] - task["fit_first"] + 1, task["fit_days"])
                self.assertEqual(task["valid_last"] - task["valid_first"] + 1, 126)
                self.assertLess(task["fit_last"] + horizon, task["valid_first"])
                self.assertEqual(task["valid_last"] + horizon, task["cutoff"])
                self.assertEqual(task["test_first"], task["cutoff"] + 1)
            self.assertEqual(len({task["valid_first"] for task in tasks}), 1)
            self.assertEqual(len({task["fit_last"] for task in tasks}), 1)

    def test_missing_input_session_is_not_silently_removed(self):
        bundle = dict(self.bundle)
        task = make_tasks(bundle, months=["2014-03"])[0]
        bundle["codes"] = bundle["codes"][bundle["codes"] != task["valid_first"]]
        with self.assertRaisesRegex(ValueError, "Missing valid input dates"):
            make_tasks(bundle, months=["2014-03"])

    def test_inadequate_history_and_boundary_month_fail(self):
        with self.assertRaisesRegex(ValueError, "Insufficient calendar history"):
            make_tasks(self.bundle, months=["2010-03"])
        with self.assertRaisesRegex(ValueError, "full-month calendar coverage"):
            make_tasks(self.bundle, months=["2014-12"])


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.bundle = synthetic_bundle()
        self.task = make_tasks(self.bundle, fit_days=20, validation_days=10, months=["2014-03"])[0]

    def test_matched_inputs_and_weights_for_feature_sets(self):
        full = make_block(self.bundle, self.task, "textall", "raw")
        core = make_block(self.bundle, self.task, "core", "raw")
        columns = feature_columns(self.bundle, "core")
        for part in ("fit", "valid", "test"):
            np.testing.assert_allclose(core[f"X_{part}"], full[f"X_{part}"][:, columns])
            np.testing.assert_array_equal(core[f"{part}_indices"], full[f"{part}_indices"])
        np.testing.assert_allclose(core["w_fit"], full["w_fit"])
        self.assertAlmostEqual(float(core["w_fit"].sum()), 1)
        np.testing.assert_array_equal(core["X_fit"][:, 1], self.bundle["X"][core["fit_indices"], 3])

    def test_missing_fit_labels_do_not_change_input_scaling(self):
        before = make_block(self.bundle, self.task, "textall", "raw")
        self.bundle["q"][before["fit_indices"][:6], 0] = np.nan
        after = make_block(self.bundle, self.task, "textall", "raw")
        np.testing.assert_array_equal(before["mean"], after["mean"])
        np.testing.assert_array_equal(before["scale"], after["scale"])
        self.assertEqual(len(before["X_fit"]) - 6, len(after["X_fit"]))
        # The remaining labels are reweighted by date rather than inheriting
        # pre-filter weights that would underweight the incomplete date.
        totals = pd.Series(after["w_fit"]).groupby(after["codes_fit"]).sum()
        np.testing.assert_allclose(totals, 1 / len(totals))

    def test_future_labels_and_values_do_not_fit_transformations(self):
        before = make_block(self.bundle, self.task, "textall", "raw")
        rows = before["test_indices"]
        self.bundle["q"][rows, 0] = np.nan
        self.bundle["y"][rows, 0] = np.nan
        self.bundle["X"][rows, 2] = 100000
        after = make_block(self.bundle, self.task, "textall", "raw")
        np.testing.assert_array_equal(before["mean"], after["mean"])
        np.testing.assert_array_equal(before["scale"], after["scale"])
        np.testing.assert_array_equal(before["X_fit"], after["X_fit"])
        self.assertEqual(len(before["X_test"]), len(after["X_test"]))
        self.assertTrue(np.isnan(after["y_test"]).all())

    def test_unranked_missing_values_imputed_from_fit_only(self):
        self.bundle["X"][self.bundle["codes"] == self.task["test_first"], 2] = np.nan
        block = make_block(self.bundle, self.task, "textall", "raw")
        self.assertTrue(np.isfinite(block["X_test"]).all())
        np.testing.assert_allclose(block["X_test"][:12, 2], 0)


class HorizonRepairTests(unittest.TestCase):
    def test_interior_gap_excluded_boundary_retained_and_targets_reranked(self):
        calendar = pd.bdate_range("2022-12-27", periods=4).to_numpy().astype("datetime64[D]")
        codes = np.repeat([0, 1, 3], 12)
        values = np.tile(np.arange(12, dtype=float), 3)
        y = np.column_stack([values, -values])
        q = np.column_stack([centered_rank(y[:, j], codes, target=True) for j in range(2)])
        original_y, original_q = y.copy(), q.copy()
        evaluation = pd.DataFrame({"date": pd.to_datetime(calendar[codes]),
            "permno": np.tile(np.arange(12), 3), "ticker": "TEST",
            "f_cumret1": y[:, 0], "ar_dgtw_1": y[:, 1],
            "next_session_ret": 1.,
            "next_stock_row_is_next_session": True}, index=np.arange(1000, 1036))
        evaluation.loc[[1000, 1035], "next_stock_row_is_next_session"] = False
        original_keys = evaluation[["date", "permno", "ticker"]].copy()
        audit = repair_horizon_labels(y, evaluation, codes, calendar)
        changed = rerank_repaired_dates(y, q, codes, audit["affected_date_codes"])
        self.assertEqual(audit["changed_y_cells"], [[0, 0], [0, 1]])
        self.assertEqual(audit["changed_y_rows"], [0])
        self.assertEqual(audit["boundary_unverified_finite_targets"]["f_cumret1"], 1)
        self.assertTrue(np.isnan(y[0]).all())
        np.testing.assert_array_equal(y[-1], original_y[-1])
        self.assertTrue(evaluation[["date", "permno", "ticker"]].equals(original_keys))
        self.assertTrue(evaluation.loc[1000, ["f_cumret1", "ar_dgtw_1"]].isna().all())
        expected = np.column_stack([centered_rank(y[:, j], codes, target=True) for j in range(2)])
        np.testing.assert_array_equal(q, expected)
        np.testing.assert_array_equal(q[12:], original_q[12:])
        actual = np.argwhere(~((q == original_q) | (np.isnan(q) & np.isnan(original_q)))).tolist()
        self.assertEqual(changed, actual)

    def test_known_gap_longer_horizon_exclusion_uses_exchange_interval(self):
        codes = np.arange(15)
        evaluation = pd.DataFrame({"permno": [1] * 14 + [2], "f_cumret3": 1.,
                                   "ar_dgtw_3": 2., "crsp_ar_dgtw_3": 2.,
                                   "f_cumret1": 3., "ar_dgtw_1": 4.})
        gap = {"permno": 1, "first_missing_code": 5, "last_missing_code": 7}
        audit = exclude_known_gap_horizons(evaluation, codes, [gap])
        # (t,t+3] intersects [5,7] exactly when 2 <= t <= 6.
        np.testing.assert_array_equal(np.flatnonzero(evaluation.f_cumret3.isna()), [2, 3, 4, 5, 6])
        self.assertEqual(audit["counts"]["3"]["f_cumret3"], 5)
        self.assertFalse(evaluation.f_cumret1.isna().any())
        self.assertFalse(evaluation.crsp_ar_dgtw_3.isna().any())


if __name__ == "__main__":
    unittest.main()
