"""Independent checks of common samples, portfolio ties and calendar inference."""
import contextlib
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_metrics import bin_returns, hac_mean, rank_correlation
from protocol_evaluate import (calendar_hac_mean, contrast_specs, main,
                               paired_contrasts, score_day, score_scope, terciles)


class ProtocolEvaluationTests(unittest.TestCase):
    def test_vectorized_scores_match_shared_tie_and_cash_rules(self):
        rng = np.random.default_rng(2)
        n = 147
        p = np.column_stack([rng.integers(0, 7, n), rng.normal(size=n), np.zeros(n)])
        y, cap = rng.normal(size=n)/100, rng.lognormal(size=n)
        cap[3:8] = np.nan
        cap[10] = -1
        y[19] = np.nan
        got = score_day(p, y, cap)
        frame = pd.DataFrame(dict(date=pd.Timestamp("2020-01-02"), y=y, cap=cap))
        for j in range(p.shape[1]):
            frame["prediction"] = p[:, j]
            expected_ic = rank_correlation(frame, "prediction", "y").iloc[0]
            self.assertAlmostEqual(got["rank_ic"][j], expected_ic)
            for label, weight in [("ew", None), ("cap", "cap")]:
                expected, counts = bin_returns(frame, "prediction", "y", weight=weight, min_per_bin=10)
                np.testing.assert_allclose(got[label+"_raw"][j], expected.iloc[0], atol=1e-14)
                valid = np.isfinite(y)
                if weight:
                    valid &= np.isfinite(cap) & (cap > 0)
                centered = frame.copy()
                centered["y"] -= y[valid].mean()
                expected, _ = bin_returns(centered, "prediction", "y", weight=weight, min_per_bin=10)
                np.testing.assert_allclose(got[label+"_demeaned"][j], expected.iloc[0], atol=1e-14)
                self.assertTrue((counts.iloc[0] >= 10).all())
        shuffled = rng.permutation(n)
        permuted = score_day(p[shuffled], y[shuffled], cap[shuffled])
        for name in ["rank_ic", "ew_raw", "cap_raw", "ew_demeaned", "cap_demeaned"]:
            np.testing.assert_allclose(got[name], permuted[name], atol=1e-14)

    def test_minimum_counts_and_constant_outcomes(self):
        small = score_day(np.arange(99), np.arange(99), np.ones(99))
        self.assertAlmostEqual(small["rank_ic"][0], 1)
        self.assertTrue(np.isnan(small["ew_raw"]).all())
        tiny = score_day(np.arange(9), np.arange(9), np.ones(9))
        self.assertTrue(np.isnan(tiny["rank_ic"]).all())
        constant = score_day(np.arange(110), np.ones(110), np.ones(110))
        self.assertTrue(np.isnan(constant["rank_ic"]).all())
        np.testing.assert_allclose(constant["ew_raw"], 1)
        np.testing.assert_allclose(constant["ew_demeaned"], 0)
        with self.assertRaises(ValueError):
            score_day([1, np.nan], [1, 2])

    def test_calendar_hac_preserves_missing_session_distances(self):
        calendar = pd.bdate_range("2020-01-01", periods=9)
        x = np.array([.4, -.2, np.nan, np.nan, .3, .7, np.nan, -.5, .1])
        finite = np.isfinite(x)
        n = finite.sum()
        u = np.where(finite, x-np.mean(x[finite]), 0)
        meat = sum(u[i]*u[j]*(1-abs(i-j)/4) for i in range(len(u))
                   for j in range(len(u)) if abs(i-j) <= 3)
        expected = np.sqrt(meat/(n*(n-1)))
        got = calendar_hac_mean(pd.Series(x[finite], index=calendar[finite]), calendar, lags=3)
        self.assertAlmostEqual(got["se"], expected)
        self.assertEqual(got["missing_days"], 3)
        collapsed = hac_mean(pd.Series(x).dropna(), lags=3)
        self.assertGreater(abs(got["se"]-collapsed["se"]), 1e-4)

    def test_calendar_hac_agrees_with_existing_hac_on_complete_dates(self):
        values = pd.Series([.2, .4, -.3, .8, -.4, -.7, .1, .3],
                           index=pd.bdate_range("2020-01-01", periods=8))
        for lag in [0, 2, 5]:
            expected, got = hac_mean(values, lag), calendar_hac_mean(values, lags=lag)
            for key in ["mean", "se", "t", "p", "ci_low", "ci_high", "n"]:
                self.assertAlmostEqual(expected[key], got[key])
        self.assertEqual(calendar_hac_mean(pd.Series(np.zeros(5)))["p"], 1)

    def test_terciles_keep_ties_and_missingness(self):
        got = terciles([1, 1, 1, 2, 3, 4, np.nan])
        self.assertEqual(len(set(got[:3])), 1)
        self.assertEqual(got[-1], -1)
        np.testing.assert_array_equal(terciles(np.ones(10)), np.ones(10))

    def test_contrast_families_are_declared_before_valid_statistics(self):
        models = [dict(model_id=f"{est}_{feature}_{fit}", estimator=est,
                       feature_set=feature, fit_days=fit)
                  for fit in [252, 504] for est in ["ols", "ridge"]
                  for feature in ["core", "textcore"]]
        specs = contrast_specs(models)
        self.assertEqual(sum(x["family"] == "feature" for x in specs), 4)
        self.assertEqual(sum(x["family"] == "estimator" for x in specs), 4)
        self.assertEqual(sum(x["family"] == "history" for x in specs), 4)
        rows = []
        for date in pd.bdate_range("2020-01-01", periods=8):
            for model in models:
                rows.append(dict(date=date, model_id=model["model_id"], rank_ic=0.,
                                 ew_spread_bp=0., cap_spread_bp=np.nan))
        got = paired_contrasts(pd.DataFrame(rows), models, lags=[5])
        self.assertTrue((got.family_size == 4).all())
        self.assertTrue((got.loc[got.metric == "rank_ic", "p_bonferroni"] == 1).all())
        self.assertTrue(got.loc[got.metric == "cap_spread_bp", "p_bonferroni"].isna().all())

    def test_score_scope_preserves_empty_calendar_sessions(self):
        calendar = pd.bdate_range("2020-01-01", periods=5)
        models = [dict(model_id="ols_core_504", estimator="ols", feature_set="core", fit_days=504)]
        p = np.tile(np.arange(110), 2)
        codes = np.repeat([0, 3], 110)
        daily, profiles = score_scope(p[:, None], p/1000, np.ones(220), codes, calendar, models)
        self.assertEqual(len(daily), 5)
        self.assertEqual(daily.rank_ic.notna().sum(), 2)
        self.assertTrue((profiles.n_days == 2).all())

    def test_neural_estimators_use_the_same_registered_contrast_rules(self):
        models = [dict(model_id=f"{est}_{feature}", estimator=est,
                       feature_set=feature, fit_days=504)
                  for est in ["ols", "nn3"] for feature in ["core", "textcore"]]
        specs = contrast_specs(models)
        estimator_specs = [s for s in specs if s["family"] == "estimator"]
        self.assertEqual(len(estimator_specs), 2)
        self.assertTrue(all(s["model_id"].startswith("nn3_") for s in estimator_specs))
        self.assertTrue(all(s["benchmark_id"].startswith("ols_") for s in estimator_specs))

    def test_full_neural_regularized_family_uses_exact_matched_inputs_and_history(self):
        models = [dict(model_id=f"{est}_{feature}_{fit}", estimator=est,
                       feature_set=feature, fit_days=fit)
                  for est in ["ols", "ridge", "lasso", "enet", "nn3"]
                  for feature in ["core", "all", "textcore", "textall"]
                  for fit in [252, 504, 756]]
        lookup = {model["model_id"]: model for model in models}
        specs = contrast_specs(models)
        self.assertEqual(sum(s["family"] == "estimator" for s in specs), 84)
        additional = [s for s in specs if s["comparison"] == "nn_versus_regularized"]
        self.assertEqual(len(additional), 36)
        for spec in additional:
            model, benchmark = lookup[spec["model_id"]], lookup[spec["benchmark_id"]]
            self.assertEqual(model["estimator"], "nn3")
            self.assertIn(benchmark["estimator"], ["ridge", "lasso", "enet"])
            self.assertEqual(model["feature_set"], benchmark["feature_set"])
            self.assertEqual(model["fit_days"], benchmark["fit_days"])
        ols_nn = [s for s in specs if s["comparison"] == "versus_ols"
                  and lookup[s["model_id"]]["estimator"] == "nn3"]
        self.assertEqual(len(ols_nn), 12)
        linear_specs = contrast_specs([m for m in models if m["estimator"] != "nn3"])
        self.assertEqual(sum(s["family"] == "estimator" for s in linear_specs), 36)

    def test_cli_economic_keys_common_coverage_and_artifacts(self):
        workspace = Path(__file__).resolve().parents[1]
        directory = workspace/".runs"/("protocol_evaluation_test_"+uuid.uuid4().hex)
        directory.mkdir(parents=True)
        try:
            dates = pd.bdate_range("2014-01-02", periods=5)
            n = 135
            keys = pd.DataFrame(dict(date=np.repeat(dates, n), permno=np.tile(np.arange(n), 5)))
            keys.index = pd.Index(np.arange(len(keys))+6000, name="mm_index")
            keys.to_pickle(directory/"keys.pkl")
            ret = np.tile(np.linspace(-.03, .03, n), 5)
            ret[4] = np.nan
            np.save(directory/"y.npy", np.column_stack([ret, ret/2]))
            np.save(directory/"calendar.npy", dates.to_numpy(dtype="datetime64[D]"))
            (directory/"manifest.json").write_text(json.dumps(dict(targets=["f_cumret1", "ar_dgtw_1"])), encoding="utf-8")
            evaluation = pd.DataFrame(dict(cap=np.tile(np.arange(n)+1., 5),
                                           log_volume=np.tile(np.arange(n)/10, 5),
                                           f_cumret3=ret*2), index=keys.index)
            evaluation.to_pickle(directory/"evaluation.pkl")
            models = []
            for estimator in ["ols", "ridge"]:
                for feature in ["core", "textcore"]:
                    for fit in [252, 504]:
                        mid = f"{estimator}_{feature}_{fit}"
                        path = directory/(mid+".pkl")
                        prediction = keys.assign(prediction=np.tile(np.arange(n), 5)/n)
                        if mid == "ridge_textcore_504":
                            prediction = prediction.iloc[1:]
                        # Stale/shuffled physical row indices must not alter matching.
                        prediction = prediction.iloc[::-1].reset_index(drop=True)
                        prediction.to_pickle(path)
                        models.append(dict(model_id=mid, estimator=estimator, feature_set=feature,
                                           fit_days=fit, target="raw", target_column="f_cumret1",
                                           predictions=str(path)))
            registry = directory/"registry.json"
            registry.write_text(json.dumps(dict(models=models)), encoding="utf-8")
            prefix = directory/"scores"
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = main(["--registry", str(registry), "--prepared", str(directory),
                                 "--out", str(prefix)])
            self.assertEqual(metadata["coverage"]["raw"]["common_prediction_stock_days"], len(keys)-1)
            self.assertEqual(metadata["coverage"]["raw"]["common_scored_stock_days"], len(keys)-2)
            coverage = pd.read_csv(directory/"scores_coverage.csv")
            self.assertTrue((coverage.common_scored_stock_days == len(keys)-2).all())
            result = pd.read_csv(directory/"scores.csv")
            np.testing.assert_allclose(result.loc[result.metric == "rank_ic", "mean"], 1)
            self.assertEqual(len(pd.read_csv(directory/"scores_daily.csv")), len(models)*5)
            self.assertTrue(len(pd.read_csv(directory/"scores_subgroups.csv")) > 0)
            self.assertEqual(set(pd.read_csv(directory/"scores_horizons.csv").horizon), {3})
            self.assertEqual(len(metadata["skipped_horizons"]), 5)
            json.loads((directory/"scores.json").read_text(encoding="utf-8"))
        finally:
            resolved = directory.resolve()
            if resolved.parent == (workspace/".runs").resolve():
                shutil.rmtree(resolved)


if __name__ == "__main__":
    unittest.main()
