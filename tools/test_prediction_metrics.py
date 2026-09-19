"""Regression tests for economically consequential evaluation edge cases."""
import importlib.util
import itertools
import json
import shutil
import uuid
import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from prediction_metrics import align_predictions, bin_returns, hac_mean, rank_correlation
from evaluate_predictions import main, summarize
from composition_check import rank_corr_by


class EvaluationTests(unittest.TestCase):
    def fixture(self, n=100):
        return pd.DataFrame({"date": pd.Timestamp("2020-01-02"), "permno": np.arange(n),
                             "signal": np.arange(n, dtype=float), "y": np.linspace(-.01, .01, n),
                             "cap": np.arange(n)+1.0})

    def test_constant_signal_is_cash_and_zero_ic_in_both_tools(self):
        df = self.fixture().assign(signal=0.0)
        for frame in [df, df.iloc[::-1]]:
            for weight in [None, "cap"]:
                p, _ = bin_returns(frame, "signal", "y", weight=weight)
                self.assertTrue((p == 0).all().all())
            self.assertEqual(rank_correlation(frame, "signal", "y").iloc[0], 0)
            self.assertEqual(rank_corr_by(frame, "signal", "y").iloc[0], 0)

    def test_constant_target_is_not_zero_ic(self):
        self.assertTrue(rank_correlation(self.fixture().assign(y=0.), "signal", "y").isna().all())

    def test_partial_ties_equal_average_over_all_tie_orders(self):
        df = self.fixture(6).assign(signal=[0, 1, 1, 1, 1, 2], y=[2., 3., 5., 7., 11., 13.])
        p, counts = bin_returns(df, "signal", "y", bins=3)
        possible = [df.iloc[[0, *perm, 5]].y.to_numpy().reshape(3, 2).mean(axis=1)
                    for perm in itertools.permutations([1, 2, 3, 4])]
        np.testing.assert_allclose(p.iloc[0], np.mean(possible, axis=0))
        np.testing.assert_allclose(counts.iloc[0], [2, 2, 2])
        for weight in [None, "cap"]:
            original, _ = bin_returns(df, "signal", "y", bins=3, weight=weight)
            shuffled, _ = bin_returns(df.sample(frac=1, random_state=8), "signal", "y", bins=3, weight=weight)
            assert_frame_equal(original, shuffled)

    def test_unique_predictions_and_minimum_counts(self):
        df = self.fixture()
        p, _ = bin_returns(df, "signal", "y")
        self.assertAlmostEqual(p.iloc[0, 0], df.y.iloc[:10].mean())
        self.assertAlmostEqual(p.iloc[0, -1], df.y.iloc[-10:].mean())
        small, _ = bin_returns(df.iloc[:9], "signal", "y")
        self.assertTrue(small.isna().all().all())
        missing, _ = bin_returns(df.assign(signal=np.nan), "signal", "y")
        self.assertTrue(missing.isna().all().all())

    def test_keyed_alignment_ignores_stale_indices_and_rejects_duplicates(self):
        panel = self.fixture(3)
        prediction = panel[["date", "permno"]].assign(prediction=[.1, .2, .3], index=[900, 700, 800])
        got = align_predictions(panel, prediction.iloc[::-1])
        np.testing.assert_allclose(got, [.1, .2, .3])
        self.assertTrue(np.isnan(align_predictions(panel, prediction.iloc[:2]).iloc[2]))
        with self.assertRaises(ValueError):
            align_predictions(panel, pd.concat([prediction, prediction.iloc[:1]]))

    def test_hac_matches_direct_bartlett_formula(self):
        x = np.array([.3, .4, -.1, -.2, .5, .6, .3, -.4])
        u = x-x.mean(); n = len(x); lags = 2
        meat = u@u + sum(2*(1-k/(lags+1))*(u[k:]@u[:-k]) for k in range(1, lags+1))
        expected = np.sqrt(meat/n**2*n/(n-1))
        self.assertAlmostEqual(hac_mean(pd.Series(x), lags)["se"], expected)
        self.assertEqual(hac_mean(pd.Series(np.zeros(10)))["p"], 1)

    def test_paired_comparison_identical_models(self):
        dates = pd.bdate_range("2020-01-01", periods=20)
        rc = pd.DataFrame({"core": np.linspace(-.1, .1, 20), "copy": np.linspace(-.1, .1, 20)}, index=dates)
        p = pd.DataFrame({1: np.zeros(20), 10: np.linspace(-.01, .01, 20)}, index=dates)
        s = summarize(rc, {"core": p, "copy": p.copy()}, "core", 5)
        self.assertEqual(s.loc["copy", "delta_rank_corr_mean"], 0)
        self.assertEqual(s.loc["copy", "delta_spread_bp_p_bonferroni"], 1)

    def test_trading_sort_preserves_unique_minimum_and_agrees(self):
        path = Path(__file__).resolve().parents[1]/"05 - trading/trading_library.py"
        spec = importlib.util.spec_from_file_location("trading_library", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        df = self.fixture()
        for signal in [df.signal, pd.Series(np.zeros(len(df)))]:
            d = df.assign(signal=signal)
            got = mod.form_portfolio_from_signals_sort(d[["date", "permno", "signal"]],
                    d[["date", "permno", "y", "cap"]], signal_col="signal", return_col="y", portfolio_weights="cap")
            p, _ = bin_returns(d, "signal", "y", weight="cap")
            np.testing.assert_allclose(got.long_ret, p[10])
            np.testing.assert_allclose(got.short_ret, p[1])

    def test_cli_writes_reproducible_common_sample_artifacts(self):
        root = Path(__file__).resolve().parents[1]/".runs"
        folder = root/f"eval_test_{uuid.uuid4().hex}"
        folder.mkdir(parents=True)
        try:
            frames = [self.fixture(20).assign(date=date) for date in pd.bdate_range("2020-01-01", periods=15)]
            panel = pd.concat(frames, ignore_index=True).rename(columns={"y": "f_cumret1"}).assign(log_volume=1)
            panel.to_pickle(folder/"panel.pkl")
            p = panel[["date", "permno"]].assign(prediction=panel.signal, index=np.arange(len(panel))+5000)
            p.sample(frac=1, random_state=2).to_pickle(folder/"prediction.pkl")
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["--data-dir", str(folder), "--panel", "panel.pkl",
                               "--models", "copy=prediction.pkl,core=prediction.pkl", "--benchmark", "core",
                               "--periods", "early:2020-01-01:2020-01-15", "--out", str(folder/"result.csv")])
            self.assertEqual(list(result.index), ["core", "copy"])
            self.assertEqual(result.loc["copy", "delta_rank_corr_mean"], 0)
            meta = json.loads((folder/"result.json").read_text())
            self.assertEqual(meta["stock_days"], len(panel))
            self.assertEqual(meta["join"], ["date", "permno"])
            self.assertIn("prediction_metrics.py", meta["code_sha256"])
            self.assertTrue((folder/"result_daily.csv").exists())
            self.assertTrue((folder/"result_periods.csv").exists())
        finally:
            assert folder.resolve().parent == root.resolve() and folder.name.startswith("eval_test_")
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
