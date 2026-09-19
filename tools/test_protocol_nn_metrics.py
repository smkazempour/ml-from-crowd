"""The cached scorer must agree with the independent shared pandas evaluator."""
import unittest

import numpy as np
import pandas as pd

from prediction_metrics import rank_correlation
from protocol_nn_metrics import DailyICScorer


class CachedDailyICTests(unittest.TestCase):
    def assert_reference(self, target, codes, prediction, min_obs=10):
        frame = pd.DataFrame({"date": codes, "target": target, "prediction": prediction})
        reference = rank_correlation(frame, "prediction", "target", min_obs=min_obs)
        scorer = DailyICScorer(target, codes, min_obs=min_obs)
        actual = scorer.daily(prediction)
        pd.testing.assert_series_equal(actual, reference, check_names=False,
                                       check_exact=False, rtol=1e-13, atol=1e-14)
        if reference.notna().any():
            self.assertAlmostEqual(scorer.score(prediction), reference.mean(), places=14)
        else:
            self.assertTrue(np.isnan(scorer.score(prediction)))

    def test_ties_missing_predictions_and_constant_targets_match_shared_metric(self):
        rng = np.random.default_rng(881)
        codes = np.repeat(np.arange(6), [12, 9, 25, 21, 16, 19]).astype(float)
        target = rng.integers(-3, 4, len(codes)).astype(float)
        prediction = rng.integers(-4, 5, len(codes)).astype(float)
        prediction[codes == 2] = 0
        target[codes == 3] = 1
        prediction[codes == 4] = np.nan
        target[3], prediction[4], codes[5] = np.nan, np.inf, np.nan
        codes[6] = np.inf
        prediction[-3:] = np.nan
        self.assert_reference(target, codes, prediction)

    def test_finite_scores_on_shuffled_dates_and_float32_targets(self):
        rng = np.random.default_rng(108)
        codes = np.repeat(pd.bdate_range("2021-01-01", periods=12).to_numpy(), 50)
        target = rng.normal(size=len(codes)).astype(np.float32)
        order = rng.permutation(len(codes))
        for prediction in [rng.normal(size=len(codes)), np.zeros(len(codes))]:
            self.assert_reference(target[order], codes[order], prediction[order])

    def test_nonfinite_predictions_recompute_target_ranks(self):
        target = np.arange(20, dtype=float)**2
        prediction = np.array([7, 8, 1, 9, 0, 5, 3, 4, 6, 2]*2, dtype=float)
        prediction[[1, 3, 7, 11, 13, 19]] = np.nan
        self.assert_reference(target, np.zeros(20), prediction)

    def test_empty_and_undefined_blocks_and_custom_minimum(self):
        self.assert_reference(np.full(20, np.nan), np.zeros(20), np.ones(20))
        self.assert_reference(np.ones(20), np.zeros(20), np.arange(20))
        self.assert_reference(np.arange(5), np.zeros(5), np.arange(5), min_obs=3)
        self.assert_reference(np.arange(5), np.zeros(5), np.full(5, np.nan), min_obs=3)

    def test_cache_isolated_from_caller_mutations_and_validates_shapes(self):
        target, codes = np.arange(20, dtype=float), np.zeros(20)
        scorer = DailyICScorer(target, codes)
        prediction = target.copy()
        before = scorer(prediction)
        target[:], codes[:] = -target, 999
        self.assertEqual(scorer(prediction), before)
        with self.assertRaises(ValueError):
            scorer(np.ones(19))
        with self.assertRaises(ValueError):
            DailyICScorer(np.ones(20), np.zeros(19))
        with self.assertRaises(ValueError):
            scorer._groups[0].target[0] = 7


if __name__ == "__main__":
    unittest.main()
