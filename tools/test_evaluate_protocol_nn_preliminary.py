"""The interim disclosure must not reduce its planned comparison correction."""
import unittest

import numpy as np
import pandas as pd

import evaluate_protocol_nn_preliminary as interim
import protocol_evaluate as evaluator


class PreliminaryTests(unittest.TestCase):
    def test_full_planned_families_and_unavailable_rows(self):
        counts = interim.planned_family_counts(evaluator)
        self.assertEqual(set(counts), {"raw", "dgtw"})
        for target in counts:
            self.assertEqual(counts[target], {"feature": 60, "estimator": 84, "history": 40})
            partial = [m for m in interim.planned_models()
                       if m["target"] == target and m["fit_days"] in interim.WINDOWS]
            observed = interim.Counter(s["family"] for s in evaluator.contrast_specs(partial))
            self.assertEqual(dict(observed), {"feature": 40, "estimator": 56, "history": 20})

    def test_adjustment_preserves_raw_statistics_and_missing_values(self):
        original = pd.DataFrame(dict(family=["feature", "estimator", "history"],
                                     family_size=[40, 56, 20], p=[.0009, .3, np.nan],
                                     p_bonferroni=[.036, 1., np.nan], mean=[.004, .007, .01],
                                     se=[.001, .008, np.nan], hac_lags=[5, 21, 63]))
        adjusted = interim.retain_planned_multiplicity(
            original, {"feature": 60, "estimator": 84, "history": 40})
        pd.testing.assert_frame_equal(original[["p", "mean", "se", "hac_lags"]],
                                      adjusted[["p", "mean", "se", "hac_lags"]])
        self.assertAlmostEqual(adjusted.p_bonferroni.iloc[0], .054)
        self.assertEqual(adjusted.p_bonferroni.iloc[1], 1.)
        self.assertTrue(np.isnan(adjusted.p_bonferroni.iloc[2]))
        self.assertEqual(adjusted.observed_family_size.tolist(), [40, 56, 20])

    def test_correction_cannot_drop_observed_comparisons(self):
        frame = pd.DataFrame(dict(family=["feature"], family_size=[40], p=[.01]))
        with self.assertRaises(ValueError):
            interim.retain_planned_multiplicity(frame, {"feature": 39})

    def test_matrix_cannot_admit_partial_months_or_f756(self):
        models = [m | dict(months=108, validation_days=126,
                           target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[m["target"]])
                  for m in interim.planned_models() if m["fit_days"] in interim.WINDOWS]
        interim.validate_included_matrix(models)
        for damaged in ([m | dict(months=107) if i == 0 else m for i, m in enumerate(models)],
                        [m | dict(fit_days=756) if i == 0 else m for i, m in enumerate(models)],
                        models[:-1]):
            with self.assertRaises(ValueError):
                interim.validate_included_matrix(damaged)


if __name__ == "__main__":
    unittest.main()
