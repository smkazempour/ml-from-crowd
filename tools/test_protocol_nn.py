"""Check NN temporal separation, weighted training and resumable publication."""
import copy
import contextlib
import io
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from protocol_nn import epoch_batches, fit_block, month_job, predict, restore_network, run_scope, validate_checkpoint


def fixture():
    rng = np.random.default_rng(44)
    xfit = rng.normal(size=(60, 3)).astype(np.float32)
    xvalid = rng.normal(size=(40, 3)).astype(np.float32)
    xtest = rng.normal(size=(40, 3)).astype(np.float32)
    # Unequal 10/50-stock fit dates: each date, rather than each row, has mass .5.
    block = {"X_fit": xfit, "X_valid": xvalid, "X_test": xtest,
             "y_fit": (xfit[:, 0]*.1).astype(np.float32),
             "y_valid": (xvalid[:, 0]*.1).astype(np.float32),
             "raw_test": (xtest[:, 0]*.1).astype(np.float32),
             "w_fit": np.r_[np.full(10, .5/10), np.full(50, .5/50)],
             "codes_valid": np.repeat([12, 13], 20), "codes_test": np.repeat([15, 16], 20),
             "test_indices": np.arange(100, 140), "feature_names": ["a", "b", "c"],
             "mean": np.zeros(3), "scale": np.ones(3), "constant": np.zeros(3, dtype=bool)}
    config = {"threads": 1, "widths": [6], "penalties": [1e-5, 1e-3],
              "seeds": [7, 1007], "learning_rate": .01, "max_epochs": 3,
              "batch_size": 16, "patience": 2}
    return block, config


class ProtocolNNTests(unittest.TestCase):
    def test_test_labels_cannot_change_selected_model_or_predictions(self):
        block, config = fixture()
        first = fit_block(block, config)
        changed = copy.deepcopy(block)
        changed["raw_test"] = np.full(40, np.nan)
        second = fit_block(changed, config)
        np.testing.assert_array_equal(first["prediction"], second["prediction"])
        self.assertEqual(first["chosen_penalty"], second["chosen_penalty"])
        self.assertEqual(first["candidate_diagnostics"], second["candidate_diagnostics"])
        self.assertTrue(np.isnan(second["test_ic"]))
        self.assertEqual(first["n_fit"], 60)
        self.assertEqual(len(first["seed_test_ic"]), 2)

    def test_ensemble_selection_uses_ensemble_validation_metric(self):
        block, config = fixture()
        result = fit_block(block, config)
        best = max(result["candidate_diagnostics"], key=lambda c: c["ensemble_validation_ic"])
        self.assertEqual(result["chosen_penalty"], best["penalty"])
        self.assertEqual(result["validation_ic"], best["ensemble_validation_ic"])
        self.assertTrue(np.isfinite(result["prediction"]).all())

    def test_saved_seed_states_restore_the_ensemble(self):
        block, config = fixture()
        result = fit_block(block, config)
        self.assertEqual(len(result["model_states"]), len(config["penalties"]))
        chosen = next(c for c in result["model_states"] if c["penalty"] == result["chosen_penalty"])
        models = [restore_network(s["state_dict"], 3, config["widths"]) for s in chosen["seeds"]]
        restored = np.mean([predict(m, torch.from_numpy(block["X_test"])) for m in models], axis=0)
        np.testing.assert_array_equal(result["prediction"], restored)

    def test_full_budget_is_distinct_from_full_coverage(self):
        self.assertEqual(run_scope("2014-01", "2022-12", None, 5, 100)["kind"], "full")
        partial = run_scope("2022-12", "2022-12", None, 5, 100)
        self.assertEqual(partial["kind"], "pilot")
        self.assertEqual(partial["training_budget"], "standard")
        self.assertEqual(run_scope("2014-01", "2022-12", None, 1, 3)["kind"], "pilot")

    def test_month_checkpoint_resumes_and_rejects_other_configuration(self):
        block, config = fixture()
        bundle = {"codes": np.r_[np.zeros(100), block["codes_test"]]}
        task = {"month": "2020-03", "test_first": 15, "test_last": 16, "fit_days": 504}
        scratch = Path(__file__).resolve().parents[1]/".runs"
        scratch.mkdir(exist_ok=True)
        directory = scratch/("protocol_nn_test_"+uuid.uuid4().hex)
        directory.mkdir()
        try:
            self.assertEqual(Path(directory).resolve().parent, scratch.resolve())
            with contextlib.redirect_stdout(io.StringIO()), patch("protocol_data.load_bundle", return_value=bundle), patch("protocol_data.make_block", return_value=block):
                first = month_job("unused", task, "core", "raw", config, directory, "identity-a")
                checkpoint = Path(directory)/"months"/"2020-03.pkl"
                content, modified = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
                with patch("protocol_nn.fit_block", side_effect=AssertionError("Resume must not fit")):
                    second = month_job("unused", task, "core", "raw", config, directory, "identity-a")
                self.assertEqual(checkpoint.read_bytes(), content)
                self.assertEqual(checkpoint.stat().st_mtime_ns, modified)
                np.testing.assert_array_equal(first["prediction"], second["prediction"])
                with self.assertRaises(ValueError):
                    month_job("unused", task, "core", "raw", config, directory, "identity-b")
        finally:
            if directory.resolve().parent == scratch.resolve():
                shutil.rmtree(directory)

    def test_batchnorm_batches_cover_each_row_once_including_singleton_tail(self):
        for n in [2, 16, 17, 32, 33, 101]:
            batches = list(epoch_batches(n, 16, np.random.default_rng(4)))
            self.assertTrue(all(len(batch) >= 2 for batch in batches))
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(n))

    def test_rejects_invalid_weight_normalization_and_checkpoint(self):
        block, config = fixture()
        block["w_fit"] *= 3
        with self.assertRaises(ValueError):
            fit_block(block, config)
        with self.assertRaises(ValueError):
            validate_checkpoint({"prediction": [1., np.nan], "test_indices": [1, 2]},
                                {"month": "2014-01"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"prediction": [1., 2.], "test_indices": [1, 1]},
                                {"month": "2014-01"})


if __name__ == "__main__":
    unittest.main()
