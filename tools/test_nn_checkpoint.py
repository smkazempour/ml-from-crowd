"""Exercise interrupted-run recovery and the notebook's training function."""
import ast
import json
import os
from pathlib import Path
import uuid
import shutil
from contextlib import contextmanager
import unittest
import contextlib
import io
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from nn_checkpoint import fingerprint, fit_month_checkpoint, atomic_pickle


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parents[1]/".runs"
    folder = root/f"nn_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        # Only this test's freshly created workspace directory may be removed.
        assert folder.resolve().parent == root.resolve() and folder.name.startswith("nn_test_")
        shutil.rmtree(folder)


class CheckpointTests(unittest.TestCase):
    def test_feature_ranks_do_not_depend_on_future_target_availability(self):
        path = Path(__file__).resolve().parents[1]/"03d - neural network/prediction_neural_network_text.ipynb"
        source = "".join(json.loads(path.read_text())["cells"][5]["source"])
        rng = np.random.default_rng(3)
        data = pd.DataFrame(rng.normal(size=(40, 384)), columns=[f"embed_{i:03}" for i in range(384)])
        for c in ["net_sentiment", "log_volume"]+[f"feature_{i}" for i in range(51)]:
            data[c] = rng.normal(size=40)
        data["date"] = np.repeat(pd.bdate_range("2020-01-01", periods=2), 20)
        data["f_cumret1"] = rng.normal(size=40)
        data["permno"] = np.tile(np.arange(20), 2)
        data["ticker"] = "FIXTURE"
        data["embed_norm"] = .5
        data["embed_cos"] = .25
        data.index.name = "mm_index"
        prepared = []
        for missing in [False, True]:
            frame = data.copy()
            if missing:
                frame.loc[35:, "f_cumret1"] = np.nan
            ns = {"data": frame, "TARGET_COL": "f_cumret1", "CORE_COLS": ["net_sentiment", "log_volume"],
                  "FEATURE_SET": "embed+norm+core", "RANK_TARGET": True, "RANK_FEATURES": True,
                  "PLACEBO": 0, "np": np, "pd": pd}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "feature_preparation", "exec"), ns)
            prepared.append(ns["model_data"][ns["FEATURES"]])
        pd.testing.assert_frame_equal(prepared[0], prepared[1])

    def test_resume_after_later_month_fails(self):
        with fixture_directory() as folder:
            cfg = {"checkpoint_dir": folder, "fingerprint": "fixture"}
            tasks = [{"month": f"2020-0{i}", "t0": 0, "t1": 2} for i in [1, 2]]
            calls = []
            def fit(task, y, codes, config):
                calls.append(task["month"])
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return {"month": task["month"], "rows": (0, 2), "pred": np.array([.1, .2])}
            first = fit_month_checkpoint(tasks[0], None, None, cfg, fit)
            with self.assertRaises(RuntimeError):
                fit_month_checkpoint(tasks[1], None, None, cfg, fit)
            self.assertFalse((Path(folder)/"2020-02.pkl").exists())
            restored = fit_month_checkpoint(tasks[0], None, None, cfg, fit)
            np.testing.assert_array_equal(first["pred"], restored["pred"])
            fit_month_checkpoint(tasks[1], None, None, cfg, fit)
            self.assertEqual(calls, ["2020-01", "2020-02", "2020-02"])
            with self.assertRaises(ValueError):
                fit_month_checkpoint(tasks[0], None, None, {**cfg, "fingerprint": "changed"}, fit)

    def test_nonfinite_result_not_saved(self):
        with fixture_directory() as folder:
            cfg = {"checkpoint_dir": folder, "fingerprint": "fixture"}
            task = {"month": "2020-01", "t0": 0, "t1": 1}
            def fit(*args):
                return {"month": "2020-01", "rows": (0, 1), "pred": [np.nan]}
            with self.assertRaises(ValueError):
                fit_month_checkpoint(task, None, None, cfg, fit)
            self.assertFalse((Path(folder)/"2020-01.pkl").exists())

    def test_config_identity_and_atomic_replacement(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint({"seeds": 1}), fingerprint({"seeds": 5}))
        with fixture_directory() as folder:
            p = Path(folder)/"checkpoint.pkl"
            atomic_pickle({"good": True}, p)
            before = p.read_bytes()
            with patch("nn_checkpoint.os.replace", side_effect=OSError("interrupted publication")):
                with self.assertRaises(OSError):
                    atomic_pickle({"new": True}, p)
            self.assertEqual(before, p.read_bytes())
            self.assertFalse(list(Path(folder).glob("*.tmp")))

    def test_notebook_training_does_not_select_on_test_labels(self):
        path = Path(__file__).resolve().parents[1]/"03d - neural network/prediction_neural_network_text.ipynb"
        nb = json.loads(path.read_text())
        tree = ast.parse("".join(nb["cells"][9]["source"]))
        ns = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(path), "exec"), ns)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 6)).astype(np.float32)
        y = (.1*X[:, 0]+rng.normal(0, .1, 500)).astype(np.float32)
        y[0] = np.nan  # missing training labels must not poison the network
        codes = np.repeat(np.arange(25), 20)
        task = {"month": "synthetic", "lo": 0, "vs": 300, "hi": 400, "t0": 400, "t1": 500}
        cfg = {"arch": [8, 4, 2], "wd_grid": [1e-5, 1e-3], "lr": .001, "batch": 100,
               "patience": 2, "max_epochs": 2, "n_seeds": 1, "threads": 1,
               "groups": {"core": [0, 1]}, "core_idx": {"net_sentiment": 0, "log_volume": 1},
               "x_path": "fixture", "p": 6, "rank_features": True}
        with patch.object(np, "load", return_value=X):
            first = ns["fit_month"](task, y, codes, cfg)
            y[400:] *= -1
            changed = ns["fit_month"](task, y, codes, cfg)
        self.assertEqual(first["chosen_wd"], changed["chosen_wd"])
        np.testing.assert_array_equal(first["pred"], changed["pred"])
        self.assertTrue(np.isfinite(first["pred"]).all())
        self.assertEqual(first["n_train"], 299)


if __name__ == "__main__":
    unittest.main()
