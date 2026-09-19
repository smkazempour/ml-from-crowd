"""Temporal-integrity, feature-schema and immutable-parent tests."""
import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import stat
import unittest
import uuid

import numpy as np
import pandas as pd

from characteristic_data import (CONTROL_NAMES, MISSING_NAMES, PRESERVED,
                                 append_transformed, feature_schema, file_record,
                                 prepare, stock_controls)


@contextmanager
def temporary_directory():
    workspace = Path(__file__).resolve().parents[1]
    directory = workspace / ".runs/characteristic_tests"
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(workspace):
        raise ValueError("Test cleanup must remain inside the workspace")
    # Python 3.13's TemporaryDirectory uses a private Windows ACL that is not
    # writable from this sandbox token. Ordinary workspace directories inherit
    # the workspace ACL and remain safe to clean up under this checked root.
    target = directory / uuid.uuid4().hex
    target.mkdir()
    try:
        yield str(target)
    finally:
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe test cleanup path")
        shutil.rmtree(target)


def history(n=320, permno=10):
    calendar = pd.bdate_range("2010-01-01", periods=n)
    frame = pd.DataFrame({"date": calendar, "permno": permno, "ret": .01,
                          "prc": 10.0, "shrout": 1000.0, "vol": 100.0})
    return calendar, frame


def parent_schema():
    return {"feature_names": ["net_sentiment", "log_volume", "other", "embed", "missing__net_sentiment"],
            "feature_sets": {"core": [0, 1, 4], "all": [0, 1, 2, 4],
                             "textcore": [0, 1, 3, 4], "textall": [0, 1, 2, 3, 4]},
            "binary_features": ["missing__net_sentiment"],
            "scalar_rank_features": ["net_sentiment", "log_volume", "other"],
            "targets": ["f_cumret1", "ar_dgtw_1"], "fingerprint": "synthetic-parent"}


class MarketHistoryTests(unittest.TestCase):
    def test_exact_products_momentum_and_units(self):
        calendar, frame = history()
        result = stock_controls(frame, calendar)
        for horizon in (1, 5, 21, 63, 126, 252):
            self.assertAlmostEqual(result.iloc[-1][f"char_ret{horizon}"], 1.01 ** horizon - 1, places=11)
        self.assertAlmostEqual(result.iloc[-1]["char_momentum252_skip21"], 1.01 ** 231 - 1, places=11)
        self.assertAlmostEqual(result.iloc[-1]["char_log_market_cap"], np.log(10))
        self.assertAlmostEqual(result.iloc[-1]["char_turnover21"], .0001)
        self.assertAlmostEqual(result.iloc[-1]["char_log_dollar_volume21"], np.log1p(1000))
        self.assertAlmostEqual(result.iloc[-1]["char_amihud21"], .01 / 1000)
        self.assertTrue(np.isnan(result.iloc[0]["char_ret1"]))
        self.assertTrue(np.isnan(result.iloc[251]["char_ret252"]))
        self.assertTrue(np.isfinite(result.iloc[252]["char_ret252"]))

    def test_future_mutations_and_future_appends_do_not_change_past(self):
        calendar, frame = history()
        baseline = stock_controls(frame, calendar)
        mutated = frame.copy()
        mutated.loc[260:, ["ret", "prc", "vol", "shrout"]] = [-.2, 1000, -99, 2000]
        changed = stock_controls(mutated, calendar)
        pd.testing.assert_frame_equal(baseline.iloc[:260], changed.iloc[:260])
        truncated = stock_controls(frame.iloc[:260], calendar[:260])
        pd.testing.assert_frame_equal(baseline.iloc[:260], truncated)

    def test_calendar_gap_cannot_be_bridged_by_stock_row_rolling(self):
        calendar, frame = history(45)
        result = stock_controls(frame.drop(index=20), calendar)
        self.assertTrue(result.loc[calendar[20]].isna().all())
        # A reported return after an absent prior close may span multiple days.
        self.assertTrue(np.isnan(result.loc[calendar[21], "char_ret1"]))
        self.assertTrue(np.isnan(result.loc[calendar[25], "char_ret5"]))
        self.assertAlmostEqual(result.loc[calendar[26], "char_ret5"], 1.01 ** 5 - 1)
        self.assertTrue(np.isnan(result.loc[calendar[40], "char_turnover21"]))
        self.assertTrue(np.isfinite(result.loc[calendar[41], "char_turnover21"]))

    def test_minus_one_returns_and_invalid_negative_gross_returns(self):
        calendar, frame = history(40)
        frame.loc[10, "ret"] = -1
        result = stock_controls(frame, calendar)
        self.assertEqual(result.iloc[10]["char_ret1"], -1)
        self.assertEqual(result.iloc[14]["char_ret5"], -1)
        self.assertAlmostEqual(result.iloc[15]["char_ret5"], 1.01 ** 5 - 1)
        frame.loc[10, "ret"] = -1.01
        result = stock_controls(frame, calendar)
        self.assertTrue(np.isnan(result.iloc[14]["char_ret5"]))

    def test_zero_volume_negative_volume_and_negative_price_quote(self):
        calendar, frame = history(45)
        frame.loc[26, "vol"] = 0
        frame.loc[27, "vol"] = -99
        frame.loc[28, "prc"] = -10
        frame.loc[29, "shrout"] = 0
        result = stock_controls(frame, calendar)
        self.assertEqual(result.iloc[26]["char_turnover1"], 0)
        self.assertEqual(result.iloc[26]["char_log_dollar_volume1"], 0)
        self.assertTrue(np.isnan(result.iloc[26]["char_amihud21"]))
        self.assertTrue(np.isnan(result.iloc[27]["char_turnover1"]))
        self.assertTrue(np.isnan(result.iloc[27]["char_log_dollar_volume21"]))
        self.assertAlmostEqual(result.iloc[28]["char_log_price"], np.log(10))
        self.assertTrue(np.isnan(result.iloc[29]["char_log_market_cap"]))
        self.assertTrue(np.isnan(result.iloc[29]["char_turnover1"]))

    def test_skipped_recent_month_does_not_enter_momentum(self):
        calendar, frame = history()
        baseline = stock_controls(frame, calendar)
        frame.loc[299:319, "ret"] = np.nan
        result = stock_controls(frame, calendar)
        self.assertEqual(result.iloc[-1]["char_momentum252_skip21"],
                         baseline.iloc[-1]["char_momentum252_skip21"])
        self.assertTrue(np.isnan(result.iloc[-1]["char_ret252"]))


class SchemaTests(unittest.TestCase):
    def test_feature_indices_and_flags_preserve_parent_semantics(self):
        parent = parent_schema()
        names, sets = feature_schema(parent)
        self.assertEqual(names[:5], parent["feature_names"])
        self.assertEqual(names[5:], CONTROL_NAMES + MISSING_NAMES)
        for name, indices in parent["feature_sets"].items():
            self.assertEqual(sets[name], indices)
        self.assertEqual(sets["characteristics_sentiment"][:2], [0, 4])
        self.assertEqual(sets["characteristics_attention"][0], 1)
        self.assertEqual(sets["characteristics_textall"], list(range(39)))
        self.assertEqual(len(sets["characteristics"]), 34)

    def test_rank_missing_flags_and_exact_social_prefix(self):
        original = np.array([[1, np.nan], [2, -0.0], [3, 9]], dtype=np.float32)
        raw = np.ones((3, 17), dtype=np.float64)
        raw[:, 0] = [4, np.nan, 2]
        with temporary_directory() as directory:
            destination = Path(directory) / "X.npy"
            append_transformed(original, raw, np.zeros(3, dtype=int), destination)
            new = np.load(destination)
            self.assertEqual(new[:, :2].tobytes(), original.tobytes())
            np.testing.assert_allclose(new[:, 2], [.5, 0, -.5])
            np.testing.assert_allclose(new[:, 19], [0, 1, 0])
            self.assertEqual(new[:, 20:].sum(), 0)

    def test_synthetic_prepare_preserves_every_parent_artifact(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            try:
                parent = root / "parent"
                parent.mkdir()
                calendar, first = history()
                second = first.assign(permno=20, ret=.02, prc=20)
                market = pd.concat([first, second], ignore_index=True)
                # One source gap must remain a missing control, not delete a parent row.
                market = market.loc[~((market.permno == 20) & (market.date == calendar[-5]))]
                market["date"] = market["date"].dt.date
                source = root / "dsf.pkl"
                market.to_pickle(source)
                keys = pd.DataFrame({"date": np.repeat(calendar[-10:], 2),
                                     "permno": [10, 20] * 10, "ticker": ["A", "B"] * 10})
                keys.index = np.arange(100, 120)
                keys.to_pickle(parent / "keys.pkl")
                keys.assign(f_cumret1=np.nan).to_pickle(parent / "evaluation.pkl")
                old_X = np.arange(100, dtype=np.float32).reshape(20, 5)
                np.save(parent / "X.npy", old_X)
                np.save(parent / "codes.npy", np.repeat(np.arange(310, 320), 2))
                np.save(parent / "calendar.npy", calendar.to_numpy())
                # Labels intentionally absent: controls must still rank all input rows.
                np.save(parent / "y.npy", np.full((20, 2), np.nan))
                np.save(parent / "q.npy", np.full((20, 2), np.nan, dtype=np.float32))
                schema = parent_schema()
                (parent / "manifest.json").write_text(json.dumps(schema))
                before = {name: file_record(parent / name) for name in (*PRESERVED, "X.npy", "manifest.json")}
                output = prepare(parent, source, root / "out")
                for name in PRESERVED:
                    self.assertEqual(file_record(output / name), before[name])
                for name, record in before.items():
                    self.assertEqual(file_record(parent / name), record)
                new_X = np.load(output / "X.npy")
                self.assertEqual(new_X[:, :5].tobytes(), old_X.tobytes())
                np.testing.assert_allclose(new_X[:2, 5], [-.5, .5])
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["derivation"]["parent_feature_count"], 5)
                self.assertEqual(manifest["feature_sets"]["core"], schema["feature_sets"]["core"])
                audit = json.loads((output / "characteristic_audit.json").read_text())
                self.assertEqual(audit["parent_rows"], 20)
                self.assertEqual(audit["parent_keys_absent_from_raw_history"], 1)
                self.assertEqual(prepare(parent, source, root / "out"), output)
            finally:
                for path in root.rglob("*"):
                    if path.is_file():
                        path.chmod(stat.S_IWRITE | stat.S_IREAD)


if __name__ == "__main__":
    unittest.main()
