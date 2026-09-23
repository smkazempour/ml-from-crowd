"""Optimization comparison budgets, bounded-memory parity and artifact provenance."""
from __future__ import annotations

from collections import Counter
import contextlib
import gzip
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

import linear_optimization_evaluate as evaluation
import linear_optimization_report as report
import protocol_evaluate as kernel


def registered_models():
    return [m | dict(months=108, rows=evaluation.EXPECTED_ROWS) for m in evaluation.planned_models()]


class OptimizationGraphTests(unittest.TestCase):
    def test_full_graph_and_semantically_matched_benchmarks(self):
        models = registered_models()
        evaluation.validate_matrix(models)
        self.assertEqual(len(models), 172)
        self.assertEqual(evaluation.PLANNED_FAMILY_COUNTS, {"transformation": 43, "estimator": 76,
            "penalty_expansion": 38, "incremental_social": 72, "group_penalty": 16, "compression": 36, "pca_expansion": 18})
        lookup = {m["model_id"]: m for m in models}
        for target in evaluation.TARGETS:
            specs = evaluation.contrast_specs([m for m in models if m["target"] == target])
            self.assertEqual(dict(Counter(s["family"] for s in specs)), evaluation.PLANNED_FAMILY_COUNTS)
            self.assertEqual(len({(s["family"], s["model_id"], s["benchmark_id"]) for s in specs}), len(specs))
            for spec in specs:
                candidate, baseline = lookup[spec["model_id"]], lookup[spec["benchmark_id"]]
                self.assertEqual(candidate["target"], baseline["target"])
                changed = {k for k in evaluation.DESIGN_FIELDS if candidate[k] != baseline[k]}
                if spec["family"] in ("transformation", "penalty_expansion", "pca_expansion", "group_penalty"):
                    axis = {"transformation": "transform", "penalty_expansion": "penalty_grid",
                            "pca_expansion": "pca_grid", "group_penalty": "estimator"}[spec["family"]]
                    self.assertEqual(changed, {axis})
                if spec["family"] == "estimator":
                    self.assertEqual(baseline["estimator"], "ols")
                    self.assertEqual(baseline["penalty_grid"], "original")
                    self.assertTrue(changed <= {"estimator", "penalty_grid"})
                if spec["comparison"] == "pca_text_given_core":
                    self.assertEqual((baseline["representation"], baseline["pca_grid"]), ("full", "none"))

    def test_production_rejects_missing_models_changed_sample_and_noop_axis(self):
        with self.assertRaisesRegex(ValueError, "172"):
            evaluation.validate_matrix(registered_models()[:-1])
        models = registered_models()
        models[0]["rows"] -= 1
        with self.assertRaisesRegex(ValueError, "unchanged"):
            evaluation.validate_matrix(models)
        models = registered_models()
        next(m for m in models if m["estimator"] == "ols")["penalty_grid"] = "expanded"
        with self.assertRaisesRegex(ValueError, "unplanned"):
            evaluation.validate_matrix(models)

    def test_registry_budgets_must_be_predeclared_even_for_subsets(self):
        with self.assertRaisesRegex(ValueError, "predeclare"):
            evaluation.registered_family_counts({})
        with self.assertRaisesRegex(ValueError, "predeclare"):
            evaluation.registered_family_counts({"planned_family_counts": {"transformation": 1}})
        self.assertEqual(evaluation.registered_family_counts({"config": {"planned_family_counts": evaluation.PLANNED_FAMILY_COUNTS}}),
                         evaluation.PLANNED_FAMILY_COUNTS)

    def test_calendar_gaps_and_budget_retained_for_one_pair(self):
        models = [m for m in registered_models() if m["target"] == "raw" and m["feature_set"] == "characteristics"
                  and m["estimator"] == "ols"]
        dates, delta = pd.bdate_range("2014-01-02", periods=6), np.array([.01, -.02, np.nan, .03, .015, .02])
        daily = pd.DataFrame([dict(date=date, model_id=m["model_id"], rank_ic=delta[i] if m["transform"] == "daily_rank" else 0.,
                                   ew_spread_bp=1., cap_spread_bp=np.nan)
                              for m in models for i, date in enumerate(dates)])
        contrasts = evaluation.paired_contrasts(daily, models, lags=(1,))
        row = contrasts[contrasts.metric == "rank_ic"].iloc[0]
        expected = kernel.calendar_hac_mean(pd.Series(delta, index=dates), lags=1)
        self.assertAlmostEqual(row["se"], expected["se"])
        self.assertEqual(row.family_size, 43)
        self.assertEqual(row.observed_family_size, 1)
        self.assertEqual(row.missing_days, 1)
        self.assertAlmostEqual(row.p_bonferroni, min(1., expected["p"] * 43))
        self.assertTrue(contrasts[contrasts.metric == "cap_spread_bp"].p_bonferroni.isna().all())

    def test_2023_forbidden_even_in_pilot(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main(["--registry", "missing", "--prepared", "missing", "--out", "missing",
                             "--end", "2023-12-31", "--allow-pilot"])


class OptimizationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (Path(__file__).resolve().parents[1] / ".runs").resolve()
        cls.root.mkdir(exist_ok=True)
        cls.directory = cls.root / ("linear_optimization_eval_test_" + uuid.uuid4().hex)
        cls.directory.mkdir()
        cls.prepared = cls.directory / "prepared"
        cls.prepared.mkdir()
        dates, n = pd.bdate_range("2014-01-02", periods=7), 110
        cls.keys = pd.DataFrame(dict(date=np.repeat(dates, n), permno=np.tile(np.arange(n), len(dates))))
        cls.keys.index = pd.Index(np.arange(len(cls.keys)) + 777, name="mm_index")
        cls.keys.to_pickle(cls.prepared / "keys.pkl")
        cls.returns = np.tile(np.linspace(-.03, .03, n), len(dates))
        cls.returns[0] = np.nan
        np.save(cls.prepared / "y.npy", np.column_stack([cls.returns, cls.returns / 2]))
        np.save(cls.prepared / "calendar.npy", dates.to_numpy(dtype="datetime64[D]"))
        cls.cap = np.tile(np.arange(n, dtype=float) + 1., len(dates))
        cls.cap[3], cls.cap[4] = 0., np.nan
        pd.DataFrame(dict(cap=cls.cap, log_volume=np.arange(len(cls.keys)) % n,
                         f_cumret3=cls.returns * 1.2, f_cumret63=cls.returns * 2),
                     index=cls.keys.index).to_pickle(cls.prepared / "evaluation.pkl")
        (cls.prepared / "manifest.json").write_text(json.dumps(dict(targets=["f_cumret1", "ar_dgtw_1"])), encoding="utf-8")
        # A tractable complete subset still exercises every comparison family.
        cls.models = [m for m in registered_models() if m["target"] == "raw" and m["estimator"] in ("ols", "ridge", "group_ridge")
                      and (m["transform"] == "standard" or (m["estimator"] == "ols" and m["feature_set"] == "characteristics"))]
        cls.predictions = []
        for j, model in enumerate(cls.models):
            path = cls.directory / (model["model_id"] + ".pkl")
            prediction = np.tile(np.arange(n), len(dates)).astype(float) + np.sin(np.arange(len(cls.keys)) + j) * 5
            if j == len(cls.models) - 1:
                prediction[5] = np.nan  # Late-batch missingness must affect every model.
            cls.predictions.append(prediction)
            cls.keys.assign(prediction=prediction).to_pickle(path)
            model.update(predictions=str(path), sha256=evaluation.file_hash(path), rows=len(cls.keys), months=1)
        cls.registry = cls.directory / "registry.json"
        cls.registry.write_text(json.dumps(dict(prepared=str(cls.prepared), config={"run_kind": "pilot"},
            planned_family_counts=evaluation.PLANNED_FAMILY_COUNTS, models=cls.models)), encoding="utf-8")
        pd.DataFrame({"stability_same_winner_fraction": [.5, 1.], "stability_distinct_winners": [3, 1]}).to_csv(
            cls.directory / "monthly_selection.csv", index=False)
        cls.prefix, cls.output, cls.digest = cls.directory / "evaluation", cls.directory / "report.md", cls.directory / "summary.md"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = evaluation.main(["--registry", str(cls.registry), "--prepared", str(cls.prepared), "--outprefix", str(cls.prefix),
                "--start", "2014-01-01", "--end", "2014-01-31", "--skip-subgroups", "--skip-horizons", "--threads", "1", "--batch-size", "3", "--allow-pilot"])
            report.build_report(cls.registry, cls.prefix, cls.output, allow_pilot=True, summary_output=cls.digest)

    @classmethod
    def tearDownClass(cls):
        if cls.directory.resolve().parent != cls.root:
            raise AssertionError("Refusing cleanup outside the explicit fixture workspace")
        shutil.rmtree(cls.directory)

    def test_bounded_batches_equal_dense_kernel_on_global_intersection(self):
        matrix = np.column_stack(self.predictions)
        common = np.isfinite(matrix).all(axis=1)
        calendar = pd.DatetimeIndex(np.load(self.prepared / "calendar.npy"))
        codes = calendar.get_indexer(self.keys.date)
        dense, _ = kernel.score_scope(matrix[common], self.returns[common], self.cap[common], codes[common], calendar, self.models)
        actual = pd.read_csv(self.prefix.with_name(self.prefix.name + "_daily.csv.gz"), parse_dates=["date"])
        columns = ["date", "model_id"] + kernel.DAY_COLUMNS
        dense = dense[columns].sort_values(["date", "model_id"]).reset_index(drop=True)
        actual = actual[columns].sort_values(["date", "model_id"]).reset_index(drop=True)
        dense["date"] = dense.date.astype("datetime64[ns]")
        pd.testing.assert_frame_equal(actual, dense, check_exact=False, atol=1e-11, rtol=1e-11)
        self.assertEqual(self.metadata["coverage"]["raw"]["common_prediction_stock_days"], 769)
        self.assertEqual(self.metadata["coverage"]["raw"]["common_scored_stock_days"], 768)

    def test_optional_subgroups_and_horizons_reuse_original_scoring(self):
        models = self.models[:2]
        prefix = self.directory / "diagnostics"
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = evaluation.score_registry(self.registry, self.prepared, prefix, models,
                evaluation.PLANNED_FAMILY_COUNTS, "2014-01-01", "2014-01-31", batch_size=1)
        subgroups = pd.read_csv(prefix.with_name(prefix.name + "_subgroups.csv"))
        horizons = pd.read_csv(prefix.with_name(prefix.name + "_horizons.csv"))
        self.assertEqual(set(subgroups.subgroup), {f"{label}_tercile_{n}" for label in ("size", "activity") for n in (1, 2, 3)})
        self.assertEqual(set(horizons.horizon), {3, 63})
        self.assertEqual(set(horizons.loc[horizons.horizon == 63, "hac_lags"]), {62})
        calendar = pd.DatetimeIndex(np.load(self.prepared / "calendar.npy"))
        codes = calendar.get_indexer(self.keys.date)
        daily, _ = kernel.score_scope(np.column_stack(self.predictions[:2]), self.returns * 1.2,
                                      self.cap, codes, calendar, models)
        expected = evaluation.summarize_daily(daily, models)
        actual = horizons[horizons.horizon == 3]
        np.testing.assert_allclose(actual["mean"], expected["mean"], atol=1e-11, rtol=1e-11, equal_nan=True)
        self.assertEqual(len(metadata["skipped_horizons"]), 4)

    def test_hac_tables_metadata_and_deterministic_gzip(self):
        raw = self.prefix.with_name(self.prefix.name + "_daily.csv")
        compressed = raw.with_suffix(".csv.gz")
        self.assertEqual(gzip.decompress(compressed.read_bytes()), raw.read_bytes())
        self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
        self.assertEqual(self.metadata["planned_models"], 172)
        self.assertEqual(self.metadata["prediction_batch_size"], 3)
        summary = pd.read_csv(self.prefix.with_suffix(".csv"))
        self.assertTrue(set(evaluation.DESIGN_FIELDS) <= set(summary.columns))
        contrasts = pd.read_csv(self.prefix.with_name(self.prefix.name + "_contrasts.csv"))
        self.assertEqual(set(contrasts.family), set(evaluation.PLANNED_FAMILY_COUNTS))
        for family, frame in contrasts.groupby("family"):
            self.assertEqual(set(frame.family_size), {evaluation.PLANNED_FAMILY_COUNTS[family]})
        for filename, digest in self.metadata["output_sha256"].items():
            self.assertEqual(evaluation.file_hash(self.directory / filename), digest)

    def test_full_report_and_concise_companion_preserve_caveats(self):
        full, digest = self.output.read_text(encoding="utf-8"), self.digest.read_text(encoding="utf-8")
        for text in (full, digest):
            self.assertIn("PLUMBING PILOT", text)
            self.assertIn("development", text)
            self.assertIn("178 of 250", text)
        self.assertIn("630 fitting sessions", full)
        self.assertIn("stability_same_winner_fraction", full)
        self.assertLess(len(digest.split()), 900)
        with self.assertRaisesRegex(ValueError, "Pilot reports"):
            report.build_report(self.registry, self.prefix, self.output)

    def test_report_rejects_tampered_artifacts_and_selection_diagnostics(self):
        for path, message in ((self.prefix.with_suffix(".csv"), "artifact changed"),
                              (self.directory / "monthly_selection.csv", "selection artifact changed")):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    report.build_report(self.registry, self.prefix, self.output, allow_pilot=True, summary_output=self.digest)
            finally:
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
