"""Tests for fixed comparison families, inherited scoring and report provenance."""
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

import linear_design_evaluate as evaluation
import linear_design_report as report
import protocol_evaluate as kernel


def registered_models():
    return [m | dict(months=108, rows=evaluation.EXPECTED_ROWS) for m in evaluation.planned_models()]


class LinearDesignComparisonTests(unittest.TestCase):
    def test_plan_and_directional_edges_match_declared_families(self):
        models = registered_models()
        evaluation.validate_matrix(models)
        self.assertEqual(len(models), 228)
        lookup = {m["model_id"]: m for m in models}
        for target in evaluation.TARGETS:
            specs = evaluation.contrast_specs([m for m in models if m["target"] == target])
            self.assertEqual(dict(Counter(s["family"] for s in specs)), evaluation.PLANNED_FAMILY_COUNTS)
            self.assertEqual(len({(s["family"], s["model_id"], s["benchmark_id"]) for s in specs}), len(specs))
            for edge in specs:
                model, reference = lookup[edge["model_id"]], lookup[edge["benchmark_id"]]
                self.assertEqual(model["target"], reference["target"])
                if edge["family"] == "refit":
                    self.assertEqual(tuple(model[k] for k in evaluation.DESIGN_FIELDS[:-1]),
                                     tuple(reference[k] for k in evaluation.DESIGN_FIELDS[:-1]))
                    self.assertEqual(edge["comparison"], model["refit_policy"] + "_minus_" + reference["refit_policy"])
                elif edge["family"] == "compression":
                    self.assertEqual((model["representation"], reference["representation"]), ("pca", "full"))
                    self.assertEqual(model["feature_set"], "characteristics_textcore")
                elif edge["family"] == "estimator":
                    self.assertEqual(reference["estimator"], "ols")

    def test_group_social_benchmark_and_pca_comparisons_are_explicit(self):
        models = [m for m in registered_models() if m["target"] == "raw"]
        lookup = {m["model_id"]: m for m in models}
        specs = evaluation.contrast_specs(models)
        group = [s for s in specs if s["comparison"] == "group_core_minus_ridge_characteristics"]
        self.assertEqual(len(group), 6)
        for edge in group:
            self.assertEqual(lookup[edge["benchmark_id"]]["feature_set"], "characteristics")
            self.assertEqual(lookup[edge["benchmark_id"]]["estimator"], "ridge")
        text = [s for s in specs if s["comparison"] == "pca_text_given_core"]
        self.assertEqual(len(text), 30)
        for edge in text:
            self.assertEqual(lookup[edge["benchmark_id"]]["representation"], "full")
            self.assertEqual(lookup[edge["benchmark_id"]]["feature_set"], "characteristics_core")

    def test_rejects_missing_procedures_rows_and_mislabeled_history(self):
        models = registered_models()
        with self.assertRaisesRegex(ValueError, "228"):
            evaluation.validate_matrix(models[:-1])
        models[0]["rows"] -= 1
        with self.assertRaisesRegex(ValueError, "unchanged"):
            evaluation.validate_matrix(models)
        models = registered_models()
        selected = next(m for m in models if m["history_policy"] == "select")
        selected["fit_days"] = 504
        with self.assertRaisesRegex(ValueError, "declared history"):
            evaluation.validate_matrix(models)

    def test_paired_hac_uses_calendar_gaps_and_full_family_denominator(self):
        models = [m for m in registered_models() if m["target"] == "raw" and m["estimator"] == "ols"
                  and m["feature_set"] == "characteristics" and m["history_policy"] == "fixed504"
                  and m["refit_policy"] in ("retain", "recent")]
        dates = pd.bdate_range("2014-01-02", periods=6)
        delta = np.array([.01, -.02, np.nan, .03, .015, .02])
        rows = []
        for m in models:
            for i, date in enumerate(dates):
                rows.append(dict(date=date, model_id=m["model_id"], rank_ic=0.0 if m["refit_policy"] == "retain" else delta[i],
                                 ew_spread_bp=1.0, cap_spread_bp=np.nan))
        contrasts = evaluation.paired_contrasts(pd.DataFrame(rows), models, lags=(1,))
        row = contrasts[contrasts.metric == "rank_ic"].iloc[0]
        expected = kernel.calendar_hac_mean(pd.Series(delta, index=dates), lags=1)
        self.assertAlmostEqual(row["mean"], expected["mean"])
        self.assertAlmostEqual(row["se"], expected["se"])
        self.assertEqual(row["family_size"], 114)
        self.assertEqual(row["observed_family_size"], 1)
        self.assertEqual(row["missing_days"], 1)
        self.assertAlmostEqual(row["p_bonferroni"], min(1., expected["p"] * 114))
        self.assertTrue(contrasts[contrasts.metric == "cap_spread_bp"].p_bonferroni.isna().all())

    def test_adapters_restore_all_original_functions_after_failure(self):
        original_contrasts, original_summary = kernel.paired_contrasts, kernel.summarize_daily
        with self.assertRaises(RuntimeError):
            with evaluation.design_adapters():
                self.assertIs(kernel.paired_contrasts, evaluation.paired_contrasts)
                self.assertIsNot(kernel.summarize_daily, original_summary)
                raise RuntimeError("synthetic evaluation failure")
        self.assertIs(kernel.paired_contrasts, original_contrasts)
        self.assertIs(kernel.summarize_daily, original_summary)

    def test_cannot_score_2023_even_in_pilot_mode(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main(["--registry", "missing", "--prepared", "missing", "--out", "missing",
                             "--end", "2023-12-31", "--allow-pilot"])


class LinearDesignArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = (Path(__file__).resolve().parents[1] / ".runs").resolve()
        root.mkdir(exist_ok=True)
        cls.directory = (root / ("linear_design_eval_test_" + uuid.uuid4().hex)).resolve()
        if cls.directory.parent != root:
            raise AssertionError("Fixture escaped the intended workspace")
        cls.directory.mkdir()
        cls.prepared = cls.directory / "prepared"
        cls.prepared.mkdir()
        dates, n = pd.bdate_range("2014-01-02", periods=7), 110
        keys = pd.DataFrame(dict(date=np.repeat(dates, n), permno=np.tile(np.arange(n), len(dates))))
        keys.index = pd.Index(np.arange(len(keys)) + 777, name="mm_index")
        keys.to_pickle(cls.prepared / "keys.pkl")
        returns = np.tile(np.linspace(-.03, .03, n), len(dates))
        returns[0] = np.nan
        np.save(cls.prepared / "y.npy", np.column_stack([returns, returns / 2]))
        np.save(cls.prepared / "calendar.npy", dates.to_numpy(dtype="datetime64[D]"))
        pd.DataFrame(dict(cap=np.ones(len(keys)), log_volume=np.arange(len(keys)) % n),
                     index=keys.index).to_pickle(cls.prepared / "evaluation.pkl")
        (cls.prepared / "manifest.json").write_text(json.dumps(dict(
            targets=["f_cumret1", "ar_dgtw_1"], feature_sets={f: ["dummy"] for f in evaluation.FEATURES})), encoding="utf-8")
        models = [m for m in registered_models() if m["history_policy"] == "fixed504" and m["refit_policy"] == "retain"]
        for j, m in enumerate(models):
            path = cls.directory / (m["model_id"] + ".pkl")
            prediction = np.tile(np.arange(n), len(dates)) + np.sin(np.arange(len(keys)) + j) * 5
            keys.assign(prediction=prediction).to_pickle(path)
            m.update(predictions=str(path), sha256=evaluation.file_hash(path), rows=len(keys), months=1)
        cls.registry = cls.directory / "registry.json"
        cls.registry.write_text(json.dumps(dict(prepared=str(cls.prepared), config={"run_kind": "pilot"}, models=models)), encoding="utf-8")
        cls.prefix, cls.out = cls.directory / "evaluation", cls.directory / "report.md"
        args = ["--registry", str(cls.registry), "--prepared", str(cls.prepared), "--out", str(cls.prefix),
                "--start", "2014-01-01", "--end", "2014-01-31", "--skip-subgroups", "--skip-horizons",
                "--threads", "1", "--allow-pilot"]
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = evaluation.main(args)
            report.build_report(cls.registry, cls.prefix, cls.out, allow_pilot=True)

    @classmethod
    def tearDownClass(cls):
        root = (Path(__file__).resolve().parents[1] / ".runs").resolve()
        if cls.directory.parent != root:
            raise AssertionError("Refusing cleanup outside the intended fixture workspace")
        shutil.rmtree(cls.directory)

    def test_gzip_is_exact_deterministic_and_hash_inventory_complete(self):
        raw = self.prefix.with_name(self.prefix.name + "_daily.csv")
        compressed = raw.with_suffix(".csv.gz")
        self.assertEqual(gzip.decompress(compressed.read_bytes()), raw.read_bytes())
        self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
        self.assertEqual(Path(self.metadata["output_files"]["_daily"]["path"]), compressed)
        self.assertIn(compressed.name, self.metadata["output_sha256"])
        self.assertNotIn(raw.name, self.metadata["output_sha256"])
        self.assertEqual(self.metadata["planned_models"], 228)
        self.assertEqual(self.metadata["included_models"], 38)
        self.assertEqual(self.metadata["reused_kernel_sha256"]["protocol_evaluate.py"], evaluation.file_hash(kernel.__file__))
        for coverage in self.metadata["coverage"].values():
            self.assertEqual(coverage["common_prediction_stock_days"], 770)
            self.assertEqual(coverage["common_scored_stock_days"], 769)

    def test_metadata_fields_survive_summary_and_family_sizes_do_not_shrink(self):
        summary = pd.read_csv(self.prefix.with_suffix(".csv"))
        for column in evaluation.DESIGN_FIELDS:
            self.assertIn(column, summary)
        self.assertEqual(set(summary.representation), {"full", "pca"})
        contrasts = pd.read_csv(self.prefix.with_name(self.prefix.name + "_contrasts.csv"))
        self.assertEqual(set(contrasts.family), {"group_penalty", "compression", "incremental_social", "estimator"})
        for family, rows in contrasts.groupby("family"):
            self.assertEqual(set(rows.family_size), {evaluation.PLANNED_FAMILY_COUNTS[family]})
            self.assertEqual(set(rows.observed_family_size), {evaluation.PLANNED_FAMILY_COUNTS[family] // 6})
            np.testing.assert_allclose(rows.p_bonferroni, np.minimum(1., rows.p * rows.family_size), atol=1e-13)

    def test_report_explains_union_development_and_unready_holdout(self):
        text = self.out.read_text(encoding="utf-8")
        for feature in report.FEATURE_LABELS.values():
            self.assertIn(feature, text)
        self.assertIn("PLUMBING PILOT", text)
        self.assertIn("630 sessions, about 2.5 years", text)
        self.assertIn("2023 is not ready", text)
        self.assertIn("178 of 250", text)
        self.assertIn("embed_norm and embed_cos", text)
        self.assertIn("_daily.csv.gz", text)
        self.assertIn("registered menu, not the ordering", text)
        with self.assertRaisesRegex(ValueError, "Pilot reports"):
            report.build_report(self.registry, self.prefix, self.out)

    def test_report_refuses_changed_evaluation_artifact(self):
        summary = self.prefix.with_suffix(".csv")
        original = summary.read_bytes()
        try:
            summary.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                report.build_report(self.registry, self.prefix, self.out, allow_pilot=True)
        finally:
            summary.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
