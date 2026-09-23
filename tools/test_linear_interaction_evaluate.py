"""Stage A's fixed graph, bounded common-sample scoring and report provenance."""
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

import linear_interaction_evaluate as evaluation
import linear_interaction_report as report
import linear_optimization_evaluate as predecessor
import protocol_evaluate as kernel


def registered_models():
    return [m | dict(months=108, rows=evaluation.EXPECTED_ROWS) for m in evaluation.planned_models()]


class InteractionGraphTests(unittest.TestCase):
    def test_complete_graph_has_correct_conditional_references(self):
        models = registered_models()
        evaluation.validate_matrix(models)
        self.assertEqual(len(models), 148)
        self.assertEqual(evaluation.PLANNED_FAMILY_COUNTS, {"characteristic_shape": 9, "social_information": 12,
            "social_shape": 9, "individual_social": 35, "text_information": 18, "text_shape": 15,
            "estimator": 39, "additive_reference": 18})
        lookup = {m["model_id"]: m for m in models}
        for target in evaluation.TARGETS:
            specs = evaluation.contrast_specs([m for m in models if m["target"] == target])
            self.assertEqual(len(specs), 155)
            self.assertEqual(dict(Counter(s["family"] for s in specs)), evaluation.PLANNED_FAMILY_COUNTS)
            self.assertEqual(len({(s["family"], s["model_id"], s["benchmark_id"]) for s in specs}), len(specs))
            individuals = []
            for spec in specs:
                candidate, baseline = lookup[spec["model_id"]], lookup[spec["benchmark_id"]]
                self.assertEqual(candidate["target"], baseline["target"])
                if spec["family"] == "individual_social":
                    self.assertEqual((candidate["estimator"], baseline["estimator"]), ("ridge", "ridge"))
                    self.assertEqual((candidate["n_features"], baseline["n_features"]), (192, 191))
                    self.assertEqual(baseline["basis"], "cq_s_squares")
                    individuals.append(candidate["interaction_term"])
                elif spec["family"] == "estimator":
                    self.assertEqual(candidate["basis"], baseline["basis"])
                    self.assertEqual(baseline["estimator"], "ridge" if spec["comparison"] == "enet_versus_ridge" else "ols")
                else:
                    self.assertEqual(candidate["estimator"], baseline["estimator"])
                if spec["comparison"] == "embedding_c_given_agreement_c":
                    self.assertEqual((candidate["basis"], baseline["basis"]), ("text_c", "agreement_c"))
                    self.assertEqual(candidate["n_features"] - baseline["n_features"], 272)
                if spec["comparison"] == "all_text_c_given_text_squares":
                    self.assertEqual(candidate["n_features"] - baseline["n_features"], 306)
                if spec["comparison"] == "text_s_given_text_c":
                    self.assertEqual(candidate["n_features"] - baseline["n_features"], 36)
            self.assertEqual(set(individuals), set(evaluation.social_terms()))

    def test_declared_menu_agrees_with_fitting_core_metadata(self):
        import linear_interaction_core as core
        actual = []
        for spec in core.procedure_specs():
            for target in evaluation.TARGETS:
                actual.append(spec | dict(model_id=core.model_name(spec, target), target=target,
                    history_policy=spec["history"], refit_policy=spec["refit"], fit_days=504, validation_days=126,
                    horizon=1, target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target]))
        evaluation.validate_matrix(actual, allow_pilot=True)
        self.assertEqual({m["model_id"] for m in actual}, {m["model_id"] for m in evaluation.planned_models()})

    def test_rejects_missing_models_changed_bases_unknown_terms_and_unregistered_budget(self):
        with self.assertRaisesRegex(ValueError, "148"):
            evaluation.validate_matrix(registered_models()[:-1])
        models = registered_models()
        models[0]["n_features"] += 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            evaluation.validate_matrix(models)
        models = registered_models()
        next(m for m in models if m["basis"] == "single_social192")["interaction_term"] = "unplanned_term"
        with self.assertRaisesRegex(ValueError, "unplanned"):
            evaluation.validate_matrix(models)
        with self.assertRaisesRegex(ValueError, "predeclare"):
            evaluation.registered_family_counts({"planned_family_counts": {"individual_social": 1}})
        models = registered_models()
        models[0]["rows"] -= 1
        with self.assertRaisesRegex(ValueError, "unchanged"):
            evaluation.validate_matrix(models)

    def test_one_term_keeps_whole_search_budget_and_calendar_gaps(self):
        models = [m for m in registered_models() if m["target"] == "raw" and m["estimator"] == "ridge"
                  and (m["basis"] == "cq_s_squares" or m["interaction_term"] == evaluation.social_terms()[0])]
        dates = pd.bdate_range("2014-01-02", periods=6)
        delta = np.array([.01, -.02, np.nan, .03, .015, .02])
        daily = pd.DataFrame([dict(date=date, model_id=m["model_id"], rank_ic=delta[i] if m["interaction_term"] else 0.,
                                  ew_spread_bp=1., cap_spread_bp=np.nan) for m in models for i, date in enumerate(dates)])
        contrasts = evaluation.paired_contrasts(daily, models, lags=(1,))
        row = contrasts[contrasts.metric == "rank_ic"].iloc[0]
        expected = kernel.calendar_hac_mean(pd.Series(delta, index=dates), lags=1)
        self.assertAlmostEqual(row["se"], expected["se"])
        self.assertEqual(row.family_size, 35)
        self.assertEqual(row.observed_family_size, 1)
        self.assertEqual(row.missing_days, 1)
        self.assertAlmostEqual(row.p_bonferroni, min(1., expected["p"] * 35))
        self.assertTrue(contrasts[contrasts.metric == "cap_spread_bp"].p_bonferroni.isna().all())

    def test_2023_forbidden_even_in_pilot(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main(["--registry", "missing", "--prepared", "missing", "--out", "missing",
                             "--end", "2023-12-31", "--allow-pilot"])


class InteractionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (Path(__file__).resolve().parents[1] / ".runs").resolve()
        cls.root.mkdir(exist_ok=True)
        cls.directory = cls.root / ("linear_interaction_eval_test_" + uuid.uuid4().hex)
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
        cls.models = [m for m in registered_models() if m["target"] == "raw" or (m["basis"] == "additive_c" and m["estimator"] != "enet")]
        cls.predictions = {}
        for j, model in enumerate(cls.models):
            path = cls.directory / (model["model_id"] + ".pkl")
            prediction = np.tile(np.arange(n), len(dates)).astype(float) + np.sin(np.arange(len(cls.keys)) + j) * 5
            if model["interaction_term"] == evaluation.social_terms()[-1]:
                prediction[5] = np.nan  # Late-batch missingness affects every raw-return model.
            cls.predictions[model["model_id"]] = prediction
            cls.keys.assign(prediction=prediction).to_pickle(path)
            model.update(predictions=str(path), sha256=evaluation.file_hash(path), rows=len(cls.keys), months=1)
        cls.registry = cls.directory / "registry.json"
        cls.registry.write_text(json.dumps(dict(prepared=str(cls.prepared), config={"run_kind": "pilot"},
            planned_family_counts=evaluation.PLANNED_FAMILY_COUNTS, models=cls.models)), encoding="utf-8")
        pd.DataFrame({"model_id": [cls.models[0]["model_id"]], "validation_ic": [.01]}).to_csv(
            cls.directory / "monthly_selection.csv", index=False)
        cls.prefix, cls.output, cls.digest = cls.directory / "evaluation", cls.directory / "report.md", cls.directory / "summary.md"
        cls.functions_before = (predecessor.summarize_daily, predecessor.paired_contrasts, kernel.score_scope, kernel.calendar_hac_mean)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = evaluation.main(["--registry", str(cls.registry), "--prepared", str(cls.prepared), "--outprefix", str(cls.prefix),
                "--start", "2014-01-01", "--end", "2014-01-31", "--skip-subgroups", "--skip-horizons", "--threads", "1", "--batch-size", "3", "--allow-pilot"])
            report.main(["--registry", str(cls.registry), "--evaluation-prefix", str(cls.prefix), "--output", str(cls.output),
                         "--summary-output", str(cls.digest), "--allow-pilot"])

    @classmethod
    def tearDownClass(cls):
        if cls.directory.resolve().parent != cls.root:
            raise AssertionError("Refusing cleanup outside the explicit fixture workspace")
        shutil.rmtree(cls.directory)

    def test_bounded_batches_match_dense_scoring_and_preserve_module_state(self):
        models = [m for m in self.models if m["target"] == "raw"]
        matrix = np.column_stack([self.predictions[m["model_id"]] for m in models])
        common = np.isfinite(matrix).all(axis=1)
        calendar = pd.DatetimeIndex(np.load(self.prepared / "calendar.npy"))
        codes = calendar.get_indexer(self.keys.date)
        dense, _ = kernel.score_scope(matrix[common], self.returns[common], self.cap[common], codes[common], calendar, models)
        actual = pd.read_csv(self.prefix.with_name(self.prefix.name + "_daily.csv.gz"), parse_dates=["date"])
        columns = ["date", "model_id"] + kernel.DAY_COLUMNS
        dense = dense[columns].sort_values(["date", "model_id"]).reset_index(drop=True)
        actual = actual[actual.target == "raw"][columns].sort_values(["date", "model_id"]).reset_index(drop=True)
        dense["date"] = dense.date.astype("datetime64[ns]")
        pd.testing.assert_frame_equal(actual, dense, check_exact=False, atol=1e-11, rtol=1e-11)
        self.assertEqual(self.metadata["coverage"]["raw"]["common_prediction_stock_days"], 769)
        self.assertEqual(self.metadata["coverage"]["raw"]["common_scored_stock_days"], 768)
        self.assertEqual(self.metadata["coverage"]["dgtw"]["common_prediction_stock_days"], 770)
        self.assertEqual(self.functions_before, (predecessor.summarize_daily, predecessor.paired_contrasts, kernel.score_scope, kernel.calendar_hac_mean))

    def test_optional_subgroups_and_horizons(self):
        models = [m for m in self.models if m["target"] == "raw"][:2]
        prefix = self.directory / "diagnostics"
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = evaluation.score_registry(self.registry, self.prepared, prefix, models,
                evaluation.PLANNED_FAMILY_COUNTS, "2014-01-01", "2014-01-31", batch_size=1)
        subgroups = pd.read_csv(prefix.with_name(prefix.name + "_subgroups.csv"))
        horizons = pd.read_csv(prefix.with_name(prefix.name + "_horizons.csv"))
        self.assertEqual(set(subgroups.subgroup), {f"{label}_tercile_{n}" for label in ("size", "activity") for n in (1, 2, 3)})
        self.assertEqual(set(horizons.horizon), {3, 63})
        self.assertEqual(set(horizons.loc[horizons.horizon == 63, "hac_lags"]), {62})
        self.assertEqual(len(metadata["skipped_horizons"]), 4)

    def test_nonempty_artifacts_complete_hashes_and_deterministic_gzip(self):
        raw = self.prefix.with_name(self.prefix.name + "_daily.csv")
        compressed = raw.with_suffix(".csv.gz")
        self.assertEqual(gzip.decompress(compressed.read_bytes()), raw.read_bytes())
        self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
        self.assertEqual(self.metadata["schema_version"], "linear_interactions_v1_evaluation")
        self.assertEqual(self.metadata["planned_models"], 148)
        for name in ("summary", "_contrasts", "_coverage", "_yearly", "_periods"):
            self.assertFalse(pd.read_csv(self.metadata["output_files"][name]["path"]).empty)
        summary = pd.read_csv(self.prefix.with_suffix(".csv"))
        self.assertTrue(set(evaluation.SUMMARY_FIELDS) <= set(summary.columns))
        contrasts = pd.read_csv(self.prefix.with_name(self.prefix.name + "_contrasts.csv"))
        self.assertEqual(set(contrasts.family), set(evaluation.PLANNED_FAMILY_COUNTS))
        for family, frame in contrasts.groupby("family"):
            self.assertEqual(set(frame.family_size), {evaluation.PLANNED_FAMILY_COUNTS[family]})
        for filename, digest in self.metadata["output_sha256"].items():
            self.assertEqual(evaluation.file_hash(self.directory / filename), digest)

    def test_reports_retain_all_terms_correct_ranks_pilot_and_development_limits(self):
        full, digest = self.output.read_text(encoding="utf-8"), self.digest.read_text(encoding="utf-8")
        for text in (full, digest):
            self.assertIn("PLUMBING PILOT", text)
            self.assertIn("development", text)
            self.assertIn("178 of 250", text)
            self.assertIn("already use daily ranks", text)
        for term in evaluation.social_terms():
            self.assertIn(report.term_label(term), full)
        self.assertIn("191-column", full)
        self.assertIn("272 embedding-PC", full)
        self.assertIn("2019-22 adjusted p", full)
        self.assertIn("no scientific answer", digest)
        self.assertLess(len(digest.split()), 1050)
        with self.assertRaisesRegex(ValueError, "Pilot reports"):
            report.build_report(self.registry, self.prefix, self.output)

    def test_report_rejects_tampered_artifact_or_selection_file(self):
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
