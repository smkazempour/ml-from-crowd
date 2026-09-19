"""Checks for the conditioning experiment's graph, inference and publication."""
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

import characteristic_evaluate as ce
import characteristic_report as cr
import protocol_evaluate as kernel


def registered_models(windows=ce.WINDOWS):
    models = [m for m in ce.planned_models() if m["fit_days"] in windows]
    for model in models:
        model.update(months=108, rows=ce.EXPECTED_ROWS,
                     target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]])
    return models


class CharacteristicContrastTests(unittest.TestCase):
    def test_registered_counts_and_exact_family_denominators(self):
        models = registered_models()
        self.assertEqual(len(models), 264)
        self.assertEqual(ce.validate_matrix(models), [504, 252, 756])
        for target in ce.TARGETS:
            specs = ce.contrast_specs([m for m in models if m["target"] == target])
            self.assertEqual(dict(Counter(s["family"] for s in specs)),
                             {"incremental_social": 144, "conditioning": 48,
                              "estimator": 99, "history": 88})
            triples = [(s["family"], s["model_id"], s["benchmark_id"]) for s in specs]
            self.assertEqual(len(triples), len(set(triples)))

    def test_edges_hold_estimator_history_and_controls_fixed(self):
        models = [m for m in registered_models() if m["target"] == "raw"]
        lookup = {m["model_id"]: m for m in models}
        allowed = {(f, r) for _, f, r in ce.SOCIAL_EDGES}
        for spec in ce.contrast_specs(models):
            model, benchmark = lookup[spec["model_id"]], lookup[spec["benchmark_id"]]
            if spec["family"] == "incremental_social":
                self.assertEqual(model["estimator"], benchmark["estimator"])
                self.assertEqual(model["fit_days"], benchmark["fit_days"])
                self.assertIn((model["feature_set"], benchmark["feature_set"]), allowed)
                self.assertTrue(benchmark["feature_set"].startswith("characteristics"))
            elif spec["family"] == "conditioning":
                self.assertEqual(model["feature_set"], "characteristics_" + benchmark["feature_set"])
            elif spec["family"] == "estimator":
                self.assertEqual(benchmark["estimator"], "ols")
                self.assertEqual(model["feature_set"], benchmark["feature_set"])
                self.assertEqual(model["fit_days"], benchmark["fit_days"])
            else:
                self.assertEqual(benchmark["fit_days"], 504)
                self.assertEqual(model["estimator"], benchmark["estimator"])
                self.assertEqual(model["feature_set"], benchmark["feature_set"])
        self.assertFalse(any(s["family"] == "feature" for s in ce.contrast_specs(models)))

    def test_partial_scope_and_missing_statistics_keep_full_adjustment(self):
        models = [m for m in registered_models((504,)) if m["target"] == "raw"
                  and m["estimator"] == "ols"
                  and m["feature_set"] in ("characteristics", "characteristics_core")]
        dates = pd.bdate_range("2014-01-02", periods=12)
        rng = np.random.default_rng(27)
        delta = rng.normal(.002, .012, len(dates))
        rows = []
        for model in models:
            values = delta if model["feature_set"] == "characteristics_core" else np.zeros(len(dates))
            rows.append(pd.DataFrame(dict(date=dates, model_id=model["model_id"],
                                          rank_ic=values, ew_spread_bp=values * 100,
                                          cap_spread_bp=np.nan)))
        got = ce.paired_contrasts(pd.concat(rows), models)
        self.assertEqual(set(got.family_size), {144})
        self.assertEqual(set(got.observed_family_size), {1})
        observed = got[got.metric == "rank_ic"]
        np.testing.assert_allclose(observed.p_bonferroni, np.minimum(1, observed.p * 144))
        expected = kernel.calendar_hac_mean(pd.Series(delta, index=dates), lags=5)
        row = observed[observed.hac_lags == 5].iloc[0]
        for name in ("mean", "se", "t", "p", "ci_low", "ci_high"):
            self.assertAlmostEqual(row[name], expected[name])
        self.assertTrue(got[got.metric == "cap_spread_bp"].p_bonferroni.isna().all())

    def test_production_rejects_selective_model_or_key_deletion(self):
        models = registered_models((504,))
        self.assertEqual(ce.validate_matrix(models), [504])
        with self.assertRaisesRegex(ValueError, "88 models"):
            ce.validate_matrix(models[:-1])
        models[0]["rows"] -= 1
        with self.assertRaisesRegex(ValueError, "unchanged"):
            ce.validate_matrix(models)

    def test_adapter_restores_original_kernel_after_failure(self):
        original = kernel.paired_contrasts
        with self.assertRaises(RuntimeError):
            with ce.characteristic_contrasts():
                self.assertIs(kernel.paired_contrasts, ce.paired_contrasts)
                raise RuntimeError("synthetic failure")
        self.assertIs(kernel.paired_contrasts, original)


class CharacteristicArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / ".runs"
        root.mkdir(exist_ok=True)
        cls.directory = (root / ("characteristic_eval_test_" + uuid.uuid4().hex)).resolve()
        if cls.directory.parent != root.resolve():
            raise AssertionError("Test workspace escaped intended directory")
        cls.directory.mkdir()
        cls.prepared = cls.directory / "prepared"
        cls.prepared.mkdir()
        dates, n = pd.bdate_range("2014-01-02", periods=7), 120
        keys = pd.DataFrame(dict(date=np.repeat(dates, n), permno=np.tile(np.arange(n), len(dates))))
        keys.index = pd.Index(np.arange(len(keys)) + 123, name="mm_index")
        keys.to_pickle(cls.prepared / "keys.pkl")
        returns = np.tile(np.linspace(-.03, .03, n), len(dates))
        returns[0] = np.nan
        np.save(cls.prepared / "y.npy", np.column_stack([returns, returns / 2]))
        np.save(cls.prepared / "calendar.npy", dates.to_numpy(dtype="datetime64[D]"))
        pd.DataFrame(dict(cap=np.ones(len(keys)), log_volume=np.arange(len(keys)) % n),
                     index=keys.index).to_pickle(cls.prepared / "evaluation.pkl")
        (cls.prepared / "manifest.json").write_text(json.dumps(dict(
            targets=["f_cumret1", "ar_dgtw_1"], feature_sets={f: ["dummy"] for f in ce.FEATURES})), encoding="utf-8")
        models = registered_models((504,))
        for j, model in enumerate(models):
            path = cls.directory / (model["model_id"] + ".pkl")
            prediction = np.tile(np.arange(n), len(dates)) + np.sin(np.arange(len(keys)) + j) * 5
            keys.assign(prediction=prediction).to_pickle(path)
            model.update(predictions=str(path), sha256=ce.file_hash(path), rows=len(keys), months=1)
        cls.registry = cls.directory / "registry.json"
        cls.registry.write_text(json.dumps(dict(prepared=str(cls.prepared), config={"run_kind": "pilot"},
                                                models=models)), encoding="utf-8")
        cls.prefix = cls.directory / "evaluation"
        cls.out = cls.directory / "report.md"
        cls.arguments = ["--registry", str(cls.registry), "--prepared", str(cls.prepared),
                         "--out", str(cls.prefix), "--start", "2014-01-01", "--end", "2014-01-31",
                         "--skip-subgroups", "--skip-horizons", "--threads", "1", "--allow-pilot"]
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = ce.main(cls.arguments)
            cr.build_report(cls.registry, cls.prefix, cls.out, allow_pilot=True)

    @classmethod
    def tearDownClass(cls):
        expected = (Path(__file__).resolve().parents[1] / ".runs").resolve()
        if cls.directory.parent != expected:
            raise AssertionError("Refusing cleanup outside the intended workspace")
        shutil.rmtree(cls.directory)

    def test_artifacts_publish_exact_deterministic_gzip_and_provenance(self):
        raw = self.prefix.with_name(self.prefix.name + "_daily.csv")
        compressed = raw.with_suffix(".csv.gz")
        self.assertEqual(gzip.decompress(compressed.read_bytes()), raw.read_bytes())
        self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
        self.assertEqual(Path(self.metadata["output_files"]["_daily"]["path"]), compressed)
        self.assertIn(compressed.name, self.metadata["output_sha256"])
        self.assertNotIn(raw.name, self.metadata["output_sha256"])
        self.assertEqual(self.metadata["planned_models"], 264)
        self.assertEqual(self.metadata["included_models"], 88)
        self.assertEqual(self.metadata["reused_kernel_sha256"]["protocol_evaluate.py"], ce.file_hash(kernel.__file__))
        for coverage in self.metadata["coverage"].values():
            self.assertEqual(coverage["common_prediction_stock_days"], 840)
            self.assertEqual(coverage["common_scored_stock_days"], 839)

    def test_full_window_partial_evaluation_has_full_family_sizes(self):
        contrasts = pd.read_csv(self.prefix.with_name(self.prefix.name + "_contrasts.csv"))
        self.assertEqual(set(contrasts.family), {"incremental_social", "conditioning", "estimator"})
        for family, rows in contrasts.groupby("family"):
            self.assertEqual(set(rows.family_size), {ce.PLANNED_FAMILY_COUNTS[family]})
            self.assertEqual(set(rows.observed_family_size), {ce.PLANNED_FAMILY_COUNTS[family] // 3})
            np.testing.assert_allclose(rows.p_bonferroni, np.minimum(1, rows.p * rows.family_size), atol=1e-13)

    def test_report_labels_all_features_and_limits_pilot_claims(self):
        text = self.out.read_text(encoding="utf-8")
        for feature, label in cr.FEATURE_LABELS.items():
            self.assertIn(feature, text)
            self.assertIn(label, text)
        self.assertIn("PLUMBING PILOT", text)
        self.assertIn("publication timing", text)
        self.assertIn("_daily.csv.gz", text)
        self.assertNotIn("Positive at all three HAC lags", text)
        with self.assertRaisesRegex(ValueError, "Pilot reports"):
            cr.build_report(self.registry, self.prefix, self.out)

    def test_report_refuses_changed_evaluation_artifacts(self):
        summary = self.prefix.with_suffix(".csv")
        content = summary.read_bytes()
        try:
            summary.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                cr.build_report(self.registry, self.prefix, self.out, allow_pilot=True)
        finally:
            summary.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
