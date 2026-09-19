"""Bounded checks for portable coverage validation and architecture inference."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

from server_nn import evaluate as ev
from server_nn.common import (ARCHITECTURES, PREPARED_FILES, TRAINING_VENDOR,
                              data_identity, ensure_vendor, hash_file,
                              phase_models, read_json, training_config, write_json)


def neural_records(phase):
    return [dict(model, target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[model["target"]],
                 validation_days=126, horizon=1, months=108, kind="full",
                 coverage_scope="2014-2022", training_budget="standard")
            for model in phase_models(phase)]


def linear_records(fits=(504,)):
    return [dict(model_id=f"{e}_{f}_{t}_{n}", estimator=e, feature_set=f, target=t,
                 fit_days=n, validation_days=126, horizon=1, months=108,
                 target_column={"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[t])
            for e in ev.LINEAR for f in ev.FEATURES for t in ("raw", "dgtw") for n in fits]


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd()/".runs"/"server_nn_tests"/uuid.uuid4().hex
        self.root.mkdir(parents=True)

    def tearDown(self):
        intended = (Path.cwd()/".runs"/"server_nn_tests").resolve()
        self.assertTrue(self.root.resolve().is_relative_to(intended))
        shutil.rmtree(self.root)


class RegistryTests(WorkspaceCase):
    def test_exact_phase_matrices_and_budgets(self):
        for phase, n in (("nn3", 24), ("depth", 32), ("width", 24)):
            records = neural_records(phase)
            self.assertEqual(len(ev.validate_matrix(records, phase)), n)
            with self.assertRaisesRegex(ValueError, "matrix"):
                ev.validate_matrix(records[:-1], phase)
            changed = copy.deepcopy(records)
            changed[0]["widths"] = [1]
            with self.assertRaisesRegex(ValueError, "widths"):
                ev.validate_matrix(changed, phase)
            changed = copy.deepcopy(records)
            changed[0]["kind"] = "pilot"
            with self.assertRaisesRegex(ValueError, "standard training budget"):
                ev.validate_matrix(changed, phase)
        ev.validate_matrix(linear_records((504, 252, 756)), "nn3", neural=False)
        ev.validate_matrix(linear_records(), "depth", neural=False)

    def test_relative_paths_resolve_after_relocation(self):
        registry = self.root/"new_location"/"registries"/"depth.json"
        write_json({"kind": "full", "path_base": "output_root", "models": [
            {"predictions": "models/predictions.pkl", "run_dir": "models"}]}, registry)
        _, records = ev.load_records(registry, registry.parent.parent)
        self.assertEqual(Path(records[0]["predictions"]), registry.parent.parent/"models"/"predictions.pkl")
        portable = ev.portable_records(records, registry.parent)
        self.assertEqual(portable[0]["predictions"], "../models/predictions.pkl")
        linear = self.root/"bundle"/"linear"/"registry.json"
        write_json({"models": [{"predictions": "ols.pkl"}]}, linear)
        self.assertEqual(ev.load_records(linear)[1][0]["predictions"], str(linear.parent/"ols.pkl"))

    def test_actual_economic_key_coverage_hash_and_finiteness(self):
        dates = pd.date_range("2014-01-01", "2022-12-01", freq="MS")
        expected = pd.DataFrame({"date": dates, "permno": 123}, index=np.arange(500, 608))
        frame = expected.copy()
        frame["prediction"] = np.linspace(-1, 1, len(frame), dtype=np.float32)
        path = self.root/"predictions.pkl"
        frame.iloc[::-1].to_pickle(path)
        record = dict(model_id="example", predictions=str(path), rows=108, sha256=hash_file(path))
        ev.validate_prediction(record, expected)
        wrong_hash = record | {"sha256": "bad"}
        with self.assertRaisesRegex(ValueError, "checksum"):
            ev.validate_prediction(wrong_hash, expected)
        changed = frame.copy()
        changed.iloc[0, changed.columns.get_loc("permno")] = 999
        changed.to_pickle(path)
        with self.assertRaisesRegex(ValueError, "economic keys"):
            ev.validate_prediction(record | {"sha256": hash_file(path)}, expected)
        changed = frame.copy()
        changed.iloc[0, changed.columns.get_loc("prediction")] = np.nan
        changed.to_pickle(path)
        with self.assertRaisesRegex(ValueError, "finite predictions"):
            ev.validate_prediction(record | {"sha256": hash_file(path)}, expected)

    def test_completion_marker_must_match_prediction(self):
        expected = pd.DataFrame({"date": [pd.Timestamp("2014-01-02")], "permno": [7]})
        frame = expected.assign(prediction=.2)
        path = self.root/"predictions.pkl"
        frame.to_pickle(path)
        record = dict(model_id="nn", predictions=str(path), rows=1, months=108,
                      sha256=hash_file(path), run_dir=str(self.root), run_fingerprint="abc")
        write_json(record, self.root/"complete.json")
        ev.validate_prediction(record, expected, neural=True)
        write_json(record | {"run_fingerprint": "other"}, self.root/"complete.json")
        with self.assertRaisesRegex(ValueError, "completion marker"):
            ev.validate_prediction(record, expected, neural=True)

    def test_actual_nn_spec_budget_and_fingerprint_are_verified(self):
        model = phase_models("depth")[0]
        spec = {"model": model, "config": training_config(model["widths"], 2), "data_identity": {"a": "b"},
                "tasks": [{"month": month} for month in ev.MONTHS]}
        record = dict(model, run_dir=str(self.root))
        for nonstandard in (False, True):
            if nonstandard:
                spec["config"]["seeds"] = [7]
            write_json(spec, self.root/"run_spec.json")
            record["run_fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if nonstandard:
                with self.assertRaisesRegex(ValueError, "standard training budget"):
                    ev.validate_nn_spec(record, {"a": "b"})
            else:
                ev.validate_nn_spec(record, {"a": "b"})
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    ev.validate_nn_spec(record | {"run_fingerprint": "wrong"}, {"a": "b"})


class ContrastTests(unittest.TestCase):
    def test_registered_families_include_all_new_comparisons(self):
        ensure_vendor()
        from protocol_evaluate import contrast_specs
        expected = {"nn3": {"feature": 60, "estimator": 84, "history": 40},
                    "depth": {"feature": 32, "estimator": 76, "architecture": 12},
                    "width": {"feature": 28, "estimator": 60, "architecture": 8}}
        for phase, families in expected.items():
            models = neural_records(phase)+linear_records((504, 252, 756) if phase == "nn3" else (504,))
            models = [x for x in models if x["target"] == "raw"]
            specs = ev.extended_contrasts(models, phase, contrast_specs)
            self.assertEqual(dict(ev.Counter(s["family"] for s in specs)), families)
            self.assertEqual(len(specs), len({(s["family"], s["model_id"], s["benchmark_id"]) for s in specs}))

    def test_adaptive_has_separate_family_and_fixed_depth_counts(self):
        ensure_vendor()
        from protocol_evaluate import contrast_specs
        models = [x for x in neural_records("depth")+linear_records() if x["target"] == "raw"]
        models += [dict(x, model_id=x["model_id"].replace("nn3", "nn_adaptive"), estimator="nn_adaptive")
                   for x in models if x["estimator"] == "nn3"]
        specs = ev.extended_contrasts(models, "depth", contrast_specs, adaptive=True)
        self.assertEqual(dict(ev.Counter(s["family"] for s in specs)),
                         {"feature": 36, "estimator": 92, "architecture": 12, "adaptive": 4})

    def test_hac_bonferroni_uses_entire_architecture_family(self):
        ensure_vendor()
        import protocol_evaluate as evaluator
        models = [x for x in neural_records("depth") if x["target"] == "raw"]
        dates = pd.bdate_range("2020-01-01", periods=20)
        daily = pd.DataFrame([dict(date=d, model_id=m["model_id"], rank_ic=.01*j+i*.001,
                                  ew_spread_bp=.1*j+i*.002, cap_spread_bp=.2*j+i*.003)
                              for i, d in enumerate(dates) for j, m in enumerate(models)])
        original = evaluator.contrast_specs
        with patch.object(evaluator, "contrast_specs", lambda records: ev.extended_contrasts(records, "depth", original)):
            result = evaluator.paired_contrasts(daily, models)
        rows = result[result.family == "architecture"]
        self.assertEqual(len(rows), 12*3*3)
        self.assertTrue((rows.family_size == 12).all())
        np.testing.assert_allclose(rows.p_bonferroni, np.minimum(1, rows.p*12), equal_nan=True)


class AdaptiveTests(WorkspaceCase):
    def test_selection_ignores_test_information_and_breaks_exact_ties(self):
        candidates = [dict(estimator=e, validation_ic=.1, parameter_count=100+j, test_ic=10-j)
                      for j, e in enumerate(ev.DEPTH)]
        self.assertEqual(ev.choose_architecture(candidates)["estimator"], "nn1")
        candidates[3]["validation_ic"] = .100000000000001
        self.assertEqual(ev.choose_architecture(candidates)["estimator"], "nn4")
        candidates[3]["validation_ic"] = .1
        candidates[3]["parameter_count"] = 99
        self.assertEqual(ev.choose_architecture(candidates)["estimator"], "nn4")
        candidates[0]["parameter_count"] = 99
        self.assertEqual(ev.choose_architecture(candidates)["estimator"], "nn1")
        with self.assertRaisesRegex(ValueError, "all four"):
            ev.choose_architecture(candidates[:3])
        candidates[0]["validation_ic"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite validation"):
            ev.choose_architecture(candidates)

    def test_parameter_count_includes_batchnorm_affine_only(self):
        self.assertEqual(ev.parameter_count(2, [128]), 2*128+128+2*128+128+1)
        self.assertGreater(ev.parameter_count(388, [128, 64, 32]), ev.parameter_count(2, [128, 64, 32]))

    def test_adaptive_predictions_follow_monthly_validation_choices(self):
        ensure_vendor()
        months = ("2014-01", "2014-02")
        prepared, output = self.root/"prepared", self.root/"output"
        prepared.mkdir()
        keys = pd.DataFrame({"date": pd.to_datetime(["2014-01-02", "2014-01-02", "2014-02-03", "2014-02-03"]),
                             "permno": [1, 2, 1, 2]}, index=[100, 102, 107, 109])
        keys.to_pickle(prepared/"keys.pkl")
        write_json({"feature_sets": {"core": [0, 1]}}, prepared/"manifest.json")
        records = []
        for target in ("raw", "dgtw"):
            for i, estimator in enumerate(ev.DEPTH):
                run = output/target/estimator
                (run/"months").mkdir(parents=True)
                tasks = [dict(month=month, fit_days=504) for month in months]
                write_json({"tasks": tasks}, run/"run_spec.json")
                frame = keys.copy()
                frame["prediction"] = np.array([i, i+.1, 10+i, 10+i+.1], dtype=np.float32)
                frame.to_pickle(run/"predictions.pkl")
                record = dict(estimator=estimator, feature_set="core", target=target, fit_days=504,
                    model_id=f"{estimator}_{target}", widths=ARCHITECTURES[estimator], run_dir=str(run),
                    run_fingerprint=estimator, predictions=str(run/"predictions.pkl"))
                for j, task in enumerate(tasks):
                    rows = np.array([2*j, 2*j+1])
                    # January chooses NN4; February chooses NN2, despite reversed test scores.
                    score = 1. if (j == 0 and estimator == "nn4") or (j == 1 and estimator == "nn2") else 0.
                    result = dict(test_indices=rows, prediction=frame.prediction.to_numpy()[rows],
                                  validation_ic=score, test_ic=-score, chosen_penalty=.001, month=task["month"])
                    pd.to_pickle(dict(fingerprint=estimator, task=task, result=result), run/"months"/f"{task['month']}.pkl")
                records.append(record)
        with patch.object(ev, "MONTHS", months), patch.object(ev, "FEATURES", ("core",)):
            generated, ledger_path = ev.build_adaptive(records, prepared, output)
        self.assertEqual(len(generated), 2)
        for record in generated:
            np.testing.assert_array_equal(pd.read_pickle(record["predictions"]).prediction.to_numpy(),
                                          np.array([3, 3.1, 11, 11.1], dtype=np.float32))
        selected = pd.read_csv(ledger_path).query("selected")
        self.assertEqual(selected.estimator.tolist(), ["nn4", "nn2", "nn4", "nn2"])
        self.assertNotIn("test_ic", selected.columns)
        self.assertTrue(selected.checkpoint_sha256.str.len().eq(64).all())


class EndToEndTests(WorkspaceCase):
    def test_complete_synthetic_phase_generates_reports_and_provenance(self):
        vendor = ensure_vendor()
        bundle, output = self.root/"bundle", self.root/"output"
        prepared = bundle/"prepared"
        prepared.mkdir(parents=True)
        dates = pd.date_range("2014-01-01", "2022-12-01", freq="MS")
        keys = pd.DataFrame({"date": np.repeat(dates, 10), "permno": np.tile(np.arange(10), 108)})
        keys.to_pickle(prepared/"keys.pkl")
        pd.DataFrame({"cap": np.ones(len(keys)), "log_volume": np.tile(np.arange(10), 108)}).to_pickle(prepared/"evaluation.pkl")
        for name, array in {"X": np.zeros((len(keys), 2), dtype=np.float32),
                            "y": np.column_stack([np.tile(np.arange(10), 108)]*2)*.001,
                            "q": np.zeros((len(keys), 2)),
                            "codes": np.repeat(np.arange(108), 10),
                            "calendar": dates.to_numpy(dtype="datetime64[D]")}.items():
            np.save(prepared/f"{name}.npy", array)
        write_json({"targets": ["f_cumret1", "ar_dgtw_1"], "fingerprint": "synthetic",
                    "feature_sets": {feature: list(range(n)) for feature, n in
                                     (("core", 2), ("all", 53), ("textcore", 388), ("textall", 439))}},
                   prepared/"manifest.json")
        write_json({}, prepared/"horizon_repair_audit.json")
        files = {"prepared/"+name: {"bytes": (prepared/name).stat().st_size,
                                   "sha256": hash_file(prepared/name)} for name in PREPARED_FILES}
        write_json({"schema": "stocktwits_server_bundle_v1", "prepared_fingerprint": "synthetic", "files": files,
                    "vendor_sha256": {name: hash_file(vendor/name) for name in TRAINING_VENDOR}},
                   bundle/"bundle_manifest.json")
        linear, neural = linear_records(), neural_records("depth")
        for i, record in enumerate(linear+neural):
            is_nn = record["estimator"].startswith("nn")
            folder = output/"models"/record["model_id"] if is_nn else bundle/"linear"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f"{record['model_id']}.pkl"
            frame = keys.assign(prediction=np.tile(np.arange(10), 108).astype(np.float32)+i*.0001)
            frame.to_pickle(path)
            record.update(rows=len(keys), predictions=ev.relative_path(path, output if is_nn else bundle/"linear"),
                          sha256=hash_file(path))
            if is_nn:
                spec = {"model": record.copy(), "config": training_config(record["widths"], 2),
                        "data_identity": data_identity(bundle), "tasks": [{"month": month} for month in ev.MONTHS]}
                write_json(spec, folder/"run_spec.json")
                signature = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
                record.update(run_dir=ev.relative_path(folder, output), run_fingerprint=signature)
                write_json(record, folder/"complete.json")
        write_json({"kind": "full", "models": linear}, bundle/"linear"/"registry.json")
        write_json({"kind": "full", "models": neural, "path_base": "output_root", "data_identity": data_identity(bundle)},
                   output/"registries"/"depth.json")
        destination = ev.main(["--bundle", str(bundle), "--output", str(output), "--phase", "depth"])
        complete = read_json(destination/"complete.json")
        self.assertEqual(complete["models"], 64)
        self.assertEqual(hash_file(destination/"results.md"), complete["results_sha256"])
        metadata = read_json(destination/"evaluation.json")
        self.assertEqual(metadata["registered_family_sizes"]["raw"]["architecture"], 12)
        self.assertIn("server_nn/evaluate.py", metadata["code_sha256"])
        self.assertIn("Fixed architectures versus NN3", (destination/"results.md").read_text(encoding="utf-8"))
        self.assertTrue((destination/"detailed_results.md").is_file())
        self.assertFalse(Path(read_json(destination/"registry.json")["models"][0]["predictions"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
