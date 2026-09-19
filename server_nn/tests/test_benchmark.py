"""Benchmark scheduling and resource-accounting checks; no real data runs."""
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

from server_nn import benchmark
from server_nn.common import write_json


class BenchmarkTests(unittest.TestCase):
    def test_default_profiles_use_full_budget_anchor_months(self):
        core = benchmark.benchmark_jobs("core_textcore")
        largest = benchmark.benchmark_jobs("largest")
        self.assertEqual(len(core), 6)
        self.assertEqual(len(largest), 3)
        self.assertEqual({month for _, month in core}, set(benchmark.ANCHOR_MONTHS))
        self.assertEqual({(m["fit_days"], m["feature_set"]) for m, _ in largest}, {(756, "textall")})

    def test_resource_sweep_schedules_two_waves_with_unique_jobs(self):
        for profile in ("core_textcore", "largest"):
            jobs = benchmark.benchmark_jobs(profile, minimum_jobs=64)
            self.assertEqual(len(jobs), 64)
            self.assertEqual(len({(m["model_id"], month) for m, month in jobs}), 64)
            self.assertTrue(set(benchmark.ANCHOR_MONTHS).issubset({month for _, month in jobs}))
        with self.assertRaises(ValueError):
            benchmark.benchmark_jobs("largest", minimum_jobs=109)

    def test_sampler_counts_only_own_processes_and_active_jobs(self):
        scratch_root = Path(__file__).resolve().parents[2] / ".runs"
        scratch = scratch_root / ("server_benchmark_test_" + uuid.uuid4().hex)
        scratch.mkdir(parents=True)
        child = SimpleNamespace(pid=44, memory_info=lambda: SimpleNamespace(rss=200, private=80))
        parent = SimpleNamespace(pid=33, memory_info=lambda: SimpleNamespace(rss=100, private=50),
                                 children=lambda recursive: [child])
        try:
            write_json({"pid": 44}, scratch / "models" / "a" / "sig" / "months" / "2014-01.running.json")
            write_json({"pid": 999}, scratch / "models" / "a" / "sig" / "months" / "2014-02.running.json")
            with patch.object(benchmark.psutil, "Process", return_value=parent):
                sampler = benchmark.ResourceSampler(scratch)
                sampler.sample()
            summary = sampler.summary()
            self.assertEqual(summary["peak_rss_bytes_sum_parent_and_children"], 300)
            self.assertEqual(summary["peak_private_bytes_sum_parent_and_children"], 130)
            self.assertEqual(summary["peak_active_month_jobs"], 1)
            self.assertEqual(summary["peak_process_count"], 2)
        finally:
            self.assertEqual(scratch.resolve().parent, scratch_root.resolve())
            shutil.rmtree(scratch)

    def test_sampler_does_not_invent_private_memory_on_linux(self):
        parent = SimpleNamespace(pid=33, memory_info=lambda: SimpleNamespace(rss=100), children=lambda recursive: [])
        with patch.object(benchmark.psutil, "Process", return_value=parent):
            sampler = benchmark.ResourceSampler(Path("unused"))
            sampler.sample()
        self.assertIsNone(sampler.summary()["peak_private_bytes_sum_parent_and_children"])


if __name__ == "__main__":
    unittest.main()
