import argparse
import unittest

from server_nn.check_environment import (GIB, effective_cpu_count, memory_budget,
                                         positive_float, slurm_memory_bytes)


class EnvironmentCheckTests(unittest.TestCase):
    def test_affinity_and_scheduler_bound_cpu_count(self):
        cpus, _ = effective_cpu_count(128, list(range(64)), {"SLURM_CPUS_PER_TASK": "48"})
        self.assertEqual(cpus, 48)
        self.assertEqual(effective_cpu_count(128, [1, 3], {})[0], 2)

    def test_slurm_memory_uses_mib_by_default(self):
        self.assertEqual(slurm_memory_bytes("184320"), 180 * GIB)
        self.assertEqual(slurm_memory_bytes("180G"), 180 * GIB)
        self.assertIsNone(slurm_memory_bytes("0"))
        self.assertIsNone(slurm_memory_bytes("invalid"))

    def test_memory_honors_cgroup_remaining(self):
        usable, _, _ = memory_budget(200 * GIB, 195 * GIB, 48,
            {"SLURM_MEM_PER_NODE": "180G"},
            [{"source": "cgroup", "limit_bytes": 175 * GIB, "remaining_bytes": 160 * GIB}])
        self.assertEqual(usable, 160 * GIB)

    def test_memory_per_cpu_allocation(self):
        usable, _, _ = memory_budget(200 * GIB, 195 * GIB, 48,
                                     {"SLURM_MEM_PER_CPU": "2G"})
        self.assertEqual(usable, 96 * GIB)

    def test_memory_rejects_nan_budget(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("nan")


if __name__ == "__main__":
    unittest.main()
