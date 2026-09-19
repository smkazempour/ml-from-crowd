"""Failure-injection checks for isolated NN controller status recovery."""
from io import StringIO
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, mock_open, patch

import resume_protocol_nn_study as recovery


def contention_error(code=5):
    error = PermissionError("Windows status-file contention")
    error.winerror = code
    return error


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class StatusRecoveryTests(unittest.TestCase):
    def writer(self, original, **kwargs):
        clock = FakeClock()
        log = []
        writer = recovery.ResilientStatusWriter(
            original, retry_seconds=0.2, retry_interval=0.1,
            clock=clock, sleep=clock.sleep, log=log.append, **kwargs
        )
        return writer, clock, log

    def test_transient_windows_contention_retries_and_recovers(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                original = Mock(side_effect=[contention_error(code), None])
                writer, clock, log = self.writer(original)
                writer("status.json", {"status": "running", "child_pid": 123})
                self.assertEqual(original.call_count, 2)
                self.assertAlmostEqual(clock.now, 0.1)
                self.assertIn("STATUS_WRITE_RETRY", log[0])
                self.assertIn("STATUS_WRITE_RECOVERED", log[1])

    def test_persistent_heartbeat_skips_and_later_recovers(self):
        original = Mock(side_effect=contention_error())
        writer, clock, log = self.writer(original)
        state = {"status": "running", "child_pid": 123}
        writer("status.json", state)
        self.assertEqual(original.call_count, 3)
        self.assertAlmostEqual(clock.now, 0.2)
        self.assertIn(Path("status.json"), writer.degraded)
        self.assertIn("STATUS_WRITE_DEGRADED", log[-1])
        original.side_effect = None
        writer("status.json", state)
        self.assertFalse(writer.degraded)
        self.assertIn("STATUS_WRITE_RECOVERED", log[-1])

    def test_terminal_startup_and_nonstatus_writes_remain_strict(self):
        for path, state, changes in (
            ("status.json", {"status": "complete", "child_pid": None}, {}),
            ("status.json", {"status": "running", "child_pid": 123}, {"status": "failed"}),
            ("status.json", {"status": "running", "child_pid": None}, {}),
            ("status.json", {"status": "running", "child_pid": 123}, {"child_pid": None}),
            ("registry.json", {"status": "running", "child_pid": 123}, {}),
        ):
            with self.subTest(path=path, state=state, changes=changes):
                writer, clock, _ = self.writer(Mock(side_effect=contention_error()))
                with self.assertRaises(PermissionError):
                    writer(path, state, **changes)
                self.assertAlmostEqual(clock.now, 0.2)

    def test_unrelated_errors_propagate_without_retry(self):
        for error in (ValueError("malformed state"), OSError("disk failure"),
                      PermissionError("non-Windows permission failure"), contention_error(87)):
            with self.subTest(error=error):
                original = Mock(side_effect=error)
                writer, clock, _ = self.writer(original)
                with self.assertRaises(type(error)):
                    writer("status.json", {"status": "running", "child_pid": 123})
                self.assertEqual(original.call_count, 1)
                self.assertEqual(clock.now, 0.0)

    def test_persistent_heartbeat_cannot_terminate_supervised_child(self):
        blocked = False
        process = Mock(pid=123)

        def wait(timeout):
            nonlocal blocked
            if not blocked:
                blocked = True
                raise subprocess.TimeoutExpired("test-child", timeout)
            blocked = False
            return 0

        process.wait.side_effect = wait

        def atomic_json(state, path):
            if blocked:
                raise contention_error()

        writer, _, log = self.writer(recovery.study.update_status)
        path = Path("test-status-recovery")
        with patch.object(recovery.study, "atomic_json", side_effect=atomic_json), \
             patch.object(recovery.study, "update_status", writer), \
             patch.object(recovery.study.subprocess, "Popen", return_value=process), \
             patch.object(Path, "open", mock_open()):
            state = {"status": "running"}
            recovery.study.run_child(
                ["test-child"], path/"child.log", "train_nn", path/"status.json", state, {}
            )
        process.terminate.assert_not_called()
        process.poll.assert_not_called()
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual(state["last_exit_code"], 0)
        self.assertEqual(state["commands"][0]["exit_code"], 0)
        self.assertTrue(any("STATUS_WRITE_DEGRADED" in line for line in log))
        self.assertTrue(any("STATUS_WRITE_RECOVERED" in line for line in log))

    def test_wrapper_passes_recipe_unchanged_and_restores_status_function(self):
        original_status = recovery.study.update_status
        original_atomic = recovery.study.atomic_json
        args = ["--prepared", "prepared", "--linear-registry", "linear.json", "--workers", "8"]

        def controller_main(received):
            self.assertEqual(received, args)
            self.assertIsInstance(recovery.study.update_status, recovery.ResilientStatusWriter)
            self.assertIs(recovery.study.atomic_json, original_atomic)
            return "done"

        with patch.object(recovery.study, "main", side_effect=controller_main), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(recovery.main(args), "done")
            self.assertIn("CONTROL_STATUS_RECOVERY", output.getvalue())
            self.assertIn("sha256=", output.getvalue())
        self.assertIs(recovery.study.update_status, original_status)
        self.assertIs(recovery.study.atomic_json, original_atomic)

    def test_wrong_wrapper_digest_prevents_controller_launch(self):
        with patch.object(recovery.study, "main") as controller_main, \
             patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit):
                recovery.main(["--expected-wrapper-sha256", "wrong"])
        controller_main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
