"""Termination detection and notification receipts must not misreport success."""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import Mock, patch
import uuid

import psutil

import watch_study as watch


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1]/".runs"
        self.root = self.base/f"watch_study_test_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

        def cleanup():
            assert self.root.resolve().parent == self.base.resolve()
            assert self.root.name.startswith("watch_study_test_")
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        self.status_path = self.root/"status.json"
        self.report = self.root/"notification_report.md"
        self.created = datetime(2026, 9, 21, 18, 0, tzinfo=timezone.utc).timestamp()
        self.status = dict(status="running", fingerprint="study-fingerprint", pid=123,
                           started_at="2026-09-21T18:00:01+00:00", phase="train_linear")
        self.save()

    def save(self):
        watch.atomic_json(self.status_path, self.status)

    def watcher(self, **kwargs):
        kwargs.setdefault("probe", Mock(return_value=dict(state="alive", create_time=self.created)))
        kwargs.setdefault("notify", Mock(return_value={"accepted": True, "method": "fixture"}))
        kwargs.setdefault("sleeper", Mock())
        return watch.Watcher(self.status_path, self.report, **kwargs)

    def complete(self):
        report, registry = self.root/"results.md", self.root/"registry.json"
        report.write_text("Study results", encoding="utf-8")
        registry.write_text('{"models": []}', encoding="utf-8")
        self.status.update(status="complete", phase="complete", finished_at="2026-09-21T19:00:00+00:00",
                           report=str(report), registry=str(registry),
                           artifact_sha256={str(p): watch.file_hash(p) for p in (report, registry)})
        self.save()

    def test_complete_requires_both_certified_artifacts(self):
        self.complete()
        watcher = self.watcher()
        self.assertEqual(watcher.observe()["outcome"], "complete")
        Path(self.status["report"]).write_text("changed")
        terminal = watcher.observe()
        self.assertEqual(terminal["outcome"], "failed")
        self.assertIn("completion certificate", terminal["reason"])
        Path(self.status["registry"]).unlink()
        self.assertEqual(watcher.observe()["outcome"], "failed")

    def test_reported_failure_retains_status_and_active_log(self):
        log = self.root/"run.log"
        log.write_text("Fitting\nTraceback: deliberate fixture failure\n")
        self.status.update(status="failed", error="ValueError: invalid checkpoint", log=str(log))
        self.save()
        watcher = self.watcher(no_notify=True)
        terminal = watcher.observe()
        self.assertEqual(terminal["outcome"], "failed")
        receipt = watcher.deliver(terminal)
        self.assertEqual(receipt["delivery_status"], "suppressed")
        self.assertIn("invalid checkpoint", self.report.read_text())
        self.assertIn("deliberate fixture failure", self.report.read_text())
        self.assertEqual(watch.read_json(self.status_path), self.status)
        self.assertTrue((self.root/"watcher_events"/f"{terminal['event_id']}.json").exists())

    def test_two_dead_observations_and_access_denied_is_unknown(self):
        probe = Mock(side_effect=[dict(state="dead"), dict(state="unknown"),
                                  dict(state="dead"), dict(state="dead"), dict(state="dead")])
        watcher = self.watcher(probe=probe)
        self.assertIsNone(watcher.observe())
        self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.dead_observations, 0)
        self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.observe()["outcome"], "interrupted")
        with patch.object(watch.psutil, "Process", side_effect=psutil.AccessDenied(123)):
            self.assertEqual(watch.process_info(123)["state"], "unknown")

    def test_reread_resolves_exit_and_completion_race(self):
        def probe(_):
            if watcher.dead_observations == 1:
                self.complete()
            return dict(state="dead")
        watcher = self.watcher(probe=probe)
        self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.observe()["outcome"], "complete")

    def test_pid_reuse_and_replacement_attempt_are_not_success(self):
        watcher = self.watcher()
        self.assertIsNone(watcher.observe())
        watcher.probe.return_value = dict(state="alive", create_time=self.created+10)
        self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.observe()["outcome"], "interrupted")
        self.status["started_at"] = "2026-09-21T18:10:00+00:00"
        self.save()
        self.assertIn("different run attempt", watcher.observe()["reason"])

    def test_read_errors_retry_and_report_monitoring_failure(self):
        watcher = self.watcher()
        self.assertIsNone(watcher.observe())
        self.status_path.write_text("{")
        for _ in range(9):
            self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.observe()["outcome"], "monitoring_error")
        self.save()
        self.assertIsNone(watcher.observe())
        self.assertEqual(watcher.read_errors, 0)

    def test_delivered_receipt_is_idempotent_across_restart(self):
        self.complete()
        watcher = self.watcher()
        terminal = watcher.observe()
        first = watcher.deliver(terminal)
        self.assertEqual(first["delivery_status"], "delivered")
        watcher.notify.assert_called_once()
        replacement = self.watcher()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(replacement.run(30), 0)
        replacement.notify.assert_not_called()
        self.assertEqual(watch.read_json(watcher.receipt_path), first)

    def test_new_attempt_does_not_replay_previous_terminal(self):
        self.complete()
        watcher = self.watcher()
        first_event = watcher.observe()["event_id"]
        watcher.deliver(watcher.observe())
        self.status.update(status="running", started_at="2026-09-21T20:00:01+00:00", pid=456)
        self.status.pop("finished_at")
        self.save()
        replacement = self.watcher(probe=Mock(return_value=dict(state="alive", create_time=self.created+7200)))
        self.assertIsNone(replacement.saved_terminal)
        self.assertIsNone(replacement.observe())
        self.status.update(status="failed", error="new attempt failure")
        self.save()
        self.assertNotEqual(replacement.observe()["event_id"], first_event)

    def test_delivery_retries_and_preserves_failure_evidence(self):
        self.complete()
        watcher = self.watcher(notify=Mock(side_effect=RuntimeError("desktop unavailable")))
        receipt = watcher.deliver(watcher.observe())
        self.assertEqual(receipt["delivery_status"], "failed")
        self.assertEqual(len(receipt["attempts"]), 5)
        self.assertEqual([x.args[0] for x in watcher.sleeper.call_args_list], [5, 15, 30, 60])
        self.assertIn("desktop unavailable", watch.read_json(watcher.receipt_path)["last_error"])
        self.assertIn("**failed**", self.report.read_text())
        watcher.notify = Mock(side_effect=[ValueError("try again"), {"accepted": True}])
        self.assertEqual(watcher.deliver(watcher.observe())["delivery_status"], "delivered")

    def test_transport_requires_acceptance_and_argument_array(self):
        response = subprocess.CompletedProcess([], 0, '{"accepted":false}', "")
        with patch.object(watch.subprocess, "run", return_value=response) as run:
            with self.assertRaisesRegex(ValueError, "accepted=true"):
                watch.notify_windows(self.root/"payload with spaces.json", self.root/"notify.ps1")
            args = run.call_args.args[0]
            self.assertEqual(args[-1], str(self.root/"payload with spaces.json"))
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertGreaterEqual(run.call_args.kwargs["timeout"], 45)

    def test_once_is_read_only(self):
        before = {p.name for p in self.root.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(watch.main(["--status", str(self.status_path), "--report", str(self.report), "--once"]), 0)
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_filelock_prevents_a_duplicate_watcher(self):
        with watch.FileLock(str(self.root/"watcher.lock"), timeout=0):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(watch.main(["--status", str(self.status_path), "--report", str(self.report)]), 3)


if __name__ == "__main__":
    unittest.main()
