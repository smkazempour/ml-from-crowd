"""Watch one existing study attempt and notify on completion or interruption.

This process never modifies the study status, artifacts, or controller. It writes
its own evidence beside status.json and a human-readable report at --report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock, Timeout
import psutil


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "failed", "interrupted", "monitoring_error"}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def read_status(path):
    value = read_json(path)
    if (value.get("status") not in {"running", "complete", "failed"}
            or not isinstance(value.get("fingerprint"), str) or not value["fingerprint"]
            or not isinstance(value.get("started_at"), str)
            or not isinstance(value.get("pid"), int) or value["pid"] <= 0):
        raise ValueError("Missing or invalid study identity/status fields")
    datetime.fromisoformat(value["started_at"])
    return value


def process_info(pid):
    """AccessDenied is unknown, never evidence that a process has died."""
    try:
        process = psutil.Process(pid)
        created = process.create_time()
        if not process.is_running() or process.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
            return {"state": "dead", "create_time": created}
        return {"state": "alive", "create_time": created}
    except psutil.NoSuchProcess:
        return {"state": "dead", "create_time": None}
    except psutil.AccessDenied as error:
        return {"state": "unknown", "create_time": None, "error": str(error)}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def certified_completion(status):
    """Verify the small final artifacts, trusting the controller's large-data checks."""
    hashes = status.get("artifact_sha256")
    if not status.get("finished_at") or not isinstance(hashes, dict) or not hashes:
        raise ValueError("Complete status has no final completion certificate")
    for field in ("report", "registry"):
        location = status.get(field)
        if not location:
            raise ValueError(f"Complete status does not declare its {field}")
        path = Path(location)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Completed {field} is missing or empty: {path}")
        if hashes.get(str(path)) != file_hash(path):
            raise ValueError(f"Completed {field} does not match its completion certificate: {path}")


def log_tail(status, limit=12000):
    if not status or not status.get("log"):
        return "No active log was declared."
    try:
        with Path(status["log"]).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            value = stream.read().decode("utf-8", errors="replace")
        return value or "The active log is empty."
    except OSError as error:
        return f"Could not read the active log: {type(error).__name__}: {error}"


def notify_windows(payload_path, script):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
         "-File", str(script), "-PayloadPath", str(payload_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Notification exit {result.returncode}: {(result.stderr or result.stdout).strip()}")
    value = json.loads(result.stdout.lstrip("\ufeff").strip())
    if not isinstance(value, dict) or value.get("accepted") is not True:
        raise ValueError("Notification transport did not confirm accepted=true")
    return value


class Watcher:
    def __init__(self, status_path, report_path, label="Study", *, no_notify=False,
                 probe=process_info, notify=notify_windows, sleeper=time.sleep,
                 notify_script=ROOT/"tools/notify_study.ps1"):
        self.status_path = Path(status_path).resolve()
        self.report_path = Path(report_path).resolve()
        self.directory = self.status_path.parent
        self.state_path = self.directory/"watcher_status.json"
        self.payload_path = self.directory/"notification.json"
        self.receipt_path = self.directory/"notification_receipt.json"
        self.label, self.no_notify = label, no_notify
        self.probe, self.notify, self.sleeper = probe, notify, sleeper
        self.notify_script = Path(notify_script).resolve()
        self.identity = None
        self.last_status = None
        self.read_errors = 0
        self.dead_observations = 0
        self.last_error = None
        self.health = None
        self.started_at = timestamp()
        self.saved_terminal = None
        # A restarted watcher remains pinned to the same attempt/process identity.
        if self.state_path.exists():
            try:
                previous = read_json(self.state_path)
                candidate = previous.get("monitored_identity")
                current = read_status(self.status_path)
                if candidate and all(candidate.get(key) == value for key, value in self.attempt_identity(current).items()):
                    self.identity = candidate
                    self.last_status = previous.get("controller_status")
                    self.saved_terminal = previous.get("terminal")
            except (OSError, ValueError):
                pass

    @staticmethod
    def attempt_identity(status):
        return {key: status[key] for key in ("fingerprint", "started_at", "pid")}

    def terminal(self, outcome, reason, status=None):
        status = status if status is not None else self.last_status
        event = {
            "fingerprint": (self.identity or {}).get("fingerprint"),
            "started_at": (self.identity or {}).get("started_at"),
            "finished_at": (status or {}).get("finished_at"),
            "outcome": outcome,
        }
        event_id = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        return dict(event, event_id=event_id, detected_at=timestamp(), reason=reason,
                    controller_status=status, active_log_tail=log_tail(status) if outcome != "complete" else None)

    def inspect(self, status):
        identity = self.attempt_identity(status)
        if self.identity is None:
            self.identity = dict(identity, create_time=None)
        elif any(self.identity.get(key) != value for key, value in identity.items()):
            return self.terminal("interrupted", "The controller status was replaced by a different run attempt.")
        self.last_status = status
        if status["status"] == "complete":
            try:
                certified_completion(status)
            except (OSError, ValueError) as error:
                return self.terminal("failed", f"Completion verification failed: {error}", status)
            return self.terminal("complete", "The study completed and its report and registry match the completion certificate.", status)
        if status["status"] == "failed":
            return self.terminal("failed", status.get("error") or "The controller reported a failure.", status)
        self.health = self.probe(status["pid"])
        created = self.health.get("create_time")
        expected = self.identity.get("create_time")
        if self.health["state"] == "alive" and created is not None:
            started = datetime.fromisoformat(status["started_at"]).timestamp()
            if (expected is not None and abs(created-expected) > 0.01) or created > started+2:
                self.health = dict(self.health, state="dead", error="The controller PID has been reused.")
            elif expected is None:
                self.identity["create_time"] = created
        self.dead_observations = self.dead_observations+1 if self.health["state"] == "dead" else 0
        return None

    def observe(self):
        try:
            status = read_status(self.status_path)
            outcome = self.inspect(status)
            self.read_errors = 0
            self.last_error = None
            if outcome is not None:
                return outcome
            if self.dead_observations >= 2:
                # The controller may have committed terminal status just after the first read.
                outcome = self.inspect(read_status(self.status_path))
                if outcome is not None:
                    return outcome
                if self.health["state"] == "dead":
                    return self.terminal("interrupted", "The controller process disappeared without a final completion or failure status.")
            return None
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.read_errors += 1
            self.dead_observations = 0
            self.last_error = f"{type(error).__name__}: {error}"
            if self.read_errors >= 10:
                return self.terminal("monitoring_error", f"The watcher could not read or inspect the study for ten consecutive polls: {self.last_error}")
            return None

    def state(self, terminal=None, receipt=None):
        return dict(status="monitoring" if terminal is None else terminal["outcome"],
                    watcher_pid=os.getpid(), watcher_started_at=self.started_at, updated_at=timestamp(),
                    status_path=str(self.status_path), report_path=str(self.report_path),
                    monitored_identity=self.identity, controller_status=self.last_status,
                    process_health=self.health, consecutive_read_errors=self.read_errors,
                    consecutive_dead_observations=self.dead_observations, last_error=self.last_error,
                    terminal=terminal, notification=receipt)

    def write_report(self, terminal=None, receipt=None):
        outcome = terminal["outcome"] if terminal else "monitoring"
        text = f"# {self.label}: {outcome}\n\nUpdated: {timestamp()} (UTC).\n\n"
        text += f"Controller status: `{self.status_path}`.\n\n"
        if terminal:
            text += terminal["reason"]+"\n\n"
            if outcome == "complete":
                text += f"Results report: `{terminal['controller_status']['report']}`.\n\n"
                text += "The watcher checked the report and registry hashes; the controller certified the other output artifacts.\n\n"
            if receipt:
                text += f"Desktop notification: **{receipt['delivery_status']}**. "
                if receipt["delivery_status"] == "delivered":
                    text += "The notification transport accepted it; this does not confirm it was viewed.\n\n"
                else:
                    text += "The durable outcome is recorded here regardless of desktop delivery.\n\n"
            if outcome != "complete":
                text += "## Last controller status\n\n```json\n"
                text += json.dumps(terminal["controller_status"], indent=2)+"\n```\n\n"
                text += "## Active log tail\n\n```text\n"+str(terminal["active_log_tail"]).replace("```", "'''" )+"\n```\n"
        else:
            text += "The watcher is waiting for successful completion, a reported failure, or an unexpected controller exit.\n\n"
            if self.last_status:
                text += f"Current phase: `{self.last_status.get('phase', 'unknown')}`.\n"
            if self.last_error:
                text += f"\nTemporary monitoring error: {self.last_error}\n"
        atomic_text(self.report_path, text)

    def persist(self, terminal=None, receipt=None):
        atomic_json(self.state_path, self.state(terminal, receipt))
        self.write_report(terminal, receipt)

    def deliver(self, terminal):
        previous = {}
        if self.receipt_path.exists():
            try:
                previous = read_json(self.receipt_path)
            except (OSError, ValueError):
                pass
        if previous.get("event_id") == terminal["event_id"] and previous.get("delivery_status") == "delivered":
            self.persist(terminal, previous)
            return previous
        names = {"complete": "completed successfully", "failed": "failed",
                 "interrupted": "stopped unexpectedly", "monitoring_error": "monitoring needs attention"}
        payload = dict(title=f"{self.label}: {names[terminal['outcome']]}",
                       message=terminal["reason"][:500], report_path=str(self.report_path),
                       tag=terminal["event_id"][:16], event_id=terminal["event_id"],
                       outcome=terminal["outcome"], detected_at=terminal["detected_at"])
        atomic_json(self.payload_path, payload)
        atomic_json(self.directory/"watcher_events"/f"{terminal['event_id']}.json", terminal)
        attempts = previous.get("attempts", []) if previous.get("event_id") == terminal["event_id"] else []
        receipt = dict(event_id=terminal["event_id"], outcome=terminal["outcome"],
                       delivery_status="suppressed" if self.no_notify else "pending", attempts=attempts)
        atomic_json(self.receipt_path, receipt)
        self.persist(terminal, receipt)
        if self.no_notify:
            return receipt
        for delay in (0, 5, 15, 30, 60):
            if delay:
                self.sleeper(delay)
            attempt = dict(attempted_at=timestamp())
            try:
                attempt["result"] = self.notify(self.payload_path, self.notify_script)
                if not isinstance(attempt["result"], dict) or attempt["result"].get("accepted") is not True:
                    raise ValueError("Notification transport did not confirm accepted=true")
                receipt.update(delivery_status="delivered", delivered_at=timestamp())
            except Exception as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
            receipt["attempts"].append(attempt)
            atomic_json(self.receipt_path, receipt)
            self.persist(terminal, receipt)
            if receipt["delivery_status"] == "delivered":
                return receipt
        receipt.update(delivery_status="failed", last_error=receipt["attempts"][-1]["error"])
        atomic_json(self.receipt_path, receipt)
        self.persist(terminal, receipt)
        return receipt

    def run(self, interval):
        # Preserve an already-observed termination across watcher restarts, including
        # interruptions with no controller finished_at and monitoring failures.
        terminal = self.saved_terminal
        while terminal is None:
            terminal = self.observe()
            if terminal is None:
                self.persist()
                self.sleeper(interval)
        receipt = self.deliver(terminal)
        print(json.dumps(dict(outcome=terminal["outcome"], notification=receipt["delivery_status"],
                              report=str(self.report_path))), flush=True)
        if receipt["delivery_status"] == "failed":
            return 2
        return 0 if terminal["outcome"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", default="Study")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="Inspect once without writing or notifying")
    parser.add_argument("--no-notify", action="store_true", help="Record the outcome without desktop delivery")
    parser.add_argument("--notify-script", type=Path, default=ROOT/"tools/notify_study.ps1")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.status.resolve() == args.report.resolve():
        parser.error("--report must not overwrite the controller status")
    watcher = Watcher(args.status, args.report, args.label, no_notify=args.no_notify,
                      notify_script=args.notify_script)
    if args.once:
        terminal = watcher.observe()
        print(json.dumps(watcher.state(terminal), indent=2))
        return 0
    if not args.status.parent.is_dir():
        parser.error("The study status directory does not exist")
    try:
        with FileLock(str(args.status.parent/"watcher.lock"), timeout=0):
            return watcher.run(args.interval)
    except Timeout:
        print("A watcher already owns this study's watcher.lock", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
