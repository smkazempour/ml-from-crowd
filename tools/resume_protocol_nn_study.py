"""Resume the frozen NN study with resilience for Windows status-file contention.

The original controller and its scientific source files stay unchanged. Only its
in-process update_status function is wrapped. Atomic prediction, checkpoint,
registry, and evaluation writes retain their original behavior.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import run_protocol_nn_study as study


RETRY_SECONDS = 2.0
RETRY_INTERVAL_SECONDS = 0.1
WINDOWS_CONTENTION_ERRORS = frozenset((5, 32, 33))


class ResilientStatusWriter:
    """Retry access/sharing failures; never kill a child for a blocked heartbeat.

    Startup, phase transitions without an active child, and terminal writes
    remain strict after retries. Other exception types always propagate.
    """

    def __init__(self, original, *, retry_seconds=RETRY_SECONDS,
                 retry_interval=RETRY_INTERVAL_SECONDS, clock=time.monotonic,
                 sleep=time.sleep, log=None):
        if retry_seconds < 0 or retry_interval <= 0:
            raise ValueError("Retry duration must be nonnegative and interval positive")
        self.original = original
        self.retry_seconds = retry_seconds
        self.retry_interval = retry_interval
        self.clock = clock
        self.sleep = sleep
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.degraded = set()

    def __call__(self, path, state, **changes):
        path = Path(path)
        deadline = self.clock() + self.retry_seconds
        attempts = 0
        while True:
            attempts += 1
            try:
                result = self.original(path, state, **changes)
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in WINDOWS_CONTENTION_ERRORS:
                    raise
                remaining = deadline - self.clock()
                if remaining > 0:
                    if attempts == 1:
                        self.log(f"STATUS_WRITE_RETRY path={path} winerror={exc.winerror}")
                    self.sleep(min(self.retry_interval, remaining))
                    continue
                # The original update_status mutates state before atomic_json.
                # Read the intended values explicitly so this stays safe if a
                # test or future compatible controller updates state later.
                status = changes.get("status", state.get("status"))
                child_pid = changes.get("child_pid", state.get("child_pid"))
                if path.name == "status.json" and status == "running" and child_pid is not None:
                    if path not in self.degraded:
                        self.log(
                            f"STATUS_WRITE_DEGRADED path={path} child_pid={child_pid} "
                            f"winerror={exc.winerror}; heartbeat skipped after bounded retries; "
                            "supervised child continues and the next poll retries"
                        )
                    self.degraded.add(path)
                    return None
                raise
            else:
                if attempts > 1 or path in self.degraded:
                    self.log(f"STATUS_WRITE_RECOVERED path={path} attempts={attempts}")
                self.degraded.discard(path)
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--expected-wrapper-sha256")
    options, controller_args = parser.parse_known_args(argv)
    wrapper_path = Path(__file__).resolve()
    digest = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    if options.expected_wrapper_sha256 is not None:
        if options.expected_wrapper_sha256.lower() != digest:
            parser.error("Recovery wrapper SHA-256 does not match the expected launch digest")
    print(
        f"CONTROL_STATUS_RECOVERY wrapper={wrapper_path} sha256={digest} "
        f"retry_seconds={RETRY_SECONDS} retry_interval_seconds={RETRY_INTERVAL_SECONDS} "
        "scope=active_child_status_writes strict_terminal_status=true",
        flush=True,
    )
    original = study.update_status
    study.update_status = ResilientStatusWriter(original)
    try:
        return study.main(controller_args)
    finally:
        study.update_status = original


if __name__ == "__main__":
    main()
