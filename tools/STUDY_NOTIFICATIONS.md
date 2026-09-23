# Automatic study notifications

As of September 22, 2026, the latest interaction study is complete. Its Windows
completion alert was accepted at 4:38:47 p.m. Central; see
[the final record](../reports/LINEAR_INTERACTION_COMPLETION.md). No watcher needs
restarting for any completed study. The example below is the historical setup
for the earlier linear-design run; attach a new watcher identity to future runs.

The independent watcher checks an existing controller every 30 seconds and sends
a Windows desktop notification when the study completes, reports an error, or
exits without a final status. It also writes a durable Markdown report. It does
not change, restart or terminate the experiment.

For the earlier linear design study, monitoring started September 21, 2026 at
1:59 p.m. Central. Watcher PID at launch: **37656**. The watcher remains active
after the conversation turn ends and does not require the IDE to stay open.
It runs locally: shutting down Windows also stops the watcher. After a reboot,
run the command below to detect and report an interrupted study.

This study subsequently completed at 5:31:46 p.m. Central. The watcher recorded
Windows acceptance of its completion alert at 5:31:56 and exited. Its final report
and delivery receipt are preserved; no watcher restart is needed for this completed run.

## Historical example and recovery command

From the repository root with the existing `py313` environment activated:

```powershell
python -B tools/watch_study.py --status .runs/linear_design_v2/study/c11478141261534f/status.json --report reports/LINEAR_DESIGN_COMPLETION.md --label "Linear design study" --interval 30
```

The current watcher was launched with a hidden window and redirected logs:

- `.runs/linear_design_v2/watcher_20260921T185943Z.log`
- `.runs/linear_design_v2/watcher_20260921T185943Z.err`

Its latest health record is `watcher_status.json` beside the controller's
`status.json`. The [notification report](../reports/LINEAR_DESIGN_COMPLETION.md)
initially says monitoring and is replaced with the final outcome. On failure it
includes the controller error and the end of its active log. On success it points
to the completed results report.

For later studies, attach the same watcher to the new status file and choose a
separate report path. A file lock prevents duplicate watchers for the same study.
The watcher pins the run fingerprint, attempt and controller process creation
time so that a reused process ID cannot masquerade as the original process.

## What triggers an alert

- Success requires the controller's complete status, completion certificate,
  results report and registry with matching hashes. Finishing training alone
  does not count as success.
- A controller-reported failure triggers an error notification.
- Two consecutive observations of a missing or replaced controller process
  trigger an interruption notification. The watcher rereads the status before
  deciding, to allow the controller to finish writing its final status.
- Ten consecutive status-reading failures trigger a monitoring-error notification.
  A process-permission error is treated as unknown rather than proof of an exit.

The outcome is saved before notification delivery. `notification.json`,
`notification_receipt.json` and archived `watcher_events/*.json` live beside the
controller status. Delivery retries are bounded and recorded. The same event is
not notified again after successful delivery, including after a watcher restart.

## Windows delivery and validation

`notify_study.ps1` reads its message from JSON and sends a Windows toast using the
installed PowerShell identity. It confirms the tagged notification appears in
Windows notification history. A shell notification is a fallback if toast
delivery is unavailable. Acceptance records do not claim that the user viewed
the notification. No email, external service or notification-setting changes are
required.

A clearly labeled setup notification was successfully accepted by Windows on
September 21. All 12 watcher tests pass, covering completion certificates,
failures, process exits, exit/status races, reused PIDs, transient read errors,
delivery retries, duplicate prevention and the read-only `--once` command.
The active experiment's 15 frozen source files remain unchanged.

The watcher uses the documented Windows
[toast interface](https://learn.microsoft.com/en-us/uwp/api/windows.ui.notifications.toastnotificationmanager.createtoastnotifier)
and [notification history](https://learn.microsoft.com/en-us/uwp/api/windows.ui.notifications.toastnotificationhistory.gethistory).
Delivery here is a local desktop alert. The IDE has no scheduled-task management
interface; see the official [scheduled-task documentation](https://learn.chatgpt.com/docs/automations).
