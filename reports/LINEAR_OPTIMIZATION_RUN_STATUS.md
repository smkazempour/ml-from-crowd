# Linear optimization run status

**Interpretation correction (September 22):** the baseline characteristics were
already daily ranks. Report 11's transformation comparison re-ranks those stored
values; it is not raw levels versus ranks. The [sample and transformation audit](11b_sample_and_transformation_audit.md)
documents this correction and confirms that all prediction rows have retained messages.

**Complete: September 22, 2026 at 3:17:55 a.m. Central** (08:17:55 UTC).
The study launched September 21 at 9:44 p.m. Central and took **5 hours,
33 minutes, 49 seconds**, including the pilot, fitting, evaluation and reporting.
All four stages exited successfully.

All **216 monthly outcome checkpoints** and **172 forecasting procedures** are
complete. Each procedure covers 108 months and 3,034,035 prediction rows.
All **13 final artifact hashes** were independently rechecked after completion.

Start with the [readable summary](11a_linear_optimization_summary.md), or see
[Report 11](11_linear_optimization_results.md) for the complete results and tests.

**Validation:** all 29 focused tests pass. They cover numerical solver checks,
exact legacy-procedure reproduction on test data, transformation timing and missing
values, validation-month omission diagnostics, resumability and tamper detection,
evaluation parity with existing kernels, report integrity, and notification startup.

**Execution:** controller PID at launch 35136; watcher PID at launch 23380. The
pilot used two workers; the full study used four workers, each with two numerical
threads. Windows accepted the completion notification at **3:18:26 a.m. Central**.
The delivery receipt does not establish whether the notification was viewed.

- Study identity: `ed854a1f5bf3da39`.
- Authoritative live status: `.runs/linear_optimization_v3/study/ed854a1f5bf3da39/status.json`.
- Pilot log: `.runs/linear_optimization_v3/study/ed854a1f5bf3da39/pilot_attempt_001.log`.
- Launch log: `.runs/linear_optimization_v3/launch_20260922T024404Z.log`.
- Launch receipt: [linear_optimization_v3_execution.json](data/linear_optimization_v3_execution.json).
- Completed registry: `.runs/linear_optimization_v3/study/ed854a1f5bf3da39/linear/full_b217f4648f5fe89a/registry.json`.

The [declared experiment](LINEAR_OPTIMIZATION_EXPERIMENT.md) contains 172 forecasting
procedures and 216 monthly outcome jobs. It crosses the proposed penalty, market
characteristic transformation and text PCA choices. Existing study outputs remain
unchanged. The January 2014 / December 2022 pilot and the full run passed
monthly reproduction checks against all 19 original procedures.

The independent watcher's final outcome and notification receipt are recorded in
[LINEAR_OPTIMIZATION_COMPLETION.md](LINEAR_OPTIMIZATION_COMPLETION.md).
