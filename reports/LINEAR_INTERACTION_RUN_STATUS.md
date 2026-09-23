# Explicit interaction study status

**Completed September 22, 2026 at 4:38:43 p.m. Central**, with no reported error.
All **216 monthly fitting jobs**, prediction verification, evaluation, and reports
finished successfully. All 148 procedures cover 108 forecast months and 3,034,035
prediction rows each. All **13 final artifact hashes** were independently verified.

Start with [the short results summary](13a_linear_interaction_summary.md), or read
[the complete register](13_linear_interaction_results.md). Windows accepted the
completion notification at **4:38:47 p.m. Central**; the delivery receipt is saved
beside the study status.

The study started at 1:32 p.m. Central; full fitting began at 1:39 p.m. after the
pilot passed and finished at 4:12:18 p.m. Evaluation finished at 4:38:37 p.m.,
followed by report generation and final certification.

All **30 focused tests passed**, covering exact reproduction of the nine additive
references, fitting-only transformations, independent solver checks, checkpoint
integrity/resume, common-sample evaluation and report generation. All ten prepared
artifact hashes and four historical pilot checkpoints passed preflight checks.

The January 2014 and December 2022 pilot completed for both targets. All 148
procedures were published and evaluated. The nine additive references reproduced
their archived forecasts and daily ICs exactly in all four pilot jobs. Each final
refit used 630 input dates. Early jobs took 51-54 seconds; late jobs took 199-208
seconds with two workers. Pilot fitting, evaluation, and reporting all exited
successfully; all 13 pilot artifact hashes were independently checked. Pilot
forecast performance did not change the declared model menu.

- Controller PID at launch: `4412`; notification watcher PID: `13572`.
- Full fitting used four workers, each with two numerical threads. The pilot used two workers.
- Authoritative status: `.runs/linear_interactions_v1/study/78b3dd65e0d5f8a6/status.json`.
- Logs and source snapshot: the same study directory.
- [Execution record](data/linear_interactions_v1_execution.json).
- [Completion/error notification record](LINEAR_INTERACTION_COMPLETION.md).

The watcher confirmed this controller's completion certificate and delivered the
configured Windows desktop alert. Its final status and delivery receipt are
preserved. No restart is needed for this completed study.

The approved Stage A menu contains 148 procedures (74 for each return outcome):
13 joint/reference designs using OLS, ridge and elastic net, plus 35 individual
social interaction additions using ridge. Monthly refitting generates 216
outcome jobs across January 2014-December 2022.

See [the experiment](LINEAR_INTERACTION_EXPERIMENT.md) and
[execution instructions](../tools/LINEAR_INTERACTION_STUDY.md).
