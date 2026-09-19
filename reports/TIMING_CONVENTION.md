# Signal timing and concrete window design

2026-09-12. This note extracts the current code's timing convention and makes the
[experimental protocol](EXPERIMENTAL_PROTOCOL.md) more specific following the user's
clarification. The timing convention is established; the longer-window design below
is now implemented in the shared protocol runners. Corrected linear training and evaluation
are complete for 96 specifications, 648 month-target checkpoints and 108 test months;
see [Report 07](07_protocol_linear_results.md). The NN runner is implemented and its
standard-budget pilot passed; the full study has started. Full NN results remain pending;
[NN_RUN_STATUS.md](NN_RUN_STATUS.md) records live execution status. Source panels
and historical predictions remain unchanged.

## 1. Time t is the first close assigned to a message

Let C_t be the market close of trading session t. A message is assigned to the first
close at or after its timestamp under the current code's boundary convention.
The stock-day row at t aggregates messages in:

    (C_(t-1), C_t]

Its forward h-session target starts at C_t and ends at C_(t+h).
In particular, the one-day target is C_t to C_(t+1), not the return ending at C_t.
The time from the message's original posting to C_t is outside that target.

For ordinary sessions:

| Message posting time | Assigned signal close t | One-day target |
|---|---|---|
| Monday 10:00 a.m. | Monday close | Monday close to Tuesday close |
| Monday 5:00 p.m. | Tuesday close | Tuesday close to Wednesday close |
| Friday after close | Next trading close, normally Monday | Monday close to Tuesday close |
| Weekend or market holiday | Next trading close | That close to the following trading close |

The table assumes no intervening holiday except where stated. For h=5, advance five
exchange sessions from the assigned close; do not add five calendar days.
After-hours messages are not assigned an additional lag after t has been determined.

The intended cumulative-return definition is:

    R(i,t,h) = product[j=1,...,h](1 + r(i,t+j)) - 1

Here r is the corresponding one-session return. This is not generally identical to a
ratio of unadjusted prices when distributions or corporate actions occur. The repository
imports the upstream cumulative-return columns; its source code does not establish the
upstream compounding or delisting treatment. The formula is the target specification
under the user-confirmed convention, and must be reconciled to those upstream columns
before extending horizons.

## 2. Code trace

Line references are notebook JSON source lines in the current working copy.

| Stage | Implementation and source |
|---|---|
| Convert timestamps | [data_cleaning.ipynb](<../00 - data cleaning/data_cleaning.ipynb>), parse_timestamp, lines 464–487: convert to US/Eastern; naive source timestamps are treated as UTC. |
| Find trading dates | Same notebook, calendar construction, lines 391–438: use Fama–French trading dates, store first_close and second_close, and backfill nontrading calendar days to the next trading day. |
| Identify after-close messages | Same notebook, lines 319 and 520: fixed 16:00 cutoff; is_after_hours is strictly time > MARKET_CLOSE. |
| Assign t | Same notebook, assign_trading_date, lines 537–575: default to first_close; after-hours messages on business days use second_close. |
| Match securities | [merge_with_crsp.ipynb](<../00 - data cleaning/merge_with_crsp.ipynb>), lines 408–413: join symbol/date to ticker/date without shifting the assigned date. |
| Construct features | Feature groups 01, 02 and 04 aggregate by symbol/date; [features_08_text_embeddings.ipynb](<../01 - feature extraction/features_08_text_embeddings.ipynb>), lines 382–410, averages message vectors on the same stock-day. |
| Join targets | [perpare_training_data.ipynb](<../02 - prepare training dataset/perpare_training_data.ipynb>), lines 74–134: import annual CRSP files, retain precomputed targets, and join social features on the same date. |
| Join text | [build_text_master.ipynb](<../02 - prepare training dataset/build_text_master.ipynb>), line 128: join embeddings on ticker/date, retaining the master targets. |
| Form portfolios | [form_portfolios.ipynb](<../05 - trading/form_portfolios.ipynb>), line 213: f_ret = ret.shift(-1) within sorted permno; signal date t therefore uses the next observed stock-return row. |
| Label portfolio returns | Same notebook, lines 296–300, and [trading_library.py](<../05 - trading/trading_library.py>), lines 74–83: shift=1 moves the completed return series to its realization date after formation. It is not a delayed-entry strategy. |

Current panel contents: raw f_cumret1, plus abnormal-return horizons 1, 3, 5, 10, 21,
42 and 63. The builder sets ar_dgtw_h = f_cumret_h - f_dgtw_ret_h. Raw cumulative
returns for h>1 must be joined from the upstream CRSP files before training those targets.
Factor-adjusted CARs are sums of abnormal one-session returns, a different construction
from compounded raw returns.

### Exact implementation boundaries

The ordinary-session convention is clear. The following are bounded implementation
details to fix or document when the calendar/data are next rebuilt:

- Exactly 16:00:00 is assigned to that same day's close. Thus the code implements
  "at or after," rather than a strictly later close at equality.
- The close is always assumed to be 16:00. Early-close sessions are not represented;
  a message after the actual early close but before 16:00 is assigned too early.
  Correcting the exchange schedule preserves the first-close rule; it changes the
  affected historical feature rows and requires a new data version.
- At the final calendar boundary, messages without a mapped next trading date are
  dropped. The current cleaning calendar ends on 2024-12-31. Extend it beyond the last
  input timestamp when rebuilding.
- The trading notebook's next observed per-stock row must be checked against the next
  exchange session. Missing stock-return dates must not silently lengthen the horizon.
- The factor-CAR builder currently skips missing values when summing and uses whichever
  horizon columns are present. Require all h observations for an h-day target; a partial
  sum or an all-missing zero is not a valid complete horizon. The known 2013 DGTW-21
  corruption is a separate repair requirement.

The pure cleaning functions were checked on synthetic weekday, after-close, weekend,
holiday and exact-close examples. UTC conversions in summer and winter preserve the
intended Eastern-time assignment. These checks establish routing behavior; they do
not validate every precomputed return in the external CRSP files.

Use this C_t-to-C_(t+h) clock for the main predictive study. An execution analysis can
add a pre-close input buffer or delayed entry as a separately named sensitivity; that
does not redefine the main prediction target.

## 3. Fixed validation, varying fitting history

Use unambiguous names: F is the number of fitting signal dates, V is the number of
validation signal dates, and h is the forward-return horizon. These are exchange-session
counts, not row counts and not calendar-day counts.

| Specification ID | Fitting dates F | Validation dates V | Role |
|---|---:|---:|---|
| fit252_val126 | 252 | 126 | Short-history comparison |
| fit504_val126 | 504 | 126 | Proposed primary specification |
| fit756_val126 | 756 | 126 | Long-history comparison |
| legacy252_80_20 | 202 | 50 | Existing NN reference; uses the historical boundary rule |

The three new specifications keep the same six-month validation period, target, input
transforms, loss, tuning grid, seed list and test dates. Longer fitting windows extend
backward while leaving validation unchanged. The comparison therefore isolates the
length of fitting history. Two years of fitting is a pragmatic starting point, not
a demonstrated optimum; all three results will be reported.

This deliberately refines proposal v1, which varied a total history of 252/504/756
dates and allocated 20% of each to validation. That design changed fitting and validation
length simultaneously. Here, 504 means 504 actual fitting dates; it does not mean
403 fitting dates plus 101 validation dates.

Refit once per month. Fit candidate parameters using only signal dates preceding the
validation period in the fitting block, use validation to select the penalty and stopping checkpoint, and
complete final selection after the preceding month's last close. Freeze the selected
parameters throughout the next month's daily forecasts. No final train-plus-validation
refit and no full-test selection of a winning window.

## 4. Exact label-end rules

Index the exchange sessions by integers. Let b be the closing session immediately
before the first test session. Complete monthly selection after C_b using only inputs
available by that selection time and labels ending no later than C_b, in time to
produce the first test forecast at C_(b+1).

For the validation signal dates, choose:

    validation_last  = b - h
    validation_first = b - h - V + 1

Their last outcome ends at C_b. Place F fitting dates before validation as follows:

    fitting_last  = validation_first - h - 1
    fitting_first = fitting_last - F + 1

This guarantees:

    fitting_last + h < validation_first
    validation_last + h = b

Fitting labels therefore finish before the first validation signal close, and every
validation label is realized by the monthly selection cutoff. The strict inequality
keeps fitting labels out of the first validation input interval. The h omitted signal
dates at each boundary implement these rules; do not add another independent embargo.
If a return/benchmark label is published after its endpoint, use its later availability
timestamp and move the eligible block back accordingly.

For the primary F=504, V=126, h=1 design:

| Block | Session indices | Count |
|---|---|---:|
| Fitting | b-631 through b-128 | 504 |
| Fitting/validation gap | b-127 | 1 |
| Validation | b-126 through b-1 | 126 |
| Signal date whose label is not yet known at C_b | b | 1 |
| Test | b+1 onward, through the month's final session | Actual sessions in that month |

The required preceding calendar span is F + V + 2h. For h=1 it is 380, 632 or 884
sessions for the three fitting lengths. For h=21, the primary span is 672 sessions;
for h=63, it is 756. These are history requirements; only F dates fit parameters.
A gap date's backward-looking feature history may still inform later features when
available; the gap excludes its forward outcome as a fitting/validation label.

When the three histories are compared at a given h, use exactly the same validation
dates and eligible observations and test months. Each longer fitting set contains the
shorter one, extending farther into the past. Label availability, target-rank construction
and feature scaling follow the shared protocol separately for each fitting block.

### Why the current two-calendar-day offset should change

Both current text-model notebooks choose their last history date using calendar
month start minus two calendar days. The NN then splits 252 observed panel dates into
202 fitting and 50 validation dates. It has no explicit label-end purge, and its date
grid depends on observed feature-complete rows.

For March 2015, the offset lands on Friday February 27; that date's one-day return
finishes Monday March 2, the first test signal close. It is not available at February
27 close, the proposed monthly selection cutoff. The old heuristic therefore does not
enforce the new pre-month rule. This does not alone establish that every historical
result used a future outcome: that judgment depends on its exact fitting clock.

Replace the offset with the explicit session and label-end rules above. Also correct
the existing first-window eligibility check, which requires 253 available panel dates
to extract a nominal 252-day history. Existing linear penalty selection from prior
out-of-sample months must likewise exclude labels not yet realized at its selection
cutoff. Prefer the same fixed validation block for the new matched comparisons.

## 5. Test periods, horizons and run order

The text-master metadata reports 2010-06-02 through 2023-12-29. A read-only audit of
the full CRSP annual panels for 2010–2014 gives the following calendar warm-up boundaries:

| Fitting dates | Validation dates | First eligible full month, h=1 | First eligible full month, h=63 |
|---:|---:|---|---|
| 252 | 126 | December 2011 | June 2012 |
| 504 | 126 | January 2013 | July 2013 |
| 756 | 126 | January 2014 | July 2014 |

The [calendar audit](data/window_calendar_audit.json) records source metadata, counts,
and exact block dates. These are dates observed across the full CRSP panels, not a
separate official session/early-close audit. Calendar warm-up does not establish
uninterrupted feature coverage or adequate labeled cross-sections; verify those before
training. Never shorten early fitting windows to fill missing warm-up.

**Specify January 2014–December 2022 for the primary common h=1 comparison**, conditional
on target eligibility. The source-input coverage audit below passes for all 108 months
and all three fitting windows. Show the 504/126 specification's January 2013 onward
results as a supplemental longer history. Historical 2012–2022 results remain their own
reference table; a difference between unequal test spans is not a training-window effect.
If all three fitting lengths and all horizons through h=63 are compared together,
the common calendar start is July 2014. Otherwise report horizon-specific coverage
explicitly. Source text coverage is incomplete across September–December 2023, including
all of October; repair and audit that source coverage before a 2023 extension.
Existing historical results remain development evidence.

### Implemented source and coverage audit

[protocol_data.py](../tools/protocol_data.py) reads the full 2010–2023 CRSP annual
panels to construct session indices, then prepares the existing 3,514,785 text-master
rows without filtering prediction eligibility by future targets. Every required input
date in the fitting, validation and test blocks is present for all 108 primary test
months, January 2014–December 2022, at F=252, 504 and 756. The supplied text panel does
have two absent early dates, June 10 and June 23, 2010, outside those required blocks.

The late-2023 issue is in `text_master.pkl` itself, rather than only in saved model
outputs. Counts below are dates with any text-master rows, not proof of complete
stock/message coverage on those dates:

| Month | Full CRSP session dates | Dates represented in text master |
|---|---:|---:|
| September 2023 | 20 | 4 |
| October 2023 | 22 | 0 |
| November 2023 | 21 | 1 |
| December 2023 | 20 | 6 |

The prescribed fitting-session count and the number of eligible labeled dates are
reported separately. For January 2014, F=756 contains all 756 input dates, but 16 early
dates have fewer than 10 observed targets. The minimum-cross-section rule excludes those
dates for both raw and DGTW fitting, leaving **740 eligible fitting dates**. This preserves
the original calendar window; it does not pull in older dates or shorten the specified
history. F=252 and F=504 have their full eligible fitting-date counts in that month, and
all three windows have 126 eligible validation dates.

The original prepared-run [manifest](../.runs/protocol_v1_1/prepared/15426a6f29a923e3/manifest.json)
and [precision audit](../.runs/protocol_v1_1/prepared/15426a6f29a923e3/source_target_precision.json)
remain as provenance. They showed a maximum raw-return difference of 8.88e-16 for
3,512,922 comparable next-session observations and exact source agreement for 3,194,624
comparable DGTW values. Source agreement alone did not validate unpaired observations.
The original cache `15426a6f29a923e3` is superseded for the new experiment by the
[corrected cache manifest](../.runs/protocol_v1_1/prepared/8f3f4eb44b399771/manifest.json)
and [horizon-repair audit](../.runs/protocol_v1_1/prepared/8f3f4eb44b399771/horizon_repair_audit.json).

The unpaired-observation audit found one finite primary-period label with the wrong
session horizon: FTNW, permno 17182, March 13, 2019 (`mm_index=9900658`). Its imported
`f_cumret1=-0.243243` equals the next observed stock return on March 26, **nine exchange
sessions later**. The stock has no CRSP rows for the eight sessions March 14–25.
The corrected cache sets that raw target and its DGTW counterpart missing and recomputes
both target cross-sections' ranks on March 13. The stock-day stays in the prediction
universe; X, keys, session codes and calendar are byte-identical to the original cache.
Future preparation applies the same exclusion whenever a finite h=1 label lacks a
consecutive stock/session successor inside the cached calendar.

For the primary 2014–2022 period, the corrected cache retains **3,034,035 prediction
rows and 3,033,080 finite raw targets, all verified against the next exchange session's
return**. Before correction there were 3,033,081 finite raw targets, including the
FTNW exception. Across the full cache, 752 finite raw targets on December 29, 2023
(677 finite DGTW targets) remain explicitly unverified because their successor falls
beyond the cached calendar. These boundary observations are outside the primary period.

Known longer-horizon diagnostic labels for FTNW are also excluded whenever the nominal
interval `(t,t+h]` intersects the identified March 14–25 missing sessions. The counts
below are newly nulled labels; previously missing labels remain missing:

| Horizon h | Raw `f_cumret_h` exclusions | DGTW `ar_dgtw_h` exclusions |
|---|---:|---:|
| 3 | 3 | 3 |
| 5 | 5 | 5 |
| 10 | 10 | 10 |
| 21 | 20 | 20 |
| 42 | 38 | 38 |
| 63 | 55 | 51 |

This is a bounded exclusion of the known gap, not a complete certification or rebuild
of all longer-horizon targets. Clearly named `crsp_ar_*` upstream audit copies retain
their original values. Early-close message routing and broader longer-horizon/factor-CAR
completeness remain separate limitations. Source files and the original cache are
unchanged. The corrected linear run combines **468 certified unaffected checkpoint
reuses and 180 completed reruns**. Certification checked unchanged fitting/tuning inputs
and code semantics; every affected fitting/validation window was rerun. Training and
evaluation are now complete for all 648 checkpoints and 96 model specifications over
108 months. [Report 07](07_protocol_linear_results.md) contains the corrected results
and migration-audit link.

For an actual calendar example, the primary 504/126 specification's first eligible
month is January 2013:

- Fitting signal dates: 2010-06-29 through 2012-06-26, 504 sessions.
- Fitting/validation gap: 2012-06-27; the last fitting return realizes at that close.
- Validation signal dates: 2012-06-28 through 2012-12-28, 126 sessions.
- Final validation outcome and monthly selection cutoff: 2012-12-31 close.
- First test signal date: 2013-01-02, forecasting that close to the next close.

The validation and test labels are therefore separated by their actual realization
dates, including the New Year holiday, without relying on calendar-day subtraction.

Execution sequence (linear stages and the standard-budget NN pilot are complete;
the full NN study has started; see [live NN status](NN_RUN_STATUS.md)):

1. Implement the shared session calendar, explicit split generator and label-end checks.
2. Verify the primary 504/126 design on January 2014, March 2015 and December 2022,
   covering early/recent samples and a weekend month boundary. Use core and text+core
   at the full NN settings (five seeds, early stopping, up to 100 epochs). Check the
   longer-horizon split logic separately before fitting any unrepaired h>1 targets.
3. Fit the primary matched OLS/ridge/lasso/elastic-net comparison, followed by NN3, using the four current feature sets,
   rank-target squared loss and equal-date weights, for raw h=1 and DGTW h=1.
4. Repeat exactly that comparison for 252/126 and 756/126. Keep a fixed common sample
   and report all three histories; do not promote whichever wins the full test afterward.
5. Treat directly trained h>1 models as separately registered tasks. Join raw h>1 targets
   and repair/validate abnormal targets first. Evaluating an h=1 signal against a longer
   outcome is a persistence diagnostic, not an h-day-trained forecast.

The split formulas were checked for F in {252,504,756}, V=126 and h in
{1,3,5,10,21,42,63}: counts, nesting at fixed h, pre-validation maturity and the
pre-test label endpoint hold. The modeling notebooks and historical predictions still
retain their historical results. [protocol_data.py](../tools/protocol_data.py) now supplies
the common calendar, transformations, weights and split rules to
[protocol_linear.py](../tools/protocol_linear.py) and
[protocol_nn.py](../tools/protocol_nn.py). Corrected full linear training and evaluation
are complete in [Report 07](07_protocol_linear_results.md); full NN results remain pending,
with execution tracked in [NN_RUN_STATUS.md](NN_RUN_STATUS.md). Direct h>1 training,
actual-close source rebuilding and the separate news/earnings/aggregate-market tracks
remain outside this implementation.
