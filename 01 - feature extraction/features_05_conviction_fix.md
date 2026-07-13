# features_05 — Conviction Index carry-forward fix

**File**: `01 - feature extraction/features_05_sentiment_dynamics_cohorts.ipynb`
**Cell**: Section 9 main processing loop
**Affected features**: `conviction_index_3`, `conviction_index_5`, `conviction_index_10` (Feature 10)
**Not affected**: flips (5–9), first-mover (16), cohorts (17–26)
**Found during**: pre-merge integration review of the `integrate-features-05` branch (July 2026)

---

## The bug

In the original Section 9 loop, the flip call **reassigns** `prev_daily_dir` to the
*current* year's directions, and that reassigned value is what conviction then
receives as its "previous years" carry:

```python
# ORIGINAL (buggy)
flip_yr, prev_daily_dir = calc_flip_features(          # <-- prev_daily_dir now = THIS year
    df_year, horizons=HORIZONS, prev_daily_dir=prev_daily_dir
)
# ... trim prev_daily_dir to last 250 td ...
conv_yr = calc_conviction_features(
    df_year, k_values=K_VALUES, prev_daily_dir=prev_daily_dir   # <-- passes THIS year as "previous"
)
```

Inside `calc_conviction_features`, the incoming `prev_daily_dir` is concatenated
with the current year's own `daily` table:

```python
daily = pd.concat([prev_daily_dir, daily]).sort_values(...)
```

Because `prev_daily_dir` at that point *is* the current year (last 250 trading
days of it), **every (user, symbol, date) row in that window appears twice**.

## Why that corrupts the Conviction Index

Duplicated rows have identical `direction`, so for the duplicate:

- `prev_dir == direction` → `dir_changed = False` → the duplicate **extends the
  streak instead of being ignored**. A user who posted Bullish on 3 distinct days
  shows `streak = 6`, not 3 — so `streak >= K` triggers roughly twice as fast as
  it should.
- Both copies of a row survive into the aggregation, so the numerator
  `(streak >= K).sum()` **double-counts users**, while the denominator uses
  `nunique(user_id)` — the ratio is inflated on both fronts.
- Meanwhile the *actual* previous year's directions are discarded, so the
  cross-year carry that `prev_daily_dir` was designed for never reaches
  conviction — every year is a cold start on top of the duplication.

**Net effect**: `conviction_index_K` is materially overstated for every year
after the first, with the bias concentrated in the ~250-trading-day overlap
window. Values can exceed their intended interpretation ("% of users on a
≥K-day streak") because streaks accumulate at ~2× speed.

## Why the notebook's own test didn't catch it

The Section 6 test cell calls `calc_conviction_features(df_sample, k_values=[3, 5])`
with the default `prev_daily_dir=None` — the concat branch never runs on a single
year, so the function looks correct in isolation. The bug only exists in how the
main loop wires the two functions together.

## The fix

Snapshot the incoming carry before the flip call reassigns it; give that snapshot
to **both** flips and conviction; only then promote the current year's directions
to become next year's carry:

```python
# FIXED
prior_daily_dir = prev_daily_dir                        # last year's directions
flip_yr, cur_daily_dir = calc_flip_features(
    df_year, horizons=HORIZONS, prev_daily_dir=prior_daily_dir
)

conv_yr = calc_conviction_features(
    df_year, k_values=K_VALUES, prev_daily_dir=prior_daily_dir   # <-- prior, not current
)

# Prepare carry for NEXT year (trim to last 250 trading days)
prev_daily_dir = cur_daily_dir
if prev_daily_dir is not None and len(prev_daily_dir) > 0:
    cutoff_td = td_idx.get(df_year["date"].max(), 0) - 250
    cutoff_date = td_sorted[max(0, cutoff_td)]
    prev_daily_dir = prev_daily_dir[prev_daily_dir["date"] >= cutoff_date].copy()
```

Semantics after the fix:

| | before fix | after fix |
|---|---|---|
| flips receive | prior years ✔ | prior years ✔ (unchanged) |
| conviction receives | **current year (duplicate)** ✘ | prior years ✔ |
| next-year carry | current year, trimmed ✔ | current year, trimmed ✔ (moved below both calls) |

The fix also gives conviction the cross-year streak continuity it was designed
for: a user Bullish through December stays on their streak in January.

## Measured impact (validation run, July 2026)

Validated on real data (`merged_with_crsp_mlcrowd`, 2023 full year = 7.19M rows,
plus the small 2024 file = 2 trading days) with the exact committed notebook code.
The 2024 file makes the bug *provable*: with only 2 days of history, no user can
have a streak ≥ 3, so a correct cold-start `conviction_index_3` must be exactly 0.

| conviction_index (mean, 2024) | buggy wiring | fixed wiring (2023 carry) | correct cold start |
|---|---|---|---|
| K = 3  | **0.083** (impossible > 0) | 0.465 | 0.000 |
| K = 5  | 0.000 | 0.347 | 0.000 |
| K = 10 | 0.000 | 0.241 | 0.000 |

Two distinct errors, both visible:

1. **Duplication manufactures streaks**: buggy K=3 = 0.083 vs a true maximum of
   0 — users with just 2 same-direction days were double-counted into phantom
   4-day streaks. In a full-year run this is the inflation channel (streaks
   accumulate at ~2× speed inside the 250-td window).
2. **The real carry is discarded**: fixed wiring carries genuine 2023 streaks
   into January (0.465 / 0.347 / 0.241), which the buggy loop lost completely.

Families A/B/C also validated clean on full 2023 (all ratios bounded [0,1],
net-sentiment in [−1,1], flip counts non-negative and weakly increasing in H,
no unexpected NaNs); family D validated on 2024 with a 2023-Q4-built carry
(all ratios bounded, carry state grows correctly across the year boundary).

## Practical consequences

- `features_05_sentiment_dynamics_cohorts.pkl` must be **regenerated** if it was
  produced with the original loop — the three `conviction_index_*` columns are
  wrong in any multi-year run (single-year test runs were unaffected).
- Downstream artifacts built from such a pickle (`features_master.pkl`,
  `merged_master.pkl`, all-features `predictions_*.pkl`) inherit the error in
  those three columns and should be rebuilt after regeneration.
- Expect regenerated `conviction_index_*` values to be **lower** on average and
  bounded as intended.

## Other review notes (informational, not changed in code)

Flagged during the same review, left as-is pending discussion:

1. **Specialist ratio** double-counts symbols discussed in both a prior year and
   the current year (`prior_nsyms + yr_nsyms_before` overlap), slightly
   *undercounting* specialists.
2. **Re-entry** (`reentry_H`) only detects returns after a *prior-year* gap
   (requires a `carry["sym_last"]` hit), so a within-year absence > H days is not
   flagged.
3. `calc_cohort_features` relies on row-wise `.apply(axis=1)` / `iterrows` in
   several places — expect very long runtimes on the full 15-year dataset.
