# features_05 — integration review fixes

**File**: `01 - feature extraction/features_05_sentiment_dynamics_cohorts.ipynb`
**Found during**: pre-merge integration review of the `integrate-features-05` branch (July 2026)
**Validated on**: real `merged_with_crsp_mlcrowd` data — full 2023 (7.19M rows), 2023-Q3/Q4 subsets, and the 2024 stub (2 trading days)

| # | Fix | Affected features | Severity |
|---|-----|-------------------|----------|
| 1 | Conviction carry-forward wiring | `conviction_index_{3,5,10}` | High — corrupt values in any multi-year run |
| 2 | Within-year re-entry detection | `reentry_volume_H` | Medium — feature was ~always zero |
| 3 | Specialist symbol counting (double-count + datetime-unit bug) | `specialist_ratio` | Medium/High — wrong on multi-year runs; wrong everywhere on pandas ≥2 |
| 4 | Night-owl/day-trader running count misalignment | `night_owl_ratio`, `day_trader_ratio` | Medium — mixed adjacent users' histories |
| 5 | Trading calendar source + coverage check | flips, conviction (silent row drops) | Low — tail-of-sample risk |
| 6 | Vectorization of `calc_cohort_features` | (performance only) | Blocker for the full run |

Unchanged-by-design: flip features (5–9), first-mover (16), fresh-blood/retention,
whale/minnow, crowding Gini — the rewrite reproduces these **exactly** (verified;
see "Verification" below).

---

## Fix 1 — Conviction Index carry-forward (Section 9 main loop)

### The bug

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
with the current year's own `daily` table — so **every (user, symbol, date) row in
the last-250-td window appears twice**. Duplicates have identical `direction`, so
`dir_changed=False` and each duplicate **extends the streak instead of being
ignored** (a 3-day streak reads as 6), and both copies survive into the
`(streak >= K).sum()` numerator. Meanwhile the *actual* previous year's directions
are discarded, so cross-year streak continuity never happens.

The Section 6 test cell calls conviction with `prev_daily_dir=None`, so the bug
is invisible on a single-year test — it only exists in how the main loop wires
the two functions together.

### The fix

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

### Measured impact

The 2024 file has only 2 trading days, which makes the bug *provable*: with 2 days
of history no user can have a streak ≥ 3, so a correct cold-start
`conviction_index_3` must be exactly 0.

| conviction_index (mean, 2024) | buggy wiring | fixed wiring (2023 carry) | correct cold start |
|---|---|---|---|
| K = 3  | **0.083** (impossible > 0) | 0.465 | 0.000 |
| K = 5  | 0.000 | 0.347 | 0.000 |
| K = 10 | 0.000 | 0.241 | 0.000 |

Two error channels, both visible: duplication manufactures phantom streaks
(0.083 where the true max is 0), and the genuine 2023 carry is lost (0.465 vs
0.083 with carry restored).

---

## Fix 2 — Re-entry only fired across year boundaries

`reentry_H` required the (user, symbol) pair to exist in `carry["sym_last"]` —
which only holds *prior years'* activity. A user who posts about a stock in
January, goes silent, and returns in December (gap > H) was never flagged,
although the definition ("inactive for >H days but returned today") clearly
covers them. Since `gap_days.notna()` already implies a previous post exists
(within-year via `shift(1)` or prior-year via carry), the carry-membership test
was simply dropped:

```python
# was: gap_days.notna() & (gap_days > hc) & <row-wise carry-membership apply>
daily[f"reentry_{H}"] = daily["gap_days"].notna() & (daily["gap_days"] > hc)
```

**Measured**: on 2023-Q3 alone (1.94M rows, no carry), the fixed code detects
**8,713** within-year re-entries at H=21 — the original code reports **0** by
construction. (H=63 correctly stays 0 on a 3-month window: 63 td ≈ 91 calendar
days > the window length.)

---

## Fix 3 — Specialist ratio: double-counting + a datetime-unit landmine

Two problems in one place.

**(a) Cross-year double-count.** `total_nsyms = prior_nsyms + yr_nsyms_before`,
where `prior_nsyms = len(carry["sym_set"][user])` and `yr_nsyms_before` counted
*all* of this year's first occurrences. A symbol discussed in both 2022 and 2023
was counted in both terms → symbol counts inflated → users misclassified as
non-specialists. Fixed by excluding carry-known symbols from this year's
first-occurrence table before counting.

**(b) Unit bug in the original counting mechanism.** The original code compared
`first_occ` dates as raw integers against `Timestamp.value`:

```python
fo_dates = first_occ[...]["first_dt"].values.astype("int64")   # unit follows the array
grp["date"].apply(lambda d: np.searchsorted(fo_dates, d.value)) # .value is ALWAYS ns
```

On pandas ≥ 2.x, date-only strings parse as `datetime64[us]` (or `[s]`), so
`astype("int64")` yields **microseconds** while `d.value` is **nanoseconds** —
every first-occurrence integer is ~1000× smaller than every message date, so the
count returned is the user's **total symbol count including today's and future
first occurrences**, regardless of date. On our environment this affected every
row: Q4-2023 `specialist_ratio` mean was **0.5755 (wrong)** vs **0.7719
(correct)**. Whether your environment was affected depends on the pandas
version; the rewritten code (`pd.merge_asof` on actual datetime columns) is
unit-safe on all versions.

---

## Fix 4 — Night-owl / day-trader counts mixed adjacent users

The exclusive running count of after-hours posts was built as:

```python
df.groupby("user_id")["is_after_hours"].cumsum().shift(1).fillna(0)
```

The `.shift(1)` sits **outside** the groupby, so it shifts the whole column: each
row inherits the running count of the *previous row's user* — and since `df` is
sorted by date (not user), that is almost always a different user. Fixed with an
exclusive per-user cumsum:

```python
_ah = df["is_after_hours"].astype(int)
df["yr_ah_cc"] = _ah.groupby(df["user_id"]).cumsum() - _ah
```

**Measured** (Q4 2023): `night_owl_ratio` mean 0.0806 → **0.0437**;
`day_trader_ratio` mean 0.4605 → **0.4681**.

---

## Fix 5 — Trading calendar: local CRSP instead of the Fama-French download

The notebook fetched the FF daily calendar from Ken French's site, which lags the
present by ~1–2 months. Any message dated beyond the fetched calendar mapped to
`NaT` in `td_idx`, and its flips/conviction rows were **silently dropped**. The
calendar is now built from the local CRSP `dsf_final_*.pkl` files — no network,
no lag, and by construction the same calendar the merged data was built on — and
the main loop raises immediately if any data date is missing from the calendar
(no more silent drops).

---

## Fix 6 — Vectorization of `calc_cohort_features`

The original implementation used row-wise `.apply(axis=1)` for carry lookups, a
per-row `searchsorted` inside a groupby-apply, a per-group Python Gini, and —
worst — a carry update that scanned the full dataframe once **per user**
(O(users × rows): 127k users × 7.2M rows for 2023). The rewrite keeps the exact
same outputs and carry-dict structure but is fully vectorized (indexed carry
lookups, `merge_asof` for first-occurrence counting, closed-form grouped Gini,
grouped aggregation for the carry update).

**Measured**: Q4-2023 8.1s → 0.3s (**×32**); **full 2023 (7.2M rows) in 15.1s**;
estimated full 15-year run ≈ **4 minutes** for the cohort family (the original
would not have finished the large years at all due to the per-user scan).

### Verification

Old vs new were run side-by-side on real Q4-2023 data (`carry=None`):

- **Exact match** (allclose, 1e-10): `n_posts`, `whale_dominance`,
  `minnow_dominance`, `crowding_gini`, `fresh_blood_count/ratio_{21,63}`,
  `retention_rate_{21,63}`, and the full carry-state dict.
- **Changed only where intended**: re-entry (fix 2), specialist (fix 3),
  night-owl/day-trader (fix 4).
- `specialist_ratio` additionally verified against an independent unit-safe
  brute-force reference across 5,224 symbol-days: exact match.
- Full-2023 output: all ratio features bounded [0,1].

---

## Practical consequences

- Any `features_05_sentiment_dynamics_cohorts.pkl` generated with the original
  code must be **regenerated**: `conviction_index_*` (multi-year runs),
  `reentry_volume_*`, `specialist_ratio`, `night_owl_ratio`, `day_trader_ratio`
  are all affected. Downstream artifacts (`features_master.pkl`,
  `merged_master.pkl`, all-features `predictions_*.pkl`) inherit the errors.
- The full run is now fast (~minutes for the cohort family).

## Reviewed and deliberately kept as-is (project decision)

Deviations from `feature_proposal.md` noted during review and retained:

| Where | Notebook does | Proposal says |
|---|---|---|
| Flip ratios (8–9) | divide by tagged users that day | "Total Active Users" |
| Conviction (10) | `streak >= K` | ">K consecutive tweets" |
| First-mover (16) | first N posts | "first N users" |
| Fresh-blood ratio (18) | divide by unique users | "Total Volume" |
| Fresh-blood horizons (17) | H ∈ {21, 63} | H ∈ {21, 250, AllTime} |
