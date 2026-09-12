# Text-embedding track (features_06 / features_08) -- integration review notes

**Source**: branch `fix/04-05-pipeline-validation-rebased` (William Matheson, 4 commits,
July-August 2026), reviewed and integrated on `integrate-text-embeddings` in September 2026.
**Validated on**: real data on this machine -- 2010-2011 message text for the embedding
notebook, the full `merged_master.pkl` (16.8M rows) and both existing prediction files for
the 04/05 notebooks, and a synthetic fixture with a planted signal for the text-feature
builder.

This file records what was taken from the branch, what was dropped, what was corrected and
why, and what was measured. It complements `features_05_integration_fixes.md` for the
previous colleague branch.

---

## 1. What the branch contained, and what happened to it

| Branch content | Decision |
|---|---|
| `features_08_text_embedding_signal.ipynb` -- sentence-transformer embedding of message text, mean-pooled to stock-day | **Kept, reworked** as `features_08_text_embeddings.ipynb` (Section 3 below) |
| `features_06_full_text_exploration.ipynb` -- validates the `messages/` join, prototypes cashtag extraction | **Kept** (exploration only; paths fixed) |
| `features_07_user_influence_accuracy.ipynb` -- user-skill features | **Excluded** -- outside `feature_proposal.md`; rejected in the July 2026 merge and must not enter the repo |
| `perpare_training_data.ipynb`: fill block for `text_signal_mean/std/n` | **Dropped** -- those columns no longer exist (the branch's own last commit replaced them with 384 `embed_*` columns and never updated the wiring); text features are attached by the new `add_text_features.ipynb` instead |
| `03a` all-features: leakage filter `startswith('ret')` | **Dropped** -- `merged_master.pkl` has no `ret` column (it is not in `perpare_training_data`'s kept columns), so the filter fixed nothing, and it would silently remove `retention_rate_21/63` once features_05 is in the feature master |
| `03a`: `Parallel(...)` without `backend='threading'` | **Not adopted** -- untested claim; process-based workers multiply memory by `N_JOBS`; can be revisited with a benchmark |
| `05/form_portfolios`: LASSO models commented out of `MODELS` | **Dropped** -- motivated by missing files on the student's machine, not by correctness |
| `05/form_portfolios`: trim to key columns, avoid chained inner merges, guard `.mode()` | **Kept, reimplemented key-based** (Section 4) |
| `04/plot_oos_r2`: pd.NA guard, `duplicates='drop'` | **Kept with a different decile rule** (Section 4) |
| `04/portfolio_returns`: portfolios are date-indexed, not date-columned | **Kept** (verified against `trading_library.py`) |
| All 16 notebooks re-pathed to `C:\Users\willi\...`, `E:\Research_data\...`, `D:\OnlineData\Dropbox\CRSP` | **Dropped** -- reverts commit c374c12; every notebook keeps this machine's paths |
| README / status doc text describing the abandoned supervised-SGD design | **Replaced** |

## 2. Defects found in the branch's final `features_08` notebook

The notebook as committed had never been run end to end after its last rewrite:

1. **Syntax error.** The universe cell ends with a stray `fg` after the `glob()` call.
2. **Failing self-check.** Section 6 asserts `EMBED_DIM + 3` columns, but `process_year()`
   appends a `year` column, so the real count is `EMBED_DIM + 4`.
3. **Multi-ticker messages attributed to one symbol only.** The universe was built with
   `drop_duplicates(subset="message_id")`, keeping only the first `symbol` row of each message.
   Every other feature notebook counts a message toward every symbol it mentions (the
   `merged_with_crsp` rows are already exploded by symbol). On 2010 data 292 of 11,198
   messages (2.6%) mention more than one matched symbol; in later years the share is larger.
   The embedding aggregates would therefore have disagreed with the sentiment aggregates on
   exactly those stock-days.
4. **Raw-text reader.** The branch relied on pandas' CSV tokenizer: C engine first, python
   engine on `ParserError`. Measured here, the C engine aborts with "Buffer overflow caught"
   on `msg_000.csv` after 600k rows and on `msg_001.csv` after 200k rows -- i.e. essentially
   every file -- and the python engine is roughly 10x slower on a 52 GB corpus. Worse, an
   unterminated quote can make either engine silently swallow every record after it, which is
   a data-loss failure that raises nothing.
5. **Wrong assumption about file layout.** The notebook text says (correctly) that
   `messages/` is not year-chunked, but the smoke test assumed that a single early file would
   cover 2010. It does not: `msg_000.csv` runs from id 4 to id ~103M within its first 640k
   rows. A year's messages are spread over all 205 files, so the join pass must always scan
   the full set.
6. **Design drift not propagated.** The last commit replaced a walk-forward supervised model
   with plain mean-pooling (output: 384 `embed_XXX_mean` columns + `embed_n`), but the
   training-data fill block, README and status document still describe three
   `text_signal_*` columns. Had the pickle been produced, the fill block would have matched
   nothing and `03a`'s `dropna()` would have discarded every stock-day without messages.
7. **Output location.** The 384-column table was written into `features_mlcrowd/`, where
   `merge_all_feature_files.ipynb` globs every `*.pkl` and would have outer-merged it into
   `features_master.pkl` (28M rows x 384 float columns, ~43 GB at float32).

## 3. What `features_08_text_embeddings.ipynb` now does

- **Universe** (Section 2): one pass over `merged_with_crsp_mlcrowd/`. Per year it saves the
  full (message_id, symbol) pair table (`symbols_{year}.pkl`) and keeps one row per message
  for routing. It asserts that every message has exactly one trading date and that no
  message_id appears in two year files. Cached to `message_universe.pkl`.
- **Join pass** (Section 3): a line-based record reader (`iter_message_records`) replaces the
  CSV tokenizer. A record starts at every physical line beginning with an id of at least
  `MIN_ID_DIGITS` (5) digits followed by a comma; other lines continue the previous body.
  Nothing inside a body is interpreted, so stray quotes cannot derail the stream. Record
  counts per file are reported and can be checked with `grep -c '^[0-9]*,'`. Matches are
  appended to `joined_{year}.csv` once per message; the pass is checkpointed per source
  file and a re-run is a no-op.
- **Encode once, fan out on aggregation** (Sections 4-5): each message is encoded once;
  `aggregate_chunk()` repeats its vector for every symbol in the year's pair table and sums
  per (symbol, date). Sums are kept in float64 across chunks and divided by the count only
  at the end; partial sums are folded together whenever they exceed `REDUCE_ROWS`, so peak
  memory scales with stock-days, not messages. Optional per-message float16 vectors
  (`SAVE_MESSAGE_EMBEDDINGS`).
- **Validation** (Section 6) checks schema, that `embed_n` sums exactly to the number of
  (message, symbol) pairs with text, that mean vectors have norm in (0, 1] with
  single-message rows at norm 1, and recomputes several stock-days by brute force (always
  including one multi-symbol message).
- **Output**: `text_embeddings_mlcrowd/text_embeddings_stock_day.pkl` (outside
  `features_mlcrowd/`), columns `symbol, date, embed_n, embed_000..embed_383` (float32),
  plus `text_embeddings_meta.json`. Section 10 refuses to save a partial corpus.
- **Environment**: `sentence-transformers` 6.0.1 / `torch` 2.14 (CPU) were installed into
  the `py313` conda env (pandas, numpy, scikit-learn unchanged). The pip `torch` wheel and
  conda's MKL each bundle an OpenMP runtime; the notebook sets
  `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch, which is the documented workaround.
  No CUDA device is present on this machine.

## 4. Downstream: how the text reaches the models

The research decision (September 2026) was to extract all 384 dimensions once and defer
the choice between using them raw and compressing them. `02 - prepare training
dataset/add_text_features.ipynb` implements that choice with `TEXT_MODE`:

| mode | columns | fit on returns? |
|---|---|---|
| `raw` | `text_emb_000..383`, `text_n` | no |
| `pca` | `text_pc_1..K` (loadings fit on stock-days up to `PCA_FIT_END`, applied forward), `text_n` | no |
| `supervised` | `text_score` (monthly walk-forward ridge on the 384 dims, `SUP_WINDOW` = 252 trading days ending before the scored month, target cross-sectionally de-meaned), `text_n` | yes, walk-forward only |

It writes `merged_master_text=<mode>.pkl` next to `merged_master.pkl` (never modifying it),
aligning on (`ticker`, `date`) exactly as `perpare_training_data.ipynb` does. Missing
stock-days get 0 and `text_n = 0`.

Every `*_all_features` model notebook (03a, 03d x5, 03e x3) gained a `TEXT_VARIANT`
switch: `None` is byte-for-byte the old behaviour; a mode name reads the matching
`merged_master_text=` file and tags output files `_text=<mode>` so baseline predictions are
never overwritten. The four 04/05 notebooks that resolve "the all-features prediction file"
gained the same switch and a regex-based `find_all_features_file(model_type, text_variant)`
that ignores tagged files unless asked for them.

`raw` mode works but is not practical at full sample on this 64 GB machine (16.8M rows x
384 float32 = 26 GB before any model notebook copies the frame); `pca` and `supervised` are
the practical routes.

## 5. Other 04/05 changes (kept from the branch, reimplemented)

- **`05/form_portfolios`**: `df` is trimmed to (date, permno, ticker) before predictions are
  attached (the ~90 feature columns were never read; merging onto them was the MemoryError).
  Predictions are attached with a **left merge on (date, permno)**, not by row position:
  a model with a coverage gap contributes NaN for those days instead of deleting them from
  every other model, and a predictions file built from a different `merged_master` build
  cannot silently misalign. The per-date mode filter tolerates all-NaN dates and skips
  models whose file is absent.
- **`04/predictive_regressions`**: the same chained inner merge became a left merge.
- **`04/plot_oos_r2`**: empty groups (nullable `Float64` target -> `pd.NA`) are written as
  `np.nan` floats; a date with fewer than 10 valid predictions gets NaN deciles rather than
  crashing (`duplicates='drop'`, the branch's choice, would silently have produced fewer than
  10 bins).
- **`04/portfolio_returns`**: uses the date index that `trading_library` returns.
- **`02/merge_all_feature_files`**: skips any `*embedding*.pkl` by name as a second line of
  defence.

## 6. Measured

- **`05/form_portfolios` end to end** on `merged_master.pkl` + `predictions_linear_regression_input={2,53}.pkl`:
  the left merge yields 13,327,784 rows with both predictions, identical to the old inner
  merge's row count, and the prediction values agree exactly on those rows. The mode filter
  survives an injected all-NaN year. Portfolios, filters and the results pickle are produced
  (both series indexed by a 2,768-day `DatetimeIndex`). Neither existing prediction file has
  a coverage gap on this machine: the "lr_all has no 2021" gap the branch worked around was
  an artefact of the student's own run.
- **`04/plot_oos_r2`, `04/predictive_regressions`, `04/rank_correlation`, `04/portfolio_returns`**
  on the real data: the OOS sample loads with 12,047,638 predictions per model (3,171,331
  stock-days with messages after the no-tweet filter); monthly, yearly, full-sample and
  by-decile metrics and all plots run; with `lr_all` blanked for all of 2021 the 2021 metrics
  come out as NaN floats (no object-dtype column), no decile is assigned that year, and every
  downstream cell still runs. `portfolio_returns` reads the scratch results from the
  `form_portfolios` run above.
- **`add_text_features` fixture** (60 tickers x 1,040 days, 35% of stock-days with text, planted
  linear signal at R2 ~ 0.006): `raw` aligns through duplicate (ticker, date) keys with zero
  fill and no NaN; `pca` components are centred, orthogonal and variance-ordered on the fit
  sample; `supervised` scores start in the first month after `SUP_TRAIN_END_DATE`, correlate
  0.25 out of sample with the planted signal, and are unchanged when every target after
  June 2012 is replaced with noise (no look-ahead). `03a` with `TEXT_VARIANT="supervised"`
  picks up `text_score`/`text_n` and would write `..._input=55_text=supervised.pkl`.
- **`features_08` on 2010-2011** (universe restricted to those years: 44,261 messages), running
  the notebook's own cells with the output folder redirected:
  - Line reader vs. pandas' python engine on `msg_000.csv`: 3,124,620 records (equal to
    `grep -c -E '^[0-9]{5,},'`), read in 3 s vs. 9 s; bodies identical on 99.999% of common
    ids (the rest differ only in embedded `\r`, which `clean_text` collapses); the 1,186
    pandas-only ids are the 2008-09 messages with ids below 10,000, outside any universe year.
  - Full join pass over all 205 files: ~3.1 s per file (~2.5M records each; ~11 min for the
    whole 52 GB); checkpoint resume and no-op re-run verified. **100.0%** of the 2010 and 2011
    universe messages found their text (11,198 and 33,063), no duplicates, no empty bodies.
  - **Join pass at full scale (2026-09-07, first full run).** The ~3 s/file above was measured
    with the universe restricted to 2010-2011 (44k ids). With the full 77,025,793-message
    universe the same code took ~187 s per file (3.1 min measured on `msg_000.csv`; ~10 h for
    the pass): `chunk.join(universe, on="message_id")` goes through pandas' merge machinery,
    which re-factorizes all 77M universe keys on every 200k-record chunk. Replaced with
    `universe.index.get_indexer(...)` on the sorted unique index plus positional takes of
    `date`/`year`: 12 s per file for the whole `msg_000.csv` (3,124,620 records, 422,814
    matches), identical rows, values and dtypes to the old join (`assert_frame_equal`, also
    checked chunk by chunk). In the live run the pass proceeds at ~4-5 s per file.
  - **Full join pass result (2026-09-07):** 205 files in 14.3 min; 77,025,602 of the 77,025,793
    universe messages found their text (191 missing, 182 of them in 2020, 10 in 2017); every
    other year is complete. Exactly one `message_id` (10655083, a 2012 $AAPL message) appears
    twice: `msg_182.csv` holds a 2023 message whose body is a multi-line list of
    `<patent number>, <title>` lines, and the line reader takes each such line as a record
    start; one of those numbers equals the 2012 id. `load_year_messages` therefore no longer
    asserts zero duplicates: it keeps the longest body per id (the fragment is a tail of the
    other message) and still asserts that duplicates stay below max(10, 0.01%) of the year's
    rows, which a Section 3 re-run on a non-clean `JOINED_DIR` would exceed. Verified: 2012
    loads 70,594 unique ids with the $AAPL body. Scale of the reader heuristic: the 205 raw files contain
    501,435,038 record-start lines (`^[0-9]{5,12},`, matching the ~501M raw messages), of which
    497,722 (0.1%) are `<digits>, ` with a following space -- the shape of the false starts
    above (or of real bodies that begin with a space); exactly one of them collided with a
    universe id, so the join is unaffected beyond that one message.
  - Encoding (CPU, 24 threads): 658 msg/s on 2010, 532 msg/s on 2011. 2010 -> 8,718 stock-days,
    2011 -> 24,482; `embed_n` sums exactly to the number of (message, symbol) pairs with text
    (11,824 and 34,180, i.e. 292 and 1,117 multi-symbol messages fanned out); mean-vector norms
    in [0.73, 1.0] with single-message rows at 1.0; brute-force recomputation of nine stock-days
    matches to 1e-5; per-year resume skips finished years; Section 10 refuses to save while
    `TEST_YEARS_ONLY` is set.
  - Extrapolation: the universe has ~85M (message, symbol) rows, on the order of 75M unique
    messages; at ~550 msg/s the full encode is ~1.5-2 days of CPU time, plus ~11 min for the
    join pass and the one-off universe build.
- **Full 15-year run (2026-09-07 18:30 -> 2026-09-09 14:04, CPU, BelowNormal priority):** 43.6 h
  wall in total; encoding rates by year 370-624 msg/s (2021: 25,598,560 messages in 14.7 h at
  485 msg/s; 2023 slowest at 410 msg/s). Peak working set 12.2 GB (while 2021 was loaded).
  Output `text_embeddings_stock_day.pkl`: 3,505,815 stock-days x 387 columns, 5.2 GB on disk,
  8,245 symbols, 2010-06-02 to 2024-01-03, no nulls; `embed_n` median 2, mean 24.4, max 98,358.
  Stock-day counts per year equal an independent count of distinct (symbol, date) pairs from
  `symbols_{year}.pkl` + the universe dates, except 2017 (-1) and 2020 (-1), which are the
  stock-days whose only messages are among the 191 without text. Disk: joined text 8.2 GB,
  per-year checkpoints + final table ~10.4 GB, all under `text_embeddings_mlcrowd/`.

## 7. Not done / to do

- ~~The full 15-year `features_08` run~~ Done 2026-09-09 (Section 6). The per-year checkpoints
  and joined text are kept so a re-aggregation does not need the join pass again.
- After the full run: `add_text_features.ipynb` in the chosen mode(s), then the model
  notebooks with `TEXT_VARIANT` set, then the 04/05 notebooks with the same setting.
- `features_05` is still not part of `features_master.pkl` (unchanged from before this
  branch).
- The 03e LASSO prediction files are absent on this machine, so `form_portfolios` prints a
  warning and skips those models (as on `main`).

---

## 8. Text-only predictive regression (2026-09-11) and the untagged-message question

**Design decisions (user, 2026-09-10/11).** Use the raw 384 dimensions, no compression yet;
exclude the message count (`embed_n` is attention, not content); keep the baseline
walk-forward OLS design so the comparison with `lr_2` / `lr_all` is like for like. The
contributed `add_text_features.ipynb` (PCA / walk-forward-ridge builder, Section 4 above) was
removed and replaced by `02 - prepare training dataset/build_text_master.ipynb`, a plain
join of the embeddings with the panel's keys, target, abnormal returns and 53 features into
`Data/text_master.pkl` (3,514,785 rows x 478 columns; `mm_index` = `merged_master` row
label). The `TEXT_VARIANT` switches left in the model and 04/05 notebooks are inert.

**Sample facts established on the way** (see also the README, "Data Pipeline Overview"):
`merged_master` is the CRSP common-stock panel (share codes 10-12, NYSE/AMEX/NASDAQ, ~4,300
stocks per day, 15,532,693 stock-days 2010-2023, not the 16.8M the README used to say);
22.6% of its rows have at least one message; `f_cumret1` is the next trading day's CRSP
return (verified on 99.6% of rows). Both the features and the embeddings are built from the
same 85M-row message-symbol table, so the stock-days with `log_volume > 0` and the stock-days
with an embedding coincide (3,514,787 vs 3,514,785; message counts agree on 99.995%). The
cleaning filter keeps only messages tagged Bullish/Bearish (~35% of the raw 501M), so
nothing downstream, text included, sees untagged messages.

**Model.** `03a - linear regression/prediction_linear_regression_text_only.ipynb`: monthly
refit, 252-trading-day window, OLS on `embed_000..383`, predictions for every tweeted
stock-day 2012-2023 (3,480,379). Runs in ~4 min with 4 threads. Registered as `lr_text` in
`plot_oos_r2`, `predictive_regressions`, `rank_correlation` and `form_portfolios`; the three
04 notebooks gained `COMMON_SAMPLE = True` (rows lacking any model's prediction are dropped),
which with `lr_text` registered is the tweeted stock-days.

**Results** (common sample 3,171,329 tweeted stock-days, 2012-2022, target and predictions
de-meaned by date):

| | lr_2 (net sentiment + log volume) | lr_all (53 features) | lr_text (384 dims) |
|---|---|---|---|
| Full-sample OOS R2 | +0.000214 | -0.000896 | -0.001541 |
| Pooled slope of return on prediction (s.e., 2-way cluster) | 1.016 (0.109) | 0.337 (0.078) | 0.060 (0.023) |
| Mean daily Spearman rank correlation | 0.0228 | 0.0146 | 0.0076 |
| Top-minus-bottom decile, equal-weighted, bp/day (t) | 17.4 (6.8) | 24.2 (10.8) | 5.8 (3.4) |
| Share of days with positive decile spread | 0.58 | 0.61 | 0.54 |

Reading: the text-only predictions order stocks in the right direction (rank correlation,
decile spread and pooled slope all positive and significant) but are far too dispersed --
a slope of 0.06 means the predictions would have to be shrunk by a factor of ~16 to be
calibrated, and the OOS R2 is negative in every year. With 384 unshrunk coefficients refit
every month on 250-600k rows this is expected; the whole-sample in-sample R2 is only
0.00006. Yearly rank correlations are positive in 9 of 11 years (negative in 2020-2021).
The signal is weaker than net sentiment alone on the same stock-days.

**Untagged messages.** A scan of the 248 raw metadata files (501,442,290 rows; script `tools/untagged_message_scan.py`,
per-file checkpoints and summary CSV under `text_embeddings_mlcrowd/_run/untagged_scan/`, ~19 min) applied the cleaning notebook's
trading-date rule and the CRSP (ticker, date) match of `merged_master` to every message,
tagged or not. It reproduces the tagged universe (77,018,883 CRSP-matched tagged messages vs
77,025,793 in features_08; the difference is the CRSP-vs-Fama-French calendar). Untagged
messages that would enter the sample if the Bullish/Bearish filter were dropped:
**96,487,680 messages / 110,956,061 message-symbol pairs, 2010-2023**, i.e. 1.25x the tagged
corpus (per year: 2017 7.5M, 2018 8.6M, 2020 12.5M, 2021 20.8M, 2022 13.3M, 2023 8.1M untagged
vs 4.7M / 5.4M / 12.1M / 25.6M / 12.2M / 6.5M tagged). Of the 325M untagged raw messages,
146M carry a cashtag and 96M of those match a CRSP common stock on their trading date. Cost
of embedding them at the measured ~500 msg/s: ~54 h of CPU (about 2.2 days), plus one join
pass; a GPU would cut this to hours. The extension was agreed to run as a *parallel* track
(a second cleaned table with the filter off and a `tagged` flag, feeding only the embedding
notebook, giving all / tagged-only / untagged-only stock-day embeddings) so the features and
baselines stay untouched.

**Next steps.** (1) Shrinkage / compression of the 384 dimensions (ridge, or PCA fit on the
pre-2012 sample and applied forward) in the same walk-forward design -- the calibration
problem, not the ordering, is what the raw OLS loses. (2) "All features + text" on
`text_master` (the 53 features are in the table). (3) The untagged-message extension as a
parallel embedding track, if the counts above justify it.

### 8b. Ridge, date-de-meaned training and the agreement features (2026-09-11, later)

Per the user's decisions: ridge in the same walk-forward design, a switch for date-de-meaned
training (target and regressors de-meaned by date inside each window, i.e. date fixed effects;
regressors also de-meaned at prediction time), and two agreement measures computed in
`build_text_master` from the stored mean vector -- `embed_norm` (its length) and `embed_cos`
(the average pairwise cosine similarity among the day's messages implied by it,
`(n*||mean||^2 - 1)/(n - 1)`, set to 1 for single-message days, which are 38.9% of the table;
among multi-message days the mean is 0.45). The message count stays excluded.

`prediction_linear_regression_text_only.ipynb` now has three switches (`ESTIMATOR`,
`TARGET_DEMEAN`, `FEATURE_SET`; environment variables `TEXTONLY_*` override them so
`tools/run_notebook.py` can launch variants) and one closed-form code path: standardise inside
the window, one eigendecomposition of the Gram matrix, all candidate penalties from it, OLS =
`lambda = 0`. Ridge penalty `alpha = lambda * n_train`, grid 0 / 1e-3 ... 1e4; the `lambda` for
month m is the one with the lowest OOS squared error (both sides de-meaned by date, as the 04
notebooks score) over the previous 12 months, `lambda = 1` until any history exists. Each run
saves a `.json` sidecar with the per-month choice and the candidate-by-month error table. Each
variant runs in 3.5-5 min. Registered as `lr_text_n`, `lr_text_n_dm`, `ridge_text_n`,
`ridge_text_n_dm` in the 04 notebooks (the best one also in `form_portfolios`).

Results, common sample 3,171,329 tweeted stock-days, 2012-2022, de-meaned by date:

| model | regressors | training target | OOS R2 | slope (s.e.) | rank corr | D10-D1 bp/day (t) |
|---|---|---|---|---|---|---|
| lr_2 | net sentiment + log volume | raw | +0.000214 | 1.02 (0.11) | 0.0228 | 17.4 (6.8) |
| lr_all | 53 features | raw | -0.000896 | 0.34 (0.08) | 0.0146 | 24.2 (10.8) |
| lr_text | 384 dims, OLS | raw | -0.001541 | 0.06 (0.02) | 0.0076 | 5.8 (3.4) |
| lr_text_n | 384 + norm/cos, OLS | raw | -0.001551 | 0.10 (0.02) | 0.0126 | 11.6 (6.1) |
| lr_text_n_dm | 384 + norm/cos, OLS | date-de-meaned | -0.001027 | 0.15 (0.03) | 0.0151 | 10.6 (5.7) |
| ridge_text_n | 384 + norm/cos, ridge | raw | +0.000004 | 0.67 (0.26) | 0.0100 | 4.6 (2.4) |
| ridge_text_n_dm | 384 + norm/cos, ridge | date-de-meaned | +0.000018 | 0.87 (0.28) | 0.0161 | 9.8 (4.8) |

Reading. (1) The two agreement measures are the single biggest improvement: with OLS they
double the decile spread (5.8 -> 11.6 bp/day) and lift the rank correlation from 0.0076 to
0.0126 at no cost in calibration. (2) Date-de-meaned training helps every metric for both
estimators. (3) Ridge does what it was meant to do -- the pooled slope goes from 0.10-0.15 to
0.67-0.87 and the OOS R2 turns (barely) positive -- but the heavy shrinkage it selects
(lambda 10-100 in most months, occasionally 10,000, on the standardised columns) costs some
ordering on the raw target; combined with de-meaning it keeps the ordering (rank corr 0.016,
above `lr_all`'s 0.015 on the same stock-days, below `lr_2`'s 0.023) and is the
best-calibrated text model. (4) Every text model has negative rank correlation in 2020 and
2021 and its best year in 2022; `lr_2` stays positive throughout. (5) All levels are tiny:
the best text-only OOS R2 is 0.00002 versus 0.0002 for net sentiment + volume.

Next candidates: a milder shrinkage family (e.g. principal-component regression, or ridge
with the penalty chosen on rank correlation rather than squared error, since the ordering is
what carries the signal); the agreement measures on their own and interacted with net
sentiment; the untagged-message extension (Section 8).

### 8c. Rank target and Phase 0 diagnostics (2026-09-11, evening)

**Rank target.** `RANK_TARGET=1` (environment variable; switch added to the two baseline 03a
notebooks and the text-only notebook) replaces `f_cumret1` in training by its daily percentile
rank minus 0.5 -- among all CRSP stocks for the baselines, among tweeted stock-days for the text
table -- and tags the output `..._rank_...`. Same walk-forward design otherwise. Registered as
`lr_2_rank`, `lr_all_rank`, `lr_text_n_rank`, `lr_text_n_dm_rank`, `ridge_text_n_rank`,
`ridge_text_n_dm_rank` in `rank_correlation` and `predictive_regressions` (not in
`plot_oos_r2`: an R2 against raw returns is meaningless for a rank prediction). Common sample
as before (3,171,329 tweeted stock-days, 2012-2022, de-meaned by date):

| model | training target | rank corr | D10-D1 bp/day (t) | pooled t |
|---|---|---|---|---|
| lr_2 | return | 0.0228 | 17.4 (6.8) | 9.3 |
| lr_all | return | 0.0146 | 24.2 (10.8) | 4.3 |
| ridge_text_n_dm | return | 0.0161 | 9.8 (4.8) | 3.2 |
| lr_2_rank | rank | 0.0294 | 21.5 (7.5) | 7.3 |
| lr_all_rank | rank | 0.0295 | 23.6 (8.6) | 5.6 |
| lr_text_n_rank | rank | 0.0359 | 14.4 (5.5) | 4.9 |
| lr_text_n_dm_rank | rank | 0.0361 | 14.5 (5.5) | 4.9 |
| ridge_text_n_rank | rank | 0.0374 | 12.7 (4.9) | 3.7 |
| ridge_text_n_dm_rank | rank | 0.0374 | 12.9 (5.0) | 3.7 |

With the rank target the text models' rank correlation more than doubles (0.016 -> 0.037) and
exceeds both baselines (0.029), and it is positive in **every** year 2012-2022 (0.045-0.048 in
2020-2021, where the return-target text models were negative); the four text variants are
indistinguishable. The baselines keep the larger decile spread and pooled t: their information
is concentrated in the tails (the bottom decile), the text's is a broad ordering across the
whole cross-section. Squared-error training on raw returns was throwing the text's ordering
away; the rank target keeps it.

**Phase 0 diagnostics** (`tools`-free script, log in `_run/phase0.log`; same evaluation sample):

A. Single features, no model (daily Spearman with the de-meaned next-day return; D10-D1):
`log_volume` -0.030 (t -17), D10-D1 **-27 bp/day** (t -10); `unique_user_count` the same;
`net_sentiment` +0.010 (t 8.7), +7.6 bp; `abnormal_sentiment_1d` +0.012, +7.8 bp;
`disagreement_index` -0.017; `embed_norm` +0.025 (t 15), +10.6 bp; `embed_cos` +0.013 but a flat
decile spread (39% of days are single-message ties). So the dominant model-free signal among
tweeted stock-days is a **negative attention effect** -- yesterday's most-discussed stocks
underperform today -- and it is what `lr_2` is mostly made of; net sentiment alone is a third
of `lr_2`. `embed_norm` inherits part of the attention effect mechanically (norm falls with
the message count), which is why adding it doubled the OLS text spread.

B. Anatomy: 65% of tweeted stock-days are unanimously bullish (net sentiment = 1), mostly
single-message days, and earn +4.7 bp; the 14% of days with mostly-but-not-all bullish
messages ([0.5, 1)) earn **-20.6 bp** the next day, -40 bp when there are more than 30 messages;
unanimously bearish days -1.9 bp. Heavy bullish chatter predicts negative returns; the bearish
tail is small.

C. The tag inside the embedding: `net_sentiment` regressed on the 386 embedding columns (fit on
year t-1, applied to year t, no returns involved) has OOS R2 0.16-0.34 and correlation ~0.5 with
the tag. Used as a return predictor this "text stance" matches the tag's decile spread (7.2 vs
7.6 bp) but adds nothing once orthogonalised to the tag (2.6 bp, t 1.7). On tagged messages the
embedding's stance content is a noisy copy of the tag; the text's incremental information, per
8c above, is not stance but a broad ordering that the rank target extracts.

**Where this points.** (1) Keep the rank target for the text models. (2) The text signal and the
attention/tag signal are different objects; a model that combines the text ranking with volume
and net sentiment (on `text_master`, rank target) is the natural next test. (3) The attention
effect deserves its own look: it is the largest single predictor in the sample and is
"information in tweets" too. (4) Untagged messages: the text ordering does not depend on the
tag, so the 96M untagged messages are usable as-is once embedded.
