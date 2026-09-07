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
  - Encoding (CPU, 24 threads): 658 msg/s on 2010, 532 msg/s on 2011. 2010 -> 8,718 stock-days,
    2011 -> 24,482; `embed_n` sums exactly to the number of (message, symbol) pairs with text
    (11,824 and 34,180, i.e. 292 and 1,117 multi-symbol messages fanned out); mean-vector norms
    in [0.73, 1.0] with single-message rows at 1.0; brute-force recomputation of nine stock-days
    matches to 1e-5; per-year resume skips finished years; Section 10 refuses to save while
    `TEST_YEARS_ONLY` is set.
  - Extrapolation: the universe has ~85M (message, symbol) rows, on the order of 75M unique
    messages; at ~550 msg/s the full encode is ~1.5-2 days of CPU time, plus ~11 min for the
    join pass and the one-off universe build.

## 7. Not done / to do

- The full 15-year `features_08` run has not been executed. Section 7 of the notebook prints
  the extrapolated cost from the measured rate before Section 8 is started.
- After the full run: `add_text_features.ipynb` in the chosen mode(s), then the model
  notebooks with `TEXT_VARIANT` set, then the 04/05 notebooks with the same setting.
- `features_05` is still not part of `features_master.pkl` (unchanged from before this
  branch).
- The 03e LASSO prediction files are absent on this machine, so `form_portfolios` prints a
  warning and skips those models (as on `main`).
