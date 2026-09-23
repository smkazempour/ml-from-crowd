# Sample and transformation clarification

Audit date: September 22, 2026. This note corrects the interpretation of the
completed Report 11 study. Its certified forecasts, reports and source snapshots
are preserved as the record of what was run.

## The tests already use message-covered stock-days

The forecasting sample comes from `text_master.pkl`, rather than the full CRSP
stock-day panel. The notebook
[build_text_master.ipynb](../02%20-%20prepare%20training%20dataset/build_text_master.ipynb)
inner-joins daily text embeddings to the broader merged panel. Rows without a
matching daily embedding are excluded. An earlier preparation stage did use a
CRSP-left-joined panel with zero-filled social features; that is not the final
sample used by these experiments.

`protocol_data.prepare` takes its keys directly from the text panel and aligns
CRSP information to those keys. `characteristic_data` calculates controls using
broader CRSP histories, then appends them to the same existing keys. The use of a
full exchange calendar and full stock histories does not add no-message days to
the forecasting sample. Characteristic-only and social-augmented models share
this sample.

Direct inspection of the certified evaluation cache over 2014-2022 found:

- 3,034,035 prediction rows, 7,155 distinct PERMNOs and 2,266 formation dates.
- `embed_n` is observed and at least 1 in every row; no zero-message embedding rows.
- `log_volume` is observed in every row, with minimum `log(2)`; upstream code
  defines it as `log(1 + retained_message_count)`.
- 3,033,080 observed next-day raw-return outcomes and 2,748,978 DGTW outcomes.

More precisely, the upstream cleaning notebook retains Bullish/Bearish-tagged
messages, and the embedding notebook obtains its message universe from the
cleaned CRSP-matched message files. The current coverage and attention variables
therefore concern retained tagged messages, not every StockTwits post. Some
retained messages have empty text bodies encoded as empty strings; the positive
embedding count establishes a retained message, not substantive text content.

Time t is the close to which messages are assigned under the existing timing
convention. Message-covered days are defined by that assignment, not midnight-to-
midnight calendar posting dates. The inherited nominal-16:00/early-close limitation
is unchanged.

Conditioning on message coverage focuses the content comparison on days where a
social signal exists. It does not attribute the entire model fit or portfolio
spread to social information: characteristics can predict within this sample,
and sample inclusion itself conditions on social activity. The current matched
increments (C+sentiment/attention minus C, and adding text beyond those inputs)
measure added predictive information conditional on coverage. They do not test
whether the presence versus absence of a message predicts returns across all stocks.

## Correction: the characteristics were already daily ranks

The earlier recommendation and Report 11's transformation description mistakenly
treated the baseline characteristics as unranked values subjected only to fitted
standardization. The actual preparation code already converts all 17 continuous
characteristics into daily cross-sectional ranks:

- `tools/characteristic_data.py`, `append_transformed`: calls
  `centered_rank(values, codes)` before writing each characteristic into `X.npy`.
- `tools/protocol_data.py`, `centered_rank`: uses average ranks, maps them to
  `2*(rank-0.5)/observed_count - 1`, and replaces missing feature values with zero.
  Separate missingness flags preserve their missing status.
- `tools/linear_optimization_core.py`, `build_block`: both transformation branches
  read `X.npy`; the `daily_rank` branch applies a second rank transformation.

The cache also retains unranked controls in `characteristics_raw.npy`, but the
completed optimization study did not use that array as its model input.

Checks on complete cross-sections for January 2, 2014 (617 rows), January 2, 2019
(1,288 rows), and January 3, 2022 (2,106 rows) found exact equality between all 17
cached columns and ranks recomputed from the unranked controls. There were 185,
232 and 1,101 missing raw characteristic cells, respectively; all were already
stored as neutral zero in the corresponding cached columns.

The two implemented branches are therefore:

1. **`standard`:** existing daily midranks, neutral-zero missing values and
   separate missingness flags, followed by fitting-only standardization.
2. **`daily_rank`:** re-ranking those stored values using endpoint ranks
   `2*(rank-1)/(count-1)-1`, including the previously imputed zeros in the rank
   calculation, followed by fitting-only standardization.

The transformation comparison mixes a change of rank convention with a change
in the treatment of missing observations. It is **not** the intended comparison
of raw characteristic levels against daily ranks. Report 11 cannot establish
whether adopting daily ranks improves upon raw standardized characteristics.
The intended comparison remains outstanding and would require a separately
identified experiment using the saved unranked controls and explicit missingness
rules.

The recorded numerical results remain results for the procedures actually fitted.
Matched penalty, PCA and social-feature comparisons remain interpretable under
those actual transformations; they should not inherit the incorrect raw-versus-
rank label. The stock-day coverage finding above is unaffected. No models were
rerun or certified artifacts replaced as part of this audit.
