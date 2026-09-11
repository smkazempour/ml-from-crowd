"""Count, per trading year, how many raw StockTwits messages WITHOUT a Bullish/Bearish tag would
enter the sample if the sentiment filter in data_cleaning.ipynb were dropped.

Reads the 248 raw metadata files (feature_wo_messages/), assigns each message its trading date
with the same rule as data_cleaning.ipynb (first CRSP close after posting; after 16:00 ET on a
trading day -> next trading day), extracts cashtags from symbol_list, and matches
(ticker, trading date) against the CRSP stock-days in merged_master.pkl. Nothing is written
except per-file JSON checkpoints and a summary CSV under OUT.
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd

RAW = r"D:\StockTwits\Data\v1\data\csv\feature_wo_messages"
OUT = r"D:\StockTwits\Data\v1\data\csv\text_embeddings_mlcrowd\_run\untagged_scan"
MM = r"D:\StockTwits\Data\merged_master.pkl"
CHUNK = 2_000_000
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---- CRSP universe: (ticker, trading date) keys and the trading calendar, from merged_master ----
t0 = time.time()
mm = pd.read_pickle(MM)[["ticker", "date"]]
mm["date"] = pd.to_datetime(mm["date"])
tdays = np.sort(mm["date"].unique()).astype("datetime64[D]")
tickers = pd.Index(pd.unique(mm["ticker"].astype(str).str.upper()))
key = tickers.get_indexer(mm["ticker"].astype(str).str.upper()).astype(np.int64) * (1 << 32) \
    + mm["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
key_set = pd.Index(pd.unique(key))
del mm, key
FIRST_DAY = tdays[0]
log(f"CRSP keys: {len(key_set):,} stock-days, {len(tickers):,} tickers, {len(tdays)} trading days "
    f"{tdays[0]}..{tdays[-1]}  [{time.time() - t0:.0f}s]")


def process_file(path):
    stats = {}
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=["message_id", "created_at", "sentiment", "symbol_list"],
                             dtype={"sentiment": "string", "symbol_list": "string"},
                             chunksize=CHUNK, on_bad_lines="skip"):
        n_rows += len(chunk)
        ts = pd.to_datetime(chunk["created_at"], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
        ts = ts.dt.tz_convert("US/Eastern").dt.tz_localize(None)
        ok = ts.notna().to_numpy()
        cal = ts.to_numpy().astype("datetime64[D]")
        secs = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy()
        after = secs > 16 * 3600                       # strictly after the 16:00:00 close
        pos = np.searchsorted(tdays, cal, side="left")  # first trading day >= calendar day
        pos_c = np.minimum(pos, len(tdays) - 1)
        is_td = (pos < len(tdays)) & (tdays[pos_c] == cal)
        pos2 = np.where(is_td & after, pos + 1, pos)
        ok &= (pos2 < len(tdays)) & (cal >= FIRST_DAY)
        pos2 = np.minimum(pos2, len(tdays) - 1)
        tdate = tdays[pos2]
        year = tdate.astype("datetime64[Y]").astype(int) + 1970

        tagged = chunk["sentiment"].isin(["Bullish", "Bearish"]).to_numpy()
        syms = chunk["symbol_list"].fillna("").str.findall(r"'([^']*)'")
        n_sym = syms.str.len().fillna(0).to_numpy().astype(int)
        has_tag = n_sym > 0

        # explode the cashtags of valid messages and match (ticker, trading date)
        rows = np.repeat(np.arange(len(chunk)), n_sym)
        flat = np.array([s for lst in syms for s in lst], dtype=object) if n_sym.sum() else np.array([], dtype=object)
        flat = pd.Index(pd.Series(flat, dtype="string").str.upper().astype(object))
        tk = tickers.get_indexer(flat).astype(np.int64)
        pair_key = tk * (1 << 32) + tdate[rows].astype(np.int64)
        matched_pair = (tk >= 0) & (key_set.get_indexer(pair_key) >= 0) & ok[rows]
        msg_matched = np.zeros(len(chunk), dtype=bool)
        np.logical_or.at(msg_matched, rows[matched_pair], True)
        pairs_per_msg = np.bincount(rows[matched_pair], minlength=len(chunk))

        df = pd.DataFrame({"year": np.where(ok, year, -1), "tagged": tagged, "ok": ok,
                           "has_tag": has_tag & ok, "matched": msg_matched & ok, "pairs": pairs_per_msg})
        g = df.groupby(["year", "tagged"]).agg(n_msgs=("ok", "size"), n_valid_ts=("ok", "sum"),
                                                n_with_cashtag=("has_tag", "sum"),
                                                n_matched=("matched", "sum"), n_pairs=("pairs", "sum"))
        for (y, t), r in g.iterrows():
            k = f"{int(y)}|{int(t)}"
            acc = stats.setdefault(k, dict(n_msgs=0, n_valid_ts=0, n_with_cashtag=0, n_matched=0, n_pairs=0))
            for c in acc:
                acc[c] += int(r[c])
    return stats, n_rows


files = sorted(glob.glob(os.path.join(RAW, "feature_wo_messages_*.csv")))
log(f"{len(files)} raw files")
t_all = time.time()
for i, f in enumerate(files, 1):
    ck = os.path.join(OUT, os.path.basename(f) + ".json")
    if os.path.exists(ck):
        continue
    t0 = time.time()
    stats, n = process_file(f)
    json.dump({"rows": n, "stats": stats}, open(ck, "w"))
    log(f"{i}/{len(files)} {os.path.basename(f)}: {n:,} rows in {time.time() - t0:.0f}s "
        f"(elapsed {(time.time() - t_all) / 60:.1f} min)")

# ---- summary ----
tot = {}
rows = 0
for ck in glob.glob(os.path.join(OUT, "*.json")):
    d = json.load(open(ck))
    rows += d["rows"]
    for k, v in d["stats"].items():
        acc = tot.setdefault(k, dict(n_msgs=0, n_valid_ts=0, n_with_cashtag=0, n_matched=0, n_pairs=0))
        for c in acc:
            acc[c] += v[c]
recs = []
for k, v in tot.items():
    y, t = k.split("|")
    recs.append(dict(year=int(y), tagged=bool(int(t)), **v))
summ = pd.DataFrame(recs).sort_values(["year", "tagged"]).reset_index(drop=True)
summ.to_csv(os.path.join(OUT, "untagged_scan_summary.csv"), index=False)
log(f"raw rows scanned: {rows:,}")
piv = summ[summ["year"] > 0].pivot_table(index="year", columns="tagged", values=["n_msgs", "n_matched", "n_pairs"], aggfunc="sum")
print(piv.to_string())
print()
tot_t = summ[(summ["year"] > 0) & summ["tagged"]].sum(numeric_only=True)
tot_u = summ[(summ["year"] > 0) & ~summ["tagged"]].sum(numeric_only=True)
print(f"TAGGED   2010-2023: messages {int(tot_t['n_msgs']):,}; with cashtag {int(tot_t['n_with_cashtag']):,}; "
      f"CRSP-matched {int(tot_t['n_matched']):,} messages / {int(tot_t['n_pairs']):,} message-symbol pairs")
print(f"UNTAGGED 2010-2023: messages {int(tot_u['n_msgs']):,}; with cashtag {int(tot_u['n_with_cashtag']):,}; "
      f"CRSP-matched {int(tot_u['n_matched']):,} messages / {int(tot_u['n_pairs']):,} message-symbol pairs")
log("done")
