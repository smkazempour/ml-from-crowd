"""Composition and horizon checks for prediction files on the tweeted stock-days.

Reads text_master.pkl (keys, embed_n, log_volume, target and DGTW abnormal returns at several
horizons), market cap from the CRSP files, and the listed prediction files; then reports, on
the common sample 2012-2022 with everything de-meaned by date:

A. mean daily rank correlation with the next-day return by message-count bucket (per model);
B. the same by size tercile (daily terciles of market cap);
C. the same by year x message-count bucket for the first listed model (does the rise over time
   come from coverage?);
D. horizon: rank correlation of each model's prediction with ar_dgtw_h, h in {1,3,5,10,21};
E. what the predictions load on: rank correlation of each model's daily ranks with the daily
   ranks of log_volume, embed_n, market cap, net_sentiment.

Usage:
    python tools/composition_check.py --models "text=predictions_ridge_textonly_rank_input=386.pkl,lr_2=..." [--out reports/data/composition.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\StockTwits\Data")
CRSP = DATA / "CRSP"
HORIZONS = [1, 3, 5, 10, 21]


def rank_corr_by(df, key, ycol, by=None):
    """Mean daily Spearman between key and ycol (both ranked within date), optionally within
    groups of `by` (a column with the group label; groups are formed within each date)."""
    keys = ["date"] + ([by] if by else [])
    df = df.dropna(subset=[key, ycol])          # a NaN target would otherwise distort the group sums
    g = df.groupby(keys)
    rx, ry = g[key].rank(pct=True), g[ycol].rank(pct=True)
    tmp = pd.DataFrame({"x": rx, "y": ry, "xy": rx * ry, "x2": rx ** 2, "y2": ry ** 2})
    for k in keys:
        tmp[k] = df[k].to_numpy()
    s = tmp.groupby(keys).agg(n=("x", "size"), x=("x", "sum"), y=("y", "sum"), xy=("xy", "sum"), x2=("x2", "sum"), y2=("y2", "sum"))
    cov = s["xy"] / s["n"] - (s["x"] / s["n"]) * (s["y"] / s["n"])
    vx = s["x2"] / s["n"] - (s["x"] / s["n"]) ** 2
    vy = s["y2"] / s["n"] - (s["y"] / s["n"]) ** 2
    # a day on which either side has no cross-sectional variance (e.g. a target that is constant
    # across stocks that day) carries no ordering information: leave it out (NaN), do not divide by 0
    ok = (s["n"] >= 10) & (vx > 0) & (vy > 0)
    rc = pd.Series(np.where(ok, cov / np.sqrt(np.where(ok, vx * vy, 1.0)), np.nan), index=s.index)
    if by:
        return rc.groupby(level=by).agg(["mean", "count"]).assign(t=lambda d: rc.groupby(level=by).mean() / rc.groupby(level=by).std() * np.sqrt(d["count"]))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = dict(kv.split("=", 1) for kv in a.models.split(","))

    cols = ["mm_index", "permno", "ticker", "date", "f_cumret1", "log_volume", "net_sentiment", "embed_n", "embed_cos"] + [f"ar_dgtw_{h}" for h in HORIZONS]
    tm = pd.read_pickle(DATA / "text_master.pkl")[cols].set_index("mm_index")
    tm["date"] = pd.to_datetime(tm["date"])
    tm = tm[(tm["date"] >= a.start) & (tm["date"] <= a.end)]
    for k, f in models.items():
        p = pd.read_pickle(DATA / f)
        tm[k] = p.set_index("index")["prediction"]
    tm = tm.dropna(subset=["f_cumret1"] + list(models)).copy()

    # market cap from CRSP (permno, date)
    caps = []
    for y in range(int(a.start[:4]), int(a.end[:4]) + 1):
        c = pd.read_pickle(CRSP / f"dsf_final_{y}.pkl")[["permno", "date", "cap"]]
        c["date"] = pd.to_datetime(c["date"]); caps.append(c)
    cap = pd.concat(caps, ignore_index=True).drop_duplicates(["permno", "date"])
    tm = tm.reset_index().merge(cap, on=["permno", "date"], how="left").set_index("mm_index")
    print(f"common sample: {len(tm):,} tweeted stock-days; cap missing for {tm['cap'].isna().mean():.1%}")

    for h in [None] + HORIZONS:
        col = "f_cumret1" if h is None else f"ar_dgtw_{h}"
        tm[f"y_{col}"] = tm[col].astype(float) - tm.groupby("date")[col].transform("mean").astype(float)
    tm["n_bucket"] = pd.cut(tm["embed_n"], [0, 1, 2, 5, 10, 30, 1e9], labels=["1", "2", "3-5", "6-10", "11-30", ">30"]).astype(str)
    tm["size_terc"] = tm.groupby("date")["cap"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["small", "mid", "large"]) if x.notna().sum() >= 30 else pd.Series(np.nan, index=x.index)).astype(str)
    tm["year"] = tm["date"].dt.year
    pd.set_option("display.width", 220)
    out = {}

    print("\nA. Rank correlation with next-day return by message-count bucket (mean of daily within-bucket correlations):")
    tabA = pd.DataFrame({k: rank_corr_by(tm, k, "y_f_cumret1", "n_bucket")["mean"] for k in models}).reindex(["1", "2", "3-5", "6-10", "11-30", ">30"])
    print(tabA.round(4).to_string()); out["A_by_msg_count"] = tabA

    print("\nB. Rank correlation with next-day return by daily size tercile:")
    sub = tm[tm["size_terc"] != "nan"]
    tabB = pd.DataFrame({k: rank_corr_by(sub, k, "y_f_cumret1", "size_terc")["mean"] for k in models}).reindex(["small", "mid", "large"])
    print(tabB.round(4).to_string()); out["B_by_size"] = tabB

    k0 = list(models)[0]
    print(f"\nC. {k0}: rank correlation by year x message-count bucket (within-bucket, within-date):")
    tm["yb"] = tm["year"].astype(str) + "|" + tm["n_bucket"]
    rc = rank_corr_by(tm, k0, "y_f_cumret1", "yb")["mean"]
    tabC = rc.to_frame("rc").assign(year=lambda d: d.index.str.split("|").str[0], b=lambda d: d.index.str.split("|").str[1]).pivot(index="year", columns="b", values="rc").reindex(columns=["1", "2", "3-5", "6-10", "11-30", ">30"])
    print(tabC.round(4).to_string()); out["C_year_x_msg_count"] = tabC
    print("share of stock-days per bucket by year:")
    print(pd.crosstab(tm["year"], tm["n_bucket"], normalize="index").reindex(columns=["1", "2", "3-5", "6-10", "11-30", ">30"]).round(3).to_string())

    print("\nD. Horizon: rank correlation of each prediction with the DGTW-adjusted return over h days (and raw next-day return):")
    tabD = pd.DataFrame({k: {("raw 1d"): rank_corr_by(tm, k, "y_f_cumret1").mean(), **{f"dgtw {h}d": rank_corr_by(tm, k, f"y_ar_dgtw_{h}").mean() for h in HORIZONS}} for k in models})
    print(tabD.round(4).to_string()); out["D_horizon"] = tabD

    print("\nE. What the predictions load on: mean daily rank correlation of the prediction with ...")
    tabE = pd.DataFrame({k: {c: rank_corr_by(tm.dropna(subset=[c]), k, c).mean() for c in ["log_volume", "embed_n", "cap", "net_sentiment", "embed_cos"]} for k in models})
    print(tabE.round(4).to_string()); out["E_loadings"] = tabE

    if a.out:
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for name, t in out.items():
                fh.write(f"# {name}\n"); t.to_csv(fh); fh.write("\n")
        print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
