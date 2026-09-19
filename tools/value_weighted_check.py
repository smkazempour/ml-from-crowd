"""Equal- versus value-weighted long-short returns of daily prediction sorts, on the tweeted
stock-days (common sample), without touching the 05 trading pipeline.

For each model and each day: sort the stocks with a prediction into quintiles (and deciles),
go long the top and short the bottom bin, hold for the next trading day (`f_cumret1`), with
equal weights and with market-cap weights (CRSP `cap` on the sort date). Reports mean daily
long-short return (bp), t-statistic, annualised Sharpe ratio, and the same by year and by
size tercile of the universe. No transaction costs.

Usage:
    python tools/value_weighted_check.py --models "text=file.pkl,lr_2=file.pkl" [--out reports/data/value_weighted.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prediction_metrics import align_predictions, bin_returns, hac_mean, sharpe

DATA = Path(r"D:\StockTwits\Data")
CRSP = DATA / "CRSP"


def ls_returns(df, key, bins, weight):
    """Daily long-short return: top bin minus bottom bin of `key`, weights 'ew' or 'cap'."""
    p, _ = bin_returns(df, key, "ret", bins=bins, weight=None if weight == "ew" else "cap", min_per_bin=10)
    return p[bins]-p[1]


def stats(x):
    x = x.dropna()
    return {"mean_bp": x.mean()*1e4, "t": hac_mean(x)["t"], "sharpe": sharpe(x), "n_days": len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = dict(kv.split("=", 1) for kv in a.models.split(","))

    df = pd.read_pickle(DATA / "merged_master.pkl")[["date", "permno", "f_cumret1", "log_volume"]].copy()
    for k, f in models.items():
        df[k] = align_predictions(df, pd.read_pickle(DATA / f))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= a.start) & (df["date"] <= a.end) & (df["log_volume"] > 0)]
    df = df.dropna(subset=["f_cumret1"] + list(models)).copy()
    df["ret"] = df["f_cumret1"].astype(float)
    caps = []
    for y in range(int(a.start[:4]), int(a.end[:4]) + 1):
        c = pd.read_pickle(CRSP / f"dsf_final_{y}.pkl")[["permno", "date", "cap"]]
        c["date"] = pd.to_datetime(c["date"]); caps.append(c)
    cap = pd.concat(caps, ignore_index=True).drop_duplicates(["permno", "date"])
    df = df.merge(cap, on=["permno", "date"], how="left")
    df = df[df["cap"].notna() & (df["cap"] > 0)].reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["size_terc"] = df.groupby("date")["cap"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["small", "mid", "large"]))
    print(f"common sample with cap: {len(df):,} tweeted stock-days, {df['date'].nunique():,} days")

    pd.set_option("display.width", 220)
    rows, yearly, bysize = [], {}, []
    for k in models:
        for bins in (5, 10):
            for w in ("ew", "cap"):
                r = ls_returns(df, k, bins, w)
                rows.append({"model": k, "bins": bins, "weight": w, **stats(r)})
                if bins == 10:
                    yearly[(k, w)] = r.groupby(r.index.year).mean() * 1e4
        for terc in ["small", "mid", "large"]:
            sub = df[df["size_terc"] == terc]
            for w in ("ew", "cap"):
                r = ls_returns(sub, k, 10, w)
                bysize.append({"model": k, "size": terc, "weight": w, **stats(r)})
    tab = pd.DataFrame(rows).set_index(["model", "bins", "weight"])
    print("\nLong-short return of daily prediction sorts (next-day, no costs):")
    print(tab.round(3).to_string())
    print("\nDecile long-short by year (bp/day), equal- vs cap-weighted:")
    yt = pd.DataFrame(yearly); yt.columns = [f"{k}|{w}" for k, w in yt.columns]
    print(yt.round(1).to_string())
    bs = pd.DataFrame(bysize).set_index(["model", "size", "weight"])
    print("\nDecile long-short within daily size terciles (sorts formed within the tercile):")
    print(bs.round(3).to_string())
    if a.out:
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# full sample\n"); tab.to_csv(fh); fh.write("\n# by year (decile, bp/day)\n"); yt.to_csv(fh)
            fh.write("\n# by size tercile (decile)\n"); bs.to_csv(fh)
        print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
