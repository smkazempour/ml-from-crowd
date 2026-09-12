"""Evaluate prediction files on a common sample, for any target column.

For every model: mean daily Spearman rank correlation between the prediction and the
date-de-meaned realised target (with a t-statistic from the daily series), the equal-weighted
top-minus-bottom prediction-decile spread in bp/day, and the same by year.

Usage:
    python tools/evaluate_predictions.py --target f_cumret1 \
        --models "lr_2=predictions_linear_regression_input=2.pkl,text=predictions_ridge_textonly_rank_input=386.pkl" \
        [--start 2012-01-01] [--end 2022-12-31] [--tweeted-only 1] [--out reports/data/eval.csv]

The sample is: rows of merged_master with log_volume > 0 (unless --tweeted-only 0), inside the
date range, with a non-missing target and a prediction from EVERY listed model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\StockTwits\Data")


def daily_rank_corr(df, key, ry, g):
    rx = g[key].rank(pct=True)
    tmp = pd.DataFrame({"date": df["date"], "x": rx, "y": ry, "xy": rx * ry, "x2": rx ** 2, "y2": ry ** 2})
    s = tmp.groupby("date").agg(n=("x", "size"), x=("x", "sum"), y=("y", "sum"), xy=("xy", "sum"),
                                x2=("x2", "sum"), y2=("y2", "sum"))
    cov = s["xy"] / s["n"] - (s["x"] / s["n"]) * (s["y"] / s["n"])
    vx = s["x2"] / s["n"] - (s["x"] / s["n"]) ** 2
    vy = s["y2"] / s["n"] - (s["y"] / s["n"]) ** 2
    # a day on which the prediction is constant carries no ordering information: count it as 0,
    # not as missing (otherwise a model that predicts nothing on its worst days would look better)
    rc = pd.Series(np.where(vx > 0, cov / np.sqrt(np.where(vx > 0, vx, 1.0) * vy), 0.0), index=s.index)
    return rc.where(s["n"] >= 10)


def daily_decile_spread(df, key, g):
    dec = np.ceil(g[key].rank(method="first", pct=True) * 10).clip(1, 10).astype(int)
    m = df.assign(dec=dec).groupby(["date", "dec"])["y"].mean().unstack()
    return m[10] - m[1], m.mean() * 1e4


def tstat(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(len(s))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="f_cumret1")
    ap.add_argument("--models", required=True, help="comma-separated key=file (files under Data/)")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--tweeted-only", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    models = dict(kv.split("=", 1) for kv in a.models.split(","))

    df = pd.read_pickle(DATA / "merged_master.pkl")[["date", "permno", a.target, "log_volume"]].copy()
    for k, f in models.items():
        p = pd.read_pickle(DATA / f)
        df[k] = p.set_index("index")["prediction"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    if a.tweeted_only:
        df = df[df["log_volume"] > 0]
    df = df.dropna(subset=[a.target] + list(models)).reset_index(drop=True)
    df["y"] = df[a.target].astype(float) - df.groupby("date")[a.target].transform("mean").astype(float)
    g = df.groupby("date")
    ry = g["y"].rank(pct=True)
    print(f"target {a.target}; common sample {len(df):,} stock-days, {df['date'].nunique():,} days, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    rc = pd.DataFrame({k: daily_rank_corr(df, k, ry, g) for k in models})
    ls, prof = {}, {}
    for k in models:
        ls[k], prof[k] = daily_decile_spread(df, k, g)
    ls = pd.DataFrame(ls)
    summ = pd.DataFrame({"rank_corr": rc.mean(), "rank_corr_t": rc.apply(tstat),
                         "spread_bp": ls.mean() * 1e4, "spread_t": ls.apply(tstat),
                         "D1_bp": {k: prof[k][1] for k in models}, "D10_bp": {k: prof[k][10] for k in models},
                         "n_days": rc.notna().sum()})
    summ.index.name = "model"
    pd.set_option("display.width", 220)
    print("\nFull sample:")
    print(summ.round(4).to_string())
    yearly = rc.groupby(rc.index.year).mean()
    print("\nRank correlation by year:")
    print(yearly.round(4).to_string())
    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out)
        yearly.to_csv(out.with_name(out.stem + "_yearly.csv"))
        print(f"\nsaved {out} and {out.with_name(out.stem + '_yearly.csv')}")


if __name__ == "__main__":
    main()
