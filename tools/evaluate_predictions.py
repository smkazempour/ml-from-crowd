"""Common-sample evaluation with keyed joins, fractional ties and paired HAC tests.

Example: python tools/evaluate_predictions.py --models "core=file.pkl,text=other.pkl"
  --benchmark core --out reports/data/comparison.csv
Files are relative to --data-dir unless absolute. --panel text_master.pkl avoids
loading the untweeted panel. --periods accepts label:start:end blocks separated
by commas, within --start/--end. Long/short legs use date-demeaned returns.
"""
import argparse
import json
import hashlib
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prediction_metrics import align_predictions, bin_returns, hac_mean, rank_correlation, sharpe

DATA = Path(r"D:\StockTwits\Data")


def daily_rank_corr(df, key, ry=None, g=None):
    return rank_correlation(df, key, "y")


def daily_decile_spread(df, key, g=None):
    p, _ = bin_returns(df, key, "y")
    return p[10]-p[1], p.mean()*1e4


def tstat(s, lags=5):
    return hac_mean(s, lags)["t"]


def summarize(rc, profiles, benchmark, lags):
    rows = []
    comparisons = max(len(rc.columns)-1, 1)
    for key in rc:
        p = profiles[key]
        spread = p[10]-p[1]
        ic, ls = hac_mean(rc[key], lags), hac_mean(spread*1e4, lags)
        row = {"model": key, "rank_corr": ic["mean"], "rank_corr_t": ic["t"],
               "spread_bp": ls["mean"], "spread_t": ls["t"],
               "D1_bp": p[1].mean()*1e4, "D10_bp": p[10].mean()*1e4,
               "spread_sharpe": sharpe(spread), "long_sharpe": sharpe(p[10]),
               "short_sharpe": sharpe(-p[1]), "n_days": ic["n"], "n_spread_days": ls["n"]}
        if benchmark:
            bp = profiles[benchmark]
            for metric, delta in {"rank_corr": rc[key]-rc[benchmark],
                                  "spread_bp": (spread-(bp[10]-bp[1]))*1e4}.items():
                stats = hac_mean(delta, lags)
                row.update({f"delta_{metric}_{k}": v for k, v in stats.items()})
                # One comparison family per metric, target and requested period.
                row[f"delta_{metric}_p_bonferroni"] = min(1.0, stats["p"]*comparisons) if np.isfinite(stats["p"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def file_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--panel", default="merged_master.pkl")
    ap.add_argument("--target", default="f_cumret1")
    ap.add_argument("--models", required=True, help="comma-separated key=file")
    ap.add_argument("--benchmark")
    ap.add_argument("--hac-lags", type=int, default=5)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--periods", default="")
    ap.add_argument("--tweeted-only", type=int, choices=[0, 1], default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    pairs = [kv.split("=", 1) for kv in a.models.split(",")]
    models = dict(pairs)
    if len(models) != len(pairs) or a.hac_lags < 0:
        ap.error("Model keys must be unique; HAC lags must be nonnegative")
    if a.benchmark:
        if a.benchmark not in models:
            ap.error("--benchmark must be a listed model key")
        models = {a.benchmark: models[a.benchmark], **models}
    if {"date", "permno", a.target, "log_volume", "y"}.intersection(models):
        ap.error("Model keys collide with panel columns")
    panel_path = a.data_dir/a.panel
    df = pd.read_pickle(panel_path)[["date", "permno", a.target, "log_volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date.between(a.start, a.end)]
    if a.tweeted_only:
        df = df[df.log_volume > 0]
    sources = {}
    for key, file in models.items():
        path = a.data_dir/file
        df[key] = align_predictions(df, pd.read_pickle(path))
        sources[key] = file_info(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[a.target]+list(models)).reset_index(drop=True)
    if df.empty:
        raise ValueError("No common stock-days for these models, target and dates")
    df["y"] = df[a.target].astype(float)-df.groupby("date")[a.target].transform("mean").astype(float)
    print(f"target {a.target}; common sample {len(df):,} stock-days, {df.date.nunique():,} days, "
          f"{df.date.min().date()} to {df.date.max().date()}", flush=True)
    rc, profiles, daily = {}, {}, {}
    for key in models:
        rc[key] = daily_rank_corr(df, key)
        profiles[key], _ = bin_returns(df, key, "y")
        daily[(key, "rank_corr")] = rc[key]
        daily[(key, "long")] = profiles[key][10]
        daily[(key, "short_asset_return")] = profiles[key][1]
        daily[(key, "spread")] = profiles[key][10]-profiles[key][1]
        print(f"Scored {key}", flush=True)
    rc = pd.DataFrame(rc)
    summary = summarize(rc, profiles, a.benchmark, a.hac_lags)
    yearly = rc.groupby(rc.index.year).mean()
    print("\nFull sample (HAC t-statistics; legs are date-demeaned):")
    print(summary.round(5).to_string())
    periods = []
    for item in filter(None, a.periods.split(",")):
        label, start, end = item.split(":")
        subset = rc.loc[start:end]
        if subset.empty:
            raise ValueError(f"No dates in requested period {item}; check --start/--end")
        table = summarize(subset, {k: p.loc[start:end] for k, p in profiles.items()}, a.benchmark, a.hac_lags)
        periods.append(table.assign(period=label).reset_index())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(a.out)
        yearly.to_csv(a.out.with_name(a.out.stem+"_yearly.csv"))
        daily_table = pd.DataFrame(daily)
        daily_table.columns = [f"{k}|{metric}" for k, metric in daily_table.columns]
        daily_table.to_csv(a.out.with_name(a.out.stem+"_daily.csv"))
        if periods:
            pd.concat(periods).to_csv(a.out.with_name(a.out.stem+"_periods.csv"), index=False)
        metadata = {"target": a.target, "benchmark": a.benchmark, "hac_lags": a.hac_lags,
                    "inference": "two-sided normal, Bartlett HAC with small-sample correction",
                    "multiplicity": "Bonferroni over non-benchmark models, separately per metric/target/period",
                    "ties": "fractional rank intervals; constant prediction = cash; constant target = undefined IC",
                    "legs": "date-demeaned; Sharpe is descriptive, not a standalone investable portfolio",
                    "join": ["date", "permno"], "stock_days": len(df), "days": df.date.nunique(),
                    "date_range": [str(df.date.min().date()), str(df.date.max().date())],
                    "panel": file_info(panel_path), "models": sources, "created": pd.Timestamp.now().isoformat(),
                    "arguments": vars(a) | {"data_dir": str(a.data_dir), "out": str(a.out)},
                    "python": platform.python_version(), "interpreter": sys.executable,
                    "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ["evaluate_predictions.py", "prediction_metrics.py"]}}
        a.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {a.out} and daily/yearly/metadata files", flush=True)
    return summary


if __name__ == "__main__":
    main()
