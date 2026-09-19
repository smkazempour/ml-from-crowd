"""Evaluate the registered protocol v1.1 models on common economic-key samples.

The primary contrast families are fixed before scoring: feature additions against
core, estimators against OLS at fixed inputs, and fitting histories against F504.
All registered models for a target share one prediction-coverage intersection.
Horizon and subgroup diagnostics are descriptive; no strategy costs or alpha are
estimated here. Daily series retain the complete supplied exchange calendar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from prediction_metrics import align_predictions


METRICS = ["rank_ic", "ew_spread_bp", "cap_spread_bp"]
DAY_COLUMNS = METRICS + ["ew_d1_raw_bp", "ew_d10_raw_bp", "ew_d1_demeaned_bp",
                        "ew_d10_demeaned_bp", "cap_d1_raw_bp", "cap_d10_raw_bp",
                        "cap_d1_demeaned_bp", "cap_d10_demeaned_bp", "n_stocks",
                        "n_cap_stocks", "constant_prediction"]
FEATURES = ("core", "all", "textcore", "textall")
ESTIMATORS = ("ols", "ridge", "lasso", "enet")


def calendar_hac_mean(series, calendar=None, lags=5):
    """Bartlett HAC for an observed-date mean, preserving missing session gaps.

    A missing observation contributes zero to the centered score, rather than
    moving two nonadjacent sessions next to each other. Normalize by the observed
    sample size and apply n/(n-1), matching the shared HAC on complete calendars.
    This conditions on the observed evaluation sample; it does not correct
    endogenous missing outcomes. Inference uses the two-sided normal reference.
    """
    if lags < 0:
        raise ValueError("HAC lags must be nonnegative")
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan)
    if s.index.has_duplicates:
        raise ValueError("HAC series must have unique calendar positions")
    if calendar is not None:
        calendar = pd.Index(calendar)
        if calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("Calendar must be unique and increasing")
        if not s.index.isin(calendar).all():
            raise ValueError("HAC observations fall outside the supplied calendar")
        s = s.reindex(calendar)
    else:
        s = s.sort_index()
    x = s.to_numpy()
    good = np.isfinite(x)
    n = int(good.sum())
    mean = float(x[good].mean()) if n else np.nan
    out = dict(mean=mean, se=np.nan, t=np.nan, p=np.nan, ci_low=np.nan,
               ci_high=np.nan, n=n, calendar_days=len(x), missing_days=len(x)-n,
               hac_lags=lags)
    if n < 2:
        return out
    u = np.where(good, x-mean, 0.0)
    meat = float(u@u)
    for k in range(1, min(lags, len(x)-1)+1):
        meat += 2*(1-k/(lags+1))*float(u[k:]@u[:-k])
    se = float(np.sqrt(max(meat, 0.0)/(n*(n-1))))
    out.update(se=se, ci_low=mean-norm.ppf(.975)*se,
               ci_high=mean+norm.ppf(.975)*se)
    if se > 0:
        t = mean/se
        out.update(t=t, p=float(2*norm.sf(abs(t))))
    elif mean == 0:
        out.update(t=0.0, p=1.0)
    return out


def score_day(predictions, target, cap=None, bins=10, min_ic=10, min_per_bin=10):
    """Vectorized equivalent of prediction_metrics' rank/fractional-tie rules.

    Every model receives the same finite-target sample. Predictions must already
    be jointly finite. Cap portfolios additionally require a positive finite cap.
    Constant forecasts hold cash in raw and demeaned legs. Counts refer to the
    fractional security membership, not an effective-cap-weight sample size.
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(target, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError("Predictions must be finite and match target rows")
    if bins < 2 or min_ic < 2 or min_per_bin < 1:
        raise ValueError("Invalid scoring minimum or bin count")
    valid = np.isfinite(y)
    p, y = p[valid], y[valid]
    m, n = p.shape[1], len(y)
    cap = (np.asarray(cap, dtype=float)[valid] if cap is not None
           else np.full(n, np.nan))
    result = {"rank_ic": np.full(m, np.nan), "n_stocks": n,
              "n_cap_stocks": int((np.isfinite(cap) & (cap > 0)).sum()),
              "constant_prediction": np.zeros(m, dtype=bool)}
    for name in ["ew_raw", "ew_demeaned", "cap_raw", "cap_demeaned"]:
        result[name] = np.full((m, bins), np.nan)
    if not n:
        return result
    left = rankdata(p, method="min", axis=0)-1
    right = rankdata(p, method="max", axis=0)
    xr = (left+right+1)/2
    xr -= xr.mean(axis=0)
    yr = rankdata(y)-((n+1)/2)
    xx, yy = (xr*xr).sum(axis=0), float(yr@yr)
    constant = xx == 0
    result["constant_prediction"] = constant
    if n >= min_ic and yy > 0:
        result["rank_ic"] = np.divide(xr.T@yr, np.sqrt(xx*yy),
                                      out=np.zeros(m), where=xx > 0)

    def portfolios(a, b, returns, weights, name):
        count = len(returns)
        if count < bins*min_per_bin:
            return
        output = np.empty((m, bins))
        for k in range(bins):
            frac = np.maximum(0, np.minimum(b, (k+1)*count/bins)
                              - np.maximum(a, k*count/bins))/(b-a)
            mass = frac*weights[:, None]
            output[:, k] = (mass*returns[:, None]).sum(axis=0)/mass.sum(axis=0)
        is_constant = np.all(a == a[0], axis=0)
        centered = output-returns.mean()
        output[is_constant] = 0.0
        centered[is_constant] = 0.0
        result[name+"_raw"] = output
        result[name+"_demeaned"] = centered

    portfolios(left, right, y, np.ones(n), "ew")
    positive = np.isfinite(cap) & (cap > 0)
    if positive.sum() >= bins*min_per_bin:
        # Missing capitalization changes the sorting universe as well as weights.
        cp = p[positive]
        portfolios(rankdata(cp, method="min", axis=0)-1,
                   rankdata(cp, method="max", axis=0), y[positive],
                   cap[positive], "cap")
    return result


def terciles(values):
    """Midrank terciles keep identical values together; missing values get -1."""
    v = np.asarray(values, dtype=float)
    good = np.isfinite(v)
    groups = np.full(len(v), -1, dtype=np.int8)
    if good.any():
        groups[good] = np.minimum(2, ((rankdata(v[good])-.5)*3/good.sum()).astype(int))
    return groups


def score_scope(predictions, target, cap, date_codes, calendar, models, mask=None):
    """Return complete-calendar daily rows and average raw/demeaned decile profiles."""
    m = len(models)
    selected = np.ones(len(target), dtype=bool) if mask is None else np.asarray(mask, bool)
    selected_rows = np.flatnonzero(selected)
    selected_rows = selected_rows[np.argsort(date_codes[selected_rows], kind="stable")]
    observed_codes, starts, sizes = np.unique(date_codes[selected_rows], return_index=True, return_counts=True)
    values = np.full((len(calendar), m, len(DAY_COLUMNS)), np.nan)
    sums = {k: np.zeros((m, 10)) for k in ["ew_raw", "ew_demeaned", "cap_raw", "cap_demeaned"]}
    counts = {k: np.zeros((m, 10), dtype=np.int64) for k in sums}
    for code, start, size in zip(observed_codes, starts, sizes):
        rows = selected_rows[start:start+size]
        day = score_day(predictions[rows], target[rows], cap[rows])
        ew, vw = day["ew_raw"]*1e4, day["cap_raw"]*1e4
        ed, vd = day["ew_demeaned"]*1e4, day["cap_demeaned"]*1e4
        values[code] = np.column_stack([
            day["rank_ic"], ew[:, -1]-ew[:, 0], vw[:, -1]-vw[:, 0],
            ew[:, 0], ew[:, -1], ed[:, 0], ed[:, -1],
            vw[:, 0], vw[:, -1], vd[:, 0], vd[:, -1],
            np.full(m, day["n_stocks"]), np.full(m, day["n_cap_stocks"]),
            day["constant_prediction"].astype(float)])
        for name in sums:
            good = np.isfinite(day[name])
            sums[name] += np.where(good, day[name], 0)
            counts[name] += good
    ids = [model["model_id"] for model in models]
    index = pd.MultiIndex.from_product([calendar, ids], names=["date", "model_id"])
    daily = pd.DataFrame(values.reshape(-1, len(DAY_COLUMNS)), index=index,
                         columns=DAY_COLUMNS).reset_index()
    profiles = []
    for name in sums:
        mean = np.divide(sums[name], counts[name], out=np.full((m, 10), np.nan), where=counts[name] > 0)
        for j, model in enumerate(models):
            for b in range(10):
                profiles.append(dict(model_id=model["model_id"], portfolio=name,
                                     decile=b+1, mean_bp=mean[j, b]*1e4,
                                     n_days=int(counts[name][j, b])))
    return daily, pd.DataFrame(profiles)


def contrast_specs(models):
    """Declared directional comparisons, counted even if a statistic is missing."""
    lookup = {(x["estimator"], x["feature_set"], int(x["fit_days"])): x["model_id"] for x in models}
    if len(lookup) != len(models):
        raise ValueError("Duplicate estimator/features/history specification")
    specs = []
    for x in models:
        estimator, feature, fit = x["estimator"], x["feature_set"], int(x["fit_days"])
        candidates = []
        if feature != "core":
            candidates.append(("feature", "versus_core", (estimator, "core", fit)))
        if feature == "textall":
            candidates.append(("feature", "text_given_all", (estimator, "all", fit)))
        if estimator != "ols":
            candidates.append(("estimator", "versus_ols", ("ols", feature, fit)))
        if estimator.startswith("nn"):
            for linear in ("ridge", "lasso", "enet"):
                candidates.append(("estimator", "nn_versus_regularized", (linear, feature, fit)))
        if fit != 504:
            candidates.append(("history", "versus_fit504", (estimator, feature, 504)))
        for family, comparison, reference in candidates:
            if reference in lookup:
                specs.append(dict(family=family, comparison=comparison,
                                  model_id=x["model_id"], benchmark_id=lookup[reference]))
    return specs


def summarize_daily(daily, models, lags=5):
    metadata = {x["model_id"]: x for x in models}
    rows = []
    for model_id, group in daily.groupby("model_id", sort=False):
        item = metadata[model_id]
        series = group.set_index("date")
        for metric in METRICS:
            stat = calendar_hac_mean(series[metric], lags=lags)
            rows.append({k: item[k] for k in ["model_id", "estimator", "feature_set", "fit_days"]}
                        | dict(metric=metric) | stat)
    return pd.DataFrame(rows)


def paired_contrasts(daily, models, lags=(5, 21, 63)):
    specs = contrast_specs(models)
    family_counts = {family: sum(s["family"] == family for s in specs)
                     for family in {s["family"] for s in specs}}
    rows = []
    for metric in METRICS:
        table = daily.pivot(index="date", columns="model_id", values=metric)
        for spec in specs:
            delta = table[spec["model_id"]]-table[spec["benchmark_id"]]
            count = family_counts[spec["family"]]
            for lag in lags:
                stats = calendar_hac_mean(delta, lags=lag)
                p = stats["p"]
                rows.append(spec | dict(metric=metric, family_size=count,
                                        p_bonferroni=min(1., p*count) if np.isfinite(p) else np.nan)
                            | stats)
    return pd.DataFrame(rows)


def _resolve(base, path):
    path = Path(path)
    return path if path.is_absolute() else (base/path).resolve()


def read_registry(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    models = doc["models"]
    if not models or len({x["model_id"] for x in models}) != len(models):
        raise ValueError("Registry must contain nonempty, unique model IDs")
    for model in models:
        for key in ["model_id", "estimator", "feature_set", "target", "fit_days", "predictions"]:
            if key not in model:
                raise ValueError(f"Registry model is missing {key}")
        model["predictions"] = str(_resolve(Path(path).resolve().parent, model["predictions"]))
    return doc, models


def _file_info(path):
    p = Path(path)
    stat = p.stat()
    return dict(path=str(p.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _write_table(out, suffix, tables):
    path = out.with_name(out.name+suffix+".csv")
    frame = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    frame.to_csv(path, index=False)
    return _file_info(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output filename prefix")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--cap-column", default="cap")
    parser.add_argument("--diagnostic-fit-days", type=int, default=504)
    parser.add_argument("--skip-horizons", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    args = parser.parse_args(argv)
    registry, all_models = read_registry(args.registry)
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    keys = pd.read_pickle(prepared/"keys.pkl")
    keys["date"] = pd.to_datetime(keys["date"])
    if keys[["date", "permno"]].isna().any().any() or keys.duplicated(["date", "permno"]).any():
        raise ValueError("Prepared economic keys must be nonmissing and unique")
    evaluation = pd.read_pickle(prepared/"evaluation.pkl")
    if len(evaluation) != len(keys) or not evaluation.index.equals(keys.index):
        raise ValueError("Evaluation table must share prepared key row order and index")
    y = np.load(prepared/"y.npy")
    targets = manifest["targets"]
    full_calendar = pd.DatetimeIndex(np.load(prepared/"calendar.npy"), name="date")
    calendar = full_calendar[(full_calendar >= pd.Timestamp(args.start))
                             & (full_calendar <= pd.Timestamp(args.end))]
    if not len(calendar) or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Evaluation calendar must be nonempty, unique and increasing")
    in_period = keys.date.between(calendar.min(), calendar.max()).to_numpy()
    positions = np.flatnonzero(in_period)
    panel = keys.iloc[positions].reset_index(drop=True)
    eval_panel = evaluation.iloc[positions].reset_index(drop=True)
    codes = calendar.get_indexer(panel.date)
    if (codes < 0).any():
        raise ValueError("Prepared keys include dates outside the exchange calendar")
    if args.cap_column not in eval_panel:
        raise ValueError(f"Evaluation table lacks capitalization column {args.cap_column}")
    cap = eval_panel[args.cap_column].to_numpy(dtype=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs = {name: [] for name in ["", "_daily", "_yearly", "_periods", "_contrasts",
                                     "_coverage", "_deciles", "_subgroups", "_horizons"]}
    sources, coverage_summary, skipped_horizons = {}, {}, []
    for target_name in dict.fromkeys(m["target"] for m in all_models):
        models = [m for m in all_models if m["target"] == target_name]
        target_column = models[0].get("target_column", {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}.get(target_name, target_name))
        if any(m.get("target_column", target_column) != target_column for m in models):
            raise ValueError("A target family cannot mix outcome columns")
        target = np.asarray(y[positions, targets.index(target_column)], dtype=float)
        # Preserve stored precision: reducing to float32 can create artificial ties.
        predictions = np.empty((len(panel), len(models)), dtype=np.float64)
        coverage = []
        for j, model in enumerate(models):
            path = Path(model["predictions"])
            frame = pd.read_pickle(path)
            predictions[:, j] = align_predictions(panel, frame).to_numpy(dtype=np.float64)
            observed = np.isfinite(predictions[:, j])
            sources[model["model_id"]] = _file_info(path)
            coverage.append(dict(target=target_name, model_id=model["model_id"],
                                 eligible_stock_days=len(panel), prediction_stock_days=int(observed.sum()),
                                 evaluated_stock_days=int((observed & np.isfinite(target)).sum()),
                                 missing_prediction_stock_days=int((~observed).sum())))
        common = np.isfinite(predictions).all(axis=1)
        if not common.any():
            raise ValueError(f"No common predictions for target {target_name}")
        coverage_summary[target_name] = dict(models=len(models), eligible_stock_days=len(panel),
                                             common_prediction_stock_days=int(common.sum()),
                                             common_scored_stock_days=int((common & np.isfinite(target)).sum()))
        for row in coverage:
            row.update(common_prediction_stock_days=int(common.sum()),
                       common_scored_stock_days=int((common & np.isfinite(target)).sum()))
        outputs["_coverage"].append(pd.DataFrame(coverage))
        # Copy once so every subsequent diagnostic can assume finite forecasts.
        p, t, c, d = predictions[common], target[common], cap[common], codes[common]
        print(f"Scoring {target_name}: {len(models)} models, {len(t):,} common predictions", flush=True)
        daily, deciles = score_scope(p, t, c, d, calendar, models)
        outputs["_daily"].append(daily.assign(target=target_name))
        outputs[""].append(summarize_daily(daily, models).assign(target=target_name))
        outputs["_deciles"].append(deciles.assign(target=target_name, scope="full", horizon=1))
        for year in sorted(daily.date.dt.year.unique()):
            subset = daily[daily.date.dt.year == year]
            outputs["_yearly"].append(summarize_daily(subset, models).assign(target=target_name, year=year))
        periods = [("full", args.start, args.end), ("2014-2018", "2014-01-01", "2018-12-31"),
                   ("2019-2022", "2019-01-01", "2022-12-31")]
        for label, first, last in periods:
            subset = daily[daily.date.between(first, last)]
            if not len(subset):
                continue
            outputs["_periods"].append(summarize_daily(subset, models).assign(target=target_name, period=label))
            outputs["_contrasts"].append(paired_contrasts(subset, models).assign(target=target_name, period=label))
        selected_models = [i for i, m in enumerate(models) if int(m["fit_days"]) == args.diagnostic_fit_days]
        diagnostic_models = [models[i] for i in selected_models]
        if not diagnostic_models:
            print(f"No F{args.diagnostic_fit_days} models; subgroup/horizon diagnostics skipped", flush=True)
            continue
        dp = p[:, selected_models]
        if not args.skip_subgroups:
            # Buckets use contemporaneous inputs before outcome filtering.
            for grouping, values in [("size", cap), ("activity", eval_panel.get("log_volume"))]:
                if values is None:
                    raise ValueError("Activity diagnostics require log_volume")
                values = np.asarray(values, dtype=float)[common]
                if grouping == "size":
                    values = np.where(values > 0, values, np.nan)
                bucket = np.full(len(values), -1, dtype=np.int8)
                ordered = np.argsort(d, kind="stable")
                _, starts, sizes = np.unique(d[ordered], return_index=True, return_counts=True)
                for start, size in zip(starts, sizes):
                    rows = ordered[start:start+size]
                    bucket[rows] = terciles(values[rows])
                for number in range(3):
                    label = f"{grouping}_tercile_{number+1}"
                    sd, sp = score_scope(dp, t, c, d, calendar, diagnostic_models, bucket == number)
                    outputs["_subgroups"].append(summarize_daily(sd, diagnostic_models).assign(
                        target=target_name, subgroup=label,
                        stock_days=int(((bucket == number) & np.isfinite(t)).sum())))
                    outputs["_deciles"].append(sp.assign(target=target_name, scope=label, horizon=1))
        if not args.skip_horizons:
            for horizon in [3, 5, 10, 21, 42, 63]:
                column = f"f_cumret{horizon}" if target_name == "raw" else f"ar_dgtw_{horizon}"
                if column not in eval_panel:
                    skipped_horizons.append(dict(target=target_name, horizon=horizon, reason="column unavailable"))
                    continue
                if target_name == "dgtw" and horizon == 21:
                    skipped_horizons.append(dict(target=target_name, horizon=horizon,
                                                 reason="Known source ar_dgtw_21 corruption; excluded pending repair"))
                    continue
                ht = eval_panel[column].to_numpy(dtype=float)[common]
                hd, hp = score_scope(dp, ht, c, d, calendar, diagnostic_models)
                outputs["_horizons"].append(summarize_daily(hd, diagnostic_models,
                                                           lags=max(5, horizon-1)).assign(
                    target=target_name, horizon=horizon, target_column=column,
                    stock_days=int(np.isfinite(ht).sum())))
                outputs["_deciles"].append(hp.assign(target=target_name, scope="full", horizon=horizon))
        print(f"Scored {target_name}; main comparisons and diagnostics complete", flush=True)
    written = {suffix or "summary": _write_table(args.out, suffix, tables)
               for suffix, tables in outputs.items()}
    metadata = dict(schema_version="protocol_v1_1_evaluation", created=pd.Timestamp.now().isoformat(),
                    registry=_file_info(args.registry), prepared=str(prepared),
                    prepared_manifest_sha256=hashlib.sha256((prepared/"manifest.json").read_bytes()).hexdigest(),
                    source_limitations=manifest.get("source_limitations", []),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    prediction_sources=sources, coverage=coverage_summary, output_files=written,
                    common_sample="Joint finite predictions across every registered model within each target; then observed outcome",
                    primary_metric="rank_ic", secondary_metrics=["ew_spread_bp", "cap_spread_bp"],
                    multiplicity="Bonferroni over all registered contrasts within target/family/metric/period; HAC lags are sensitivities, not selected. A partial registry is a partial experiment.",
                    contrast_families="Feature: noncore versus core plus textall versus all. Estimator: every non-OLS versus OLS, plus every NN versus each ridge/lasso/enet at fixed inputs/history. History: each non-F504 versus F504.",
                    inference="Two-sided normal Bartlett HAC, n/(n-1) correction, missing trading sessions retain actual calendar distance",
                    hac_lags=[5, 21, 63], min_ic_stocks=10, minimum_fractional_stocks_per_decile=10,
                    capitalization=args.cap_column, ties="Fractional rank intervals; constant forecasts hold cash",
                    legs="Raw realized target returns and target returns demeaned by that scope/day's unweighted eligible-stock mean; cash stays zero",
                    portfolio_limit="Descriptive gross sorts. No attainable same-close execution, turnover, costs, factor alpha, or investment claim.",
                    subgroups="F504 by default; contemporaneous midrank terciles before outcome filtering; equal values stay together",
                    horizons="F504 forecasts trained on h1 evaluated against cumulative close-t-to-close-(t+h) returns; not disjoint future-return intervals or h-specific fits",
                    skipped_horizons=skipped_horizons, python=platform.python_version(), interpreter=sys.executable,
                    code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ["protocol_evaluate.py", "prediction_metrics.py"]})
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved evaluation artifacts: {args.out}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
