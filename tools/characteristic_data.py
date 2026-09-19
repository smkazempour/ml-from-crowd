"""Append close-t market controls to an immutable protocol-v1.1 prepared panel.

Only this derived cache is written. Social inputs, labels, row universe and splits
are preserved. Accounting characteristics are deliberately excluded: the legacy
daily book-to-market join uses fiscal-period dates without publication lags.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time

from filelock import FileLock
import numpy as np
import pandas as pd

from nn_checkpoint import atomic_json, fingerprint
from protocol_data import centered_rank, source_stat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / ".runs/protocol_v1_1/prepared/8f3f4eb44b399771"
DEFAULT_SOURCE = ROOT.parent / "Data/CRSP/dsf.pkl"
DEFAULT_OUT_ROOT = ROOT / ".runs/characteristics_v1/prepared"
PRESERVED = ("keys.pkl", "evaluation.pkl", "y.npy", "q.npy", "codes.npy", "calendar.npy")
PRIMITIVES = ("permno", "date", "ret", "prc", "shrout", "vol")
RECIPES = {
    **{f"char_ret{h}": f"product(1+ret[s], s=t-{h-1}..t)-1; {h} complete exchange sessions"
       for h in (1, 5, 21, 63, 126, 252)},
    "char_momentum252_skip21": "product(1+ret[s], s=t-251..t-21)-1; 231 complete exchange sessions",
    "char_log_market_cap": "ln(abs(prc[t])*shrout[t]/1000); positive price and shares; security cap in USD millions",
    "char_log_price": "ln(abs(prc[t])); strictly positive price",
    "char_turnover1": "vol[t]/(1000*shrout[t]); nonnegative volume, positive shares",
    "char_turnover21": "mean(vol[s]/(1000*shrout[s]), s=t-20..t); 21 complete sessions",
    "char_log_dollar_volume1": "ln(1+abs(prc[t])*vol[t]); positive price, nonnegative volume; USD",
    "char_log_dollar_volume21": "ln(1+mean(abs(prc[s])*vol[s], s=t-20..t)); 21 complete sessions; USD",
    "char_volatility21": "sample std(ret[s], s=t-20..t), ddof=1; 21 complete sessions",
    "char_volatility63": "sample std(ret[s], s=t-62..t), ddof=1; 63 complete sessions",
    "char_max_return21": "max(ret[s], s=t-20..t); 21 complete sessions",
    "char_amihud21": "mean(abs(ret[s])/(abs(prc[s])*vol[s]), s=t-20..t); 21 complete sessions with positive dollar volume",
}
CONTROL_NAMES = list(RECIPES)
MISSING_NAMES = [f"missing__{name}" for name in CONTROL_NAMES]
TIMING = {
    "signal": "first nominal 16:00 close after message; inherited parent convention",
    "market_controls": "observed through signal close t; no forward returns",
    "ret_validity": "finite ret >= -1 and an observed strictly positive price at the immediately preceding exchange close",
    "history": "calendar is union of dates in the full raw CRSP source; each stock is reindexed onto every exchange session in its observed span",
    "window_missingness": "require all constituent values; no forward-fill, zero-fill, or crossing a missing return/price observation",
    "zero_volume": "valid zero turnover and log1p dollar volume; unavailable Amihud denominator",
    "negative_volume": "unavailable, including CRSP negative sentinel codes",
    "ranks": "date-local observed ranks within all parent input rows, without conditioning on outcome availability; missing becomes zero",
    "missing_flags": "one unscaled binary flag for each of all 17 controls, even if no missing values occur",
    "universe": "exact parent social-data stock-day universe; no additional security filter",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def _cumulative_return(returns, window):
    """Complete-window products, preserving a gross return of zero (ret=-1)."""
    gross = returns + 1.0
    logs = np.log(gross.where(gross > 0, 1.0)).where(returns.notna())
    log_sum = logs.rolling(window, min_periods=window).sum()
    zeros = gross.eq(0).astype(float).where(returns.notna()).rolling(window, min_periods=window).sum()
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(log_sum)
    result = result.mask(zeros > 0, -1.0)
    return result.where(np.isfinite(result))


def stock_controls(frame, calendar):
    """Compute one stock's controls on a full exchange grid, indexed by date.

    This small pure function is also the executable definition tested below.
    It does not inspect outcomes or another stock's future values.
    """
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate stock dates")
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and increasing")
    frame = frame.sort_values("date").set_index("date")
    if not frame.index.isin(calendar).all():
        raise ValueError("Stock dates outside exchange calendar")
    calendar = calendar[(calendar >= frame.index.min()) & (calendar <= frame.index.max())]
    daily = frame.reindex(calendar)
    price = daily["prc"].astype(float).abs()
    price = price.where(np.isfinite(price) & (price > 0))
    shares = daily["shrout"].astype(float)
    shares = shares.where(np.isfinite(shares) & (shares > 0))
    volume = daily["vol"].astype(float)
    volume = volume.where(np.isfinite(volume) & (volume >= 0))
    returns = daily["ret"].astype(float)
    returns = returns.where(np.isfinite(returns) & (returns >= -1) & price.shift(1).notna())
    turnover = volume / (1000.0 * shares)
    dollars = price * volume
    with np.errstate(divide="ignore", invalid="ignore"):
        result = {f"char_ret{h}": _cumulative_return(returns, h)
                  for h in (1, 5, 21, 63, 126, 252)}
        result.update({
            "char_momentum252_skip21": _cumulative_return(returns, 231).shift(21),
            "char_log_market_cap": np.log(price * shares / 1000),
            "char_log_price": np.log(price),
            "char_turnover1": turnover,
            "char_turnover21": turnover.rolling(21, min_periods=21).mean(),
            "char_log_dollar_volume1": np.log1p(dollars),
            "char_log_dollar_volume21": np.log1p(dollars.rolling(21, min_periods=21).mean()),
            "char_volatility21": returns.rolling(21, min_periods=21).std(ddof=1),
            "char_volatility63": returns.rolling(63, min_periods=63).std(ddof=1),
            "char_max_return21": returns.rolling(21, min_periods=21).max(),
            "char_amihud21": (returns.abs() / dollars.where(dollars > 0)).rolling(21, min_periods=21).mean(),
        })
    return pd.DataFrame(result, index=calendar).replace([np.inf, -np.inf], np.nan)


def feature_schema(parent_manifest):
    """Keep old indices intact; always add all 17 missingness flags."""
    names = list(parent_manifest["feature_names"])
    if len(set(names)) != len(names) or set(names) & set(CONTROL_NAMES + MISSING_NAMES):
        raise ValueError("Feature-name collision")
    offset = len(names)
    names += CONTROL_NAMES + MISSING_NAMES
    sets = copy.deepcopy(parent_manifest["feature_sets"])
    controls = list(range(offset, len(names)))
    sets["characteristics"] = controls
    for suffix, column in (("sentiment", "net_sentiment"), ("attention", "log_volume")):
        if column not in names[:offset]:
            raise ValueError(f"Missing parent social feature: {column}")
        selected = [names.index(column)]
        missing_name = f"missing__{column}"
        if missing_name in names[:offset]:
            selected.append(names.index(missing_name))
        sets[f"characteristics_{suffix}"] = sorted(set(selected + controls))
    for suffix in ("core", "all", "textcore", "textall"):
        sets[f"characteristics_{suffix}"] = sorted(set(sets[suffix] + controls))
    return names, sets


def append_transformed(parent_X, raw, codes, destination):
    """Copy social inputs exactly and append observed-date ranks and flags."""
    n, original_p = parent_X.shape
    if raw.shape != (n, len(CONTROL_NAMES)) or len(codes) != n:
        raise ValueError("Characteristic shape does not match parent")
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=parent_X.dtype,
                                     shape=(n, original_p + 2 * len(CONTROL_NAMES)))
    for start in range(0, n, 32768):
        stop = min(start + 32768, n)
        output[start:stop, :original_p] = parent_X[start:stop]
        if not np.array_equal(output[start:stop, :original_p].view(np.uint8),
                              parent_X[start:stop].view(np.uint8)):
            raise ValueError("Parent feature values changed during append")
    for j in range(len(CONTROL_NAMES)):
        values = np.asarray(raw[:, j], dtype=np.float64)
        output[:, original_p + j] = centered_rank(values, codes)
        output[:, original_p + len(CONTROL_NAMES) + j] = ~np.isfinite(values)
    output.flush()
    del output


def _copy_preserved(parent, directory, records):
    for name, record in records.items():
        destination = directory / name
        if destination.exists():
            if file_record(destination) != record:
                raise ValueError(f"Existing copied artifact differs: {destination}")
        else:
            shutil.copy2(parent / name, destination)
            if file_record(destination) != record:
                raise ValueError(f"Copied artifact differs: {destination}")
        destination.chmod(stat.S_IREAD)


def prepare(parent=DEFAULT_PARENT, source=DEFAULT_SOURCE, out_root=DEFAULT_OUT_ROOT):
    parent, source, out_root = map(lambda p: Path(p).resolve(), (parent, source, out_root))
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    source_before = source_stat(source)
    print("Hashing parent and primitive CRSP sources", flush=True)
    preserved = {name: file_record(parent / name) for name in PRESERVED}
    if (parent / "horizon_repair_audit.json").exists():
        preserved["horizon_repair_audit.json"] = file_record(parent / "horizon_repair_audit.json")
    parent_manifest_hash = sha256_file(parent / "manifest.json")
    parent_X_hash = sha256_file(parent / "X.npy")
    source_hash = sha256_file(source)
    identity = {"kind": "market_characteristics_v1", "parent_fingerprint": parent_manifest["fingerprint"],
                "parent_manifest_sha256": parent_manifest_hash, "parent_X_sha256": parent_X_hash,
                "preserved_files": preserved, "primitive_source": {**source_before, "sha256": source_hash},
                "code_sha256": sha256_file(__file__),
                "kernel_code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                       for name in ("protocol_data.py", "nn_checkpoint.py")},
                "recipes": RECIPES, "timing": TIMING,
                "numpy": np.__version__, "pandas": pd.__version__}
    digest = fingerprint(identity)
    directory = out_root / digest[:16]
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock", timeout=0):
        if (directory / "manifest.json").exists():
            existing = json.loads((directory / "manifest.json").read_text())
            if existing["fingerprint"] != digest:
                raise ValueError("Prepared identity mismatch")
            for name, record in existing["artifact_records"].items():
                if file_record(directory / name) != record:
                    raise ValueError(f"Prepared artifact changed: {name}")
            print(f"Prepared: {directory}", flush=True)
            return directory

        keys = pd.read_pickle(parent / "keys.pkl")
        if keys.duplicated(["permno", "date"]).any() or keys[["permno", "date"]].isna().any().any():
            raise ValueError("Invalid parent input keys")
        codes = np.load(parent / "codes.npy", mmap_mode="r")
        parent_X = np.load(parent / "X.npy", mmap_mode="r")
        if len(keys) != len(codes) or len(keys) != len(parent_X):
            raise ValueError("Parent row counts do not agree")
        names, sets = feature_schema(parent_manifest)
        begin = time.monotonic()
        print("Loading raw CRSP (one pass, no source changes)", flush=True)
        frame = pd.read_pickle(source)
        if any(name not in frame for name in PRIMITIVES):
            raise ValueError("Primitive CRSP schema missing required columns")
        # The raw WRDS pickle stores Python datetime.date objects; annual
        # derived files store datetime64. Normalize before Timestamp comparisons.
        frame["date"] = pd.to_datetime(frame["date"])
        calendar = pd.DatetimeIndex(frame["date"].drop_duplicates()).sort_values()
        if calendar.hasnans or not pd.DatetimeIndex(keys["date"]).isin(calendar).all():
            raise ValueError("Raw exchange calendar does not cover parent input dates")
        earliest = pd.Timestamp(keys["date"].min()) - pd.Timedelta(days=730)
        keep = (frame["permno"].isin(keys["permno"].unique()) & frame["date"].ge(earliest)
                & frame["date"].le(keys["date"].max()))
        market = frame.loc[keep, list(PRIMITIVES)].copy()
        source_rows = len(frame)
        del frame, keep
        gc.collect()
        for name in ("ret", "prc", "shrout", "vol"):
            market[name] = market[name].to_numpy(dtype=np.float64, na_value=np.nan)
        market["date"] = pd.to_datetime(market["date"])
        market.sort_values(["permno", "date"], inplace=True, ignore_index=True)
        if market.duplicated(["permno", "date"]).any():
            raise ValueError("Duplicate primitive stock-date keys")
        audit = {"source_rows": source_rows, "selected_history_rows": len(market),
                 "calendar_first": str(calendar.min().date()), "calendar_last": str(calendar.max().date()),
                 "calendar_sessions": len(calendar), "history_first": str(market["date"].min().date()),
                 "history_last": str(market["date"].max().date()), "parent_rows": len(keys),
                 "parent_stock_count": int(keys["permno"].nunique()), "output_stock_count": 0,
                 "invalid_return_below_minus_one": int((market["ret"] < -1).sum()),
                 "return_minus_one": int((market["ret"] == -1).sum()),
                 "nonpositive_or_missing_price": int((~np.isfinite(market["prc"]) | market["prc"].eq(0)).sum()),
                 "nonpositive_or_missing_shares": int((~np.isfinite(market["shrout"]) | market["shrout"].le(0)).sum()),
                 "negative_volume": int(market["vol"].lt(0).sum()), "zero_volume": int(market["vol"].eq(0).sum()),
                 "missing_volume": int((~np.isfinite(market["vol"])).sum()), "absent_stock_session_rows": 0,
                 "parent_keys_absent_from_raw_history": 0}
        raw = np.lib.format.open_memmap(directory / "characteristics_raw.npy", mode="w+", dtype="float64",
                                       shape=(len(keys), len(CONTROL_NAMES)))
        raw[:] = np.nan
        input_groups = keys.groupby("permno", sort=False).indices
        represented = set()
        for number, (permno, history) in enumerate(market.groupby("permno", sort=False), 1):
            rows = input_groups[permno]
            controls = stock_controls(history, calendar)
            requested_dates = pd.DatetimeIndex(keys.iloc[rows]["date"])
            raw[rows] = controls.reindex(requested_dates)[CONTROL_NAMES].to_numpy()
            audit["absent_stock_session_rows"] += len(controls) - len(history)
            audit["parent_keys_absent_from_raw_history"] += int((~requested_dates.isin(history["date"])).sum())
            represented.add(permno)
            if number % 250 == 0:
                print(f"Controls: {number}/{len(input_groups)} stocks; {time.monotonic()-begin:.1f}s", flush=True)
        for permno in set(input_groups) - represented:
            audit["parent_keys_absent_from_raw_history"] += len(input_groups[permno])
        audit["output_stock_count"] = len(represented)
        raw.flush()
        del market
        gc.collect()
        append_transformed(parent_X, raw, codes, directory / "X.npy")
        missing_counts = {name: int((~np.isfinite(raw[:, j])).sum()) for j, name in enumerate(CONTROL_NAMES)}
        audit["missing_counts"] = missing_counts
        audit["coverage_by_year"] = {
            str(year): {name: int(np.isfinite(raw[rows, j]).sum()) for j, name in enumerate(CONTROL_NAMES)}
            for year, rows in keys.groupby(keys["date"].dt.year, sort=True).indices.items()}
        audit["feature_summary"] = {}
        for j, name in enumerate(CONTROL_NAMES):
            values = np.asarray(raw[:, j])
            observed = values[np.isfinite(values)]
            audit["feature_summary"][name] = ({"n": len(observed), "min": float(observed.min()),
                "p01": float(np.quantile(observed, .01)), "median": float(np.median(observed)),
                "p99": float(np.quantile(observed, .99)), "max": float(observed.max())} if len(observed) else {"n": 0})
        del raw, parent_X
        _copy_preserved(parent, directory, preserved)
        if (source_stat(source) != source_before
                or sha256_file(parent / "manifest.json") != parent_manifest_hash
                or sha256_file(parent / "X.npy") != parent_X_hash):
            raise ValueError("Source or parent manifest changed while preparing")
        audit["elapsed_seconds"] = time.monotonic() - begin
        audit["parent_feature_prefix_verified"] = True
        atomic_json(audit, directory / "characteristic_audit.json")
        artifact_records = {name: file_record(directory / name) for name in
                            [*preserved, "X.npy", "characteristics_raw.npy", "characteristic_audit.json"]}
        manifest = copy.deepcopy(parent_manifest)
        manifest.update({"protocol": "characteristics_v1", "fingerprint": digest,
                         "code_sha256": identity["code_sha256"], "feature_names": names,
                         "feature_sets": sets, "binary_features": parent_manifest.get("binary_features", []) + MISSING_NAMES,
                         "scalar_rank_features": parent_manifest.get("scalar_rank_features", []) + CONTROL_NAMES,
                         "missing_counts": {**parent_manifest.get("missing_counts", {}), **missing_counts},
                         "characteristics": identity, "artifact_records": artifact_records,
                         "derivation": {"kind": "append_market_characteristics", "parent_path": str(parent),
                            "parent_fingerprint": parent_manifest["fingerprint"], "parent_manifest_sha256": parent_manifest_hash,
                            "parent_feature_count": len(parent_manifest["feature_names"]), "parent_X_sha256": parent_X_hash,
                            "preserved_files": preserved, "parent_feature_prefix_verified": True},
                         "source_limitations": parent_manifest.get("source_limitations", []) + [
                            "Market-based controls only; no audited point-in-time accounting characteristics.",
                            "Legacy upstream daily bm joins fiscal datadate without reporting lag; excluded from predictors; inherited DGTW target has this limitation.",
                            "Close-t market inputs and nominal-close message routing inherit the executable-close and early-close limitations."]})
        atomic_json(manifest, directory / "manifest.json")
        for path in directory.iterdir():
            if path.is_file():
                path.chmod(stat.S_IREAD)
        print(f"Prepared: {directory}", flush=True)
        return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    arguments = parser.parse_args()
    prepare(arguments.parent, arguments.source, arguments.out_root)


if __name__ == "__main__":
    main()
