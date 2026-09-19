"""Shared data, calendar splits and transformations for protocol v1.1.

Run ``python tools/protocol_data.py --prepare`` once. Only an immutable prepared
cache under .runs is written; source panels and historical predictions are read-only.
The calendar is the union of full CRSP panel dates, not a repaired early-close map.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from filelock import FileLock
import numpy as np
import pandas as pd

from nn_checkpoint import atomic_json, atomic_pickle, fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Data"
PROTOCOL = "v1.1"
TARGETS = ["f_cumret1", "ar_dgtw_1"]
CORE = ["net_sentiment", "log_volume"]
HORIZONS = [1, 3, 5, 10, 21, 42, 63]
HORIZON_REPAIR_RULE = ("Set raw and DGTW h=1 labels missing when the next observed stock "
                       "row is not the next exchange session, provided the signal is "
                       "strictly before the prepared calendar's last date. Preserve "
                       "prediction rows; boundary labels remain explicitly unverified.")


def source_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def equal_date_weights(codes):
    """Positive weights summing to one, with each represented date equal weight."""
    codes = np.asarray(codes)
    if not len(codes):
        raise ValueError("No eligible observations")
    _, inverse, count = np.unique(codes, return_inverse=True, return_counts=True)
    return 1.0 / (len(count) * count[inverse])


def centered_rank(values, codes, *, target=False, min_count=10):
    """Date-local average ranks; unavailable outcomes cannot influence features.

    Scalar features use observed ranks in [-1,1], missing entries neutral zero.
    Targets use centered ranks in [-.5,.5]; dates with <10 labels or constant
    outcomes are unavailable for fitting, selection and IC scoring.
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    frame = pd.DataFrame({"value": values, "code": np.asarray(codes)})
    group = frame.groupby("code", sort=False)["value"]
    ranks = group.rank(method="average").to_numpy()
    counts = group.transform("count").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (ranks - .5) / counts - .5
    if target:
        distinct = group.transform("nunique").to_numpy()
        result[(counts < min_count) | (distinct < 2)] = np.nan
    else:
        result = np.nan_to_num(result * 2, nan=0.0)
    return result.astype(np.float32)


def feature_columns(bundle, feature_set):
    return list(bundle["manifest"]["feature_sets"][feature_set])


def load_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = {"path": str(path), "manifest": manifest}
    for name in ("X", "y", "q", "codes", "calendar"):
        result[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    result["keys"] = pd.read_pickle(path / "keys.pkl")
    n = len(result["keys"])
    if any(len(result[k]) != n for k in ("X", "y", "q", "codes")):
        raise ValueError("Prepared cache row counts do not agree")
    if np.any(np.diff(result["codes"]) < 0):
        raise ValueError("Prepared cache dates are not ordered")
    return result


def make_tasks(bundle, fit_days=504, validation_days=126, horizon=1,
               start="2014-01", end="2022-12", months=None):
    """Build full-month forecasts with explicit, strict label-end boundaries.

    Signal date s has a target ending on session s+h. Later publication times
    are not supplied by this source; the one-day targets assume close maturity.
    """
    if min(fit_days, validation_days, horizon) < 1:
        raise ValueError("Window lengths and horizon must be positive")
    calendar = pd.DatetimeIndex(np.asarray(bundle["calendar"]))
    codes = np.asarray(bundle["codes"])
    present = np.zeros(len(calendar), dtype=bool)
    present[np.unique(codes)] = True
    periods = calendar.to_period("M")
    requested = ([pd.Period(m, "M") for m in months] if months is not None
                 else list(pd.period_range(start, end, freq="M")))
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate requested test months")
    tasks = []
    for month in sorted(requested):
        test = np.flatnonzero(periods == month)
        if not len(test):
            raise ValueError(f"No calendar coverage for {month}")
        first, last = int(test[0]), int(test[-1])
        # Require adjacent months in the observed calendar to establish that the
        # test month has not simply been truncated at a cache boundary.
        if first == 0 or last == len(calendar) - 1:
            raise ValueError(f"Cannot establish full-month calendar coverage: {month}")
        b = first - 1
        valid_last = b - horizon
        valid_first = valid_last - validation_days + 1
        fit_last = valid_first - horizon - 1
        fit_first = fit_last - fit_days + 1
        if fit_first < 0:
            raise ValueError(f"Insufficient calendar history: {month}, F={fit_days}")
        blocks = {"fit": (fit_first, fit_last), "valid": (valid_first, valid_last),
                  "test": (first, last)}
        for name, (lo, hi) in blocks.items():
            if not present[lo:hi + 1].all():
                absent = calendar[lo:hi + 1][~present[lo:hi + 1]].strftime("%Y-%m-%d").tolist()
                raise ValueError(f"Missing {name} input dates for {month}: {absent[:8]}")
        task = {"month": str(month), "fit_days": int(fit_days),
                "validation_days": int(validation_days), "horizon": int(horizon),
                "cutoff": b}
        for name, (lo, hi) in blocks.items():
            task[f"{name}_first"], task[f"{name}_last"] = lo, hi
        tasks.append(task)
    return tasks


def _weighted_scaler(values, codes, binary=None, chunk_size=32768):
    """Fit weighted continuous-input moments with finite observations only.

    Missing unranked values are imputed to fitting weighted means. Each input
    date has equal initial weight; within a column its observed weights are
    renormalized. Scalar-rank missing entries have already become neutral zero.
    """
    weights = equal_date_weights(codes)
    p = values.shape[1]
    sums, second, mass = np.zeros(p), np.zeros(p), np.zeros(p)
    for first in range(0, len(values), chunk_size):
        x = np.asarray(values[first:first + chunk_size], dtype=np.float64)
        w = weights[first:first + chunk_size, None]
        finite = np.isfinite(x)
        x = np.where(finite, x, 0)
        sums += np.sum(w * x, axis=0)
        second += np.sum(w * x * x, axis=0)
        mass += np.sum(w * finite, axis=0)
    mean = np.divide(sums, mass, out=np.zeros(p), where=mass > 0)
    variance = np.divide(second, mass, out=np.zeros(p), where=mass > 0) - mean ** 2
    constant = (variance <= 1e-20) | (mass == 0)
    scale = np.sqrt(np.maximum(variance, 0))
    scale[constant] = 1
    # Missingness indicators retain their binary meaning, rather than scaling.
    if binary is not None:
        binary = np.asarray(binary, dtype=bool)
        mean[binary], scale[binary] = 0, 1
    return mean, scale, constant, mass == 0


def _transform(values, mean, scale, chunk_size=32768):
    result = np.empty(values.shape, dtype=np.float32)
    for first in range(0, len(values), chunk_size):
        x = np.asarray(values[first:first + chunk_size], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        result[first:first + chunk_size] = (x - mean) / scale
    return result


def make_block(bundle, task, feature_set, target):
    """Return identical transformed arrays and weights to any estimator family."""
    if task["horizon"] != 1:
        raise ValueError("Prepared fitting targets only support h=1; longer-horizon splits are testable")
    aliases = {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}
    target = aliases.get(target, target)
    target_index = bundle["manifest"]["targets"].index(target)
    columns = feature_columns(bundle, feature_set)
    names = [bundle["manifest"]["feature_names"][i] for i in columns]
    codes = np.asarray(bundle["codes"])
    selections = {}
    for part in ("fit", "valid", "test"):
        lo = int(np.searchsorted(codes, task[f"{part}_first"], side="left"))
        hi = int(np.searchsorted(codes, task[f"{part}_last"], side="right"))
        selections[part] = np.arange(lo, hi)
    full_columns = columns == list(range(bundle["X"].shape[1]))

    def get_values(indices):
        # The rows of each block are contiguous, so a full-feature block can be
        # read directly from mmap without a second multi-GB advanced-index copy.
        x = bundle["X"][indices[0]:indices[-1] + 1]
        return x if full_columns else x[:, columns]

    fit_indices = selections["fit"]
    binary_names = set(bundle["manifest"].get("binary_features", []))
    moments = _weighted_scaler(get_values(fit_indices), codes[fit_indices],
                               [name in binary_names for name in names])
    mean, scale, constant, all_missing = moments
    result = {"mean": mean, "scale": scale, "constant": constant,
              "all_missing": all_missing, "feature_names": names,
              "split": dict(task), "target": target}
    calendar = np.asarray(bundle["calendar"])
    for part, indices in selections.items():
        q = np.asarray(bundle["q"][indices, target_index])
        transformed = _transform(get_values(indices), mean, scale)
        if part != "test":
            eligible = np.isfinite(q)
            indices, q, transformed = indices[eligible], q[eligible], transformed[eligible]
            if not len(indices):
                raise ValueError(f"No eligible {part} labels: {task['month']} {target}")
            result[f"w_{part}"] = equal_date_weights(codes[indices])
        result[f"X_{part}"] = transformed
        result[f"y_{part}"] = q
        result[f"raw_{part}"] = np.asarray(bundle["y"][indices, target_index])
        result[f"codes_{part}"] = codes[indices]
        result[f"{part}_indices"] = indices
        result["split"][f"{part}_rows"] = len(indices)
        result["split"][f"{part}_labeled_days"] = int(np.unique(codes[indices][np.isfinite(q)]).size)
        result["split"][f"{part}_first_date"] = str(calendar[task[f"{part}_first"]])[:10]
        result["split"][f"{part}_last_date"] = str(calendar[task[f"{part}_last"]])[:10]
    return result


def _compact_crsp(data_dir, directory, files):
    """Read each large annual source once and retain a small reusable audit panel."""
    source_records = []
    for source in files:
        year = int(source.stem.rsplit("_", 1)[-1])
        dest = directory / f"crsp_{year}.pkl"
        if not dest.exists():
            begin = time.monotonic()
            frame = pd.read_pickle(source)
            if frame.duplicated(["date", "permno"]).any():
                raise ValueError(f"Duplicate CRSP keys: {source}")
            columns = ["date", "permno", "cap", "ret"] + [f"f_cumret{h}" for h in HORIZONS]
            compact = frame[columns].copy()
            compact["date"] = pd.to_datetime(compact["date"]).dt.normalize()
            for h in HORIZONS:
                compact[f"crsp_ar_dgtw_{h}"] = frame[f"f_cumret{h}"] - frame[f"f_dgtw_ret{h}"]
            del frame
            atomic_pickle(compact, dest)
            del compact
            gc.collect()
            print(f"CRSP {year} cached in {time.monotonic() - begin:.1f}s", flush=True)
        source_records.append(source_stat(source))
    frames = [pd.read_pickle(directory / f"crsp_{int(p.stem.rsplit('_', 1)[-1])}.pkl") for p in files]
    crsp = pd.concat(frames, ignore_index=True)
    del frames
    crsp.sort_values(["permno", "date"], inplace=True, ignore_index=True)
    calendar = np.sort(crsp["date"].unique()).astype("datetime64[D]")
    code = np.searchsorted(calendar, crsp["date"].to_numpy().astype("datetime64[D]"))
    group = crsp.groupby("permno", sort=False)
    previous_code = pd.Series(code).groupby(crsp["permno"], sort=False).shift(1).to_numpy()
    next_code = pd.Series(code).groupby(crsp["permno"], sort=False).shift(-1).to_numpy()
    crsp["lag_cap"] = group["cap"].shift(1)
    crsp.loc[previous_code != code - 1, "lag_cap"] = np.nan
    crsp["next_session_ret"] = group["ret"].shift(-1)
    crsp["next_observed_stock_date"] = group["date"].shift(-1)
    crsp.loc[next_code != code + 1, "next_session_ret"] = np.nan
    crsp["next_stock_row_is_next_session"] = next_code == code + 1
    return crsp, calendar, source_records


def repair_horizon_labels(y, evaluation, codes, calendar):
    """Repair writable in-memory labels/evaluation; never modify source artifacts.

    A signal on the last cached session cannot be checked against its successor.
    It is recorded as unverified, not automatically classified as a calendar gap.
    This rule concerns h=1 only; no claim is made about imported longer horizons.
    """
    codes = np.asarray(codes)
    known_next = evaluation["next_stock_row_is_next_session"].fillna(False).to_numpy(dtype=bool)
    interior = codes < len(calendar) - 1
    finite = np.isfinite(y)
    dates = np.asarray(calendar)[codes]
    comparable = known_next & np.isfinite(evaluation["next_session_ret"].to_numpy(dtype=float))

    def coverage():
        raw_finite = np.isfinite(y[:, 0])
        subsets = {"full": np.ones(len(y), dtype=bool), "primary_2014_2022":
                   (dates >= np.datetime64("2014-01-01")) & (dates <= np.datetime64("2022-12-31"))}
        return {name: {"rows": int(mask.sum()), "finite_raw_targets": int((mask & raw_finite).sum()),
                       "compared_next_session_raw_targets": int((mask & raw_finite & comparable).sum()),
                       "finite_raw_uncompared": int((mask & raw_finite & ~comparable).sum()),
                       "missing_raw_targets": int((mask & ~raw_finite).sum())}
                for name, mask in subsets.items()}
    bad = (~known_next) & interior & finite.any(axis=1)
    positions = np.flatnonzero(bad)
    audit = {"rule": HORIZON_REPAIR_RULE, "excluded_rows": len(positions), "coverage_before": coverage(),
             "changed_y_cells": np.argwhere(bad[:, None] & finite).tolist(),
             "changed_y_rows": positions.tolist(),
             "affected_date_codes": np.unique(codes[positions]).tolist(),
             "affected_dates": [str(d) for d in np.asarray(calendar)[np.unique(codes[positions])]],
             "boundary_unverified_finite_targets": {
                 name: int(((~known_next) & (~interior) & finite[:, j]).sum())
                 for j, name in enumerate(TARGETS)},
             "excluded_observations": [{"row": int(i), "mm_index": int(evaluation.index[i]),
                  "date": str(evaluation.iloc[i]["date"])[:10],
                  "permno": int(evaluation.iloc[i]["permno"]),
                  "ticker": str(evaluation.iloc[i]["ticker"]),
                  "original_targets": {name: float(y[i, j]) if np.isfinite(y[i, j]) else None
                                       for j, name in enumerate(TARGETS)}} for i in positions]}
    y[positions, :] = np.nan
    for name in TARGETS:
        evaluation.iloc[positions, evaluation.columns.get_loc(name)] = np.nan
    audit["coverage_after"] = coverage()
    return audit


def rerank_repaired_dates(y, q, codes, affected_codes):
    """Update only affected date-local rank targets, returning exact changed cells."""
    changed = []
    for code in affected_codes:
        positions = np.flatnonzero(np.asarray(codes) == code)
        for column in range(y.shape[1]):
            before = q[positions, column].copy()
            after = centered_rank(y[positions, column], np.asarray(codes)[positions], target=True)
            different = ~(np.equal(before, after) | (np.isnan(before) & np.isnan(after)))
            changed.extend([[int(row), column] for row in positions[different]])
            q[positions, column] = after
    return sorted(changed)


def known_gap_records(evaluation, codes, calendar, repair_rows, compact_directory):
    """Describe only the observed gaps identified by the finite h=1 audit."""
    records = []
    for position in repair_rows:
        row = evaluation.iloc[position]
        next_date = row.get("next_observed_stock_date", pd.NaT)
        if pd.isna(next_date):
            for year in range(row["date"].year, pd.Timestamp(calendar[-1]).year + 1):
                compact = pd.read_pickle(Path(compact_directory) / f"crsp_{year}.pkl")
                future = compact.loc[(compact["permno"] == row["permno"])
                                     & (compact["date"] > row["date"]), "date"]
                if len(future):
                    next_date = future.min()
                    break
        if pd.isna(next_date):
            continue
        next_code = int(np.searchsorted(calendar, np.datetime64(next_date, "D")))
        first_missing, last_missing = int(codes[position]) + 1, next_code - 1
        if first_missing <= last_missing:
            records.append({"permno": int(row["permno"]), "signal_date": str(row["date"])[:10],
                            "next_observed_stock_date": str(next_date)[:10],
                            "first_missing_code": first_missing, "last_missing_code": last_missing,
                            "first_missing_date": str(calendar[first_missing]),
                            "last_missing_date": str(calendar[last_missing]),
                            "missing_sessions": last_missing - first_missing + 1})
    return records


def exclude_known_gap_horizons(evaluation, codes, gaps):
    """Conservatively exclude imported raw/DGTW h>1 labels crossing known gaps.

    This bounded diagnostic repair is not a certification of other h>1 labels.
    Original crsp_ar_* copies remain explicitly named upstream audit values.
    """
    changes, counts = [], {}
    codes = np.asarray(codes)
    for horizon in HORIZONS[1:]:
        affected = np.zeros(len(evaluation), dtype=bool)
        for gap in gaps:
            affected |= ((evaluation["permno"].to_numpy() == gap["permno"])
                         & (codes < gap["last_missing_code"])
                         & (codes + horizon >= gap["first_missing_code"]))
        counts[str(horizon)] = {"intersecting_origins": int(affected.sum())}
        for name in (f"f_cumret{horizon}", f"ar_dgtw_{horizon}"):
            if name not in evaluation:
                continue
            positions = np.flatnonzero(affected & evaluation[name].notna().to_numpy())
            changes.extend([[int(row), name] for row in positions])
            evaluation.iloc[positions, evaluation.columns.get_loc(name)] = np.nan
            counts[str(horizon)][name] = len(positions)
    return {"scope": "raw/DGTW h>1 diagnostic labels whose nominal (t,t+h] interval intersects an identified missing-session gap",
            "gaps": gaps, "counts": counts, "changed_evaluation_cells": changes}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_prepared(parent_path, run_root=None):
    """Publish a new immutable prepared version without rerunning feature work.

    Invoke with --repair-prepared OLD_DIR, then pass the printed new directory as
    --prepared to model runners. Unchanged arrays are hardlinked (copy fallback);
    y, q, evaluation and manifests are new files. Existing sources remain untouched.
    A fresh --prepare applies the same rule before its initial target ranking.
    """
    parent_path = Path(parent_path).resolve()
    parent = load_bundle(parent_path)
    identity = {"kind": "prepared_horizon_repair", "parent_prepared": str(parent_path),
                "parent_fingerprint": parent["manifest"]["fingerprint"],
                "rule": HORIZON_REPAIR_RULE,
                "code_sha256": _sha256_file(__file__),
                "numpy": np.__version__, "pandas": pd.__version__}
    signature = fingerprint(identity)
    directory = Path(run_root or parent_path.parent) / signature[:16]
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock", timeout=0):
        if (directory / "manifest.json").exists():
            print(f"PREPARED {directory}", flush=True)
            return directory
        unchanged = ["X.npy", "keys.pkl", "codes.npy", "calendar.npy"]
        proof = {}
        for name in unchanged + sorted(p.name for p in parent_path.glob("crsp_*.pkl")):
            source, destination = parent_path / name, directory / name
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            if name in unchanged:
                before = _sha256_file(source)
                after = _sha256_file(destination)
                if before != after:
                    raise ValueError(f"Unchanged artifact differs: {name}")
                proof[name] = {"parent_sha256": before, "corrected_sha256": after,
                               "bytes": source.stat().st_size,
                               "same_file": os.path.samefile(source, destination)}
        y, q = np.array(parent["y"]), np.array(parent["q"])
        evaluation = pd.read_pickle(parent_path / "evaluation.pkl")
        audit = repair_horizon_labels(y, evaluation, parent["codes"], parent["calendar"])
        audit["changed_q_cells"] = rerank_repaired_dates(y, q, parent["codes"], audit["affected_date_codes"])
        audit["changed_q_rows"] = sorted({row for row, _ in audit["changed_q_cells"]})
        audit["changed_cells"] = {"y": audit["changed_y_cells"], "q": audit["changed_q_cells"]}
        gaps = known_gap_records(evaluation, parent["codes"], parent["calendar"],
                                 audit["changed_y_rows"], directory)
        audit["longer_horizon_exclusions"] = exclude_known_gap_horizons(evaluation, parent["codes"], gaps)
        audit.update(parent_fingerprint=parent["manifest"]["fingerprint"], corrected_fingerprint=signature,
                     parent_data_code_sha256=parent["manifest"]["code_sha256"],
                     corrected_data_code_sha256=identity["code_sha256"],
                     unchanged_artifacts=proof,
                     parent_target_sha256={name: _sha256_file(parent_path / name) for name in ("y.npy", "q.npy")})
        for name, values in (("y", y), ("q", q)):
            np.save(directory / f"{name}.npy", values)
        atomic_pickle(evaluation, directory / "evaluation.pkl")
        audit["corrected_target_sha256"] = {name: _sha256_file(directory / name) for name in ("y.npy", "q.npy")}
        manifest = {**parent["manifest"], "fingerprint": signature,
                    "code_sha256": identity["code_sha256"], "derivation": identity,
                    "target_horizon_repair": audit,
                    "eligible_target_rows": {TARGETS[j]: int(np.isfinite(q[:, j]).sum()) for j in range(2)},
                    "source_audit": {**parent["manifest"]["source_audit"],
                        "source_comparison_scope": "original finite source targets before the separately documented horizon exclusion",
                        "horizon_excluded_rows": audit["excluded_rows"]}}
        audit_path = directory / "horizon_repair_audit.json"
        atomic_json(audit, audit_path)
        atomic_json(manifest, directory / "manifest.json")
        print(json.dumps({"excluded_rows": audit["excluded_rows"], "changed_y_cells": audit["changed_y_cells"],
                          "changed_q_cells": len(audit["changed_q_cells"]),
                          "affected_dates": audit["affected_dates"], "audit": str(audit_path)}, indent=2), flush=True)
        print(f"PREPARED {directory}", flush=True)
    return directory


def prepare(data_dir=DATA, run_root=None):
    """Prepare rank inputs once, retaining missing-label prediction rows."""
    data_dir = Path(data_dir).resolve()
    source = data_dir / "text_master.pkl"
    files = [data_dir / "CRSP" / f"dsf_final_{year}.pkl" for year in range(2010, 2024)]
    identity = {"protocol": PROTOCOL, "source": source_stat(source),
                "crsp_sources": [source_stat(p) for p in files],
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "rank_min_labels": 10, "source_timing": "legacy first nominal 16:00 close"}
    digest = fingerprint(identity)
    directory = Path(run_root or ROOT / ".runs" / "protocol_v1_1" / "prepared") / digest[:16]
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock", timeout=0):
        if (directory / "manifest.json").exists():
            print(f"PREPARED {directory}", flush=True)
            return directory
        crsp, calendar, source_records = _compact_crsp(data_dir, directory, files)
        print("Loading text master", flush=True)
        frame = pd.read_pickle(source)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if frame.duplicated(["date", "permno"]).any() or frame["mm_index"].duplicated().any():
            raise ValueError("Text panel keys are not unique")
        if frame[["date", "permno", "mm_index"]].isna().any().any():
            raise ValueError("Text panel contains missing keys")
        frame.sort_values(["date", "permno"], inplace=True, kind="stable")
        frame.set_index("mm_index", inplace=True)
        keys = frame[["date", "permno", "ticker"]].copy()
        dates = frame["date"].to_numpy().astype("datetime64[D]")
        codes = np.searchsorted(calendar, dates).astype(np.int32)
        if not np.array_equal(calendar[codes], dates):
            raise ValueError("Text dates are absent from full CRSP calendar")
        scalars = [c for c in frame if not (c.startswith(("ar_", "embed_", "px_"))
                    or c in ("date", "permno", "ticker", "f_cumret1", "year_month"))]
        embeds = [c for c in frame if c.startswith("embed_") and c not in ("embed_n", "embed_norm", "embed_cos")]
        if len(scalars) != 53 or len(embeds) != 384:
            raise ValueError(f"Unexpected feature schema: {len(scalars)} scalars, {len(embeds)} embeddings")
        base_names = scalars + embeds + ["embed_norm", "embed_cos"]
        missing = {name: int((~np.isfinite(frame[name].to_numpy(dtype=np.float64))).sum()) for name in base_names}
        flags = [name for name in base_names if missing[name]]
        names = base_names + [f"missing__{name}" for name in flags]
        X = np.lib.format.open_memmap(directory / "X.npy", mode="w+", dtype="float32", shape=(len(frame), len(names)))
        for j, name in enumerate(base_names):
            values = frame[name].to_numpy(dtype=np.float64)
            X[:, j] = (centered_rank(values, codes) if name in scalars
                       else np.where(np.isfinite(values), values, np.nan))
            if (j + 1) % 50 == 0:
                print(f"Prepared features {j + 1}/{len(base_names)}", flush=True)
        for k, name in enumerate(flags):
            X[:, len(base_names) + k] = ~np.isfinite(frame[name].to_numpy(dtype=np.float64))
        X.flush()
        del X
        y = frame[TARGETS].to_numpy(dtype=np.float64)
        y[~np.isfinite(y)] = np.nan
        groups_by_name = {"core": CORE, "all": scalars,
                          "textcore": [c for c in base_names if c in set(CORE + embeds + ["embed_norm", "embed_cos"])],
                          "textall": base_names}
        sets = {}
        for name, selected in groups_by_name.items():
            chosen = set(selected + [f"missing__{c}" for c in flags if c in selected])
            sets[name] = [i for i, c in enumerate(names) if c in chosen]
        eval_columns = ["date", "permno", "ticker", "log_volume", "embed_n"] + [c for c in frame if c.startswith("ar_")]
        evaluation = frame[eval_columns].copy()
        # Source y values are retained exactly. Recomputed upstream DGTW horizons
        # have explicit names and do not silently repair historical corruption.
        text_y = frame[TARGETS].copy()
        del frame
        gc.collect()
        lookup = pd.MultiIndex.from_frame(keys[["date", "permno"]])
        compact = crsp.set_index(["date", "permno"]).reindex(lookup)
        compact.index = keys.index
        del crsp
        for name in compact:
            evaluation[name] = compact[name].to_numpy()
        raw_finite = np.isfinite(y[:, 0]) & np.isfinite(evaluation["f_cumret1"])
        dgtw_finite = np.isfinite(y[:, 1]) & np.isfinite(evaluation["crsp_ar_dgtw_1"])
        audit = {"raw_panel_vs_crsp_mismatches": int(np.sum(~np.isclose(y[raw_finite, 0], evaluation.loc[raw_finite, "f_cumret1"], atol=1e-10, rtol=1e-8))),
                 "dgtw_panel_vs_crsp_mismatches": int(np.sum(~np.isclose(y[dgtw_finite, 1], evaluation.loc[dgtw_finite, "crsp_ar_dgtw_1"], atol=1e-10, rtol=1e-8))),
                 "missing_lag_cap": int(evaluation["lag_cap"].isna().sum()),
                 "missing_current_cap": int(evaluation["cap"].isna().sum())}
        same_next = np.isfinite(evaluation["next_session_ret"]) & np.isfinite(evaluation["f_cumret1"])
        audit["one_day_vs_next_session_return_mismatches"] = int(np.sum(~np.isclose(evaluation.loc[same_next, "f_cumret1"], evaluation.loc[same_next, "next_session_ret"], atol=1e-10, rtol=1e-8)))
        audit["one_day_vs_next_session_return_compared"] = int(same_next.sum())
        evaluation["f_cumret1"] = text_y["f_cumret1"]
        horizon_repair = repair_horizon_labels(y, evaluation, codes, calendar)
        gaps = known_gap_records(evaluation, codes, calendar, horizon_repair["changed_y_rows"], directory)
        horizon_repair["longer_horizon_exclusions"] = exclude_known_gap_horizons(evaluation, codes, gaps)
        q = np.column_stack([centered_rank(y[:, j], codes, target=True) for j in range(2)])
        coverage = keys.groupby(keys["date"].dt.to_period("M")).size()
        all_months = pd.period_range(coverage.index.min(), coverage.index.max(), freq="M")
        missing_dates = calendar[(calendar >= dates.min()) & (calendar <= dates.max()) & ~np.isin(calendar, np.unique(dates))]
        manifest = {**identity, "fingerprint": digest, "rows": len(keys), "feature_names": names,
                    "feature_sets": sets, "binary_features": names[len(base_names):],
                    "scalar_rank_features": scalars, "unranked_features": embeds + ["embed_norm", "embed_cos"],
                    "missing_counts": missing, "targets": TARGETS,
                    "calendar_source": "union of dates in full CRSP 2010-2023; early closes not repaired",
                    "calendar_sessions": len(calendar), "calendar_first": str(calendar[0]), "calendar_last": str(calendar[-1]),
                    "input_first": str(dates.min()), "input_last": str(dates.max()),
                    "missing_input_dates": [str(d) for d in missing_dates],
                    "missing_input_months": [str(p) for p in all_months.difference(coverage.index)],
                    "month_rows": {str(k): int(v) for k, v in coverage.items()},
                    "eligible_target_rows": {TARGETS[j]: int(np.isfinite(q[:, j]).sum()) for j in range(2)},
                    "target_availability": "assumed at session endpoint; no publication-delay metadata",
                    "source_audit": audit,
                    "target_horizon_repair": horizon_repair,
                    "source_limitations": ["nominal 16:00 message routing ignores early closes", "ar_dgtw_21 has known 2013 panel corruption; crsp_ar_dgtw_21 is separately retained", "h>1 imported labels and factor-CAR completeness not certified"]}
        if audit["raw_panel_vs_crsp_mismatches"] or audit["dgtw_panel_vs_crsp_mismatches"]:
            raise ValueError(f"One-day source target disagreement requires investigation: {audit}")
        for name, values in (("y", y), ("q", q), ("codes", codes), ("calendar", calendar)):
            np.save(directory / f"{name}.npy", values)
        atomic_pickle(keys, directory / "keys.pkl")
        atomic_pickle(evaluation, directory / "evaluation.pkl")
        if source_stat(source) != identity["source"] or [source_stat(p) for p in files] != source_records:
            raise RuntimeError("Input source changed during preparation")
        atomic_json(manifest, directory / "manifest.json")
        print(json.dumps({"rows": len(keys), "features": len(names), "source_audit": audit, "missing_input_months": manifest["missing_input_months"]}, indent=2), flush=True)
        print(f"PREPARED {directory}", flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--repair-prepared", type=Path,
                        help="Create a new prepared version with calendar-gap h=1 targets excluded")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.repair_prepared:
        repair_prepared(args.repair_prepared, args.run_root)
    else:
        prepare(args.data_dir, args.run_root)


if __name__ == "__main__":
    main()
