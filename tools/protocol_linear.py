"""Matched OLS/ridge/lasso/elastic-net experiments under protocol v1.1.

All four estimators share the prepared universe, fit/validation dates, transformed
inputs, equal-date normalized loss, unpenalized intercept, and validation IC.
The selected fitting-block model is retained without a validation-block refit.

The objective is 0.5 * sum(w * residual**2) + alpha*l1_ratio*|beta|_1
                 + 0.5*alpha*(1-l1_ratio)*|beta|_2**2, where sum(w)=1.
OLS has alpha=0; ridge has l1_ratio=0. Candidate fits use reusable Gram matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from filelock import FileLock
from joblib import Parallel, delayed
from scipy.stats import rankdata
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path
from threadpoolctl import threadpool_limits

try:
    from .nn_checkpoint import atomic_json, atomic_pickle, fingerprint
except ImportError:
    from nn_checkpoint import atomic_json, atomic_pickle, fingerprint


SCHEMA = "protocol_v1_1_linear"
FEATURE_SETS = ("core", "all", "textcore", "textall")
ESTIMATORS = ("ols", "ridge", "lasso", "enet")
TARGET_COLUMNS = {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}
RIDGE_ALPHAS = tuple(float(x) for x in np.logspace(3, -6, 10))
ALPHA_FRACTIONS = (1.0, .3, .1, .03, .01, .003, .001, .0003, .0001)
L1_RATIOS = (.1, .5, .9)
SOLVER_TOL = 1e-8
KKT_TOL = 2e-7
MAX_ITER = 5_000
MAX_POLISH_STEPS = 2_000
EIGEN_RCOND = 1e-12
TIE_TOL = 1e-12


def protocol_data():
    try:
        from . import protocol_data as data_module
    except ImportError:
        import protocol_data as data_module
    return data_module


def grid_specification():
    return {
        "objective": "0.5*sum(w*residual^2)+alpha*l1_ratio*L1(beta)"
                     "+0.5*alpha*(1-l1_ratio)*L2(beta)^2; sum(w)=1",
        "intercept": "unpenalized, fitted using normalized equal-date label weights",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "alpha_fractions": list(ALPHA_FRACTIONS),
        "lasso_alpha": "fraction * max(abs(centered_X_transpose_weighted_y))",
        "enet_alpha": "fraction * max(abs(centered_X_transpose_weighted_y)) / l1_ratio",
        "enet_l1_ratios": list(L1_RATIOS),
        "selection": "largest mean daily validation Spearman IC, minimum 10 labels/day",
        "tie_rule": "first declared candidate within 1e-12 of maximum; ridge alpha descending; "
                    "lasso fraction descending; enet fraction descending then l1_ratio ascending",
        "refit": False, "solver": "public sklearn.enet_path on equivalent spectral design, "
                                   "then active-set Newton refinement with original-objective certificates",
        "solver_tolerance": SOLVER_TOL, "kkt_absolute_tolerance": KKT_TOL,
        "coordinate_descent_max_iterations": MAX_ITER, "active_set_max_steps": MAX_POLISH_STEPS,
        "normalized_dual_gap_bound": "max(solver_tolerance * fitting_target_variance, 1e-12)",
        "coordinate_descent_limit": "recorded; candidate must pass post-refinement KKT and dual-gap checks",
        "eigen_pseudoinverse_rcond": EIGEN_RCOND,
    }


def weighted_moments(X, y, weights, chunk_rows=32768):
    """Accumulate float64 sufficient statistics, independent of row chunk size."""
    X, y, weights = np.asarray(X), np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or weights.shape != y.shape:
        raise ValueError("Incompatible fitting arrays")
    if not len(X) or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError("Empty or nonfinite fitting values")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("Fitting weights must be nonnegative and sum to one")
    p = X.shape[1]
    gram, cross, mean = np.zeros((p, p)), np.zeros(p), np.zeros(p)
    minimum, maximum = np.full(p, np.inf), np.full(p, -np.inf)
    ymean = float(weights @ y)
    for lo in range(0, len(X), chunk_rows):
        hi = min(lo + chunk_rows, len(X))
        block = np.array(X[lo:hi], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite fitting input")
        w = weights[lo:hi]
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        mean += w @ block
        cross += block.T @ (w * y[lo:hi])
        block *= np.sqrt(w[:, None])
        gram += block.T @ block
    gram -= np.outer(mean, mean)
    cross -= mean * ymean
    gram = .5 * (gram + gram.T)
    constant = minimum == maximum
    gram[constant, :], gram[:, constant], cross[constant] = 0., 0., 0.
    return {"gram": gram, "cross": cross, "x_mean": mean, "y_mean": ymean,
            "y_variance": float(weights @ ((y-ymean)**2))}


def kkt_violation(gram, cross, beta, alpha, l1_ratio):
    gradient = gram @ beta - cross + alpha * (1-l1_ratio) * beta
    nz = beta != 0
    error = np.maximum(np.abs(gradient) - alpha*l1_ratio, 0)
    error[nz] = np.abs(gradient[nz] + alpha*l1_ratio*np.sign(beta[nz]))
    return float(np.max(error, initial=0))


def objective_gap(gram, cross, beta, alpha, l1_ratio, y_variance):
    """Fenchel primal-dual gap for the original normalized weighted objective."""
    l1, l2 = alpha*l1_ratio, alpha*(1-l1_ratio)
    quadratic = float(beta @ gram @ beta)
    cross_beta = float(cross @ beta)
    residual_variance = y_variance-2*cross_beta+quadratic
    primal = .5*residual_variance+l1*np.abs(beta).sum()+.5*l2*(beta@beta)
    score = cross-gram@beta
    if l2 > 0:
        soft = np.sign(score)*np.maximum(np.abs(score)-l1, 0.)
        dual = .5*y_variance-.5*quadratic-.5*(soft@soft)/l2
    else:
        maximum = float(np.max(np.abs(score), initial=0))
        shrink = min(1., l1/maximum) if maximum > 0 else 1.
        dual = shrink*(y_variance-cross_beta)-.5*shrink**2*residual_variance
    return float(max(primal-dual, 0.))


def polish_solution(gram, cross, beta, alpha, l1_ratio, max_steps=MAX_POLISH_STEPS):
    """Resolve slow coordinate drift using a convex active-set Newton method.

    Solve the quadratic exactly on the current active orthant. If a coefficient
    would change sign, move only to the first zero crossing and remove it. Once
    that orthant is minimized, add the most KKT-violating inactive coordinate.
    This preserves the original objective, including nearly collinear inputs.
    """
    beta = np.asarray(beta, dtype=float).copy()
    l1, l2 = alpha*l1_ratio, alpha*(1-l1_ratio)
    hessian = gram.copy()
    hessian.flat[::len(beta)+1] += l2
    tolerance = min(1e-11, max(l1*1e-6, 1e-13))
    for step in range(max_steps):
        active = np.flatnonzero(beta != 0)
        if len(active):
            signs = np.sign(beta[active])
            submatrix = hessian[np.ix_(active, active)]
            rhs = cross[active]-l1*signs
            proposal = (np.linalg.solve(submatrix, rhs) if l2 > 0
                        else np.linalg.lstsq(submatrix, rhs, rcond=EIGEN_RCOND)[0])
            crossing = proposal*signs <= 0
            if crossing.any():
                direction = proposal-beta[active]
                fractions = -beta[active[crossing]]/direction[crossing]
                fraction = float(fractions.min())
                beta[active] += fraction*direction
                leaving = active[crossing][np.abs(fractions-fraction) <= 1e-12]
                beta[leaving] = 0.
                continue
            beta[active] = proposal
        gradient = hessian@beta-cross
        inactive = beta == 0
        violation = np.where(inactive, np.abs(gradient)-l1, -np.inf)
        entering = int(np.argmax(violation))
        if violation[entering] <= tolerance:
            return beta, step+1
        if hessian[entering, entering] <= 0:
            raise RuntimeError("Nonpositive curvature at a violating coordinate")
        beta[entering] = -np.sign(gradient[entering])*violation[entering]/hessian[entering, entering]
    raise RuntimeError("Active-set Newton refinement did not converge")


def fit_candidates(moments, columns):
    """Fit every predeclared candidate once from a submatrix of shared moments.

    A p+1-row spectral design preserves centered weighted Gram, cross product,
    and target variance. Public enet_path and active-set refinement minimize the
    original normalized objective without repeatedly copying the large fitting
    matrix. Each penalized candidate must pass original Gram KKT and dual-gap checks.
    """
    columns = np.asarray(columns, dtype=int)
    gram = moments["gram"][np.ix_(columns, columns)]
    cross = moments["cross"][columns]
    eig, basis = np.linalg.eigh(gram)
    cutoff = max(float(eig.max(initial=0)), 1.0)*EIGEN_RCOND
    if eig.min(initial=0) < -cutoff*10:
        raise ValueError("Weighted Gram matrix is not positive semidefinite")
    eig = np.maximum(eig, 0)
    active = eig > cutoff
    cross_basis = basis.T @ cross
    candidates = []

    def append(estimator, beta, alpha=0., l1_ratio=0., fraction=None, iterations=0,
               dual_gap=None, polish_iterations=0, cd_warning=False):
        violation = kkt_violation(gram, cross, beta, alpha, l1_ratio)
        if not np.isfinite(beta).all() or violation > KKT_TOL:
            raise RuntimeError(f"{estimator} failed KKT: {violation:.3g} > {KKT_TOL}; "
                               f"alpha={alpha:.8g}, l1_ratio={l1_ratio}")
        candidates.append({"estimator": estimator, "alpha": float(alpha),
                           "l1_ratio": float(l1_ratio), "alpha_fraction": fraction,
                           "coefficient": np.asarray(beta, dtype=float),
                           "intercept": float(moments["y_mean"]-moments["x_mean"][columns]@beta),
                           "iterations": int(iterations), "dual_gap": dual_gap,
                           "polish_iterations": int(polish_iterations),
                           "coordinate_descent_limit": bool(cd_warning),
                           "kkt_violation": violation})

    inv = np.divide(cross_basis, eig, out=np.zeros_like(eig), where=active)
    append("ols", basis @ inv)
    for alpha in RIDGE_ALPHAS:
        append("ridge", basis @ (cross_basis/(eig+alpha)), alpha=alpha)
    alpha_max = float(np.max(np.abs(cross), initial=0))
    if alpha_max <= np.finfo(float).eps:
        for estimator, ratios in (("lasso", (1.,)), ("enet", L1_RATIOS)):
            for fraction in ALPHA_FRACTIONS:
                for ratio in ratios:
                    append(estimator, np.zeros(len(columns)), l1_ratio=ratio, fraction=fraction)
        return candidates

    p, effective_n = len(columns), len(columns)+1
    # Preserve the original response variance as well as Gram and cross product.
    # Otherwise coordinate descent's relative dual-gap tolerance is scaled by
    # explained variance alone and becomes needlessly severe for weak signals.
    projected_response = np.divide(cross_basis, np.sqrt(eig),
                                   out=np.zeros_like(eig), where=active)
    residual_variance = float(moments["y_variance"]-projected_response@projected_response)
    if residual_variance < -1e-10:
        raise ValueError("Gram projection exceeds original target variance")
    surrogate_X = np.zeros((effective_n, p), dtype=float, order="F")
    surrogate_X[:p] = np.sqrt(effective_n*eig)[:, None]*basis.T
    surrogate_y = np.r_[np.sqrt(effective_n)*projected_response,
                         np.sqrt(effective_n*max(residual_variance, 0.))]
    paths = {}
    for ratio in (1.,) + L1_RATIOS:
        alphas = alpha_max / ratio * np.asarray(ALPHA_FRACTIONS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            _, coefficients, gaps, iterations = enet_path(
                surrogate_X, surrogate_y, l1_ratio=ratio, alphas=alphas,
                precompute=True, tol=SOLVER_TOL, max_iter=MAX_ITER,
                selection="cyclic", return_n_iter=True)
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                raise RuntimeError(f"Unexpected path warning: {warning.message}")
        paths[ratio] = (alphas, coefficients, gaps, iterations)
    for estimator, ratios in (("lasso", (1.,)), ("enet", L1_RATIOS)):
        for k, fraction in enumerate(ALPHA_FRACTIONS):
            for ratio in ratios:
                alphas, coefficients, gaps, iterations = paths[ratio]
                beta, polish_steps = polish_solution(gram, cross, coefficients[:, k], alphas[k], ratio)
                gap = objective_gap(gram, cross, beta, alphas[k], ratio, moments["y_variance"])
                if gap > max(SOLVER_TOL*moments["y_variance"], 1e-12):
                    raise RuntimeError(f"{estimator} failed objective certificate: dual gap={gap:.3g}")
                append(estimator, beta, alphas[k], ratio, fraction,
                       iterations[k], gap, polish_steps, iterations[k] >= MAX_ITER)
    return candidates


def validation_scores(predictions, targets, codes, min_obs=10):
    """Mean daily Spearman IC with the conventions in prediction_metrics.py."""
    predictions = np.asarray(predictions, dtype=float)
    targets, codes = np.asarray(targets), np.asarray(codes)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if len(predictions) != len(targets) or len(codes) != len(targets):
        raise ValueError("Incompatible validation arrays")
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite validation predictions")
    totals, days = np.zeros(predictions.shape[1]), 0
    # Sorting once keeps this valid for fixtures and for arrays not already ordered.
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order]))+1, len(order)]
    for first, last in zip(starts[:-1], starts[1:]):
        indices = order[first:last]
        indices = indices[np.isfinite(targets[indices])]
        if len(indices) < min_obs:
            continue
        yr = rankdata(targets[indices], method="average")
        yr -= yr.mean()
        yy = yr @ yr
        if yy == 0:
            continue
        xr = rankdata(predictions[indices], method="average", axis=0)
        xr -= xr.mean(axis=0)
        denominator = np.sqrt(np.sum(xr*xr, axis=0)*yy)
        totals += np.divide(yr @ xr, denominator, out=np.zeros_like(totals), where=denominator > 0)
        days += 1
    if not days:
        raise ValueError("No eligible validation dates")
    return totals/days, days


def select_candidates(candidates, X_valid, y_valid, codes_valid):
    coefficients = np.column_stack([x["coefficient"] for x in candidates])
    intercepts = np.asarray([x["intercept"] for x in candidates])
    scores, days = validation_scores(X_valid @ coefficients + intercepts, y_valid, codes_valid)
    for candidate, score in zip(candidates, scores):
        candidate["validation_ic"] = float(score)
        candidate["validation_days"] = days
    selected = {}
    for estimator in ESTIMATORS:
        options = [c for c in candidates if c["estimator"] == estimator]
        maximum = max(c["validation_ic"] for c in options)
        selected[estimator] = next(c for c in options if c["validation_ic"] >= maximum-TIE_TOL)
    return selected


def model_name(estimator, feature_set, target, fit_days, validation_days=126):
    return f"{estimator}_{feature_set}_{target}_fit{fit_days}_val{validation_days}"


def fit_month(bundle, task, target):
    data = protocol_data()
    block = data.make_block(bundle, task, "textall", target)
    moments = weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
    predictions, diagnostics, names = [], {}, []
    for feature_set in FEATURE_SETS:
        columns = data.feature_columns(bundle, feature_set)
        candidates = fit_candidates(moments, columns)
        selected = select_candidates(candidates, block["X_valid"][:, columns],
                                     block["y_valid"], block["codes_valid"])
        coef = np.column_stack([selected[estimator]["coefficient"] for estimator in ESTIMATORS])
        intercepts = np.asarray([selected[estimator]["intercept"] for estimator in ESTIMATORS])
        predictions.append(block["X_test"][:, columns] @ coef + intercepts)
        for estimator in ESTIMATORS:
            name = model_name(estimator, feature_set, target, task["fit_days"])
            names.append(name)
            chosen = selected[estimator]
            diagnostics[name] = {
                "selected": chosen,
                "grid": [{key: value for key, value in candidate.items()
                          if key not in ("coefficient", "intercept")}
                         for candidate in candidates if candidate["estimator"] == estimator],
                "features": [block["feature_names"][j] for j in columns],
                "feature_mean": block["mean"][columns], "feature_scale": block["scale"][columns],
                "constant": block["constant"][columns],
            }
    return {"month": task["month"], "target": target,
            "test_indices": np.asarray(block["test_indices"], dtype=np.int64),
            "predictions": np.column_stack(predictions), "model_names": names,
            "fit_rows": len(block["X_fit"]), "validation_rows": len(block["X_valid"]),
            "fit_dates": int(len(np.unique(block["codes_fit"]))),
            "validation_dates": int(len(np.unique(block["codes_valid"]))),
            "diagnostics": diagnostics}


def validate_month(result, task, target, bundle):
    indices = np.asarray(result["test_indices"])
    expected = np.flatnonzero((bundle["codes"] >= task["test_first"]) &
                              (bundle["codes"] <= task["test_last"]))
    names = [model_name(estimator, feature_set, target, task["fit_days"])
             for feature_set in FEATURE_SETS for estimator in ESTIMATORS]
    if (result["month"] != task["month"] or result["target"] != target
            or not np.array_equal(indices, expected) or result["model_names"] != names
            or result["predictions"].shape != (len(expected), len(names))
            or not np.isfinite(result["predictions"]).all()):
        raise ValueError(f"Invalid prediction checkpoint for {target}/{task['month']}")


def checkpoint_month(prepared, task, target, run_dir, run_fingerprint, threads=2):
    bundle = protocol_data().load_bundle(prepared)
    path = Path(run_dir)/"checkpoints"/f"fit{task['fit_days']}_{target}_{task['month']}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+".lock", timeout=0):
        if path.exists():
            with path.open("rb") as stream:
                saved = pickle.load(stream)
            if saved["fingerprint"] != run_fingerprint or saved["task"] != task:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            validate_month(saved["result"], task, target, bundle)
            print(f"RESUME fit{task['fit_days']} {target} {task['month']}", flush=True)
            return str(path)
        started = time.monotonic()
        with threadpool_limits(limits=threads):
            result = fit_month(bundle, task, target)
        validate_month(result, task, target, bundle)
        result["elapsed_seconds"] = time.monotonic()-started
        atomic_pickle({"fingerprint": run_fingerprint, "task": task, "result": result}, path)
        print(f"SAVED fit{task['fit_days']} {target} {task['month']}: "
              f"{result['elapsed_seconds']:.1f}s", flush=True)
        return str(path)


def code_identity():
    paths = [Path(__file__), Path(__file__).with_name("protocol_data.py"),
             Path(__file__).with_name("nn_checkpoint.py"),
             Path(__file__).with_name("prediction_metrics.py")]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def publish_run(bundle, run_dir, config, checkpoint_paths):
    """Publish complete keyed model series only after all requested months finish."""
    run_dir = Path(run_dir)
    grouped = {}
    monthly_summary = []
    for path in checkpoint_paths:
        with Path(path).open("rb") as stream:
            saved = pickle.load(stream)
        if saved["fingerprint"] != config["fingerprint"]:
            raise ValueError("Publication checkpoint fingerprint mismatch")
        result, task = saved["result"], saved["task"]
        validate_month(result, task, result["target"], bundle)
        for j, name in enumerate(result["model_names"]):
            grouped.setdefault(name, []).append((result["test_indices"], result["predictions"][:, j]))
            chosen = result["diagnostics"][name]["selected"]
            monthly_summary.append({"model_id": name, "month": result["month"],
                                    "fit_rows": result["fit_rows"],
                                    "validation_rows": result["validation_rows"],
                                    "test_rows": len(result["test_indices"]),
                                    "elapsed_month_seconds": result["elapsed_seconds"],
                                    **{k: v for k, v in chosen.items()
                                       if k not in ("coefficient", "intercept")}})
    models = []
    for fit_days in config["fit_days"]:
        for target in config["targets"]:
            for feature_set in FEATURE_SETS:
                for estimator in ESTIMATORS:
                    name = model_name(estimator, feature_set, target, fit_days)
                    pieces = grouped.pop(name)
                    indices = np.concatenate([piece[0] for piece in pieces])
                    frame = bundle["keys"].iloc[indices].copy()
                    frame["prediction"] = np.concatenate([piece[1] for piece in pieces])
                    frame.sort_values(["date", "permno"], inplace=True)
                    if frame.duplicated(["date", "permno"]).any() or not frame.index.is_unique:
                        raise ValueError(f"Duplicate published prediction keys: {name}")
                    path = run_dir/f"predictions_{name}.pkl"
                    atomic_pickle(frame, path)
                    entry = {"model_id": name, "estimator": estimator,
                             "feature_set": feature_set, "target": target,
                             "target_column": TARGET_COLUMNS[target], "fit_days": fit_days,
                             "validation_days": 126, "predictions": str(path.resolve()),
                             "rows": len(frame), "months": int(frame.date.dt.to_period("M").nunique()),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    atomic_json({**entry, "fingerprint": config["fingerprint"],
                                 "schema_version": SCHEMA}, path.with_suffix(".json"))
                    models.append(entry)
    summary_path = run_dir/"monthly_selection.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    pd.DataFrame(monthly_summary).to_csv(temporary, index=False)
    os.replace(temporary, summary_path)
    registry = {"schema_version": SCHEMA, "prepared": config["prepared"],
                "fingerprint": config["fingerprint"], "models": models,
                "config": config, "monthly_selection": str(summary_path.resolve())}
    atomic_json(registry, run_dir/"registry.json")
    atomic_json({"fingerprint": config["fingerprint"], "models": len(models),
                 "checkpoints": len(checkpoint_paths)}, run_dir/"complete.json")
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--fit-days", nargs="+", type=int, default=[504, 252, 756],
                        choices=[252, 504, 756])
    parser.add_argument("--targets", nargs="+", choices=["raw", "dgtw"], default=["raw", "dgtw"])
    parser.add_argument("--start", default="2014-01")
    parser.add_argument("--end", default="2022-12")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-root", type=Path,
                        default=Path(__file__).resolve().parents[1]/".runs"/"protocol_v1_1"/"linear")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    args.fit_days, args.targets = list(dict.fromkeys(args.fit_days)), list(dict.fromkeys(args.targets))
    data = protocol_data()
    bundle = data.load_bundle(args.prepared)
    tasks = [task for fit in args.fit_days for task in data.make_tasks(
        bundle, fit_days=fit, validation_days=126, horizon=1,
        start=args.start, end=args.end, months=args.months)]
    if not tasks:
        raise ValueError("No eligible test months")
    config = {"schema_version": SCHEMA, "prepared": str(args.prepared.resolve()),
              "data_manifest": bundle["manifest"], "fit_days": args.fit_days,
              "validation_days": 126, "horizon": 1, "targets": args.targets,
              "feature_sets": list(FEATURE_SETS), "estimators": list(ESTIMATORS),
              "tasks": tasks, "grid": grid_specification(), "threads": args.threads,
              "code": code_identity(), "versions": {"python": sys.version, "numpy": np.__version__,
                  "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "run_kind": "pilot" if args.months else "full"}
    config["fingerprint"] = fingerprint(config)
    run_dir = args.run_root/f"{config['run_kind']}_{config['fingerprint'][:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config, run_dir/"config.json")
    work = [(task, target) for task in tasks for target in args.targets]
    print(f"RUN {run_dir.resolve()} | {len(work)} month-target fits | workers={args.workers}, "
          f"threads={args.threads}", flush=True)
    if args.workers == 1:
        paths = [checkpoint_month(str(args.prepared), task, target, run_dir,
                                  config["fingerprint"], args.threads) for task, target in work]
    else:
        paths = Parallel(n_jobs=args.workers, backend="loky", pre_dispatch=args.workers)(
            delayed(checkpoint_month)(str(args.prepared), task, target, str(run_dir),
                                      config["fingerprint"], args.threads) for task, target in work)
    publish_run(bundle, run_dir, config, paths)
    print(f"Registry: {(run_dir/'registry.json').resolve()}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
