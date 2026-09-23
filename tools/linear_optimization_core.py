"""Prospective linear optimization; frozen v2 kernels remain unchanged.

Candidate fits and validation scores are shared across the nested menu axes.
Calendar-month deletion measures selection stability only: every forecast still
uses the setting selected on all original validation dates and a union refit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import warnings

import numpy as np
from scipy.stats import rankdata

try:
    from . import linear_design_core as legacy, protocol_data as data, protocol_linear as linear
except ImportError:
    import linear_design_core as legacy
    import protocol_data as data
    import protocol_linear as linear

UNION = legacy.UNION
FEATURE_SETS = legacy.FEATURE_SETS
TRANSFORMS = ("standard", "daily_rank")
PENALTY_GRIDS = ("original", "expanded")
PCA_MENUS = {"original": (16, 32, 64, 128), "expanded": (4, 8, 16, 32, 64, 128)}
RIDGE_EXPANDED = tuple(sorted(set(linear.RIDGE_ALPHAS) |
                             {float(3 * 10. ** exponent) for exponent in range(-6, 3)}, reverse=True))
FRACTIONS_EXPANDED = tuple(sorted(set(linear.ALPHA_FRACTIONS) |
    {float(np.sqrt(a * b)) for a, b in zip(linear.ALPHA_FRACTIONS[:-1], linear.ALPHA_FRACTIONS[1:])}, reverse=True))
RATIOS_EXPANDED = (.01, .03, .1, .5, .9)
TEXT_EXPANDED = (1., 3., 10., 30., 100., 300., 1000.)


def procedure_specs():
    return [{**base, "transform": transform, "penalty_grid": penalty,
             "pca_grid": pca_grid, "history": "fixed504", "refit": "union"}
            for transform in TRANSFORMS for base in legacy.base_specs()
            for penalty in (("original",) if base["estimator"] == "ols" else PENALTY_GRIDS)
            for pca_grid in (("original", "expanded") if base["representation"] == "pca" else ("none",))]


def model_name(spec, target):
    return (f"ld3_{spec['estimator']}_{spec['feature_set']}_{spec['representation']}_"
            f"{target}_{spec['transform']}_{spec['penalty_grid']}_{spec['pca_grid']}_fixed504_union")


def grid_specification():
    return {"original": {"global": linear.grid_specification(),
            "social_multipliers": list(legacy.SOCIAL_MULTIPLIERS),
            "text_multipliers": list(legacy.TEXT_MULTIPLIERS),
            "drop_order": ["none", "text", "social", "text_social"]},
        "fit_windows": [504], "validation_days": 126,
        "refit": "union of original504 fitting dates and126 validation dates; preserve interior purge",
        "transforms": {"standard": "unchanged v2 training-only scaler and numerical path",
            "daily_rank": "rank continuous characteristics within all input rows of each formation date before label filtering; "
                "average ties;2*(rank-1)/(n_finite-1)-1;singleton0;missing stays missing until training-mean imputation; "
                "then training-only standardization;binary flags/social inputs/embeddings unchanged"},
        "expanded": {"ridge_alphas": list(RIDGE_EXPANDED), "alpha_fractions": list(FRACTIONS_EXPANDED),
            "enet_l1_ratios": list(RATIOS_EXPANDED), "social_multipliers": list(legacy.SOCIAL_MULTIPLIERS),
            "text_multipliers": list(TEXT_EXPANDED), "drop_order": ["none", "text", "social", "text_social"]},
        "pca_dimensions": {key: list(value) for key, value in PCA_MENUS.items()},
        "selection": "maximum full-validation mean daily Spearman IC;first declared within1e-12",
        "candidate_order": "PCA dimension ascending;ridge alpha descending;sparse fraction descending then ratio ascending;"
                           "group drop order, social multiplier ascending,text multiplier ascending,alpha descending",
        "stability": "diagnostic only: omit each calendar month of validation in turn and reselect from remaining daily ICs;"
                     "full-validation setting remains the forecast choice",
        "candidate_reuse": "original candidate solutions retained exactly;additional settings parameterized independently;"
                           "shared coefficients and scores across nested grid/PCA axes",
        "model_count_per_target": len(procedure_specs())}


def continuous_characteristic_columns(bundle):
    binary = set(bundle["manifest"].get("binary_features", []))
    names = bundle["manifest"]["feature_names"]
    return [column for column in data.feature_columns(bundle, "characteristics") if names[column] not in binary]


def daily_rank_characteristics(values, codes, columns):
    """Rank a complete set of same-date inputs without consulting any labels."""
    values = np.asarray(values).copy()
    codes = np.asarray(codes)
    if values.ndim != 2 or len(values) != len(codes):
        raise ValueError("Incompatible daily rank inputs")
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
    for first, last in zip(starts[:-1], starts[1:]):
        rows = order[first:last]
        for column in columns:
            present = rows[np.isfinite(values[rows, column])]
            if len(present) == 1:
                values[present, column] = 0.
            elif len(present) > 1:
                values[present, column] = 2 * (rankdata(values[present, column], method="average") - 1) / (len(present) - 1) - 1
    return values


def build_block(bundle, task, target, refit="retain", transform="standard"):
    if transform == "standard":
        return legacy.build_block(bundle, task, target, refit)
    if transform != "daily_rank" or refit not in ("retain", "union"):
        raise ValueError("Unknown transform or refit")
    if task["horizon"] != 1:
        raise ValueError("Only h=1 fitting is supported")
    columns = data.feature_columns(bundle, UNION)
    inverse = {column: j for j, column in enumerate(columns)}
    rank_columns = [inverse[column] for column in continuous_characteristic_columns(bundle)]
    names = [bundle["manifest"]["feature_names"][column] for column in columns]
    binary = set(bundle["manifest"].get("binary_features", []))
    codes = np.asarray(bundle["codes"])
    selections = {"fit": legacy.input_indices(bundle, task, refit),
        "test": np.flatnonzero((codes >= task["test_first"]) & (codes <= task["test_last"]))}
    if refit == "retain":
        selections["valid"] = np.flatnonzero((codes >= task["valid_first"]) & (codes <= task["valid_last"]))
    target_column = linear.TARGET_COLUMNS.get(target, target)
    target_index = bundle["manifest"]["targets"].index(target_column)
    indices = selections["fit"]
    # Each selection consists of complete dates. Ranking precedes outcome filtering.
    raw = daily_rank_characteristics(bundle["X"][np.ix_(indices, columns)], codes[indices], rank_columns)
    mean, scale, constant, missing = data._weighted_scaler(raw, codes[indices], [name in binary for name in names])
    result = dict(mean=mean, scale=scale, constant=constant, all_missing=missing,
                  feature_names=names, input_indices=indices, split=dict(task), target=target_column)
    for part, indices in selections.items():
        source = raw if part == "fit" else daily_rank_characteristics(
            bundle["X"][np.ix_(indices, columns)], codes[indices], rank_columns)
        transformed = data._transform(source, mean, scale)
        q = np.asarray(bundle["q"][indices, target_index])
        if part != "test":
            eligible = np.isfinite(q)
            indices, q, transformed = indices[eligible], q[eligible], transformed[eligible]
            if not len(indices):
                raise ValueError(f"No eligible {part} outcomes")
            result[f"w_{part}"] = data.equal_date_weights(codes[indices])
        result.update({f"X_{part}": transformed, f"y_{part}": q,
                       f"codes_{part}": codes[indices], f"{part}_indices": indices})
    return result


def fit_candidates(moments, ridge_alphas, fractions, ratios):
    """Parameterized spectral paths with the same original-objective certificates."""
    gram, cross = moments["gram"], moments["cross"]
    eig, basis = np.linalg.eigh(gram)
    cutoff = max(float(eig.max(initial=0)), 1.) * linear.EIGEN_RCOND
    if eig.min(initial=0) < -cutoff * 10:
        raise ValueError("Weighted Gram matrix is not positive semidefinite")
    eig = np.maximum(eig, 0)
    active, projected = eig > cutoff, basis.T @ cross
    candidates = []

    def append(estimator, beta, alpha=0., ratio=0., fraction=None, iterations=0, polish=0):
        violation = linear.kkt_violation(gram, cross, beta, alpha, ratio)
        if not np.isfinite(beta).all() or violation > linear.KKT_TOL:
            raise RuntimeError(f"{estimator} failed stationarity certificate")
        gap = linear.objective_gap(gram, cross, beta, alpha, ratio, moments["y_variance"]) if ratio else None
        if gap is not None and gap > max(linear.SOLVER_TOL * moments["y_variance"], 1e-12):
            raise RuntimeError(f"{estimator} failed objective certificate")
        candidates.append(dict(estimator=estimator, coefficient=beta,
            intercept=float(moments["y_mean"] - moments["x_mean"] @ beta), alpha=float(alpha),
            l1_ratio=float(ratio), alpha_fraction=fraction, iterations=int(iterations),
            polish_iterations=int(polish), coordinate_descent_limit=bool(iterations >= linear.MAX_ITER),
            dual_gap=gap, kkt_violation=violation))

    append("ols", basis @ np.divide(projected, eig, out=np.zeros_like(eig), where=active))
    for alpha in ridge_alphas:
        append("ridge", basis @ (projected / (eig + alpha)), alpha)
    maximum = float(np.max(np.abs(cross), initial=0))
    if maximum <= np.finfo(float).eps:
        for estimator, menu in (("lasso", (1.,)), ("enet", ratios)):
            for fraction in fractions:
                for ratio in menu:
                    append(estimator, np.zeros(len(cross)), ratio=ratio, fraction=fraction)
        return candidates
    p, n = len(cross), len(cross) + 1
    response = np.divide(projected, np.sqrt(eig), out=np.zeros_like(eig), where=active)
    remaining = float(moments["y_variance"] - response @ response)
    if remaining < -1e-10:
        raise ValueError("Gram projection exceeds target variance")
    surrogate_X = np.zeros((n, p), dtype=float, order="F")
    surrogate_X[:p] = np.sqrt(n * eig)[:, None] * basis.T
    surrogate_y = np.r_[np.sqrt(n) * response, np.sqrt(n * max(remaining, 0.))]
    paths = {}
    for ratio in (1.,) + tuple(ratios):
        alphas = maximum / ratio * np.asarray(fractions)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", linear.ConvergenceWarning)
            _, coefficients, _, iterations = linear.enet_path(surrogate_X, surrogate_y,
                l1_ratio=ratio, alphas=alphas, precompute=True, tol=linear.SOLVER_TOL,
                max_iter=linear.MAX_ITER, selection="cyclic", return_n_iter=True)
        if any(not issubclass(warning.category, linear.ConvergenceWarning) for warning in caught):
            raise RuntimeError("Unexpected path warning")
        paths[ratio] = alphas, coefficients, iterations
    for estimator, menu in (("lasso", (1.,)), ("enet", ratios)):
        for k, fraction in enumerate(fractions):
            for ratio in menu:
                alphas, coefficients, iterations = paths[ratio]
                beta, polish = linear.polish_solution(gram, cross, coefficients[:, k], alphas[k], ratio)
                append(estimator, beta, alphas[k], ratio, fraction, iterations[k], polish)
    return candidates


def group_ridge_candidates(moments, groups, ridge_alphas, text_multipliers):
    groups = np.asarray(groups)
    drops = ("none", "text", "social", "text_social") if np.any(groups == "text") else ("none", "social")
    candidates = []
    for drop in drops:
        keep = np.ones(len(groups), dtype=bool)
        for group in ("text", "social"):
            if group in drop:
                keep &= groups != group
        columns = np.flatnonzero(keep)
        social_menu = legacy.SOCIAL_MULTIPLIERS if np.any(groups[columns] == "social") else (1.,)
        text_menu = text_multipliers if np.any(groups[columns] == "text") else (1.,)
        gram, cross = moments["gram"][np.ix_(columns, columns)], moments["cross"][columns]
        for social in social_menu:
            for text in text_menu:
                multipliers = np.where(groups[columns] == "social", social, np.where(groups[columns] == "text", text, 1.))
                whitening = 1. / np.sqrt(multipliers)
                eig, basis = np.linalg.eigh(gram * whitening[:, None] * whitening[None, :])
                eig = np.maximum(eig, 0)
                projected = basis.T @ (whitening * cross)
                for alpha in ridge_alphas:
                    beta = np.zeros(len(groups))
                    beta[columns] = whitening * (basis @ (projected / (eig + alpha)))
                    violation = float(np.max(np.abs(gram @ beta[columns] - cross + alpha * multipliers * beta[columns])))
                    if not np.isfinite(beta).all() or violation > linear.KKT_TOL:
                        raise RuntimeError("Group ridge failed stationarity certificate")
                    candidates.append(dict(estimator="group_ridge", coefficient=beta,
                        intercept=float(moments["y_mean"] - moments["x_mean"] @ beta), alpha=float(alpha),
                        l1_ratio=0., alpha_fraction=None, group_social_multiplier=social,
                        group_text_multiplier=text, drop_block=drop, kkt_violation=violation))
    return candidates


def setting_key(candidate):
    return (candidate["estimator"], candidate.get("pca_dim"),
        candidate["alpha"] if candidate["estimator"] in ("ridge", "group_ridge") else None,
        candidate.get("alpha_fraction"), candidate.get("l1_ratio"),
        candidate.get("drop_block"), candidate.get("group_social_multiplier"), candidate.get("group_text_multiplier"))


def candidate_order(candidate):
    return (candidate.get("pca_dim") or 0,
            {"none": 0, "text": 1, "social": 2, "text_social": 3}.get(candidate.get("drop_block"), 0),
            candidate.get("group_social_multiplier", 1.), candidate.get("group_text_multiplier", 1.),
            -float(candidate.get("alpha_fraction") or candidate.get("alpha") or 0.), candidate.get("l1_ratio", 0.))


def score_candidates(candidates, X, y, codes, calendar, batch_size=16):
    """Retain per-month sums, not the N by candidate validation predictions."""
    codes, y = np.asarray(codes), np.asarray(y)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
    days = []
    for first, last in zip(starts[:-1], starts[1:]):
        rows = order[first:last]
        rows = rows[np.isfinite(y[rows])]
        if len(rows) < 10:
            continue
        ranks = rankdata(y[rows], method="average")
        ranks -= ranks.mean()
        yy = ranks @ ranks
        if yy:
            month = str(np.asarray(calendar)[codes[rows[0]]].astype("datetime64[M]"))
            days.append((rows, ranks, yy, month))
    if not days:
        raise ValueError("No eligible validation dates")
    for first in range(0, len(candidates), batch_size):
        batch = candidates[first:first + batch_size]
        predictions = X @ np.column_stack([item["coefficient"] for item in batch]) + np.array([item["intercept"] for item in batch])
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite validation predictions")
        total, month_sums, month_counts = np.zeros(len(batch)), {}, {}
        for rows, ranks, yy, month in days:
            xr = rankdata(predictions[rows], method="average", axis=0)
            xr -= xr.mean(axis=0)
            denominator = np.sqrt(np.sum(xr * xr, axis=0) * yy)
            scores = np.divide(ranks @ xr, denominator, out=np.zeros(len(batch)), where=denominator > 0)
            total += scores
            month_sums[month] = month_sums.get(month, np.zeros(len(batch))) + scores
            month_counts[month] = month_counts.get(month, 0) + 1
        for j, candidate in enumerate(batch):
            candidate.update(validation_ic=float(total[j] / len(days)), validation_days=len(days),
                validation_sum=float(total[j]), month_sum={month: float(values[j]) for month, values in month_sums.items()},
                month_count=dict(month_counts))
    return candidates


def selection_stability(options):
    """A leave-one-calendar-month-out diagnostic; never changes forecast selection."""
    chosen = legacy.best_candidate(options)
    ordered_scores = sorted((item["validation_ic"] for item in options), reverse=True)
    runner_up_gap = float(ordered_scores[0] - ordered_scores[1]) if len(ordered_scores) > 1 else None
    months = sorted(set().union(*(item["month_count"] for item in options)))
    rows = []
    for month in months:
        remaining = []
        for item in options:
            count = item["validation_days"] - item["month_count"].get(month, 0)
            if count:
                score = (item["validation_sum"] - item["month_sum"].get(month, 0.)) / count
                remaining.append((item, score, count))
        if not remaining:
            rows.append({"omitted_month": month, "winner_candidate_id": None, "remaining_days": 0, "remaining_ic": None})
            continue
        maximum = max(score for _, score, _ in remaining)
        winner, score, count = next(item for item in remaining if item[1] >= maximum - linear.TIE_TOL)
        scores = sorted((value for _, value, _ in remaining), reverse=True)
        rows.append({"omitted_month": month, "winner_candidate_id": winner["candidate_id"],
                     "remaining_days": int(count), "remaining_ic": float(score),
                     "runner_up_gap": float(scores[0] - scores[1]) if len(scores) > 1 else None,
                     "same_as_full": winner["candidate_id"] == chosen["candidate_id"]})
    valid = [row for row in rows if row["winner_candidate_id"] is not None]
    return {"diagnostic_only": True, "full_candidate_id": chosen["candidate_id"],
        "month_count": len(months), "omissions": rows, "runner_up_gap": runner_up_gap,
        "same_winner_fraction": float(np.mean([row["same_as_full"] for row in valid])) if valid else None,
        "distinct_winners": len({row["winner_candidate_id"] for row in valid})}


def procedure_key(spec):
    return tuple(spec[field] for field in ("feature_set", "estimator", "representation", "penalty_grid", "pca_grid"))


def fit_menu(bundle, block, transform, pca_menus=None):
    pca_menus = PCA_MENUS if pca_menus is None else pca_menus
    moments = block.get("shared_moments")
    if moments is None:
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
    pca = legacy.pca_state(bundle, block)
    all_candidates = {}
    for feature in FEATURE_SETS:
        dimensions = (None,) + tuple(sorted(set().union(*pca_menus.values()))) if feature == UNION else (None,)
        for dimension in dimensions:
            matrix, groups, _ = legacy.representation_map(bundle, feature, pca, dimension)
            represented = legacy.mapped_moments(moments, matrix)
            original = linear.fit_candidates(represented, np.arange(matrix.shape[1]))
            expanded = fit_candidates(represented, RIDGE_EXPANDED, FRACTIONS_EXPANDED, RATIOS_EXPANDED)
            if feature != "characteristics":
                original += legacy.group_ridge_candidates(represented, groups)
                expanded += group_ridge_candidates(represented, groups, RIDGE_EXPANDED, TEXT_EXPANDED)
            # Original numerical solutions are the common candidates in both menus.
            shared = {}
            for original_grid, menu in ((True, original), (False, expanded)):
                for candidate in menu:
                    candidate["pca_dim"] = dimension
                    key = setting_key(candidate)
                    if key not in shared:
                        candidate["original_penalty"] = original_grid
                        shared[key] = candidate
            candidates = list(shared.values())
            for candidate in candidates:
                candidate["coefficient"] = matrix @ candidate["coefficient"]
                candidate["selected_fit_days"] = 504
                identity = (feature, transform, setting_key(candidate))
                candidate["candidate_id"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
            score_candidates(candidates, block["X_valid"], block["y_valid"], block["codes_valid"], bundle["calendar"])
            for candidate in candidates:
                key = (feature, candidate["estimator"], "full" if dimension is None else "pca")
                all_candidates.setdefault(key, []).append(candidate)
    selected, stability = {}, {}
    for spec in (spec for spec in procedure_specs() if spec["transform"] == transform):
        base = tuple(spec[field] for field in ("feature_set", "estimator", "representation"))
        options = [candidate for candidate in all_candidates[base]
                   if (spec["penalty_grid"] == "expanded" or candidate["original_penalty"])
                   and (spec["pca_grid"] == "none" or candidate["pca_dim"] in pca_menus[spec["pca_grid"]])]
        options.sort(key=candidate_order)
        key = procedure_key(spec)
        selected[key] = copy.deepcopy(legacy.best_candidate(options))
        stability[key] = selection_stability(options)
    diagnostics = {"grid": {"|".join(key): [legacy.compact_candidate(item) for item in options]
                            for key, options in all_candidates.items()},
        "fit_rows": len(block["X_fit"]), "fit_dates": len(np.unique(block["codes_fit"])),
        "validation_rows": len(block["X_valid"]), "validation_dates": len(np.unique(block["codes_valid"])),
        "pca_eigenvalues": pca["eigenvalues"], "pca_input_rows": pca["input_rows"],
        "mean": block["mean"], "scale": block["scale"]}
    return selected, stability, diagnostics


def fit_month(bundle, tasks, target, pca_menus=None):
    if len(tasks) != 1 or int(tasks[0]["fit_days"]) != 504:
        raise ValueError("Exactly one fixed504 task is required")
    task, specs = tasks[0], procedure_specs()
    diagnostics, window_diagnostics, test_indices, values = {}, {"504": {}}, None, None
    coefficient_features = [bundle["manifest"]["feature_names"][i] for i in data.feature_columns(bundle, UNION)]
    for transform in TRANSFORMS:
        print(f"FIT {target} {task['month']} {transform}: training and validation search", flush=True)
        block = build_block(bundle, task, target, transform=transform)
        selected, stability, window_diagnostics["504"][transform] = fit_menu(bundle, block, transform, pca_menus)
        if test_indices is None:
            test_indices = block["test_indices"]
            values = np.empty((len(test_indices), len(specs)))
        elif not np.array_equal(test_indices, block["test_indices"]):
            raise ValueError("Transformation forecast keys differ")
        del block
        print(f"REFIT {target} {task['month']} {transform}: selected settings on training plus validation", flush=True)
        block = build_block(bundle, task, target, refit="union", transform=transform)
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        pca, cache = legacy.pca_state(bundle, block), {}
        for j, spec in enumerate(specs):
            if spec["transform"] != transform:
                continue
            key = procedure_key(spec)
            chosen = selected[key]
            name = model_name(spec, target)
            diagnostics[name] = {"selected": legacy.compact_candidate(chosen), "procedure": spec,
                "stability": stability[key], "features": [bundle["manifest"]["feature_names"][i]
                    for i in data.feature_columns(bundle, spec["feature_set"])]}
            cache_key = (spec["feature_set"], setting_key(chosen))
            if cache_key not in cache:
                matrix, groups, names = legacy.representation_map(bundle, spec["feature_set"], pca, chosen["pca_dim"])
                fitted = legacy.fit_selected(legacy.mapped_moments(moments, matrix), chosen, groups)
                beta = matrix @ fitted["coefficient"]
                prediction = block["X_test"] @ beta + fitted["intercept"]
                input_codes = np.asarray(bundle["codes"])[block["input_indices"]]
                meta = {**legacy.compact_candidate(fitted), "refit": "union", "fit_rows": len(block["X_fit"]),
                    "fit_dates": int(len(np.unique(block["codes_fit"]))), "input_dates": int(len(np.unique(input_codes))),
                    "input_first": int(input_codes.min()), "input_last": int(input_codes.max()),
                    "coefficient": beta, "intercept": fitted["intercept"], "coefficient_features": coefficient_features,
                    "mean": block["mean"], "scale": block["scale"], "represented_features": names}
                cache[cache_key] = prediction, meta
            values[:, j], diagnostics[name]["final"] = cache[cache_key]
        del block
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite monthly predictions")
    return {"month": task["month"], "target": target, "test_indices": test_indices,
        "predictions": values, "model_names": [model_name(spec, target) for spec in specs],
        "diagnostics": diagnostics, "window_diagnostics": window_diagnostics}
