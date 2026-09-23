"""Fitting kernels for the prospective linear-design comparison.

Historical kernels are imported, never changed. All hyperparameters and PCA
dimensions are selected on the original validation dates. Final refits preserve
that selection and use only labels matured before the outer forecast month.
"""
from __future__ import annotations

import copy
import re
import warnings

import numpy as np

try:
    from . import protocol_data as data, protocol_linear as linear
except ImportError:
    import protocol_data as data
    import protocol_linear as linear


UNION = "characteristics_textcore"
HISTORICAL_UNION = "characteristics_textall"
FEATURE_SETS = ("characteristics", "characteristics_core", UNION)
HISTORIES = ("fixed504", "select")
REFITS = ("retain", "recent", "union")
FIT_WINDOWS = (504, 252, 756)
PCA_DIMS = (16, 32, 64, 128)
SOCIAL_MULTIPLIERS = (1., 10.)
TEXT_MULTIPLIERS = (1., 10., 100.)


def base_specs():
    result = []
    for feature in FEATURE_SETS:
        for representation in (("full", "pca") if feature == UNION else ("full",)):
            estimators = linear.ESTIMATORS + (("group_ridge",) if feature != "characteristics" else ())
            result.extend(dict(feature_set=feature, estimator=estimator,
                               representation=representation) for estimator in estimators)
    return result


def procedure_specs():
    return [{**base, "history": history, "refit": refit}
            for base in base_specs() for history in HISTORIES for refit in REFITS]


def model_name(spec, target):
    return (f"ld2_{spec['estimator']}_{spec['feature_set']}_{spec['representation']}_"
            f"{target}_{spec['history']}_{spec['refit']}")


def grid_specification():
    return {"global": linear.grid_specification(), "fit_windows": list(FIT_WINDOWS),
            "history_tie_order": list(FIT_WINDOWS), "validation_days": 126,
            "selection": "maximum mean daily validation Spearman IC; first declared within 1e-12",
            "group_ridge": {"base_alphas": list(linear.RIDGE_ALPHAS),
                "characteristic_multiplier": 1., "social_multipliers": list(SOCIAL_MULTIPLIERS),
                "text_multipliers": list(TEXT_MULTIPLIERS),
                "drop_order": ["none", "text", "social", "text_social"],
                "objective": "0.5*weighted_squared_error + 0.5*alpha*sum(group_multiplier*beta^2)",
                "dropped_blocks": "exactly zero coefficients, including both agreement inputs with text"},
            "pca_dimensions": list(PCA_DIMS),
            "pca": "equal-date covariance of fit-standardized embedding coordinates on all fitting inputs; "
                   "no outcomes; descending eigenvectors with deterministic signs; component SD scaling; "
                   "agreement inputs retained outside PCA; rebuild on final fitting inputs",
            "refits": {"retain": "original selected fitting coefficients",
                "recent": "most recent F input dates whose horizon label ends no later than cutoff",
                "union": "original F fitting dates union original126 validation dates; preserve interior purge"},
            "sparse_refit": "transfer alpha_fraction and l1_ratio; recompute alpha_max on final fitting rows",
            "loss": "unchanged rank target, equal date weights, unpenalized intercept"}


def local_columns(bundle, feature):
    inverse = {value: j for j, value in enumerate(data.feature_columns(bundle, UNION))}
    return np.asarray([inverse[value] for value in data.feature_columns(bundle, feature)], dtype=int)


def group_labels(bundle):
    n = len(data.feature_columns(bundle, UNION))
    groups = np.full(n, "text", dtype="U14")
    groups[local_columns(bundle, "characteristics_core")] = "social"
    groups[local_columns(bundle, "characteristics")] = "characteristic"
    return groups


def input_indices(bundle, task, refit="retain"):
    """Full input rows, before filtering missing outcomes, for a final fit."""
    codes = np.asarray(bundle["codes"])
    if refit == "retain":
        mask = (codes >= task["fit_first"]) & (codes <= task["fit_last"])
    elif refit == "union":
        mask = (((codes >= task["fit_first"]) & (codes <= task["fit_last"]))
                | ((codes >= task["valid_first"]) & (codes <= task["valid_last"])))
    elif refit == "recent":
        last = task["cutoff"] - task["horizon"]
        first = last - task["fit_days"] + 1
        mask = (codes >= first) & (codes <= last)
    else:
        raise ValueError(f"Unknown refit: {refit}")
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("Empty final fitting inputs")
    if np.max(codes[indices]) + task["horizon"] >= task["test_first"]:
        raise ValueError("Final fitting label reaches forecast month")
    return indices


def build_block(bundle, task, target, refit="retain"):
    """Fit-only scaling, with the original block path unchanged for retention."""
    if refit == "retain":
        # Matrix width/layout changes floating-point scaler reductions and BLAS
        # accumulation. Preserve the archived 473-column numerical path before
        # restricting inputs to this experiment's 422-column feature union.
        block = data.make_block(bundle, task, HISTORICAL_UNION, target)
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        inverse = {value: j for j, value in enumerate(data.feature_columns(bundle, HISTORICAL_UNION))}
        columns = np.asarray([inverse[value] for value in data.feature_columns(bundle, UNION)])
        block["shared_moments"] = {"gram": moments["gram"][np.ix_(columns, columns)],
            "cross": moments["cross"][columns], "x_mean": moments["x_mean"][columns],
            "y_mean": moments["y_mean"], "y_variance": moments["y_variance"]}
        for part in ("fit", "valid", "test"):
            block[f"X_{part}"] = np.ascontiguousarray(block[f"X_{part}"][:, columns])
        for field in ("mean", "scale", "constant", "all_missing"):
            block[field] = block[field][columns]
        block["feature_names"] = [block["feature_names"][j] for j in columns]
        block["input_indices"] = input_indices(bundle, task, refit)
        return block
    indices = input_indices(bundle, task, refit)
    columns = data.feature_columns(bundle, UNION)
    names = [bundle["manifest"]["feature_names"][j] for j in columns]
    codes = np.asarray(bundle["codes"])
    raw = np.asarray(bundle["X"][np.ix_(indices, columns)])
    binary = set(bundle["manifest"].get("binary_features", []))
    mean, scale, constant, missing = data._weighted_scaler(raw, codes[indices], [n in binary for n in names])
    transformed = data._transform(raw, mean, scale)
    del raw
    target_column = linear.TARGET_COLUMNS.get(target, target)
    target_index = bundle["manifest"]["targets"].index(target_column)
    q = np.asarray(bundle["q"][indices, target_index])
    eligible = np.isfinite(q)
    test = np.flatnonzero((codes >= task["test_first"]) & (codes <= task["test_last"]))
    result = dict(mean=mean, scale=scale, constant=constant, all_missing=missing,
                  feature_names=names, input_indices=indices, split=dict(task),
                  fit_indices=indices[eligible], X_fit=transformed[eligible], y_fit=q[eligible],
                  codes_fit=codes[indices[eligible]], w_fit=data.equal_date_weights(codes[indices[eligible]]),
                  test_indices=test, X_test=data._transform(bundle["X"][np.ix_(test, columns)], mean, scale))
    return result


def pca_state(bundle, block):
    """Unsupervised covariance uses every fitting input, including missing labels."""
    embedding = np.asarray([j for j, name in enumerate(block["feature_names"])
                            if re.fullmatch(r"embed_\d{3}", name)])
    if not len(embedding):
        raise ValueError("No embedding coordinates for PCA")
    source_columns = np.asarray(data.feature_columns(bundle, UNION))[embedding]
    indices = block["input_indices"]
    weights = data.equal_date_weights(np.asarray(bundle["codes"])[indices])
    mean = np.zeros(len(embedding))
    second = np.zeros((len(embedding), len(embedding)))
    for lo in range(0, len(indices), 32768):
        chosen = indices[lo:lo + 32768]
        raw = bundle["X"][np.ix_(chosen, source_columns)]
        # Same float32 transform as the historical model inputs.
        x = data._transform(raw, block["mean"][embedding], block["scale"][embedding]).astype(float)
        w = weights[lo:lo + len(chosen)]
        mean += w @ x
        second += x.T @ (w[:, None] * x)
    covariance = second - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(.5 * (covariance + covariance.T))
    order = np.argsort(values, kind="stable")[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    for j in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, j]))
        if vectors[pivot, j] < 0:
            vectors[:, j] *= -1
    scales = np.sqrt(values)
    scales[values <= 1e-20] = 1.
    return dict(embedding_columns=embedding, mean=mean, eigenvalues=values,
                components=vectors, scale=scales,
                input_rows=len(indices), input_dates=int(len(np.unique(np.asarray(bundle["codes"])[indices]))))


def representation_map(bundle, feature, pca=None, dimension=None):
    """Map represented coefficients back to the unchanged full standardized inputs."""
    n = len(data.feature_columns(bundle, UNION))
    columns = local_columns(bundle, feature)
    groups = group_labels(bundle)
    if dimension is None:
        return np.eye(n)[:, columns], groups[columns], [bundle["manifest"]["feature_names"][i]
                                                      for i in data.feature_columns(bundle, feature)]
    if feature != UNION or pca is None or dimension > len(pca["embedding_columns"]):
        raise ValueError("Invalid PCA representation")
    nonembedding = np.asarray([j for j in columns if j not in set(pca["embedding_columns"])])
    matrix = np.zeros((n, len(nonembedding) + dimension))
    matrix[nonembedding, np.arange(len(nonembedding))] = 1.
    matrix[pca["embedding_columns"], len(nonembedding):] = pca["components"][:, :dimension] / pca["scale"][:dimension]
    names = [bundle["manifest"]["feature_names"][data.feature_columns(bundle, UNION)[j]] for j in nonembedding]
    names.extend(f"embedding_pc_{j + 1:03d}" for j in range(dimension))
    return matrix, np.r_[groups[nonembedding], np.repeat("text", dimension)], names


def mapped_moments(moments, matrix):
    return {"gram": matrix.T @ moments["gram"] @ matrix,
            "cross": matrix.T @ moments["cross"], "x_mean": moments["x_mean"] @ matrix,
            "y_mean": moments["y_mean"], "y_variance": moments["y_variance"]}


def group_ridge_candidates(moments, groups):
    """Solve each diagonal-penalty path once, including exact smaller-model paths."""
    groups = np.asarray(groups)
    has_text = bool(np.any(groups == "text"))
    drops = ("none", "text", "social", "text_social") if has_text else ("none", "social")
    candidates = []
    for drop in drops:
        keep = np.ones(len(groups), dtype=bool)
        if "text" in drop:
            keep &= groups != "text"
        if "social" in drop:
            keep &= groups != "social"
        columns = np.flatnonzero(keep)
        social_menu = SOCIAL_MULTIPLIERS if np.any(groups[columns] == "social") else (1.,)
        text_menu = TEXT_MULTIPLIERS if np.any(groups[columns] == "text") else (1.,)
        gram = moments["gram"][np.ix_(columns, columns)]
        cross = moments["cross"][columns]
        for social in social_menu:
            for text in text_menu:
                multipliers = np.where(groups[columns] == "social", social,
                                       np.where(groups[columns] == "text", text, 1.))
                whitening = 1. / np.sqrt(multipliers)
                eig, basis = np.linalg.eigh(gram * whitening[:, None] * whitening[None, :])
                eig = np.maximum(eig, 0)
                projected = basis.T @ (whitening * cross)
                for alpha in linear.RIDGE_ALPHAS:
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


def score_candidates(candidates, X, y, codes, batch_size=16):
    """Bound validation prediction storage to N times batch_size columns."""
    for first in range(0, len(candidates), batch_size):
        batch = candidates[first:first + batch_size]
        beta = np.column_stack([candidate["coefficient"] for candidate in batch])
        intercept = np.asarray([candidate["intercept"] for candidate in batch])
        scores, days = linear.validation_scores(X @ beta + intercept, y, codes)
        for candidate, score in zip(batch, scores):
            candidate.update(validation_ic=float(score), validation_days=int(days))
    return candidates


def best_candidate(options):
    maximum = max(item["validation_ic"] for item in options)
    return next(item for item in options if item["validation_ic"] >= maximum - linear.TIE_TOL)


def compact_candidate(candidate):
    return {key: value for key, value in candidate.items() if key not in ("coefficient", "intercept")}


def fit_menu(bundle, block, pca_dimensions=PCA_DIMS):
    """Return nineteen selected procedures, sharing moments and PCA across models."""
    moments = block.get("shared_moments")
    if moments is None:
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
    pca = pca_state(bundle, block)
    all_candidates = {}
    for feature in FEATURE_SETS:
        dimensions = (None,) + tuple(pca_dimensions) if feature == UNION else (None,)
        for dimension in dimensions:
            matrix, groups, names = representation_map(bundle, feature, pca, dimension)
            represented = mapped_moments(moments, matrix)
            candidates = linear.fit_candidates(represented, np.arange(matrix.shape[1]))
            if feature != "characteristics":
                candidates += group_ridge_candidates(represented, groups)
            for candidate in candidates:
                candidate["coefficient"] = matrix @ candidate["coefficient"]
                candidate.update(pca_dim=dimension, selected_fit_days=int(block["split"]["fit_days"]))
            score_candidates(candidates, block["X_valid"], block["y_valid"], block["codes_valid"])
            for candidate in candidates:
                key = (feature, candidate["estimator"], "full" if dimension is None else "pca")
                all_candidates.setdefault(key, []).append(candidate)
    selected = {key: copy.deepcopy(best_candidate(options)) for key, options in all_candidates.items()}
    diagnostics = {"grid": {"|".join(key): [compact_candidate(item) for item in options]
                            for key, options in all_candidates.items()},
                   "fit_rows": len(block["X_fit"]), "fit_dates": len(np.unique(block["codes_fit"])),
                   "validation_rows": len(block["X_valid"]),
                   "validation_dates": len(np.unique(block["codes_valid"])),
                   "pca_eigenvalues": pca["eigenvalues"], "pca_input_rows": pca["input_rows"],
                   "mean": block["mean"], "scale": block["scale"]}
    return selected, diagnostics


def fit_selected(moments, candidate, groups):
    """Refit one selected setting, with sparse alpha_max recomputed on these rows."""
    estimator = candidate["estimator"]
    gram, cross = moments["gram"], moments["cross"]
    if estimator == "group_ridge":
        groups = np.asarray(groups)
        keep = np.ones(len(groups), dtype=bool)
        for group in ("text", "social"):
            if group in candidate["drop_block"]:
                keep &= groups != group
        columns = np.flatnonzero(keep)
        multipliers = np.where(groups[columns] == "social", candidate["group_social_multiplier"],
                               np.where(groups[columns] == "text", candidate["group_text_multiplier"], 1.))
        hessian = gram[np.ix_(columns, columns)].copy()
        hessian.flat[::len(columns) + 1] += candidate["alpha"] * multipliers
        beta = np.zeros(len(groups))
        beta[columns] = np.linalg.solve(hessian, cross[columns])
        violation = float(np.max(np.abs(hessian @ beta[columns] - cross[columns])))
        if not np.isfinite(beta).all() or violation > linear.KKT_TOL:
            raise RuntimeError("Selected group ridge failed stationarity certificate")
        return {**compact_candidate(candidate), "coefficient": beta,
                "intercept": float(moments["y_mean"] - moments["x_mean"] @ beta),
                "kkt_violation": violation}
    eig, basis = np.linalg.eigh(gram)
    eig = np.maximum(eig, 0)
    cross_basis = basis.T @ cross
    active = eig > max(float(eig.max(initial=0)), 1.) * linear.EIGEN_RCOND
    alpha, ratio, iterations, polish_iterations = candidate["alpha"], candidate["l1_ratio"], 0, 0
    gap = None
    if estimator == "ols":
        beta = basis @ np.divide(cross_basis, eig, out=np.zeros_like(eig), where=active)
    elif estimator == "ridge":
        beta = basis @ (cross_basis / (eig + alpha))
    else:
        maximum = float(np.max(np.abs(cross), initial=0))
        alpha = float(candidate["alpha_fraction"] * maximum / ratio)
        if maximum <= np.finfo(float).eps:
            beta = np.zeros(len(cross))
            alpha = 0.
        else:
            p, effective_n = len(cross), len(cross) + 1
            projected = np.divide(cross_basis, np.sqrt(eig), out=np.zeros_like(eig), where=active)
            remaining = float(moments["y_variance"] - projected @ projected)
            if remaining < -1e-10:
                raise ValueError("Gram projection exceeds final target variance")
            surrogate_X = np.zeros((effective_n, p), dtype=float, order="F")
            surrogate_X[:p] = np.sqrt(effective_n * eig)[:, None] * basis.T
            surrogate_y = np.r_[np.sqrt(effective_n) * projected,
                                 np.sqrt(effective_n * max(remaining, 0.))]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", linear.ConvergenceWarning)
                _, coefficients, _, counts = linear.enet_path(surrogate_X, surrogate_y, l1_ratio=ratio,
                    alphas=[alpha], precompute=True, tol=linear.SOLVER_TOL, max_iter=linear.MAX_ITER,
                    selection="cyclic", return_n_iter=True)
            if any(not issubclass(item.category, linear.ConvergenceWarning) for item in caught):
                raise RuntimeError("Unexpected selected-model solver warning")
            beta, polish_iterations = linear.polish_solution(gram, cross, coefficients[:, 0], alpha, ratio)
            iterations = int(counts[0])
            gap = linear.objective_gap(gram, cross, beta, alpha, ratio, moments["y_variance"])
            if gap > max(linear.SOLVER_TOL * moments["y_variance"], 1e-12):
                raise RuntimeError("Selected sparse refit failed objective certificate")
    violation = linear.kkt_violation(gram, cross, beta, alpha, ratio)
    if not np.isfinite(beta).all() or violation > linear.KKT_TOL:
        raise RuntimeError("Selected refit failed stationarity certificate")
    return {**compact_candidate(candidate), "coefficient": beta,
            "intercept": float(moments["y_mean"] - moments["x_mean"] @ beta),
            "alpha": float(alpha), "kkt_violation": violation, "dual_gap": gap,
            "iterations": iterations, "polish_iterations": polish_iterations}


def fit_month(bundle, tasks, target, pca_dimensions=PCA_DIMS):
    """Fit one month/target; keep only one large history or refit block in RAM."""
    lookup = {int(task["fit_days"]): task for task in tasks}
    if set(lookup) != set(FIT_WINDOWS) or len(tasks) != 3 or len({t["month"] for t in tasks}) != 1:
        raise ValueError("A month requires exactly the three declared fitting histories")
    if len({(t["valid_first"], t["valid_last"], t["test_first"], t["test_last"], t["cutoff"]) for t in tasks}) != 1:
        raise ValueError("Histories must share validation and forecast dates")
    selected_by_window, retained, window_diagnostics = {}, {}, {}
    test_indices = None
    for fit_days in FIT_WINDOWS:
        block = build_block(bundle, lookup[fit_days], target)
        selected, window_diagnostics[str(fit_days)] = fit_menu(bundle, block, pca_dimensions)
        selected_by_window[fit_days] = selected
        retained[fit_days] = {key: block["X_test"] @ value["coefficient"] + value["intercept"]
                              for key, value in selected.items()}
        if test_indices is not None and not np.array_equal(test_indices, block["test_indices"]):
            raise ValueError("History forecast keys differ")
        test_indices = block["test_indices"]
        del block
    specs = procedure_specs()
    coefficient_features = [bundle["manifest"]["feature_names"][i]
                            for i in data.feature_columns(bundle, UNION)]
    values = np.empty((len(test_indices), len(specs)), dtype=float)
    diagnostics, pending = {}, {}
    for j, spec in enumerate(specs):
        key = (spec["feature_set"], spec["estimator"], spec["representation"])
        options = [selected_by_window[days][key] for days in (FIT_WINDOWS if spec["history"] == "select" else (504,))]
        chosen = best_candidate(options)
        fit_days = chosen["selected_fit_days"]
        name = model_name(spec, target)
        diagnostics[name] = {"selected": compact_candidate(chosen), "procedure": spec,
                             "features": [bundle["manifest"]["feature_names"][i]
                                          for i in data.feature_columns(bundle, spec["feature_set"])]}
        if spec["refit"] == "retain":
            values[:, j] = retained[fit_days][key]
            info = window_diagnostics[str(fit_days)]
            diagnostics[name]["final"] = {"refit": "retain", "fit_rows": info["fit_rows"],
                "fit_dates": info["fit_dates"], "alpha": chosen["alpha"],
                "coefficient": chosen["coefficient"], "coefficient_features": coefficient_features,
                "intercept": chosen["intercept"]}
        else:
            pending.setdefault((fit_days, spec["refit"]), []).append((j, spec, chosen))
    for (fit_days, refit), requests in pending.items():
        task = lookup[fit_days]
        block = build_block(bundle, task, target, refit)
        moments = linear.weighted_moments(block["X_fit"], block["y_fit"], block["w_fit"])
        pca = pca_state(bundle, block) if any(spec["representation"] == "pca" for _, spec, _ in requests) else None
        cache = {}
        for j, spec, chosen in requests:
            key = (spec["feature_set"], spec["estimator"], spec["representation"])
            if key not in cache:
                matrix, groups, represented_names = representation_map(bundle, spec["feature_set"], pca, chosen["pca_dim"])
                fitted = fit_selected(mapped_moments(moments, matrix), chosen, groups)
                beta = matrix @ fitted["coefficient"]
                prediction = block["X_test"] @ beta + fitted["intercept"]
                meta = {**compact_candidate(fitted), "refit": refit, "fit_rows": len(block["X_fit"]),
                    "fit_dates": int(len(np.unique(block["codes_fit"]))),
                    "input_dates": int(len(np.unique(np.asarray(bundle["codes"])[block["input_indices"]]))),
                    "input_first": int(np.min(np.asarray(bundle["codes"])[block["input_indices"]])),
                    "input_last": int(np.max(np.asarray(bundle["codes"])[block["input_indices"]])),
                    "coefficient": beta, "intercept": fitted["intercept"],
                    "coefficient_features": coefficient_features,
                    "mean": block["mean"], "scale": block["scale"],
                    "represented_features": represented_names}
                cache[key] = prediction, meta
            values[:, j], diagnostics[model_name(spec, target)]["final"] = cache[key]
        del block
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite monthly predictions")
    return {"month": tasks[0]["month"], "target": target, "test_indices": test_indices,
            "predictions": values, "model_names": [model_name(spec, target) for spec in specs],
            "diagnostics": diagnostics, "window_diagnostics": window_diagnostics}
