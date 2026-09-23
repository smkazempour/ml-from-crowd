"""Stage A: named products of certified daily ranks and fitting-only text PCs.

The nine additive controls retain the frozen v2 matrix/scaling path. All other
procedures share one 604-column derived matrix and sufficient statistics, rather
than keeping separate stock-day matrices for every information set.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np

try:
    from . import linear_design_core as legacy, protocol_data as data, protocol_linear as linear
    from .characteristic_data import CONTROL_NAMES, MISSING_NAMES
except ImportError:
    import linear_design_core as legacy
    import protocol_data as data
    import protocol_linear as linear
    from characteristic_data import CONTROL_NAMES, MISSING_NAMES

UNION = legacy.UNION
FEATURE_SETS = legacy.FEATURE_SETS
ESTIMATORS = ("ols", "ridge", "enet")
SOCIAL_NAMES = ("net_sentiment", "log_volume")
AGREEMENT_NAMES = ("embed_norm", "embed_cos")
PCA_DIM = 16
BASES = (
    ("additive_c", 34, "characteristics"),
    ("additive_cs", 36, "characteristics_core"),
    ("additive_cst_full", 422, UNION),
    ("c_squares", 51, "characteristics"),
    ("c_quadratic", 187, "characteristics"),
    ("cq_s", 189, "characteristics_core"),
    ("cq_s_squares", 191, "characteristics_core"),
    ("cs_joint", 226, "characteristics_core"),
    ("text_main", 244, UNION),
    ("text_squares", 262, UNION),
    ("agreement_c", 296, UNION),
    ("text_c", 568, UNION),
    ("text_s", 604, UNION),
)
SOCIAL_INTERACTIONS = tuple(f"{social}__x__{control}" for social in SOCIAL_NAMES for control in CONTROL_NAMES) + (
    "net_sentiment__x__log_volume",)


def procedure_specs():
    result = []
    for basis, count, feature in BASES:
        additive = basis.startswith("additive_")
        text = feature == UNION and not additive
        for estimator in ESTIMATORS:
            result.append(dict(basis=basis, n_features=count, interaction_term=None,
                estimator=estimator, feature_set=feature,
                representation="full" if additive else "pca_interactions" if text else "explicit",
                transform="standard", penalty_grid="original", pca_grid="fixed16" if text else "none",
                history="fixed504", refit="union"))
    for term in SOCIAL_INTERACTIONS:
        result.append(dict(basis="single_social192", n_features=192, interaction_term=term,
            estimator="ridge", feature_set="characteristics_core", representation="explicit",
            transform="standard", penalty_grid="original", pca_grid="none", history="fixed504", refit="union"))
    return result


def model_name(spec, target):
    return (f"li1_{spec['estimator']}_{spec['basis']}_{spec.get('interaction_term') or 'joint'}_"
            f"{target}_fixed504_union")


def legacy_spec(spec):
    if not spec["basis"].startswith("additive_"):
        return None
    return dict(feature_set=spec["feature_set"], estimator=spec["estimator"],
                representation="full", history="fixed504", refit="union")


def grid_specification():
    global_grid = linear.grid_specification()
    return dict(fit_days=504, validation_days=126, refit="union; preserve original interior purge", horizon=1,
        estimators=list(ESTIMATORS), ridge_alphas=list(linear.RIDGE_ALPHAS),
        enet_alpha_fractions=list(linear.ALPHA_FRACTIONS), enet_l1_ratios=list(linear.L1_RATIOS),
        solver={key: value for key, value in global_grid.items() if key.startswith(("solver", "kkt", "coordinate", "active", "normalized", "eigen"))},
        objective=global_grid["objective"], selection="maximum full-validation mean daily Spearman IC;first declared within1e-12",
        tie_order="ridge alpha descending;elastic-net fraction descending then l1_ratio ascending",
        feature_source="certified cached C and S daily ranks, already transformed to approximately[-1,1];no second ranking",
        missingness="C and S cached neutral-zero ranks;17 C missingness flags retained as unscaled main effects only;"
                    "no flag products;raw agreement missingness imputed with original fitting-input weighted mean before products",
        characteristic_names=list(CONTROL_NAMES), social_names=list(SOCIAL_NAMES),
        agreement_names=list(AGREEMENT_NAMES), social_interactions=list(SOCIAL_INTERACTIONS),
        pca=dict(dimensions=PCA_DIM, fit_rows="all original fitting inputs before outcome filtering;rebuild on union",
            transformation="legacy fit-only embedding standardization and equal-date covariance;"
                "subtract PCA fitting mean,project deterministic eigenvectors,divide by component fitting SD",
            agreement="the two raw agreement variables remain outside PCA"),
        products="construct squares and distinct products from cached C/S ranks,centered/scaled16 PCs,and raw/imputed agreements;"
                 "then fit-only equal-date standardize derived continuous columns;never rerank products",
        final_refit="rebuild PCA,derived means/scales,and sparse alpha_max on original504+126 input dates;"
                    "retain validation-selected alpha or alpha_fraction/l1_ratio",
        additive_controls="nine unchanged v2 fixed504/union full-representation paths;independent of604-column scaling",
        bases=[dict(basis=basis,n_features=count,feature_set=feature) for basis,count,feature in BASES],
        single_basis="191-column cq_s_squares plus exactly one of35 named social products",
        model_count_per_target=len(procedure_specs()))


def basis_schema(bundle):
    """The 604-column union; the full-embedding additive baseline has its own path."""
    source_names = bundle["manifest"]["feature_names"]
    required = list(CONTROL_NAMES) + list(MISSING_NAMES) + list(SOCIAL_NAMES) + list(AGREEMENT_NAMES)
    if len(set(source_names)) != len(source_names) or any(name not in source_names for name in required):
        raise ValueError("Expected seventeen named characteristics/flags, sentiment, attention and agreement inputs")
    characteristic_names = [source_names[i] for i in data.feature_columns(bundle, "characteristics")]
    if set(characteristic_names) != set(CONTROL_NAMES + MISSING_NAMES):
        raise ValueError("Characteristic main effects must be exactly17 continuous inputs and17 flags")
    names = list(CONTROL_NAMES) + list(MISSING_NAMES)
    definitions = [("source", name) for name in names]

    def square(name):
        names.append(f"{name}__sq")
        definitions.append(("square", name))

    def product(first, second):
        names.append(f"{first}__x__{second}")
        definitions.append(("product", first, second))

    for name in CONTROL_NAMES:
        square(name)
    for first, left in enumerate(CONTROL_NAMES):
        for right in CONTROL_NAMES[first + 1:]:
            product(left, right)
    names += list(SOCIAL_NAMES)
    definitions += [("source", name) for name in SOCIAL_NAMES]
    for name in SOCIAL_NAMES:
        square(name)
    interactions = {}
    for social in SOCIAL_NAMES:
        for control in CONTROL_NAMES:
            interactions[f"{social}__x__{control}"] = len(names)
            product(social, control)
    interactions[SOCIAL_INTERACTIONS[-1]] = len(names)
    product(*SOCIAL_NAMES)
    pc_names = [f"embedding_pc_{i + 1:03d}" for i in range(PCA_DIM)]
    text_names = pc_names + list(AGREEMENT_NAMES)
    names += text_names
    definitions += [("pc", i) for i in range(PCA_DIM)] + [("source", name) for name in AGREEMENT_NAMES]
    for name in text_names:
        square(name)
    for agreement in AGREEMENT_NAMES:
        for control in CONTROL_NAMES:
            product(agreement, control)
    for pc in pc_names:
        for control in CONTROL_NAMES:
            product(pc, control)
    for text in text_names:
        for social in SOCIAL_NAMES:
            product(text, social)
    if len(names) != 604 or len(set(names)) != 604:
        raise ValueError("Derived feature union must contain604 unique columns")
    columns = {basis: list(range(count)) for basis,count,_ in BASES if not basis.startswith("additive_")}
    columns["additive_c"] = list(range(34))
    columns["additive_cs"] = list(range(34)) + [names.index(name) for name in SOCIAL_NAMES]
    return dict(feature_names=names, definitions=definitions, columns=columns,
                interactions=interactions, characteristic_names=list(CONTROL_NAMES), text_names=text_names)


def derived_values(bundle, block, indices, pca, schema=None):
    """Build products before fitting-scale transformations; no labels are read."""
    schema = basis_schema(bundle) if schema is None else schema
    source_names = bundle["manifest"]["feature_names"]
    block_lookup = {name: j for j,name in enumerate(block["feature_names"])}
    names = list(CONTROL_NAMES) + list(MISSING_NAMES) + list(SOCIAL_NAMES) + list(AGREEMENT_NAMES)
    columns = [source_names.index(name) for name in names]
    raw = np.asarray(bundle["X"][np.ix_(indices, columns)])
    source = {name: raw[:,j] for j,name in enumerate(names)}
    # The certified cache's rank imputation is zero, not a fitting-sample mean.
    for name in list(CONTROL_NAMES) + list(SOCIAL_NAMES):
        source[name] = np.where(np.isfinite(source[name]), source[name], 0.)
    for name in MISSING_NAMES:
        if not np.isin(source[name], (0.,1.)).all():
            raise ValueError("Characteristic missingness indicators must remain binary")
    for control, missing in zip(CONTROL_NAMES, MISSING_NAMES):
        if np.any(source[control][source[missing] == 1] != 0):
            raise ValueError("A missing characteristic must carry the certified neutral-zero rank")
    for name in AGREEMENT_NAMES:
        source[name] = np.where(np.isfinite(source[name]), source[name], block["mean"][block_lookup[name]])
    embedding = pca["embedding_columns"]
    if len(embedding) < PCA_DIM:
        raise ValueError("At least16 embedding coordinates are required")
    embedding_source = np.asarray(data.feature_columns(bundle, UNION))[embedding]
    pcs = np.empty((len(indices),PCA_DIM),dtype=float)
    for first in range(0,len(indices),32768):
        rows = indices[first:first+32768]
        standardized = data._transform(bundle["X"][np.ix_(rows,embedding_source)],
                                       block["mean"][embedding],block["scale"][embedding]).astype(float)
        standardized -= pca["mean"]
        pcs[first:first+len(rows)] = (standardized @ pca["components"][:,:PCA_DIM]) / pca["scale"][:PCA_DIM]
    del standardized
    for j in range(PCA_DIM):
        source[f"embedding_pc_{j + 1:03d}"] = pcs[:,j]
    result = np.empty((len(indices), len(schema["feature_names"])), dtype=np.float32)
    for j, definition in enumerate(schema["definitions"]):
        operation, first, *rest = definition
        if operation == "source":
            result[:,j] = source[first]
        elif operation == "pc":
            result[:,j] = pcs[:,first]
        elif operation == "square":
            result[:,j] = source[first] ** 2
        else:
            result[:,j] = source[first] * source[rest[0]]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite derived features")
    return result


def build_derived_block(bundle, task, target, refit="retain", source_block=None):
    if refit not in ("retain", "union"):
        raise ValueError("Stage A supports only original tuning and union refit inputs")
    block = legacy.build_block(bundle, task, target, refit) if source_block is None else source_block
    if source_block is None:
        for part in ("fit","valid","test"):
            block.pop(f"X_{part}",None)
        block.pop("shared_moments",None)
    schema = basis_schema(bundle)
    pca = legacy.pca_state(bundle, block)
    indices = block["input_indices"]
    codes = np.asarray(bundle["codes"])
    raw = derived_values(bundle, block, indices, pca, schema)
    binary = [name in MISSING_NAMES for name in schema["feature_names"]]
    mean, scale, constant, missing = data._weighted_scaler(raw, codes[indices], binary)
    result = dict(mean=mean, scale=scale, constant=constant, all_missing=missing,
        feature_names=schema["feature_names"], input_indices=indices, split=dict(task),
        schema=schema, pca=pca, parent_mean=block["mean"], parent_scale=block["scale"],
        agreement_imputation={name:float(block["mean"][block["feature_names"].index(name)]) for name in AGREEMENT_NAMES})
    selections = {"fit": indices, "test": block["test_indices"]}
    if refit == "retain":
        selections["valid"] = np.flatnonzero((codes >= task["valid_first"]) & (codes <= task["valid_last"]))
    target_index = bundle["manifest"]["targets"].index(linear.TARGET_COLUMNS.get(target,target))
    for part, chosen in selections.items():
        source = raw if part == "fit" else derived_values(bundle, block, chosen, pca, schema)
        transformed = transform_inplace(source,mean,scale)
        q = np.asarray(bundle["q"][chosen,target_index])
        if part != "test":
            eligible = np.isfinite(q)
            chosen,q,transformed = chosen[eligible],q[eligible],compact_rows_inplace(transformed,eligible)
            if not len(chosen):
                raise ValueError(f"No eligible {part} outcomes")
            result[f"w_{part}"] = data.equal_date_weights(codes[chosen])
        result.update({f"X_{part}":transformed, f"y_{part}":q, f"codes_{part}":codes[chosen], f"{part}_indices":chosen})
    return result


def transform_inplace(values,mean,scale,chunk_size=32768):
    """The frozen float64-to-float32 scaling arithmetic without a second big matrix."""
    if values.dtype != np.float32 or not values.flags.writeable:
        raise ValueError("Expected an owned writable float32 derived matrix")
    for first in range(0,len(values),chunk_size):
        last = min(first+chunk_size,len(values))
        chunk = values[first:last].astype(np.float64)
        chunk -= mean
        chunk /= scale
        values[first:last] = chunk
    return values


def compact_rows_inplace(values,keep,chunk_size=32768):
    """Stable forward compaction; destination positions never overtake unread rows."""
    positions = np.flatnonzero(keep)
    if len(positions) == len(values):
        return values
    for first in range(0,len(positions),chunk_size):
        selected = positions[first:first+chunk_size]
        values[first:first+len(selected)] = values[selected]
    # Keep a view of the one original allocation instead of allocating another3GB.
    return values[:len(positions)]


def subset_moments(moments, columns):
    columns = np.asarray(columns,dtype=int)
    return {"gram":moments["gram"][np.ix_(columns,columns)], "cross":moments["cross"][columns],
        "x_mean":moments["x_mean"][columns], "y_mean":moments["y_mean"], "y_variance":moments["y_variance"]}


def fit_candidates(moments, estimators=ESTIMATORS):
    """Fit only declared OLS/ridge/enet candidates; certify the original objective."""
    gram,cross = moments["gram"],moments["cross"]
    eigen,basis = np.linalg.eigh(gram)
    cutoff = max(float(eigen.max(initial=0)),1.) * linear.EIGEN_RCOND
    if eigen.min(initial=0) < -cutoff*10:
        raise ValueError("Weighted Gram is not positive semidefinite")
    eigen = np.maximum(eigen,0)
    active,projected = eigen > cutoff,basis.T @ cross
    result = []

    def append(estimator,beta,alpha=0.,ratio=0.,fraction=None,iterations=0,polish=0):
        violation = linear.kkt_violation(gram,cross,beta,alpha,ratio)
        if not np.isfinite(beta).all() or violation > linear.KKT_TOL:
            raise RuntimeError(f"{estimator} stationarity certificate failed:{violation}")
        gap = linear.objective_gap(gram,cross,beta,alpha,ratio,moments["y_variance"]) if ratio else None
        if gap is not None and gap > max(linear.SOLVER_TOL*moments["y_variance"],1e-12):
            raise RuntimeError("Elastic-net objective certificate failed")
        result.append(dict(estimator=estimator,coefficient=beta,
            intercept=float(moments["y_mean"]-moments["x_mean"]@beta),alpha=float(alpha),
            l1_ratio=float(ratio),alpha_fraction=fraction,iterations=int(iterations),
            polish_iterations=int(polish),coordinate_descent_limit=bool(iterations>=linear.MAX_ITER),
            dual_gap=gap,kkt_violation=violation))
    if "ols" in estimators:
        append("ols",basis @ np.divide(projected,eigen,out=np.zeros_like(eigen),where=active))
    if "ridge" in estimators:
        for alpha in linear.RIDGE_ALPHAS:
            append("ridge",basis @ (projected/(eigen+alpha)),alpha)
    if "enet" not in estimators:
        return result
    maximum = float(np.max(np.abs(cross),initial=0))
    if maximum <= np.finfo(float).eps:
        for fraction in linear.ALPHA_FRACTIONS:
            for ratio in linear.L1_RATIOS:
                append("enet",np.zeros(len(cross)),ratio=ratio,fraction=fraction)
        return result
    p,n = len(cross),len(cross)+1
    response = np.divide(projected,np.sqrt(eigen),out=np.zeros_like(eigen),where=active)
    remaining = float(moments["y_variance"]-response@response)
    if remaining < -1e-10:
        raise ValueError("Gram projection exceeds target variance")
    surrogate = np.zeros((n,p),dtype=float,order="F")
    surrogate[:p] = np.sqrt(n*eigen)[:,None]*basis.T
    response = np.r_[np.sqrt(n)*response,np.sqrt(n*max(remaining,0.))]
    paths = {}
    for ratio in linear.L1_RATIOS:
        alphas = maximum/ratio*np.asarray(linear.ALPHA_FRACTIONS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",linear.ConvergenceWarning)
            _,coefficients,_,iterations = linear.enet_path(surrogate,response,l1_ratio=ratio,alphas=alphas,
                precompute=True,tol=linear.SOLVER_TOL,max_iter=linear.MAX_ITER,selection="cyclic",return_n_iter=True)
        if any(not issubclass(warning.category,linear.ConvergenceWarning) for warning in caught):
            raise RuntimeError("Unexpected solver warning")
        paths[ratio] = alphas,coefficients,iterations
    for j,fraction in enumerate(linear.ALPHA_FRACTIONS):
        for ratio in linear.L1_RATIOS:
            alphas,coefficients,iterations = paths[ratio]
            beta,steps = linear.polish_solution(gram,cross,coefficients[:,j],alphas[j],ratio)
            append("enet",beta,alphas[j],ratio,fraction,iterations[j],steps)
    return result


def specification_columns(schema,spec):
    if spec["basis"] == "single_social192":
        return schema["columns"]["cq_s_squares"] + [schema["interactions"][spec["interaction_term"]]]
    return schema["columns"][spec["basis"]]


def fit_additive_menu(bundle,block):
    moments = block.get("shared_moments")
    if moments is None:
        moments = linear.weighted_moments(block["X_fit"],block["y_fit"],block["w_fit"])
    selected,grid = {},{}
    for feature in FEATURE_SETS:
        matrix,groups,_ = legacy.representation_map(bundle,feature)
        represented = legacy.mapped_moments(moments,matrix)
        candidates = linear.fit_candidates(represented,np.arange(matrix.shape[1]))
        if feature != "characteristics":
            # This reproduces v2's final global-candidate batch width exactly.
            # Group candidates are numerical controls, not Stage A procedures.
            candidates += legacy.group_ridge_candidates(represented,groups)
        for candidate in candidates:
            candidate["coefficient"] = matrix@candidate["coefficient"]
            candidate.update(pca_dim=None,selected_fit_days=504)
        # Retain the original complete candidate ordering/batches for numerical identity.
        legacy.score_candidates(candidates,block["X_valid"],block["y_valid"],block["codes_valid"])
        for estimator in ESTIMATORS:
            options = [candidate for candidate in candidates if candidate["estimator"] == estimator]
            key = (feature,estimator)
            selected[key] = copy.deepcopy(legacy.best_candidate(options))
            grid["|".join(key)] = [legacy.compact_candidate(item) for item in options]
    return selected,grid


def fit_derived_menu(block,specs):
    moments = linear.weighted_moments(block["X_fit"],block["y_fit"],block["w_fit"])
    selected,grid = {},{}
    grouped = {}
    for spec in specs:
        if legacy_spec(spec) is None:
            grouped.setdefault((spec["basis"],spec["interaction_term"]),[]).append(spec)
    for key,requests in grouped.items():
        columns = specification_columns(block["schema"],requests[0])
        estimators = tuple(spec["estimator"] for spec in requests)
        menu = fit_candidates(subset_moments(moments,columns),estimators)
        # Predict in the represented matrix; storage is bounded by16 candidates.
        legacy.score_candidates(menu,block["X_valid"][:,columns],block["y_valid"],block["codes_valid"])
        for candidate in menu:
            candidate.update(selected_fit_days=504,pca_dim=PCA_DIM if requests[0]["pca_grid"] == "fixed16" else None)
        for spec in requests:
            options = [candidate for candidate in menu if candidate["estimator"] == spec["estimator"]]
            chosen = copy.deepcopy(legacy.best_candidate(options))
            name = model_name(spec,"selection")
            selected[name] = chosen
            grid[name] = [legacy.compact_candidate(item) for item in options]
    return selected,grid


def block_summary(block):
    return dict(fit_rows=len(block["X_fit"]),fit_dates=int(len(np.unique(block["codes_fit"]))),
        validation_rows=len(block.get("X_valid",[])),
        validation_dates=int(len(np.unique(block.get("codes_valid",[])))),
        input_rows=len(block["input_indices"]), mean=block["mean"],scale=block["scale"])


def final_metadata(bundle,block,fitted,names,pca=None):
    input_codes = np.asarray(bundle["codes"])[block["input_indices"]]
    result = {**legacy.compact_candidate(fitted),"refit":"union","fit_rows":len(block["X_fit"]),
        "fit_dates":int(len(np.unique(block["codes_fit"]))),"input_dates":int(len(np.unique(input_codes))),
        "input_first":int(input_codes.min()),"input_last":int(input_codes.max()),
        "coefficient":fitted["coefficient"],"intercept":fitted["intercept"],
        "coefficient_features":names,"mean":block["mean"],"scale":block["scale"]}
    if pca is not None:
        result.update(pca=pca,agreement_imputation=block["agreement_imputation"],
                      parent_mean=block["parent_mean"],parent_scale=block["parent_scale"])
    return result


def fit_month(bundle,tasks,target):
    if len(tasks) != 1 or tasks[0]["fit_days"] != 504 or tasks[0]["horizon"] != 1:
        raise ValueError("Stage A requires one fixed504,h1 task")
    task,specs = tasks[0],procedure_specs()
    diagnostics,window = {},{}
    source = legacy.build_block(bundle,task,target)
    additive,window["additive_grid"] = fit_additive_menu(bundle,source)
    window["additive"] = block_summary(source)
    test_indices = source["test_indices"]
    values = np.empty((len(test_indices),len(specs)))
    for part in ("fit","valid","test"):
        del source[f"X_{part}"]
    source.pop("shared_moments",None)
    derived = build_derived_block(bundle,task,target,source_block=source)
    del source
    selected,window["derived_grid"] = fit_derived_menu(derived,specs)
    window["derived"] = {**block_summary(derived),"feature_names":derived["feature_names"],
        "pca":derived["pca"],"agreement_imputation":derived["agreement_imputation"],
        "parent_mean":derived["parent_mean"],"parent_scale":derived["parent_scale"]}
    del derived
    source = legacy.build_block(bundle,task,target,"union")
    moments = linear.weighted_moments(source["X_fit"],source["y_fit"],source["w_fit"])
    for j,spec in enumerate(specs):
        if legacy_spec(spec) is None:
            continue
        chosen = additive[(spec["feature_set"],spec["estimator"])]
        matrix,groups,names = legacy.representation_map(bundle,spec["feature_set"])
        fitted = legacy.fit_selected(legacy.mapped_moments(moments,matrix),chosen,groups)
        fitted["coefficient"] = matrix@fitted["coefficient"]
        values[:,j] = source["X_test"]@fitted["coefficient"]+fitted["intercept"]
        diagnostics[model_name(spec,target)] = dict(selected=legacy.compact_candidate(chosen),procedure=spec,
            features=names,final=final_metadata(bundle,source,fitted,source["feature_names"]))
    for part in ("fit","test"):
        del source[f"X_{part}"]
    derived = build_derived_block(bundle,task,target,"union",source_block=source)
    del source,moments
    moments = linear.weighted_moments(derived["X_fit"],derived["y_fit"],derived["w_fit"])
    for j,spec in enumerate(specs):
        if legacy_spec(spec) is not None:
            continue
        chosen = selected[model_name(spec,"selection")]
        columns = specification_columns(derived["schema"],spec)
        names = [derived["feature_names"][column] for column in columns]
        fitted = legacy.fit_selected(subset_moments(moments,columns),chosen,np.repeat("derived",len(columns)))
        values[:,j] = derived["X_test"][:,columns]@fitted["coefficient"]+fitted["intercept"]
        metadata = final_metadata(bundle,derived,fitted,names,
            pca=derived["pca"] if spec["pca_grid"] == "fixed16" else None)
        # Coefficients and transformation metadata have the same represented order.
        metadata["mean"],metadata["scale"] = derived["mean"][columns],derived["scale"][columns]
        diagnostics[model_name(spec,target)] = dict(selected=legacy.compact_candidate(chosen),procedure=spec,
                                                   features=names,final=metadata)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite Stage A predictions")
    return dict(month=task["month"],target=target,test_indices=test_indices,predictions=values,
        model_names=[model_name(spec,target) for spec in specs],diagnostics=diagnostics,window_diagnostics={"504":window})
