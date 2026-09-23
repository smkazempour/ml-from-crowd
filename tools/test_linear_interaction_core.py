"""Independent tests for named interactions, fitting boundaries and additive controls."""
import copy
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from threadpoolctl import threadpool_limits

import linear_interaction_core as core
import linear_design_core as legacy
import protocol_data as data
import protocol_linear as linear


def fixture():
    rng = np.random.default_rng(580341)
    calendar = pd.bdate_range("2020-01-01",periods=60).to_numpy()
    codes = np.repeat(np.arange(len(calendar)),24)
    names = list(core.SOCIAL_NAMES)+[f"other_social_{j}" for j in range(51)]
    names += [f"embed_{j:03d}" for j in range(384)]+list(core.AGREEMENT_NAMES)
    names += list(core.CONTROL_NAMES)+list(core.MISSING_NAMES)
    X = rng.normal(size=(len(codes),len(names))).astype(np.float32)
    for j,control in enumerate(core.CONTROL_NAMES):
        values = rng.normal(loc=.5,size=len(codes))
        missing = rng.uniform(size=len(codes)) < .06
        values[missing] = np.nan
        X[:,names.index(control)] = data.centered_rank(values,codes)
        X[:,names.index(core.MISSING_NAMES[j])] = missing
    X[:,0] = data.centered_rank(rng.choice([-1.,0.,1.],size=len(codes)),codes)
    X[:,1] = data.centered_rank(np.log1p(rng.integers(1,35,size=len(codes))),codes)
    X[:,names.index("embed_norm")] = rng.uniform(.1,.9,size=len(codes))
    X[:,names.index("embed_cos")] = rng.uniform(-.2,.5,size=len(codes))
    c0 = X[:,names.index(core.CONTROL_NAMES[0])]
    y = .05*c0+.08*c0*X[:,0]+rng.normal(size=len(codes))
    q = data.centered_rank(y,codes,target=True)
    q[codes == 8] = np.nan
    q[3] = np.nan
    C = [names.index(name) for name in list(core.CONTROL_NAMES)+list(core.MISSING_NAMES)]
    union = [0,1]+list(range(53,len(names)))
    bundle = dict(X=X,y=np.column_stack([y,y]),q=np.column_stack([q,q]),codes=codes,calendar=calendar,
        manifest=dict(feature_names=names,binary_features=list(core.MISSING_NAMES),targets=["f_cumret1","ar_dgtw_1"],
            feature_sets={"characteristics":C,"characteristics_core":[0,1]+C,
                          core.UNION:union,legacy.HISTORICAL_UNION:list(range(len(names)))}))
    task = dict(month="2020-03",fit_days=504,validation_days=126,horizon=1,
                fit_first=0,fit_last=35,valid_first=37,valid_last=47,cutoff=48,test_first=49,test_last=52)
    return bundle,task


class LinearInteractionCoreTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.limit.restore_original_limits()

    def test_counts_model_identity_and_parent_hierarchy(self):
        bundle,_ = fixture()
        specs,schema = core.procedure_specs(),core.basis_schema(bundle)
        self.assertEqual(len(specs),74)
        self.assertEqual(len({core.model_name(spec,"raw") for spec in specs}),74)
        self.assertEqual(sum(core.legacy_spec(spec) is not None for spec in specs),9)
        self.assertEqual(len(schema["feature_names"]),604)
        self.assertEqual(len(schema["interactions"]),35)
        self.assertEqual(len(set(core.SOCIAL_INTERACTIONS)),35)
        self.assertFalse(any("missing__" in name for name in schema["feature_names"][34:]))
        self.assertEqual(len(data.feature_columns(bundle,core.UNION)),422)
        for spec in specs:
            if spec["basis"] == "additive_cst_full":
                continue
            columns = core.specification_columns(schema,spec)
            self.assertEqual(len(columns),spec["n_features"])
            if spec["basis"] == "single_social192":
                self.assertEqual(columns[:-1],list(range(191)))
                self.assertEqual(schema["feature_names"][columns[-1]],spec["interaction_term"])
            included = {schema["feature_names"][column] for column in columns}
            for column in columns:
                definition = schema["definitions"][column]
                if definition[0] in ("square","product"):
                    for parent in definition[1:]:
                        self.assertIn(parent,included)
        self.assertNotIn("lasso",{spec["estimator"] for spec in specs})

    def test_products_use_cached_ranks_and_raw_agreement_before_scaling(self):
        bundle,task = fixture()
        source = legacy.build_block(bundle,task,"raw")
        pca,schema = legacy.pca_state(bundle,source),core.basis_schema(bundle)
        indices = source["input_indices"]
        raw = core.derived_values(bundle,source,indices,pca,schema)
        names = schema["feature_names"]
        source_names = bundle["manifest"]["feature_names"]
        c = core.CONTROL_NAMES[0]
        original = bundle["X"][indices,source_names.index(c)]
        np.testing.assert_array_equal(raw[:,names.index(c+"__sq")],original**2)
        product = core.SOCIAL_NAMES[0]+"__x__"+c
        np.testing.assert_array_equal(raw[:,names.index(product)],original*bundle["X"][indices,0])
        flags = bundle["X"][indices,source_names.index(core.MISSING_NAMES[0])] == 1
        self.assertTrue(flags.any())
        np.testing.assert_array_equal(raw[flags,names.index(c+"__sq")],0.)
        np.testing.assert_array_equal(raw[flags,names.index(product)],0.)
        agreement = bundle["X"][indices,source_names.index("embed_norm")]
        np.testing.assert_array_equal(raw[:,names.index("embed_norm__x__"+c)],agreement*original)
        pc = raw[:,names.index("embedding_pc_001")]
        weights = data.equal_date_weights(bundle["codes"][indices])
        self.assertAlmostEqual(float(weights@pc),0.,places=7)
        self.assertAlmostEqual(float(weights@(pc**2)),1.,places=6)

    def test_feature_state_all_inputs_not_outcomes_and_no_future_information(self):
        bundle,task = fixture()
        first = core.build_derived_block(bundle,task,"raw")
        self.assertGreater(first["pca"]["input_rows"],len(first["X_fit"]))
        changed = copy.deepcopy(bundle)
        changed["X"][changed["codes"] >= task["valid_first"],53:439] *= 300.
        changed["q"][changed["codes"] >= task["test_first"]] = np.nan
        second = core.build_derived_block(changed,task,"raw")
        for key in ("mean","scale","X_fit"):
            np.testing.assert_array_equal(first[key],second[key])
        np.testing.assert_array_equal(first["pca"]["components"],second["pca"]["components"])
        changed = copy.deepcopy(bundle)
        changed["q"][:100] = np.nan
        third = core.build_derived_block(changed,task,"raw")
        np.testing.assert_array_equal(first["mean"],third["mean"])
        np.testing.assert_array_equal(first["scale"],third["scale"])
        np.testing.assert_array_equal(first["pca"]["components"],third["pca"]["components"])
        binary = [first["feature_names"].index(name) for name in core.MISSING_NAMES]
        original = [bundle["manifest"]["feature_names"].index(name) for name in core.MISSING_NAMES]
        np.testing.assert_array_equal(first["X_fit"][:,binary],bundle["X"][np.ix_(first["fit_indices"],original)])

    def test_union_rebuild_preserves_purge_and_agreement_imputation(self):
        bundle,task = fixture()
        train = core.build_derived_block(bundle,task,"raw")
        union = core.build_derived_block(bundle,task,"raw","union")
        expected_dates = list(range(36))+list(range(37,48))
        self.assertEqual(np.unique(bundle["codes"][union["input_indices"]]).tolist(),expected_dates)
        self.assertEqual(union["pca"]["input_dates"],47)
        self.assertFalse(np.array_equal(train["pca"]["components"],union["pca"]["components"]))
        names = bundle["manifest"]["feature_names"]
        bundle["X"][0,names.index("embed_norm")] = np.nan
        source = legacy.build_block(bundle,task,"raw")
        pca = legacy.pca_state(bundle,source)
        schema = core.basis_schema(bundle)
        raw = core.derived_values(bundle,source,np.array([0]),pca,schema)
        expected = source["mean"][source["feature_names"].index("embed_norm")]
        self.assertAlmostEqual(float(raw[0,schema["feature_names"].index("embed_norm")]),expected,places=7)

    def test_inplace_transform_and_row_compaction_are_bitwise_equivalent(self):
        rng = np.random.default_rng(735)
        raw = rng.normal(size=(159,604)).astype(np.float32)
        mean,scale = rng.normal(size=604),rng.uniform(.1,3.,size=604)
        expected = data._transform(raw,mean,scale)
        working = raw.copy()
        actual = core.transform_inplace(working,mean,scale,chunk_size=13)
        self.assertIs(actual,working)
        np.testing.assert_array_equal(actual,expected)
        for keep in (rng.uniform(size=159)>.3,np.ones(159,dtype=bool),np.zeros(159,dtype=bool),
                     np.arange(159)%29==0):
            storage = actual.copy()
            compacted = core.compact_rows_inplace(storage,keep,chunk_size=7)
            if keep.any():
                self.assertTrue(np.shares_memory(compacted,storage))
            np.testing.assert_array_equal(compacted,expected[keep])

    def test_weighted_candidate_and_union_refit_against_sklearn(self):
        bundle,task = fixture()
        block = core.build_derived_block(bundle,task,"raw","union")
        columns = core.specification_columns(block["schema"],next(spec for spec in core.procedure_specs() if spec["basis"] == "cq_s_squares"))
        X = block["X_fit"][:,columns]
        moments = linear.weighted_moments(X,block["y_fit"],block["w_fit"])
        candidates = core.fit_candidates(moments)
        self.assertEqual(len(candidates),38)
        for estimator in ("ridge","enet"):
            chosen = next(candidate for candidate in candidates if candidate["estimator"] == estimator
                          and (candidate["alpha"] == .1 if estimator == "ridge" else candidate["alpha_fraction"] == .03 and candidate["l1_ratio"] == .5))
            if estimator == "ridge":
                reference = Ridge(alpha=chosen["alpha"])
            else:
                reference = ElasticNet(alpha=chosen["alpha"],l1_ratio=chosen["l1_ratio"],tol=1e-12,max_iter=100000)
            reference.fit(X.astype(float),block["y_fit"],sample_weight=block["w_fit"])
            np.testing.assert_allclose(chosen["coefficient"],reference.coef_,atol=1e-8)
            fitted = legacy.fit_selected(moments,{**chosen,"alpha":999.} if estimator == "enet" else chosen,
                                         np.repeat("derived",len(columns)))
            np.testing.assert_allclose(fitted["coefficient"],reference.coef_,atol=1e-8)
            self.assertLess(fitted["kkt_violation"],linear.KKT_TOL)

    def test_month_all74_additive_identity_and_holdout_label_invariance(self):
        bundle,task = fixture()
        result = core.fit_month(bundle,[task],"raw")
        self.assertEqual(result["predictions"].shape,(96,74))
        original = legacy.build_block(bundle,task,"raw")
        selected,_ = legacy.fit_menu(bundle,original,pca_dimensions=(16,))
        final = legacy.build_block(bundle,task,"raw","union")
        moments = linear.weighted_moments(final["X_fit"],final["y_fit"],final["w_fit"])
        for spec in core.procedure_specs():
            old_spec = core.legacy_spec(spec)
            name = core.model_name(spec,"raw")
            final_meta = result["diagnostics"][name]["final"]
            self.assertEqual(len(final_meta["coefficient"]),len(final_meta["coefficient_features"]))
            self.assertEqual(len(final_meta["mean"]),len(final_meta["coefficient_features"]))
            if old_spec is None:
                continue
            chosen = selected[(spec["feature_set"],spec["estimator"],"full")]
            matrix,groups,_ = legacy.representation_map(bundle,spec["feature_set"])
            fitted = legacy.fit_selected(legacy.mapped_moments(moments,matrix),chosen,groups)
            expected = final["X_test"]@(matrix@fitted["coefficient"])+fitted["intercept"]
            np.testing.assert_array_equal(result["predictions"][:,result["model_names"].index(name)],expected)
        bundle["q"][bundle["codes"] >= task["test_first"]] = np.nan
        bundle["y"][bundle["codes"] >= task["test_first"]] = np.nan
        second = core.fit_month(bundle,[task],"raw")
        np.testing.assert_array_equal(result["predictions"],second["predictions"])


if __name__ == "__main__":
    unittest.main()
