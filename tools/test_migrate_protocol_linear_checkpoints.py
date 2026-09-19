"""Dependency boundaries and strict rejection of unapproved data/code changes."""
from contextlib import contextmanager
import copy
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

import migrate_protocol_linear_checkpoints as migration
import protocol_data as data


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parents[1]/".runs"
    folder = root/f"linear_migration_test_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        assert folder.resolve().parent == root.resolve() and folder.name.startswith("linear_migration_test_")
        shutil.rmtree(folder)


def bundles():
    rng = np.random.default_rng(387)
    codes = np.repeat([10, 11, 12], 12)
    dates = np.repeat(pd.to_datetime(["2019-03-12", "2019-03-13", "2019-03-14"]), 12)
    index = np.arange(100, 136)
    index[15] = migration.BAD_MM_INDEX
    keys = pd.DataFrame({"date": dates, "permno": np.tile(np.arange(12)+500, 3)}, index=index)
    keys.loc[migration.BAD_MM_INDEX, "permno"] = migration.BAD_PERMNO
    y = rng.normal(size=(36, 2))
    old = {"keys": keys, "codes": codes, "y": y,
           "q": np.column_stack([data.centered_rank(y[:, j], codes, target=True) for j in range(2)])}
    new = copy.deepcopy(old)
    new["y"][15] = np.nan
    for column in range(2):
        new["q"][codes == 11, column] = data.centered_rank(new["y"][codes == 11, column], codes[codes == 11], target=True)
    declaration = {name: migration.changed_cells(old[name], new[name]) for name in ("y", "q")}
    return old, new, declaration


class MigrationTests(unittest.TestCase):
    def test_fit_and_validation_boundaries_are_inclusive_but_test_and_gaps_do_not_depend(self):
        task = {"fit_first": 10, "fit_last": 20, "valid_first": 22, "valid_last": 30,
                "test_first": 32, "test_last": 40}
        for code in [10, 15, 20, 22, 25, 30]:
            self.assertTrue(migration.depends_on_changed_labels(task, [code]))
        for code in [9, 21, 31, 32, 40, 41]:
            self.assertFalse(migration.depends_on_changed_labels(task, [code]))
        self.assertTrue(migration.depends_on_changed_labels(task, [9, 22, 35]))

    def test_exact_null_and_same_date_rerank_are_accepted(self):
        old, new, declared = bundles()
        result = migration.verify_array_delta(old, new, declared)
        self.assertEqual(result["changed_date_codes"], {"raw": [11], "dgtw": [11]})
        self.assertEqual(result["changed_cells"]["y"], [[15, 0], [15, 1]])
        self.assertGreater(len(result["changed_cells"]["q"]), 2)

    def test_undeclared_or_semantically_wrong_array_changes_are_rejected(self):
        old, new, declared = bundles()
        new["q"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            migration.verify_array_delta(old, new, declared)
        declared["q"] = migration.changed_cells(old["q"], new["q"])
        with self.assertRaisesRegex(ValueError, "extend beyond"):
            migration.verify_array_delta(old, new, declared)
        old, new, declared = bundles()
        new["q"][16, 0] = -.49
        declared["q"] = migration.changed_cells(old["q"], new["q"])
        with self.assertRaisesRegex(ValueError, "freshly computed"):
            migration.verify_array_delta(old, new, declared)
        old, new, declared = bundles()
        new["y"][0, 0] = np.nan
        declared["y"] = migration.changed_cells(old["y"], new["y"])
        with self.assertRaisesRegex(ValueError, "beyond nulling"):
            migration.verify_array_delta(old, new, declared)

    def test_ast_proof_ignores_new_preparation_helpers_but_detects_runtime_edits(self):
        source = Path(data.__file__).read_text(encoding="utf-8")
        original = migration.runtime_ast(source)
        self.assertEqual(original, migration.runtime_ast(source+"\n\ndef new_preparation_helper():\n    return 1\n"))
        modified = source.replace("def make_block(bundle, task, feature_set, target):",
                                  "def make_block(bundle, task, feature_set, target):\n    raise RuntimeError('changed')")
        self.assertNotEqual(original, migration.runtime_ast(modified))

    def test_unchanged_file_hash_proof_rejects_modified_predictors(self):
        with fixture_directory() as folder:
            old, new = folder/"old", folder/"new"
            old.mkdir()
            new.mkdir()
            for name in migration.INPUT_FILES:
                (old/name).write_bytes(b"unchanged fixture "+name.encode())
                (new/name).write_bytes((old/name).read_bytes())
            bundle = {"manifest": {"feature_names": ["a"], "rows": 1}}
            proof = migration.verify_inputs(old, new, bundle, bundle)
            self.assertEqual(set(proof), set(migration.INPUT_FILES))
            (new/"X.npy").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "X.npy"):
                migration.verify_inputs(old, new, bundle, bundle)


if __name__ == "__main__":
    unittest.main()
