import contextlib
import io
import unittest

import numpy as np

from qec_trajectory_split import make_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(15), 20)
        self.cycles = np.tile(np.arange(20), 15)
        self.rates = np.where(self.cycles % 2, 0.12, 0.04)
        self.p = np.column_stack([1 - self.rates, self.rates,
                                  np.zeros(300), np.zeros(300)])

    def split(self, p=None, groups=None, cycles=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return make_splits(self.p if p is None else p, self.rates,
                               self.groups if groups is None else groups,
                               self.cycles if cycles is None else cycles)

    def test_disjoint_groups_rows_and_next_cycle_labels(self):
        parts = self.split()
        seen_groups, seen_rows = set(), set()
        for name, expected in [("train", 9), ("val", 3), ("test", 3)]:
            part = parts[name]
            groups = set(part["trajectory_ids"].tolist())
            rows = set(part["source_rows"].ravel().tolist())
            self.assertEqual(len(groups), expected)
            self.assertFalse(groups & seen_groups)
            self.assertFalse(rows & seen_rows)
            seen_groups |= groups
            seen_rows |= rows
            self.assertEqual(part["X"].shape, (expected * 12, 8, 4, 1))
            source = part["source_rows"]
            self.assertTrue((self.groups[source] == part["trajectory_ids"][:, None]).all())
            self.assertTrue((np.diff(self.cycles[source], axis=1) == 1).all())
            np.testing.assert_array_equal(part["y"], self.rates[source[:, -1]] > .08)
            np.testing.assert_allclose(part["X"][..., 0], self.p[source[:, :-1]])
        self.assertEqual(seen_groups, set(range(15)))

    def test_missing_or_duplicate_cycles_are_rejected(self):
        for bad in (30, 18):
            cycles = self.cycles.copy()
            cycles[19] = bad
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                self.split(cycles=cycles)

    def test_invalid_probability_vectors_are_rejected(self):
        p = self.p.copy()
        p[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            self.split(p=p)

    def test_fixed_seed_is_repeatable(self):
        a, b = self.split(), self.split()
        for name in a:
            np.testing.assert_array_equal(a[name]["source_rows"], b[name]["source_rows"])


if __name__ == "__main__":
    unittest.main()
