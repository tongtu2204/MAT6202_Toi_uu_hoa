"""Small deterministic checks for the isolated GD/AGD update."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tu_update"))

from evaluation import sigmoid, split_indices, train_valid_indices  # noqa: E402
from gd_agd_runner import run_agd, run_gd  # noqa: E402


class Quadratic:
    def value(self, w):
        return 0.5 * float(w @ w)

    def grad(self, w):
        return np.asarray(w, dtype=float)


class UpdateTests(unittest.TestCase):
    def test_sigmoid_is_stable(self):
        actual = sigmoid(np.array([-1_000.0, 0.0, 1_000.0]))
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, [0.0, 0.5, 1.0], atol=1e-15)

    def test_train_valid_split_has_no_overlap(self):
        y = np.tile([0, 1], 100)
        train, valid, mode = train_valid_indices(y, 0.2, 42)
        self.assertEqual(mode, "stratified-row")
        self.assertEqual(len(set(train) & set(valid)), 0)
        self.assertEqual(len(train) + len(valid), len(y))

    def test_three_way_split_has_no_overlap(self):
        y = np.tile([0, 1], 100)
        train, valid, test, _ = split_indices(y, 0.2, 0.2, 42)
        self.assertFalse(set(train) & set(valid))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(valid) & set(test))
        self.assertEqual(len(train) + len(valid) + len(test), len(y))

    def test_gd_and_agd_reduce_a_quadratic(self):
        obj = Quadratic()
        w0 = np.array([2.0, -1.0])
        gd = run_gd(obj, w0, 0.25, (1, 20), 0.0)
        agd = run_agd(obj, w0, 0.25, 0.5, (1, 20), 0.0)
        self.assertLess(gd.snapshots[20].objective, gd.snapshots[1].objective)
        self.assertLess(agd.snapshots[20].objective, agd.snapshots[1].objective)


if __name__ == "__main__":
    unittest.main()

