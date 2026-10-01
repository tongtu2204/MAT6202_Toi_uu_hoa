"""Kiểm tra nhỏ, xác định cho mã GD/AGD trong ``tu_update``."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tu_update"))

from evaluation import sigmoid  # noqa: E402
from gd_agd_runner import (  # noqa: E402
    run_agd,
    run_gd_backtracking,
    run_gd_fixed,
)
from run_experiments import fine_linear_grid, fine_log_grid  # noqa: E402


class Quadratic:
    def value(self, w):
        return 0.5 * float(w @ w)

    def grad(self, w):
        return np.asarray(w, dtype=float)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.obj = Quadratic()
        self.w0 = np.array([2.0, -1.0])

    def test_sigmoid_is_numerically_stable(self):
        actual = sigmoid(np.array([-1_000.0, 0.0, 1_000.0]))
        np.testing.assert_allclose(actual, [0.0, 0.5, 1.0], atol=1e-15)

    def test_gd_fixed_reduces_objective(self):
        run = run_gd_fixed(self.obj, self.w0, 0.25, 20, 1e-12, False)
        self.assertLess(run.objectives[-1], run.objectives[0])
        self.assertEqual(run.iterations[-1], 20)

    def test_backtracking_satisfies_armijo_and_counts_trials(self):
        run = run_gd_backtracking(
            self.obj, self.w0, t0=4.0, rho=0.5, c=1e-4,
            max_iter=10, tol=1e-12, stop_on_convergence=False,
        )
        self.assertTrue(run.finite)
        self.assertGreater(run.backtracking_trials, 10)
        self.assertLess(run.objectives[-1], run.objectives[0])

    def test_two_agd_schemes_reduce_objective(self):
        constant = run_agd(
            self.obj, self.w0, 0.25, 20, 1e-12, False,
            "constant", 0.5,
        )
        dynamic = run_agd(
            self.obj, self.w0, 0.25, 20, 1e-12, False,
            "dynamic", None,
        )
        self.assertLess(constant.objectives[-1], constant.objectives[0])
        self.assertLess(dynamic.objectives[-1], dynamic.objectives[0])

    def test_stop_on_convergence_records_iteration(self):
        run = run_gd_fixed(self.obj, self.w0, 0.5, 200, 1e-8, True)
        self.assertTrue(run.converged)
        self.assertLess(run.converged_at, 200)
        self.assertEqual(run.iterations[-1], run.converged_at)

    def test_fine_grids_keep_coarse_winner_at_center(self):
        linear = fine_linear_grid((1.0, 2.0, 5.0), 2.0, 9)
        logarithmic = fine_log_grid((1e-4, 0.1, 0.5), 0.1, 3)
        self.assertEqual(linear[4], 2.0)
        self.assertEqual(logarithmic[1], 0.1)


if __name__ == "__main__":
    unittest.main()
