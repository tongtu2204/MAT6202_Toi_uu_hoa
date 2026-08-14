"""Correctness tests for the hand-coded optimizer stack.

The whole talk rests on gradients/Hessians/prox operators coded from formulas, so
these pin them down against finite differences and known behaviour. Fast: tiny
synthetic problems, no artifacts needed. Run either way:

    pytest tests/                 # or:  python tests/test_optim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optim.objective import LogisticObjective                       # noqa: E402
from optim.optimizers import (gradient_descent, accelerated_gd, newton,  # noqa: E402
                              coordinate_descent)
from optim.optimizers.ista import _soft_threshold, fista, _composite       # noqa: E402
from optim.linesearch import armijo_backtrack                       # noqa: E402
from optim.reference import newton_reference, best_known            # noqa: E402
from churn_opt.diagnostics import conditioning                      # noqa: E402


def _toy(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ w)))).astype(float)
    return X, y


def test_gradient_matches_finite_difference():
    X, y = _toy()
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=False)
    w0 = np.random.default_rng(1).standard_normal(obj.d)
    assert check_grad(obj.value, obj.grad, w0) < 1e-5


def test_hessian_matches_finite_difference():
    X, y = _toy()
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=False)
    w = np.random.default_rng(2).standard_normal(obj.d)
    H, eps = obj.hessian(w), 1e-6
    Hfd = np.zeros_like(H)
    for j in range(obj.d):
        e = np.zeros(obj.d); e[j] = eps
        Hfd[:, j] = (obj.grad(w + e) - obj.grad(w - e)) / (2 * eps)
    assert np.max(np.abs(H - Hfd)) < 1e-5


def test_intercept_is_unregularized():
    X, y = _toy()
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=True)
    assert obj._reg[-1] == 0.0 and np.all(obj._reg[:-1] == 1.0)


def test_soft_threshold_leaves_intercept():
    reg = np.array([1.0, 1.0, 0.0])          # last coord = intercept
    out = _soft_threshold(np.array([3.0, -0.5, 5.0]), 1.0, reg)
    assert np.allclose(out, [2.0, 0.0, 5.0]) # shrink, zero, untouched


def test_armijo_gives_descent():
    A = np.diag([1.0, 10.0])
    f = lambda w: 0.5 * float(w @ A @ w)
    w = np.array([1.0, 1.0]); g = A @ w
    t = armijo_backtrack(f, w, f(w), g, -g, t0=1.0)
    assert f(w - t * g) < f(w) and t > 0


def test_gd_converges_within_kappa_bound():
    X, y = _toy(d=6)
    obj = LogisticObjective(X, y, lam=1e-1, fit_intercept=False)
    cond = conditioning(obj.X, 1e-1)
    res = gradient_descent(obj, np.zeros(obj.d), eta=1.0 / cond["L"],
                           max_iter=20000, tol=1e-8)
    assert res.converged and res.grad_norm[-1] < 1e-8
    # linear-rate sanity: iters <= a few * kappa * ln(1/tol)
    budget = 5 * cond["kappa_upper"] * np.log(1e8)
    assert len(res.f_history) < budget


def test_newton_quadratic_convergence():
    X, y = _toy(n=400, d=6)
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=True)
    res = newton(obj, np.zeros(obj.d), max_iter=50, tol=1e-12)
    assert res.converged and len(res.f_history) < 15    # ~quadratic


def test_sklearn_objective_match():
    X, y = _toy(n=500, d=6)
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=True)
    f_star = newton_reference(obj).f_star
    from optim.benchmark import sklearn_baseline
    _, f_sk = sklearn_baseline(obj, X, y)
    assert abs(f_sk - f_star) < 1e-7, (f_sk, f_star)


def test_fstar_is_final_value_and_certified():
    """f* must be the value AT the returned iterate (not min over the history),
    and the PL certificate ||g||^2/(2mu) must be far below the 1e-16 plot floor."""
    X, y = _toy(n=400, d=6)
    obj = LogisticObjective(X, y, lam=1e-2, fit_intercept=True)
    ref = newton_reference(obj)
    assert ref.f_star == float(obj.value(ref.w_star))
    assert ref.bound < 1e-16, ref.bound
    # Armijo makes Newton monotone, so last == min here: nothing is gained by
    # taking a min, which is exactly why the code does not.
    assert abs(ref.f_star - min(ref.result.f_history)) < 1e-15


def test_newton_raises_on_singular_hessian():
    """A rank-deficient design must FAIL loudly, not return converged=False."""
    X, y = _toy(n=100, d=4)
    X = np.hstack([X, X[:, :1]])                 # exact duplicate column
    obj = LogisticObjective(X, y, lam=0.0, fit_intercept=False)
    with pytest.raises(np.linalg.LinAlgError):
        newton(obj, np.zeros(obj.d), max_iter=20, tol=1e-12)
    with pytest.warns(RuntimeWarning):           # opt-out path for the demos
        res = newton(obj, np.zeros(obj.d), max_iter=20, tol=1e-12, strict=False)
    assert not res.converged and "positive definite" in res.note


def test_agd_logs_and_stops_at_the_same_point():
    """Row k of the AGD history must pair f(w_k) with a gradient norm that
    certifies it: f(w_k) - f* <= ||g_k||^2 / (2 mu)."""
    X, y = _toy(n=300, d=6)
    lam = 1e-2
    obj = LogisticObjective(X, y, lam=lam, fit_intercept=False)
    cond = conditioning(obj.X, lam)
    L, mu = cond["L"], max(cond["mu"], lam)
    f_star = newton_reference(obj).f_star
    res = accelerated_gd(obj, np.zeros(obj.d), L=L, mu=mu, max_iter=5000, tol=1e-9)
    assert res.converged
    gap = np.asarray(res.f_history) - f_star
    cert = np.asarray(res.grad_norm) ** 2 / (2 * mu)
    assert np.all(gap <= cert + 1e-15), np.max(gap - cert)
    assert res.f_history[-1] == float(obj.value(res.w))


def test_cd_reaches_fista_optimum():
    X, y = _toy(n=300, d=10)
    obj = LogisticObjective(X, y, lam=0.0, fit_intercept=True)   # pure L1 smooth part
    from optim.benchmark import _smooth_lipschitz
    L, alpha = _smooth_lipschitz(obj), 1e-2
    w0 = np.zeros(obj.d)
    # composite case: no strong convexity => no PL certificate, so f* is the best
    # value any run reached (best_known), and FISTA is non-monotone so its whole
    # history is scanned. See optim/reference.py.
    f_ref = best_known(*fista(obj, w0, alpha=alpha, L=L,
                              max_iter=20000, tol=1e-12).f_history)
    cd = coordinate_descent(obj, w0, alpha=alpha, max_epochs=2000, tol=1e-12)
    assert _composite(obj, cd.w, alpha) <= f_ref + 1e-8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
