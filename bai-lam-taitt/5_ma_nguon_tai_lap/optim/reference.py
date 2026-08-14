"""The reference optimum f* — and the certificate that says it is trustworthy.

Every convergence plot in this project is log(f_k - f*), so f* is load-bearing:
pick it badly and every curve is wrong. Two rules, one per problem class.

1. SMOOTH, STRONGLY CONVEX (all the ridge/GD/AGD/Newton/SGD experiments)
   f* = the objective AT THE FINAL ITERATE of a deep damped-Newton run.
   NOT `min(f_history)`: the smallest value a run ever touched is a different
   point from the one the run returned, and reporting it would shave the gaps
   f_k - f* of every OTHER method in our favour. Armijo makes Newton monotone,
   so here min and last coincide anyway — which is exactly why there is no
   reason to write `min`.

   What actually makes f* trustworthy is strong convexity. For mu-strongly
   convex f,

       f(w) - f(w*) <= ||grad f(w)||^2 / (2 mu)                          (PL)

   so the gradient norm at the point we stopped at BOUNDS how far our reference
   sits above the true optimum. `Reference.bound` carries that number so a slide
   can quote it instead of asking for trust (typically ~1e-24, eight orders below
   the 1e-16 floor of the log plots).

   CAREFUL about mu. The usual claim "mu >= lambda because of the ridge term" is
   NOT true here: the intercept is deliberately unpenalized, so the ridge term is
   lambda*diag(reg) with a ZERO on the intercept, and along that one direction the
   only curvature is (1/n) sum_i p_i(1-p_i) — which decays to 0 as |w| grows. f is
   therefore strongly convex on bounded sublevel sets, not globally. (This is not
   academic: pure Newton from a far start walks out to ||w|| ~ 1e16, S underflows,
   and the Hessian becomes exactly singular in the intercept direction — that is
   what the pure-vs-damped Newton slide is showing.)

   So `mu` defaults to the MEASURED lambda_min(grad^2 f(w*)) at the reference
   point, not to lambda. On `ridge` that is ~2.8e-3, i.e. ~2.8x larger than
   lambda, so the certificate is both correct and tighter.

2. COMPOSITE / NON-SMOOTH (the L1 lasso extension)
   With lambda = 0 the composite is convex but NOT strongly convex, so there is
   no PL certificate. There f* = the smallest composite value found by ANY run
   (a 20k-iteration FISTA plus the saga baseline). That direction is safe:
   f* can only be an OVER-estimate of the true minimum, and a lower f* makes the
   reported gaps LARGER, never smaller. `best_known` implements it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .objective import LogisticObjective
from .optimizers import newton
from .optimizers.base import OptResult


@dataclass
class Reference:
    """Deep-Newton reference optimum plus its strong-convexity certificate."""
    f_star: float
    w_star: np.ndarray
    grad_norm: float      # ||grad f(w*)|| at the point we stopped
    mu: float             # strong-convexity constant used in the bound
    bound: float          # ||g||^2 / (2 mu): certified over-estimate of f_star
    converged: bool
    result: OptResult


def newton_reference(
    obj: LogisticObjective,
    w0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-14,
    mu: float | None = None,
    warn_bound: float = 1e-16,
    t0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> Reference:
    """Run damped Newton to machine precision; return f* with its PL certificate.

    `mu` defaults to the measured lambda_min of the Hessian AT the optimum — see
    the module docstring for why obj.lam is the wrong constant (unpenalized
    intercept). Warns if the certificate is weaker than the 1e-16 plot floor,
    i.e. if f* is not tight enough to trust the curves drawn against it.
    """
    if w0 is None:
        w0 = np.zeros(obj.d)
    # (t0, rho, c) default to the library's damped-Newton settings; run_benchmark
    # passes the HAND-TUNED ones so the Newton curve on the money plot is the same
    # configuration the deck's tuning section chose, not an unrelated default.
    res = newton(obj, w0, max_iter=max_iter, tol=tol, t0=t0, rho=rho, c=c)
    w_star = res.w
    f_star = float(obj.value(w_star))          # value at the RETURNED iterate
    gnorm = float(np.linalg.norm(obj.grad(w_star)))
    if mu is None:
        mu = float(np.linalg.eigvalsh(obj.hessian(w_star))[0])
        if not np.isfinite(mu) or mu <= 0:     # should not happen at a minimizer
            warnings.warn(f"lambda_min(H(w*)) = {mu:.3e} <= 0; falling back to "
                          f"lambda = {obj.lam:g} for the certificate.",
                          RuntimeWarning, stacklevel=2)
            mu = float(obj.lam)
    mu = float(mu)
    bound = gnorm * gnorm / (2.0 * mu) if mu > 0 else float("inf")
    if bound > warn_bound:
        warnings.warn(
            f"f* certificate weak: ||g||={gnorm:.3e}, mu={mu:.3e} => "
            f"f* - f_true <= {bound:.3e} (> plot floor {warn_bound:.0e}). "
            "Convergence curves below that gap are not meaningful.",
            RuntimeWarning, stacklevel=2)
    return Reference(f_star, w_star, gnorm, mu, bound, bool(res.converged), res)


def best_known(*values: float) -> float:
    """f* for the non-strongly-convex (composite) case: the smallest objective
    value ANY run achieved. Conservative by construction — see module docstring."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        raise ValueError("best_known: no finite objective value given")
    return min(finite)
