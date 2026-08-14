"""Damped Newton (= IRLS).  w_{k+1} = w_k - t (∇²f)^{-1} ∇f.

Solve H d = -g with np.linalg.solve (never invert); t from armijo_backtrack
(pure Newton t=1 near the optimum -> quadratic convergence). This is why the
'ridge' variant must be full rank: singular H makes the solve blow up.
Cost per step is O(n d^2) to form H + O(d^3) to solve — the expensive axis of
the trade-off.  Run this deep (tol ~ 1e-14) to produce f* for the log-plots.
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import OptResult, Recorder
from ..linesearch import armijo_backtrack
from ..objective import LogisticObjective


def newton(
    obj: LogisticObjective,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
    line_search: bool = True,
    strict: bool = True,
    t0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> OptResult:
    """Damped Newton via Cholesky solve of the (SPD) regularized Hessian.
    `t0` is the step: with line_search it seeds the Armijo backtracking, without it
    the step is simply t0 every iteration. t0=1 and line_search=False is pure
    Newton; t0!=1 and line_search=False is "damped Newton with a fixed step", which
    the reference deck sweeps as its own variant. Stops on ||∇f|| < tol.

    A non-positive-definite Hessian is normally a BUG, not a result, so
    `strict=True` (default) raises instead of silently returning
    converged=False — a silent break here would look like "Newton just didn't
    converge" and hide a rank-deficient design matrix. strict=False downgrades it
    to a RuntimeWarning + `note` on the result, for the runs where the breakdown
    IS the finding (pure Newton from a far start: see newton_init_study)."""
    rec = Recorder("Newton")
    w = w0.copy()
    converged = False
    note = ""
    for k in range(max_iter):
        g = obj.grad(w)
        H = obj.hessian(w)
        rec.tick(grad=1, hess=1)
        f = obj.value(w)
        gnorm = float(np.linalg.norm(g))
        rec.log(f, gnorm)
        if gnorm < tol:
            converged = True
            break
        try:
            Lc = np.linalg.cholesky(H)          # SPD factor (fails if not PD)
            u = np.linalg.solve(Lc, g)          # forward solve
            d = -np.linalg.solve(Lc.T, u)       # back solve -> d = -H^{-1} g
        except np.linalg.LinAlgError as exc:
            note = (f"Hessian not positive definite at iteration {k} "
                    f"(||g||={gnorm:.3e}, f={f:.3e}, ||w||={np.linalg.norm(w):.3e}, "
                    f"lambda={obj.lam:g}, d={obj.d}): {exc}. Either the design "
                    "matrix is rank-deficient, or the iterate ran away far enough "
                    "that S=diag(p(1-p)) underflowed and the unpenalized intercept "
                    "direction lost its only curvature.")
            if strict:
                raise np.linalg.LinAlgError(note) from exc
            warnings.warn(note, RuntimeWarning, stacklevel=2)
            break
        if line_search:
            t = armijo_backtrack(obj.value, w, f, g, d, t0=t0, beta=rho, c=c)
        else:
            t = t0
        rec.res.steps.append(float(t))
        w = w + t * d
    return rec.finish(w, converged, note)
