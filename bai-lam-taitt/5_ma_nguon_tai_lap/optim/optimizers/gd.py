"""Gradient descent.  w_{k+1} = w_k - eta ∇f(w_k).

Fixed eta = 1/L gives the baseline linear rate ~ (1 - mu/L); optionally use
armijo_backtrack from linesearch for an adaptive step. The non-finite guard below
just stops cleanly if an over-large step blows the objective up. Step-size
stability (the honest eta>2/L story) is studied in optim/experiments.py.
"""
from __future__ import annotations

import numpy as np

from .base import OptResult, Recorder
from ..linesearch import armijo_backtrack
from ..objective import LogisticObjective


def gradient_descent(
    obj: LogisticObjective,
    w0: np.ndarray,
    eta: float,
    max_iter: int = 2000,
    tol: float = 1e-8,
    line_search: bool = False,
    rho: float = 0.5,
    c: float = 1e-4,
) -> OptResult:
    """Full-batch GD. `eta` is the fixed step (with line_search=True it seeds the
    backtracking t0). `rho` shrinks the trial step and `c` is the Armijo constant —
    both exposed so they can be swept, since which (rho, c) is best is an empirical
    question, not a theoretical one. Logs f and ||g|| every iteration; a non-finite
    objective (the eta>2/L demo) stops the run and is recorded."""
    rec = Recorder("GD")
    w = w0.copy()
    converged = False
    for _ in range(max_iter):
        g = obj.grad(w)
        rec.tick(grad=1)
        f = obj.value(w)
        gnorm = float(np.linalg.norm(g))
        rec.log(f, gnorm)
        if not np.isfinite(f):            # divergent config: bail once it blows up
            break
        if gnorm < tol:
            converged = True
            break
        if line_search:
            t = armijo_backtrack(obj.value, w, f, g, -g, t0=eta, beta=rho, c=c)
        else:
            t = eta
        rec.res.steps.append(float(t))   # what backtracking actually chose
        w = w - t * g
    return rec.finish(w, converged)
