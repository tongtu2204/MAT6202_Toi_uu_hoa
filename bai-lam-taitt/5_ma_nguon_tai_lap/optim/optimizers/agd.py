"""Nesterov accelerated gradient descent.

    y_{k}   = w_k + beta (w_k - w_{k-1})
    w_{k+1} = y_k - (1/L) ∇f(y_k)
Two momentum schemes, selected by `scheme`:

  "const" (default)  beta = (sqrt(kappa)-1)/(sqrt(kappa)+1), kappa = L/mu.
                     The STRONGLY CONVEX form. Rate ~ (1 - 1/sqrt(kappa)).
                     Derived assuming step 1/L: (t, beta) is a matched PAIR.
  "k"                beta_k = (k-2)/(k+1) for k = 2, 3, ... — the form for merely
                     convex f, rate O(1/k^2). It carries NO problem constant, so
                     the step t is a genuinely free parameter here, which is why
                     it can be swept by hand while the "const" pair cannot.

`t` overrides the step; leaving it None keeps the classical 1/L.
"""
from __future__ import annotations

import numpy as np

from .base import OptResult, Recorder
from ..objective import LogisticObjective


def accelerated_gd(
    obj: LogisticObjective,
    w0: np.ndarray,
    L: float,
    mu: float,
    max_iter: int = 2000,
    tol: float = 1e-8,
    scheme: str = "const",
    t: float | None = None,
) -> OptResult:
    """Nesterov AGD with constant momentum from kappa = L/mu. Step 1/L. Costs
    exactly one gradient per iteration, evaluated at the momentum point y.

    Logging is deliberately CONSISTENT: row k holds f(w_k) together with
    ||∇f(y_{k-1})||, and those two refer to the same iterate because

        f(w_k) - f*  <=  f(y_{k-1}) - f*  <=  ||∇f(y_{k-1})||^2 / (2 mu)

    (descent lemma with step 1/L, then Polyak-Lojasiewicz). So the stopping test
    ||g|| < tol certifies exactly the quantity the plot draws, at the same row —
    logging f(w_k) while testing ||∇f(y_k)|| (the next momentum point, one step
    ahead) would mix two different iterates. Evaluating ||∇f(w_k)|| directly
    instead would cost a second gradient per iteration and make the wall-clock
    comparison against GD dishonest."""
    if scheme not in ("const", "k"):
        raise ValueError(f"scheme phải là 'const' hoặc 'k', nhận {scheme!r}")
    kappa = L / mu
    beta_const = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    rec = Recorder("AGD" if scheme == "const" else "AGD (k-2)/(k+1)")
    step = 1.0 / L if t is None else float(t)
    w = w0.copy()
    w_prev = w0.copy()
    converged = False

    # row 0: the starting point, same quantity every method logs at index 0
    g = obj.grad(w)
    rec.tick(grad=1)
    gnorm = float(np.linalg.norm(g))
    rec.log(obj.value(w), gnorm)
    if gnorm < tol:
        return rec.finish(w, True)

    for i in range(max_iter):
        # k = 2, 3, ... exactly as the (k-2)/(k+1) scheme is stated: the first
        # momentum coefficient is 0, so step 1 is plain gradient descent.
        beta = beta_const if scheme == "const" else (i) / (i + 3.0)
        y = w + beta * (w - w_prev)
        g = obj.grad(y)
        rec.tick(grad=1)
        gnorm = float(np.linalg.norm(g))
        w_prev = w
        w = y - step * g
        f = obj.value(w)
        rec.log(f, gnorm)                 # (f(w_k), ||∇f(y_{k-1})||) — see above
        if not np.isfinite(f):
            break
        if gnorm < tol:
            converged = True
            break
    return rec.finish(w, converged)
