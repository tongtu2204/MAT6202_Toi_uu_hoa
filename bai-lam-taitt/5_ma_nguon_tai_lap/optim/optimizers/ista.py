"""ISTA / FISTA — the L1 (Lasso) extension, run on the 'lasso' variant.

The composite objective splits into a smooth part and a nonsmooth part:
    F(w) = g(w) + alpha * ||w||_1 ,   g(w) = obj.value(w)  (logloss [+ optional L2])
`obj` supplies the SMOOTH part only: build it with lam=0 for a pure Lasso, or
lam>0 for elastic net. The L1 term (and its prox) is added here and excludes the
intercept column.

Proximal-gradient step with t = 1/L is soft-thresholding:
    prox_{t*alpha}(v) = sign(v) * max(|v| - t*alpha, 0)      (intercept untouched)
    ISTA:  w_{k+1} = prox( w_k - t ∇g(w_k) )                 -> O(1/k)
    FISTA: same prox on a momentum point y_k                 -> O(1/k^2)

Stopping uses the norm of the gradient mapping G_t(w) = (w - w_plus)/t (the
prox-gradient analogue of ∇f). We also record nnz — how many coordinates the L1
term drove to exactly zero, the sparsity vs the redundant columns churn_opt kept
in this variant on purpose.
"""
from __future__ import annotations

import numpy as np

from .base import OptResult, Recorder
from ..objective import LogisticObjective, sigmoid


def _soft_threshold(v: np.ndarray, thr: float, reg_mask: np.ndarray) -> np.ndarray:
    """Soft-threshold regularized coords by `thr`; leave the intercept as-is."""
    shrunk = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
    return np.where(reg_mask > 0, shrunk, v)


def _composite(obj: LogisticObjective, w: np.ndarray, alpha: float) -> float:
    """F(w) = g(w) + alpha * ||w||_1 over the regularized (non-intercept) coords."""
    l1 = float(np.sum(np.abs(obj._reg * w)))
    return obj.value(w) + alpha * l1


def _nnz(obj: LogisticObjective, w: np.ndarray) -> int:
    """Number of nonzero regularized coordinates (intercept excluded)."""
    return int(np.count_nonzero((obj._reg > 0) & (w != 0.0)))


def ista(
    obj: LogisticObjective,
    w0: np.ndarray,
    alpha: float,
    L: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> OptResult:
    """Proximal gradient descent on F = g + alpha||w||_1. Step t = 1/L."""
    rec = Recorder("ISTA")
    t = 1.0 / L
    w = w0.copy()
    converged = False
    for _ in range(max_iter):
        g = obj.grad(w)
        rec.tick(grad=1)
        w_plus = _soft_threshold(w - t * g, t * alpha, obj._reg)
        gmap = float(np.linalg.norm(w - w_plus) / t)   # gradient-mapping residual
        rec.log(_composite(obj, w, alpha), gmap)
        rec.res.nnz.append(_nnz(obj, w))
        w = w_plus
        if gmap < tol:
            converged = True
            break
    return rec.finish(w, converged)


def fista(
    obj: LogisticObjective,
    w0: np.ndarray,
    alpha: float,
    L: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> OptResult:
    """Accelerated proximal gradient (Beck & Teboulle 2009). Step t = 1/L,
    momentum from the classic t_k = (1 + sqrt(1 + 4 t_{k-1}^2))/2 schedule."""
    rec = Recorder("FISTA")
    t = 1.0 / L
    w = w0.copy()
    w_prev = w0.copy()
    y = w0.copy()
    theta = 1.0
    converged = False
    for _ in range(max_iter):
        g = obj.grad(y)
        rec.tick(grad=1)
        w = _soft_threshold(y - t * g, t * alpha, obj._reg)
        theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        y = w + ((theta - 1.0) / theta_next) * (w - w_prev)
        gmap = float(np.linalg.norm(w - w_prev) / t)
        rec.log(_composite(obj, w, alpha), gmap)
        rec.res.nnz.append(_nnz(obj, w))
        w_prev = w
        theta = theta_next
        if gmap < tol:
            converged = True
            break
    return rec.finish(w, converged)


def coordinate_descent(
    obj: LogisticObjective,
    w0: np.ndarray,
    alpha: float,
    max_epochs: int = 500,
    tol: float = 1e-9,
) -> OptResult:
    """Cyclic proximal coordinate descent on F = g + alpha||w||_1 — the family
    scikit-learn's Lasso uses. Each coordinate takes a 1-D proximal step with its
    OWN Lipschitz constant L_j = (1/4n)||x_j||^2 (logistic curvature bound), then
    soft-thresholds; the intercept is updated but never thresholded. Xw is kept
    incrementally so an epoch costs O(nd), like one full gradient. Logs per epoch
    so the curve overlays ISTA/FISTA.
    """
    rec = Recorder("CD")
    w = w0.copy()
    Xw = obj.X @ w
    col_sq = np.sum(obj.X * obj.X, axis=0)          # ||x_j||^2 per column
    Lj = col_sq / (4.0 * obj.n) + obj.lam * obj._reg  # coordinate Lipschitz
    Lj = np.where(Lj > 1e-12, Lj, 1e-12)
    converged = False
    # p = sigmoid(Xw) only changes when Xw does, i.e. when a coordinate ACTUALLY
    # moves. Recomputing it for every j costs an O(n) exp() per coordinate — O(nd)
    # exponentials per epoch, which measured 474 ms/epoch at n=75k, d=520 and made
    # CD look 15x more expensive per step than ISTA. In the sparse regime this is
    # nearly all waste: most coordinates sit at exactly zero and do not move. The
    # dirty flag keeps the iteration mathematically identical and only skips
    # recomputation that cannot have changed anything.
    p = sigmoid(Xw)
    for _ in range(max_epochs):
        max_delta = 0.0
        for j in range(obj.d):
            g_j = float(obj.X[:, j] @ (p - obj.y)) / obj.n + obj.lam * obj._reg[j] * w[j]
            wj_new = w[j] - g_j / Lj[j]
            if obj._reg[j] > 0:                      # regularized coord: prox = soft-threshold
                thr = alpha / Lj[j]
                wj_new = np.sign(wj_new) * max(abs(wj_new) - thr, 0.0)
            delta = wj_new - w[j]
            if delta != 0.0:
                Xw += obj.X[:, j] * delta
                p = sigmoid(Xw)                 # only now can p have changed
                w[j] = wj_new
                max_delta = max(max_delta, abs(delta))
        rec.tick(grad=1)                            # one O(nd) sweep ~ one gradient
        rec.log(_composite(obj, w, alpha), max_delta)
        rec.res.nnz.append(_nnz(obj, w))
        if max_delta < tol:
            converged = True
            break
    return rec.finish(w, converged)


def subgradient(
    obj: LogisticObjective,
    w0: np.ndarray,
    alpha: float,
    eta0: float,
    max_iter: int = 5000,
    tol: float = 0.0,
) -> OptResult:
    """Subgradient descent on F(w) = g(w) + alpha*||w||_1 — the naive alternative
    to the proximal step, kept as the control that makes ISTA/FISTA's case.

        w_{k+1} = w_k - eta_k * (grad g(w_k) + alpha * sign(w_k)),
        eta_k   = eta0 / sqrt(k+1)

    Two things this is here to demonstrate, and both are properties of the METHOD,
    not of the tuning:

    1. RATE. The 1/sqrt(k) step schedule is what non-smooth convergence theory
       requires (sum eta_k = inf, sum eta_k^2 < inf) and it buys only O(1/sqrt k) —
       against O(1/k) for ISTA and O(1/k^2) for FISTA. There is no step size that
       fixes this; the subgradient of |.| does not shrink as w -> 0.
    2. SPARSITY. `sign(w)` is +-1 for every nonzero coordinate, so a coordinate is
       pushed toward zero but overshoots and changes sign instead of landing on it.
       Exact zeros essentially never occur, which is why `nnz` stays at full width
       while the prox methods drive it down. Soft-thresholding is not a faster way
       to do the same thing — it is the part that produces sparsity at all.

    F is not differentiable at 0, so the iterates are not monotone; the history is
    the raw F(w_k), and `best_known` (see optim/reference.py) is what the composite
    f* rule is for.
    """
    rec = Recorder("Subgradient")
    w = w0.copy()
    for k in range(max_iter):
        g = obj.grad(w) + alpha * np.sign(w) * obj._reg   # intercept unpenalized
        rec.tick(grad=1)
        gnorm = float(np.linalg.norm(g))
        rec.log(_composite(obj, w, alpha), gnorm)
        rec.res.nnz.append(_nnz(obj, w))
        if gnorm < tol:
            return rec.finish(w, True)
        w = w - (eta0 / np.sqrt(k + 1.0)) * g
    return rec.finish(w, False)
