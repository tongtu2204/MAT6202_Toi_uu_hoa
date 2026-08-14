"""Minibatch stochastic gradient descent.

Stochastic gradient uses a batch B: g ≈ (1/|B|) X_Bᵀ(p_B - y_B) + lambda w.
Diminishing step eta_k = eta0/(1 + gamma k) (or constant, to show the noise
floor). Log per EPOCH so its curve is comparable to the full-batch methods —
the point is fast early progress, noisy tail.
"""
from __future__ import annotations

import numpy as np

from .base import OptResult, Recorder
from ..objective import LogisticObjective
from ..objective import sigmoid


def _batch_grad(obj: LogisticObjective, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Minibatch gradient of the same regularized objective (mean over |B|)."""
    Xb = obj.X[idx]
    p = sigmoid(Xb @ w)
    return Xb.T @ (p - obj.y[idx]) / len(idx) + obj.lam * obj._reg * w


def sgd(
    obj: LogisticObjective,
    w0: np.ndarray,
    eta0: float,
    batch_size: int = 256,
    epochs: int = 50,
    gamma: float = 2e-4,
    seed: int = 0,
    tol: float = 1e-4,
) -> OptResult:
    """Shuffle-each-epoch minibatch SGD with 1/(1+gamma*k) step decay (k counts
    minibatch steps). Logs the FULL objective + true gradient norm once per
    epoch, so its curve overlays the batch methods; per-batch grad cost is
    counted as |B|/n of a full gradient.

    gamma default 2e-4 (not 1e-2): the decay counts MINIBATCH steps (~n/batch per
    epoch), so 1e-2 collapses the step within one epoch. `converged` is reported
    honestly from the final full-gradient norm vs `tol` (constant-step SGD floors
    above any tol; a diminishing schedule can dip below). The run always uses the
    full epoch budget so per-epoch curves stay length-comparable across configs."""
    rec = Recorder("SGD")
    rng = np.random.default_rng(seed)
    n = obj.n
    w = w0.copy()
    step = 0
    # log the starting point (epoch 0)
    g0 = obj.grad(w)
    rec.tick(grad=1)
    gnorm = float(np.linalg.norm(g0))
    rec.log(obj.value(w), gnorm)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            g = _batch_grad(obj, w, idx)
            eta = eta0 / (1.0 + gamma * step)
            w = w - eta * g
            step += 1
            rec.tick(grad=len(idx) / n)     # fractional full-gradient equivalent
        gnorm = float(np.linalg.norm(obj.grad(w)))
        rec.log(obj.value(w), gnorm)
        if not np.isfinite(obj.value(w)):
            break
    return rec.finish(w, converged=gnorm < tol)
