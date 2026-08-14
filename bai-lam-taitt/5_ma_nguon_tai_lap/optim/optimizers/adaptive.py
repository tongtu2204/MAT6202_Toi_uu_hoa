"""Adaptive-step first-order methods: AdaGrad, RMSprop, Adam (+AdamW, AMSGrad).

All of them are GD with a CHEAP DIAGONAL PRECONDITIONER learned online from the
gradient history:

    w_{k+1} = w_k - eta * P_k^{-1} g_k,   P_k diagonal

    AdaGrad   P_k = diag(sqrt(G_k) + eps),  G_k = G_{k-1} + g_k^2      (only grows)
    RMSprop   same, but G_k = rho G_{k-1} + (1-rho) g_k^2              (forgets)
    Adam      RMSprop + momentum, with bias correction on both moments

Placing them on the map of this project: GD (P=I) -> Ada/Adam (P diagonal) ->
Newton (P = full Hessian). The diagonal costs O(d) per step instead of Newton's
O(nd^2 + d^3), but it cannot fix CORRELATION between columns — which is exactly
what limits it here (see optim.experiments.adaptive_family).

Full-batch by default so the curves are directly comparable to GD/AGD/Newton on
the same iteration axis; pass batch_size to get the stochastic version, which
then logs once per epoch like `sgd` does.
"""
from __future__ import annotations

import numpy as np

from .base import OptResult, Recorder
from ..objective import LogisticObjective, sigmoid


def _batch_grad(obj: LogisticObjective, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    Xb = obj.X[idx]
    p = sigmoid(Xb @ w)
    return Xb.T @ (p - obj.y[idx]) / len(idx) + obj.lam * obj._reg * w


def _step_adagrad(state, g, eta, eps, **_):
    state["G"] += g * g
    return eta * g / (np.sqrt(state["G"]) + eps)


def _step_rmsprop(state, g, eta, eps, rho=0.9, **_):
    state["G"] = rho * state["G"] + (1.0 - rho) * g * g
    return eta * g / (np.sqrt(state["G"]) + eps)


def _step_adam(state, g, eta, eps, beta1=0.9, beta2=0.999,
               amsgrad=False, **_):
    state["k"] += 1
    k = state["k"]
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * g
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * g * g
    m_hat = state["m"] / (1.0 - beta1 ** k)
    v_hat = state["v"] / (1.0 - beta2 ** k)
    if amsgrad:                       # non-decreasing denominator (Reddi et al.)
        state["v_max"] = np.maximum(state["v_max"], v_hat)
        v_hat = state["v_max"]
    return eta * m_hat / (np.sqrt(v_hat) + eps)


_RULES = {
    "AdaGrad": _step_adagrad,
    "RMSprop": _step_rmsprop,
    "Adam": _step_adam,
    "AdamW": _step_adam,             # decoupled decay handled below
    "AMSGrad": _step_adam,
}


def adaptive(
    obj: LogisticObjective,
    w0: np.ndarray,
    method: str = "Adam",
    eta: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-8,
    eps: float = 1e-8,
    rho: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    batch_size: int | None = None,
    epochs: int | None = None,
    seed: int = 0,
) -> OptResult:
    """One driver for the whole family; `method` picks the update rule.

    Full batch (batch_size=None): logs f and ||g|| every iteration, stops at
    ||g|| < tol — same contract as `gradient_descent`, so results drop straight
    into the same money plot.

    Mini-batch (batch_size=b): runs `epochs` passes and logs the FULL objective
    once per epoch, same contract as `sgd`.

    AdamW differs from Adam by applying the ridge term as decoupled weight decay
    (w -= eta*lam*w) instead of letting it flow through the 1/sqrt(v) scaling —
    on this objective the two are genuinely different, not a reparametrization.
    """
    if method not in _RULES:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_RULES)}")
    rule = _RULES[method]
    decoupled = method == "AdamW"
    kwargs = dict(rho=rho, beta1=beta1, beta2=beta2, amsgrad=(method == "AMSGrad"))

    rec = Recorder(method)
    w = w0.copy()
    d = obj.d
    state = {"G": np.zeros(d), "m": np.zeros(d), "v": np.zeros(d),
             "v_max": np.zeros(d), "k": 0}

    # AdamW optimizes the SAME f, but applies the decay outside the
    # preconditioner, so the gradient fed to the rule excludes lam*w.
    if batch_size is None:
        converged = False
        for _ in range(max_iter):
            g_full = obj.grad(w)
            rec.tick(grad=1)
            f = obj.value(w)
            gnorm = float(np.linalg.norm(g_full))
            rec.log(f, gnorm)
            if not np.isfinite(f):
                break
            if gnorm < tol:
                converged = True
                break
            g = g_full - obj.lam * obj._reg * w if decoupled else g_full
            w = w - rule(state, g, eta, eps, **kwargs)
            if decoupled:
                w = w - eta * obj.lam * obj._reg * w
        return rec.finish(w, converged)

    # ---- stochastic variant: log once per epoch, like optim.optimizers.sgd ----
    rng = np.random.default_rng(seed)
    n = obj.n
    n_epochs = epochs if epochs is not None else 50
    g0 = obj.grad(w)
    rec.tick(grad=1)
    gnorm = float(np.linalg.norm(g0))
    rec.log(obj.value(w), gnorm)
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            g = _batch_grad(obj, w, idx)
            if decoupled:
                g = g - obj.lam * obj._reg * w
            w = w - rule(state, g, eta, eps, **kwargs)
            if decoupled:
                w = w - eta * obj.lam * obj._reg * w
            rec.tick(grad=len(idx) / n)
        f = obj.value(w)
        gnorm = float(np.linalg.norm(obj.grad(w)))
        rec.log(f, gnorm)
        if not np.isfinite(f):
            break
    return rec.finish(w, converged=gnorm < tol)


def adagrad(obj, w0, **kw) -> OptResult:
    return adaptive(obj, w0, method="AdaGrad", **kw)


def rmsprop(obj, w0, **kw) -> OptResult:
    return adaptive(obj, w0, method="RMSprop", **kw)


def adam(obj, w0, **kw) -> OptResult:
    return adaptive(obj, w0, method="Adam", **kw)
