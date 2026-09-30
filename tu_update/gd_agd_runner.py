"""GD/AGD runners that retain weights and diagnostics at fixed checkpoints.

The original project stores only the final weight vector.  The update needs to
compare optimization and predictive metrics at several equal budgets, so this
module records immutable snapshots without changing the original optimizers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    iteration: int
    w: np.ndarray
    objective: float
    grad_norm: float
    time_s: float


@dataclass
class Run:
    method: str
    eta: float
    beta: float | None
    snapshots: dict[int, Snapshot]
    converged_at: int | None
    finite: bool


def _wanted(checkpoints) -> tuple[int, ...]:
    values = tuple(sorted({int(k) for k in checkpoints if int(k) > 0}))
    if not values:
        raise ValueError("checkpoints phải chứa ít nhất một số nguyên dương")
    return values


def run_gd(obj, w0, eta: float, checkpoints, tol: float = 1e-10) -> Run:
    wanted = _wanted(checkpoints)
    w = np.asarray(w0, dtype=float).copy()
    snapshots: dict[int, Snapshot] = {}
    converged_at = None
    finite = True
    start = time.perf_counter()

    for k in range(1, wanted[-1] + 1):
        g = obj.grad(w)
        w = w - eta * g
        if k in wanted:
            f = float(obj.value(w))
            gn = float(np.linalg.norm(obj.grad(w)))
            snapshots[k] = Snapshot(k, w.copy(), f, gn, time.perf_counter() - start)
            finite = finite and bool(np.isfinite(f) and np.isfinite(gn))
        if not np.all(np.isfinite(w)):
            finite = False
            break
        if converged_at is None and float(np.linalg.norm(g)) < tol:
            converged_at = k

    return Run("GD", float(eta), None, snapshots, converged_at, finite)


def run_agd(obj, w0, eta: float, beta: float, checkpoints,
            tol: float = 1e-10) -> Run:
    wanted = _wanted(checkpoints)
    w = np.asarray(w0, dtype=float).copy()
    w_prev = w.copy()
    snapshots: dict[int, Snapshot] = {}
    converged_at = None
    finite = True
    start = time.perf_counter()

    for k in range(1, wanted[-1] + 1):
        y = w + beta * (w - w_prev)
        g = obj.grad(y)
        w_prev, w = w, y - eta * g
        if k in wanted:
            # Dùng gradient tại chính w_k cho báo cáo checkpoint. Chi phí bổ sung
            # này áp dụng như nhau cho mọi cấu hình và không được tính là một bước.
            f = float(obj.value(w))
            gn = float(np.linalg.norm(obj.grad(w)))
            snapshots[k] = Snapshot(k, w.copy(), f, gn, time.perf_counter() - start)
            finite = finite and bool(np.isfinite(f) and np.isfinite(gn))
        if not np.all(np.isfinite(w)):
            finite = False
            break
        if converged_at is None and float(np.linalg.norm(g)) < tol:
            converged_at = k

    return Run("AGD", float(eta), float(beta), snapshots, converged_at, finite)


def beta_from_eta(eta: float, mu: float) -> float:
    """Empirical matched-pair rule used by the current project.

    This is a tuning rule, not a theorem when 1/eta is smaller than the true
    global smoothness constant.
    """
    kappa_eff = (1.0 / float(eta)) / float(mu)
    root = np.sqrt(max(kappa_eff, 1.0))
    return float((root - 1.0) / (root + 1.0))


def beta_theory(L: float, mu: float) -> float:
    root = np.sqrt(float(L) / float(mu))
    return float((root - 1.0) / (root + 1.0))

