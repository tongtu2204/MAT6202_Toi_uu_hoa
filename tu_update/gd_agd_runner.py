"""Instrumented GD and AGD loops used only by the revised experiment."""
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
    convergence_time_s: float | None
    finite: bool


def _wanted(checkpoints) -> tuple[int, ...]:
    values = tuple(sorted({int(k) for k in checkpoints if int(k) > 0}))
    if not values:
        raise ValueError("checkpoints must contain a positive integer")
    return values


def _safe_state(w: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(w)) and np.linalg.norm(w) < 1e12)


def run_gd(obj, w0, eta: float, checkpoints, tol: float = 1e-10,
           stop_on_convergence: bool = False) -> Run:
    wanted = _wanted(checkpoints)
    w = np.asarray(w0, dtype=float).copy()
    snapshots: dict[int, Snapshot] = {}
    converged_at = None
    convergence_time_s = None
    finite = True
    start = time.perf_counter()

    for k in range(1, wanted[-1] + 1):
        g = obj.grad(w)
        if not np.all(np.isfinite(g)):
            finite = False
            break
        if converged_at is None and float(np.linalg.norm(g)) <= tol:
            converged_at = k - 1
            convergence_time_s = time.perf_counter() - start
            if stop_on_convergence:
                break

        w = w - eta * g
        if not _safe_state(w):
            finite = False
            break
        if k in wanted:
            # Capture algorithm time before extra diagnostic calls.
            elapsed = time.perf_counter() - start
            f = float(obj.value(w))
            gn = float(np.linalg.norm(obj.grad(w)))
            snapshots[k] = Snapshot(k, w.copy(), f, gn, elapsed)
            finite = finite and bool(np.isfinite(f) and np.isfinite(gn))

    return Run("GD", float(eta), None, snapshots, converged_at,
               convergence_time_s, finite)


def run_agd(obj, w0, eta: float, beta: float, checkpoints,
            tol: float = 1e-10, stop_on_convergence: bool = False) -> Run:
    wanted = _wanted(checkpoints)
    w = np.asarray(w0, dtype=float).copy()
    w_prev = w.copy()
    snapshots: dict[int, Snapshot] = {}
    converged_at = None
    convergence_time_s = None
    finite = True
    start = time.perf_counter()

    for k in range(1, wanted[-1] + 1):
        y = w + beta * (w - w_prev)
        g = obj.grad(y)
        if not np.all(np.isfinite(g)):
            finite = False
            break
        if converged_at is None and float(np.linalg.norm(g)) <= tol:
            converged_at = k - 1
            convergence_time_s = time.perf_counter() - start
            if stop_on_convergence:
                break

        w_prev, w = w, y - eta * g
        if not _safe_state(w):
            finite = False
            break
        if k in wanted:
            elapsed = time.perf_counter() - start
            f = float(obj.value(w))
            gn = float(np.linalg.norm(obj.grad(w)))
            snapshots[k] = Snapshot(k, w.copy(), f, gn, elapsed)
            finite = finite and bool(np.isfinite(f) and np.isfinite(gn))

    return Run("AGD", float(eta), float(beta), snapshots, converged_at,
               convergence_time_s, finite)


def beta_theory(L: float, mu: float) -> float:
    """Curvature-based constant momentum used by the untuned baseline."""
    root = np.sqrt(float(L) / float(mu))
    return float((root - 1.0) / (root + 1.0))

