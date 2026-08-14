"""Shared optimizer plumbing: the OptResult history container + a recorder.

Every optimizer logs, per outer iteration:
    f_history   -> objective value (for log(f - f*) curves)
    grad_norm   -> ||∇f|| (stopping / diagnostics)
    n_grad      -> cumulative gradient evaluations (per-iteration-cost axis)
    n_hess      -> cumulative Hessian solves (Newton's O(d^3) cost)
    time_s      -> wallclock (secondary cost axis)

Keeping cost counters is the whole point of the benchmark: the thesis is
per-iteration cost vs convergence rate, so curves get plotted against BOTH
iteration index and cumulative work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptResult:
    name: str
    w: np.ndarray | None = None
    f_history: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    n_grad: list[float] = field(default_factory=list)   # fractional: SGD counts |B|/n
    n_hess: list[int] = field(default_factory=list)
    time_s: list[float] = field(default_factory=list)
    nnz: list[int] = field(default_factory=list)   # nonzeros per iter (L1 methods)
    steps: list[float] = field(default_factory=list)  # step size ACTUALLY taken per iter
    converged: bool = False
    note: str = ""                                 # why a run stopped abnormally

    def suboptimality(self, f_star: float) -> np.ndarray:
        """f_k - f* clipped to a tiny floor so log-plots stay finite."""
        return np.maximum(np.asarray(self.f_history) - f_star, 1e-16)


class Recorder:
    """Mutable accumulator an optimizer's inner loop calls once per step."""

    def __init__(self, name: str):
        self.res = OptResult(name=name)
        self._t0 = time.perf_counter()
        self._grad = 0.0
        self._hess = 0

    def tick(self, *, grad: float = 0.0, hess: int = 0) -> None:
        """Count work. `grad` is FRACTIONAL on purpose: a minibatch of size |B|
        costs |B|/n of a full gradient, so SGD's cost axis stays comparable with
        the full-batch methods."""
        self._grad += grad
        self._hess += hess

    def log(self, f: float, gnorm: float) -> None:
        r = self.res
        r.f_history.append(float(f))
        r.grad_norm.append(float(gnorm))
        r.n_grad.append(self._grad)
        r.n_hess.append(self._hess)
        r.time_s.append(time.perf_counter() - self._t0)

    def finish(self, w: np.ndarray, converged: bool, note: str = "") -> OptResult:
        self.res.w = w
        self.res.converged = converged
        self.res.note = note
        return self.res
