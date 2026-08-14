"""Shared step-size utilities: backtracking (Armijo) line search.

Used by GD (optional adaptive step) and Newton (damping). Kept separate so the
optimizer files stay focused on the update rule, not the step selection.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def armijo_backtrack(
    f: Callable[[np.ndarray], float],
    w: np.ndarray,
    fw: float,
    grad: np.ndarray,
    direction: np.ndarray,
    t0: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Return step t satisfying f(w+t d) <= f(w) + c t ∇fᵀd. `direction` is a
    descent direction (e.g. -grad for GD, -H^{-1}grad for Newton)."""
    slope = float(grad @ direction)
    t = t0
    for _ in range(max_iter):
        if f(w + t * direction) <= fw + c * t * slope:
            return t
        t *= beta
    return t
