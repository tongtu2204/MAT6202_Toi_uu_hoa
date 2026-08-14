"""Conditioning diagnostics — the numbers that go on the opening experiment slide.

For f(w) = (1/n)logloss + (lambda/2)||w||^2 the Hessian is
    H(w) = (1/n) Xᵀ S X + lambda I,   S = diag(p_i(1-p_i)) in [0, 1/4].
So smoothness uses S <= 1/4 and strong convexity uses S >= 0:
    L   = (1/4) * lambda_max((1/n)XᵀX) + lambda
    mu  = lambda                      (lower bound; data curvature can vanish)
    kappa_upper = L / lambda
Full column rank of X is what keeps the Hessian invertible so Newton can run.

CAVEAT on mu. The formula above assumes the ridge term is lambda*I, which holds for X
AS PASSED HERE — without an intercept column. The benchmark appends an intercept and
does NOT penalize it, so its Hessian is (1/n)XᵀSX + lambda*diag(reg) with a zero on the
intercept: along that direction the only curvature is (1/n)sum_i p_i(1-p_i), which
decays to 0 as ||w|| grows. f is then strongly convex on bounded sublevel sets, not
globally. Anything that needs a real strong-convexity constant (certificates, PL bounds)
must use the measured lambda_min at the optimum — see optim/reference.py. `kappa_upper`
below is exactly what its name says: an UPPER bound, ~9x the local kappa* at the optimum.
"""
from __future__ import annotations

import numpy as np


def conditioning(X: np.ndarray, lam: float) -> dict:
    n, d = X.shape
    gram = (X.T @ X) / n
    eig = np.linalg.eigvalsh(gram)              # ascending, symmetric
    lambda_max = float(eig[-1])
    lambda_min = float(eig[0])
    L = 0.25 * lambda_max + lam
    mu = lam
    rank = int(np.linalg.matrix_rank(X))
    return {
        "n": n,
        "d": d,
        "lambda": lam,
        "gram_lambda_max": lambda_max,
        "gram_lambda_min": lambda_min,
        "L": L,
        "mu": mu,
        "kappa_upper": L / mu,
        "rank": rank,
        "full_rank": rank == d,
        "newton_safe": rank == d,               # invertible Hessian
    }


def format_report(name: str, c: dict) -> str:
    return (
        f"[{name}]  n={c['n']}  d={c['d']}  rank={c['rank']}"
        f"  full_rank={c['full_rank']}\n"
        f"      lambda={c['lambda']:.3g}  L={c['L']:.4g}  mu={c['mu']:.3g}"
        f"  kappa_upper=L/mu={c['kappa_upper']:.4g}\n"
        f"      gram eig: min={c['gram_lambda_min']:.3e}  max={c['gram_lambda_max']:.3e}"
        f"  ({'Newton OK' if c['newton_safe'] else 'SINGULAR — Newton will fail'})"
    )
