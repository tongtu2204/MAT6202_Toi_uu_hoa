"""The one objective everything optimizes: L2-regularized logistic regression.

    p_i   = sigma(w . x_i)
    f(w)  = (1/n) Σ [ -y_i log p_i - (1-y_i) log(1-p_i) ] + (lambda/2)||w||^2
    ∇f(w) = (1/n) Xᵀ(p - y) + lambda w
    ∇²f(w)= (1/n) Xᵀ S X + lambda I,   S = diag(p_i(1-p_i))

Newton on this = IRLS. Bias handling: an all-ones column is expected to be
absent from X (features are standardized), so `fit_intercept` appends one and
leaves it UNREGULARIZED — matching how the sklearn baseline is compared.

sklearn note (see CLAUDE.md): sklearn minimizes a SUM with penalty (1/2)wᵀw and
C = 1/(lambda*n) (NOT 1/(2*lambda*n) — our objective averages the loss, so matching
the penalty/loss ratio gives 1/(n*lambda); see `sklearn_C`). Use it to build a
matching baseline and compare OBJECTIVE VALUE, not accuracy.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticObjective:
    def __init__(self, X: np.ndarray, y: np.ndarray, lam: float,
                 fit_intercept: bool = True):
        self.lam = float(lam)
        self.fit_intercept = fit_intercept
        if fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.X = X
        self.y = y.astype(float)
        self.n, self.d = X.shape
        # per-coordinate regularization mask (0 on the intercept column)
        self._reg = np.ones(self.d)
        if fit_intercept:
            self._reg[-1] = 0.0

    def probs(self, w: np.ndarray) -> np.ndarray:
        return sigmoid(self.X @ w)

    def value(self, w: np.ndarray) -> float:
        z = self.X @ w
        # log-sum-exp stable cross-entropy: log(1+e^z) - y z
        ll = np.mean(np.logaddexp(0.0, z) - self.y * z)
        reg = 0.5 * self.lam * np.sum((self._reg * w) ** 2)
        return float(ll + reg)

    def grad(self, w: np.ndarray) -> np.ndarray:
        p = self.probs(w)
        return self.X.T @ (p - self.y) / self.n + self.lam * self._reg * w

    def hessian(self, w: np.ndarray) -> np.ndarray:
        p = self.probs(w)
        s = p * (1.0 - p)
        H = (self.X * s[:, None]).T @ self.X / self.n
        H[np.diag_indices(self.d)] += self.lam * self._reg
        return H

    def sklearn_C(self) -> float:
        """C such that sklearn's summed objective matches this averaged one."""
        return 1.0 / (self.lam * self.n)


def empirical_lipschitz(obj: LogisticObjective, w: np.ndarray) -> float:
    """Largest Hessian eigenvalue at `w` — the LOCAL smoothness constant.

    The reported L = (1/4n)lambda_max(XᵀX) + lambda is the GLOBAL bound, using
    S = p(1-p) <= 1/4. That bound is attained EXACTLY at w=0 (there p_i=1/2, so
    S=1/4): lambda_max(H(0)) == L. Away from 0 the predictions sharpen, S shrinks,
    and the local curvature drops well below L — which is precisely why plain GD on
    this REGULARIZED objective tolerates steps far past the textbook 2/L before it
    misbehaves (true blow-up needs eta > 2/lambda). Use this to report the honest
    curvature at any point along the path.
    """
    return float(np.linalg.eigvalsh(obj.hessian(w))[-1])
