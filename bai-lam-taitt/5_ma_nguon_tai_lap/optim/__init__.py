"""optim — from-scratch optimizer benchmark for the MAT6202 test bed.

Second deliverable of the project: implement GD, Accelerated GD (Nesterov),
Newton (damped / line-search = IRLS) and SGD by hand on the L2-regularized
logistic objective produced by `churn_opt`, then compare their per-iteration
cost vs convergence rate (governed by kappa). ISTA/FISTA on the `lasso` variant
is the optional L1 bonus.

Module order mirrors the execution flow:
    data        -> load artifacts/<variant>.npz design matrices
    objective   -> f(w), grad(w), hessian(w) (sklearn-matched form)
    linesearch  -> shared Armijo / backtracking utilities
    optimizers/ -> one file per algorithm, all returning an OptResult history
    benchmark   -> run every optimizer on a variant, estimate f*, record curves
    experiments -> parameter sweeps + honesty studies (divergence, dim scaling)
    plots       -> log(f - f*) convergence curves and the sweep/study figures
"""
from .objective import LogisticObjective
from .data import load_variant, Dataset

__all__ = ["LogisticObjective", "load_variant", "Dataset"]
