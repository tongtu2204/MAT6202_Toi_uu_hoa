"""TIER 0 GATE — Hessian-spectrum + BLAS diagnostics that decide the whole revision.

Everything in Tier 2 (kappa-sweep T2.1, thesis-reframe T2.4, the honest kappa
table T1.1) is locked until these numbers come back. The single decision number
is the data-Hessian condition ceiling

    kappa_max = lambda_max(data H) / lambda_min(data H)   (the lambda -> 0 limit)

because mu_star(lambda) = lambda_min(data H) + lambda can only DRIVE kappa DOWN
toward kappa_max, never above it: kappa_max is a property of the DATA, not lambda.

    kappa_max >= 1e3   -> branch A  (lambda-sweep spans >= 2-3 decades on real data)
    1e2 <= .. < 1e3    -> branch A' (thin but fittable; add 2 lambda points)
    kappa_max < 1e2    -> branch B  (lambda can't move kappa -> synthetic test bed)

Run:  python -m optim.diagnostics      # writes artifacts/gate.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import load_variant, ARTIFACTS
from .objective import LogisticObjective


# --------------------------------------------------------------------------
def solve_deep(obj: LogisticObjective, tol: float = 1e-13,
               max_it: int = 100) -> np.ndarray:
    """Damped Newton to deep convergence -> reference w_star. Solve, never inv()."""
    w = np.zeros(obj.d)
    for _ in range(max_it):
        g = obj.grad(w)
        if np.max(np.abs(g)) < tol:
            break
        H = obj.hessian(w)
        delta = np.linalg.solve(H, g)          # Cholesky/LU, not inv()
        t, f0 = 1.0, obj.value(w)
        gd = float(g @ delta)
        while obj.value(w - t * delta) > f0 - 1e-4 * t * gd:
            t *= 0.5
            if t < 1e-12:
                break
        w = w - t * delta
    return w


def _data_hessian(obj: LogisticObjective, w: np.ndarray) -> np.ndarray:
    """(1/n) Xᵀ S X  at w — the pure data curvature, no lambda term."""
    p = obj.probs(w)
    s = p * (1.0 - p)
    return (obj.X * s[:, None]).T @ obj.X / obj.n


# --------------------------------------------------------------------------
def spectrum_gate(X: np.ndarray, y: np.ndarray, variant: str,
                  lam_ref: float = 1e-3,
                  lam_grid=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)) -> dict:
    """Full local-spectrum report for one variant.

    kappa_max comes from the data Hessian at w_star(lam_ref). Each row then
    RE-SOLVES w_star(lambda) and reports the EXACT local mu/L/kappa (this is the
    T2.1 data), alongside the cheap r+lambda approximation for cross-check.
    """
    obj_ref = LogisticObjective(X, y, lam=lam_ref)
    n, d = obj_ref.n, obj_ref.d                 # d includes the intercept column
    w_star = solve_deep(obj_ref)

    # data-only Hessian spectrum -> the lambda-independent ceiling
    ev_data = np.linalg.eigvalsh(_data_hessian(obj_ref, w_star))
    r, L_data = float(ev_data[0]), float(ev_data[-1])
    kappa_max = L_data / r if r > 1e-14 else float("inf")

    rows = []
    for lam in lam_grid:
        obj_l = LogisticObjective(X, y, lam=lam)
        w_l = solve_deep(obj_l)
        ev = np.linalg.eigvalsh(obj_l.hessian(w_l))
        mu_exact, L_exact = float(ev[0]), float(ev[-1])
        rows.append(dict(
            lam=lam,
            mu_star=mu_exact, L_star=L_exact, kappa_star=L_exact / mu_exact,
            mu_approx=r + lam, L_approx=L_data + lam,
            kappa_approx=(L_data + lam) / (r + lam),
        ))

    # upper-bound (deck) numbers on the intercept-augmented design, matching
    # churn_opt.diagnostics.conditioning(obj.X, lam)
    XtX = obj_ref.X.T @ obj_ref.X
    lam_max_XtX = float(np.linalg.eigvalsh(XtX)[-1])
    L_bound = lam_max_XtX / (4 * n) + lam_ref

    return dict(
        variant=variant, n=n, d=d,
        rank_X=int(np.linalg.matrix_rank(obj_ref.X)),
        lam_min_data=r, L_data=L_data, kappa_max=kappa_max,
        lam_max_XtX=lam_max_XtX, trace_XtX=float(np.trace(XtX)),
        L_bound=L_bound, kappa_bound=L_bound / lam_ref,
        lam_ref=lam_ref, rows=rows,
    )


# --------------------------------------------------------------------------
def blas_probe(n: int, d: int, reps: int = 7, seed: int = 0) -> dict:
    """Time the Hessian build (GEMM/BLAS-3) vs the gradient (GEMV/BLAS-2).

    If ratio_time << ratio_flop (= d) the BLAS-3 hypothesis for C2 holds: Newton
    costs d x the FLOPs of a GD step but only ratio_time x the wall-clock, and the
    d/ratio_time factor is the cache gift that pushes the crossover out past
    feasible d.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    s, res = rng.random(n), rng.random(n)

    def med(f):
        f()                                     # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    t_gemm = med(lambda: (X.T * s) @ X)         # ~2 n d^2 FLOP (BLAS-3)
    t_gemv = med(lambda: X.T @ res)             # ~2 n d   FLOP (BLAS-2)
    return dict(
        n=n, d=d,
        t_gemm=t_gemm, t_gemv=t_gemv,
        gemm_gflops=2 * n * d * d / t_gemm / 1e9,
        gemv_gflops=2 * n * d / t_gemv / 1e9,
        ratio_time=t_gemm / t_gemv,             # measured wall-clock ratio
        ratio_flop=d,                           # theoretical FLOP ratio
    )


# --------------------------------------------------------------------------
def _branch(kappa_max: float) -> str:
    if kappa_max >= 1e3:
        return "A"
    if kappa_max >= 1e2:
        return "A'"
    return "B"


def main(artifacts: Path = ARTIFACTS) -> dict:
    variants = ["ridge", "lasso", "poly"]
    gates = {}
    for v in variants:
        ds = load_variant(v)
        gates[v] = spectrum_gate(ds.X_train, ds.y_train, v)

    n_ref = gates["ridge"]["n"]
    blas = [blas_probe(n_ref, d) for d in (40, 160, 640)]

    # branch decision uses the BEST variant (largest kappa_max)
    best = max(variants, key=lambda v: gates[v]["kappa_max"])
    branch = _branch(gates[best]["kappa_max"])

    out = dict(
        gates=gates, blas=blas,
        kappa_max_by_variant={v: gates[v]["kappa_max"] for v in variants},
        best_variant=best, branch=branch,
    )
    (artifacts / "gate.json").write_text(json.dumps(out, indent=2))
    return out


def _print_summary(out: dict) -> None:
    print("=" * 72)
    print("TIER 0 GATE SUMMARY")
    print("=" * 72)
    for v, g in out["gates"].items():
        print(f"\n[{v}]  n={g['n']}  d={g['d']}  rank_X={g['rank_X']}"
              f"  (full_rank={g['rank_X'] == g['d']})")
        print(f"   data spectrum:  lam_min={g['lam_min_data']:.4e}  "
              f"L_data={g['L_data']:.4e}  ->  kappa_max={g['kappa_max']:.4g}")
        print(f"   deck bound:     L_bound={g['L_bound']:.4g}  "
              f"kappa_bound={g['kappa_bound']:.4g}")
        print(f"   XtX:            lam_max={g['lam_max_XtX']:.4e}  "
              f"trace={g['trace_XtX']:.4e}")
        print("   lambda-sweep (exact local kappa*):")
        for row in g["rows"]:
            print(f"      lam={row['lam']:.0e}  mu*={row['mu_star']:.4e}  "
                  f"L*={row['L_star']:.4e}  kappa*={row['kappa_star']:.4g}"
                  f"   (approx {row['kappa_approx']:.4g})")
    print("\n" + "-" * 72)
    print("BLAS probe (GEMM vs GEMV):")
    for b in out["blas"]:
        print(f"   d={b['d']:4d}  GEMM={b['gemm_gflops']:.1f} GFLOP/s  "
              f"GEMV={b['gemv_gflops']:.1f} GFLOP/s  "
              f"ratio_time={b['ratio_time']:.2f}  (ratio_flop=d={b['ratio_flop']})")
    print("-" * 72)
    print(f"\nkappa_max by variant: {out['kappa_max_by_variant']}")
    print(f"best variant: {out['best_variant']}   =>   BRANCH {out['branch']}")
    print("=" * 72)


if __name__ == "__main__":
    _print_summary(main())
