#!/usr/bin/env python3
"""Run the from-scratch optimizer benchmark on a design-matrix variant.

Prerequisite: run_pipeline.py must have written artifacts/<variant>.npz.

Usage:
    python run_benchmark.py                       # ridge variant, all optimizers
    python run_benchmark.py --variant poly --lam 1e-3
    python run_benchmark.py --variant lasso --l1   # ISTA/FISTA bonus

Produces the convergence figures (artifacts/figures/) and prints the L/mu/kappa
and f* used for the log(f - f*) slides.
"""
from __future__ import annotations

import blas_threads  # must precede numpy: pins BLAS to 1 thread
import argparse

from optim.benchmark import run_benchmark, run_l1_benchmark
from optim import plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="ridge", choices=["ridge", "lasso", "poly"])
    p.add_argument("--lam", type=float, default=1e-3, help="L2 lambda")
    p.add_argument("--l1", action="store_true", help="run ISTA/FISTA (lasso variant)")
    p.add_argument("--alpha", type=float, default=1e-3, help="L1 strength (--l1)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_l1(a: argparse.Namespace) -> None:
    variant = a.variant if a.variant != "ridge" else "lasso"
    bench = run_l1_benchmark(variant=variant, alpha=a.alpha)
    print(f"[{bench.variant} L1]  alpha={bench.alpha:g}  L={bench.L:.4g}  "
          f"d_reg={bench.d_reg}")
    print(f"  F*        (FISTA deep)  = {bench.f_star:.12f}")
    print(f"  F_sklearn (saga)        = {bench.f_sklearn:.12f}  "
          f"nnz={bench.nnz_sklearn}/{bench.d_reg}")
    print(f"  gap = F_sklearn - F*    = {bench.f_sklearn - bench.f_star:+.3e}"
          "   (near 0 -> composite objective & F* validated)")
    for name in ("ISTA", "FISTA"):
        r = bench.results.get(name)
        if r is not None:
            print(f"    {name:<5} iters={len(r.f_history):<5} "
                  f"F-F*={r.f_history[-1] - bench.f_star:.3e}  "
                  f"nnz={r.nnz[-1]}/{bench.d_reg}  conv={r.converged}")
    fig = plots.l1_convergence(bench)
    print(f"  figure: {fig}")


def main() -> None:
    a = parse_args()
    if a.l1:
        run_l1(a)
        return

    bench = run_benchmark(variant=a.variant, lam=a.lam, seed=a.seed)

    c = bench.conditioning
    print(f"[{bench.variant}]  n={c['n']}  d={c['d']}  "
          f"kappa_upper={c['kappa_upper']:.4g}")
    print(f"  f*        (Newton deep) = {bench.f_star:.12f}")
    print(f"  f_sklearn (lbfgs)       = {bench.f_sklearn:.12f}")
    print(f"  gap = f_sklearn - f*    = {bench.f_sklearn - bench.f_star:+.3e}"
          "   (near 0 -> objective form & f* validated)")
    print("  per-optimizer final suboptimality:")
    for name in ("Newton", "AGD", "GD", "SGD"):
        r = bench.results.get(name)
        if r is not None:
            print(f"    {name:<7} iters={len(r.f_history):<5} "
                  f"f-f*={r.f_history[-1] - bench.f_star:.3e}  conv={r.converged}")

    fig1 = plots.convergence_curves(bench)
    print(f"  figure: {fig1}")
    print("  step-size / divergence study: python run_experiments.py quad-div gd-sweep")


if __name__ == "__main__":
    main()
