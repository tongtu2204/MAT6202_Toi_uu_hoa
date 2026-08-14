#!/usr/bin/env python3
"""Run the presentation's parameter sweeps and the two honesty studies (A, B).

These sit on top of the headline benchmark (run_benchmark.py) and produce the
per-slide figures in artifacts/figures/. Prerequisite: run_pipeline.py has
written the design matrices (ridge/lasso/poly, and ridge_raw for --std).

Usage:
    python run_experiments.py all                 # every study
    python run_experiments.py gd-sweep quad-div   # pick a subset
    python run_experiments.py dim --dmax 640      # dimension scaling

Studies:
    gd-sweep    (Sweep 1 + B) GD across eta = mult/L; the 2/L bound is pessimistic
                on the regularized logistic (self-conditioning off w0).
    quad-div    (B)  clean eta>2/L divergence on the local quadratic model.
    newton-init (Sweep 2) pure vs damped Newton from a far w0.
    sgd         (Sweep 3) batch size + constant-vs-diminishing step (variance floor).
    sgd-hyper   (Sweep 3b) grid over (eta0, batch) + gamma: WHERE the SGD numbers
                in optim/benchmark.py come from. Writes artifacts/sgd_hyper.json.
    std         (Sweep 4) standardization contrast (needs ridge_raw.npz).
    dim         (A)  Newton vs first-order cost as dimension grows.
"""
from __future__ import annotations

import blas_threads  # must precede numpy: pins BLAS to 1 thread
import argparse
import json
from pathlib import Path

from optim import experiments as E
from optim import plots

ART = Path(__file__).resolve().parent / "artifacts"
STUDIES = ("gd-sweep", "quad-div", "newton-init", "sgd", "sgd-hyper", "std", "dim")


def _gd_sweep(a):
    sw = E.gd_step_sweep(variant=a.variant, lam=a.lam)
    print(f"  L={sw.L:.3f}  L_star={sw.L_star:.3f}  f*={sw.f_star:.6f}")
    for m in sw.mults:
        r = sw.runs[m]
        print(f"    eta={m:>5}/L  iters={len(r.f_history):5d}  "
              f"f-f*={r.f_history[-1] - sw.f_star:.2e}  conv={r.converged}")
    print("  figure:", plots.step_sweep_plot(sw))


def _quad_div(a):
    qd = E.quadratic_divergence(variant=a.variant, lam=a.lam)
    for m in qd.mults:
        h = qd.histories[m]
        print(f"    eta={m:>5}/L  last q={h[-1]:.3e}  "
              f"{'converges' if m < 2 else 'DIVERGES'}")
    print("  figure:", plots.quadratic_divergence_plot(qd))


def _newton_init(a):
    ni = E.newton_init_study(variant=a.variant, lam=a.lam, w0_scale=a.scale)
    print(f"    pure   : conv={ni.pure.converged}  maxf={max(ni.pure.f_history):.1f}")
    print(f"    damped : conv={ni.damped.converged}  iters={len(ni.damped.f_history)}")
    print("  figure:", plots.newton_init_plot(ni))


def _sgd(a):
    sw = E.sgd_config_sweep(variant=a.variant, lam=a.lam)
    print(f"  f*={sw.f_star:.6f}")
    for label, r in {**sw.floor_runs, **sw.batch_runs}.items():
        print(f"    {label:<24} final f-f*={r.f_history[-1] - sw.f_star:.2e}")
    print("  figure:", plots.sgd_sweep_plot(sw))


def _sgd_hyper(a):
    hp = E.sgd_hyper_sweep(variant=a.variant, lam=a.lam)
    print(f"  f*={hp.f_star:.9f}   budget={hp.epochs} epoch   tol={hp.tol:g}")
    print(f"  {'batch':>6} {'eta0':>6} {'f-f*':>11} {'epoch->tol':>11} {'s':>6}")
    for r in hp.grid:
        hit = "-" if r["epochs_to_tol"] < 0 else str(r["epochs_to_tol"])
        gap = "diverged" if r["diverged"] else f"{r['final_gap']:.3e}"
        print(f"  {r['batch']:>6} {r['eta0']:>6g} {gap:>11} {hit:>11} {r['time_s']:>6.1f}")
    print(f"  gamma sweep at eta0={hp.best['eta0']:g}, batch={hp.best['batch']}:")
    for r in hp.gamma_rows:
        hit = "-" if r["epochs_to_tol"] < 0 else str(r["epochs_to_tol"])
        print(f"    gamma={r['gamma']:<8g} f-f*={r['final_gap']:.3e}  epoch->tol={hit}")
    print(f"  CHOSEN: {hp.best}")
    print("  figure:", plots.sgd_hyper_plot(hp))
    out = ART / "sgd_hyper.json"
    out.write_text(json.dumps(dict(
        variant=hp.variant, lam=hp.lam, epochs=hp.epochs, tol=hp.tol,
        f_star=hp.f_star, eta_grid=hp.eta_grid, batch_grid=hp.batch_grid,
        gamma_grid=hp.gamma_grid, grid=hp.grid, gamma_rows=hp.gamma_rows,
        best=hp.best, eta_max_stable=hp.eta_max_stable), indent=1))
    print("  numbers:", out)


def _std(a):
    sc = E.standardization_contrast(lam=a.lam)
    print(f"    kappa: z-scored={sc.kappa_std:.0f}  raw={sc.kappa_raw:.3e}")
    for tag, runs, fstar in (("std", sc.std_runs, sc.f_star_std),
                             ("raw", sc.raw_runs, sc.f_star_raw)):
        for name, r in runs.items():
            print(f"    [{tag}] {name:<7} iters={len(r.f_history):5d}  "
                  f"f-f*={r.f_history[-1] - fstar:.2e}")
    print("  figure:", plots.standardization_plot(sc))


def _dim(a):
    d_list = tuple(d for d in (40, 80, 160, 320, 640, 1280) if d <= a.dmax)
    scan = E.dim_scaling(variant=a.variant, lam=a.lam, d_list=d_list)
    print(f"  {'d':>5} {'kappa':>7} {'pi_N(ms)':>9} {'pi_AGD(ms)':>11} "
          f"{'N/AGD':>6} {'t_N':>6} {'t_AGD':>7}")
    for r in scan.rows:
        print(f"  {r['d']:>5} {r['kappa']:>7.0f} {r['pi_newton']*1e3:>9.2f} "
              f"{r['pi_agd']*1e3:>11.2f} {r['pi_newton']/r['pi_agd']:>6.1f} "
              f"{r['t_newton']:>6.2f} {r['t_agd']:>7.2f}")
    print("  figure:", plots.dim_scaling_plot(scan))


DISPATCH = {
    "gd-sweep": _gd_sweep, "quad-div": _quad_div, "newton-init": _newton_init,
    "sgd": _sgd, "sgd-hyper": _sgd_hyper, "std": _std, "dim": _dim,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("studies", nargs="+", choices=("all",) + STUDIES,
                   help="which studies to run")
    p.add_argument("--variant", default="ridge", choices=["ridge", "lasso", "poly"])
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--scale", type=float, default=4.0, help="far-init scale (newton-init)")
    p.add_argument("--dmax", type=int, default=640, help="largest d (dim study)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    print("BLAS threads:", blas_threads.verify())   # timings are void if this != 1
    chosen = STUDIES if "all" in a.studies else a.studies
    for name in chosen:
        print(f"[{name}]")
        DISPATCH[name](a)


if __name__ == "__main__":
    main()
