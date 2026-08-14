#!/usr/bin/env python3
"""Run ONE stage of the revision suite and merge its numbers into rev1_numbers.json.

`run_revision.py` does all of these in one go. At d=415 that is a multi-hour job,
and a multi-hour job that writes its results once at the end is a job that loses
everything when it is interrupted — which is exactly what happened. This driver
runs a single stage, merges (read-modify-write) into artifacts/rev1_numbers.json,
and exits, so progress is durable at stage granularity.

    python run_stage.py kappa-honesty
    python run_stage.py step-sweep newton-init three-regimes
    python run_stage.py --list

Measured stage costs on ridge (n=75,026, d=415), 1 BLAS thread:
    kappa-honesty  ~55 min      kappa-sweep   ~35 min (subsampled to 15k rows)
    step-sweep     ~15 min      newton-init   ~2 min
    three-regimes  ~10 min      affine        ~9 min
    sgd-multiseed  ~5 min       armijo        ~90 min (incl. fixed-vs-backtracking)
    sklearn        ~15 min      breakeven     ~3 min
    money          ~4 min       l1            ~60 min (re-measured: 3596s / 3956s on two
                                              runs; the "~40 min" here was stale)
"""
from __future__ import annotations

import blas_threads  # must precede numpy: pins BLAS to 1 thread
import numpy as np
import argparse
import json
import time
from pathlib import Path

from optim import experiments as E
from optim import plots
from optim.benchmark import run_benchmark, run_l1_benchmark

ART = Path(__file__).resolve().parent / "artifacts"
NUMS = ART / "rev1_numbers.json"


def _merge(key: str, value) -> None:
    """Read-modify-write so a stage never clobbers another stage's numbers."""
    nums = json.loads(NUMS.read_text()) if NUMS.exists() else {}
    nums[key] = value
    NUMS.write_text(json.dumps(nums, indent=2, default=float))


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def _kappa_honesty():
    kh = E.kappa_honesty()
    plots.kappa_honesty_plot(kh)
    _merge("kappa_honesty", dict(
        kappa_bound=kh.kappa_bound, kappa_star=kh.kappa_star,
        L_bound=kh.L_bound, L_star=kh.L_star, mu_star=kh.mu_star,
        pred_gd_bound=kh.pred_gd_bound, pred_gd_star=kh.pred_gd_star,
        meas_gd=kh.meas_gd, eps=kh.eps, n_used=kh.n_used))
    print(f"  n_used={kh.n_used}  kappa_bound={kh.kappa_bound:.4g} "
          f"kappa*={kh.kappa_star:.4g}  predicted GD: bound={kh.pred_gd_bound:.4g} "
          f"local={kh.pred_gd_star:.4g}  measured={kh.meas_gd}")


def _kappa_sweep():
    ks = E.kappa_sweep()
    plots.kappa_sweep_plot(ks)
    _merge("kappa_sweep", dict(
        n_used=ks.n_used, slope_gd=ks.slope_gd, slope_agd=ks.slope_agd,
        rows=[dict(lam=r["lam"], kappa_star=r["kappa_star"],
                   it_gd=r["it_gd"], it_agd=r["it_agd"]) for r in ks.rows]))
    print(f"  n_used={ks.n_used}  slope_gd={ks.slope_gd:.3f}  "
          f"slope_agd={ks.slope_agd:.3f}")
    for r in ks.rows:
        print(f"    lam={r['lam']:<8g} kappa*={r['kappa_star']:>9.1f} "
              f"GD={r['it_gd']:>7} AGD={r['it_agd']:>6}")


def _step_sweep():
    sw = E.gd_step_sweep()
    plots.step_sweep_plot(sw)
    _merge("gd_step_sweep", dict(
        L=sw.L, L_star=sw.L_star, f_star=sw.f_star, mults=sw.mults,
        rows=[dict(mult=m, iters=len(sw.runs[m].f_history),
                   gap=float(sw.runs[m].f_history[-1] - sw.f_star),
                   converged=bool(sw.runs[m].converged)) for m in sw.mults]))
    print(f"  L={sw.L:.4g}  L*={sw.L_star:.4g}")


def _newton_init():
    ni = E.newton_init_study()
    plots.newton_init_plot(ni)
    _merge("newton_init", dict(
        w0_scale=ni.w0_scale, f_star=ni.f_star,
        pure=dict(iters=len(ni.pure.f_history), converged=bool(ni.pure.converged),
                  max_f=float(max(ni.pure.f_history)), note=ni.pure.note),
        damped=dict(iters=len(ni.damped.f_history),
                    converged=bool(ni.damped.converged))))
    print(f"  pure: {len(ni.pure.f_history)} it conv={ni.pure.converged} | "
          f"damped: {len(ni.damped.f_history)} it conv={ni.damped.converged}")
    if ni.pure.note:
        print(f"  pure note: {ni.pure.note}")


def _three_regimes():
    tr = E.three_regimes()
    plots.three_regimes_plot(tr)
    # The run costs ~12 min but the figure is pure styling; cache the curves so a
    # colour/legend tweak can be re-rendered offline (see replot_three_regimes.py).
    np.savez_compressed(ART / "three_regimes_ridge.npz",
                        etas=np.asarray(tr.etas, dtype=float),
                        two_over_L=tr.two_over_L, two_over_lam=tr.two_over_lam,
                        f_star=tr.f_star, variant=tr.variant,
                        **{f"h_{i}": tr.histories[e] for i, e in enumerate(tr.etas)})
    _merge("three_regimes", dict(two_over_L=tr.two_over_L,
                                 two_over_lam=tr.two_over_lam, etas=tr.etas))
    print(f"  2/L={tr.two_over_L:.4g}  2/lam={tr.two_over_lam:.4g}")


def _affine():
    ai = E.affine_invariance()
    plots.affine_plot(ai)
    _merge("affine", dict(kappa_std=ai.kappa_std, kappa_raw=ai.kappa_raw,
                          iters_newton_std=ai.iters_newton_std,
                          iters_newton_raw=ai.iters_newton_raw,
                          n_nonpos_std=ai.n_nonpos_std,
                          n_nonpos_raw=ai.n_nonpos_raw,
                          note_raw=ai.newton_raw.note))
    print(f"  Newton std={ai.iters_newton_std} it, raw={ai.iters_newton_raw} it | "
          f"kappa std={ai.kappa_std:.3e} raw={ai.kappa_raw:.3e} | "
          f"non-positive eigenvalues of H(w*): std={ai.n_nonpos_std} raw={ai.n_nonpos_raw}")
    if ai.newton_raw.note:
        print(f"  raw note: {ai.newton_raw.note}")


def _sgd_multiseed():
    sm = E.sgd_multiseed()
    plots.sgd_multiseed_plot(sm)
    _merge("sgd", dict(
        batches=[int(b) for b in sm.mean_gap],
        final_std={int(b): float(sm.std_gap[b][-1]) for b in sm.mean_gap},
        final_mean={int(b): float(sm.mean_gap[b][-1]) for b in sm.mean_gap}))


def _armijo():
    asw = E.armijo_sweep()
    _merge("armijo_sweep", dict(t0=asw.t0, rows=asw.rows))
    fb = E.fixed_vs_backtracking()
    _merge("fixed_vs_backtracking", fb.rows)
    for r in fb.rows:
        print(f"    {r['method']:<7} {r['mode']:<13} iters={r['iters']:>6} "
              f"fevals={r['fevals']:>6} time={r['time']:.2f}s")


def _sklearn():
    st = E.sklearn_timing()
    _merge("sklearn_timing", dict(f_star=st.f_star, rows=st.rows))
    for r in st.rows:
        print(f"    {r['solver']:<28} gap={r['gap']:.3e} iters={r['iters']:>6} "
              f"t={r['time']:.3f}s")


def _breakeven():
    be = E.breakeven_analysis()
    plots.breakeven_plot(be)
    _merge("breakeven", dict(kappa_star=be.kappa_star, d_star=be.d_star, n=be.n,
                             eps=be.eps, k_newton=be.k_newton,
                             d_cube_ratio=be.d_cube_ratio, blas=be.blas))
    print(f"  d*={be.d_star:.1f}  k_newton={be.k_newton}")


def _money():
    bench = run_benchmark(variant="ridge")
    plots.money_plots(bench)
    plots.convergence_curves(bench)
    _merge("money", dict(
        f_star=bench.f_star, f_star_bound=bench.f_star_bound,
        f_star_grad_norm=bench.f_star_grad_norm, f_sklearn=bench.f_sklearn,
        sgd_config=bench.sgd_config, gd_config=bench.gd_config,
        agd_config=bench.agd_config,
        kappa_upper=bench.conditioning["kappa_upper"], L=bench.conditioning["L"],
        rows={k: dict(iters=len(v.f_history),
                      gap=float(v.f_history[-1] - bench.f_star),
                      time=float(v.time_s[-1]), converged=bool(v.converged))
              for k, v in bench.results.items()}))
    for k, v in bench.results.items():
        print(f"    {k:<7} iters={len(v.f_history):>5} "
              f"gap={v.f_history[-1] - bench.f_star:.3e} t={v.time_s[-1]:.2f}s "
              f"conv={v.converged}")


def _l1():
    l1 = run_l1_benchmark(alpha=1e-3, max_iter=8000)
    plots.l1_convergence(l1)
    _merge("l1", dict(f_star=l1.f_star, saga_f=l1.f_sklearn,
                      nnz_saga=l1.nnz_sklearn, d_reg=l1.d_reg,
                      subgrad_eta0=l1.subgrad_eta0,
                      methods={k: dict(iters=len(v.f_history),
                                       gap=float(v.f_history[-1] - l1.f_star),
                                       nnz=int(v.nnz[-1]) if v.nnz else -1,
                                       converged=bool(v.converged))
                               for k, v in l1.results.items()}))
    for k, v in l1.results.items():
        print(f"    {k:<6} iters={len(v.f_history):>5} "
              f"gap={v.f_history[-1] - l1.f_star:.3e} nnz={v.nnz[-1]}")


STAGES = {
    "kappa-honesty": _kappa_honesty, "kappa-sweep": _kappa_sweep,
    "step-sweep": _step_sweep, "newton-init": _newton_init,
    "three-regimes": _three_regimes, "affine": _affine,
    "sgd-multiseed": _sgd_multiseed, "armijo": _armijo,
    "sklearn": _sklearn, "breakeven": _breakeven,
    "money": _money, "l1": _l1,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", choices=list(STAGES) + ["all"])
    ap.add_argument("--list", action="store_true", help="print stage names and exit")
    a = ap.parse_args()
    if a.list or not a.stages:
        print("stages:", " ".join(STAGES))
        return

    chosen = list(STAGES) if "all" in a.stages else a.stages
    print("BLAS threads:", blas_threads.verify(), flush=True)
    for name in chosen:
        t = time.perf_counter()
        print(f"[{name}] ...", flush=True)
        STAGES[name]()
        print(f"[{name}] done in {time.perf_counter() - t:.1f}s -> {NUMS.name}",
              flush=True)


if __name__ == "__main__":
    main()
