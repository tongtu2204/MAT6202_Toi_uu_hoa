#!/usr/bin/env python3
"""Regenerate every figure + number for REVISION LẦN 1 in one shot.

Prereqs: run_pipeline.py (writes artifacts/*.npz incl. ridge_raw) and
`python -m optim.diagnostics` (writes artifacts/gate.json). Then:

    python run_revision.py

Writes figures to artifacts/figures/ and the scalar table values to
artifacts/rev1_numbers.json (consumed by the LaTeX tables in the deck).
BLAS is pinned to 1 thread automatically (see `import blas_threads` above) so the
per-core timing on the experimental-protocol slide is reproducible without any
shell setup.
"""
from __future__ import annotations

import blas_threads  # must precede numpy: pins BLAS to 1 thread
import json
import time
from pathlib import Path

from optim import experiments as E
from optim import plots
from optim.benchmark import run_benchmark, run_l1_benchmark

ART = Path(__file__).resolve().parent / "artifacts"
NUMS = ART / "rev1_numbers.json"


def _flush(nums: dict, stage: str, t0: float) -> None:
    """Write rev1_numbers.json after EVERY stage, not once at the end.

    At d=415 this script runs for hours, and a single write at the end means an
    interrupt (or a crash in a later stage) throws away every scalar computed so
    far even though the figures are already on disk. Learned the hard way.
    """
    NUMS.write_text(json.dumps(nums, indent=2, default=float))
    print(f"    [{stage} done at {time.perf_counter() - t0:.0f}s -> {NUMS.name}]",
          flush=True)


def main() -> None:
    t0 = time.perf_counter()
    nums: dict = {}
    nums["blas_threads"] = blas_threads.verify()   # timings are void if this != 1
    print("  BLAS threads:", nums["blas_threads"])

    print("[T1.1] kappa honesty ...", flush=True)
    kh = E.kappa_honesty()
    plots.kappa_honesty_plot(kh)
    nums["kappa_honesty"] = dict(
        kappa_bound=kh.kappa_bound, kappa_star=kh.kappa_star,
        L_bound=kh.L_bound, L_star=kh.L_star, mu_star=kh.mu_star,
        pred_gd_bound=kh.pred_gd_bound, pred_gd_star=kh.pred_gd_star,
        meas_gd=kh.meas_gd, eps=kh.eps)
    _flush(nums, "kappa_honesty", t0)

    print("[T2.1] kappa sweep (slope fit) ...", flush=True)
    ks = E.kappa_sweep()
    plots.kappa_sweep_plot(ks)
    nums["kappa_sweep"] = dict(
        slope_gd=ks.slope_gd, slope_agd=ks.slope_agd,
        rows=[dict(lam=r["lam"], kappa_star=r["kappa_star"],
                   it_gd=r["it_gd"], it_agd=r["it_agd"]) for r in ks.rows])
    _flush(nums, "kappa_sweep", t0)

    print("[slide10] GD step sweep (reused) ...", flush=True)
    plots.step_sweep_plot(E.gd_step_sweep())
    print("[slide12] Newton pure-vs-damped (reused) ...", flush=True)
    plots.newton_init_plot(E.newton_init_study())

    print("[T2.2] three regimes ...", flush=True)
    tr = E.three_regimes()
    plots.three_regimes_plot(tr)
    nums["three_regimes"] = dict(two_over_L=tr.two_over_L, two_over_lam=tr.two_over_lam,
                                 etas=tr.etas)
    _flush(nums, "three_regimes", t0)

    print("[T2.3] affine invariance (lambda=0) ...", flush=True)
    ai = E.affine_invariance()
    plots.affine_plot(ai)
    nums["affine"] = dict(kappa_std=ai.kappa_std, kappa_raw=ai.kappa_raw,
                          iters_newton_std=ai.iters_newton_std,
                          iters_newton_raw=ai.iters_newton_raw)
    _flush(nums, "affine", t0)

    print("[T1.4] SGD multiseed ...", flush=True)
    sm = E.sgd_multiseed()
    plots.sgd_multiseed_plot(sm)
    nums["sgd"] = dict(
        batches=[int(b) for b in sm.mean_gap],
        final_std={int(b): float(sm.std_gap[b][-1]) for b in sm.mean_gap},
        final_mean={int(b): float(sm.mean_gap[b][-1]) for b in sm.mean_gap})
    _flush(nums, "sgd_multiseed", t0)

    print("[T1.3] Armijo sweep + fixed-vs-backtracking ...", flush=True)
    asw = E.armijo_sweep()
    fb = E.fixed_vs_backtracking()
    nums["armijo_sweep"] = dict(t0=asw.t0, rows=asw.rows)
    nums["fixed_vs_backtracking"] = fb.rows
    _flush(nums, "armijo+fixed_vs_backtracking", t0)

    print("[T1.5] sklearn timing ...", flush=True)
    st = E.sklearn_timing()
    nums["sklearn_timing"] = dict(f_star=st.f_star, rows=st.rows)
    _flush(nums, "sklearn_timing", t0)

    print("[T2.4] breakeven ...", flush=True)
    be = E.breakeven_analysis()
    plots.breakeven_plot(be)
    nums["breakeven"] = dict(kappa_star=be.kappa_star, d_star=be.d_star,
                             n=be.n, eps=be.eps, k_newton=be.k_newton,
                             d_cube_ratio=be.d_cube_ratio, blas=be.blas)
    _flush(nums, "breakeven", t0)

    print("[T1.2] money plots (ridge benchmark) ...", flush=True)
    bench = run_benchmark(variant="ridge")
    plots.money_plots(bench)
    plots.convergence_curves(bench)              # keep the 2-panel too

    print("[T2.5] L1 benchmark with Coordinate Descent ...", flush=True)
    l1 = run_l1_benchmark(alpha=1e-3, max_iter=8000)
    plots.l1_convergence(l1)
    nums["l1"] = dict(f_star=l1.f_star, saga_f=l1.f_sklearn,
                      nnz_saga=l1.nnz_sklearn, d_reg=l1.d_reg,
                      methods={k: dict(iters=len(v.f_history),
                                       gap=float(v.f_history[-1] - l1.f_star),
                                       nnz=int(v.nnz[-1]) if v.nnz else -1,
                                       converged=bool(v.converged))
                               for k, v in l1.results.items()})

    _flush(nums, "l1", t0)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. "
          f"Numbers -> artifacts/rev1_numbers.json, figures -> artifacts/figures/")


if __name__ == "__main__":
    main()
