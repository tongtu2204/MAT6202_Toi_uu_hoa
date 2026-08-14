#!/usr/bin/env python3
"""Run the adaptive-family study for section 5 of the deck.

AdaGrad / RMSprop / Adam / AdamW / AMSGrad, each given its own best step size from
a shared grid, against the GD / AGD / Newton baselines on identical footing.

    python run_adaptive.py

Writes artifacts/figures/adaptive_family_ridge.png (picked up automatically by
presentation_v2/main.tex via \\IfFileExists) and artifacts/adaptive_numbers.json
(the step-size sweep + the final table the slide quotes).
"""
from __future__ import annotations

import blas_threads  # must precede numpy: pins BLAS to 1 thread
import json
import time
from pathlib import Path

from optim import experiments as E
from optim import plots

ART = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    print("  BLAS threads:", blas_threads.verify())   # timings are meaningless if != 1
    t0 = time.perf_counter()
    print("[T3.1] adaptive family (step-size sweep + baselines) ...")
    af = E.adaptive_family(methods=("AdaGrad", "RMSprop", "Adam",
                                    "AdamW", "AMSGrad"))
    print("  figure:", plots.adaptive_family_plot(af))

    table = {}
    for name, r in af.runs.items():
        gap = float(r.f_history[-1] - af.f_star)
        table[name] = dict(
            eta=af.best_eta.get(name),
            iters=len(r.f_history) - 1,
            f_gap=gap,
            time_s=float(r.time_s[-1]),
            converged=bool(r.converged),
        )
    nums = dict(variant=af.variant, lam=af.lam, f_star=af.f_star,
                eta_grid=af.eta_grid, best_eta=af.best_eta,
                sweep=af.sweep, table=table)
    (ART / "adaptive_numbers.json").write_text(json.dumps(nums, indent=1))

    print(f"\n  f* = {af.f_star:.12f}")
    print(f"  {'method':10s} {'eta':>6s} {'iters':>7s} {'f-f*':>11s} {'time(s)':>8s}  conv")
    for name, row in table.items():
        eta = "-" if row["eta"] is None else f"{row['eta']:g}"
        print(f"  {name:10s} {eta:>6s} {row['iters']:7d} {row['f_gap']:11.3e} "
              f"{row['time_s']:8.3f}  {row['converged']}")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s -> artifacts/adaptive_numbers.json")


if __name__ == "__main__":
    main()
