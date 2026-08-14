#!/usr/bin/env python3
"""Re-render artifacts/figures/three_regimes_ridge.png from the cached curves.

`run_stage.py three-regimes` costs ~12.5 min (9 step sizes x 4000 GD iterations at
n=75,026, d=415) but the figure is pure styling, so the stage also writes
artifacts/three_regimes_ridge.npz. Use this to iterate on colours/legend without
paying for the sweep again:

    python replot_three_regimes.py

Re-run the stage itself only when the eta grid or the objective changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from optim import plots
from optim.experiments import ThreeRegimes

ART = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    f = ART / "three_regimes_ridge.npz"
    if not f.exists():
        raise SystemExit(f"{f} chưa có — chạy `python run_stage.py three-regimes` trước.")
    d = np.load(f, allow_pickle=False)
    etas = [float(e) for e in d["etas"]]
    tr = ThreeRegimes(variant=str(d["variant"]), lam=float(d["two_over_lam"]) and 2.0 / float(d["two_over_lam"]),
                      L=2.0 / float(d["two_over_L"]),
                      two_over_L=float(d["two_over_L"]),
                      two_over_lam=float(d["two_over_lam"]),
                      etas=etas,
                      histories={e: d[f"h_{i}"] for i, e in enumerate(etas)},
                      f_star=float(d["f_star"]))
    print("hình:", plots.three_regimes_plot(tr))


if __name__ == "__main__":
    main()
