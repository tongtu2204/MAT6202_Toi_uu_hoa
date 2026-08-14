#!/usr/bin/env python3
"""Run the feature-engineering pipeline and save design matrices for the benchmark.

Usage:
    python run_pipeline.py                # defaults from churn_opt.config
    python run_pipeline.py --snapshots 2019-09-30            # single snapshot
    python run_pipeline.py --lam 1e-2 --no-poly

Outputs (in artifacts/):
    <variant>.npz        X_train, X_test, y_train, y_test
    <variant>.features.txt   ordered feature names
    diagnostics.txt          L, mu, kappa, rank per variant + churn rate

The two checks the spec says to put on the opening experiment slide are printed
here: the actual kappa of X (predicts convergence behaviour) and confirmation that
XᵀX is full rank (so Newton can run at all).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from churn_opt import FeatureConfig, conditioning, format_report, make_datasets
from churn_opt.config import OUT_DIR


def parse_args() -> FeatureConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", nargs="+", default=None,
                   help="snapshot dates T (YYYY-MM-DD); several are STACKED into one "
                        "design matrix, one row per (customer, snapshot)")
    p.add_argument("--obs-months", type=int, default=None, help="observation window K")
    p.add_argument("--out-months", type=int, default=None, help="outcome window H")
    p.add_argument("--lam", type=float, default=None, help="lambda for conditioning report")
    p.add_argument("--no-poly", action="store_true", help="skip the high-kappa poly variant")
    p.add_argument("--holdout", action="store_true",
                   help="keep a 25%% test split (only needed for out-of-sample AUC/F1); "
                        "default trains on the full eligible population")
    a = p.parse_args()

    cfg = FeatureConfig()
    if a.snapshots:
        cfg.snapshots = tuple(a.snapshots)
    if a.obs_months:
        cfg.windows.observation_months = a.obs_months
    if a.out_months:
        cfg.windows.outcome_months = a.out_months
    if a.lam is not None:
        cfg.lam = a.lam
    if a.no_poly:
        cfg.add_polynomial = False
    cfg.use_holdout = a.holdout
    return cfg


def main() -> None:
    cfg = parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"Snapshots: {list(cfg.snapshots)}  (K={cfg.windows.observation_months}mo, "
          f"H={cfg.windows.outcome_months}mo)")
    print("Building feature matrices (this reads the 16M-row activity file)...")
    datasets = make_datasets(cfg)
    print(f"  done in {time.time() - t0:.1f}s\n")

    report_lines = []
    for variant, ds in datasets.items():
        np.savez_compressed(
            OUT_DIR / f"{variant}.npz",
            X_train=ds.X_train, X_test=ds.X_test,
            y_train=ds.y_train, y_test=ds.y_test,
        )
        (OUT_DIR / f"{variant}.features.txt").write_text("\n".join(ds.feature_names))

        cond = conditioning(ds.X_train, cfg.lam)
        line = format_report(variant, cond)
        print(line)
        report_lines.append(line)

    # Population summary (same y across variants).
    y = next(iter(datasets.values())).y_train
    yte = next(iter(datasets.values())).y_test
    churn = (y.sum() + yte.sum()) / (len(y) + len(yte))
    summary = (
        f"\nsnapshots={list(cfg.snapshots)}  obs=(T-{cfg.windows.observation_months}mo, T]"
        f"  outcome=(T, T+{cfg.windows.outcome_months}mo]  holdout={cfg.use_holdout}\n"
        f"rows (customer x snapshot): train={len(y)} test={len(yte)}"
        f"  churn_rate={churn:.3f}"
    )
    print(summary)

    (OUT_DIR / "diagnostics.txt").write_text("\n".join(report_lines) + summary + "\n")
    print(f"\nArtifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
