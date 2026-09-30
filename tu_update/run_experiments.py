#!/usr/bin/env python3
"""Extended GD/AGD search plus before/after predictive evaluation.

This driver is intentionally separate from the original project. It writes only
to tu_update/artifacts/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ORIGINAL = REPO / "bai-lam-taitt" / "5_ma_nguon_tai_lap"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    AGD_BETA_GRID,
    AGD_STEP_GRID,
    AGD_TOP_STEPS_FOR_BETA_SEARCH,
    CHECKPOINTS,
    GD_STEP_GRID,
    LAMBDA,
    RANDOM_STATE,
    TEST_SIZE,
    TOL,
    VALID_SIZE,
)
from evaluation import (  # noqa: E402
    classification_metrics,
    split_indices,
    train_valid_indices,
)
from gd_agd_runner import (  # noqa: E402
    beta_from_eta,
    beta_theory,
    run_agd,
    run_gd,
)
from churn_opt.diagnostics import conditioning  # noqa: E402
from optim.objective import LogisticObjective  # noqa: E402
from optim.reference import newton_reference  # noqa: E402

OUT = HERE / "artifacts"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("gd", "agd", "all"), default="all")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--groups-path", type=Path, default=None,
                   help="optional .npy customer ids for group-aware splitting")
    p.add_argument("--quick", action="store_true",
                   help="small smoke run: short grids and checkpoints up to 150")
    return p.parse_args()


def locate_data(explicit=None):
    candidates = [
        explicit,
        ORIGINAL / "artifacts" / "ridge.npz",
        REPO / "data" / "ridge.npz",
    ]
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path)
    raise FileNotFoundError("Không tìm thấy ridge.npz; truyền --data-path")


def load_data(path):
    data = np.load(path)
    Xtr, ytr = data["X_train"], data["y_train"].astype(float)
    Xte = data["X_test"] if "X_test" in data else np.empty((0, Xtr.shape[1]))
    yte = data["y_test"].astype(float) if "y_test" in data else np.empty(0)
    return Xtr, ytr, Xte, yte


def row(run, f_star, checkpoint, X_valid, y_valid):
    snap = run.snapshots.get(checkpoint)
    if snap is None:
        return {"iteration": checkpoint, "finite": False}
    return {
        "iteration": checkpoint,
        "eta": run.eta,
        "beta": run.beta,
        "objective": snap.objective,
        "f_gap": float(snap.objective - f_star),
        "grad_norm": snap.grad_norm,
        "time_s": snap.time_s,
        "finite": run.finite,
        "validation": classification_metrics(X_valid, y_valid, snap.w),
    }


def best_at(rows, checkpoint):
    eligible = [r for r in rows if r.get("iteration") == checkpoint
                and r.get("finite") and np.isfinite(r.get("f_gap", np.inf))]
    if not eligible:
        raise RuntimeError(f"Không có cấu hình hữu hạn tại checkpoint {checkpoint}")
    return min(eligible, key=lambda r: r["f_gap"])


def main():
    args = parse_args()
    data_path = locate_data(args.data_path)
    X, y, X_external_test, y_external_test = load_data(data_path)
    groups = np.load(args.groups_path) if args.groups_path else None

    # Nếu artifact đã có test, giữ nguyên test đó; chỉ tách validation từ train.
    if len(y_external_test):
        train, valid, split_mode = train_valid_indices(
            y, valid_size=VALID_SIZE,
            random_state=RANDOM_STATE, groups=groups)
        X_train, y_train = X[train], y[train]
        X_valid, y_valid = X[valid], y[valid]
        X_test, y_test = X_external_test, y_external_test
        split_mode = "provided-test+" + split_mode
    else:
        train, valid, test, split_mode = split_indices(
            y, VALID_SIZE, TEST_SIZE, RANDOM_STATE, groups)
        X_train, y_train = X[train], y[train]
        X_valid, y_valid = X[valid], y[valid]
        X_test, y_test = X[test], y[test]

    obj = LogisticObjective(X_train, y_train, lam=LAMBDA)
    w0 = np.zeros(obj.d)
    cond = conditioning(obj.X, LAMBDA)
    reference = newton_reference(obj, w0)
    L, mu = float(cond["L"]), float(reference.mu)

    checkpoints = (50, 150) if args.quick else CHECKPOINTS
    gd_grid = GD_STEP_GRID[::4] if args.quick else GD_STEP_GRID
    agd_grid = AGD_STEP_GRID[::4] if args.quick else AGD_STEP_GRID
    beta_grid = AGD_BETA_GRID[::3] if args.quick else AGD_BETA_GRID
    target = checkpoints[-1]
    output = {
        "data_path": str(data_path),
        "split_mode": split_mode,
        "sizes": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test)},
        "lambda": LAMBDA,
        "L_global": L,
        "mu_at_reference": mu,
        "f_star": reference.f_star,
        "checkpoints": list(checkpoints),
    }

    if args.method in ("gd", "all"):
        baseline = run_gd(obj, w0, 1.0 / L, checkpoints, TOL)
        runs = [run_gd(obj, w0, eta, checkpoints, TOL) for eta in gd_grid]
        rows = [row(r, reference.f_star, k, X_valid, y_valid)
                for r in runs for k in checkpoints]
        chosen = best_at(rows, target)
        tuned = next(r for r in runs if r.eta == chosen["eta"])
        output["gd"] = {
            "baseline": row(baseline, reference.f_star, target, X_valid, y_valid),
            "search": rows,
            "chosen": chosen,
            "test_before": classification_metrics(X_test, y_test,
                                                   baseline.snapshots[target].w),
            "test_after": classification_metrics(X_test, y_test,
                                                  tuned.snapshots[target].w),
        }

    if args.method in ("agd", "all"):
        b_theory = beta_theory(L, mu)
        baseline = run_agd(obj, w0, 1.0 / L, b_theory, checkpoints, TOL)
        paired = [run_agd(obj, w0, eta, beta_from_eta(eta, mu), checkpoints, TOL)
                  for eta in agd_grid]
        paired_rows = [row(r, reference.f_star, k, X_valid, y_valid)
                       for r in paired for k in checkpoints]
        ranked = sorted(
            [r for r in paired_rows if r["iteration"] == target and r.get("finite")],
            key=lambda r: r["f_gap"],
        )[:AGD_TOP_STEPS_FOR_BETA_SEARCH]
        top_steps = sorted({r["eta"] for r in ranked})
        refined = [run_agd(obj, w0, eta, beta, checkpoints, TOL)
                   for eta in top_steps for beta in beta_grid]
        refined_rows = [row(r, reference.f_star, k, X_valid, y_valid)
                        for r in refined for k in checkpoints]
        chosen = best_at(refined_rows, target)
        tuned = next(r for r in refined
                     if r.eta == chosen["eta"] and r.beta == chosen["beta"])
        output["agd"] = {
            "baseline": row(baseline, reference.f_star, target, X_valid, y_valid),
            "paired_search": paired_rows,
            "refined_search": refined_rows,
            "chosen": chosen,
            "test_before": classification_metrics(X_test, y_test,
                                                   baseline.snapshots[target].w),
            "test_after": classification_metrics(X_test, y_test,
                                                  tuned.snapshots[target].w),
        }

    OUT.mkdir(exist_ok=True)
    result_path = OUT / ("gd_agd_quick.json" if args.quick else "gd_agd_results.json")
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Đã ghi {result_path}")


if __name__ == "__main__":
    main()
