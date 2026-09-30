#!/usr/bin/env python3
"""Run the revised GD/AGD search and held-out before/after comparison.

The original source and presentation are read-only inputs.  Every new artifact
is written below ``tu_update/artifacts``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ORIGINAL = REPO / "bai-lam-taitt" / "5_ma_nguon_tai_lap"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    AGD_BETA_GRID,
    AGD_STEP_GRID,
    CHECKPOINTS,
    CONVERGENCE_MAX_ITER,
    GD_STEP_GRID,
    LAMBDA,
    RANDOM_STATE,
    SELECTION_ITER,
    STABILITY_ITER,
    STABILITY_TOP_N,
    TEST_SIZE,
    TIMING_REPEATS,
    TOL,
    VALID_SIZE,
)
from evaluation import (  # noqa: E402
    classification_metrics,
    split_indices,
    train_valid_indices,
)
from gd_agd_runner import beta_theory, run_agd, run_gd  # noqa: E402
from optim.objective import LogisticObjective  # noqa: E402
from optim.reference import newton_reference  # noqa: E402

OUT = HERE / "artifacts"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gd", "agd", "all"), default="all")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--groups-path", type=Path, default=None,
                        help="optional .npy customer ids for group-aware validation")
    parser.add_argument("--quick", action="store_true",
                        help="reduced smoke grid; does not replace the full result")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def locate_data(explicit=None):
    candidates = [
        explicit,
        ORIGINAL / "artifacts" / "ridge.npz",
        REPO / "data" / "ridge.npz",
    ]
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path)
    raise FileNotFoundError("ridge.npz not found; pass --data-path")


def load_data(path):
    data = np.load(path)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float)
    X_test = (np.asarray(data["X_test"], dtype=float)
              if "X_test" in data else np.empty((0, X_train.shape[1])))
    y_test = (np.asarray(data["y_test"], dtype=float)
              if "y_test" in data else np.empty(0))
    return X_train, y_train, X_test, y_test


def split_data(X, y, X_external_test, y_external_test, groups=None):
    if len(y_external_test):
        train, valid, mode = train_valid_indices(
            y, valid_size=VALID_SIZE, random_state=RANDOM_STATE, groups=groups
        )
        return (X[train], y[train], X[valid], y[valid],
                X_external_test, y_external_test, "provided-test+" + mode)

    train, valid, test, mode = split_indices(
        y, VALID_SIZE, TEST_SIZE, RANDOM_STATE, groups
    )
    return X[train], y[train], X[valid], y[valid], X[test], y[test], mode


def search_row(run, f_star, selection_iter, X_valid, y_valid):
    snap = run.snapshots.get(selection_iter)
    if snap is None:
        return {
            "eta": run.eta,
            "beta": run.beta,
            "iteration": selection_iter,
            "finite": False,
        }
    raw_gap = float(snap.objective - f_star)
    return {
        "eta": run.eta,
        "beta": run.beta,
        "iteration": selection_iter,
        "objective": snap.objective,
        "f_gap": max(raw_gap, 0.0),
        "raw_f_gap": raw_gap,
        "grad_norm": snap.grad_norm,
        "finite": bool(run.finite),
        "validation": classification_metrics(X_valid, y_valid, snap.w),
    }


def select_best(rows):
    eligible = [
        row for row in rows
        if row.get("finite") and np.isfinite(row.get("f_gap", np.inf))
    ]
    if not eligible:
        raise RuntimeError("no finite candidate in the search grid")
    return min(eligible, key=lambda row: (row["f_gap"], row["grad_norm"]))


def stability_screen(method, rows, obj, w0, f_star, stability_iter, top_n):
    """Reject transient winners that look good only at the selection checkpoint."""
    eligible = [row for row in rows if row.get("finite")]
    ranked = sorted(eligible, key=lambda row: (row["f_gap"], row["grad_norm"]))
    for row in ranked[:top_n]:
        if method == "gd":
            run = run_gd(obj, w0, row["eta"], (stability_iter,), 0.0)
        else:
            run = run_agd(obj, w0, row["eta"], row["beta"],
                          (stability_iter,), 0.0)
        snap = run.snapshots.get(stability_iter)
        if snap is None:
            row.update(stability_checked=True, stable=False)
            continue
        long_gap = max(float(snap.objective - f_star), 0.0)
        stable = bool(
            run.finite
            and long_gap <= max(1.05 * row["f_gap"], 1e-12)
            and snap.grad_norm <= max(1.05 * row["grad_norm"], 1e-10)
        )
        row.update(
            stability_checked=True,
            stability_iteration=stability_iter,
            stability_f_gap=long_gap,
            stability_grad_norm=snap.grad_norm,
            stable=stable,
        )
    stable_rows = [row for row in ranked[:top_n] if row.get("stable")]
    if not stable_rows:
        raise RuntimeError("no stable candidate among the top search results")
    return min(stable_rows, key=lambda row: (row["f_gap"], row["grad_norm"]))


def timed_trace(method, obj, w0, eta, beta, checkpoints):
    wanted = set(checkpoints)
    final = max(wanted)
    times = {}
    w = np.asarray(w0, dtype=float).copy()
    w_prev = w.copy()
    start = time.perf_counter()
    for k in range(1, final + 1):
        if method == "gd":
            w = w - eta * obj.grad(w)
        else:
            y = w + beta * (w - w_prev)
            w_prev, w = w, y - eta * obj.grad(y)
        if k in wanted:
            times[k] = time.perf_counter() - start
    return times


def median_times(method, obj, w0, eta, beta, checkpoints, repeats):
    # One short warm-up prevents import/BLAS initialization from entering timing.
    timed_trace(method, obj, w0, eta, beta, (3,))
    traces = [
        timed_trace(method, obj, w0, eta, beta, checkpoints)
        for _ in range(repeats)
    ]
    return {
        int(k): float(np.median([trace[k] for trace in traces]))
        for k in checkpoints
    }


def convergence_summary(method, obj, w0, eta, beta, max_iter):
    if method == "gd":
        run = run_gd(obj, w0, eta, (max_iter,), TOL, stop_on_convergence=True)
    else:
        run = run_agd(obj, w0, eta, beta, (max_iter,), TOL,
                      stop_on_convergence=True)
    return {
        "tolerance": TOL,
        "max_iter": max_iter,
        "converged": run.converged_at is not None,
        "iteration": run.converged_at,
        "time_s": run.convergence_time_s,
    }


def comparison_rows(baseline, tuned, f_star, checkpoints,
                    X_valid, y_valid, X_test, y_test,
                    baseline_times, tuned_times):
    rows = []
    for k in checkpoints:
        base = baseline.snapshots[k]
        tune = tuned.snapshots[k]
        rows.append({
            "iteration": k,
            "baseline": {
                "objective": base.objective,
                "f_gap": max(float(base.objective - f_star), 0.0),
                "grad_norm": base.grad_norm,
                "gradient_evaluations": k,
                "median_time_s": baseline_times[k],
                "validation": classification_metrics(X_valid, y_valid, base.w),
                "test": classification_metrics(X_test, y_test, base.w),
            },
            "tuned": {
                "objective": tune.objective,
                "f_gap": max(float(tune.objective - f_star), 0.0),
                "grad_norm": tune.grad_norm,
                "gradient_evaluations": k,
                "median_time_s": tuned_times[k],
                "validation": classification_metrics(X_valid, y_valid, tune.w),
                "test": classification_metrics(X_test, y_test, tune.w),
            },
        })
    return rows


def run_method(method, obj, w0, f_star, L, mu, checkpoints,
               selection_iter, X_valid, y_valid, X_test, y_test,
               gd_grid, agd_steps, agd_betas, timing_repeats,
               convergence_max, stability_iter, stability_top_n):
    if method == "gd":
        print(f"[GD] searching {len(gd_grid)} step sizes at k={selection_iter}", flush=True)
        candidates = [run_gd(obj, w0, eta, (selection_iter,), 0.0)
                      for eta in gd_grid]
        search = [search_row(run, f_star, selection_iter, X_valid, y_valid)
                  for run in candidates]
        chosen = stability_screen(
            method, search, obj, w0, f_star, stability_iter, stability_top_n
        )
        baseline_eta, baseline_beta = 1.0 / L, None
        tuned_eta, tuned_beta = chosen["eta"], None
        baseline = run_gd(obj, w0, baseline_eta, checkpoints, TOL)
        tuned = run_gd(obj, w0, tuned_eta, checkpoints, TOL)
    else:
        total = len(agd_steps) * len(agd_betas)
        print(f"[AGD] searching {total} (step, beta) pairs at k={selection_iter}",
              flush=True)
        candidates = [
            run_agd(obj, w0, eta, beta, (selection_iter,), 0.0)
            for eta in agd_steps for beta in agd_betas
        ]
        search = [search_row(run, f_star, selection_iter, X_valid, y_valid)
                  for run in candidates]
        chosen = stability_screen(
            method, search, obj, w0, f_star, stability_iter, stability_top_n
        )
        baseline_eta, baseline_beta = 1.0 / L, beta_theory(L, mu)
        tuned_eta, tuned_beta = chosen["eta"], chosen["beta"]
        baseline = run_agd(obj, w0, baseline_eta, baseline_beta, checkpoints, TOL)
        tuned = run_agd(obj, w0, tuned_eta, tuned_beta, checkpoints, TOL)

    print(f"[{method.upper()}] selected eta={tuned_eta:g}"
          + ("" if tuned_beta is None else f", beta={tuned_beta:g}"), flush=True)
    base_times = median_times(method, obj, w0, baseline_eta, baseline_beta,
                              checkpoints, timing_repeats)
    tune_times = median_times(method, obj, w0, tuned_eta, tuned_beta,
                              checkpoints, timing_repeats)

    return {
        "baseline_parameters": {
            "label": "curvature-based, no hyperparameter search",
            "eta": baseline_eta,
            "beta": baseline_beta,
        },
        "tuned_parameters": {
            "label": "best training objective gap in the searched grid",
            "eta": tuned_eta,
            "beta": tuned_beta,
        },
        "search": search,
        "selected_search_row": chosen,
        "comparison": comparison_rows(
            baseline, tuned, f_star, checkpoints,
            X_valid, y_valid, X_test, y_test, base_times, tune_times,
        ),
        "convergence": {
            "baseline": convergence_summary(
                method, obj, w0, baseline_eta, baseline_beta, convergence_max
            ),
            "tuned": convergence_summary(
                method, obj, w0, tuned_eta, tuned_beta, convergence_max
            ),
        },
    }


def main():
    args = parse_args()
    data_path = locate_data(args.data_path)
    X, y, X_external_test, y_external_test = load_data(data_path)
    groups = np.load(args.groups_path) if args.groups_path else None
    (X_train, y_train, X_valid, y_valid,
     X_test, y_test, split_mode) = split_data(
        X, y, X_external_test, y_external_test, groups
    )

    obj = LogisticObjective(X_train, y_train, lam=LAMBDA)
    w0 = np.zeros(obj.d)
    reference = newton_reference(obj, w0)
    # Exact curvature at w0, including the unregularized intercept convention.
    L = float(np.linalg.eigvalsh(obj.hessian(w0))[-1])
    mu = float(reference.mu)

    if args.quick:
        checkpoints = (50, 150)
        selection_iter = 150
        gd_grid = GD_STEP_GRID[::4]
        agd_steps = AGD_STEP_GRID[::4]
        agd_betas = AGD_BETA_GRID[::3]
        timing_repeats = 1
        convergence_max = 500
        stability_iter = 500
        stability_top_n = 3
    else:
        checkpoints = CHECKPOINTS
        selection_iter = SELECTION_ITER
        gd_grid = GD_STEP_GRID
        agd_steps = AGD_STEP_GRID
        agd_betas = AGD_BETA_GRID
        timing_repeats = TIMING_REPEATS
        convergence_max = CONVERGENCE_MAX_ITER
        stability_iter = STABILITY_ITER
        stability_top_n = STABILITY_TOP_N

    output = {
        "experiment": {
            "quick": args.quick,
            "selection_criterion": (
                f"minimum training objective gap at iteration {selection_iter} "
                f"among top candidates stable through iteration {stability_iter}"
            ),
            "selection_iteration": selection_iter,
            "stability_iteration": stability_iter,
            "stability_top_n": stability_top_n,
            "test_usage": "held out from search; evaluated after selection",
            "checkpoints": list(checkpoints),
            "timing_repeats": timing_repeats,
            "random_state": RANDOM_STATE,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "data": {
            "path": str(data_path.relative_to(REPO)
                        if data_path.is_relative_to(REPO) else data_path),
            "split_mode": split_mode,
            "sizes": {
                "train": len(y_train),
                "validation": len(y_valid),
                "test": len(y_test),
                "features": X_train.shape[1],
            },
            "positive_rates": {
                "train": float(np.mean(y_train)),
                "validation": float(np.mean(y_valid)),
                "test": float(np.mean(y_test)),
            },
        },
        "objective": {
            "lambda": LAMBDA,
            "L_at_w0": L,
            "mu_at_reference": mu,
            "f_star": reference.f_star,
            "reference_grad_norm": reference.grad_norm,
            "reference_bound": reference.bound,
            "reference_converged": reference.converged,
        },
    }

    if args.method in ("gd", "all"):
        output["gd"] = run_method(
            "gd", obj, w0, reference.f_star, L, mu, checkpoints,
            selection_iter, X_valid, y_valid, X_test, y_test,
            gd_grid, agd_steps, agd_betas, timing_repeats, convergence_max,
            stability_iter, stability_top_n,
        )
    if args.method in ("agd", "all"):
        output["agd"] = run_method(
            "agd", obj, w0, reference.f_star, L, mu, checkpoints,
            selection_iter, X_valid, y_valid, X_test, y_test,
            gd_grid, agd_steps, agd_betas, timing_repeats, convergence_max,
            stability_iter, stability_top_n,
        )

    OUT.mkdir(exist_ok=True)
    filename = "gd_agd_quick.json" if args.quick else "gd_agd_results.json"
    result_path = OUT / filename
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {result_path}", flush=True)

    if not args.skip_plots:
        from make_artifacts import generate_artifacts
        generate_artifacts(result_path)


if __name__ == "__main__":
    main()
