#!/usr/bin/env python3
"""Chạy bốn thí nghiệm GD/AGD đã chốt trên ``data/ridge.npz``.

Quy trình:
1. Mỗi ứng viên tham số chạy đúng 500 vòng trên toàn bộ tập train.
2. Dò thô, sinh lưới tinh quanh ứng viên tốt nhất, vẫn 500 vòng/ứng viên.
3. Khóa một cấu hình cho từng biến thể và chạy lại từ w0 tới hội tụ.
4. So nội bộ GD, so nội bộ AGD, rồi mới so hai phương án thắng.
5. Chỉ sau khi khóa cấu hình mới đọc metric trên tập test.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ORIGINAL = REPO / "bai-lam-taitt" / "5_ma_nguon_tai_lap"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    AGD_STEP_COARSE,
    BACKTRACKING_C_COARSE,
    BACKTRACKING_FINE_POINTS,
    BACKTRACKING_MAX_TRIALS,
    BACKTRACKING_RHO_COARSE,
    BACKTRACKING_T0,
    CONVERGENCE_MAX_ITER,
    CONVERGENCE_TOL,
    GD_STEP_COARSE,
    LAMBDA,
    SEARCH_ITERATIONS,
    STEP_FINE_POINTS,
    TIMING_REPEATS,
)
from evaluation import classification_metrics  # noqa: E402
from gd_agd_runner import (  # noqa: E402
    OptimizationRun,
    constant_momentum,
    run_agd,
    run_gd_backtracking,
    run_gd_fixed,
)
from optim.objective import LogisticObjective  # noqa: E402
from optim.reference import newton_reference  # noqa: E402

OUT = HERE / "artifacts"


def load_ridge(path: Path):
    with np.load(path) as data:
        required = {"X_train", "y_train", "X_test", "y_test"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"ridge.npz thiếu khóa: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=float)
        y_train = np.asarray(data["y_train"], dtype=float)
        X_test = np.asarray(data["X_test"], dtype=float)
        y_test = np.asarray(data["y_test"], dtype=float)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train và X_test phải là ma trận 2 chiều")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Số đặc trưng train/test không khớp")
    for name, array in (("X_train", X_train), ("y_train", y_train),
                        ("X_test", X_test), ("y_test", y_test)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} chứa NaN/Inf")
    return X_train, y_train, X_test, y_test


def fine_linear_grid(grid, best: float, points: int, positive: bool = True):
    values = np.array(sorted({float(x) for x in grid}), dtype=float)
    index = int(np.argmin(np.abs(values - best)))
    if index == 0:
        lower = best / 2.0 if positive else best - (values[1] - best)
        upper = values[1]
    elif index == len(values) - 1:
        lower = values[-2]
        upper = best + (best - values[-2])
    else:
        lower, upper = values[index - 1], values[index + 1]
    if positive:
        lower = max(lower, np.finfo(float).eps)
    return tuple(float(x) for x in np.linspace(lower, upper, points))


def fine_log_grid(grid, best: float, points: int):
    values = np.array(sorted({float(x) for x in grid}), dtype=float)
    index = int(np.argmin(np.abs(values - best)))
    lower = best / 10.0 if index == 0 else values[index - 1]
    upper = best * 10.0 if index == len(values) - 1 else values[index + 1]
    upper = min(upper, 0.99)
    return tuple(float(x) for x in np.geomspace(lower, upper, points))


def run_summary(run: OptimizationRun, f_star: float, stage: str) -> dict:
    return {
        "stage": stage,
        "parameters": run.parameters,
        "finite": bool(run.finite),
        "iterations_run": int(run.iterations[-1]) if run.iterations else 0,
        "converged_within_500": run.converged_at is not None,
        "first_converged_at": run.converged_at,
        "final_objective": float(run.final_objective),
        "final_f_gap": max(float(run.final_objective - f_star), 0.0),
        "final_grad_norm": float(run.final_grad_norm),
        "gradient_evals": int(run.gradient_evals),
        "objective_evals": int(run.objective_evals),
        "backtracking_trials": int(run.backtracking_trials),
        "elapsed_s": float(run.elapsed_s),
        "mean_step": float(np.mean(run.steps)) if run.steps else None,
        "min_step": float(np.min(run.steps)) if run.steps else None,
        "max_step": float(np.max(run.steps)) if run.steps else None,
        "note": run.note,
    }


def candidate_key(row: dict):
    if not row["finite"]:
        return (2, float("inf"), float("inf"), float("inf"))
    if row["first_converged_at"] is not None:
        return (0, row["first_converged_at"], row["objective_evals"], row["final_grad_norm"])
    return (1, row["final_grad_norm"], row["final_f_gap"], row["objective_evals"])


def select_best(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["finite"]]
    if not eligible:
        raise RuntimeError("Không có ứng viên hữu hạn trong lưới")
    return min(eligible, key=candidate_key)


def search_1d(name: str, coarse_grid, runner, f_star: float):
    rows: list[dict] = []
    print(f"[{name}] lưới thô: {len(coarse_grid)} cấu hình x {SEARCH_ITERATIONS} vòng", flush=True)
    for index, step in enumerate(coarse_grid, start=1):
        run = runner(float(step), SEARCH_ITERATIONS, False)
        rows.append(run_summary(run, f_star, "coarse"))
        if index % 5 == 0 or index == len(coarse_grid):
            print(f"[{name}] đã chạy {index}/{len(coarse_grid)} cấu hình thô", flush=True)

    coarse_best = select_best(rows)
    best_step = float(coarse_best["parameters"]["step"])
    fine_grid = fine_linear_grid(coarse_grid, best_step, STEP_FINE_POINTS)
    existing = {round(float(x), 14) for x in coarse_grid}
    fine_only = [x for x in fine_grid if round(float(x), 14) not in existing]
    print(f"[{name}] lưới tinh quanh step={best_step:g}: {len(fine_only)} cấu hình", flush=True)
    for step in fine_only:
        run = runner(float(step), SEARCH_ITERATIONS, False)
        rows.append(run_summary(run, f_star, "fine"))
    selected = select_best(rows)
    return rows, coarse_best, selected


def search_backtracking(obj, w0, f_star: float):
    rows: list[dict] = []
    pairs = [(rho, c) for rho in BACKTRACKING_RHO_COARSE for c in BACKTRACKING_C_COARSE]
    print(f"[GD backtracking] lưới thô: {len(pairs)} cấu hình x {SEARCH_ITERATIONS} vòng", flush=True)
    for index, (rho, c) in enumerate(pairs, start=1):
        run = run_gd_backtracking(
            obj, w0, BACKTRACKING_T0, rho, c, SEARCH_ITERATIONS,
            CONVERGENCE_TOL, False, BACKTRACKING_MAX_TRIALS,
        )
        rows.append(run_summary(run, f_star, "coarse"))
        if index % 7 == 0 or index == len(pairs):
            print(f"[GD backtracking] đã chạy {index}/{len(pairs)} cấu hình thô", flush=True)

    coarse_best = select_best(rows)
    best_rho = float(coarse_best["parameters"]["rho"])
    best_c = float(coarse_best["parameters"]["c"])
    rho_fine = fine_linear_grid(
        BACKTRACKING_RHO_COARSE, best_rho, BACKTRACKING_FINE_POINTS
    )
    c_fine = fine_log_grid(BACKTRACKING_C_COARSE, best_c, BACKTRACKING_FINE_POINTS)
    existing = {
        (round(float(rho), 14), round(float(c), 14))
        for rho, c in pairs
    }
    fine_pairs = [
        (rho, c) for rho in rho_fine for c in c_fine
        if (round(rho, 14), round(c, 14)) not in existing
    ]
    print(
        f"[GD backtracking] lưới tinh quanh (rho={best_rho:g}, c={best_c:g}): "
        f"{len(fine_pairs)} cấu hình",
        flush=True,
    )
    for index, (rho, c) in enumerate(fine_pairs, start=1):
        run = run_gd_backtracking(
            obj, w0, BACKTRACKING_T0, rho, c, SEARCH_ITERATIONS,
            CONVERGENCE_TOL, False, BACKTRACKING_MAX_TRIALS,
        )
        rows.append(run_summary(run, f_star, "fine"))
        if index % 5 == 0 or index == len(fine_pairs):
            print(f"[GD backtracking] đã chạy {index}/{len(fine_pairs)} cấu hình tinh", flush=True)
    selected = select_best(rows)
    return rows, coarse_best, selected


def finalist_summary(run: OptimizationRun, f_star: float, median_time: float,
                     repeat_times: list[float]) -> dict:
    return {
        "method": run.method,
        "parameters": run.parameters,
        "converged": bool(run.converged),
        "converged_at": run.converged_at,
        "max_iter": CONVERGENCE_MAX_ITER,
        "tolerance": CONVERGENCE_TOL,
        "final_objective": float(run.final_objective),
        "final_f_gap": max(float(run.final_objective - f_star), 0.0),
        "final_grad_norm": float(run.final_grad_norm),
        "gradient_evals": int(run.gradient_evals),
        "objective_evals": int(run.objective_evals),
        "backtracking_trials": int(run.backtracking_trials),
        "median_time_s": float(median_time),
        "repeat_times_s": [float(x) for x in repeat_times],
        "mean_step": float(np.mean(run.steps)) if run.steps else None,
        "min_step": float(np.min(run.steps)) if run.steps else None,
        "max_step": float(np.max(run.steps)) if run.steps else None,
        "note": run.note,
        "trace": {
            "iteration": run.iterations,
            "objective": run.objectives,
            "grad_norm": run.grad_norms,
            "time_s": run.times_s,
        },
    }


def timed_final(name: str, runner, f_star: float):
    runner(3, False)  # warm-up nhỏ; không đưa vào kết quả
    runs = []
    print(f"[{name}] chạy lại từ w0 tới hội tụ ({TIMING_REPEATS} lần đo thời gian)", flush=True)
    for repeat in range(TIMING_REPEATS):
        run = runner(CONVERGENCE_MAX_ITER, True)
        runs.append(run)
        status = f"hội tụ ở vòng {run.converged_at}" if run.converged else "chưa hội tụ"
        print(f"[{name}] lần {repeat + 1}/{TIMING_REPEATS}: {status}, {run.elapsed_s:.3f}s", flush=True)
    times = [run.elapsed_s for run in runs]
    representative = runs[int(np.argsort(times)[len(times) // 2])]
    return representative, finalist_summary(
        representative, f_star, float(np.median(times)), times
    )


def comparison_winner(first_key: str, second_key: str, finalists: dict):
    first, second = finalists[first_key], finalists[second_key]

    def key(row):
        return (
            0 if row["converged"] else 1,
            row["median_time_s"] if row["converged"] else row["final_grad_norm"],
            row["converged_at"] if row["converged_at"] is not None else row["max_iter"],
            row["objective_evals"],
        )

    winner = first_key if key(first) <= key(second) else second_key
    return {
        "candidates": [first_key, second_key],
        "winner": winner,
        "criterion": "ưu tiên hội tụ; sau đó thời gian trung vị tới hội tụ, số vòng và số objective",
    }


def sklearn_solution(X_train, y_train):
    model = LogisticRegression(
        C=1.0 / (LAMBDA * len(y_train)),
        solver="lbfgs",
        max_iter=10_000,
        tol=1e-12,
        random_state=42,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    w = np.concatenate([model.coef_.ravel(), model.intercept_.ravel()])
    return w, float(elapsed), int(model.n_iter_[0])


def main():
    data_path = REPO / "data" / "ridge.npz"
    X_train, y_train, X_test, y_test = load_ridge(data_path)
    print(
        f"Dữ liệu: train={X_train.shape}, test={X_test.shape}; "
        "không chia lại và chưa dùng test trong bước dò",
        flush=True,
    )

    obj = LogisticObjective(X_train, y_train, lam=LAMBDA)
    w0 = np.zeros(obj.d)
    reference = newton_reference(obj, w0)
    L = float(np.linalg.eigvalsh(obj.hessian(w0))[-1])
    mu = float(reference.mu)
    beta = constant_momentum(L, mu)
    f_star = float(reference.f_star)
    print(
        f"Tham chiếu: f*={f_star:.12f}, L={L:.6f}, mu={mu:.6f}, beta={beta:.6f}",
        flush=True,
    )

    gd_fixed_runner = lambda step, n, stop: run_gd_fixed(  # noqa: E731
        obj, w0, step, n, CONVERGENCE_TOL, stop
    )
    gd_fixed_rows, gd_fixed_coarse, gd_fixed_selected = search_1d(
        "GD bước cố định", GD_STEP_COARSE, gd_fixed_runner, f_star
    )

    gd_bt_rows, gd_bt_coarse, gd_bt_selected = search_backtracking(obj, w0, f_star)

    agd_const_runner = lambda step, n, stop: run_agd(  # noqa: E731
        obj, w0, step, n, CONVERGENCE_TOL, stop, "constant", beta
    )
    agd_const_rows, agd_const_coarse, agd_const_selected = search_1d(
        "AGD momentum cố định", AGD_STEP_COARSE, agd_const_runner, f_star
    )

    agd_dynamic_runner = lambda step, n, stop: run_agd(  # noqa: E731
        obj, w0, step, n, CONVERGENCE_TOL, stop, "dynamic", None
    )
    agd_dynamic_rows, agd_dynamic_coarse, agd_dynamic_selected = search_1d(
        "AGD momentum động", AGD_STEP_COARSE, agd_dynamic_runner, f_star
    )

    selected_gd_step = float(gd_fixed_selected["parameters"]["step"])
    selected_rho = float(gd_bt_selected["parameters"]["rho"])
    selected_c = float(gd_bt_selected["parameters"]["c"])
    selected_agd_const_step = float(agd_const_selected["parameters"]["step"])
    selected_agd_dynamic_step = float(agd_dynamic_selected["parameters"]["step"])

    final_runs: dict[str, OptimizationRun] = {}
    finalists: dict[str, dict] = {}

    final_runs["gd_fixed"], finalists["gd_fixed"] = timed_final(
        "GD bước cố định",
        lambda n, stop: run_gd_fixed(
            obj, w0, selected_gd_step, n, CONVERGENCE_TOL, stop
        ),
        f_star,
    )
    final_runs["gd_backtracking"], finalists["gd_backtracking"] = timed_final(
        "GD backtracking",
        lambda n, stop: run_gd_backtracking(
            obj, w0, BACKTRACKING_T0, selected_rho, selected_c, n,
            CONVERGENCE_TOL, stop, BACKTRACKING_MAX_TRIALS,
        ),
        f_star,
    )
    final_runs["agd_constant"], finalists["agd_constant"] = timed_final(
        "AGD momentum cố định",
        lambda n, stop: run_agd(
            obj, w0, selected_agd_const_step, n, CONVERGENCE_TOL,
            stop, "constant", beta,
        ),
        f_star,
    )
    final_runs["agd_dynamic"], finalists["agd_dynamic"] = timed_final(
        "AGD momentum động",
        lambda n, stop: run_agd(
            obj, w0, selected_agd_dynamic_step, n, CONVERGENCE_TOL,
            stop, "dynamic", None,
        ),
        f_star,
    )

    gd_internal = comparison_winner("gd_fixed", "gd_backtracking", finalists)
    agd_internal = comparison_winner("agd_constant", "agd_dynamic", finalists)
    overall = comparison_winner(gd_internal["winner"], agd_internal["winner"], finalists)

    # Test chỉ được chạm tới sau khi toàn bộ tham số và người thắng đã khóa.
    model_metrics = {
        "untrained_w0": {
            "label": "Chưa tối ưu (w0=0)",
            "train": classification_metrics(X_train, y_train, w0),
            "test": classification_metrics(X_test, y_test, w0),
            "objective": float(obj.value(w0)),
        }
    }
    for key, run in final_runs.items():
        model_metrics[key] = {
            "label": finalists[key]["method"],
            "train": classification_metrics(X_train, y_train, run.w),
            "test": classification_metrics(X_test, y_test, run.w),
            "objective": float(obj.value(run.w)),
        }

    sklearn_w, sklearn_time, sklearn_iterations = sklearn_solution(X_train, y_train)
    model_metrics["sklearn_reference"] = {
        "label": "LogisticRegression (scikit-learn)",
        "train": classification_metrics(X_train, y_train, sklearn_w),
        "test": classification_metrics(X_test, y_test, sklearn_w),
        "objective": float(obj.value(sklearn_w)),
        "time_s": sklearn_time,
        "iterations": sklearn_iterations,
    }

    result = {
        "protocol": {
            "search_iterations_per_candidate": SEARCH_ITERATIONS,
            "search_stages": ["coarse", "fine"],
            "selection_rule": (
                "ưu tiên đạt ||grad||<=tol sớm trong 500 vòng; nếu chưa đạt, "
                "chọn gradient norm cuối nhỏ nhất rồi objective gap"
            ),
            "final_run": "khởi tạo lại w0 và chạy tới hội tụ",
            "convergence_tolerance": CONVERGENCE_TOL,
            "safety_max_iterations": CONVERGENCE_MAX_ITER,
            "timing_repeats": TIMING_REPEATS,
            "test_usage": "không dùng khi dò; chỉ đánh giá sau khi khóa tham số",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "data": {
            "path": "data/ridge.npz",
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "train_positive_rate": float(np.mean(y_train)),
            "test_positive_rate": float(np.mean(y_test)),
            "split_again": False,
        },
        "objective": {
            "lambda": LAMBDA,
            "f_star": f_star,
            "reference_grad_norm": float(reference.grad_norm),
            "reference_bound": float(reference.bound),
            "L_at_w0": L,
            "mu_at_reference": mu,
            "constant_momentum_beta": beta,
        },
        "searches": {
            "gd_fixed": {
                "coarse_best": gd_fixed_coarse,
                "selected": gd_fixed_selected,
                "candidates": gd_fixed_rows,
            },
            "gd_backtracking": {
                "coarse_best": gd_bt_coarse,
                "selected": gd_bt_selected,
                "candidates": gd_bt_rows,
            },
            "agd_constant": {
                "coarse_best": agd_const_coarse,
                "selected": agd_const_selected,
                "candidates": agd_const_rows,
            },
            "agd_dynamic": {
                "coarse_best": agd_dynamic_coarse,
                "selected": agd_dynamic_selected,
                "candidates": agd_dynamic_rows,
            },
        },
        "finalists": finalists,
        "comparisons": {
            "gd_internal": gd_internal,
            "agd_internal": agd_internal,
            "gd_vs_agd": overall,
        },
        "model_metrics": model_metrics,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã ghi {result_path}", flush=True)

    from make_artifacts import generate_artifacts
    generate_artifacts(result_path)


if __name__ == "__main__":
    main()
