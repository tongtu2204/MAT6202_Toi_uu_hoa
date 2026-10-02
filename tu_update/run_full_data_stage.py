"""Run and checkpoint ONE full-data method/stage, never the entire sweep."""
from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import ast
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ORIGINAL = REPO / "bai-lam-taitt" / "5_ma_nguon_tai_lap"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(HERE))

from config import (AGD_STEP_COARSE, BACKTRACKING_C_COARSE,
                    BACKTRACKING_FINE_POINTS, BACKTRACKING_MAX_TRIALS,
                    BACKTRACKING_RHO_COARSE, BACKTRACKING_T0,
                    CONVERGENCE_MAX_ITER, CONVERGENCE_TOL, GD_STEP_COARSE,
                    LAMBDA, SEARCH_ITERATIONS, STEP_FINE_POINTS, TIMING_REPEATS)
from evaluation import classification_metrics
from gd_agd_runner import constant_momentum, run_agd, run_gd_backtracking, run_gd_fixed
from optim.objective import LogisticObjective
from optim.reference import newton_reference
from run_experiments import fine_linear_grid, fine_log_grid, run_summary, select_best
from run_full_data_gd_coarse import DATA_PATH, EXPECTED_SHA256

OUT = HERE / "artifacts" / "full_data"
LABELS = {"gd_fixed": "GD bước cố định", "gd_backtracking": "GD backtracking",
          "agd_constant": "AGD momentum hằng", "agd_dynamic": "AGD momentum biến thiên"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_hash():
    digest = hashlib.sha256()
    # Rendering changes must not invalidate already-computed optimization runs.
    nodes = ast.parse(Path(__file__).read_text(encoding="utf-8")).body
    computation = [node for node in nodes if isinstance(node, ast.FunctionDef)
                   and node.name not in {"source_hash", "export_stage"}]
    digest.update(ast.dump(ast.Module(body=computation, type_ignores=[])).encode())
    for path in (HERE / "config.py", HERE / "gd_agd_runner.py",
                 HERE / "run_experiments.py", ORIGINAL / "optim/objective.py",
                 ORIGINAL / "optim/reference.py", ORIGINAL / "optim/optimizers/newton.py",
                 ORIGINAL / "optim/linesearch.py", HERE / "evaluation.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_input():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Tai's full dataset: {DATA_PATH}")
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Full ridge.npz does not match Tai's MANIFEST SHA-256")
    with np.load(DATA_PATH) as data:
        X, y = np.asarray(data["X_train"], float), np.asarray(data["y_train"], float)
        test_shape = list(data["X_test"].shape)
    if X.shape != (75026, 415) or y.shape != (75026,):
        raise ValueError(f"Unexpected full-data shapes: {X.shape}, {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid features or binary labels")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("All BLAS pools must use one thread")
    metadata = {"path": str(DATA_PATH.relative_to(REPO)), "sha256": digest,
                "train_shape": list(X.shape), "test_shape": test_shape,
                "positive_rate": float(y.mean()), "split_again": False,
                "model_evaluation": "in-sample; original X_test is empty"}
    return X, y, metadata


def reference_for(obj, fingerprint):
    path = OUT / "reference.json"
    if path.exists():
        reference = json.loads(path.read_text(encoding="utf-8"))
        if reference["fingerprint"] != fingerprint:
            raise ValueError("Reference source/data mismatch; archive earlier results before rerunning")
        w = np.asarray(reference["w_star"], float)
        if abs(obj.value(w) - reference["f_star"]) > 1e-12 or np.linalg.norm(obj.grad(w)) > 1e-8:
            raise ValueError("Cached Newton reference failed validation")
        return reference
    print("Computing full-data Newton reference...", flush=True)
    result = newton_reference(obj, np.zeros(obj.d))
    if result.grad_norm > 1e-8:
        raise RuntimeError("Newton reference has not converged")
    L = float(np.linalg.eigvalsh(obj.hessian(np.zeros(obj.d)))[-1])
    reference = {"fingerprint": fingerprint, "f_star": float(result.f_star),
                 "grad_norm": float(result.grad_norm), "mu_local": float(result.mu),
                 "L_at_w0": L, "beta": constant_momentum(L, result.mu),
                 "w_star": result.w_star.tolist()}
    write_json(path, reference)
    return reference


def previous_stage(method, stage, fingerprint):
    path = OUT / method / stage / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Run {method}/{stage} first")
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous["status"] != "complete" or previous["fingerprint"] != fingerprint:
        raise ValueError(f"Incomplete or incompatible prerequisite: {path}")
    return previous


def parameters_for(method, stage, fingerprint, beta):
    if stage == "final":
        return [previous_stage(method, "fine", fingerprint)["selected"]["parameters"]] * TIMING_REPEATS
    grid = GD_STEP_COARSE if method == "gd_fixed" else AGD_STEP_COARSE
    if method == "gd_backtracking":
        rho, c = BACKTRACKING_RHO_COARSE, BACKTRACKING_C_COARSE
        if stage == "fine":
            best = previous_stage(method, "coarse", fingerprint)["selected"]["parameters"]
            rho = fine_linear_grid(rho, best["rho"], BACKTRACKING_FINE_POINTS)
            if any(not 0 < value < 1 for value in rho):
                # Refine inside the valid coarse interval when extrapolation
                # crosses rho=1; retain three points and the coarse winner.
                lower = max(value for value in BACKTRACKING_RHO_COARSE
                            if value < best["rho"])
                rho = tuple(float(value) for value in
                            np.linspace(lower, best["rho"], BACKTRACKING_FINE_POINTS))
            c = fine_log_grid(c, best["c"], BACKTRACKING_FINE_POINTS)
        # Boundary winner may expand the linear grid beyond Armijo's domain.
        rho = tuple(x for x in rho if 0 < x < 1)
        return [{"t0": BACKTRACKING_T0, "rho": r, "c": a} for r in rho for a in c]
    if stage == "fine":
        best = previous_stage(method, "coarse", fingerprint)["selected"]["parameters"]["step"]
        grid = fine_linear_grid(grid, best, STEP_FINE_POINTS)
    return [{"step": step, **({"beta": beta} if method == "agd_constant" else {})} for step in grid]


def run_candidate(method, obj, parameters, stage):
    w0 = np.zeros(obj.d)
    final = stage == "final"
    iterations = 3 if stage == "warmup" else (CONVERGENCE_MAX_ITER if final else SEARCH_ITERATIONS)
    if method == "gd_fixed":
        return run_gd_fixed(obj, w0, parameters["step"], iterations, CONVERGENCE_TOL, final)
    if method == "gd_backtracking":
        return run_gd_backtracking(obj, w0, parameters["t0"], parameters["rho"], parameters["c"],
                                   iterations, CONVERGENCE_TOL, final, BACKTRACKING_MAX_TRIALS)
    scheme = "constant" if method == "agd_constant" else "dynamic"
    return run_agd(obj, w0, parameters["step"], iterations, CONVERGENCE_TOL, final,
                   scheme, parameters.get("beta"))


def export_stage(result, directory):
    rows = result["candidates"]
    fields = [key for key in rows[0] if key not in {"trace", "weights"}]
    with (directory / "search_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for row in rows:
        label = ", ".join(f"{k}={v:g}" for k, v in row["parameters"].items() if k != "beta")
        if result["stage"] == "final":
            label = f"Lần {row['candidate_index'] + 1}: {row['elapsed_s']:.2f}s"
        if not row["finite"]:
            label += " [không tìm được bước]"
        ax.semilogy(row["trace"]["iteration"],
                    np.maximum(np.asarray(row["trace"]["objective"]) - result["reference"]["f_star"], 1e-16),
                    label=label, linewidth=1.5,
                    marker="x" if not row["finite"] else None, markersize=6)
    stage_label = {"coarse": "dò thô", "fine": "dò tinh", "final": "chạy tới hội tụ"}[result["stage"]]
    ax.set(xlabel="Số bước cập nhật k", ylabel=r"$f(w_k)-f^*$ (thang log)")
    ax.grid(True, which="both", alpha=0.23)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "convergence.png", dpi=190)
    plt.close(fig)
    selected = result["selected"]
    lines = [f"# {LABELS[result['method']]} — {stage_label}", "",
             f"- Dữ liệu: `{result['data']['path']}`; 75.026 × 415; không chia lại.",
             f"- SHA-256: `{result['data']['sha256']}`.",
             f"- f* (Newton) = {result['reference']['f_star']:.12f}; λ=0,001; w0=0.",
             "- BLAS một luồng.",
             "- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.",
             "- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.",
             "- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.",
             f"- Cấu hình chọn: `{selected['parameters']}`.", ""]
    final = result["stage"] == "final"
    if final:
        lines.extend([f"- Chuẩn dừng: ||gradient|| ≤ {CONVERGENCE_TOL:g}.", ""])
    lines.extend(["| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) |"
                  + (" Đạt chuẩn dừng |" if final else ""),
                  "|---|---:|---:|---:|---:|" + ("---|" if final else "")])
    for row in rows:
        lines.append(f"| {row['parameters']} | {row['iterations_run']} | {row['final_f_gap']:.3e} | "
                     f"{row['final_grad_norm']:.3e} | {row['elapsed_s']:.3f} |"
                     + ((" Có |" if row['first_converged_at'] is not None else " Chưa |")
                        if final else ""))
    for row in rows:
        if not row["finite"]:
            lines.extend(["", f"Cấu hình `{row['parameters']}` dừng sau {row['iterations_run']} "
                          f"bước cập nhật: {row['note']}. Không đưa cấu hình này vào chọn tham số.",
                          "Dấu × ở k=0 trên hình là lần thử dừng trước bước cập nhật đầu tiên."])
    if result["stage"] == "final":
        lines.extend(["", f"Thời gian trung vị: {result['median_time_s']:.3f}s.",
                      "Chỉ số mô hình trong results.json là in-sample; tập test gốc rỗng."])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=LABELS, required=True)
    parser.add_argument("--stage", choices=("coarse", "fine", "final"), required=True)
    args = parser.parse_args()
    X, y, data = load_input()
    fingerprint = {"data_sha256": data["sha256"], "source_sha256": source_hash()}
    obj = LogisticObjective(X, y, lam=LAMBDA)
    reference = reference_for(obj, fingerprint)
    parameters = parameters_for(args.method, args.stage, fingerprint, reference["beta"])
    directory = OUT / args.method / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint or result["parameter_grid"] != parameters:
            raise ValueError("Checkpoint differs from current source/data/grid; archive it first")
    else:
        result = {"status": "running", "fingerprint": fingerprint, "data": data,
                  "method": args.method, "stage": args.stage, "parameter_grid": parameters,
                  "reference": {k: v for k, v in reference.items() if k not in {"w_star", "fingerprint"}},
                  "protocol": {"lambda": LAMBDA, "w0": "zero", "search_iterations": SEARCH_ITERATIONS,
                               "tol": CONVERGENCE_TOL, "final_max_iter": CONVERGENCE_MAX_ITER,
                               "timing_repeats": TIMING_REPEATS, "stop_on_convergence": args.stage == "final"},
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "blas": threadpool_info()}, "candidates": []}
    warmed_up = False
    for index in range(len(result["candidates"]), len(parameters)):
        print(f"[{args.method}/{args.stage}] {index + 1}/{len(parameters)}: {parameters[index]}", flush=True)
        if args.stage == "final" and not warmed_up:
            # Warm-up outside the recorded timed run.
            run_candidate(args.method, obj, parameters[index], "warmup")
            warmed_up = True
        run = run_candidate(args.method, obj, parameters[index], args.stage)
        row = run_summary(run, reference["f_star"], args.stage)
        row.pop("converged_within_500")
        row.update(candidate_index=index, weights=run.w.tolist())
        row["trace"].update(gradient_norm=run.grad_norms, time_s=run.times_s, accepted_steps=run.steps)
        result["candidates"].append(row)
        write_json(path, result)
        print(f"  saved: gap={row['final_f_gap']:.3e}; gradient={row['final_grad_norm']:.3e}; "
              f"time={row['elapsed_s']:.2f}s", flush=True)
    result["selected"] = {k: v for k, v in select_best(result["candidates"]).items()
                          if k not in {"trace", "weights"}}
    if args.stage == "final":
        times = [row["elapsed_s"] for row in result["candidates"]]
        result["median_time_s"] = float(np.median(times))
        index = int(np.argsort(times)[len(times) // 2])
        result["representative_candidate_index"] = index
        result["model_metrics_in_sample"] = classification_metrics(X, y, np.asarray(result["candidates"][index]["weights"]))
        result["model_metrics_w0_in_sample"] = classification_metrics(X, y, np.zeros(obj.d))
    result["status"] = "complete"
    export_stage(result, directory)
    write_json(path, result)
    print(f"Complete: {directory.relative_to(REPO)}; selected={result['selected']['parameters']}", flush=True)


if __name__ == "__main__":
    main()
