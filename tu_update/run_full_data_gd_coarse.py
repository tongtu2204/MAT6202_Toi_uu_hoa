"""Run only the existing GD coarse grid on Tai's verified full ridge matrix."""
from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ORIGINAL = REPO / "bai-lam-taitt" / "5_ma_nguon_tai_lap"
sys.path.insert(0, str(ORIGINAL))

from config import CONVERGENCE_TOL, GD_STEP_COARSE, LAMBDA, SEARCH_ITERATIONS
from gd_agd_runner import run_gd_fixed
from optim.objective import LogisticObjective
from optim.reference import newton_reference

EXPECTED_SHA256 = "0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4"
DATA_PATH = ORIGINAL / "artifacts" / "ridge.npz"
OUT = HERE / "artifacts" / "full_data_gd_coarse"


def main():
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Input does not match Tai's MANIFEST SHA-256")
    with np.load(DATA_PATH) as archive:
        X = np.asarray(archive["X_train"], dtype=float)
        y = np.asarray(archive["y_train"], dtype=float)
        test_shape = list(archive["X_test"].shape)
    if X.shape != (75026, 415) or len(y) != len(X):
        raise ValueError(f"Unexpected full-data shape: {X.shape}, {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite input")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("BLAS must use one thread")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Verified full dataset: {X.shape}; test={test_shape}; churn={y.mean():.6%}", flush=True)
    obj = LogisticObjective(X, y, lam=LAMBDA)
    w0 = np.zeros(obj.d)
    print("Computing Newton reference on the same full dataset...", flush=True)
    reference = newton_reference(obj, w0)
    print(f"f*={reference.f_star:.12f}, reference gradient={reference.grad_norm:.3e}", flush=True)

    result = {
        "data": {"path": str(DATA_PATH.relative_to(REPO)), "sha256": digest,
                 "train_shape": list(X.shape), "test_shape": test_shape,
                 "positive_rate": float(y.mean()), "split_again": False},
        "protocol": {"stage": "coarse", "lambda": LAMBDA,
                     "iterations_per_candidate": SEARCH_ITERATIONS,
                     "steps": list(GD_STEP_COARSE), "initialization": "w0=0",
                     "stop_on_convergence": False, "convergence_tol": CONVERGENCE_TOL,
                     "runner": "tu_update/gd_agd_runner.py:run_gd_fixed"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "blas": threadpool_info()},
        "reference": {"f_star": float(reference.f_star),
                      "grad_norm": float(reference.grad_norm), "mu_local": float(reference.mu)},
        "candidates": [],
    }
    for step in GD_STEP_COARSE:
        print(f"GD coarse: t={step:g}, {SEARCH_ITERATIONS} updates...", flush=True)
        run = run_gd_fixed(obj, w0, step, SEARCH_ITERATIONS, CONVERGENCE_TOL, False)
        differences = np.diff(run.objectives)
        row = {
            "step": float(step), "updates": int(run.iterations[-1]),
            "final_objective": float(run.final_objective),
            "final_gap": float(run.final_objective - reference.f_star),
            "final_grad_norm": float(run.final_grad_norm),
            "elapsed_s": float(run.elapsed_s), "finite": bool(run.finite),
            "monotone_decrease": bool(np.all(differences <= 1e-12)),
            "objective_increases": int(np.sum(differences > 1e-12)),
            "converged_at": run.converged_at,
            "trace": {"iteration": run.iterations, "objective": run.objectives,
                      "gradient_norm": run.grad_norms, "time_s": run.times_s},
        }
        result["candidates"].append(row)
        (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  f={row['final_objective']:.9f}; gap={row['final_gap']:.3e}; "
              f"gradient={row['final_grad_norm']:.3e}; increases={row['objective_increases']}; "
              f"time={row['elapsed_s']:.2f}s", flush=True)

    fields = [key for key in result["candidates"][0] if key != "trace"]
    with (OUT / "search_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in result["candidates"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for index, row in enumerate(result["candidates"]):
        trace = row["trace"]
        color = plt.get_cmap("tab10")(index)
        label = f"t={row['step']:g}"
        axes[0].plot(trace["iteration"], trace["objective"], label=label, color=color, lw=1.6)
        axes[1].semilogy(trace["iteration"],
                         np.maximum(np.asarray(trace["objective"]) - reference.f_star, 1e-16),
                         label=label, color=color, lw=1.6)
    axes[0].axhline(reference.f_star, color="black", ls="--", lw=1, label="f* (Newton)")
    axes[0].set(title="Hàm mục tiêu trong giai đoạn dò thô", ylabel="f(w_k)")
    axes[1].set(title="Sai số so với nghiệm tham chiếu", ylabel=r"$f(w_k)-f^*$ (thang log)")
    for ax in axes:
        ax.set_xlabel("Số bước cập nhật k")
        ax.set_xlim(0, SEARCH_ITERATIONS)
        ax.grid(True, which="both", alpha=0.23)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"GD bước cố định — dò thô trên dữ liệu đầy đủ của Tài\n"
                 f"75.026 dòng × 415 đặc trưng; λ={LAMBDA:g}; {SEARCH_ITERATIONS} bước/cấu hình")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "search_gd_fixed_coarse.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    lines = ["# GD dò thô trên dữ liệu đầy đủ của Tài", "",
             f"- Dữ liệu: `{result['data']['path']}` — 75.026 dòng × 415 đặc trưng; không chia lại.",
             f"- SHA-256 khớp MANIFEST: `{digest}`.",
             "- Nguồn: https://drive.google.com/file/d/1mehXGDef4XyBnGdZp4gG89EbDUnAbU6Q/view",
             f"- λ={LAMBDA:g}; w0=0; {SEARCH_ITERATIONS} bước/cấu hình; BLAS một luồng.",
             "- Giữ lưới dò thô và mã GD của tu_update; chỉ thay đầu vào bằng ma trận đầy đủ của Tài.",
             f"- Nghiệm tham chiếu Newton: f*={reference.f_star:.12f}, chuẩn gradient={reference.grad_norm:.3e}.",
             "", "| t | Số bước | f cuối | f−f* | Chuẩn gradient cuối | Thời gian (s) | Giảm đều |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    for row in result["candidates"]:
        lines.append(f"| {row['step']:g} | {row['updates']} | {row['final_objective']:.9f} | "
                     f"{row['final_gap']:.3e} | {row['final_grad_norm']:.3e} | "
                     f"{row['elapsed_s']:.2f} | {'Có' if row['monotone_decrease'] else 'Không'} |")
    lines.extend(["", "Đây là kết quả dò thô; chưa chạy dò tinh, AGD hay đánh giá chất lượng phân loại.", ""])
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved coarse GD results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
