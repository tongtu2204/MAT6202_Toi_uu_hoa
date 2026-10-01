#!/usr/bin/env python3
"""Sinh bảng, hình và tóm tắt từ ``artifacts/results.json``."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "gd_fixed": "GD bước cố định",
    "gd_backtracking": "GD backtracking",
    "agd_constant": "AGD momentum cố định",
    "agd_dynamic": r"AGD $\beta_k=(k-2)/(k+1)$",
}


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    value = float(value)
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _parameter_text(parameters):
    parts = []
    for key in ("step", "t0", "rho", "c", "beta"):
        if key in parameters:
            parts.append(f"{key}={parameters[key]:.6g}")
    return ", ".join(parts)


def write_search_csv(result, path):
    fields = [
        "method", "stage", "step", "t0", "rho", "c", "beta",
        "finite", "converged_within_500", "first_converged_at",
        "final_objective", "final_f_gap", "final_grad_norm",
        "gradient_evals", "objective_evals", "elapsed_s",
        "mean_step", "min_step", "max_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, search in result["searches"].items():
            for row in search["candidates"]:
                parameters = row["parameters"]
                writer.writerow({
                    "method": method,
                    "stage": row["stage"],
                    **{key: parameters.get(key) for key in ("step", "t0", "rho", "c", "beta")},
                    **{key: row.get(key) for key in fields[7:]},
                })


def write_optimization_csv(result, path):
    fields = [
        "method", "parameters", "converged", "converged_at",
        "final_objective", "final_f_gap", "final_grad_norm",
        "gradient_evals", "objective_evals", "backtracking_trials",
        "median_time_s", "mean_step", "min_step", "max_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, row in result["finalists"].items():
            writer.writerow({
                "method": method,
                "parameters": _parameter_text(row["parameters"]),
                **{key: row.get(key) for key in fields[2:]},
            })


def write_model_csv(result, path):
    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "log_loss",
    ]
    fields = ["model", "label", "dataset", "objective", *metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model, row in result["model_metrics"].items():
            for dataset in ("train", "test"):
                writer.writerow({
                    "model": model,
                    "label": row["label"],
                    "dataset": dataset,
                    "objective": row["objective"],
                    **{metric: row[dataset][metric] for metric in metrics},
                })


def _search_xy(search):
    return [row for row in search["candidates"] if row["finite"]]


def plot_searches(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    line_methods = [
        ("gd_fixed", axes[0, 0], "GD: dò bước cố định"),
        ("agd_constant", axes[1, 0], "AGD momentum cố định: dò bước"),
        ("agd_dynamic", axes[1, 1], r"AGD $\beta_k=(k-2)/(k+1)$: dò bước"),
    ]
    for method, ax, title in line_methods:
        rows = _search_xy(result["searches"][method])
        for stage, marker, color in (("coarse", "o", "#315f9f"), ("fine", "x", "#d35400")):
            stage_rows = sorted(
                (row for row in rows if row["stage"] == stage),
                key=lambda row: row["parameters"]["step"],
            )
            if stage_rows:
                ax.semilogy(
                    [row["parameters"]["step"] for row in stage_rows],
                    [max(row["final_grad_norm"], 1e-18) for row in stage_rows],
                    marker, ms=5, color=color, label=stage,
                )
        selected = result["searches"][method]["selected"]
        ax.scatter(
            selected["parameters"]["step"], max(selected["final_grad_norm"], 1e-18),
            marker="*", s=180, color="#c51b29", edgecolor="black", linewidth=0.5,
            label="được chọn", zorder=5,
        )
        ax.set(title=title, xlabel="Độ dài bước t", ylabel=r"$\|\nabla f\|$ sau 500 vòng")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes[0, 1]
    rows = _search_xy(result["searches"]["gd_backtracking"])
    values = np.log10([max(row["final_grad_norm"], 1e-18) for row in rows])
    scatter = ax.scatter(
        [row["parameters"]["rho"] for row in rows],
        [row["parameters"]["c"] for row in rows],
        c=values, cmap="viridis_r", s=48,
        marker="o", edgecolor="white", linewidth=0.3,
    )
    selected = result["searches"]["gd_backtracking"]["selected"]
    ax.scatter(
        selected["parameters"]["rho"], selected["parameters"]["c"],
        marker="*", s=220, color="#c51b29", edgecolor="black", linewidth=0.6,
        label="được chọn", zorder=5,
    )
    ax.set_yscale("log")
    ax.set(title=r"GD backtracking: dò $(\rho,c)$", xlabel=r"$\rho$", ylabel=r"$c$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=ax, label=r"$\log_{10}\|\nabla f\|$ sau 500 vòng")

    fig.suptitle("Dò thô và dò tinh — mỗi cấu hình chạy 500 vòng", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _plot_trace(ax, result, keys, title):
    f_star = float(result["objective"]["f_star"])
    colors = ("#315f9f", "#d35400")
    for key, color in zip(keys, colors):
        trace = result["finalists"][key]["trace"]
        iterations = np.asarray(trace["iteration"])
        gap = np.maximum(np.asarray(trace["objective"]) - f_star, 1e-18)
        ax.semilogy(iterations, gap, color=color, lw=1.7, label=METHOD_LABELS[key])
    ax.set(title=title, xlabel="Vòng lặp", ylabel=r"$f(w_k)-f^*$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def plot_hierarchical_comparison(result, path):
    gd_winner = result["comparisons"]["gd_internal"]["winner"]
    agd_winner = result["comparisons"]["agd_internal"]["winner"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _plot_trace(axes[0], result, ["gd_fixed", "gd_backtracking"], "1. So sánh nội bộ GD")
    _plot_trace(axes[1], result, ["agd_constant", "agd_dynamic"], "2. So sánh nội bộ AGD")
    _plot_trace(axes[2], result, [gd_winner, agd_winner], "3. GD thắng vs AGD thắng")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_model_metrics(result, path):
    gd_winner = result["comparisons"]["gd_internal"]["winner"]
    agd_winner = result["comparisons"]["agd_internal"]["winner"]
    keys = ["untrained_w0", gd_winner, agd_winner, "sklearn_reference"]
    labels = ["Chưa tối ưu", "GD tốt nhất", "AGD tốt nhất", "scikit-learn"]
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Balanced acc.", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ("#9e9e9e", "#315f9f", "#d35400", "#4c9f70")
    for index, (key, label, color) in enumerate(zip(keys, labels, colors)):
        values = [result["model_metrics"][key]["test"][metric] for metric in metrics]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Giá trị trên tập test")
    ax.set_title("Chất lượng mô hình trước và sau tối ưu")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_summary(result, path):
    comparisons = result["comparisons"]
    gd_winner = comparisons["gd_internal"]["winner"]
    agd_winner = comparisons["agd_internal"]["winner"]
    overall = comparisons["gd_vs_agd"]["winner"]
    lines = [
        "# Kết quả cập nhật GD và AGD",
        "",
        "> Sinh tự động từ `results.json`. `main.tex` chưa được sửa.",
        "",
        "## Giao thức",
        "",
        f"- Dữ liệu: train `{result['data']['train_shape']}`, test `{result['data']['test_shape']}` từ `data/ridge.npz`.",
        "- Không chia lại train; test không tham gia dò tham số.",
        f"- Mỗi ứng viên ở lưới thô và lưới tinh chạy đúng **{result['protocol']['search_iterations_per_candidate']} vòng**.",
        f"- Sau khi khóa tham số, chạy lại từ `w0=0` tới `||grad|| <= {result['protocol']['convergence_tolerance']:.0e}`.",
        f"- Trần kỹ thuật: {result['protocol']['safety_max_iterations']:,} vòng; thời gian lấy trung vị {result['protocol']['timing_repeats']} lần.",
        "",
        "## Tham số được chọn sau dò 500 vòng",
        "",
        "| Biến thể | Tham số | Số cấu hình | Gradient norm cuối |",
        "|---|---|---:|---:|",
    ]
    for key in ("gd_fixed", "gd_backtracking", "agd_constant", "agd_dynamic"):
        search = result["searches"][key]
        selected = search["selected"]
        lines.append(
            f"| {METHOD_LABELS[key]} | `{_parameter_text(selected['parameters'])}` | "
            f"{len(search['candidates'])} | {_fmt(selected['final_grad_norm'])} |"
        )

    lines += [
        "",
        "## Chạy chính thức đến hội tụ",
        "",
        "| Biến thể | Hội tụ | Số vòng | Thời gian trung vị (s) | Gradient eval | Objective eval | f−f* |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("gd_fixed", "gd_backtracking", "agd_constant", "agd_dynamic"):
        row = result["finalists"][key]
        lines.append(
            f"| {METHOD_LABELS[key]} | {'Có' if row['converged'] else 'Không'} | "
            f"{_fmt(row['converged_at'])} | {_fmt(row['median_time_s'])} | "
            f"{_fmt(row['gradient_evals'])} | {_fmt(row['objective_evals'])} | "
            f"{_fmt(row['final_f_gap'])} |"
        )

    lines += [
        "",
        "## So sánh theo ba tầng",
        "",
        f"1. **Nội bộ GD:** `{METHOD_LABELS[gd_winner]}` thắng theo thời gian tới hội tụ.",
        f"2. **Nội bộ AGD:** `{METHOD_LABELS[agd_winner]}` thắng theo thời gian tới hội tụ.",
        f"3. **GD với AGD:** `{METHOD_LABELS[overall]}` là phương án nhanh hơn trong hai người thắng.",
        "",
        "## Chất lượng mô hình trên test",
        "",
        "| Mô hình | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Log-loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_keys = ["untrained_w0", gd_winner, agd_winner, "sklearn_reference"]
    for key in model_keys:
        row = result["model_metrics"][key]
        metric = row["test"]
        lines.append(
            f"| {row['label']} | {_fmt(metric['accuracy'])} | {_fmt(metric['balanced_accuracy'])} | "
            f"{_fmt(metric['precision'])} | {_fmt(metric['recall'])} | {_fmt(metric['f1'])} | "
            f"{_fmt(metric['roc_auc'])} | {_fmt(metric['pr_auc'])} | {_fmt(metric['log_loss'])} |"
        )

    before = result["model_metrics"]["untrained_w0"]["test"]
    after = result["model_metrics"][overall]["test"]
    lines += [
        "",
        "## Diễn giải chính",
        "",
        f"- So với `w0=0`, phương án cuối tăng Accuracy **{100*(after['accuracy']-before['accuracy']):.2f} điểm %**, "
        f"F1 **{100*(after['f1']-before['f1']):.2f} điểm %** và ROC-AUC **{100*(after['roc_auc']-before['roc_auc']):.2f} điểm %**.",
        "- Khi các thuật toán đều hội tụ về cùng nghiệm Ridge logistic, metric dự đoán gần như trùng nhau; khác biệt chính nằm ở số vòng, thời gian và số lần đánh giá hàm.",
        "- Accuracy không được dùng một mình vì tỷ lệ churn chỉ khoảng 13,7%; cần đọc cùng Balanced Accuracy, F1, ROC-AUC, PR-AUC và Log-loss.",
        "",
        "## Tệp kết quả",
        "",
        "- `results.json`: toàn bộ lưới, quỹ đạo và metric.",
        "- `search_results.csv`: toàn bộ ứng viên tham số.",
        "- `optimization_results.csv`: bốn cấu hình chạy tới hội tụ.",
        "- `model_metrics.csv`: metric train/test trước và sau tối ưu.",
        "- `parameter_search.png`: bốn phép dò tham số.",
        "- `hierarchical_comparison.png`: so GD, so AGD, rồi GD–AGD.",
        "- `model_comparison.png`: metric test trước và sau tối ưu.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(result_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = result_path.parent
    write_search_csv(result, output / "search_results.csv")
    write_optimization_csv(result, output / "optimization_results.csv")
    write_model_csv(result, output / "model_metrics.csv")
    plot_searches(result, output / "parameter_search.png")
    plot_hierarchical_comparison(result, output / "hierarchical_comparison.png")
    plot_model_metrics(result, output / "model_comparison.png")
    write_summary(result, output / "summary.md")
    print(f"Đã sinh bảng, hình và tóm tắt tại {output}", flush=True)


if __name__ == "__main__":
    generate_artifacts(Path(__file__).parent / "artifacts" / "results.json")
