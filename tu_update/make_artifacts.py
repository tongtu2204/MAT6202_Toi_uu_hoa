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
    fields = [
        "model", "label", "dataset", "objective",
        "distance_to_reference", *metrics,
    ]
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
                    "distance_to_reference": row["distance_to_reference"],
                    **{metric: row[dataset][metric] for metric in metrics},
                })


def _same_parameters(left, right):
    keys = set(left).union(right)
    return all(
        key in left and key in right
        and np.isclose(float(left[key]), float(right[key]), rtol=0.0, atol=1e-12)
        for key in keys
    )


def _candidate_label(method, parameters):
    if method == "gd_backtracking":
        return rf"$\rho$={parameters['rho']:.3g}, $c$={parameters['c']:.3g}"
    return rf"$t$={parameters['step']:.4g}"


def plot_method_search(result, method, path):
    """Vẽ riêng đường hội tụ của dò thô và dò tinh cho một phương pháp."""
    search = result["searches"][method]
    f_star = float(result["objective"]["f_star"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True, sharey=True)
    stage_specs = (
        ("coarse", "Dò thử nghiệm thô", search["coarse_best"]),
        ("fine", "Dò thử nghiệm tinh chỉnh", search["selected"]),
    )
    colors = plt.get_cmap("tab10")

    for ax, (stage, stage_title, chosen) in zip(axes, stage_specs):
        rows = [row for row in search["candidates"] if row["stage"] == stage]
        if method == "gd_backtracking":
            rows.sort(key=lambda row: (row["parameters"]["rho"], row["parameters"]["c"]))
        else:
            rows.sort(key=lambda row: row["parameters"]["step"])

        for index, row in enumerate(rows):
            trace = row["trace"]
            gap = np.maximum(np.asarray(trace["objective"], dtype=float) - f_star, 1e-18)
            selected = _same_parameters(row["parameters"], chosen["parameters"])
            label = _candidate_label(method, row["parameters"])
            if selected:
                label += "  ← chọn"
            ax.semilogy(
                trace["iteration"], gap,
                color="#c51b29" if selected else colors(index % 10),
                lw=2.8 if selected else 1.15,
                alpha=1.0 if selected else 0.72,
                label=label,
                zorder=5 if selected else 2,
            )

        ax.set_title(stage_title, fontweight="bold")
        ax.set_xlabel("Vòng lặp k (ngân sách 500 vòng)")
        ax.grid(True, which="both", alpha=0.23)
        ax.set_xlim(0, result["protocol"]["search_iterations_per_candidate"])
        ax.set_ylim(1e-18, 10.0)
        ax.legend(fontsize=7.3, ncol=1, loc="best", framealpha=0.92)

    axes[0].set_ylabel(r"Sai số mục tiêu $f(w_k)-f^*$ (thang log)")
    fig.suptitle(
        f"{METHOD_LABELS[method]} — đường hội tụ khi dò tham số",
        fontsize=14,
    )
    fig.text(
        0.5, 0.015,
        "Mỗi đường là một cấu hình; đường đỏ đậm là cấu hình được chọn ở từng giai đoạn.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
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


def plot_all_methods(result, path):
    """So sánh cuối cùng cả bốn cấu hình đã khóa theo vòng lặp và thời gian."""
    f_star = float(result["objective"]["f_star"])
    colors = {
        "gd_fixed": "#315f9f",
        "gd_backtracking": "#4c9f70",
        "agd_constant": "#d35400",
        "agd_dynamic": "#8e5aa7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for key in ("gd_fixed", "gd_backtracking", "agd_constant", "agd_dynamic"):
        row = result["finalists"][key]
        trace = row["trace"]
        gap = np.maximum(np.asarray(trace["objective"], dtype=float) - f_star, 1e-18)
        axes[0].semilogy(
            trace["iteration"], gap, lw=2.0, color=colors[key], label=METHOD_LABELS[key]
        )
        axes[1].semilogy(
            trace["time_s"], gap, lw=2.0, color=colors[key], label=METHOD_LABELS[key]
        )

    axes[0].set(
        title="Độ chính xác theo số vòng",
        xlabel="Vòng lặp k",
        ylabel=r"$f(w_k)-f^*$ (thang log)",
    )
    axes[1].set(
        title="Độ chính xác theo thời gian",
        xlabel="Thời gian (giây)",
        ylabel=r"$f(w_k)-f^*$ (thang log)",
    )
    for ax in axes:
        ax.grid(True, which="both", alpha=0.23)
        ax.legend(fontsize=8)
    fig.suptitle("So sánh cuối cùng bốn phương pháp sau khi khóa tham số", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics(result, path):
    keys = [
        "untrained_w0", "gd_fixed", "gd_backtracking",
        "agd_constant", "agd_dynamic", "sklearn_reference",
    ]
    labels = [
        "Chưa tối ưu", "GD cố định", "GD backtracking",
        "AGD cố định", "AGD động", "scikit-learn",
    ]
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Balanced acc.", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.135
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [3.2, 1]})
    ax = axes[0]
    colors = ("#9e9e9e", "#315f9f", "#4c9f70", "#d35400", "#8e5aa7", "#333333")
    for index, (key, label, color) in enumerate(zip(keys, labels, colors)):
        values = [result["model_metrics"][key]["test"][metric] for metric in metrics]
        ax.bar(x + (index - 2.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Giá trị trên tập test")
    ax.set_title("Các chỉ số phân loại (cao hơn tốt hơn)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    log_losses = [result["model_metrics"][key]["test"]["log_loss"] for key in keys]
    axes[1].bar(np.arange(len(keys)), log_losses, color=colors)
    axes[1].set_xticks(np.arange(len(keys)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("Log-loss trên tập test")
    axes[1].set_title("Log-loss (thấp hơn tốt hơn)")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Chất lượng mô hình trước và sau tối ưu", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
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
        "| Mô hình | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Log-loss | ||w−w*|| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_keys = [
        "untrained_w0", "gd_fixed", "gd_backtracking",
        "agd_constant", "agd_dynamic", "sklearn_reference",
    ]
    for key in model_keys:
        row = result["model_metrics"][key]
        metric = row["test"]
        lines.append(
            f"| {row['label']} | {_fmt(metric['accuracy'])} | {_fmt(metric['balanced_accuracy'])} | "
            f"{_fmt(metric['precision'])} | {_fmt(metric['recall'])} | {_fmt(metric['f1'])} | "
            f"{_fmt(metric['roc_auc'])} | {_fmt(metric['pr_auc'])} | {_fmt(metric['log_loss'])} | "
            f"{_fmt(row['distance_to_reference'])} |"
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
        "- `search_gd_fixed.png`: dò thô/tinh của GD bước cố định.",
        "- `search_gd_backtracking.png`: dò thô/tinh của GD backtracking.",
        "- `search_agd_constant.png`: dò thô/tinh của AGD momentum cố định.",
        "- `search_agd_dynamic.png`: dò thô/tinh của AGD momentum động.",
        "- `hierarchical_comparison.png`: so GD, so AGD, rồi GD–AGD.",
        "- `all_methods_comparison.png`: cả bốn đường theo số vòng và thời gian.",
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
    for method in ("gd_fixed", "gd_backtracking", "agd_constant", "agd_dynamic"):
        plot_method_search(result, method, output / f"search_{method}.png")
    plot_hierarchical_comparison(result, output / "hierarchical_comparison.png")
    plot_all_methods(result, output / "all_methods_comparison.png")
    plot_model_metrics(result, output / "model_comparison.png")
    write_summary(result, output / "summary.md")
    print(f"Đã sinh bảng, hình và tóm tắt tại {output}", flush=True)


if __name__ == "__main__":
    generate_artifacts(Path(__file__).parent / "artifacts" / "results.json")
