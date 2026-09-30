#!/usr/bin/env python3
"""Create reproducible tables and figures from gd_agd_results.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _row_at(result, method, iteration):
    return next(row for row in result[method]["comparison"]
                if row["iteration"] == iteration)


def _safe_gap(value):
    return max(float(value), 1e-18)


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    value = float(value)
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def write_comparison_csv(result, path):
    fields = [
        "method", "configuration", "iteration", "eta", "beta",
        "f_gap", "grad_norm", "median_time_s", "accuracy",
        "balanced_accuracy", "roc_auc", "pr_auc", "f1",
        "precision", "recall", "log_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in ("gd", "agd"):
            if method not in result:
                continue
            for row in result[method]["comparison"]:
                for config_name in ("baseline", "tuned"):
                    values = row[config_name]
                    params = result[method][f"{config_name}_parameters"]
                    metrics = values["test"]
                    writer.writerow({
                        "method": method.upper(),
                        "configuration": config_name,
                        "iteration": row["iteration"],
                        "eta": params["eta"],
                        "beta": params["beta"],
                        "f_gap": values["f_gap"],
                        "grad_norm": values["grad_norm"],
                        "median_time_s": values["median_time_s"],
                        **{key: metrics[key] for key in fields[8:]},
                    })


def plot_gd_search(result, path):
    rows = result["gd"]["search"]
    eta = np.array([row["eta"] for row in rows], dtype=float)
    gap = np.array([_safe_gap(row.get("f_gap", np.inf)) for row in rows])
    finite = np.isfinite(gap)
    chosen = result["gd"]["tuned_parameters"]["eta"]
    baseline = result["gd"]["baseline_parameters"]["eta"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.semilogy(eta[finite], gap[finite], "o-", color="#275dad", lw=1.5,
                ms=4, label="Grid search")
    unstable = [row for row in rows
                if row.get("stability_checked") and not row.get("stable")]
    if unstable:
        ax.scatter(
            [row["eta"] for row in unstable],
            [_safe_gap(row["f_gap"]) for row in unstable],
            marker="x", s=75, linewidth=2, color="#9c1c1c",
            label="Rejected: unstable by 500 iterations", zorder=5,
        )
    ax.axvline(baseline, color="#666666", ls="--", lw=1.3,
               label=f"Baseline 1/L = {baseline:.3f}")
    ax.axvline(chosen, color="#d62728", ls="-", lw=1.5,
               label=f"Selected t = {chosen:g}")
    ax.set(xlabel="Step size t", ylabel=r"Training objective gap $f-f^*$",
           title="GD search at the fixed selection budget")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_agd_heatmap(result, path):
    rows = result["agd"]["search"]
    steps = sorted({float(row["eta"]) for row in rows})
    betas = sorted({float(row["beta"]) for row in rows})
    matrix = np.full((len(betas), len(steps)), np.nan)
    step_idx = {value: i for i, value in enumerate(steps)}
    beta_idx = {value: i for i, value in enumerate(betas)}
    for row in rows:
        if row.get("finite") and np.isfinite(row.get("f_gap", np.inf)):
            matrix[beta_idx[float(row["beta"])], step_idx[float(row["eta"])]] = (
                np.log10(_safe_gap(row["f_gap"]))
            )

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(steps)), [f"{x:g}" for x in steps], rotation=55)
    ax.set_yticks(range(len(betas)), [f"{x:g}" for x in betas])
    ax.set(xlabel="Step size t", ylabel=r"Momentum $\beta$",
           title=r"AGD joint search: $\log_{10}(f-f^*)$")
    selected = result["agd"]["tuned_parameters"]
    ax.scatter(step_idx[selected["eta"]], beta_idx[selected["beta"]],
               marker="*", s=180, facecolor="white", edgecolor="black",
               linewidth=0.8, label="Selected")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}(f-f^*)$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(result, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, method in zip(axes, ("gd", "agd")):
        rows = result[method]["comparison"]
        x = [row["iteration"] for row in rows]
        for config, color in (("baseline", "#666666"), ("tuned", "#d62728")):
            y = [_safe_gap(row[config]["f_gap"]) for row in rows]
            ax.loglog(x, y, "o-", color=color, label=config.capitalize())
        ax.set_title(method.upper())
        ax.set_xlabel("Iterations / gradient evaluations")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel(r"Training objective gap $f-f^*$")
    fig.suptitle("Equal-budget comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_predictive_delta(result, path):
    selection = int(result["experiment"]["selection_iteration"])
    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "f1"]
    labels = ["Accuracy", "Balanced acc.", "ROC-AUC", "PR-AUC", "F1"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, method, color in ((-width / 2, "gd", "#275dad"),
                                  (width / 2, "agd", "#e07a1f")):
        row = _row_at(result, method, selection)
        delta = [100.0 * (row["tuned"]["test"][metric]
                          - row["baseline"]["test"][metric])
                 for metric in metrics]
        ax.bar(x + offset, delta, width, label=method.upper(), color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Tuned − baseline (percentage points)")
    ax.set_title(f"Held-out test metric change at {selection} iterations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(result, path):
    selection = int(result["experiment"]["selection_iteration"])
    final_iter = max(result["experiment"]["checkpoints"])
    lines = [
        "# Kết quả cập nhật GD và AGD",
        "",
        "> Kết quả này được sinh tự động từ `gd_agd_results.json`; chưa sửa `main.tex`.",
        "",
        "## Giao thức",
        "",
        f"- Chọn cấu hình theo `f-f*` trên train tại {selection} vòng, sau khi "
        f"loại ứng viên không ổn định tới {result['experiment']['stability_iteration']} vòng.",
        "- Tập test được giữ ngoài quá trình dò và chỉ dùng để báo cáo sau khi chọn.",
        f"- Thời gian là trung vị của {result['experiment']['timing_repeats']} lần chạy.",
        f"- Kích thước: train={result['data']['sizes']['train']:,}, "
        f"validation={result['data']['sizes']['validation']:,}, "
        f"test={result['data']['sizes']['test']:,}, "
        f"d={result['data']['sizes']['features']}.",
        "",
        "## Tham số được chọn",
        "",
        "| Thuật toán | Baseline | Sau dò | Số cấu hình dò |",
        "|---|---|---|---:|",
    ]
    for method in ("gd", "agd"):
        if method not in result:
            continue
        base = result[method]["baseline_parameters"]
        tune = result[method]["tuned_parameters"]
        base_text = f"t={base['eta']:.6g}"
        tune_text = f"t={tune['eta']:.6g}"
        if base["beta"] is not None:
            base_text += f", β={base['beta']:.6g}"
            tune_text += f", β={tune['beta']:.6g}"
        lines.append(
            f"| {method.upper()} | {base_text} | {tune_text} | "
            f"{len(result[method]['search'])} |"
        )

    lines += [
        "",
        f"## So sánh tại cùng ngân sách {selection} vòng",
        "",
        "| Thuật toán | Cấu hình | f−f* | ‖∇f‖ | Giây | Accuracy | F1 | ROC-AUC | PR-AUC | Log-loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("gd", "agd"):
        if method not in result:
            continue
        row = _row_at(result, method, selection)
        for config in ("baseline", "tuned"):
            values = row[config]
            metrics = values["test"]
            lines.append(
                f"| {method.upper()} | {config} | {_fmt(values['f_gap'])} | "
                f"{_fmt(values['grad_norm'])} | {_fmt(values['median_time_s'])} | "
                f"{_fmt(metrics['accuracy'])} | {_fmt(metrics['f1'])} | "
                f"{_fmt(metrics['roc_auc'])} | {_fmt(metrics['pr_auc'])} | "
                f"{_fmt(metrics['log_loss'])} |"
            )

    lines += ["", "## Mức thay đổi tại ngân sách chọn", ""]
    for method in ("gd", "agd"):
        if method not in result:
            continue
        row = _row_at(result, method, selection)
        base, tune = row["baseline"], row["tuned"]
        gap_gain = 100.0 * (1.0 - tune["f_gap"] / base["f_gap"])
        time_change = 100.0 * (tune["median_time_s"] / base["median_time_s"] - 1.0)
        accuracy_pp = 100.0 * (tune["test"]["accuracy"] - base["test"]["accuracy"])
        f1_pp = 100.0 * (tune["test"]["f1"] - base["test"]["f1"])
        lines.append(
            f"- **{method.upper()}**: giảm objective gap {_fmt(gap_gain, 2)}%; "
            f"thời gian thay đổi {_fmt(time_change, 2)}%; "
            f"Accuracy {accuracy_pp:+.3f} điểm %, F1 {f1_pp:+.3f} điểm %."
        )

    lines += [
        "",
        f"## Trạng thái ở {final_iter:,} vòng và hội tụ",
        "",
        "| Thuật toán | Cấu hình | f−f* ở mốc cuối | Đạt ‖∇f‖≤1e-10 | Vòng hội tụ | Giây hội tụ |",
        "|---|---|---:|---|---:|---:|",
    ]
    for method in ("gd", "agd"):
        if method not in result:
            continue
        final_row = _row_at(result, method, final_iter)
        for config in ("baseline", "tuned"):
            conv = result[method]["convergence"][config]
            lines.append(
                f"| {method.upper()} | {config} | {_fmt(final_row[config]['f_gap'])} | "
                f"{'Có' if conv['converged'] else 'Không'} | "
                f"{_fmt(conv['iteration'])} | {_fmt(conv['time_s'])} |"
            )

    lines += [
        "",
        "## Tệp kết quả",
        "",
        "- `gd_agd_results.json`: toàn bộ lưới, checkpoint và metric.",
        "- `comparison.csv`: bảng dài để kiểm tra hoặc vẽ lại.",
        "- `gd_step_search.png`: lưới bước GD.",
        "- `agd_joint_search.png`: heatmap dò đồng thời `(t, beta)`.",
        "- `equal_budget_convergence.png`: baseline và sau dò tại cùng ngân sách.",
        "- `test_metric_delta.png`: thay đổi metric trên test.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(result_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = result_path.parent
    quick = bool(result["experiment"].get("quick"))
    suffix = "_quick" if quick else ""

    write_comparison_csv(result, output / f"comparison{suffix}.csv")
    if "gd" in result:
        plot_gd_search(result, output / f"gd_step_search{suffix}.png")
    if "agd" in result:
        plot_agd_heatmap(result, output / f"agd_joint_search{suffix}.png")
    if "gd" in result and "agd" in result:
        plot_convergence(result, output / f"equal_budget_convergence{suffix}.png")
        plot_predictive_delta(result, output / f"test_metric_delta{suffix}.png")
    write_summary(result, output / f"summary{suffix}.md")
    print(f"wrote tables and figures to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path,
                        nargs="?", default=Path(__file__).parent / "artifacts" /
                        "gd_agd_results.json")
    args = parser.parse_args()
    generate_artifacts(args.result)


if __name__ == "__main__":
    main()
