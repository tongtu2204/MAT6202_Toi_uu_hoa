"""Các vòng lặp GD/AGD có đo hội tụ và chi phí tính toán.

Quy ước đếm một vòng ngoài:
- GD cố định: một gradient; một objective để ghi lịch sử.
- GD backtracking: một gradient; nhiều objective do Armijo.
- AGD: một gradient tại điểm momentum; một objective tại điểm cập nhật.

AGD chỉ tính thêm gradient tại nghiệm ứng viên khi chuẩn dừng tại điểm momentum
đã nhỏ hơn tolerance. Lần kiểm tra hiếm này được tính vào ``gradient_evals``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationRun:
    method: str
    parameters: dict[str, float]
    w: np.ndarray
    iterations: list[int] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    times_s: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)
    converged: bool = False
    converged_at: int | None = None
    finite: bool = True
    gradient_evals: int = 0
    objective_evals: int = 0
    backtracking_trials: int = 0
    final_grad_norm: float = float("inf")
    note: str = ""

    @property
    def final_objective(self) -> float:
        return self.objectives[-1] if self.objectives else float("inf")

    @property
    def elapsed_s(self) -> float:
        return self.times_s[-1] if self.times_s else 0.0


def _safe(w: np.ndarray, value: float, grad: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(w))
        and np.isfinite(value)
        and np.all(np.isfinite(grad))
        and np.linalg.norm(w) < 1e12
    )


def _record(run: OptimizationRun, iteration: int, value: float,
            grad_norm: float, started: float) -> None:
    run.iterations.append(int(iteration))
    run.objectives.append(float(value))
    run.grad_norms.append(float(grad_norm))
    run.times_s.append(float(time.perf_counter() - started))


def _mark_first_convergence(run: OptimizationRun, iteration: int,
                            grad_norm: float, tol: float) -> None:
    if run.converged_at is None and grad_norm <= tol:
        run.converged_at = int(iteration)


def run_gd_fixed(obj, w0: np.ndarray, step: float, max_iter: int,
                 tol: float, stop_on_convergence: bool) -> OptimizationRun:
    run = OptimizationRun("GD bước cố định", {"step": float(step)}, w0.copy())
    w = np.asarray(w0, dtype=float).copy()
    started = time.perf_counter()

    value = float(obj.value(w))
    grad = obj.grad(w)
    run.objective_evals += 1
    run.gradient_evals += 1
    grad_norm = float(np.linalg.norm(grad))
    _record(run, 0, value, grad_norm, started)

    for iteration in range(1, max_iter + 1):
        w = w - float(step) * grad
        value = float(obj.value(w))
        grad = obj.grad(w)
        run.objective_evals += 1
        run.gradient_evals += 1
        grad_norm = float(np.linalg.norm(grad))
        if not _safe(w, value, grad):
            run.finite = False
            run.note = "phân kỳ hoặc xuất hiện giá trị không hữu hạn"
            break
        _record(run, iteration, value, grad_norm, started)
        _mark_first_convergence(run, iteration, grad_norm, tol)
        if stop_on_convergence and run.converged_at is not None:
            break

    run.w = w
    run.final_grad_norm = grad_norm
    run.converged = run.converged_at is not None
    return run


def run_gd_backtracking(obj, w0: np.ndarray, t0: float, rho: float, c: float,
                        max_iter: int, tol: float, stop_on_convergence: bool,
                        max_trials: int = 60) -> OptimizationRun:
    if not 0.0 < rho < 1.0:
        raise ValueError("rho phải nằm trong (0, 1)")
    if not 0.0 < c < 1.0:
        raise ValueError("c phải nằm trong (0, 1)")

    params = {"t0": float(t0), "rho": float(rho), "c": float(c)}
    run = OptimizationRun("GD backtracking", params, w0.copy())
    w = np.asarray(w0, dtype=float).copy()
    started = time.perf_counter()

    value = float(obj.value(w))
    grad = obj.grad(w)
    run.objective_evals += 1
    run.gradient_evals += 1
    grad_norm = float(np.linalg.norm(grad))
    _record(run, 0, value, grad_norm, started)

    for iteration in range(1, max_iter + 1):
        slope = -float(grad @ grad)
        step = float(t0)
        accepted_value = float("inf")
        trials = 0
        for trials in range(1, max_trials + 1):
            candidate = w - step * grad
            accepted_value = float(obj.value(candidate))
            run.objective_evals += 1
            if accepted_value <= value + c * step * slope:
                break
            step *= rho
        else:
            run.finite = False
            run.note = "Armijo không tìm được bước trong giới hạn thử"
            break

        run.backtracking_trials += trials
        run.steps.append(float(step))
        w = candidate
        value = accepted_value
        grad = obj.grad(w)
        run.gradient_evals += 1
        grad_norm = float(np.linalg.norm(grad))
        if not _safe(w, value, grad):
            run.finite = False
            run.note = "phân kỳ hoặc xuất hiện giá trị không hữu hạn"
            break
        _record(run, iteration, value, grad_norm, started)
        _mark_first_convergence(run, iteration, grad_norm, tol)
        if stop_on_convergence and run.converged_at is not None:
            break

    run.w = w
    run.final_grad_norm = grad_norm
    run.converged = run.converged_at is not None
    return run


def run_agd(obj, w0: np.ndarray, step: float, max_iter: int, tol: float,
            stop_on_convergence: bool, scheme: str, beta_const: float | None) \
        -> OptimizationRun:
    if scheme not in {"constant", "dynamic"}:
        raise ValueError("scheme phải là 'constant' hoặc 'dynamic'")
    if scheme == "constant" and beta_const is None:
        raise ValueError("AGD momentum cố định cần beta_const")

    label = "AGD momentum cố định" if scheme == "constant" else "AGD momentum (k-2)/(k+1)"
    params = {"step": float(step)}
    if beta_const is not None:
        params["beta"] = float(beta_const)
    run = OptimizationRun(label, params, w0.copy())
    w = np.asarray(w0, dtype=float).copy()
    w_prev = w.copy()
    started = time.perf_counter()

    value = float(obj.value(w))
    grad0 = obj.grad(w)
    run.objective_evals += 1
    run.gradient_evals += 1
    grad_norm = float(np.linalg.norm(grad0))
    _record(run, 0, value, grad_norm, started)

    for iteration in range(1, max_iter + 1):
        # iteration=1 tương ứng k=2, nên beta_2=(2-2)/(2+1)=0.
        beta = float(beta_const) if scheme == "constant" else (iteration - 1.0) / (iteration + 2.0)
        momentum_point = w + beta * (w - w_prev)
        grad_momentum = obj.grad(momentum_point)
        run.gradient_evals += 1
        grad_norm = float(np.linalg.norm(grad_momentum))
        new_w = momentum_point - float(step) * grad_momentum
        value = float(obj.value(new_w))
        run.objective_evals += 1
        if not _safe(new_w, value, grad_momentum):
            run.finite = False
            run.note = "phân kỳ hoặc xuất hiện giá trị không hữu hạn"
            break

        w_prev, w = w, new_w
        run.steps.append(float(step))
        _record(run, iteration, value, grad_norm, started)

        # Chuẩn tại điểm momentum là đại lượng thuật toán đã tính. Chỉ khi nó
        # đạt tolerance mới kiểm chứng thêm chuẩn gradient tại nghiệm mới.
        if run.converged_at is None and grad_norm <= tol:
            exact_grad = obj.grad(w)
            run.gradient_evals += 1
            exact_norm = float(np.linalg.norm(exact_grad))
            if exact_norm <= tol:
                run.converged_at = iteration
                run.final_grad_norm = exact_norm
        if stop_on_convergence and run.converged_at is not None:
            break

    run.w = w
    # Khi dò tham số, vòng lặp vẫn chạy đủ 500 bước kể cả đã từng chạm tol.
    # Vì vậy phải đo lại gradient tại đúng nghiệm cuối, không giữ chuẩn ở lần
    # đầu chạm ngưỡng. Khi chạy chính thức và dừng ngay tại hội tụ thì phép đo
    # xác nhận phía trên đã là gradient của chính nghiệm trả về.
    stopped_at_convergence = bool(
        stop_on_convergence
        and run.converged_at is not None
        and run.iterations[-1] == run.converged_at
    )
    if not stopped_at_convergence:
        exact_grad = obj.grad(w)
        run.gradient_evals += 1
        run.final_grad_norm = float(np.linalg.norm(exact_grad))
    run.converged = run.converged_at is not None
    return run


def constant_momentum(L: float, mu: float) -> float:
    root = np.sqrt(float(L) / float(mu))
    return float((root - 1.0) / (root + 1.0))
