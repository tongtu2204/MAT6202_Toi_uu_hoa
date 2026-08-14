"""Thử-và-sai để chọn tham số — theo đúng lối trình bày của bài mẫu.

Cách làm ở đây cố ý CHÂN PHƯƠNG. Không có hàm chấm điểm, không có quy tắc xếp
hạng, không có bộ chọn tự động. Quy trình đúng như một người ngồi dò tay:

    1. Chạy một lưới THÔ, cách nhau theo bậc 10 (0.01, 0.1, 1, 10). Vẽ f(w_k) trên
       trục thẳng — ở tầng này ta chỉ cần thấy bậc nào phân kỳ, bậc nào bò chậm.
    2. Chạy một lưới TINH quanh giá trị thắng ở tầng 1 (3, 4, 5, 5.4, 5.5). Vẽ
       log(f - f*) — ở tầng này các đường mới tách nhau ra để so được.
    3. NHÌN hình rồi chọn. Giá trị chọn được viết tay vào driver kèm lý do.

Vì sao để người chọn chứ không để code chọn: tiêu chí thật sự là "đường nào vừa
xuống nhanh vừa không dao động", mà điều đó nhìn hình thì rõ còn viết thành một
con số thì luôn thiếu. Một bộ chọn tự động cũng che mất chuyện đáng nói nhất —
rằng bước tốt nhất nằm XA cận lý thuyết 2/L đến mức nào.

Chi phí: các lần chạy ở đây đều ngắn (10 vòng ở tầng thô, 100-150 vòng ở tầng
tinh) nên cả module này rẻ hơn nhiều so với việc chạy tới hội tụ.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .optimizers.base import OptResult


@dataclass
class Sweep:
    """Một tầng thử: chạy cùng một thuật toán ở nhiều giá trị tham số."""
    label: str                 # "GD — độ dài bước cố định"
    param: str                 # tên tham số, ví dụ "t"
    grid: list                 # các giá trị đã thử
    runs: dict                 # giá trị -> OptResult
    n_iter: int
    f_star: float

    def gaps(self, value):
        return np.maximum(np.asarray(self.runs[value].f_history) - self.f_star, 1e-16)

    def iters_to(self, value, tol: float) -> int:
        """Số vòng đầu tiên đạt f - f* < tol (-1 nếu không đạt).

        Với Newton thì đây mới là đại lượng phân biệt được các bước: mọi t hợp lý
        đều về tới độ chính xác máy, nên gap CUỐI giống hệt nhau và không nói gì.
        """
        g = np.asarray(self.runs[value].f_history) - self.f_star
        hit = np.flatnonzero(g < tol)
        return int(hit[0]) if hit.size else -1

    def table(self, tol: float = 1e-12) -> list[dict]:
        """Bảng gọn để in ra và ghi vào artifact."""
        out = []
        for v in self.grid:
            r = self.runs[v]
            h = np.asarray(r.f_history)
            out.append(dict(
                value=v,
                iters=len(h),
                iters_to_tol=self.iters_to(v, tol),
                f_last=float(h[-1]),
                gap=float(h[-1] - self.f_star),
                finite=bool(np.isfinite(h).all()),
                monotone=bool(np.all(np.diff(h) <= 1e-12)),
                time_s=float(r.time_s[-1]) if r.time_s else float("nan"),
            ))
        return out


def sweep(label: str, param: str, run_one: Callable[[object], OptResult],
          grid, n_iter: int, f_star: float, verbose: bool = True) -> Sweep:
    """Chạy `run_one(value)` cho từng giá trị trong `grid`. Không chọn gì cả."""
    runs = {}
    for v in grid:
        t0 = time.perf_counter()
        runs[v] = run_one(v)
        if verbose:
            h = np.asarray(runs[v].f_history)
            tag = "phân kỳ" if not np.isfinite(h).all() else f"gap={h[-1]-f_star:.3e}"
            print(f"    {param}={v:<10g} {len(h):>5d} vòng  {tag:<20s}"
                  f"{time.perf_counter()-t0:6.1f}s", flush=True)
    return Sweep(label, param, list(grid), runs, n_iter, f_star)


@dataclass
class Choice:
    """Kết quả một lần dò hai tầng, kèm giá trị đã chọn và lý do chọn."""
    label: str
    coarse: Sweep
    fine: Sweep
    chosen: float
    reason: str                # vì sao chọn — viết tay, hiện lên slide
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(label=self.label, param=self.coarse.param, chosen=self.chosen,
                    reason=self.reason,
                    coarse=dict(grid=self.coarse.grid, rows=self.coarse.table()),
                    fine=dict(grid=self.fine.grid, rows=self.fine.table()),
                    **self.extra)


# --------------------------------------------------------------------------- #
# Xác định L — bốn cách, để đối chiếu với nhau
# --------------------------------------------------------------------------- #
@dataclass
class LStudy:
    """L theo bốn đường khác nhau, trên cùng một bài toán."""
    lam: float
    L_formula: float           # (1/4)*lambda_max(X^T X / n) + lambda
    lambda_max_gram: float
    power_iters: list[float]   # lịch sử power iteration -> lambda_max
    L_local: float             # lambda_max(Hessian tại w*)
    L_empirical: float         # suy từ bước lớn nhất còn ổn định: L ~ 2/t_max
    t_max_stable: float
    t_grid: list[float]
    stable: list[bool]


def l_study(obj, w_star, t_grid, run_one, lam: float,
            power_iter: int = 40, seed: int = 0) -> LStudy:
    """Xác định hằng số trơn L bằng bốn cách và so chúng với nhau.

    1. CÔNG THỨC. Vì 0 < p(1-p) <= 1/4 nên
           grad^2 f = (1/n)X'SX + lambda*diag(r)  <=  [(1/4)lambda_max((1/n)X'X) + lambda] I
       Đây là cận trên đúng với MỌI w — an toàn nhưng bi quan.
    2. POWER ITERATION. Cùng lambda_max đó nhưng tính bằng phép lặp
           v <- (X'(Xv)) / ||.||,
       chỉ tốn O(nd) mỗi vòng thay vì O(nd^2 + d^3) như eigvalsh. Ở d=415 đây là
       cách duy nhất còn rẻ khi d lớn hơn nữa, và nó hội tụ trong vài chục vòng.
    3. ĐỘ CONG CỤC BỘ tại nghiệm: lambda_max(grad^2 f(w*)). Nhỏ hơn (1) vì tại w*
       thì p(1-p) đã rời xa 1/4.
    4. THỰC NGHIỆM: bước cố định lớn nhất mà GD còn giảm đơn điệu. Lý thuyết nói
       ngưỡng là 2/L, nên bước lớn nhất còn ổn định cho ta L_emp ~ 2/t_max.

    Khoảng cách giữa (1) và (4) chính là lý do bài mẫu dò tay ra bước 5.4 trong khi
    1/L_công_thức nhỏ hơn thế rất nhiều.
    """
    X = obj.X
    n = X.shape[0]

    # (1) + (2)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    hist = []
    for _ in range(power_iter):
        u = X.T @ (X @ v) / n
        nv = float(np.linalg.norm(u))
        v = u / nv
        hist.append(nv)
    lambda_max = hist[-1]
    L_formula = 0.25 * lambda_max + lam

    # (3)
    L_local = float(np.linalg.eigvalsh(obj.hessian(w_star))[-1])

    # (4)
    stable = []
    for t in t_grid:
        h = np.asarray(run_one(t).f_history)
        ok = bool(np.isfinite(h).all() and np.all(np.diff(h) <= 1e-9))
        stable.append(ok)
        print(f"    t={t:<8g} {'giảm đơn điệu' if ok else 'KHÔNG đơn điệu'}", flush=True)
    t_max = max((t for t, ok in zip(t_grid, stable) if ok), default=float("nan"))

    return LStudy(lam=lam, L_formula=L_formula, lambda_max_gram=lambda_max,
                  power_iters=hist, L_local=L_local,
                  L_empirical=2.0 / t_max if t_max == t_max else float("nan"),
                  t_max_stable=t_max, t_grid=list(t_grid), stable=stable)


# --------------------------------------------------------------------------- #
# SGD: dò lịch bước theo từng cỡ mini-batch (bố cục của bài mẫu)
# --------------------------------------------------------------------------- #
SCHEDULES = {
    "hằng":        lambda a, k, m: a,
    "a/k":         lambda a, k, m: a / (k + 1),
    "a/sqrt(k)":   lambda a, k, m: a / np.sqrt(k + 1),
    "a/(k//m)":    lambda a, k, m: a / (1 + k // m),
    "a/k + reset": lambda a, k, m: a / (1 + (k % m)),
}


def sgd_steps(obj, w0, eta0, batch, n_steps, schedule="hằng", m=100,
              seed=0, log_every=5):
    """SGD ghi log theo BƯỚC mini-batch (không theo epoch), để vẽ đúng như bài mẫu.

    `sgd` trong optim/optimizers ghi mỗi epoch — hợp cho money plot nhưng ở đây
    ta cần trục "Số bước" như bài mẫu, và cần thử cả họ lịch bước:

        hằng        eta_k = a              (không hội tụ, dừng ở sàn phương sai)
        a/k         eta_k = a/(k+1)        (tắt quá nhanh: sum eta_k hữu hạn thì kẹt)
        a/sqrt(k)   eta_k = a/sqrt(k+1)    (Robbins-Monro: sum=inf, sum^2<inf)
        a/(k//m)    bậc thang, giảm mỗi m bước
        a/k + reset bậc thang có đặt lại — để thấy mỗi lần reset là một cú nhảy

    Hàm mục tiêu ĐẦY ĐỦ được tính mỗi `log_every` bước (một matvec O(nd)), nên đây
    là phần tốn nhất; log thưa ra để rẻ mà đường vẫn mượt.
    """
    from .objective import sigmoid
    from .optimizers.base import Recorder

    rec = Recorder(f"SGD b={batch} {schedule}")
    rng = np.random.default_rng(seed)
    step_fn = SCHEDULES[schedule]
    w = w0.copy()
    n = obj.n
    rec.tick(grad=0.0)
    rec.log(obj.value(w), 0.0)
    for k in range(n_steps):
        idx = rng.integers(0, n, size=batch)
        Xb = obj.X[idx]
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - obj.y[idx]) / batch + obj.lam * obj._reg * w
        eta = step_fn(eta0, k, m)
        w = w - eta * g
        rec.tick(grad=batch / n)
        rec.res.steps.append(float(eta))
        if (k + 1) % log_every == 0:
            rec.log(obj.value(w), 0.0)
    return rec.finish(w, False)
