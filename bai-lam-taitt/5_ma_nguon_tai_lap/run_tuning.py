#!/usr/bin/env python3
"""Thử-và-sai chọn tham số cho từng thuật toán, theo lối trình bày của bài mẫu.

    python run_tuning.py --list
    python run_tuning.py L gd-fixed gd-bt gd-accel newton-fixed newton-bt sgd

Mỗi mục chạy hai tầng — lưới THÔ theo bậc 10, rồi lưới TINH quanh giá trị thắng —
vẽ ra một hình hai panel, và ghi số vào artifacts/tuning.json.

GIÁ TRỊ ĐƯỢC CHỌN NẰM NGAY TRONG FILE NÀY, viết tay, kèm lý do. Đó là chủ ý: tiêu
chí thật sự là "đường nào vừa xuống nhanh vừa không dao động", nhìn hình thì rõ mà
viết thành công thức chấm điểm thì luôn thiếu. Muốn đổi thì sửa hằng số CHON_* bên
dưới rồi chạy lại — hình và slide sẽ tự khớp.

Chi phí: các lần chạy đều ngắn (10 vòng tầng thô, 100-150 vòng tầng tinh), cả file
này chạy khoảng 15-20 phút ở n=75.026, d=415.
"""
from __future__ import annotations

import blas_threads  # phải đứng trước numpy: ghim BLAS về 1 luồng
import argparse
import json
import time
from pathlib import Path

import numpy as np

from churn_opt.diagnostics import conditioning
from optim import plots
from optim.data import load_variant
from optim.objective import LogisticObjective
from optim.optimizers import (gradient_descent, accelerated_gd, newton, sgd,
                              adaptive, ista, fista, subgradient)
from optim.reference import newton_reference
from optim.tuning import SCHEDULES, Choice, l_study, sgd_steps, sweep

ART = Path(__file__).resolve().parent / "artifacts"
OUT = ART / "tuning.json"

# ----------------------------------------------------------------------------- #
# CÁC GIÁ TRỊ ĐÃ CHỌN — đọc từ hình, viết tay, kèm lý do hiện lên slide
# ----------------------------------------------------------------------------- #
# Vòng 1 đã chạy với lưới tinh đặt sai hướng (thô chọn t=1 nhưng lưới tinh lại dò
# LÊN 1->10). Vòng 2 dò quanh giá trị thô mới ra kết quả dùng được. Ghi lại ở đây
# vì đó chính là quy trình: lưới thô chỉ ra vùng, lưới tinh phải bao quanh vùng đó.
CHON_GD_FIXED = (0.4, "gap nhỏ nhất sau 150 vòng (4,5e-3); t=1 chậm hơn 30 lần, "
                      "t=10 phân kỳ. Đáng chú ý: 0,4 gấp 2,8 lần ngưỡng đơn điệu "
                      "2/L=0,142 — GD dao động nhưng về đích nhanh hơn.")
CHON_AGD_K = (0.3, "ở 1000 vòng: 0,1 cho 5,8e-7; 0,2 cho 1,6e-7; 0,3 cho 5,6e-8 "
                   "rồi 0,5 SẬP xuống 1,1e-2. Cực tiểu nội trong lưới, nhưng là giá "
                   "trị lớn nhất TRƯỚC vách sập — tối ưu thật nằm đâu đó trong "
                   "(0,3; 0,5) mà lưới này không tách được. Lược đồ không ràng buộc "
                   "t với beta nên t là tham số tự do thật sự.")
CHON_GD_ACCEL = (0.3, "ĐỔI theo ngân sách: ở 1000 vòng 0,3 cho 2,3e-15, hơn 0,2 "
                      "(4,2e-13) 180 lần, còn 0,5 sập xuống 1,6e-2. Ở 150 vòng thì "
                      "0,2 mới thắng — bước tốt nhất phụ thuộc ngân sách, và ta chọn "
                      "theo 1000 vòng vì money plot cùng mục 4.11/4.12 chạy ở đó. "
                      "Trùng đúng bước của lược đồ (k-2)/(k+1), nên mục 4.11 so hai "
                      "lược đồ ở CÙNG một t — chỉ momentum khác nhau.")
CHON_NEWTON_FIXED = (1.0, "mọi t trong [0,8; 1,2] đều về tới độ chính xác máy, nên "
                          "phải so bằng SỐ VÒNG: t=1 (pure Newton) ít vòng nhất. "
                          "Damping không giúp gì khi khởi tạo gần nghiệm.")
# Hai trục cho HAI câu trả lời khác nhau ở GD — đây là điều đáng nói nhất của lưới này.
CHON_BT_GD = ((0.8, 0.5),
              "1000 vòng, 9 cấu hình. Chọn theo ĐỘ CHÍNH XÁC: rho=0,8;c=0,5 cho 1,2e-6, "
              "tốt nhất lưới — hơn (0,5;0,5) 4,2 lần và hơn rho=0,2 tới 115 lần. Giá "
              "phải trả là 180,1s so với 81,7s, vì rho lớn nghĩa là lùi bước tinh hơn "
              "nên tốn nhiều lần tính f hơn mỗi vòng. Hai trục vẫn cho hai câu trả lời "
              "(theo thời gian thì rho=0,2 thắng với 58,9s) — đây là biên Pareto thật, "
              "rho điều khiển cả hai trục cùng lúc còn c gần như vô can. Ta chọn trục "
              "độ chính xác vì mục 4.12 so các thuật toán ở CÙNG số vòng, nên tham số "
              "phải là tham số cho nghiệm tốt nhất, không phải cho nghiệm rẻ nhất. "
              "Góc tệ nhất vẫn là rho=0,8;c=0,2 — vừa chậm nhất 192,3s vừa kém nhất.")
# Cấu hình SGD dùng cho money plot (mục 4.10). Chọn tay như mọi tham số khác.
CHON_SGD = dict(
    batch=1024, schedule="hằng", eta0=0.2,
    reason="Theo SỐ BƯỚC thì b=16384 tốt nhất (3,1e-4 so với 3,3e-3), nhưng money "
           "plot đọc theo TRỤC THỜI GIAN: b=1024 đạt ~2e-3 sau 4,5s trong khi "
           "b=16384 cần 20s mới tới cùng mức. Lịch hằng vì với batch lớn thì mọi "
           "lịch giảm dần đều tệ hơn trong ngân sách này (mục 4.8).")

CHON_L1_SUB = (2.0, "cực tiểu nội: 1,0 cho 9,1e-3 và 3,0 cho 1,2e-2, còn 2,0 cho 2,6e-3")
CHON_L1_FISTA = (4.0, "dò RIÊNG cho FISTA, KHÔNG dùng chung 7/L của ISTA. Lưới tinh 600 "
                      "vòng: 2 cho 1,17e-6; 3 cho 4,26e-7; 4 cho 2,06e-7; 5 cho 2,72e-7; "
                      "rồi 6 SẬP xuống 8,15e-3 (346 hệ số khác 0 thay vì 125). Cực tiểu "
                      "nội tại t=4, và vách sập nằm giữa 5 và 6 — tức trần bước của FISTA "
                      "THẤP HƠN của ISTA (>=7): momentum tích lũy khuếch đại phần vượt "
                      "ngưỡng mà ISTA thuần hấp thụ được, đúng kiểu AGD mong manh hơn GD "
                      "ở mục 4.12.")
CHON_L1_ISTA = (7.0, "phải dò BA vòng mới tới: 2,5 rồi 7 đều chạm mép trên. Cận lý thuyết của proximal là t<=1/L, nhưng bước tốt nhất là 7/L — bi quan gấp 7 lần, cùng kiểu với GD ở mục 4.3.")
CHON_BT_NEWTON = ((0.5, 0.2),
                  "Khác GD, ở đây hai trục cho CÙNG một câu trả lời: rho=0,5;c=0,2 về "
                  "tới độ chính xác máy sau ~8 vòng / 5,5s, nhanh nhất trên cả hai. "
                  "c nhỏ = điều kiện Armijo dễ thỏa = giữ được bước gần 1, tức gần "
                  "pure Newton — đúng thứ ta muốn khi đã ở gần nghiệm.")


def _merge(key, value):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = value
    OUT.write_text(json.dumps(d, indent=1, default=float))
    print(f"  -> {OUT.name}[{key}]", flush=True)


def _setup(lam=1e-3, variant="ridge"):
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    w0 = np.zeros(obj.d)
    ref = newton_reference(obj, w0, mu=lam)
    cond = conditioning(obj.X, lam)
    return obj, w0, ref, cond


# ----------------------------------------------------------------------------- #
def do_L(a):
    """Xác định L bằng bốn cách và đối chiếu."""
    obj, w0, ref, cond = _setup(a.lam)
    print(f"  L theo công thức (conditioning) = {cond['L']:.6g}")
    # Lưới TRUNG TÍNH: cố ý KHÔNG chứa sẵn giá trị 2/L, để ngưỡng đo được rơi vào
    # một khoảng chứ không trùng khít một điểm ta đã biết trước — nếu không thì
    # kết quả trông như đã gài sẵn.
    t_grid = [0.05, 0.1, 0.12, 0.14, 0.16, 0.2, 0.5, 1.0, 5.0, 20.0]
    print("  Thử bước cố định, xem bước nào GD còn giảm đơn điệu (200 vòng):")
    ls = l_study(obj, ref.w_star, t_grid,
                 lambda t: gradient_descent(obj, w0, eta=t, max_iter=200, tol=0.0),
                 lam=a.lam)
    print(f"\n  (1) công thức        L = {ls.L_formula:.6g}   (1/L = {1/ls.L_formula:.4g})")
    print(f"  (2) power iteration  lambda_max = {ls.lambda_max_gram:.6g} "
          f"sau {len(ls.power_iters)} vòng")
    print(f"  (3) cục bộ tại w*    L = {ls.L_local:.6g}   (1/L = {1/ls.L_local:.4g})")
    print(f"  (4) thực nghiệm      t_max = {ls.t_max_stable:g} "
          f"-> L ~ {ls.L_empirical:.6g}")
    print(f"  => bước ổn định lớn nhất gấp {ls.t_max_stable*ls.L_formula/2:.1f} lần "
          f"ngưỡng 2/L của công thức")
    print("  hình:", plots.l_study_plot(ls))
    _merge("L", dict(L_formula=ls.L_formula, lambda_max_gram=ls.lambda_max_gram,
                     L_local=ls.L_local, L_empirical=ls.L_empirical,
                     t_max_stable=ls.t_max_stable, t_grid=ls.t_grid,
                     stable=ls.stable, power_iters=ls.power_iters[-5:],
                     n_power_iters=len(ls.power_iters)))


# Lưới THÔ dùng chung cho mọi thuật toán có "bước cố định": cùng một lưới thì
# bốn mục 4.2/4.6/4.9/4.10 so được trực tiếp với nhau, thay vì mỗi mục một lưới.
# Trải từ 0,01 tới 5 nên bao được cả 1/L=0,071, 2/L=0,142 lẫn t=1 của Newton.
COARSE_T = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
COARSE_ITER = 20


def _two_stage(label, param, run_one, coarse, fine, f_star, chosen, reason, fname,
               fine_iter=150, coarse_iter=10):
    print(f"  [{label}] lưới THÔ {coarse} ({coarse_iter} vòng)")
    c = sweep(label, param, lambda v: run_one(v, coarse_iter), coarse, coarse_iter, f_star)
    print(f"  [{label}] lưới TINH {fine} ({fine_iter} vòng)")
    f = sweep(label, param, lambda v: run_one(v, fine_iter), fine, fine_iter, f_star)
    if chosen is None:
        print(f"  !! chưa chọn {param} — xem hình rồi điền vào CHON_* trong run_tuning.py")
        chosen = fine[len(fine) // 2]
        reason = "TẠM THỜI: chưa đọc hình"
    ch = Choice(label, c, f, chosen, reason)
    print("  hình:", plots.choice_plot(ch, fname=fname))
    return ch


def do_gd_fixed(a):
    obj, w0, ref, cond = _setup(a.lam)
    ch = _two_stage(
        "GD — bước cố định", "t",
        lambda t, k: gradient_descent(obj, w0, eta=t, max_iter=k, tol=0.0),
        coarse=COARSE_T, fine=[0.2, 0.4, 0.7, 1.0, 1.5], coarse_iter=COARSE_ITER,
        f_star=ref.f_star, chosen=CHON_GD_FIXED[0], reason=CHON_GD_FIXED[1],
        fname="tune_gd_fixed.png")
    _merge("gd_fixed", ch.summary())


def do_gd_accel(a):
    obj, w0, ref, cond = _setup(a.lam)
    mu = max(cond["mu"], a.lam)
    ch = _two_stage(
        "AGD — bước cố định", "t",
        lambda t, k: accelerated_gd(obj, w0, L=1.0 / t, mu=mu, max_iter=k, tol=0.0),
        coarse=COARSE_T, fine=[0.1, 0.2, 0.3, 0.5, 0.8], coarse_iter=COARSE_ITER,
        f_star=ref.f_star, chosen=CHON_GD_ACCEL[0], reason=CHON_GD_ACCEL[1],
        fname="tune_gd_accel.png",
        # 1000 vòng thay vì 150: ở 150 vòng AGD còn chưa qua pha khởi động nên
        # lưới tinh xếp hạng theo một giai đoạn không đại diện. Bước chọn ở đây
        # được money plot và mục 4.11/4.12 dùng lại, nên nó phải đúng ở ngân sách
        # mà các mục đó chạy.
        fine_iter=1000)
    _merge("gd_accel", ch.summary())


def do_agd_k(a):
    """Lược đồ momentum (k-2)/(k+1) — dò bước, rồi so với AGD beta-hằng."""
    obj, w0, ref, cond = _setup(a.lam)
    L, mu = cond["L"], max(cond["mu"], a.lam)

    # Lưới TINH đặt đúng bằng lưới của mục AGD beta-hằng, để hai bảng so được
    # trực tiếp chứ không phải so hai lưới khác nhau.
    ch = _two_stage(
        "AGD (k-2)/(k+1) — bước cố định", "t",
        lambda t_, k: accelerated_gd(obj, w0, L=L, mu=mu, t=t_, scheme="k",
                                     max_iter=k, tol=0.0),
        coarse=COARSE_T, fine=[0.1, 0.2, 0.3, 0.5, 0.8], coarse_iter=COARSE_ITER,
        f_star=ref.f_star, chosen=CHON_AGD_K[0], reason=CHON_AGD_K[1],
        fname="tune_agd_k.png",
        # 1000 vòng: ở 150 vòng lược đồ này còn đang khởi động momentum nên các
        # giá trị t chưa tách nhau đủ để kết luận.
        fine_iter=1000)
    _merge("agd_k", ch.summary())

    # Đối đầu ở ngân sách dài: mỗi lược đồ chạy ở cấu hình tốt nhất CỦA NÓ.
    print("  [đối đầu] 3 cấu hình, trần 3000 vòng ...", flush=True)
    runs = {
        r"$\beta$ hằng, bước $1/L$ (bản dùng ở money plot)":
            accelerated_gd(obj, w0, L=L, mu=mu, max_iter=3000, tol=1e-10),
        rf"$\beta$ hằng, cặp $(t,\beta)$ dò tay $t={CHON_GD_ACCEL[0]:g}$":
            accelerated_gd(obj, w0, L=1.0 / CHON_GD_ACCEL[0], mu=mu,
                           max_iter=3000, tol=1e-10),
        rf"$(k-2)/(k+1)$, $t={CHON_AGD_K[0]:g}$":
            accelerated_gd(obj, w0, L=L, mu=mu, t=CHON_AGD_K[0], scheme="k",
                           max_iter=3000, tol=1e-10),
    }
    rows = {}
    for label, r in runs.items():
        gap = float(r.f_history[-1] - ref.f_star)
        rows[label] = dict(iters=len(r.f_history) - 1, gap=gap,
                           time_s=float(r.time_s[-1]), converged=bool(r.converged))
        print(f"    {len(r.f_history)-1:>5d} vòng  gap={gap:.3e}  "
              f"{r.time_s[-1]:6.1f}s  hội tụ={r.converged}")
    # Hình của mục 4.11 chỉ vẽ hai đường DÒ TAY. Đường bước 1/L vẫn được chạy vì
    # mục 5.2 trích số của nó, nhưng nó không được lên hình ở mục 4: cả mục 4 là
    # "dò tay, chưa dùng lý thuyết", mà 1/L thì cần L — thứ tới mục 5 mới định nghĩa.
    ve = {k: v for k, v in runs.items() if "1/L" not in k}
    assert len(ve) == 2, sorted(ve)
    # Điểm giao được ĐO, không đọc bằng mắt. Phải lấy lần vượt CUỐI CÙNG: hai đường
    # cắt nhau vài lần trong ~15 vòng đầu lúc momentum còn đang khởi động, nên "lần
    # vượt đầu tiên" trả về vòng 1 và không nói lên điều gì.
    import numpy as _np
    a, b = [_np.asarray(r.f_history) - ref.f_star for r in ve.values()]
    m = min(len(a), len(b))
    behind = _np.flatnonzero(a[:m] >= b[:m])
    cross = int(behind[-1]) + 1 if behind.size and behind[-1] + 1 < m else -1
    print(f"    beta hằng vượt hẳn từ vòng {cross} (và không bị cắt lại)")
    _np.savez_compressed(ART / "agd_schemes_ridge.npz", f_star=ref.f_star,
                         labels=_np.array(list(ve), dtype=object),
                         **{f"h_{i}": _np.asarray(r.f_history) for i, r in enumerate(ve.values())},
                         **{f"t_{i}": _np.asarray(r.time_s) for i, r in enumerate(ve.values())})
    _merge("agd_schemes_cross", dict(cross_iter=cross,
                                     labels=[k for k in ve], n_compared=m))
    print("  hình:", plots.agd_schemes_plot(ve, ref.f_star))
    _merge("agd_schemes", dict(f_star=ref.f_star, L=L, mu=mu, rows=rows))


def do_money7(a):
    """Money plot: BẢY cấu hình đã dò tay, không cấu hình nào lấy từ nơi khác.

    Khác `run_stage.py money` ở chỗ đó. Bản cũ vẽ bốn đường, trong đó Newton là
    lần chạy sinh ra f* — tức chạy ở tham số MẶC ĐỊNH của thư viện, không phải
    tham số dò tay. Ở đây f* vẫn do lần chạy ấy sinh ra (mục 4.2 của deck trình
    bày riêng), nhưng nó KHÔNG còn là một đường trên hình: bảy đường ở đây đều
    đọc tham số từ tuning.json.
    """
    obj, w0, ref, cond = _setup(a.lam)
    mu = max(cond["mu"], a.lam)
    L = cond["L"]
    TOL, CAP = 1e-10, 3000                # cùng một tiêu chí dừng cho mọi đường

    cfg = [
        (rf"Newton --- bước cố định $t={CHON_NEWTON_FIXED[0]:g}$",
         lambda: newton(obj, w0, max_iter=100, tol=TOL, line_search=False,
                        t0=CHON_NEWTON_FIXED[0], strict=False)),
        (rf"Newton --- backtracking $\rho={CHON_BT_NEWTON[0][0]:g}$, $c={CHON_BT_NEWTON[0][1]:g}$",
         lambda: newton(obj, w0, max_iter=100, tol=TOL, line_search=True, t0=2.0,
                        strict=False, rho=CHON_BT_NEWTON[0][0], c=CHON_BT_NEWTON[0][1])),
        (rf"GD --- bước cố định $t={CHON_GD_FIXED[0]:g}$",
         lambda: gradient_descent(obj, w0, eta=CHON_GD_FIXED[0], max_iter=CAP, tol=TOL)),
        (rf"GD --- backtracking $\rho={CHON_BT_GD[0][0]:g}$, $c={CHON_BT_GD[0][1]:g}$",
         lambda: gradient_descent(obj, w0, eta=20.0, max_iter=CAP, tol=TOL,
                                  line_search=True, rho=CHON_BT_GD[0][0], c=CHON_BT_GD[0][1])),
        (rf"AGD --- $\beta$ hằng, $t={CHON_GD_ACCEL[0]:g}$",
         lambda: accelerated_gd(obj, w0, L=1.0 / CHON_GD_ACCEL[0], mu=mu,
                                max_iter=CAP, tol=TOL)),
        (rf"AGD --- $\beta_k=\frac{{k-2}}{{k+1}}$, $t={CHON_AGD_K[0]:g}$",
         lambda: accelerated_gd(obj, w0, L=L, mu=mu, t=CHON_AGD_K[0], scheme="k",
                                max_iter=CAP, tol=TOL)),
        (rf"SGD --- $\eta_0={CHON_SGD['eta0']:g}$, $b={CHON_SGD['batch']}$",
         lambda: sgd(obj, w0, eta0=CHON_SGD["eta0"], batch_size=CHON_SGD["batch"],
                     epochs=50, gamma=0.0, seed=0)),
    ]

    runs, rows = {}, {}
    for label, fn in cfg:
        t0 = time.perf_counter()
        r = fn(); runs[label] = r
        gap = float(r.f_history[-1] - ref.f_star)
        rows[label] = dict(iters=len(r.f_history) - 1, gap=gap,
                           time_s=float(r.time_s[-1]), converged=bool(r.converged))
        print(f"    {len(r.f_history)-1:>5d} vòng  gap={gap:11.3e}  "
              f"{r.time_s[-1]:7.1f}s  hội tụ={str(r.converged):5s}  {label[:46]}",
              flush=True)
    print("  hình:", plots.money7_plot(runs, ref.f_star, fname="money7_iter.png", x="iter"))
    print("  hình:", plots.money7_plot(runs, ref.f_star, fname="money7_time.png", x="time"))
    _merge("money7", dict(f_star=ref.f_star, grad_norm=ref.grad_norm,
                          bound=ref.bound, mu_star=ref.mu, tol=TOL, cap=CAP, rows=rows))


def do_newton_fixed(a):
    obj, w0, ref, cond = _setup(a.lam)
    ch = _two_stage(
        "Newton — bước cố định", "t",
        # strict=False là bắt buộc ở đây: lưới thô có t=2 và t=5, pure Newton ở
        # bước đó vọt đi rất xa, S=diag(p(1-p)) chạm đáy và Cholesky hỏng thật.
        # Đó là KẾT QUẢ của lưới thô ("bước này không dùng được"), không phải lỗi.
        lambda t, k: newton(obj, w0, max_iter=min(k, 40), tol=0.0,
                            line_search=False, t0=t, strict=False),
        coarse=COARSE_T, fine=[0.8, 0.9, 1.0, 1.1, 1.2], coarse_iter=COARSE_ITER,
        f_star=ref.f_star, chosen=CHON_NEWTON_FIXED[0], reason=CHON_NEWTON_FIXED[1],
        fname="tune_newton_fixed.png")
    _merge("newton_fixed", ch.summary())


def _bt_grid(label, run_one, t_init, f_star, chosen, fname, tag):
    rhos, cs = (0.2, 0.5, 0.8), (0.2, 0.5, 0.8)
    runs = {}
    for rho in rhos:
        for c in cs:
            t0 = time.perf_counter()
            runs[(rho, c)] = run_one(rho, c)
            r = runs[(rho, c)]
            print(f"    rho={rho} c={c}  {len(r.f_history):>4d} vòng  "
                  f"gap={r.f_history[-1]-f_star:.3e}  "
                  f"bước TB={np.mean(r.steps) if r.steps else float('nan'):.3g}  "
                  f"{time.perf_counter()-t0:5.1f}s", flush=True)
    if chosen is None:
        chosen = (0.5, 0.5)
        print("  !! chưa chọn (rho,c) — xem hình rồi điền CHON_BT_* ")
    print("  hình:", plots.bt_grid_plot(runs, f_star,
                                        f"{label} — backtracking (t_init={t_init:g})",
                                        chosen, fname=fname))
    print("  hình:", plots.step_trace_plot(
        runs[chosen], f"{label}: độ dài bước backtracking chọn ở mỗi vòng "
                      rf"($\rho$={chosen[0]:g}, c={chosen[1]:g})",
        fname=f"tune_step_trace_{tag}.png"))
    # Chỉ có phần TÔ ĐẬM trong hình và hình vết bước là phụ thuộc `chosen`, còn nội
    # dung chín lần chạy thì không — mà chạy lại chín lần tốn ~17 phút. Lưu lại để
    # đổi lựa chọn sau này chỉ tốn vài giây (xem replot_bt_grid.py).
    np.savez_compressed(ART / f"bt_grid_{tag}.npz", f_star=f_star, t_init=t_init,
                        keys=np.array([f"{r},{c}" for (r, c) in sorted(runs)]),
                        **{f"h_{i}": np.asarray(runs[k].f_history)
                           for i, k in enumerate(sorted(runs))},
                        **{f"t_{i}": np.asarray(runs[k].time_s)
                           for i, k in enumerate(sorted(runs))},
                        **{f"s_{i}": np.asarray(runs[k].steps)
                           for i, k in enumerate(sorted(runs))})
    rows = [dict(rho=r, c=c, iters=len(v.f_history),
                 gap=float(v.f_history[-1] - f_star),
                 time=float(v.time_s[-1]),
                 mean_step=float(np.mean(v.steps)) if v.steps else float("nan"),
                 max_step=float(np.max(v.steps)) if v.steps else float("nan"))
            for (r, c), v in sorted(runs.items())]
    return runs, dict(t_init=t_init, chosen=list(chosen), rows=rows)


def do_gd_bt(a):
    obj, w0, ref, cond = _setup(a.lam)
    t_init = 20.0
    runs, summary = _bt_grid(
        # 1000 vòng, không phải 150: ở 150 vòng trục "số bước" và trục "thời gian"
        # cho hai câu trả lời khác nhau, phần lớn vì cả chín cấu hình còn chưa tách
        # nhau đủ theo độ chính xác. Ngân sách dài hơn cho một câu trả lời duy nhất.
        "GD", lambda rho, c: gradient_descent(obj, w0, eta=t_init, max_iter=1000,
                                              tol=0.0, line_search=True,
                                              rho=rho, c=c),
        t_init, ref.f_star, CHON_BT_GD[0], "tune_gd_bt_grid.png", "gd")
    summary["reason"] = CHON_BT_GD[1]
    _merge("gd_backtracking", summary)


def do_newton_bt(a):
    obj, w0, ref, cond = _setup(a.lam)
    t_init = 2.0
    runs, summary = _bt_grid(
        "Newton", lambda rho, c: newton(obj, w0, max_iter=40, tol=0.0,
                                        line_search=True, t0=t_init,
                                        strict=False, rho=rho, c=c),
        t_init, ref.f_star, CHON_BT_NEWTON[0], "tune_newton_bt_grid.png", "newton")
    summary["reason"] = CHON_BT_NEWTON[1]
    _merge("newton_backtracking", summary)


def do_compare(a):
    """Chốt lại: với CÙNG một thuật toán, ba cách chọn bước hơn kém nhau thế nào."""
    obj, w0, ref, cond = _setup(a.lam)
    mu = max(cond["mu"], a.lam)
    # 1000 vòng, không phải 150: ở 150 vòng lược đồ (k-2)/(k+1) còn đang ở pha
    # khởi động momentum, nên so ở đó là so nửa chừng. GD backtracking là cấu hình
    # đắt nhất ở đây (~3,8 lần tính f mỗi vòng).
    k = 1000

    gd = {
        f"Bước cố định (t={CHON_GD_FIXED[0]:g})":
            gradient_descent(obj, w0, eta=CHON_GD_FIXED[0], max_iter=k, tol=0.0),
        (f"Backtracking (t0=20, "
         rf"$\rho$={CHON_BT_GD[0][0]:g}, c={CHON_BT_GD[0][1]:g})"):
            gradient_descent(obj, w0, eta=20.0, max_iter=k, tol=0.0, line_search=True,
                             rho=CHON_BT_GD[0][0], c=CHON_BT_GD[0][1]),
        rf"AGD --- $\beta$ hằng (t={CHON_GD_ACCEL[0]:g})":
            accelerated_gd(obj, w0, L=1.0 / CHON_GD_ACCEL[0], mu=mu, max_iter=k, tol=0.0),
        rf"AGD --- $\beta_k=\frac{{k-2}}{{k+1}}$ (t={CHON_AGD_K[0]:g})":
            accelerated_gd(obj, w0, L=cond["L"], mu=mu, t=CHON_AGD_K[0], scheme="k",
                           max_iter=k, tol=0.0),
    }
    print("  hình:", plots.variants_plot(
        gd, ref.f_star, f"GD và AGD: bốn cách chọn bước ({k} vòng)",
        fname="tune_cmp_gd.png"))

    nt = {
        "Pure Newton (t=1)":
            newton(obj, w0, max_iter=40, tol=0.0, line_search=False, t0=1.0, strict=False),
        f"Damped — bước cố định (t={CHON_NEWTON_FIXED[0]:g})":
            newton(obj, w0, max_iter=40, tol=0.0, line_search=False,
                   t0=CHON_NEWTON_FIXED[0], strict=False),
        (rf"Damped — backtracking ($\rho$={CHON_BT_NEWTON[0][0]:g}, "
         f"c={CHON_BT_NEWTON[0][1]:g})"):
            newton(obj, w0, max_iter=40, tol=0.0, line_search=True, t0=2.0,
                   strict=False, rho=CHON_BT_NEWTON[0][0], c=CHON_BT_NEWTON[0][1]),
    }
    print("  hình:", plots.variants_plot(
        nt, ref.f_star, "Newton: ba cách chọn độ dài bước", fname="tune_cmp_newton.png"))

    def rows(runs):
        return {name: dict(iters=len(r.f_history),
                           gap=float(r.f_history[-1] - ref.f_star),
                           time=float(r.time_s[-1]),
                           fevals=int(r.n_grad[-1])) for name, r in runs.items()}
    for name, r in {**gd, **nt}.items():
        print(f"    {name:<46s} {len(r.f_history):>4d} vòng  "
              f"gap={r.f_history[-1]-ref.f_star:.3e}  t={r.time_s[-1]:.1f}s")
    _merge("compare", dict(gd=rows(gd), newton=rows(nt), f_star=ref.f_star))


# lưới TINH của từng thuật toán, điền sau khi nhìn lưới thô (vòng 1)
# Vòng 1 đặt lưới tinh theo kết quả cũ ở 3000 vòng; ba thuật toán chạm mép TRÊN,
# vì ở ngân sách 150 vòng thì bước tối ưu lớn hơn hẳn. Vòng 2 nới lên.
FINE_ADA = {"AdaGrad": [0.05, 0.1, 0.2, 0.3, 0.5],
            "RMSprop": [0.01, 0.02, 0.03, 0.05, 0.08],
            "Adam":    [0.03, 0.05, 0.08, 0.12, 0.2],
            "AdamW":   [0.05, 0.08, 0.12, 0.2, 0.3],
            "AMSGrad": [0.05, 0.1, 0.2, 0.3, 0.5]}
CHON_ADA = {"AdaGrad": (0.3, "cực tiểu nội: 0,2 và 0,5 đều tệ hơn"),
            "RMSprop": (0.02, "cực tiểu nội; nhưng gap 5,1e-2 vẫn tệ nhất cả họ"),
            "Adam":    (0.12, "cực tiểu nội sau khi nới lưới lên"),
            "AdamW":   (0.08, "cực tiểu nội"),
            "AMSGrad": (0.2, "cực tiểu nội, và là kết quả tốt nhất cả họ")}


def do_adaptive(a):
    """Dò eta cho cả họ Ada/Adam, mỗi thuật toán một hình hai tầng."""
    obj, w0, ref, cond = _setup(a.lam)
    out = {}
    for m in ("AdaGrad", "RMSprop", "Adam", "AdamW", "AMSGrad"):
        ch = _two_stage(
            f"{m}", "eta",
            lambda e, k, m=m: adaptive(obj, w0, method=m, eta=e, max_iter=k, tol=0.0),
            coarse=[0.001, 0.01, 0.1, 1.0], fine=FINE_ADA[m],
            f_star=ref.f_star, chosen=CHON_ADA.get(m, (None, ""))[0],
            reason=CHON_ADA.get(m, (None, ""))[1],
            fname=f"tune_ada_{m.lower()}.png")
        out[m] = ch.summary()
    _merge("adaptive", out)


def do_l1(a):
    """Dò bước cho họ L1: subgradient (eta0) và proximal (bội của 1/L)."""
    from optim.benchmark import _smooth_lipschitz
    ds = load_variant("lasso")
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=0.0)
    w0 = np.zeros(obj.d)
    L = _smooth_lipschitz(obj)
    alpha = 1e-3
    # f* của bài composite: giá trị nhỏ nhất mọi lần chạy đạt được (không lồi mạnh
    # nên không có chứng chỉ) — lấy từ một lần FISTA sâu.
    deep = fista(obj, w0, alpha=alpha, L=L, max_iter=4000, tol=0.0)
    f_star = float(min(deep.f_history))
    print(f"  L (phần trơn) = {L:.4g}, 1/L = {1/L:.4g}, f* (FISTA sâu) = {f_star:.9f}")

    ch = _two_stage(
        "Subgradient", "eta0",
        lambda e, k: subgradient(obj, w0, alpha=alpha, eta0=e, max_iter=k),
        coarse=[0.01, 0.1, 1.0, 10.0], fine=[0.3, 0.5, 1.0, 2.0, 3.0],
        f_star=f_star, chosen=CHON_L1_SUB[0], reason=CHON_L1_SUB[1],
        fname="tune_l1_subgradient.png")
    _merge("l1_subgradient", ch.summary())

    ch3 = _two_stage(
        "FISTA — bước (bội của 1/L)", "t (bội của 1/L)",
        lambda t_, k: fista(obj, w0, alpha=alpha, L=L / t_, max_iter=k, tol=0.0),
        coarse=[0.5, 1.0, 2.0, 5.0], fine=[2.0, 3.0, 4.0, 5.0, 6.0],
        f_star=f_star, chosen=CHON_L1_FISTA[0], reason=CHON_L1_FISTA[1],
        fname="tune_l1_fista.png", fine_iter=600)
    _merge("l1_fista", ch3.summary())

    ch2 = _two_stage(
        "ISTA", "t (bội của 1/L)",
        lambda mu_, k: ista(obj, w0, alpha=alpha, L=L / mu_, max_iter=k, tol=0.0),
        coarse=[0.1, 0.5, 1.0, 2.0], fine=[3.0, 5.0, 7.0, 10.0, 15.0],
        f_star=f_star, chosen=CHON_L1_ISTA[0], reason=CHON_L1_ISTA[1],
        fname="tune_l1_ista.png")
    _merge("l1_ista", ch2.summary())


def do_sgd(a):
    """Mỗi cỡ mini-batch một panel; trong panel thử các lịch bước — như bài mẫu.

    Bước hai tầng đúng tinh thần dò tay: với MỖI cỡ batch, dò nhanh 3 giá trị alpha
    bằng lịch hằng để biết cỡ nào hợp lý, rồi chạy cả 5 lịch ở alpha đó. Alpha phải
    dò riêng cho từng batch vì batch lớn nhiễu ít hơn nên chịu được bước dài hơn.
    """
    obj, w0, ref, cond = _setup(a.lam)
    n_steps, log_every, m = 1000, 5, 100
    batches = [1, 64, 1024, 16384]
    alpha_probe = {1: [0.01, 0.03, 0.1], 64: [0.05, 0.15, 0.5],
                   1024: [0.2, 0.6, 2.0], 16384: [0.5, 1.5, 5.0]}

    panels, best, chosen_alpha = {}, {}, {}
    for b in batches:
        print(f"  [batch={b}] dò alpha (lịch hằng):", flush=True)
        probe = {}
        for al in alpha_probe[b]:
            r = sgd_steps(obj, w0, al, b, n_steps // 2, "hằng", m, log_every=20)
            probe[al] = min(np.asarray(r.f_history)) - ref.f_star
            print(f"    alpha={al:<6g} gap={probe[al]:.3e}", flush=True)
        al = min(probe, key=probe.get)
        chosen_alpha[b] = al
        print(f"    -> alpha = {al:g}", flush=True)

        panels[b] = {}
        for name in SCHEDULES:
            t0 = time.perf_counter()
            r = sgd_steps(obj, w0, al, b, n_steps, name, m, log_every=log_every)
            panels[b][f"{name} (a={al:g})"] = r
            gap = float(np.asarray(r.f_history)[-1] - ref.f_star)
            print(f"    {name:<12s} gap={gap:.3e}  {time.perf_counter()-t0:5.1f}s",
                  flush=True)
        # lịch tốt nhất của cỡ batch này, để đưa sang hình so sánh
        key = min(panels[b], key=lambda kk: panels[b][kk].f_history[-1])
        best[f"batch={b} — {key}"] = panels[b][key]

    print("  hình:", plots.sgd_schedule_panels(panels, ref.f_star, log_every))
    print("  hình:", plots.sgd_batch_compare(best, ref.f_star, log_every))
    _merge("sgd_chosen", CHON_SGD)
    print(f"  cấu hình cho money plot: {CHON_SGD['batch']=} "
          f"{CHON_SGD['schedule']=} {CHON_SGD['eta0']=}")
    _merge("sgd_schedules", dict(
        n_steps=n_steps, log_every=log_every, m=m, batches=batches,
        alpha=chosen_alpha,
        rows={f"b={b}|{k}": dict(gap=float(v.f_history[-1] - ref.f_star),
                                 time=float(v.time_s[-1]))
              for b in batches for k, v in panels[b].items()},
        best={k: float(v.f_history[-1] - ref.f_star) for k, v in best.items()}))


DISPATCH = {"L": do_L, "gd-fixed": do_gd_fixed, "gd-accel": do_gd_accel,
            "agd-k": do_agd_k, "money7": do_money7,
            "gd-bt": do_gd_bt, "newton-fixed": do_newton_fixed,
            "newton-bt": do_newton_bt, "compare": do_compare, "sgd": do_sgd,
            "adaptive": do_adaptive, "l1": do_l1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", choices=list(DISPATCH) + ["all"])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.steps:
        print("các mục:", " ".join(DISPATCH))
        return
    print("BLAS:", blas_threads.verify(), flush=True)
    for name in (list(DISPATCH) if "all" in a.steps else a.steps):
        t0 = time.perf_counter()
        print(f"\n[{name}]", flush=True)
        DISPATCH[name](a)
        print(f"[{name}] xong sau {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
