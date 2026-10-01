"""Cấu hình duy nhất cho thí nghiệm GD/AGD đã chốt."""
from __future__ import annotations

# Mỗi ứng viên tham số được quan sát đúng 500 vòng. Sau khi khóa tham số,
# thuật toán được chạy lại từ w0 tới hội tụ (MAX_ITER chỉ là van an toàn).
SEARCH_ITERATIONS = 500
# 1e-8 là ngưỡng chuẩn gradient đủ chặt cho bài toán float64 này. Với 1e-10,
# Armijo đã chạm sàn sai số của objective (xấp xỉ 1e-16) trước khi chứng nhận
# được điều kiện giảm, dù nghiệm thực tế đã trùng f* tới độ chính xác máy.
CONVERGENCE_TOL = 1e-8
CONVERGENCE_MAX_ITER = 50_000
TIMING_REPEATS = 3

LAMBDA = 1e-3
BACKTRACKING_T0 = 20.0
BACKTRACKING_MAX_TRIALS = 60

# Lưới rộng, sau đó mã tự sinh một lưới tinh giữa hai điểm lân cận quanh ứng
# viên tốt nhất. Mọi điểm của cả lưới thô và lưới tinh đều chạy 500 vòng.
GD_STEP_COARSE = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80,
    1.00, 1.20, 1.50, 1.80, 2.00, 2.20, 2.40, 2.60,
    2.80, 3.00, 3.20, 3.40, 3.60, 4.00, 5.00, 8.00, 10.00,
)

AGD_STEP_COARSE = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80,
    1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.40,
    2.60, 2.80, 3.00, 3.50, 4.00,
)

BACKTRACKING_RHO_COARSE = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90)
BACKTRACKING_C_COARSE = (1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.25, 0.50)

STEP_FINE_POINTS = 9
BACKTRACKING_FINE_POINTS = 5

RANDOM_STATE = 42
