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

# Lưới thô cố ý thưa: chỉ dùng để nhận diện vùng tham số ổn định/hội tụ.
# Sau đó mã sinh lưới tinh trong khoảng hai điểm lân cận quanh ứng viên tốt
# nhất. Mọi ứng viên ở cả hai giai đoạn đều chạy đúng 500 vòng.
STEP_COARSE = (0.05, 0.10, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00)
GD_STEP_COARSE = STEP_COARSE
AGD_STEP_COARSE = STEP_COARSE

# Backtracking dùng 3 x 3 = 9 cặp ở vòng thô; vòng tinh tiếp tục dùng 3 x 3
# quanh cặp tốt nhất. t0 được giữ cố định để chỉ đánh giá vai trò của (rho, c).
BACKTRACKING_RHO_COARSE = (0.10, 0.50, 0.90)
BACKTRACKING_C_COARSE = (1e-4, 0.10, 0.50)

STEP_FINE_POINTS = 9
BACKTRACKING_FINE_POINTS = 3

RANDOM_STATE = 42
