"""Cấu hình duy nhất cho thí nghiệm cập nhật GD/AGD."""
from __future__ import annotations

# Một lần chạy tới MAX_ITER cho phép đọc kết quả ở toàn bộ checkpoint, tránh chạy
# lại cùng một cấu hình cho từng ngân sách.
CHECKPOINTS = (50, 150, 500, 1_000, 3_000)
MAX_ITER = max(CHECKPOINTS)

# Lưới GD rộng hơn bản cũ và có mật độ cao quanh vùng 0.3--0.5.
GD_STEP_GRID = (
    0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.85, 1.00, 1.50, 2.00,
)

# Vòng một của AGD: dò bước và ghép beta theo L_eff=1/t. Đây là cấu hình thực
# nghiệm, không được trình bày như bảo đảm lý thuyết khi L_eff < L thật.
AGD_STEP_GRID = (
    0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
    0.24, 0.28, 0.30, 0.32, 0.35, 0.38, 0.42, 0.50,
)

# Vòng hai: với các bước tốt nhất, dò beta độc lập quanh vùng momentum cao.
AGD_BETA_GRID = (0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995)
AGD_TOP_STEPS_FOR_BETA_SEARCH = 3

LAMBDA = 1e-3
TOL = 1e-10
RANDOM_STATE = 42
VALID_SIZE = 0.20
TEST_SIZE = 0.20
TIMING_REPEATS = 3

