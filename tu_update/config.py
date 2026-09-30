"""Single source of truth for the revised GD/AGD experiment."""
from __future__ import annotations

# All four configurations are compared at exactly the same iteration budgets.
CHECKPOINTS = (50, 150, 500, 1_000, 3_000)
SELECTION_ITER = 150
STABILITY_ITER = 500
STABILITY_TOP_N = 10
CONVERGENCE_MAX_ITER = 10_000

# The public ridge artifact has L ~= 1.96. The grid spans both the textbook
# region and the empirically stable region, with a dense band near the observed
# GD stability boundary instead of only five hand-picked values.
GD_STEP_GRID = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80,
    1.00, 1.20, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00,
    3.10, 3.20, 3.30, 3.40, 3.50, 4.00, 5.00, 8.00, 10.00,
)

# AGD is more sensitive than GD, so t and beta are searched independently.
# This avoids the old experiment's hidden coupling beta=beta(1/t, mu).
AGD_STEP_GRID = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80,
    1.00, 1.20, 1.40, 1.50, 1.60, 1.80, 2.00, 2.20,
    2.40, 2.60, 3.00, 3.50, 4.00,
)
AGD_BETA_GRID = (
    0.00, 0.50, 0.70, 0.80, 0.85, 0.88,
    0.90, 0.92, 0.94, 0.96, 0.98, 0.99,
)

LAMBDA = 1e-3
TOL = 1e-10
RANDOM_STATE = 42
VALID_SIZE = 0.20
TEST_SIZE = 0.20
TIMING_REPEATS = 3
