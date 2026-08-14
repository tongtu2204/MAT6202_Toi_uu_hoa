"""Invariants of the feature pipeline (`churn_opt/build.py`).

These exist because the rank invariant BROKE and nothing caught it: widening the
design matrix from d=39 to d=416 silently produced rank 415, which would have made
Newton crash on the examiner's machine. Correlation and VIF filters do not
guarantee full rank (see `_rank_prune`), so the guarantee needs a test.

Fast: tiny synthetic matrices, no artifacts and no data/ needed. The one test that
does touch artifacts/ skips cleanly when the pipeline has not been run.

    pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from churn_opt.build import _corr_prune, _rank_prune, _vif_prune   # noqa: E402
from churn_opt.config import WindowConfig                          # noqa: E402


def _frame(cols: dict) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_rank_prune_removes_an_exact_three_column_identity():
    """The failure that actually happened: SPAN = FIRST_AGE - RECENCY.

    Correlation and VIF are the filters that were supposed to catch this and did
    not, so the test asserts BOTH that they miss it and that _rank_prune fixes it —
    if a future change makes corr/VIF sufficient, this test should be revisited
    rather than silently passing for the wrong reason.
    """
    rng = np.random.default_rng(0)
    n = 400
    recency = rng.normal(size=n)
    first_age = rng.normal(size=n)
    df = _frame({
        "RECENCY": recency,
        "FIRST_AGE": first_age,
        "SPAN": first_age - recency,          # exact linear combination
        "OTHER": rng.normal(size=n),
    })
    assert np.linalg.matrix_rank(df.to_numpy()) == 3   # 4 columns, rank 3

    # the statistical filters leave the identity in place: no PAIR is extreme
    kept = _vif_prune(df[_corr_prune(df, 0.98)], 100.0)
    assert np.linalg.matrix_rank(df[kept].to_numpy()) < len(kept)

    kept = _rank_prune(df)
    assert len(kept) == 3
    assert np.linalg.matrix_rank(df[kept].to_numpy()) == len(kept)


def test_rank_prune_removes_an_exact_rescaling():
    """The other one: DAYS_PER_MONTH = ACTIVE_DAYS / K. After z-scoring these are
    literally the same column."""
    rng = np.random.default_rng(1)
    days = rng.normal(size=300)
    df = _frame({"ACTIVE_DAYS": days, "DAYS_PER_MONTH": days / 3.0,
                 "OTHER": rng.normal(size=300)})
    kept = _rank_prune(df)
    assert len(kept) == 2
    assert np.linalg.matrix_rank(df[kept].to_numpy()) == 2


def test_rank_prune_keeps_a_healthy_matrix_intact():
    """It must not prune anything when the matrix is already full rank."""
    rng = np.random.default_rng(2)
    df = _frame({f"c{j}": rng.normal(size=200) for j in range(12)})
    assert _rank_prune(df) == list(df.columns)


def test_observation_windows_tile_2019_and_never_touch_2018():
    """The three snapshots must partition the year: half-open on the left, month-end
    boundaries, and the earliest window must not reach into 2018 (no data there)."""
    snaps = ["2019-03-31", "2019-06-30", "2019-09-30"]
    wins = [WindowConfig(snapshot=s) for s in snaps]

    assert wins[0].obs_start == pd.Timestamp("2018-12-31")   # EXCLUDED endpoint
    for w in wins:
        # left endpoint is excluded, so the earliest day actually used is 2019-01-01
        assert w.obs_start + pd.Timedelta(days=1) >= pd.Timestamp("2019-01-01")
        assert w.outcome_end <= pd.Timestamp("2019-12-31")
        assert w.month_buckets == w.observation_months == 3

    # consecutive windows tile: each observation window starts where the previous ends
    for prev, nxt in zip(wins, wins[1:]):
        assert nxt.obs_start == prev.T
        assert prev.outcome_end == nxt.T        # and outcome_i lines up with T_{i+1}


def test_month_buckets_is_exactly_k_for_every_snapshot():
    """MonthEnd offsets, not DateOffset(months=): from 2019-06-30 the latter gives
    2019-03-30, letting a stray 2019-03-31 in and making that one snapshot span four
    calendar months instead of three — which would unalign the M0..M2 block."""
    for s in ("2019-03-31", "2019-06-30", "2019-09-30"):
        w = WindowConfig(snapshot=s)
        months = pd.period_range(w.obs_start + pd.Timedelta(days=1), w.T, freq="M")
        assert len(months) == w.month_buckets == 3, (s, list(months))


@pytest.mark.parametrize("variant,expect_full_rank",
                         [("ridge", True), ("poly", True)])
def test_saved_variants_are_full_rank(variant, expect_full_rank):
    """The shipped design matrices must satisfy the invariant Newton depends on.

    Skips when artifacts/ has not been built, so the suite stays runnable without
    the 974 MB data/ directory.
    """
    npz = ROOT / "artifacts" / f"{variant}.npz"
    if not npz.exists():
        pytest.skip(f"{npz.name} not built; run run_pipeline.py")
    X = np.load(npz)["X_train"]
    assert np.isfinite(X).all(), "a NaN/Inf would poison every gradient"
    rank = np.linalg.matrix_rank(X)
    assert (rank == X.shape[1]) == expect_full_rank, (variant, rank, X.shape)
