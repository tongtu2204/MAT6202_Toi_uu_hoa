"""Step 0 — define the eligible population and the churn label, leakage-free.

Three disjoint windows around snapshot T, both half-open on the left so that no
calendar day falls in two of them and the earliest snapshot never reaches into 2018
(which the raw data does not cover):
  observation (T-K, T]  -> features only (built elsewhere)
  outcome     (T, T+H]  -> label only (here)

Eligible = customers active (>=1 transaction OR activity) in the observation window.
Label y_i = 1 iff the customer has NO transaction AND NO activity in the outcome
window. Customers onboarded after T are dropped (their absence in the outcome window
would be mislabeled as churn).
"""
from __future__ import annotations

import pandas as pd

from .config import ID_COL, WindowConfig


def _active_ids(df: pd.DataFrame, date_col: str, start, end, inclusive: str) -> set:
    mask = df[date_col].between(start, end, inclusive=inclusive)
    return set(df.loc[mask, ID_COL].unique())


def build_labels(customer: pd.DataFrame, transaction: pd.DataFrame,
                 activity: pd.DataFrame, win: WindowConfig) -> pd.DataFrame:
    """Return a frame [CUSTOMER_NUMBER, y] for the eligible population only."""
    T, obs_start, outcome_end = win.T, win.obs_start, win.outcome_end

    # Active in the observation window (strictly after obs_start, inclusive of T).
    tx_obs = _active_ids(transaction, "TRANS_DATE", obs_start, T, "right")
    act_obs = _active_ids(activity, "ACTIVITY_DATE", obs_start, T, "right")
    eligible = tx_obs | act_obs

    # Active in the outcome window (strictly after T, up to T+H).
    tx_out = _active_ids(transaction, "TRANS_DATE", T, outcome_end, "right")
    act_out = _active_ids(activity, "ACTIVITY_DATE", T, outcome_end, "right")
    active_outcome = tx_out | act_out

    # Drop customers onboarded after the snapshot (CIF created in the outcome window).
    onboarded_after_T = set(
        customer.loc[customer["CLIENT_CREATE_DATE"] > T, ID_COL].unique()
    )
    eligible -= onboarded_after_T

    ids = sorted(eligible)
    y = [0 if cid in active_outcome else 1 for cid in ids]
    labels = pd.DataFrame({ID_COL: ids, "y": y})
    return labels
