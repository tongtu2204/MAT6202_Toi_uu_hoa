"""Step 1 — fold each multi-row table down to one row per customer.

Every aggregate here is computed strictly on the observation window (T-K, T]. This
is also where dimensionality d is pumped toward the target (see CLAUDE.md): the raw
tables carry only a handful of columns, so d comes from PIVOTING the categorical
axes they do have —

    TRANS_LV1     3 levels   -> count / amount / share            (9 cols)
    TRANS_LV2    14 levels   -> count / amount / share / recency  (56 cols)
    ACTIVITY_NAME 45 levels  -> count / share / recency / days    (180 cols)
    hour-of-day  24 levels   -> count + share, transactions AND activity (96 cols)
    day-of-week   7 levels   -> count + share                     (14 cols)

plus per-month levels (M0 = oldest month in the window) and shape statistics of the
amount / inter-arrival distributions. Roughly 500 raw columns before pruning.

Naming rule for anything time-indexed: columns are named RELATIVE to the window
(`_M0.._M{K-1}`), never by calendar month. Several snapshots are stacked into one design
matrix, so a calendar name would put June-2019 and September-2019 in different
columns and make the stack ragged.

Heavy-tailed money/count columns are produced raw; the log1p compression the spec
calls for happens later in build.py (Step 3) so the transform stays centralized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_COL, WindowConfig

WEEKEND = {"Sat", "Sun"}
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = list(range(24))


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def restrict(df: pd.DataFrame, date_col: str, win: WindowConfig,
             ids: pd.Index | None = None) -> pd.DataFrame:
    """Rows inside the observation window, optionally limited to the eligible ids.

    Filtering by `ids` FIRST matters: the activity table is 16M rows and the pivots
    below are customer x level, so cutting to the ~45k eligible customers before
    unstacking is the difference between a 200 MB frame and a 4 GB one.
    """
    out = df[df[date_col].between(win.obs_start, win.T, inclusive="right")]
    if ids is not None:
        out = out[out[ID_COL].isin(ids)]
    return out


def _pivot_counts(df: pd.DataFrame, level_col: str, prefix: str,
                  levels: list, value_col: str | None = None,
                  aggfunc: str = "sum") -> pd.DataFrame:
    """customer x level matrix, reindexed onto a FIXED level list.

    The fixed list is what keeps the three snapshots column-compatible: a level that
    happens to be absent in one window still gets its (all-zero) column.
    """
    if value_col is None:
        g = df.groupby([ID_COL, level_col], observed=True).size()
    else:
        g = df.groupby([ID_COL, level_col], observed=True)[value_col].agg(aggfunc)
    wide = g.unstack(level_col, fill_value=0.0)
    wide = wide.reindex(columns=levels, fill_value=0.0)
    wide.columns = [f"{prefix}{c}" for c in levels]
    return wide.astype(float)


def _share_of(wide: pd.DataFrame, prefix: str, levels: list) -> pd.DataFrame:
    """Row-normalized version of `wide`, named `{prefix}{level}`, LAST level dropped.

    Shares sum to 1 for anyone with at least one event, so keeping all k of them next
    to an intercept is (near-)exact collinearity — the same trap as one-hot without
    drop_first, and it makes the Hessian singular. Drop one, exactly as
    `pd.get_dummies(drop_first=True)` does.
    """
    total = wide.sum(axis=1).replace(0.0, np.nan)
    sh = wide.div(total, axis=0).fillna(0.0)
    sh.columns = [f"{prefix}{lv}" for lv in levels]
    return sh.iloc[:, :-1]          # drop_first-style: avoid the sum-to-one identity


def _recency_by_level(df: pd.DataFrame, date_col: str, level_col: str,
                      prefix: str, levels: list, T: pd.Timestamp) -> pd.DataFrame:
    """Days since the customer last did each level (NaN = never; imputed in build)."""
    last = (df.groupby([ID_COL, level_col], observed=True)[date_col].max()
              .unstack(level_col))
    # A level nobody used in this window comes back from reindex as a float NaN
    # column, and `T - float` raises. Insert it as NaT so the whole frame stays
    # datetime and the subtraction below is well typed.
    for lv in levels:
        if lv not in last.columns:
            last[lv] = pd.NaT
    last = last[levels]
    days = (np.datetime64(T) - last.to_numpy(dtype="datetime64[ns]")) \
        / np.timedelta64(1, "D")
    return pd.DataFrame(days, index=last.index,
                        columns=[f"{prefix}{lv}" for lv in levels], dtype=float)


def _month_matrix(df: pd.DataFrame, date_col: str, value_col=None,
                  aggfunc="sum") -> pd.DataFrame:
    """Customer x month matrix (one column per calendar month in the window)."""
    m = df[[ID_COL, date_col] + ([value_col] if value_col else [])].copy()
    m["_month"] = m[date_col].dt.to_period("M")
    if value_col is None:
        g = m.groupby([ID_COL, "_month"], observed=True).size()
    else:
        g = m.groupby([ID_COL, "_month"], observed=True)[value_col].agg(aggfunc)
    return g.unstack("_month")


def _monthly_levels(wide: pd.DataFrame, prefix: str, n_months: int) -> pd.DataFrame:
    """The last `n_months` calendar columns, renamed M0 (oldest) .. M{n-1} (newest).

    Relative naming is mandatory for the multi-snapshot stack — see module docstring.
    """
    cols = sorted(wide.columns)[-n_months:]
    vals = np.nan_to_num(wide[cols].to_numpy(dtype=float), nan=0.0)
    if vals.shape[1] < n_months:            # short window: left-pad with zeros
        pad = np.zeros((vals.shape[0], n_months - vals.shape[1]))
        vals = np.hstack([pad, vals])
    return pd.DataFrame(vals, index=wide.index,
                        columns=[f"{prefix}_M{i}" for i in range(n_months)])


def _slope_momentum(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-customer trend features from a customer x month matrix.

    Missing months are treated as zero (absence of activity = zero that month), so
    a fading customer gets a negative slope — the strongest churn signal per spec.
    """
    cols = sorted(wide.columns)
    vals = np.nan_to_num(wide[cols].to_numpy(dtype=float), nan=0.0)
    x = np.arange(len(cols), dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum() or 1.0
    slope = (vals * xc).sum(axis=1) / denom
    first, last = vals[:, 0], vals[:, -1]
    momentum = (last + 1.0) / (first + 1.0)        # smoothed; never divides by zero
    mean = vals.mean(axis=1)
    std = vals.std(axis=1)
    return pd.DataFrame(
        {f"{prefix}_slope": slope, f"{prefix}_momentum": momentum,
         f"{prefix}_std": std, f"{prefix}_mean": mean,
         f"{prefix}_cv": std / np.where(mean > 0, mean, np.nan),
         f"{prefix}_last_share": last / np.where(vals.sum(axis=1) > 0,
                                                 vals.sum(axis=1), np.nan)},
        index=wide.index,
    )


def _gap_stats(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    """Inter-arrival statistics of a customer's active DAYS.

    Recency says how long since the last event; these say whether the customer's
    rhythm was regular or bursty, which recency cannot express.
    """
    days = (df[[ID_COL, date_col]].drop_duplicates()
              .sort_values([ID_COL, date_col]))
    delta = days.groupby(ID_COL, observed=True)[date_col].diff().dt.days
    g = delta.groupby(days[ID_COL], observed=True)
    return pd.DataFrame({
        f"{prefix}_GAP_MEAN": g.mean(),
        f"{prefix}_GAP_STD": g.std(),
        f"{prefix}_GAP_MAX": g.max(),
        f"{prefix}_GAP_MIN": g.min(),
    })


def _entropy(wide: pd.DataFrame) -> pd.Series:
    """Shannon entropy of a customer's distribution over levels (in nats).

    A customer who only ever logs in has low entropy; one who uses many features has
    high entropy. Entirely different information from the counts themselves.
    """
    total = wide.sum(axis=1).replace(0.0, np.nan)
    p = wide.div(total, axis=0)
    return -(p * np.log(p.where(p > 0))).sum(axis=1, skipna=True)


# --------------------------------------------------------------------------- #
# Table 1 — CUSTOMER (static demographics; already one row per customer)
# --------------------------------------------------------------------------- #
def agg_customer(customer: pd.DataFrame, win: WindowConfig) -> pd.DataFrame:
    df = customer.set_index(ID_COL)
    T = win.T
    out = pd.DataFrame(index=df.index)
    out["AGE"] = (T - df["DATE_OF_BIRTH"]).dt.days / 365.25
    out["TENURE_CIF"] = (T - df["CLIENT_CREATE_DATE"]).dt.days
    out["TENURE_EBANK"] = (T - df["IB_REGISTER_DATE"]).dt.days
    out["ADOPTION_LAG"] = (df["IB_REGISTER_DATE"] - df["CLIENT_CREATE_DATE"]).dt.days
    out["EBANK_TENURE_RATIO"] = out["TENURE_EBANK"] / out["TENURE_CIF"].replace(0, np.nan)
    out["CREATE_MONTH"] = df["CLIENT_CREATE_DATE"].dt.month
    out["CREATE_DOW"] = df["CLIENT_CREATE_DATE"].dt.dayofweek
    # Binary flags -> {0,1}; unknowns stay NaN and are imputed in build.
    out["IS_MALE"] = df["CLIENT_SEX"].map({"M": 1, "F": 0})
    out["IS_STAFF"] = df["STAFF_VIB"].map({"Y": 1, "N": 0})
    out["HAS_SMS"] = df["SMS"].map({"Y": 1, "N": 0})
    # Categoricals carried through; one-hot (drop_first) applied in build.py.
    out["EB_REGISTER_CHANNEL"] = df["EB_REGISTER_CHANNEL"]
    out["VERIFY_METHOD"] = df["VERIFY_METHOD"]
    return out


# --------------------------------------------------------------------------- #
# Table 2 — MYVIB TRANSACTION (richest source: RFM + behaviour + three pivots)
# --------------------------------------------------------------------------- #
def agg_transaction(transaction: pd.DataFrame, win: WindowConfig,
                    ids: pd.Index | None = None,
                    levels: dict | None = None) -> pd.DataFrame:
    tx = restrict(transaction, "TRANS_DATE", win, ids)
    lv1 = levels["TRANS_LV1"]
    lv2 = levels["TRANS_LV2"]
    n_months = win.month_buckets
    g = tx.groupby(ID_COL, observed=True)

    out = pd.DataFrame(index=g.size().index)
    # --- RFM core ---
    out["TX_RECENCY"] = (win.T - g["TRANS_DATE"].max()).dt.days
    out["TX_FIRST_AGE"] = (win.T - g["TRANS_DATE"].min()).dt.days
    out["TX_COUNT"] = g["TRANS_NO"].sum()
    out["TX_ROWS"] = g.size()
    out["TX_PER_MONTH"] = out["TX_COUNT"] / win.observation_months
    out["TX_ACTIVE_DAYS"] = g["TRANS_DATE"].nunique()
    out["TX_PER_ACTIVE_DAY"] = out["TX_COUNT"] / out["TX_ACTIVE_DAYS"]

    # --- monetary shape (a mean hides everything that matters about a tail) ---
    amt = g["TRANS_AMOUNT"]
    out["TX_AMOUNT_SUM"] = amt.sum()
    out["TX_AMOUNT_MEAN"] = amt.mean()
    out["TX_AMOUNT_STD"] = amt.std()
    out["TX_AMOUNT_MIN"] = amt.min()
    out["TX_AMOUNT_MAX"] = amt.max()
    out["TX_AMOUNT_MED"] = amt.median()
    out["TX_AMOUNT_Q25"] = amt.quantile(0.25)
    out["TX_AMOUNT_Q75"] = amt.quantile(0.75)
    out["TX_AMOUNT_IQR"] = out["TX_AMOUNT_Q75"] - out["TX_AMOUNT_Q25"]
    out["TX_AMOUNT_CV"] = out["TX_AMOUNT_STD"] / out["TX_AMOUNT_MEAN"].replace(0, np.nan)
    out["TX_AMOUNT_MAXSHARE"] = out["TX_AMOUNT_MAX"] / out["TX_AMOUNT_SUM"].replace(0, np.nan)
    out["TX_AMOUNT_PER_DAY"] = out["TX_AMOUNT_SUM"] / out["TX_ACTIVE_DAYS"]

    # --- diversity ---
    out["TX_LV1_NUNIQUE"] = g["TRANS_LV1"].nunique()
    out["TX_LV2_NUNIQUE"] = g["TRANS_LV2"].nunique()
    out["TX_HOUR_NUNIQUE"] = g["TRANS_HOUR"].nunique()
    out["TX_DOW_NUNIQUE"] = g["DAY_OF_WEEK"].nunique()
    out["TX_HOUR_MEAN"] = g["TRANS_HOUR"].mean()
    out["TX_HOUR_STD"] = g["TRANS_HOUR"].std()

    # --- behavioural ratios ---
    is_weekend = tx["DAY_OF_WEEK"].isin(WEEKEND)
    off_hours = ~tx["TRANS_HOUR"].between(8, 17)
    night = tx["TRANS_HOUR"].between(0, 5)
    out["TX_WEEKEND_RATIO"] = is_weekend.groupby(tx[ID_COL], observed=True).mean()
    out["TX_OFFHOURS_RATIO"] = off_hours.groupby(tx[ID_COL], observed=True).mean()
    out["TX_NIGHT_RATIO"] = night.groupby(tx[ID_COL], observed=True).mean()

    # --- pivots: LV1 (3) and LV2 (14), each as count / amount / share ---
    c1 = _pivot_counts(tx, "TRANS_LV1", "TXG_", lv1, "TRANS_NO", "sum")
    a1 = _pivot_counts(tx, "TRANS_LV1", "TXGA_", lv1, "TRANS_AMOUNT", "sum")
    c2 = _pivot_counts(tx, "TRANS_LV2", "TXL2_", lv2, "TRANS_NO", "sum")
    a2 = _pivot_counts(tx, "TRANS_LV2", "TXL2A_", lv2, "TRANS_AMOUNT", "sum")
    out = out.join([c1, a1, c2, a2,
                    _share_of(c1, "TXGS_", lv1),
                    _share_of(c2, "TXL2S_", lv2),
                    _recency_by_level(tx, "TRANS_DATE", "TRANS_LV2", "TXL2R_", lv2, win.T)])
    out["TX_LV1_ENTROPY"] = _entropy(c1)
    out["TX_LV2_ENTROPY"] = _entropy(c2)

    # --- pivots: hour-of-day (24) and day-of-week (7), count + share ---
    ch = _pivot_counts(tx, "TRANS_HOUR", "TXH_", HOURS, "TRANS_NO", "sum")
    cd = _pivot_counts(tx, "DAY_OF_WEEK", "TXD_", DOW_ORDER, "TRANS_NO", "sum")
    out = out.join([ch, cd,
                    _share_of(ch, "TXHS_", HOURS),
                    _share_of(cd, "TXDS_", DOW_ORDER)])
    out["TX_HOUR_ENTROPY"] = _entropy(ch)
    out["TX_DOW_ENTROPY"] = _entropy(cd)

    # --- per-month levels + trend, on counts and on amount ---
    m_cnt = _month_matrix(tx, "TRANS_DATE", "TRANS_NO", "sum")
    m_amt = _month_matrix(tx, "TRANS_DATE", "TRANS_AMOUNT", "sum")
    out = out.join([_monthly_levels(m_cnt, "TX_CNT", n_months),
                    _monthly_levels(m_amt, "TX_AMT", n_months),
                    _slope_momentum(m_cnt, "TX"),
                    _slope_momentum(m_amt, "TX_AMT"),
                    _gap_stats(tx, "TRANS_DATE", "TX")])
    return out


# --------------------------------------------------------------------------- #
# Table 3 — MYVIB ACTIVITY (non-financial engagement; 45-level name pivot)
# --------------------------------------------------------------------------- #
def agg_activity(activity: pd.DataFrame, win: WindowConfig,
                 ids: pd.Index | None = None,
                 levels: dict | None = None) -> pd.DataFrame:
    act = restrict(activity, "ACTIVITY_DATE", win, ids)
    names = levels["ACTIVITY_NAME"]
    n_months = win.month_buckets
    g = act.groupby(ID_COL, observed=True)

    out = pd.DataFrame(index=g.size().index)
    out["ACT_RECENCY"] = (win.T - g["ACTIVITY_DATE"].max()).dt.days
    out["ACT_FIRST_AGE"] = (win.T - g["ACTIVITY_DATE"].min()).dt.days
    out["ACT_COUNT"] = g.size()
    out["ACT_ACTIVE_DAYS"] = g["ACTIVITY_DATE"].nunique()
    out["ACT_PER_ACTIVE_DAY"] = out["ACT_COUNT"] / out["ACT_ACTIVE_DAYS"]
    out["ACT_NAME_NUNIQUE"] = g["ACTIVITY_NAME"].nunique()
    out["ACT_HOUR_NUNIQUE"] = g["ACTIVITY_HOUR"].nunique()
    out["ACT_HOUR_MEAN"] = g["ACTIVITY_HOUR"].mean()
    out["ACT_HOUR_STD"] = g["ACTIVITY_HOUR"].std()
    night = act["ACTIVITY_HOUR"].between(0, 5)
    off_hours = ~act["ACTIVITY_HOUR"].between(8, 17)
    out["ACT_NIGHT_RATIO"] = night.groupby(act[ID_COL], observed=True).mean()
    out["ACT_OFFHOURS_RATIO"] = off_hours.groupby(act[ID_COL], observed=True).mean()

    # --- the 45-level name pivot: count / share / recency ---
    cn = _pivot_counts(act, "ACTIVITY_NAME", "ACTN_", names)
    out = out.join([cn, _share_of(cn, "ACTNS_", names),
                    _recency_by_level(act, "ACTIVITY_DATE", "ACTIVITY_NAME",
                                      "ACTNR_", names, win.T)])
    out["ACT_NAME_ENTROPY"] = _entropy(cn)

    # --- hour pivot (24), count + share ---
    ch = _pivot_counts(act, "ACTIVITY_HOUR", "ACTH_", HOURS)
    out = out.join([ch, _share_of(ch, "ACTHS_", HOURS)])
    out["ACT_HOUR_ENTROPY"] = _entropy(ch)

    monthly = _month_matrix(act, "ACTIVITY_DATE")
    out = out.join([_monthly_levels(monthly, "ACT_CNT", n_months),
                    _slope_momentum(monthly, "ACT"),
                    _gap_stats(act, "ACTIVITY_DATE", "ACT")])
    return out


# --------------------------------------------------------------------------- #
# Table 4 — DEPOSIT (monthly balance panel)
# --------------------------------------------------------------------------- #
def _latest_in_window(df: pd.DataFrame, win: WindowConfig,
                      ids: pd.Index | None = None):
    d = df[df["MONTH"].between(win.obs_start, win.T, inclusive="right")]
    if ids is not None:
        d = d[d[ID_COL].isin(ids)]
    idx = d.groupby(ID_COL, observed=True)["MONTH"].idxmax()
    return d, d.loc[idx].set_index(ID_COL)


def _panel_block(d: pd.DataFrame, latest: pd.DataFrame, col: str, prefix: str,
                 n_months: int) -> pd.DataFrame:
    """The standard treatment of one monthly balance/holding column: level, summary
    stats, per-month levels and trend. Applied identically to deposits, loans, cards
    so the three panels contribute comparable features."""
    g = d.groupby(ID_COL, observed=True)[col]
    out = pd.DataFrame(index=latest.index)
    out[f"{prefix}_LATEST"] = latest[col]
    out[f"{prefix}_MEAN"] = g.mean()
    out[f"{prefix}_STD"] = g.std()
    out[f"{prefix}_MIN"] = g.min()
    out[f"{prefix}_MAX"] = g.max()
    out[f"{prefix}_RANGE"] = out[f"{prefix}_MAX"] - out[f"{prefix}_MIN"]
    monthly = _month_matrix(d, "MONTH", col, "mean")
    return out.join([_monthly_levels(monthly, prefix, n_months),
                     _slope_momentum(monthly, prefix)])


def agg_deposit(deposit: pd.DataFrame, win: WindowConfig,
                ids: pd.Index | None = None, levels: dict | None = None):
    d, latest = _latest_in_window(deposit, win, ids)
    n_months = win.month_buckets
    out = pd.DataFrame(index=latest.index)
    # NB: CA_ACCT_LATEST is produced by the COUNT_CA_ACCT panel block below —
    # setting it here too would collide on the join.
    out["TD_ACCT_LATEST"] = latest["COUNT_TD_ACCT"]
    out["HAS_TD"] = (latest["COUNT_TD_ACCT"] > 0).astype(int)
    out["DEP_MONTHS_SEEN"] = d.groupby(ID_COL, observed=True)["MONTH"].nunique()
    out = out.join([_panel_block(d, latest, "AVG_CA_BALANCE", "CA_BAL", n_months),
                    _panel_block(d, latest, "AVG_TD_BALANCE", "TD_BAL", n_months),
                    _panel_block(d, latest, "COUNT_CA_ACCT", "CA_ACCT", n_months)])
    total = out["CA_BAL_LATEST"].fillna(0) + out["TD_BAL_LATEST"].fillna(0)
    out["TD_BAL_SHARE"] = out["TD_BAL_LATEST"] / total.replace(0, np.nan)
    out["CA_BAL_PER_ACCT"] = out["CA_BAL_LATEST"] / out["CA_ACCT_LATEST"].replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Table 5 — LENDING / Table 6 — CARD (monthly holdings panels)
# --------------------------------------------------------------------------- #
def agg_lending(lending: pd.DataFrame, win: WindowConfig,
                ids: pd.Index | None = None, levels: dict | None = None):
    d, latest = _latest_in_window(lending, win, ids)
    n_months = win.month_buckets
    out = pd.DataFrame(index=latest.index)
    out["HAS_LOAN"] = 1
    out["LOAN_MONTHS_SEEN"] = d.groupby(ID_COL, observed=True)["MONTH"].nunique()
    out = out.join([_panel_block(d, latest, "AVG_LOAN_AMOUNT", "LOAN_AMOUNT", n_months),
                    _panel_block(d, latest, "COUNT_OF_LOAN", "LOAN_CNT", n_months)])
    out["LOAN_PER_ACCT"] = (out["LOAN_AMOUNT_LATEST"]
                            / out["LOAN_CNT_LATEST"].replace(0, np.nan))
    return out


def agg_card(card: pd.DataFrame, win: WindowConfig,
             ids: pd.Index | None = None, levels: dict | None = None):
    d, latest = _latest_in_window(card, win, ids)
    n_months = win.month_buckets
    out = pd.DataFrame(index=latest.index)
    out["HAS_CREDITCARD"] = (latest["COUNT_CREDITCARD"] > 0).astype(int)
    out["CARD_MONTHS_SEEN"] = d.groupby(ID_COL, observed=True)["MONTH"].nunique()
    out = out.join([_panel_block(d, latest, "COUNT_CREDITCARD", "CRED_CNT", n_months),
                    _panel_block(d, latest, "COUNT_DEBITCARD", "DEB_CNT", n_months)])
    out["CARD_TOTAL"] = out["CRED_CNT_LATEST"].fillna(0) + out["DEB_CNT_LATEST"].fillna(0)
    out["CRED_SHARE"] = out["CRED_CNT_LATEST"] / out["CARD_TOTAL"].replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Level inventory — computed ONCE on the full tables, shared by every snapshot
# --------------------------------------------------------------------------- #
def level_inventory(transaction: pd.DataFrame, activity: pd.DataFrame) -> dict:
    """The fixed category lists every pivot is reindexed onto.

    Taken from the WHOLE table, not from a window: otherwise a rare TRANS_LV2 that is
    missing from one quarter would shift the column layout of that snapshot and the
    stack would not line up.
    """
    def _levels(s: pd.Series) -> list:
        if isinstance(s.dtype, pd.CategoricalDtype):
            return list(s.cat.categories)
        return sorted(s.dropna().unique().tolist())

    return {
        "TRANS_LV1": _levels(transaction["TRANS_LV1"]),
        "TRANS_LV2": _levels(transaction["TRANS_LV2"]),
        "ACTIVITY_NAME": _levels(activity["ACTIVITY_NAME"]),
    }
