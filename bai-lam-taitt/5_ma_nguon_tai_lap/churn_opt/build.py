"""Steps 2-7 — join, clean, encode, split, scale, prune, and (optionally) expand.

Produces three design-matrix variants that the optimizer benchmark consumes:
  - "ridge": collinearity-pruned, full column rank -> Ridge + Newton (needs a
    healthy, invertible Hessian).
  - "lasso": fuller matrix that intentionally keeps redundant/noisy features so the
    L1 soft-thresholding in ISTA/FISTA has something to drive to zero.
  - "poly": ridge core plus degree-2 interactions on a small behavioural core -- a
    deliberate knob to inflate kappa so AGD's sqrt(kappa) edge over GD becomes visible.

Leakage discipline: split FIRST, then fit the scaler / pruning column choice on the
training fold only and apply the frozen choice to test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import aggregate as agg
from . import loaders
from .config import ID_COL, FeatureConfig
from .labeling import build_labels

# Columns whose right-skewed money/count tails get log1p-compressed (Step 3): this
# pulls down lambda_max(XᵀX) so GD does not crawl for purely scaling reasons.
HEAVY_TAILED_PREFIXES = (
    "TX_COUNT", "TX_ROWS", "TX_PER_MONTH", "TX_AMOUNT", "TX_CNT", "TX_AMT",
    "TXG_", "TXGA_", "TXL2_", "TXL2A_", "TXH_", "TXD_",
    "ACT_COUNT", "ACT_CNT", "ACTN_", "ACTH_",
    "CA_BAL", "TD_BAL", "CA_ACCT", "LOAN_AMOUNT", "LOAN_CNT",
    "CRED_CNT", "DEB_CNT",
)
# ...but never a column that can go NEGATIVE. log1p clips at 0, which would silently
# collapse every declining customer's trend to the same value — the exact signal the
# feature exists to carry. Trend slopes are the only such columns.
NEVER_LOG_SUFFIXES = ("_slope",)

CATEGORICAL_COLS = ["EB_REGISTER_CHANNEL", "VERIFY_METHOD", "SNAPSHOT"]

# "Days since ..." columns. NaN means the event never happened in the window, which
# is not zero — it is maximally stale — so they are filled with the window span.
# Gap stats are NaN when the customer had a single active day: one long gap, same fill.
DAYS_PREFIXES = ("TX_RECENCY", "ACT_RECENCY", "TXL2R_", "ACTNR_",
                 "TX_FIRST_AGE", "ACT_FIRST_AGE", "TX_GAP_", "ACT_GAP_")


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    variant: str


# --------------------------------------------------------------------------- #
# Step 2-4: assemble one customer-level frame (still raw scale, with HAS_ flags)
# --------------------------------------------------------------------------- #
def _assemble_one(cfg: FeatureConfig, snapshot: str, tables: dict,
                  levels: dict) -> pd.DataFrame:
    """One snapshot's worth of rows: features from [T-K, T], label from (T, T+H]."""
    win = replace(cfg.windows, snapshot=snapshot)
    labels = build_labels(tables["customer"], tables["transaction"],
                          tables["activity"], win).set_index(ID_COL)
    ids = labels.index

    feats = agg.agg_customer(tables["customer"], win).reindex(ids)
    blocks = {
        "TX": agg.agg_transaction(tables["transaction"], win, ids, levels),
        "ACT": agg.agg_activity(tables["activity"], win, ids, levels),
        "DEP": agg.agg_deposit(tables["deposit"], win, ids, levels),
        "LOAN": agg.agg_lending(tables["lending"], win, ids, levels),
        "CARD": agg.agg_card(tables["card"], win, ids, levels),
    }
    for tag, block in blocks.items():
        present = block.reindex(ids).notna().any(axis=1)
        feats = feats.join(block.reindex(ids))
        feats[f"HAS_{tag}"] = present.astype(int)     # "missing = information"

    feats["SNAPSHOT"] = snapshot                      # one-hot later: calendar effect
    feats["y"] = labels["y"]
    return feats.reset_index(drop=True)


def assemble(cfg: FeatureConfig) -> pd.DataFrame:
    """Stack every snapshot into one design matrix.

    The raw tables are loaded ONCE and reused across snapshots (the activity CSV
    alone is 16M rows / ~50s), and the category inventory for every pivot is taken
    from the whole table so all snapshots produce identical columns.

    Imputation, log1p and one-hot run on the STACKED frame, so a single constant is
    shared by all snapshots — imputing per snapshot would put three different
    "maximally stale" values into the same column.
    """
    tables = {
        "customer": loaders.load_customer(),
        "transaction": loaders.load_transaction(),
        "activity": loaders.load_activity(),
        "deposit": loaders.load_deposit(),
        "lending": loaders.load_lending(),
        "card": loaders.load_card(),
    }
    levels = agg.level_inventory(tables["transaction"], tables["activity"])

    frames = [_assemble_one(cfg, snap, tables, levels) for snap in cfg.snapshots]
    feats = pd.concat(frames, ignore_index=True, sort=False)
    feats = _clean(feats)
    feats = _encode(feats)
    return feats


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — impute (no NaN may survive) and log1p heavy tails."""
    # "Days since": absent => maximally stale, filled with the largest span observed.
    days = [c for c in df.columns if c.startswith(DAYS_PREFIXES)]
    if days:
        span = df[days].max().max()
        df[days] = df[days].fillna(span if pd.notna(span) else 0.0)

    heavy = [c for c in df.columns
             if c.startswith(HEAVY_TAILED_PREFIXES) and c != "y"
             and not c.endswith(NEVER_LOG_SUFFIXES)]
    df[heavy] = df[heavy].fillna(0)
    # log1p needs non-negative input; clip tiny negatives from float noise.
    df[heavy] = np.log1p(df[heavy].clip(lower=0))

    # Everything else numeric: fill remaining NaN with 0 (counts/flags) — categoricals
    # are handled in _encode. A single surviving NaN would NaN-poison the gradient.
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(0)
    # Ratios like std/mean can produce +/-inf when the denominator underflows.
    df[num] = df[num].replace([np.inf, -np.inf], 0.0)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 — one-hot with drop_first=True (mandatory: avoids a singular Hessian)."""
    present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=True, dummy_na=False)
    # get_dummies yields bool columns; cast to float for the linear algebra downstream.
    bool_cols = df.select_dtypes(include=[bool]).columns
    df[bool_cols] = df[bool_cols].astype(float)
    return df


# --------------------------------------------------------------------------- #
# Step 6 helpers — collinearity pruning (correlation filter, then iterative VIF)
# --------------------------------------------------------------------------- #
def _drop_zero_variance(df: pd.DataFrame) -> list[str]:
    return df.columns[df.std(axis=0) > 1e-12].tolist()


def _corr_prune(df: pd.DataFrame, threshold: float) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if (upper[c] > threshold).any()}
    return [c for c in df.columns if c not in drop]


def _rank_prune(df: pd.DataFrame) -> list[str]:
    """Drop columns until the matrix has FULL COLUMN RANK. The last line of defence.

    Correlation and VIF filters are statistical: they catch strong *pairwise* and
    *near*-collinearity. Neither is guaranteed to remove an EXACT multi-column
    identity — VIF cannot even see one reliably, because a singular correlation
    matrix sends np.linalg.inv down the pinv fallback, whose finite diagonal looks
    like a perfectly ordinary VIF. An exact dependency that survives makes X'SX
    singular and Newton crashes, which is precisely the failure this project must
    not ship.

    Rank-revealing QR with column pivoting solves it directly: the permutation
    orders columns by how much new direction each one adds, and |diag(R)| falling
    below the numerical tolerance marks where the matrix stops gaining rank. Keep
    the pivots above that point, in the original column order.
    """
    from scipy.linalg import qr

    A = df.to_numpy(dtype=float)
    _, R, piv = qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    tol = diag[0] * max(A.shape) * np.finfo(float).eps
    rank = int((diag > tol).sum())
    keep = sorted(piv[:rank])
    return [df.columns[i] for i in keep]


def _vif_prune(df: pd.DataFrame, threshold: float) -> list[str]:
    """Iteratively drop the highest-VIF column until all VIF < threshold.

    VIF_i is the i-th diagonal of the inverse correlation matrix.

    The full correlation matrix is computed ONCE and then sub-indexed. Pairwise
    correlations do not change when other columns are dropped, so recomputing
    corrcoef inside the loop is pure waste — and at n=75k x p=500 it is O(n p^2) =
    2e10 flops PER ITERATION, which turned a 1-second prune into a 20-minute one.
    """
    cols = list(df.columns)
    full = np.corrcoef(df.to_numpy(), rowvar=False)
    full = np.nan_to_num(full, nan=0.0)          # constant columns -> zero correlation
    idx = list(range(len(cols)))
    while len(idx) > 1:
        sub = full[np.ix_(idx, idx)]
        try:
            vif = np.diag(np.linalg.inv(sub))
        except np.linalg.LinAlgError:
            vif = np.diag(np.linalg.pinv(sub))
        i = int(np.argmax(vif))
        if vif[i] > threshold:
            idx.pop(i)
        else:
            break
    return [cols[i] for i in idx]


# --------------------------------------------------------------------------- #
# Step 5 + 7 — split, scale, prune, expand into the three variants
# --------------------------------------------------------------------------- #
def _transform(scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    """scaler.transform, but tolerant of the empty test slice used when
    cfg.use_holdout is False (sklearn rejects arrays with 0 samples)."""
    if len(df) == 0:
        return np.empty((0, df.shape[1]), dtype=float)
    return scaler.transform(df)


def make_datasets(cfg: FeatureConfig) -> dict[str, Dataset]:
    feats = assemble(cfg)
    y = feats.pop("y").to_numpy().astype(float)
    X = feats

    # Step 5: optionally split FIRST, then fit everything on train only.
    # Default (cfg.use_holdout=False) keeps the whole eligible population as "train":
    # the object of study is the optimizer, and f(w) is well defined on any (X, y),
    # so a hold-out would only cost 25% of n. The empty test slice keeps the .npz
    # schema (and optim.data.Dataset) unchanged for every downstream consumer.
    if cfg.use_holdout:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
    else:
        Xtr, ytr = X, y
        Xte, yte = X.iloc[:0], y[:0]
    keep = _drop_zero_variance(Xtr)
    Xtr, Xte = Xtr[keep], Xte[keep]

    scaler = StandardScaler().fit(Xtr)
    Ztr = pd.DataFrame(scaler.transform(Xtr), columns=keep, index=Xtr.index)
    Zte = pd.DataFrame(_transform(scaler, Xte), columns=keep, index=Xte.index)

    # "lasso": the full standardized matrix (redundancy kept on purpose).
    datasets = {"lasso": Dataset(Ztr.to_numpy(), Zte.to_numpy(), ytr, yte,
                                 list(Ztr.columns), "lasso")}

    # "ridge": correlation + VIF pruned -> well-conditioned, full rank for Newton.
    cols = _corr_prune(Ztr, cfg.corr_threshold)
    cols = _vif_prune(Ztr[cols], cfg.vif_threshold)
    cols = _rank_prune(Ztr[cols])          # guarantee Newton can invert the Hessian
    Rtr, Rte = Ztr[cols], Zte[cols]
    datasets["ridge"] = Dataset(Rtr.to_numpy(), Rte.to_numpy(), ytr, yte,
                                list(Rtr.columns), "ridge")

    # "ridge_raw": the SAME columns as ridge but WITHOUT z-scoring — the
    # standardization-contrast test bed. Only StandardScaler is dropped (log1p is
    # feature construction, not conditioning), so the two differ by exactly the
    # z-score step. Raw column scales span orders of magnitude, so kappa explodes
    # and first-order methods stall, while affine-invariant Newton is unaffected.
    datasets["ridge_raw"] = Dataset(Xtr[cols].to_numpy(), Xte[cols].to_numpy(),
                                    ytr, yte, list(cols), "ridge_raw")

    # "poly": ridge core + degree-2 interactions on the highest-variance core.
    if cfg.add_polynomial:
        datasets["poly"] = _add_polynomial(Rtr, Rte, ytr, yte, cfg)

    return datasets


def _add_polynomial(Rtr, Rte, ytr, yte, cfg: FeatureConfig) -> Dataset:
    # Core must be continuous only: squaring/interacting a binary (0/1) column
    # reproduces an affine function of it -> exact collinearity -> singular Hessian.
    continuous = [c for c in Rtr.columns if Rtr[c].nunique() > 2]
    core = (Rtr[continuous].var().sort_values(ascending=False)
            .head(cfg.poly_core_size).index.tolist())
    new_tr, new_te, names = {}, {}, []
    for i, a in enumerate(core):
        for b in core[i:]:                       # squares (a==b) and pairwise products
            name = f"POLY_{a}__{b}"
            new_tr[name] = Rtr[a] * Rtr[b]
            new_te[name] = Rte[a] * Rte[b]
            names.append(name)
    Ptr = pd.concat([Rtr, pd.DataFrame(new_tr, index=Rtr.index)], axis=1)
    Pte = pd.concat([Rte, pd.DataFrame(new_te, index=Rte.index)], axis=1)

    # Re-standardize only the new interaction columns (the products are off-scale).
    scaler = StandardScaler().fit(Ptr[names])
    Ptr[names] = scaler.transform(Ptr[names])
    Pte[names] = _transform(scaler, Pte[names])

    # Products of near-collinear parents are themselves near-collinear, so the poly
    # block can reintroduce exact dependencies the ridge core no longer had.
    cols = _rank_prune(Ptr)
    Ptr, Pte = Ptr[cols], Pte[cols]
    return Dataset(Ptr.to_numpy(), Pte.to_numpy(), ytr, yte, list(Ptr.columns), "poly")
