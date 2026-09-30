"""Data splitting and held-out classification metrics for the GD/AGD update."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def train_valid_indices(y, valid_size=0.20, random_state=42, groups=None):
    """Return train/validation indices while preserving an external test set."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    if not 0 < valid_size < 1:
        raise ValueError("valid_size phải nằm trong (0, 1)")

    if groups is not None:
        groups = np.asarray(groups)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=valid_size, random_state=random_state
        )
        train, valid = next(splitter.split(idx, y, groups))
        return train, valid, "group"

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=valid_size, random_state=random_state
    )
    train, valid = next(splitter.split(idx, y))
    return train, valid, "stratified-row"


def split_indices(y, valid_size=0.20, test_size=0.20, random_state=42,
                  groups=None):
    """Return train/validation/test indices.

    When customer ids are available, group splitting prevents the same customer
    from appearing in more than one split. Otherwise a deterministic stratified
    row split is used and the limitation must be disclosed in the report.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    holdout = valid_size + test_size
    if not 0 < holdout < 1:
        raise ValueError("valid_size + test_size phải nằm trong (0, 1)")

    if groups is not None:
        groups = np.asarray(groups)
        first = GroupShuffleSplit(n_splits=1, test_size=holdout,
                                  random_state=random_state)
        train, rest = next(first.split(idx, y, groups))
        test_fraction = test_size / holdout
        second = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                   random_state=random_state + 1)
        valid_rel, test_rel = next(second.split(rest, y[rest], groups[rest]))
        return train, rest[valid_rel], rest[test_rel], "group"

    first = StratifiedShuffleSplit(n_splits=1, test_size=holdout,
                                    random_state=random_state)
    train, rest = next(first.split(idx, y))
    test_fraction = test_size / holdout
    second = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction,
                                     random_state=random_state + 1)
    valid_rel, test_rel = next(second.split(rest, y[rest]))
    return train, rest[valid_rel], rest[test_rel], "stratified-row"


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def predict_proba(X, w):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return sigmoid(X @ w[:-1] + w[-1])


def classification_metrics(X, y, w, threshold=0.5):
    y = np.asarray(y, dtype=float)
    p = predict_proba(X, w)
    pred = (p >= threshold).astype(float)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0.0, 1.0])),
    }
