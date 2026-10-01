"""Metric dự đoán cho so sánh mô hình trước và sau tối ưu."""
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return sigmoid(X @ w[:-1] + w[-1])


def classification_metrics(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                           threshold: float = 0.5) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    probability = predict_proba(X, w)
    prediction = (probability >= threshold).astype(float)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0.0, 1.0])),
    }
