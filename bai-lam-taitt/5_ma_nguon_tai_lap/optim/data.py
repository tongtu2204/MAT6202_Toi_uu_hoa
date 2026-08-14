"""Load the design matrices `churn_opt` wrote to artifacts/<variant>.npz.

Each .npz holds X_train, X_test, y_train, y_test (X already standardized,
one-hot drop_first, no NaNs). Variants: 'ridge' (Newton-safe, full rank),
'lasso' (redundant, for ISTA/FISTA), 'poly' (kappa-inflated).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    features: list[str]

    @property
    def n(self) -> int:
        return self.X_train.shape[0]

    @property
    def d(self) -> int:
        return self.X_train.shape[1]


def load_variant(name: str = "ridge", artifacts: Path = ARTIFACTS) -> Dataset:
    """Read <name>.npz (+ <name>.features.txt) into a Dataset."""
    npz = np.load(artifacts / f"{name}.npz")
    feat_path = artifacts / f"{name}.features.txt"
    features = feat_path.read_text().split() if feat_path.exists() else []
    return Dataset(
        name=name,
        X_train=npz["X_train"], X_test=npz["X_test"],
        y_train=npz["y_train"], y_test=npz["y_test"],
        features=features,
    )
