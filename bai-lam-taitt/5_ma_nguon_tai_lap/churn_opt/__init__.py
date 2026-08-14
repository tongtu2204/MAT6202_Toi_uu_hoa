"""churn_opt — feature-engineering pipeline for the MAT6202 optimization test bed.

The package folds the multi-table VIB panel data into standardized design matrices
whose conditioning (kappa, rank) is controlled on purpose, so the downstream GD /
AGD / Newton / SGD benchmark has a clean, predictable f(w) to optimize.
"""
from .config import FeatureConfig, WindowConfig
from .build import make_datasets, assemble, Dataset
from .diagnostics import conditioning, format_report

__all__ = [
    "FeatureConfig", "WindowConfig", "Dataset",
    "make_datasets", "assemble", "conditioning", "format_report",
]
