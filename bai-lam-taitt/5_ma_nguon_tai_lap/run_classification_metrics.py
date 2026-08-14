#!/usr/bin/env python3
"""Classification metrics for the appendix table.

Since the pipeline now trains on the full eligible population (no hold-out —
see churn_opt.config.FeatureConfig.use_holdout), these are **in-sample** metrics
and the deck labels them as such. They are reference-only: the object of study is
the optimizer, not the classifier.

    python run_classification_metrics.py            # -> artifacts/metrics.json
"""
from __future__ import annotations

import blas_threads  # noqa: F401  (must precede numpy: pins BLAS to 1 thread)
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from optim.data import load_variant
from optim.objective import LogisticObjective
from optim.optimizers.newton import newton

ART = Path(__file__).resolve().parent / "artifacts"


def main(variant: str = "ridge", lam: float = 1e-3) -> None:
    ds = load_variant(variant)
    obj = LogisticObjective(ds.X_train, ds.y_train, lam=lam)
    res = newton(obj, np.zeros(obj.d), max_iter=100, tol=1e-13)
    w = res.w

    p = 1.0 / (1.0 + np.exp(-(obj.X @ w)))
    yhat = (p >= 0.5).astype(float)
    y = ds.y_train

    out = {
        "variant": variant,
        "lam": lam,
        "n": int(len(y)),
        "d": int(obj.d),
        "in_sample": True,
        "churn_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat)),
        "recall": float(recall_score(y, yhat)),
    }
    (ART / "metrics.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
