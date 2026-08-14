"""Pin BLAS/OpenMP to a single thread for REPRODUCIBLE wall-clock timing.

Import this FIRST — before numpy (hence before `optim`/`churn_opt`) — because the
thread-count env vars are read once when the BLAS backend loads. `setdefault` means
an explicit `export OPENBLAS_NUM_THREADS=8` still wins, so this only sets a floor.

    import blas_threads   # noqa: F401  (must precede numpy)
    blas_threads.verify()  # optional: assert it actually took effect

Setting the variables is NOT proof that it worked: if something already imported
numpy (a conftest, a notebook, another module), the backend is loaded and the
variables are ignored — the pin fails silently and every reported time becomes
unreproducible. So this module refuses to fail quietly: it warns at import time if
numpy was already loaded, and `verify()` interrogates the live BLAS backend
through threadpoolctl instead of trusting the environment.
"""
from __future__ import annotations

import os
import sys
import warnings

_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

#: True if numpy was already imported when this module ran -> the pin is a no-op.
IMPORTED_LATE = "numpy" in sys.modules

for _v in _VARS:
    os.environ.setdefault(_v, "1")

if IMPORTED_LATE:
    warnings.warn(
        "blas_threads was imported AFTER numpy: the BLAS backend has already "
        "read its thread count, so this pin had no effect and wall-clock "
        "timings are not reproducible. Import blas_threads first.",
        RuntimeWarning, stacklevel=2)


def threads() -> dict[str, int]:
    """Actual thread count per loaded native backend (needs threadpoolctl)."""
    from threadpoolctl import threadpool_info
    return {i.get("internal_api", "?"): int(i["num_threads"])
            for i in threadpool_info()}


def verify(expected: int = 1, strict: bool = False) -> dict[str, int]:
    """Check the LIVE backends really run on `expected` threads.

    Returns the per-backend counts. Warns (or raises, with strict=True) on any
    mismatch — call this from a timing driver so a broken pin shows up in the
    log next to the numbers it invalidates, rather than never.
    """
    try:
        info = threads()
    except ImportError:                     # threadpoolctl ships with sklearn
        warnings.warn("threadpoolctl not installed: cannot verify BLAS threads.",
                      RuntimeWarning, stacklevel=2)
        return {}
    bad = {k: v for k, v in info.items() if v != expected}
    if bad:
        msg = (f"BLAS thread pin FAILED: {bad} (expected {expected} thread(s)); "
               f"numpy already imported at pin time: {IMPORTED_LATE}. "
               "Wall-clock timings from this run are not comparable.")
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return info
