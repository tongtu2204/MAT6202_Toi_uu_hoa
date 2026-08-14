"""Pipeline configuration: paths, time windows, and feature-engineering knobs.

The whole project is an optimization test bed (see CLAUDE.md): every choice here
is made to control the conditioning of the design matrix X, not to maximize churn
AUC. Where prediction quality and optimization behaviour conflict, optimization wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# data/ lives next to this package
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Customer ids are 8-char zero-padded strings in the CSVs but plain ints in the
# xlsx files (Excel stripped the leading zeros). Everything is normalized to this.
ID_WIDTH = 8
ID_COL = "CUSTOMER_NUMBER"

# Raw file names (kept verbatim, including the typo-prone TRANS_ vs TRXN_ headers).
FILES = {
    "customer": "1.Data_Customer.csv",
    "transaction": "2.Data_MyVIB_Transaction.csv",
    "activity": "3.Data_MyVIB_Activity.csv",
    "deposit": "4.Data_Deposit.csv",
    "lending": "5.Data_Lending.xlsx",
    "card": "6.Data_Card.xlsx",
}


@dataclass
class WindowConfig:
    """Three disjoint time windows around the snapshot T (Step 0 of the spec).

    All raw data lives in calendar 2019. K=3 (not 6) so that THREE such
    (observation, outcome) pairs fit inside the year — see FeatureConfig.snapshots.

    BOTH windows are half-open on the left: observation is (T-K, T], outcome is
    (T, T+H]. Two reasons, and the first is not cosmetic:
      * the earliest snapshot is T=2019-03-31, whose observation window would
        otherwise reach back to 2018-12-31 — a date the raw data does not cover.
        Excluding the left endpoint keeps every window strictly inside 2019.
      * the endpoints then tile exactly: (12-31, 03-31], (03-31, 06-30],
        (06-30, 09-30], so no calendar day is counted in two windows.
    Offsets are MonthEnd, not DateOffset(months=): from 2019-06-30, DateOffset gives
    2019-03-30 and lets a stray 2019-03-31 into the window, which would make that one
    snapshot span four calendar months instead of three and unalign the M0..M2 block.
    """

    snapshot: str = "2019-09-30"          # T: the "standing-and-looking" instant
    observation_months: int = 3           # K: features come ONLY from (T-K, T]
    outcome_months: int = 3               # H: label comes ONLY from (T, T+H]

    @property
    def T(self):
        import pandas as pd
        return pd.Timestamp(self.snapshot)

    @property
    def obs_start(self):
        import pandas as pd
        return self.T - pd.offsets.MonthEnd(self.observation_months)

    @property
    def outcome_end(self):
        import pandas as pd
        return self.T + pd.offsets.MonthEnd(self.outcome_months)

    @property
    def month_buckets(self) -> int:
        """Calendar months the observation window touches — exactly K, because the
        window is half-open on the left and both endpoints are month ends.

        Per-month features are named M0..M{K-1} RELATIVE to the window so every
        snapshot yields the same column layout; a calendar name would make the
        stacked matrix ragged.
        """
        return self.observation_months


@dataclass
class FeatureConfig:
    windows: WindowConfig = field(default_factory=WindowConfig)

    # Step 0b — MULTI-SNAPSHOT STACKING. One row per (customer, snapshot).
    #
    # With K=3 and H=3, three disjoint (observation, outcome) pairs fit inside 2019:
    #   T=2019-03-31: obs [2018-12-31, 03-31], outcome (03-31, 06-30]
    #   T=2019-06-30: obs [2019-03-30, 06-30], outcome (06-30, 09-30]
    #   T=2019-09-30: obs [2019-06-30, 09-30], outcome (09-30, 12-31]
    # Stacking them raises n from 44,026 (single snapshot) to ~75,000. Note a shorter
    # K by itself SHRINKS n — eligibility means "active in [T-K, T]", so K=3 is a
    # subset of K=6; the stacking is what buys the observations back and then some.
    #
    # Caveat to state on the slide: a customer can appear at more than one snapshot,
    # so the rows are not iid. For an OPTIMIZATION study that is irrelevant — f(w) is
    # defined by whatever (X, y) it is handed, and L, mu, kappa, iteration counts and
    # wall-clock are all properties of that matrix. It would matter for inference.
    snapshots: tuple[str, ...] = ("2019-03-31", "2019-06-30", "2019-09-30")

    # Step 6 — collinearity pruning thresholds.
    corr_threshold: float = 0.98          # drop one of any |rho| > this pair
    vif_threshold: float = 100.0          # iteratively drop highest VIF until all below

    # Step 7 — degree-2 interactions on a small behavioural core only (a kappa knob).
    poly_core_size: int = 8               # top-k features by variance get squared/crossed
    add_polynomial: bool = True

    # Split + reproducibility.
    # This is an OPTIMIZATION study: f(w) is defined by whatever (X, y) we hand it,
    # so a hold-out buys nothing — it only shrinks n. Default is therefore no split
    # (the full eligible population becomes X_train). Pass --holdout to run_pipeline
    # to get the old 75/25 behaviour back (the only thing it enables is reporting
    # out-of-sample AUC/F1 in the appendix).
    use_holdout: bool = False
    test_size: float = 0.25
    random_state: int = 42

    # Regularization used purely for reporting L, mu, kappa. Single source of truth:
    # kept equal to the optimizer/deck default (1e-3) so the pipeline's printed kappa
    # matches the benchmark. X itself is independent of lam; the benchmark may sweep it.
    lam: float = 1e-3
