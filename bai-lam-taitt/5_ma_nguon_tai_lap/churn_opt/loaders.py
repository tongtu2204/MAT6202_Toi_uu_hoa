"""Raw table loaders with id normalization and date parsing.

Each loader returns a tidy DataFrame with a canonical 8-char string CUSTOMER_NUMBER
and parsed datetimes. Date formats differ across files (Activity is M/D/YYYY; the
transaction/customer CSVs are YYYY-MM-DD), so parsing is per-file, not global.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, FILES, ID_COL, ID_WIDTH


def _norm_id(s: pd.Series) -> pd.Series:
    """Coerce any CUSTOMER_NUMBER representation to a zero-padded 8-char string."""
    # xlsx gives ints/floats (leading zeros lost); CSVs give padded strings.
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype(str)
    else:
        s = s.astype(str).str.strip()
        # strip any accidental ".0" from float-parsed ids
        s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(ID_WIDTH)


def load_customer() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / FILES["customer"], dtype={ID_COL: str})
    df[ID_COL] = _norm_id(df[ID_COL])
    for col in ["CLIENT_CREATE_DATE", "DATE_OF_BIRTH", "IB_REGISTER_DATE"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_transaction() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / FILES["transaction"],
        dtype={ID_COL: str, "TRANS_LV1": "category", "TRANS_LV2": "category",
               "DAY_OF_WEEK": "category"},
    )
    df[ID_COL] = _norm_id(df[ID_COL])
    df["TRANS_DATE"] = pd.to_datetime(df["TRANS_DATE"], errors="coerce")
    return df


def load_activity() -> pd.DataFrame:
    # 16M rows — read only what we use and parse the M/D/YYYY format explicitly.
    df = pd.read_csv(
        DATA_DIR / FILES["activity"],
        usecols=["ACTIVITY_DATE", "ACTIVITY_HOUR", "ACTIVITY_NAME", ID_COL],
        dtype={ID_COL: str, "ACTIVITY_NAME": "category"},
    )
    df[ID_COL] = _norm_id(df[ID_COL])
    df["ACTIVITY_DATE"] = pd.to_datetime(df["ACTIVITY_DATE"], format="%m/%d/%Y",
                                         errors="coerce")
    return df


def load_deposit() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / FILES["deposit"], dtype={ID_COL: str})
    df[ID_COL] = _norm_id(df[ID_COL])
    df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")
    return df


def load_lending() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / FILES["lending"])
    df[ID_COL] = _norm_id(df[ID_COL])
    df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")
    return df


def load_card() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / FILES["card"])
    df[ID_COL] = _norm_id(df[ID_COL])
    df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")
    return df
