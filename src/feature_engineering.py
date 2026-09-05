"""
FraudSentry - Feature Engineering
==================================
Builds the model-facing feature set from raw transactions.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
DEFAULT_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "features.csv"


def load_raw(path=None):
    if path is None:
        path = DEFAULT_RAW_PATH
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _p(msg):
    print(msg, flush=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    _p(f"[1/6] Sorting {len(df):,} rows by customer/time...")
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    _p("[2/6] Computing transaction velocity (rolling 1h count)...")
    df["_ones"] = 1
    roll = df.groupby("customer_id").rolling("3600s", on="timestamp")["_ones"].count()
    df["txn_velocity_1h"] = (roll.reset_index(level=0, drop=True).astype(int) - 1).values
    df = df.drop(columns=["_ones"])

    _p("[3/6] Computing amount z-score vs customer history...")
    g_amount = df.groupby("customer_id")["amount"]
    exp_mean = g_amount.expanding().mean().reset_index(level=0, drop=True)
    exp_std = g_amount.expanding().std().reset_index(level=0, drop=True)
    exp_mean_shifted = exp_mean.groupby(df["customer_id"]).shift(1)
    exp_std_shifted = exp_std.groupby(df["customer_id"]).shift(1).fillna(1).replace(0, 1)
    df["amount_zscore"] = ((df["amount"] - exp_mean_shifted) / exp_std_shifted).fillna(0)

    _p("[4/6] Computing geo mismatch flag...")
    df["geo_mismatch"] = (df["txn_country"] != df["customer_home_country"]).astype(int)

    _p("[5/6] Computing temporal features...")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    _p("[6/6] Encoding merchant category...")
    df["merchant_category"] = df["merchant_category"].astype("category")

    _p("Feature engineering complete.")
    return df


FEATURE_COLUMNS = [
    "amount", "txn_velocity_1h", "amount_zscore", "geo_mismatch",
    "new_device", "hour_of_day", "is_night", "day_of_week",
]
CATEGORICAL_COLUMNS = ["merchant_category"]
LABEL_COLUMN = "is_fraud"


if __name__ == "__main__":
    raw = load_raw()
    feat = engineer_features(raw)
    feat.to_csv(DEFAULT_OUT_PATH, index=False)
    print(f"Engineered features for {len(feat):,} transactions -> {DEFAULT_OUT_PATH}")
    print(feat[FEATURE_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN]].head())
