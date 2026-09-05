"""
FraudSentry - Real Dataset Loader (IEEE-CIS Fraud Detection)
================================================================
Run this on a machine WITH internet access and Kaggle API credentials
configured (see LOCAL_SETUP.md). Downloads the real IEEE-CIS Fraud
Detection dataset via the Kaggle API and maps it onto the SAME feature
schema `feature_engineering.py` already produces, so nothing downstream
(train_models.py, explainability.py, case_tracking.py) needs to change.

IMPORTANT DISCLOSED LIMITATION: IEEE-CIS has no explicit customer_id
column. This loader reconstructs a pseudo-customer identity by hashing
(card1, card2, card3, card5, addr1, D1) -- a documented community
technique for this specific competition (D1 approximates account age
in days, which helps stabilize the grouping), NOT a guaranteed-correct
customer identity. Treat `customer_id` from this loader as a
best-effort proxy, and say so if you present results from it.
"""
import hashlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw_ieee_cis"
OUT_PATH = Path(__file__).resolve().parent / "features.csv"

# Map IEEE-CIS's ProductCD (W, C, R, H, S) onto merchant-category-like labels.
# These are Vesta's actual product/transaction type codes for this dataset;
# the mapping to human labels below is illustrative, not an official Vesta mapping.
PRODUCT_CD_MAP = {
    "W": "online_retail",
    "C": "electronics",
    "R": "travel",
    "H": "entertainment",
    "S": "grocery",
}


def download_if_needed():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    txn_path = RAW_DIR / "train_transaction.csv"
    if txn_path.exists():
        print("Raw IEEE-CIS files already present, skipping download.")
        return

    print("Downloading IEEE-CIS Fraud Detection dataset via Kaggle API...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "ieee-fraud-detection",
         "-p", str(RAW_DIR)],
        check=True,
    )
    zip_path = RAW_DIR / "ieee-fraud-detection.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    print("Downloaded and extracted.")


def reconstruct_pseudo_customer_id(df: pd.DataFrame) -> pd.Series:
    """Documented community heuristic for this competition. See module
    docstring -- this is a best-effort proxy, not a verified identity."""
    key_cols = ["card1", "card2", "card3", "card5", "addr1", "D1"]
    for c in key_cols:
        if c not in df.columns:
            df[c] = -999  # sentinel for missing
    cols_str = df[key_cols].fillna(-999).astype(str)
    key = cols_str[key_cols[0]]
    for c in key_cols[1:]:
        key = key + "_" + cols_str[c]
    return key.apply(lambda s: "CUST" + hashlib.md5(s.encode()).hexdigest()[:10])


def load_and_map():
    download_if_needed()
    txn = pd.read_csv(RAW_DIR / "train_transaction.csv")
    identity_path = RAW_DIR / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        txn = txn.merge(identity, on="TransactionID", how="left")

    df = pd.DataFrame()
    df["transaction_id"] = "TXN" + txn["TransactionID"].astype(str)
    df["customer_id"] = reconstruct_pseudo_customer_id(txn)

    # TransactionDT is seconds relative to an arbitrary reference point (not
    # a real calendar date in the public IEEE-CIS release). We synthesize a
    # relative timestamp from it purely to support the time-based split and
    # hour-of-day features -- hour-of-day derived this way is only as
    # meaningful as the (undisclosed) reference offset Vesta used.
    ref = pd.Timestamp("2025-01-01")
    df["timestamp"] = ref + pd.to_timedelta(txn["TransactionDT"], unit="s")

    df["amount"] = txn["TransactionAmt"]
    df["merchant_category"] = txn["ProductCD"].map(PRODUCT_CD_MAP).fillna("grocery")

    # Geo mismatch proxy: addr1 (billing region) changing transaction-to-transaction
    # for the same pseudo-customer. Real geo mismatch (txn country vs home country)
    # isn't directly available in this dataset -- disclosed approximation.
    df["txn_country"] = txn["addr1"].fillna(-1).astype(str)
    counts = df.groupby(["customer_id", "txn_country"]).size().reset_index(name="cnt")
    counts = counts.sort_values("cnt", ascending=False).drop_duplicates("customer_id")
    mode_map = counts.set_index("customer_id")["txn_country"]
    df["customer_home_country"] = df["customer_id"].map(mode_map)

    # Device novelty: real signal available via DeviceInfo, if the identity
    # table was present. Falls back to 0 (unknown) if identity data missing.
    if "DeviceInfo" in txn.columns:
        df["_device"] = txn["DeviceInfo"].fillna("unknown")
        seen = set()
        new_device_flags = []
        for cust, dev in zip(df["customer_id"], df["_device"]):
            key = (cust, dev)
            new_device_flags.append(int(key not in seen))
            seen.add(key)
        df["new_device"] = new_device_flags
    else:
        df["new_device"] = 0

    df["is_fraud"] = txn["isFraud"]

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Now run this through the EXACT SAME feature_engineering.py used by the
    # synthetic pipeline, so train_models.py needs zero changes.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from feature_engineering import engineer_features

    engineered = engineer_features(df)
    engineered.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(engineered):,} real IEEE-CIS transactions to {OUT_PATH}")
    print(f"Fraud rate: {engineered['is_fraud'].mean()*100:.3f}%")
    print(
        "\nNOTE: 'customer_id' and 'txn_country'/'geo_mismatch' are "
        "reconstructed proxies, not verified ground truth -- see this "
        "file's module docstring before presenting results from real data."
    )


if __name__ == "__main__":
    load_and_map()

