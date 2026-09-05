"""
FraudSentry - Synthetic Transaction Data Generator
====================================================
NOTE ON DATA PROVENANCE (read this first):
This sandbox environment has no internet access, so the originally-planned
public datasets (Kaggle Credit Card Fraud Detection / IEEE-CIS Fraud
Detection) could not be downloaded. Instead, this script generates a
SYNTHETIC transaction dataset with fraud patterns modeled on well-documented,
publicly known characteristics of real card-fraud data (low base rate,
elevated fraud likelihood at night, cross-border and new-device
transactions, and unusual transaction velocity/amount).

This is disclosed plainly in the README. The modeling pipeline downstream
(feature engineering, imbalanced classification, explainability, alerting)
is built exactly as it would be against real data -- only the data source
differs. If real data becomes available later (e.g. via Kaggle on a
machine with internet access), this generator can be swapped out for a
loader with no other pipeline changes required.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_CUSTOMERS = 8_000
N_TRANSACTIONS = 150_000
FRAUD_RATE_TARGET = 0.0045  # ~0.45%, in line with published card-fraud base rates

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "pharmacy", "utilities",
    "electronics", "online_retail", "travel", "jewelry", "entertainment",
]
# Baseline fraud propensity by category (jewelry/electronics/online_retail/travel
# are documented as higher-risk categories in card-fraud literature)
CATEGORY_RISK = {
    "grocery": 0.2, "gas_station": 0.3, "restaurant": 0.3, "pharmacy": 0.2,
    "utilities": 0.1, "electronics": 1.6, "online_retail": 1.8,
    "travel": 1.7, "jewelry": 2.2, "entertainment": 0.6,
}
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "NG", "RO", "VN", "BR", "IN"]
HIGH_RISK_COUNTRIES = {"NG", "RO", "VN"}  # used only to shape synthetic signal


def make_customers(n):
    home_country = RNG.choice(COUNTRIES, size=n, p=[
        0.55, 0.10, 0.08, 0.06, 0.05, 0.03, 0.03, 0.03, 0.04, 0.03
    ])
    avg_amount = np.clip(RNG.normal(65, 30, size=n), 5, 400)
    return pd.DataFrame({
        "customer_id": [f"CUST{i:06d}" for i in range(n)],
        "home_country": home_country,
        "typical_amount_mean": avg_amount,
        "typical_amount_std": avg_amount * RNG.uniform(0.15, 0.35, size=n),
    })


def simulate():
    customers = make_customers(N_CUSTOMERS)
    cust_idx = RNG.integers(0, N_CUSTOMERS, size=N_TRANSACTIONS)
    cust = customers.iloc[cust_idx].reset_index(drop=True)

    start = datetime(2025, 1, 1)
    minutes_offset = np.sort(RNG.integers(0, 60 * 24 * 180, size=N_TRANSACTIONS))
    timestamps = [start + timedelta(minutes=int(m)) for m in minutes_offset]

    merchant_category = RNG.choice(MERCHANT_CATEGORIES, size=N_TRANSACTIONS)
    category_risk = np.array([CATEGORY_RISK[c] for c in merchant_category])

    txn_country = cust["home_country"].to_numpy().copy()
    cross_border_mask = RNG.random(N_TRANSACTIONS) < 0.06
    txn_country[cross_border_mask] = RNG.choice(COUNTRIES, size=cross_border_mask.sum())

    new_device = (RNG.random(N_TRANSACTIONS) < 0.08).astype(int)

    amount = np.clip(
        RNG.normal(cust["typical_amount_mean"], cust["typical_amount_std"] + 1e-6),
        1, 5000,
    )

    hour_of_day = np.array([t.hour for t in timestamps])
    is_night = ((hour_of_day >= 23) | (hour_of_day <= 5)).astype(int)

    # ---- fraud propensity score (drives label, not visible to model directly) ----
    geo_mismatch = (txn_country != cust["home_country"].to_numpy()).astype(int)
    high_risk_country = np.isin(txn_country, list(HIGH_RISK_COUNTRIES)).astype(int)

    risk_score = (
        0.35 * category_risk
        + 1.4 * geo_mismatch
        + 1.1 * high_risk_country
        + 0.9 * new_device
        + 0.6 * is_night
        + 0.02 * (amount / (cust["typical_amount_mean"].to_numpy() + 1))
    )
    risk_score += RNG.normal(0, 0.5, size=N_TRANSACTIONS)  # noise so it's learnable, not deterministic

    # Calibrate threshold so overall fraud rate lands near FRAUD_RATE_TARGET
    threshold = np.quantile(risk_score, 1 - FRAUD_RATE_TARGET)
    is_fraud = (risk_score >= threshold).astype(int)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(N_TRANSACTIONS)],
        "customer_id": cust["customer_id"].to_numpy(),
        "timestamp": timestamps,
        "amount": np.round(amount, 2),
        "merchant_category": merchant_category,
        "txn_country": txn_country,
        "customer_home_country": cust["home_country"].to_numpy(),
        "new_device": new_device,
        "is_fraud": is_fraud,
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = simulate()
    out_path = "/home/claude/fraudsentry/data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} transactions to {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%  ({df['is_fraud'].sum():,} fraudulent)")
