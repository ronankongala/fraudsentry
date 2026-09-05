"""
FraudSentry - Post-Incident Investigation & Trend Analysis
=============================================================
Takes the alerts that were resolved as confirmed_fraud (see
case_tracking.py) and looks for shared characteristics across them --
this is the "trend analysis to identify patterns and strengthen future
prevention efforts" piece of the JD, done for real against the model's
actual confirmed-fraud outputs rather than the ground-truth labels
directly (i.e., this analyzes what the SYSTEM caught and confirmed,
which is the realistic fraud-ops workflow).
"""
import json
import sqlite3
import pandas as pd
from collections import Counter

from case_tracking import DB_PATH


def load_confirmed_fraud_cases():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM alerts WHERE resolution = 'confirmed_fraud'", conn
    )
    conn.close()
    return df


def trend_analysis(cases: pd.DataFrame, features: pd.DataFrame):
    if cases.empty:
        return {"note": "No confirmed fraud cases in this run to analyze."}

    merged = cases.merge(
        features, on="transaction_id", how="left", suffixes=("", "_feat")
    )

    trends = {
        "total_confirmed_cases": len(merged),
        "top_merchant_categories": Counter(merged["merchant_category"].astype(str)).most_common(5),
        "pct_geo_mismatch": round(float(merged["geo_mismatch"].mean()) * 100, 1),
        "pct_new_device": round(float(merged["new_device"].mean()) * 100, 1),
        "pct_at_night": round(float(merged["is_night"].mean()) * 100, 1),
        "avg_amount": round(float(merged["amount"].mean()), 2),
        "avg_txn_velocity_1h": round(float(merged["txn_velocity_1h"].mean()), 2),
        "unique_customers_affected": int(merged["customer_id"].nunique()),
    }
    return trends


def write_case_narratives(cases: pd.DataFrame, features: pd.DataFrame, n=3):
    """Produce short, human-readable investigation write-ups for a sample
    of confirmed cases -- the JD asks specifically for recommended remedial
    actions communicated to stakeholders, not just a model score."""
    narratives = []
    sample = cases.head(n)
    for _, case in sample.iterrows():
        feat_row = features[features["transaction_id"] == case["transaction_id"]]
        if feat_row.empty:
            continue
        feat_row = feat_row.iloc[0]
        factors = json.loads(case["top_factors_json"]) if case["top_factors_json"] else []

        narrative = {
            "case_id": int(case["alert_id"]),
            "transaction_id": case["transaction_id"],
            "customer_id": case["customer_id"],
            "amount": float(feat_row["amount"]),
            "merchant_category": str(feat_row["merchant_category"]),
            "model_score": round(float(case["score"]), 3),
            "contributing_factors": factors,
            "summary": (
                f"Transaction {case['transaction_id']} for customer {case['customer_id']} "
                f"was flagged with a model score of {case['score']:.3f}. "
                f"Primary contributing factors: "
                + ", ".join(f"{f['feature']} ({f['direction']}, z={f['z_score']})" for f in factors)
                + f". Merchant category: {feat_row['merchant_category']}, amount: ${feat_row['amount']:.2f}."
            ),
            "recommended_action": (
                "Confirm cardholder identity via out-of-band verification before restoring "
                "full account privileges; monitor customer's account for 30 days for recurrence; "
                "consider temporary velocity limits on new-device transactions for this customer."
            ),
        }
        narratives.append(narrative)
    return narratives


if __name__ == "__main__":
    cases = load_confirmed_fraud_cases()
    features = pd.read_csv("/home/claude/fraudsentry/data/features.csv", parse_dates=["timestamp"])

    trends = trend_analysis(cases, features)
    print("=== Trend Analysis (confirmed fraud cases) ===")
    print(json.dumps(trends, indent=2, default=str))

    narratives = write_case_narratives(cases, features)
    print("\n=== Sample Case Narratives ===")
    for n in narratives:
        print(f"\n--- Case {n['case_id']} ---")
        print(n["summary"])
        print(f"Recommended action: {n['recommended_action']}")

    with open("/home/claude/fraudsentry/results/trend_analysis.json", "w") as f:
        json.dump(trends, f, indent=2, default=str)
    with open("/home/claude/fraudsentry/results/case_narratives.json", "w") as f:
        json.dump(narratives, f, indent=2)
    print("\nSaved: results/trend_analysis.json, results/case_narratives.json")
