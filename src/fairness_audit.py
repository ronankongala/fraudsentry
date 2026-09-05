"""
FraudSentry - Subgroup Fairness Audit
========================================
Addresses the gap explicitly flagged in dpia.md Section 6: "No subgroup
fairness audit was performed."

Checks whether the model's false-positive rate is roughly consistent
across subgroups of legitimate transactions, rather than concentrated
on one segment (e.g., international customers, a specific merchant
category, a specific device type). A model that's accurate overall but
disproportionately flags one group's legitimate transactions is a real
harm, not just a technical footnote -- this is what GDPR Art. 35's risk
assessment is asking a real DPIA to check for.

IMPORTANT: this checks proxies available in the dataset (merchant
category, transaction country/geography, device novelty), NOT
protected characteristics like race, religion, or nationality directly
-- those aren't present in transaction data and testing for them would
itself raise separate privacy/legal questions (see dpia.md). Geography
and merchant category are reasonable, available proxies for a
first-pass check, not a complete fairness audit.

Run after train_models_real.py (or train_models.py for the offline
synthetic version -- just change SCORE_COL and INPUT paths below).
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SCORE_COL = "xgb_score"  # change to "rf_score" if using the offline synthetic pipeline
TEST_SCORED_PATH = RESULTS_DIR / "test_scored_real.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "features.csv"
TARGET_FPR = 0.03


def threshold_for_target_fpr(y_true, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not valid.any():
        return scores.max()
    return thresholds[np.where(valid)[0][-1]]


def subgroup_fpr(df, group_col, threshold):
    rows = []
    for group_val, sub in df.groupby(group_col):
        legit = sub[sub["is_fraud"] == 0]
        if len(legit) < 30:  # skip tiny groups, not statistically meaningful
            continue
        flagged = (legit[SCORE_COL] >= threshold).sum()
        fpr = flagged / len(legit)
        rows.append({
            "group": str(group_val),
            "n_legitimate_txns": int(len(legit)),
            "n_flagged": int(flagged),
            "false_positive_rate": round(float(fpr), 4),
        })
    return sorted(rows, key=lambda r: -r["false_positive_rate"])


def main():
    print("[1/3] Loading scored test set and features...", flush=True)
    scored = pd.read_csv(TEST_SCORED_PATH)
    features = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    merged = scored.merge(
        features[["transaction_id", "geo_mismatch", "new_device"]],
        on="transaction_id", how="left",
    )

    print("[2/3] Computing threshold and subgroup FPRs...", flush=True)
    threshold = threshold_for_target_fpr(merged["is_fraud"], merged[SCORE_COL], TARGET_FPR)
    print(f"Using threshold={threshold:.4f} (calibrated to overall {TARGET_FPR*100:.0f}% FPR)\n")

    report = {"overall_threshold": round(float(threshold), 4), "subgroups": {}}

    for group_col in ["merchant_category", "geo_mismatch", "new_device"]:
        if group_col not in merged.columns:
            continue
        rows = subgroup_fpr(merged, group_col, threshold)
        report["subgroups"][group_col] = rows
        print(f"=== False positive rate by {group_col} ===")
        for r in rows:
            print(f"  {r['group']:20s} FPR={r['false_positive_rate']*100:5.2f}%  "
                  f"(n_legit={r['n_legitimate_txns']}, flagged={r['n_flagged']})")
        if rows:
            spread = rows[0]["false_positive_rate"] - rows[-1]["false_positive_rate"]
            print(f"  --> spread across groups: {spread*100:.2f} percentage points\n")
            if spread > 0.05:
                print(f"  FLAG: >5 point FPR spread across {group_col} groups -- "
                      f"worth investigating before production use.\n")

    print("[3/3] Saving report...", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "fairness_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved: results/fairness_audit.json")
    print(
        "\nReminder: this checks available proxies (category, geography, device), "
        "not protected characteristics directly. See this file's docstring and "
        "dpia.md Section 6 for what this audit does and does not cover."
    )


if __name__ == "__main__":
    main()
