"""
FraudSentry - Subgroup Fairness Audit (Offline / Synthetic Data Version)
============================================================================
This is the version of fairness_audit.py that runs NOW, against the
synthetic data and offline pipeline already in this project -- it does
not require internet access or the real IEEE-CIS dataset.

See fairness_audit.py's docstring for the full methodology notes and
caveats (proxies vs. protected characteristics, etc.) -- they apply
here identically. The only difference is this script points at
results/test_scored.csv (RandomForest score, offline pipeline) instead
of results/test_scored_real.csv (XGBoost score, real-library pipeline).

Run this any time after train_models.py has been run.
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

SCORE_COL = "rf_score"
TEST_SCORED_PATH = "/home/claude/fraudsentry/results/test_scored.csv"
FEATURES_PATH = "/home/claude/fraudsentry/data/features.csv"
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
        if len(legit) < 30:
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
    scored = pd.read_csv(TEST_SCORED_PATH)
    features = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    merged = scored.merge(
        features[["transaction_id", "geo_mismatch", "new_device"]],
        on="transaction_id", how="left",
    )

    threshold = threshold_for_target_fpr(merged["is_fraud"], merged[SCORE_COL], TARGET_FPR)
    print(f"Using threshold={threshold:.4f} (calibrated to overall {TARGET_FPR*100:.0f}% FPR)\n")

    report = {"overall_threshold": round(float(threshold), 4), "subgroups": {}, "flags": []}

    for group_col in ["merchant_category", "geo_mismatch", "new_device"]:
        rows = subgroup_fpr(merged, group_col, threshold)
        report["subgroups"][group_col] = rows
        print(f"=== False positive rate by {group_col} ===")
        for r in rows:
            print(f"  {r['group']:20s} FPR={r['false_positive_rate']*100:6.2f}%  "
                  f"(n_legit={r['n_legitimate_txns']}, flagged={r['n_flagged']})")
        if rows:
            spread = rows[0]["false_positive_rate"] - rows[-1]["false_positive_rate"]
            print(f"  --> spread across groups: {spread*100:.2f} percentage points\n")
            if spread > 0.05:
                flag_msg = (
                    f">5-point FPR spread across '{group_col}' groups "
                    f"({spread*100:.1f} points) -- worth investigating before "
                    f"production use, not just a modeling footnote."
                )
                print(f"  FLAG: {flag_msg}\n")
                report["flags"].append({"group_col": group_col, "spread_pct_points": round(spread*100, 1), "message": flag_msg})

    with open("/home/claude/fraudsentry/results/fairness_audit_synthetic.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved: results/fairness_audit_synthetic.json")


if __name__ == "__main__":
    main()
