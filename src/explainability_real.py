"""
FraudSentry - Explainability with Real SHAP
===============================================
Replaces the offline substitutes in explainability.py:
  - Global importance: SHAP's mean |SHAP value| per feature, instead of
    permutation_importance.
  - Local (per-alert) explanation: SHAP's actual Shapley-value
    attribution per prediction, instead of the deviation-based
    heuristic. This is the meaningful upgrade -- SHAP accounts for
    feature interactions (e.g., "new_device only matters this much
    WHEN combined with geo_mismatch"), which the deviation heuristic
    cannot represent.

Run after train_models_real.py.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from feature_engineering import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"


def load_artifacts():
    with open(RESULTS_DIR / "pipeline_real.pkl", "rb") as f:
        artifacts = pickle.load(f)
    features = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return artifacts, features


def get_feature_names(pre):
    return (
        FEATURE_COLUMNS +
        list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS))
    )


def main():
    print("[1/5] Loading artifacts and features...", flush=True)
    artifacts, features = load_artifacts()
    pre = artifacts["preprocessor"]
    model = artifacts["xgb_model"]  # SHAP's TreeExplainer works natively on XGBoost

    X = pre.transform(features[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    feature_names = get_feature_names(pre)

    print("[2/5] Building SHAP TreeExplainer...", flush=True)
    explainer = shap.TreeExplainer(model)

    print("[3/5] Computing SHAP values on sample (this can take a minute)...", flush=True)
    sample_idx = np.random.default_rng(42).choice(len(X), size=min(5000, len(X)), replace=False)
    shap_values = explainer.shap_values(X[sample_idx])

    global_importance = sorted(
        zip(feature_names, np.abs(shap_values).mean(axis=0)),
        key=lambda x: -x[1],
    )
    print("=== Global feature importance (SHAP mean |value|, XGBoost) ===")
    for name, val in global_importance[:10]:
        print(f"  {name:30s} {val:+.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "global_importance_shap.json", "w") as f:
        json.dump([{"feature": n, "mean_abs_shap": round(float(v), 5)} for n, v in global_importance], f, indent=2)

    print("[4/5] Saving SHAP summary plot...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, X[sample_idx], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved results/plots/shap_summary.png")
    except Exception as e:
        print(f"Could not save SHAP summary plot: {e}")

    print("[5/5] Computing local explanations for top alerts...", flush=True)
    scored = pd.read_csv(RESULTS_DIR / "test_scored_real.csv")
    top_alerts = scored.sort_values("xgb_score", ascending=False).head(5)

    print("\n=== Sample local SHAP explanations (top 5 highest-risk alerts) ===")
    explanations = []
    for _, row in top_alerts.iterrows():
        feat_row = features[features["transaction_id"] == row["transaction_id"]]
        if feat_row.empty:
            continue
        X_row = pre.transform(feat_row[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
        if hasattr(X_row, "toarray"):
            X_row = X_row.toarray()
        sv = explainer.shap_values(X_row)[0]
        top_factors = sorted(zip(feature_names, sv), key=lambda x: -abs(x[1]))[:3]
        exp = [{"feature": f, "shap_value": round(float(v), 4)} for f, v in top_factors]
        explanations.append({
            "transaction_id": row["transaction_id"],
            "is_fraud_actual": int(row["is_fraud"]),
            "xgb_score": round(float(row["xgb_score"]), 4),
            "top_shap_factors": exp,
        })
        print(f"  {row['transaction_id']} (actual_fraud={row['is_fraud']}, score={row['xgb_score']:.3f}): {exp}")

    with open(RESULTS_DIR / "sample_explanations_shap.json", "w") as f:
        json.dump(explanations, f, indent=2)

    print("\nSaved: results/global_importance_shap.json, results/sample_explanations_shap.json")


if __name__ == "__main__":
    main()
