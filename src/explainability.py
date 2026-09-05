"""
FraudSentry - Explainability
==============================
NOTE ON METHOD: `shap` could not be installed in this offline sandbox.
Rather than skip explainability, this module uses two dependency-free
substitutes that are honest, defensible alternatives:

  1. GLOBAL explainability: sklearn's `permutation_importance` -- a
     model-agnostic method that measures how much each feature's
     contribution matters by shuffling it and watching performance drop.
     This is a real, published technique (Breiman 2001 for the concept;
     sklearn.inspection.permutation_importance is the standard library
     implementation), not a placeholder.

  2. LOCAL (per-alert) explainability: a deviation-based heuristic --
     for each flagged transaction, report which features deviate most
     (in standard-deviation terms) from the legitimate-transaction
     population. This is simpler than SHAP's Shapley-value attribution
     and does NOT account for feature interactions the way SHAP does,
     but it gives an analyst a genuine, interpretable "why was this
     flagged" answer for each alert. This limitation is disclosed here
     and in the README -- a production system would upgrade to SHAP's
     TreeExplainer for game-theoretically fair per-prediction attribution.
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from feature_engineering import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN


def load_artifacts():
    with open("/home/claude/fraudsentry/results/pipeline.pkl", "rb") as f:
        artifacts = pickle.load(f)
    features = pd.read_csv("/home/claude/fraudsentry/data/features.csv", parse_dates=["timestamp"])
    return artifacts, features


def global_importance(artifacts, features, model_key="lr_model", n_repeats=8):
    pre = artifacts["preprocessor"]
    model = artifacts[model_key]
    X = pre.transform(features[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = features[LABEL_COLUMN].to_numpy()

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42,
        scoring="average_precision", n_jobs=-1,
    )
    feature_names = (
        FEATURE_COLUMNS +
        list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS))
    )
    ranked = sorted(
        zip(feature_names, result.importances_mean),
        key=lambda x: -x[1],
    )
    return ranked


def local_explanation(row: pd.Series, population_stats: dict, top_k=3):
    """Rank this transaction's numeric features by how many std devs they
    deviate from the legitimate-population mean. Returns top_k contributing
    factors as human-readable strings."""
    deviations = []
    for col in FEATURE_COLUMNS:
        mean, std = population_stats[col]
        std = std if std > 1e-9 else 1.0
        z = (row[col] - mean) / std
        deviations.append((col, z))
    deviations.sort(key=lambda x: -abs(x[1]))
    top = deviations[:top_k]
    return [
        {"feature": col, "z_score": round(float(z), 2),
         "direction": "above normal" if z > 0 else "below normal"}
        for col, z in top
    ]


def compute_population_stats(features: pd.DataFrame):
    legit = features[features[LABEL_COLUMN] == 0]
    return {col: (legit[col].mean(), legit[col].std()) for col in FEATURE_COLUMNS}


if __name__ == "__main__":
    artifacts, features = load_artifacts()

    print("=== Global feature importance (permutation, LogisticRegression) ===")
    ranked = global_importance(artifacts, features)
    for name, score in ranked[:10]:
        print(f"  {name:30s} {score:+.4f}")

    with open("/home/claude/fraudsentry/results/global_importance.json", "w") as f:
        json.dump([{"feature": n, "importance": round(float(s), 5)} for n, s in ranked], f, indent=2)

    stats = compute_population_stats(features)
    scored = pd.read_csv("/home/claude/fraudsentry/results/test_scored.csv")
    top_alerts = scored.sort_values("rf_score", ascending=False).head(5)

    print("\n=== Sample local explanations (top 5 highest-risk test alerts) ===")
    explanations = []
    for _, row in top_alerts.iterrows():
        full_row = features[features["transaction_id"] == row["transaction_id"]].iloc[0]
        exp = local_explanation(full_row, stats)
        explanations.append({"transaction_id": row["transaction_id"], "is_fraud_actual": int(row["is_fraud"]),
                              "rf_score": round(float(row["rf_score"]), 4), "top_factors": exp})
        print(f"  {row['transaction_id']} (actual_fraud={row['is_fraud']}, score={row['rf_score']:.3f}): {exp}")

    with open("/home/claude/fraudsentry/results/sample_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
