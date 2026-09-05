"""
FraudSentry - Model Training with Real Libraries
====================================================
Same logic as train_models.py, but using the REAL libraries this
project's offline version substituted for:
  - xgboost.XGBClassifier instead of HistGradientBoostingClassifier
  - imblearn.over_sampling.SMOTE instead of the from-scratch smote.py

Run this only after installing requirements-local.txt (see
LOCAL_SETUP.md) and after data/features.csv contains either the
synthetic data (data/generate_data.py) or the real IEEE-CIS data
(data/load_ieee_cis.py).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from feature_engineering import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN

DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_features(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURE_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def time_based_split(df, train_frac=0.8):
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def recall_at_fpr(y_true, scores, target_fpr=0.03):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0, 1.0
    best_i = np.where(valid)[0][-1]
    return tpr[best_i], thresholds[best_i]


def evaluate(y_true, scores, model_name, target_fpr=0.03):
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    recall_at_target, threshold = recall_at_fpr(y_true, scores, target_fpr)
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    achieved_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "model": model_name,
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        f"recall_at_{int(target_fpr*100)}pct_fpr": round(float(recall_at_target), 4),
        "achieved_fpr": round(float(achieved_fpr), 4),
        "threshold_used": round(float(threshold), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    print("[1/7] Loading features...", flush=True)
    df = load_features()
    train_df, test_df = time_based_split(df)

    print("[2/7] Building preprocessor and transforming data...", flush=True)
    pre = build_preprocessor()
    X_train = pre.fit_transform(train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    X_test = pre.transform(test_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_test = test_df[LABEL_COLUMN].to_numpy()
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    print(f"Train: {len(y_train):,} ({y_train.sum()} fraud) | Test: {len(y_test):,} ({y_test.sum()} fraud)", flush=True)

    results = []

    print("[3/7] Training LogisticRegression...", flush=True)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    results.append(evaluate(y_test, lr.predict_proba(X_test)[:, 1], "LogisticRegression"))

    print("[4/7] Running imblearn SMOTE oversampling...", flush=True)
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After imblearn SMOTE: {len(y_train_bal):,} rows ({int(y_train_bal.sum())} fraud)", flush=True)

    print("[5/7] Training RandomForest...", flush=True)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced_subsample",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)
    results.append(evaluate(y_test, rf.predict_proba(X_test)[:, 1], "RandomForest (imblearn SMOTE)"))

    print("[6/7] Training XGBoost...", flush=True)
    xgb_model = xgb.XGBClassifier(
        max_depth=8, learning_rate=0.08, n_estimators=300,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train_bal, y_train_bal)
    results.append(evaluate(y_test, xgb_model.predict_proba(X_test)[:, 1], "XGBoost (imblearn SMOTE)"))

    print("[7/7] Training IsolationForest...", flush=True)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    results.append(evaluate(y_test, iso_scores, "IsolationForest (unsupervised)"))

    print("\n=== Results (real libraries, target FPR = 3%) ===", flush=True)
    for r in results:
        print(json.dumps(r, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "metrics_real.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "pipeline_real.pkl", "wb") as f:
        pickle.dump({
            "preprocessor": pre, "rf_model": rf, "xgb_model": xgb_model,
            "lr_model": lr, "iso_model": iso,
            "feature_columns": FEATURE_COLUMNS, "categorical_columns": CATEGORICAL_COLUMNS,
        }, f)

    test_out = test_df[["transaction_id", "customer_id", "timestamp", "amount",
                         "merchant_category", "is_fraud"]].copy()
    test_out["rf_score"] = rf.predict_proba(X_test)[:, 1]
    test_out["xgb_score"] = xgb_model.predict_proba(X_test)[:, 1]
    test_out.to_csv(RESULTS_DIR / "test_scored_real.csv", index=False)
    print("Saved: results/metrics_real.json, results/pipeline_real.pkl, results/test_scored_real.csv", flush=True)


if __name__ == "__main__":
    main()
