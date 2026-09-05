"""
FraudSentry - Model Training & Evaluation
============================================
Trains and compares:
  1. Logistic Regression (baseline, class-weighted)
  2. Random Forest (class-weighted + SMOTE-balanced training set)
  3. HistGradientBoostingClassifier (sklearn's native gradient boosting --
     used in place of XGBoost, which could not be installed in this
     offline sandbox; algorithmically the same family of model)
  4. Isolation Forest (unsupervised anomaly detector, trained WITHOUT
     labels, evaluated against them -- this is the "catch fraud patterns
     you haven't labeled yet" layer real fraud systems pair with
     supervised models)

Split strategy: TIME-BASED (train on first 80% of transactions by
timestamp, test on the last 20%). This is deliberate -- a random split
would leak future transaction patterns into training and overstate
performance. Real fraud systems are evaluated this way because fraud
patterns drift over time.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

from feature_engineering import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN
from smote import smote_balance


def load_features(path="/home/claude/fraudsentry/data/features.csv"):
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
    """Find the recall achievable at (or just under) a target false-positive rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0, 1.0
    best_i = np.where(valid)[0][-1]  # highest tpr among thresholds under target FPR
    return tpr[best_i], thresholds[best_i]


def evaluate(y_true, scores, model_name, target_fpr=0.03):
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    recall_at_target, threshold = recall_at_fpr(y_true, scores, target_fpr)
    preds_at_threshold = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_at_threshold).ravel()
    achieved_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    result = {
        "model": model_name,
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        f"recall_at_{int(target_fpr*100)}pct_fpr": round(float(recall_at_target), 4),
        "achieved_fpr": round(float(achieved_fpr), 4),
        "threshold_used": round(float(threshold), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return result


def main():
    df = load_features()
    train_df, test_df = time_based_split(df)

    pre = build_preprocessor()
    X_train = pre.fit_transform(train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    X_test = pre.transform(test_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_test = test_df[LABEL_COLUMN].to_numpy()

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    print(f"Train: {len(y_train):,} txns ({y_train.sum()} fraud)  |  "
          f"Test: {len(y_test):,} txns ({y_test.sum()} fraud)")

    results = []

    # ---- 1. Logistic Regression (class-weighted, no SMOTE) ----
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate(y_test, lr_scores, "LogisticRegression (class-weighted)"))

    # ---- 2. SMOTE-balanced training set for RF and GB ----
    X_train_bal, y_train_bal = smote_balance(X_train, y_train, target_ratio=0.5)
    print(f"After SMOTE: {len(y_train_bal):,} training rows "
          f"({int(y_train_bal.sum())} fraud, {y_train_bal.mean()*100:.1f}%)")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_bal, y_train_bal)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate(y_test, rf_scores, "RandomForest (SMOTE-balanced)"))

    # ---- 3. HistGradientBoostingClassifier (XGBoost-family substitute) ----
    gb = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=300, random_state=42,
    )
    gb.fit(X_train_bal, y_train_bal)
    gb_scores = gb.predict_proba(X_test)[:, 1]
    results.append(evaluate(y_test, gb_scores, "HistGradientBoosting (SMOTE-balanced)"))

    # ---- 4. Isolation Forest (unsupervised, no labels used in training) ----
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_train)  # labels NOT used
    iso_scores = -iso.score_samples(X_test)  # higher = more anomalous
    results.append(evaluate(y_test, iso_scores, "IsolationForest (unsupervised)"))

    print("\n=== Results (target FPR = 3%) ===")
    for r in results:
        print(json.dumps(r, indent=2))

    with open("/home/claude/fraudsentry/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save the best supervised model + preprocessor + test predictions for downstream
    # explainability / case-tracking steps
    best = max(results[:3], key=lambda r: r[f"recall_at_3pct_fpr"])
    print(f"\nBest supervised model by recall@3%FPR: {best['model']}")

    import pickle
    with open("/home/claude/fraudsentry/results/pipeline.pkl", "wb") as f:
        pickle.dump({
            "preprocessor": pre,
            "rf_model": rf,
            "gb_model": gb,
            "lr_model": lr,
            "iso_model": iso,
            "feature_columns": FEATURE_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
        }, f)

    test_out = test_df[["transaction_id", "customer_id", "timestamp", "amount",
                         "merchant_category", "is_fraud"]].copy()
    test_out["lr_score"] = lr_scores
    test_out["rf_score"] = rf_scores
    test_out["gb_score"] = gb_scores
    test_out["iso_score"] = iso_scores
    test_out.to_csv("/home/claude/fraudsentry/results/test_scored.csv", index=False)
    print("Saved: results/metrics.json, results/pipeline.pkl, results/test_scored.csv")


if __name__ == "__main__":
    main()
