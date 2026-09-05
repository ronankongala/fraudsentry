"""
FraudSentry - Visualizations for README (REAL DATA VERSION)
================================================================
Generates charts from the real IEEE-CIS pipeline outputs:
train_models_real.py + explainability_real.py + fairness_audit.py.
Saves to results/plots_real/ so the original synthetic charts in
results/plots/ are preserved for comparison.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots_real"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SCORED_PATH = RESULTS_DIR / "test_scored_real.csv"
METRICS_PATH = RESULTS_DIR / "metrics_real.json"
SHAP_IMPORTANCE_PATH = RESULTS_DIR / "global_importance_shap.json"
FAIRNESS_PATH = RESULTS_DIR / "fairness_audit.json"

# Only these two models have per-row scores saved in test_scored_real.csv
SCORE_COLS = {
    "Random Forest": "rf_score",
    "XGBoost": "xgb_score",
}


def plot_roc_curves():
    from sklearn.metrics import roc_curve, auc
    scored = pd.read_csv(TEST_SCORED_PATH)
    y = scored["is_fraud"]

    plt.figure(figsize=(7, 6))
    for label, col in SCORE_COLS.items():
        fpr, tpr, _ = roc_curve(y, scored[col])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    plt.axvline(0.03, linestyle=":", color="gray", label="3% FPR operating point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves - Fraud Detection Models (Real IEEE-CIS Data)")
    plt.legend()
    plt.xlim(0, 0.2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR / 'roc_curves.png'}")


def plot_metrics_comparison():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    names = [m["model"].split(" (")[0] for m in metrics]
    recalls = [m["recall_at_3pct_fpr"] * 100 for m in metrics]
    aucs = [m["roc_auc"] * 100 for m in metrics]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, recalls, width, label="Recall @ 3% FPR (%)", color="#2E86AB")
    ax.bar(x + width/2, aucs, width, label="ROC-AUC (x100)", color="#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Recall vs. ROC-AUC (Real IEEE-CIS Data)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR / 'model_comparison.png'}")


def plot_feature_importance():
    with open(SHAP_IMPORTANCE_PATH) as f:
        importance = json.load(f)
    top = importance[:8]
    names = [d["feature"] for d in top]
    values = [d["mean_abs_shap"] for d in top]

    plt.figure(figsize=(8, 5))
    plt.barh(names[::-1], values[::-1], color="#F18F01")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top Predictive Features (XGBoost, Real SHAP)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR / 'feature_importance.png'}")


def plot_fairness_disparity():
    with open(FAIRNESS_PATH) as f:
        audit = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    geo = audit["subgroups"].get("geo_mismatch", [])
    labels = ["Country matches" if g["group"] == "0" else "Country mismatch" for g in geo]
    fprs = [g["false_positive_rate"] * 100 for g in geo]
    geo_spread = (max(fprs) - min(fprs)) if fprs else 0
    colors = ["#2E86AB" if v < 20 else "#C73E1D" for v in fprs]
    axes[0].bar(labels, fprs, color=colors)
    axes[0].set_ylabel("False Positive Rate (%)")
    axes[0].set_title(f"FPR by Geo-Mismatch\n({geo_spread:.1f}-point spread)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(fprs):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    cat = audit["subgroups"].get("merchant_category", [])
    cat_sorted = sorted(cat, key=lambda x: -x["false_positive_rate"])
    cat_labels = [c["group"] for c in cat_sorted]
    cat_fprs = [c["false_positive_rate"] * 100 for c in cat_sorted]
    cat_spread = (max(cat_fprs) - min(cat_fprs)) if cat_fprs else 0
    axes[1].barh(cat_labels[::-1], cat_fprs[::-1], color="#A23B72")
    axes[1].set_xlabel("False Positive Rate (%)")
    axes[1].set_title(f"FPR by Merchant Category\n({cat_spread:.1f}-point spread)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fairness_disparity.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR / 'fairness_disparity.png'}")


def plot_confusion_matrix():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    best = max(metrics, key=lambda m: m["recall_at_3pct_fpr"])
    cm = best["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            plt.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                      fontsize=14, fontweight="bold", color=color)
    plt.xticks([0, 1], ["Predicted Legit", "Predicted Fraud"])
    plt.yticks([0, 1], ["Actual Legit", "Actual Fraud"])
    model_short = best["model"].split(" (")[0]
    plt.title(f"Confusion Matrix - {model_short} (Real Data)\n(at 3% FPR threshold)", pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    print("[1/5] Plotting ROC curves...", flush=True)
    plot_roc_curves()
    print("[2/5] Plotting model comparison...", flush=True)
    plot_metrics_comparison()
    print("[3/5] Plotting feature importance...", flush=True)
    plot_feature_importance()
    print("[4/5] Plotting fairness disparity...", flush=True)
    plot_fairness_disparity()
    print("[5/5] Plotting confusion matrix...", flush=True)
    plot_confusion_matrix()
    print(f"\nAll real-data plots saved to {PLOTS_DIR}/")
