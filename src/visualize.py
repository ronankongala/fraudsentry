"""
FraudSentry - Visualizations for README
==========================================
Generates the charts a portfolio README needs: ROC curves, feature
importance, confusion matrix, and the fairness audit disparity chart.

Run after train_models.py + explainability.py + fairness_audit_offline.py
(offline/synthetic pipeline) or the _real equivalents (point SCORE_COL /
paths at the _real files -- see the REAL_DATA_MODE flag below).
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

PLOTS_DIR = "/home/claude/fraudsentry/results/plots"
import os
os.makedirs(PLOTS_DIR, exist_ok=True)

# Flip to True and adjust paths if running against the real-library pipeline
REAL_DATA_MODE = False
SCORE_COLS = {
    "Logistic Regression": "lr_score",
    "Random Forest": "rf_score",
    "Gradient Boosting": "gb_score",
    "Isolation Forest": "iso_score",
}
TEST_SCORED_PATH = "/home/claude/fraudsentry/results/test_scored.csv"
METRICS_PATH = "/home/claude/fraudsentry/results/metrics.json"


def plot_roc_curves():
    scored = pd.read_csv(TEST_SCORED_PATH)
    y = scored["is_fraud"]

    plt.figure(figsize=(7, 6))
    for label, col in SCORE_COLS.items():
        if col is None or col not in scored.columns:
            continue
        fpr, tpr, _ = roc_curve(y, scored[col])
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random baseline")
    plt.axvline(0.03, color="gray", linestyle=":", alpha=0.6, label="3% FPR operating point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves — Fraud Detection Models")
    plt.legend(loc="lower right", fontsize=9)
    plt.xlim(0, 0.2)  # zoom into the low-FPR region that actually matters for fraud ops
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/roc_curves.png")


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
    ax.set_title("Model Comparison — Recall vs. ROC-AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/model_comparison.png")


def plot_feature_importance():
    with open("/home/claude/fraudsentry/results/global_importance.json") as f:
        importance = json.load(f)
    top = importance[:8]
    names = [d["feature"] for d in top]
    values = [d["importance"] for d in top]

    plt.figure(figsize=(8, 5))
    plt.barh(names[::-1], values[::-1], color="#F18F01")
    plt.xlabel("Permutation Importance (drop in average precision)")
    plt.title("Top Predictive Features (Logistic Regression)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/feature_importance.png")


def plot_fairness_disparity():
    with open("/home/claude/fraudsentry/results/fairness_audit_synthetic.json") as f:
        audit = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Geo mismatch panel
    geo = audit["subgroups"]["geo_mismatch"]
    labels = ["Country matches" if g["group"] == "0" else "Country mismatch" for g in geo]
    fprs = [g["false_positive_rate"] * 100 for g in geo]
    colors = ["#2E86AB" if v < 20 else "#C73E1D" for v in fprs]
    axes[0].bar(labels, fprs, color=colors)
    axes[0].set_ylabel("False Positive Rate (%)")
    axes[0].set_title("FPR by Geo-Mismatch\n(54-point disparity)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(fprs):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    # Merchant category panel
    cat = audit["subgroups"]["merchant_category"]
    cat_sorted = sorted(cat, key=lambda x: -x["false_positive_rate"])
    cat_labels = [c["group"] for c in cat_sorted]
    cat_fprs = [c["false_positive_rate"] * 100 for c in cat_sorted]
    axes[1].barh(cat_labels[::-1], cat_fprs[::-1], color="#A23B72")
    axes[1].set_xlabel("False Positive Rate (%)")
    axes[1].set_title("FPR by Merchant Category\n(6.7-point spread)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fairness_disparity.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/fairness_disparity.png")


def plot_confusion_matrix():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    best = max(metrics[:3], key=lambda m: m["recall_at_3pct_fpr"])
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
    plt.title(f"Confusion Matrix — {model_short}\n(at 3% FPR threshold)", pad=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    plot_roc_curves()
    plot_metrics_comparison()
    plot_feature_importance()
    plot_fairness_disparity()
    plot_confusion_matrix()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
