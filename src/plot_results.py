import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)


# ── paths ────────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = Path("results/predictions/predictions.csv")
MODEL_PATH       = Path("results/models/model.pkl")
OUTPUT_PATH      = Path("results/figures/results.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── CV scores from the pipeline run (binary classification) ──────────────────
CV_RESULTS = {
    "Decision\nTree":   {"train": 1.000, "val": 0.644, "std": 0.021},
    "Random\nForest":   {"train": 0.987, "val": 0.714, "std": 0.019},
    "KNN":              {"train": 0.795, "val": 0.676, "std": 0.041},
}

FEATURE_NAMES = [
    "area", "height", "width", "perimeter", "compactness",
    "asymmetry",
    "border_irregularity", "border_gradient",
    "diameter_0", "diameter_45", "diameter_90", "diameter_135", "diameter_ratio",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    "rel_r", "rel_g", "rel_b",
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
    "hair_coverage", "hair_in_lesion",
]


def load_artifacts():
    preds = pd.read_csv(PREDICTIONS_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return preds, model


def plot_confusion_matrix(ax, preds):
    cm = confusion_matrix(preds["true_label"], preds["predicted_label"],
                          labels=["Cancer", "Non-Cancer"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cancer", "Non-Cancer"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def plot_roc_curve(ax, preds):
    y_true  = (preds["true_label"] == "Cancer").astype(int)
    y_score = preds["prob_Cancer"]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="#2563eb", lw=2,
            label=f"Random Forest (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)


def plot_cv_comparison(ax):
    names  = list(CV_RESULTS.keys())
    vals   = [CV_RESULTS[n]["val"]   for n in names]
    trains = [CV_RESULTS[n]["train"] for n in names]
    stds   = [CV_RESULTS[n]["std"]   for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars_val   = ax.bar(x - w/2, vals,   w, label="Val (CV)",   color="#2563eb", alpha=0.85)
    bars_train = ax.bar(x + w/2, trains, w, label="Train (CV)", color="#93c5fd", alpha=0.85)

    ax.errorbar(x - w/2, vals, yerr=stds, fmt="none",
                color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("CV Model Comparison — Binary Classification", fontweight="bold")
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.legend(fontsize=9)

    for bar in bars_val:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)


def plot_feature_importance(ax, model, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names   = [FEATURE_NAMES[i] for i in indices]
    values  = importances[indices]

    colors = ["#2563eb" if v >= np.percentile(values, 60) else "#93c5fd" for v in values]
    ax.barh(range(top_n), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)


def main():
    preds, model = load_artifacts()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Skin Lesion Classification Results\n"
        "Random Forest · Binary (Cancer vs Non-Cancer) · PAD-UFES-20",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_cm   = fig.add_subplot(gs[0, 0])
    ax_roc  = fig.add_subplot(gs[0, 1])
    ax_cv   = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[1, 1])

    plot_confusion_matrix(ax_cm, preds)
    plot_roc_curve(ax_roc, preds)
    plot_cv_comparison(ax_cv)
    plot_feature_importance(ax_feat, model)

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
