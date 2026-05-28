# open question: do the hair features actually help?
# train the model twice (with and without the two hair features) and compare.
# run with: python -m src.open_question

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, balanced_accuracy_score
from src.feature_names import FEATURE_COLS, HAIR_FEATURES

FEATURES_PATH = Path("data/features.csv")
OUTPUT_FIG = Path("results/figures/open_question_hair.png")
OUTPUT_JSON = Path("results/open_question_results.json")

def load_binary():
    #Load features.csv and return X, y, groups for binary cancer/non-cancer
    df = pd.read_csv(FEATURES_PATH)
    cancer = {"BCC", "MEL", "SCC"}
    y = df["diagnostic"].apply(lambda c: "Cancer" if c in cancer else "Non-Cancer").values
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    groups = df["patient_id"].values
    return X, y, groups


def cv_scores(X, y, groups, scorer, cv):
    #No repeat the pipeline definition twice
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    return cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring=scorer, n_jobs=-1)


def main():
    print("Loading data...")
    X_all, y, groups = load_binary()
    X_no_hair = X_all.drop(columns=HAIR_FEATURES)

    print(f"With hair features:    {X_all.shape[1]} columns")
    print(f"Without hair features: {X_no_hair.shape[1]} columns")

    gkf = GroupKFold(n_splits=5)
    bal_acc = make_scorer(balanced_accuracy_score)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if list(le.classes_).index("Cancer") != 1:
        y_enc = 1 - y_enc

    print("\n--- With all features ---")
    bal_full = cv_scores(X_all, y_enc, groups, bal_acc, gkf)
    auc_full = cv_scores(X_all, y_enc, groups, "roc_auc", gkf)
    print(f"  Balanced acc: {bal_full.mean():.3f} +/- {bal_full.std():.3f}")
    print(f"  AUC:          {auc_full.mean():.3f} +/- {auc_full.std():.3f}")

    print("\n--- Without hair features ---")
    bal_no_hair = cv_scores(X_no_hair, y_enc, groups, bal_acc, gkf)
    auc_no_hair = cv_scores(X_no_hair, y_enc, groups, "roc_auc", gkf)
    print(f"  Balanced acc: {bal_no_hair.mean():.3f} +/- {bal_no_hair.std():.3f}")
    print(f"  AUC:          {auc_no_hair.mean():.3f} +/- {auc_no_hair.std():.3f}")

    delta_bal = bal_full.mean() - bal_no_hair.mean()
    delta_auc = auc_full.mean() - auc_no_hair.mean()
    print(f"\nDifference (with - without):")
    print(f"  Balanced acc: {delta_bal:+.3f}")
    print(f"  AUC:          {delta_auc:+.3f}")

    # save results
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "with_hair":    {"bal_acc": list(map(float, bal_full)),
                             "auc":     list(map(float, auc_full))},
            "without_hair": {"bal_acc": list(map(float, bal_no_hair)),
                             "auc":     list(map(float, auc_no_hair))},
            "removed_features": HAIR_FEATURES,
        }, f, indent=2)
    print(f"Saved numbers to {OUTPUT_JSON}")

    # plot
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (full, no_hair, label) in zip(
        axes,
        [(bal_full, bal_no_hair, "Balanced Accuracy"),
         (auc_full, auc_no_hair, "AUC")],
    ):
        means = [full.mean(), no_hair.mean()]
        stds  = [full.std(),  no_hair.std()]
        ax.bar([0, 1], means, yerr=stds, color=["steelblue", "cornflowerblue"],
               capsize=6, edgecolor="white", linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["With hair feats", "Without hair feats"])
        ax.set_ylabel(label)
        ax.set_title(f"{label} -- RF, 5-fold GroupKFold", fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis="y", alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}\n± {s:.3f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Open question: do hair features actually help?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
