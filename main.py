import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold, cross_validate, GroupShuffleSplit, RandomizedSearchCV, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score, classification_report, confusion_matrix
import pickle

from src.feature_names import FEATURE_COLS

def load_data(features_path, binary=False):
    """
    Loads the features CSV and returns features, labels and groups.
    
    :param features_path: path to features.csv
    :param binary: if True, merge classes into Cancer vs Non-Cancer
    :return: X (features), y (labels), groups (patient_id)
    """
    df = pd.read_csv(features_path)
    feature_cols = FEATURE_COLS

    if binary:
        # Merge into Cancer vs Non-Cancer
        cancer_classes = ['BCC', 'MEL', 'SCC']
        df['label'] = df['diagnostic'].apply(
            lambda x: 'Cancer' if x in cancer_classes else 'Non-Cancer'
        )
        y = df['label'].values
    else:
        y = df['diagnostic'].values
    
    X = df[feature_cols].values
    groups = df['patient_id'].values

    # Validate: catch any NaN/inf that slipped through feature extraction
    bad_mask = np.isnan(X) | np.isinf(X)
    if bad_mask.any():
        bad_rows = np.where(bad_mask.any(axis=1))[0]
        bad_cols = [feature_cols[c] for c in np.where(bad_mask.any(axis=0))[0]]
        print(f"WARNING: {bad_mask.sum()} invalid value(s) found in features.")
        print(f"  Affected rows (samples): {bad_rows.tolist()}")
        print(f"  Affected columns (features): {bad_cols}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, groups


def split_dev_test(X, y, groups, test_size=0.2, random_state=42):
    """
    Splits data into development and test sets using GroupShuffleSplit
    to ensure patients don't appear in both sets.
    
    :param X: feature matrix
    :param y: labels
    :param groups: patient_id groups
    :param test_size: fraction of data to use as test set
    :return: X_dev, X_test, y_dev, y_test, groups_dev
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    
    X_dev, X_test = X[dev_idx], X[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]
    groups_dev = groups[dev_idx]
    
    return X_dev, X_test, y_dev, y_test, groups_dev


def run_cross_validation(X, y, groups):
    """
    Runs GroupKFold cross-validation for multiple classifiers.

    Each classifier is wrapped in a Pipeline with StandardScaler so that
    scaling is fit only on the training fold and never sees the validation fold.

    :param X: feature matrix
    :param y: labels
    :param groups: patient_id groups
    :return: dictionary of results per classifier
    """
    # Define classifiers to compare
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    # GroupKFold with 5 folds
    gkf = GroupKFold(n_splits=5)

    # Scoring metric
    scorer = make_scorer(balanced_accuracy_score)

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        cv_results = cross_validate(
            pipeline, X, y,
            groups=groups,
            cv=gkf,
            scoring=scorer,
            return_train_score=True
        )

        mean_val = np.mean(cv_results['test_score'])
        std_val = np.std(cv_results['test_score'])
        mean_train = np.mean(cv_results['train_score'])

        print(f"  Train score: {mean_train:.3f}")
        print(f"  Val score:   {mean_val:.3f} +/- {std_val:.3f}")

        results[name] = {
            "mean_val": mean_val,
            "std_val": std_val,
            "mean_train": mean_train
        }

    return results


def tune_hyperparameters(X_dev, y_dev, groups_dev):
    """
    Tunes hyperparameters for both Random Forest and KNN using
    RandomizedSearchCV with GroupKFold so no patient leaks across folds.
    Returns the name and best params of whichever model scores highest.

    :param X_dev: development feature matrix
    :param y_dev: development labels
    :param groups_dev: patient_id groups for development set
    :return: (best_model_name, best_params)
    """
    scorer = make_scorer(balanced_accuracy_score)
    gkf = GroupKFold(n_splits=5)

    search_configs = {
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "classifier__n_estimators":     [100, 200, 300],
                "classifier__max_depth":         [3, 5, 7, 10, None],
                "classifier__min_samples_leaf":  [1, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features":      ["sqrt", "log2"],
            },
            50,
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "classifier__weights":     ["uniform", "distance"],
                "classifier__metric":      ["euclidean", "manhattan", "minkowski"],
            },
            42,  # exhausts the full grid (7 * 2 * 3)
        ),
    }

    # KNN in sklearn 1.8 fails with string labels due to an internal optimisation
    # that casts class labels to int. Encode once and reuse for KNN searches.
    le = LabelEncoder()
    y_dev_encoded = le.fit_transform(y_dev)

    best_name = None
    best_score = -1
    best_params = None

    for name, (clf, param_dist, n_iter) in search_configs.items():
        print(f"\nTuning {name} ({n_iter} iterations × 5 folds)...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=gkf,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        y_fit = y_dev_encoded if name == "KNN" else y_dev
        search.fit(X_dev, y_fit, groups=groups_dev)

        params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        print(f"  Best CV balanced accuracy: {search.best_score_:.3f}")
        print(f"  Best parameters: {params}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_params = params

    print(f"\nWinner: {best_name} (CV balanced accuracy: {best_score:.3f})")
    return best_name, best_params


def compute_validation_curves(X_dev, y_dev, groups_dev, output_dir):
    """
    Computes validation curves for Random Forest hyperparameters.

    These plots show how training and validation balanced accuracy
    change as n_estimators and max_depth vary, while all other
    hyperparameters are held fixed. They justify the choices made
    during tuning by showing where over/underfitting kicks in.

    Saves two figures to `output_dir`:
      - val_curve_n_estimators.png
      - val_curve_max_depth.png

    :param X_dev: development feature matrix
    :param y_dev: development labels
    :param groups_dev: patient_id groups
    :param output_dir: Path-like, directory to save figures into
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = make_scorer(balanced_accuracy_score)
    gkf = GroupKFold(n_splits=5)

    # Vary n_estimators with max_depth=10 fixed
    n_est_range = [10, 25, 50, 100, 200, 300, 500]
    print(f"\nValidation curve over n_estimators in {n_est_range}...")
    pipeline_n = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(max_depth=10,
                                              class_weight="balanced",
                                              random_state=42)),
    ])
    train_scores, val_scores = validation_curve(
        pipeline_n, X_dev, y_dev,
        param_name="classifier__n_estimators",
        param_range=n_est_range,
        groups=groups_dev, cv=gkf, scoring=scorer, n_jobs=-1,
    )
    _plot_validation_curve(
        n_est_range, train_scores, val_scores,
        xlabel="n_estimators (max_depth=10 fixed)",
        title="Validation curve: n_estimators",
        path=output_dir / "val_curve_n_estimators.png",
    )

    # Vary max_depth with n_estimators=100 fixed; encode None as a finite
    # sentinel so the x-axis can be plotted
    depths = [2, 3, 5, 7, 10, 15, 20, None]
    depth_labels = [d if d is not None else "None" for d in depths]
    print(f"Validation curve over max_depth in {depth_labels}...")
    pipeline_d = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100,
                                              class_weight="balanced",
                                              random_state=42)),
    ])
    train_scores, val_scores = validation_curve(
        pipeline_d, X_dev, y_dev,
        param_name="classifier__max_depth",
        param_range=depths,
        groups=groups_dev, cv=gkf, scoring=scorer, n_jobs=-1,
    )
    _plot_validation_curve(
        list(range(len(depths))), train_scores, val_scores,
        xlabel="max_depth (n_estimators=100 fixed)",
        title="Validation curve: max_depth",
        path=output_dir / "val_curve_max_depth.png",
        xtick_labels=depth_labels,
    )

    print(f"Validation curves saved to {output_dir}")


def _plot_validation_curve(x, train_scores, val_scores, xlabel, title, path,
                            xtick_labels=None):
    """Helper to plot a validation curve with mean ± std bands."""
    import matplotlib.pyplot as plt

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, train_mean, "-o", color="#93c5fd", label="Train (CV)")
    ax.fill_between(x, train_mean - train_std, train_mean + train_std,
                     color="#93c5fd", alpha=0.2)
    ax.plot(x, val_mean, "-o", color="#2563eb", label="Val (CV)")
    ax.fill_between(x, val_mean - val_std, val_mean + val_std,
                     color="#2563eb", alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    if xtick_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cv_results(results_bin, results_6, output_path):
    """
    Persist cross-validation results to JSON so the dashboard can
    load them without hardcoded numbers.
    """
    payload = {
        "binary": {
            name: {
                "train": float(res["mean_train"]),
                "val": float(res["mean_val"]),
                "std": float(res["std_val"]),
            }
            for name, res in results_bin.items()
        },
        "six_class": {
            name: {
                "train": float(res["mean_train"]),
                "val": float(res["mean_val"]),
                "std": float(res["std_val"]),
            }
            for name, res in results_6.items()
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"CV results saved to {output_path}")


def train_and_save(X_dev, X_test, y_dev, y_test, model_path,
                   prediction_results_path, load_model, model_name=None, model_params=None):
    """
    Trains the best model on development set, evaluates on test set.

    :param X_dev: development feature matrix
    :param X_test: test feature matrix
    :param y_dev: development labels
    :param y_test: test labels
    :param model_path: path to save/load the model
    :param prediction_results_path: path to save predictions
    :param load_model: if True, load model from model_path
    :param model_name: "Random Forest" or "KNN" (from tuning)
    :param model_params: dict of hyperparameters (from tuning)
    """
    scaler_path = Path(model_path).with_suffix('.scaler.pkl')
    encoder_path = Path(model_path).with_suffix('.encoder.pkl')

    if load_model:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        # Load label encoder if it exists (saved for KNN models)
        le = pickle.load(open(encoder_path, 'rb')) if encoder_path.exists() else None
        print("Model and scaler loaded!")
    else:
        scaler = StandardScaler()
        scaler.fit(X_dev)

        params = model_params or {}
        if model_name == "KNN":
            # Encode string labels to integers — sklearn 1.8 KNN requires numeric labels
            le = LabelEncoder()
            y_dev_fit = le.fit_transform(y_dev)
            print(f"Training final KNN model with params: {params}")
            clf = KNeighborsClassifier(**params)
            clf.fit(scaler.transform(X_dev), y_dev_fit)
            with open(encoder_path, 'wb') as f:
                pickle.dump(le, f)
            print(f"Label encoder saved to {encoder_path}")
        else:
            le = None
            params = params or {"n_estimators": 100, "max_depth": 10}
            print(f"Training final Random Forest model with params: {params}")
            clf = RandomForestClassifier(class_weight="balanced", random_state=42, **params)
            clf.fit(scaler.transform(X_dev), y_dev)

        # Save the model and scaler together
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    raw_predictions = clf.predict(X_test_scaled)
    probabilities = clf.predict_proba(X_test_scaled)

    # Decode integer predictions back to string labels for KNN models
    if le is not None:
        predictions = le.inverse_transform(raw_predictions)
        class_names = le.classes_
    else:
        predictions = raw_predictions
        class_names = clf.classes_

    test_score = balanced_accuracy_score(y_test, predictions)
    print(f"Test balanced accuracy: {test_score:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': predictions,
    })
    for i, class_name in enumerate(class_names):
        pred_df[f'prob_{class_name}'] = probabilities[:, i]
    
    Path(prediction_results_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(prediction_results_path, index=False)
    print(f"\nPredictions saved to {prediction_results_path}")


def main(features_path, prediction_results_path, model_path, load_model):
    """
    Full model pipeline: cross-validation for model selection,
    saving models and predictions.
    """
    # Load binary data
    X, y, groups = load_data(features_path, binary=True)
    
    # Split into development and test sets
    print("Splitting data into development and test sets...")
    X_dev, X_test, y_dev, y_test, groups_dev = split_dev_test(X, y, groups)
    print(f"Development set: {len(y_dev)} samples")
    print(f"Test set: {len(y_test)} samples")

    if not load_model:
        # Compare 6-class vs binary on development set
        print("\n" + "=" * 50)
        print("6-CLASS CLASSIFICATION (development set)")
        print("=" * 50)
        X_6, y_6, groups_6 = load_data(features_path, binary=False)
        X_6_dev, _, y_6_dev, _, groups_6_dev = split_dev_test(X_6, y_6, groups_6)
        results_6 = run_cross_validation(X_6_dev, y_6_dev, groups_6_dev)

        print("\n" + "=" * 50)
        print("BINARY CLASSIFICATION (development set)")
        print("=" * 50)
        results_bin = run_cross_validation(X_dev, y_dev, groups_dev)

        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print("\n6-class:")
        for name, res in results_6.items():
            print(f"  {name}: {res['mean_val']:.3f} +/- {res['std_val']:.3f}")
        print("\nBinary:")
        for name, res in results_bin.items():
            print(f"  {name}: {res['mean_val']:.3f} +/- {res['std_val']:.3f}")

        # Persist CV results so the dashboard can read them
        save_cv_results(results_bin, results_6, "results/cv_results.json")

        # Validation curves for the chosen model (Random Forest)
        print("\n" + "=" * 50)
        print("VALIDATION CURVES")
        print("=" * 50)
        compute_validation_curves(X_dev, y_dev, groups_dev,
                                   output_dir="results/figures")

        # Tune hyperparameters on the binary development set
        print("\n" + "=" * 50)
        print("HYPERPARAMETER TUNING")
        print("=" * 50)
        best_model_name, best_params = tune_hyperparameters(X_dev, y_dev, groups_dev)
    else:
        best_model_name, best_params = None, None

    # Train final model and evaluate on test set
    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 50)
    train_and_save(X_dev, X_test, y_dev, y_test, model_path,
                   prediction_results_path, load_model,
                   model_name=best_model_name, model_params=best_params)
    print("\nDone!")


if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./results/predictions/predictions.csv"
    model_path = "./results/models/model.pkl"
    load_model = False

    main(features_path, prediction_results_path, model_path, load_model)