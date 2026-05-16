import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from models.gradient_boosting import GradientBoosting
from models.l1_regression import L1Regression
from models.l2_regression import L2Regression
from models.mlp import MLP


HYPERPARAM_GRIDS = {
    "L1Regression": {
        "cls": L1Regression,
        "grid": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },
    "L2Regression": {
        "cls": L2Regression,
        "grid": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },
    "GradientBoosting": {
        "cls": GradientBoosting,
        "grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5],
        },
    },
    "MLP": {
        "cls": MLP,
        "grid": {
            "hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],
            "learning_rate_init": [1e-3, 5e-4, 1e-4],
        },
    },
}


def load_data(path, target_col, feature_cols=None, drop_cols=None):
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].values
    X = df[feature_cols].values if feature_cols else df.drop(columns=[target_col]).values
    return X.astype(float), y


def cross_validate_model(model_cls, param_grid, X, y, cv=5, threshold=0.5, average="binary"):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best = None

    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = model_cls(**params)
            model.fit(X_train, y_train)
            preds = (model.predict(X_val) >= threshold).astype(int)
            fold_scores.append(f1_score(y_val, preds, average=average, zero_division=0))

        result = {
            "params": params,
            "mean_f1": np.mean(fold_scores),
            "std_f1": np.std(fold_scores),
            "fold_scores": fold_scores,
        }
        if best is None or result["mean_f1"] > best["mean_f1"]:
            best = result

    return best


def run_all_experiments(X, y, cv=5, threshold=0.5, average="binary"):
    results = {}
    for name, spec in HYPERPARAM_GRIDS.items():
        print(f"[{name}] searching {len(list(ParameterGrid(spec['grid'])))} configs...")
        results[name] = cross_validate_model(
            spec["cls"], spec["grid"], X, y, cv=cv, threshold=threshold, average=average
        )
        best = results[name]
        print(f"  best params : {best['params']}")
        print(f"  mean F1     : {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")
    return results


def train_best_models(X_train, y_train, cv_results):
    fitted = {}
    for name, result in cv_results.items():
        cls = HYPERPARAM_GRIDS[name]["cls"]
        model = cls(**result["params"])
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def plot_precision_recall_curves(fitted_models, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted_models.items():
        scores = model.predict(X_test)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"baseline (AP={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--drop", nargs="*", help="Columns to drop")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--average", default="binary", choices=["binary", "macro", "weighted"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--save-plot", default=None, help="Path to save PR curve PNG")
    args = parser.parse_args()

    X, y = load_data(args.data, args.target, drop_cols=args.drop)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    cv_results = run_all_experiments(X_train, y_train, cv=args.cv, threshold=args.threshold, average=args.average)
    fitted = train_best_models(X_train, y_train, cv_results)
    plot_precision_recall_curves(fitted, X_test, y_test, save_path=args.save_plot)