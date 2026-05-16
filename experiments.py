import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from data_pipeline import load_and_sample, clean, make_target, split
from models.gradient_boosting import GradientBoosting
from models.l1_regression import L1Regression
from models.l2_regression import L2Regression
from models.mlp import MLP


HYPERPARAM_GRIDS = {
    "L1Regression": {
        "cls": L1Regression,
        "grid": {"alpha": [0.01, 1.0, 10.0]},
    },
    "L2Regression": {
        "cls": L2Regression,
        "grid": {"alpha": [0.01, 1.0, 10.0]},
    },
    "GradientBoosting": {
        "cls": GradientBoosting,
        "grid": {
            "n_estimators": [100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
    "MLP": {
        "cls": MLP,
        "grid": {
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "learning_rate_init": [1e-3, 1e-4],
        },
    },
}


def load_data(path):
    df = load_and_sample(path)
    df = clean(df)
    X_df, y = make_target(df)
    feature_names = list(X_df.columns)
    X_tr, X_val, X_test, y_tr, y_val, y_test = split(X_df, y)
    return (
        X_tr.values.astype(float),
        X_val.values.astype(float),
        X_test.values.astype(float),
        y_tr.values,
        y_val.values,
        y_test.values,
        feature_names,
    )


def cross_validate_model(model_cls, param_grid, X, y, cv=5, threshold=0.5, average="binary"):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best = None
    configs = list(ParameterGrid(param_grid))

    for i, params in enumerate(configs):
        print(f"    config {i+1}/{len(configs)}: {params}", flush=True)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = model_cls(**params)
            model.fit(X_train, y_train)
            preds = (model.predict(X_val) >= threshold).astype(int)
            score = f1_score(y_val, preds, average=average, zero_division=0)
            fold_scores.append(score)
            print(f"      fold {fold+1}/{cv}  F1={score:.4f}", flush=True)

        result = {
            "params": params,
            "mean_f1": np.mean(fold_scores),
            "std_f1": np.std(fold_scores),
            "fold_scores": fold_scores,
        }
        print(f"    => mean F1={result['mean_f1']:.4f} ± {result['std_f1']:.4f}", flush=True)
        if best is None or result["mean_f1"] > best["mean_f1"]:
            best = result

    return best


def run_all_experiments(X, y, cv=5, threshold=0.5, average="binary", grids=None):
    if grids is None:
        grids = HYPERPARAM_GRIDS
    results = {}
    for name, spec in grids.items():
        print(f"\n[{name}] searching {len(list(ParameterGrid(spec['grid'])))} configs...", flush=True)
        results[name] = cross_validate_model(
            spec["cls"], spec["grid"], X, y, cv=cv, threshold=threshold, average=average
        )
        best = results[name]
        print(f"  best params : {best['params']}")
        print(f"  mean F1     : {best['mean_f1']:.4f} ± {best['std_f1']:.4f}", flush=True)
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


def analyze_l1_features(fitted_l1_model, feature_names, save_path=None):
    coefs = fitted_l1_model.coef_
    selected = [(name, coef) for name, coef in zip(feature_names, coefs) if coef != 0]
    selected.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n[L1 Feature Selection] {len(selected)}/{len(feature_names)} features selected:")
    for name, coef in selected:
        print(f"  {name:40s}  {coef:+.4f}")

    if not selected:
        return None

    names, vals = zip(*selected)
    colors = ["steelblue" if v > 0 else "tomato" for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(4, len(selected) * 0.35)))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("L1 Logistic Regression — Selected Features")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def analyze_shap(fitted_gb_model, X_test, feature_names, max_display=20, save_path=None):
    import shap

    pipeline = fitted_gb_model.model
    X_scaled = pipeline.named_steps["scaler"].transform(X_test)
    explainer = shap.TreeExplainer(pipeline.named_steps["gbc"])
    shap_values = explainer.shap_values(X_scaled)

    # shap_values is a list [class0, class1] for classifiers; take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"\n[SHAP] mean |SHAP| per feature (top {max_display}):", flush=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    for i in order:
        print(f"  {feature_names[i]:40s}  {mean_abs[i]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Summary — Gradient Boosting")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gmp_data_2026-04-19_03-59-23.csv", help="Path to raw CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score cutoff for binary prediction")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--average", default="binary", choices=["binary", "macro", "weighted"], help="F1 averaging strategy")
    parser.add_argument("--save-plot", default=None, help="Path to save PR curve PNG")
    parser.add_argument("--save-shap", default=None, help="Path to save SHAP summary PNG")
    parser.add_argument("--model", default=None, choices=list(HYPERPARAM_GRIDS.keys()), help="Train only this model")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data(args.data)
    print(f"Loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}", flush=True)

    grids = {args.model: HYPERPARAM_GRIDS[args.model]} if args.model else HYPERPARAM_GRIDS
    cv_results = run_all_experiments(X_train, y_train, cv=args.cv, threshold=args.threshold, average=args.average, grids=grids)
    fitted = train_best_models(X_train, y_train, cv_results)
    plot_precision_recall_curves(fitted, X_test, y_test, save_path=args.save_plot)
    if "L1Regression" in fitted:
        analyze_l1_features(fitted["L1Regression"], feature_names)
    if "GradientBoosting" in fitted:
        analyze_shap(fitted["GradientBoosting"], X_test, feature_names, save_path=args.save_shap)