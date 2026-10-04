"""Train and compare fixed logistic-regression and random-forest configurations."""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    RAW_DATA, BASELINE_FEATURES, SELECTED_FEATURES, TARGET,
    RANDOM_STATE, N_SPLITS, N_REPEATS, METRIC_DIR, FIGURE_DIR,
)


def logistic_pipeline():
    """Return the fixed standardized logistic-regression pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=3000,
            class_weight=None,
            random_state=RANDOM_STATE,
        )),
    ])


def random_forest():
    """Return the fixed random-forest classifier used in the study."""
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "neg_brier": "neg_brier_score",
}


def evaluate_configuration(name, estimator, X, y, cv):
    """Evaluate one model-feature configuration across repeated cross-validation."""
    scores = cross_validate(
        estimator, X, y, cv=cv, scoring=SCORING, n_jobs=-1,
        return_train_score=False, error_score="raise"
    )
    return pd.DataFrame({
        "model": name,
        "fold": np.arange(1, len(scores["test_roc_auc"]) + 1),
        "roc_auc": scores["test_roc_auc"],
        "average_precision": scores["test_average_precision"],
        "accuracy": scores["test_accuracy"],
        "balanced_accuracy": scores["test_balanced_accuracy"],
        "precision": scores["test_precision"],
        "recall": scores["test_recall"],
        "f1": scores["test_f1"],
        "brier": -scores["test_neg_brier"],
    })


def make_discrimination_figure(summary):
    """Write an SVG comparison of mean ROC-AUC and average precision."""
    order = [
        ("all_feature_logistic_regression", "All LR"),
        ("selected_logistic_regression", "Selected LR"),
        ("all_feature_random_forest", "All RF"),
        ("selected_random_forest", "Selected RF"),
    ]
    indexed = summary.set_index("model")
    bars = []
    for i, (model, label) in enumerate(order):
        row = indexed.loc[model]
        base_x = 130 + i * 175
        roc = float(row["roc_auc"])
        ap = float(row["average_precision"])
        h1, h2 = roc * 400, ap * 400
        bars.append(
            f'<rect x="{base_x}" y="{420-h1:.3f}" width="45" height="{h1:.3f}" fill="#555"/>'
            f'<rect x="{base_x+50}" y="{420-h2:.3f}" width="45" height="{h2:.3f}" fill="#aaa"/>'
            f'<text x="{base_x+47}" y="450" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="450" y="34" text-anchor="middle" font-family="Arial" font-size="22">Repeated Cross-Validation Discrimination</text>\n'
        '<line x1="80" y1="420" x2="860" y2="420" stroke="black"/>\n'
        '<line x1="80" y1="70" x2="80" y2="420" stroke="black"/>\n'
        '<text x="24" y="250" transform="rotate(-90 24 250)" text-anchor="middle" font-family="Arial" font-size="16">Mean score</text>\n'
        + ''.join(bars) + '\n'
        '<rect x="650" y="65" width="18" height="18" fill="#555"/><text x="675" y="79" font-family="Arial" font-size="13">ROC-AUC</text>\n'
        '<rect x="750" y="65" width="18" height="18" fill="#aaa"/><text x="775" y="79" font-family="Arial" font-size="13">AP</text>\n'
        '</svg>'
    )
    (FIGURE_DIR / "figure_02_repeated_cv.svg").write_text(svg, encoding="utf-8")


def main():
    """Run repeated cross-validation and persist model-comparison artifacts."""
    df = pd.read_csv(RAW_DATA)
    y = df[TARGET]
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    configs = [
        ("all_feature_logistic_regression", logistic_pipeline(), df[BASELINE_FEATURES]),
        ("selected_logistic_regression", logistic_pipeline(), df[SELECTED_FEATURES]),
        ("all_feature_random_forest", random_forest(), df[BASELINE_FEATURES]),
        ("selected_random_forest", random_forest(), df[SELECTED_FEATURES]),
    ]
    folds = pd.concat(
        [evaluate_configuration(name, estimator, X, y, cv) for name, estimator, X in configs],
        ignore_index=True,
    )
    folds.to_csv(METRIC_DIR / "04_fold_metrics.csv", index=False)

    means = folds.groupby("model").mean(numeric_only=True).drop(columns=["fold"])
    sds = folds.groupby("model")[["roc_auc", "average_precision", "brier"]].std()
    summary = means.reset_index()
    for metric in ["roc_auc", "average_precision", "brier"]:
        summary[f"{metric}_sd"] = summary["model"].map(sds[metric])
    ordered = [
        "model", "roc_auc", "roc_auc_sd", "average_precision", "average_precision_sd",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "brier", "brier_sd",
    ]
    summary = summary[ordered]
    summary.round(6).to_csv(METRIC_DIR / "04_model_summary.csv", index=False)
    make_discrimination_figure(summary)

    metadata = {
        "crisp_dm_phase": "Modeling",
        "modeling_design": "Fixed configurations; no grid/random search on repeated-CV results.",
        "cross_validation": f"{N_SPLITS}-fold stratified CV repeated {N_REPEATS} times",
        "random_state": RANDOM_STATE,
        "average_precision_definition": "scikit-learn average_precision scorer",
        "threshold_for_class_metrics": 0.5,
    }
    (METRIC_DIR / "04_modeling_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
