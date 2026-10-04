"""Evaluate selected models with out-of-fold calibration and subgroup metrics."""

import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import (
    RAW_DATA, SELECTED_FEATURES, TARGET, RANDOM_STATE,
    METRIC_DIR, FIGURE_DIR,
)
from utils import classification_metrics, write_json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

modeling_path = Path(__file__).with_name("04_modeling.py")
spec = spec_from_file_location("modeling_module", modeling_path)
modeling = module_from_spec(spec)
spec.loader.exec_module(modeling)


def calibration_intercept_slope(y, probability):
    """Estimate calibration intercept and slope from predicted probabilities."""
    eps = 1e-6
    p = np.clip(np.asarray(probability), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def subgroup_rows(df, y, probability):
    """Compute descriptive performance metrics for prespecified age and sex subgroups."""
    result = []
    definitions = [
        ("Female", df["sex"] == 0),
        ("Male", df["sex"] == 1),
        ("Age <65", df["age"] < 65),
        ("Age >=65", df["age"] >= 65),
    ]
    for name, mask in definitions:
        metrics = classification_metrics(y[mask], probability[mask])
        result.append({
            "subgroup": name,
            "n": int(mask.sum()),
            "deaths": int(y[mask].sum()),
            "mortality": float(y[mask].mean()),
            **metrics,
        })
    return pd.DataFrame(result)


def make_calibration_summary_figure(rounded_overall):
    """Write an SVG summary of calibration intercepts and slopes for selected models."""
    lr = rounded_overall["selected_logistic_regression"]
    rf = rounded_overall["selected_random_forest"]

    def point(item):
        """Map calibration statistics to SVG coordinates."""
        slope = item["calibration_slope"]
        intercept = item["calibration_intercept"]
        x = 90 + (slope - 0.8) / 0.25 * 560
        y = 410 - (intercept + 0.10) / 0.20 * 340
        return x, y

    lrx, lry = point(lr)
    rfx, rfy = point(rf)
    ideal_x = 90 + (1.0 - 0.8) / 0.25 * 560
    ideal_y = 410 - (0.0 + 0.10) / 0.20 * 340

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="500" viewBox="0 0 700 500">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="350" y="34" text-anchor="middle" font-family="Arial" font-size="22">Out-of-Fold Calibration Summary</text>\n'
        '<line x1="90" y1="410" x2="650" y2="410" stroke="black"/>\n'
        '<line x1="90" y1="70" x2="90" y2="410" stroke="black"/>\n'
        f'<line x1="{ideal_x:.3f}" y1="70" x2="{ideal_x:.3f}" y2="410" stroke="#888" stroke-dasharray="6,5"/>\n'
        f'<line x1="90" y1="{ideal_y:.3f}" x2="650" y2="{ideal_y:.3f}" stroke="#888" stroke-dasharray="6,5"/>\n'
        '<text x="350" y="460" text-anchor="middle" font-family="Arial" font-size="15">Calibration slope (ideal = 1)</text>\n'
        '<text x="24" y="240" transform="rotate(-90 24 240)" text-anchor="middle" font-family="Arial" font-size="15">Calibration intercept (ideal = 0)</text>\n'
        f'<circle cx="{lrx:.3f}" cy="{lry:.3f}" r="8" fill="#555"/><text x="245" y="370" font-family="Arial" font-size="14">Selected LR: slope {lr["calibration_slope"]:.3f}, intercept {lr["calibration_intercept"]:.3f}</text>\n'
        f'<circle cx="{rfx:.3f}" cy="{rfy:.3f}" r="8" fill="#999"/><text x="230" y="185" font-family="Arial" font-size="14">Selected RF: slope {rf["calibration_slope"]:.3f}, intercept {rf["calibration_intercept"]:.3f}</text>\n'
        f'<text x="{ideal_x:.3f}" y="435" text-anchor="middle" font-family="Arial" font-size="12">Ideal slope</text>\n'
        f'<text x="615" y="{ideal_y-8:.3f}" font-family="Arial" font-size="12">Ideal intercept</text>\n'
        '</svg>'
    )
    (FIGURE_DIR / "figure_03_calibration_summary.svg").write_text(svg, encoding="utf-8")


def round_metrics(mapping, digits=6):
    """Round nested model-metric mappings to a fixed number of decimal places."""
    return {
        model: {key: round(float(value), digits) for key, value in metrics.items()}
        for model, metrics in mapping.items()
    }


def main():
    """Generate out-of-fold predictions, calibration statistics, and subgroup metrics."""
    df = pd.read_csv(RAW_DATA)
    X = df[SELECTED_FEATURES]
    y = df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "selected_logistic_regression": modeling.logistic_pipeline(),
        "selected_random_forest": modeling.random_forest(),
    }

    all_predictions = pd.DataFrame({"row_id": np.arange(len(df)), "y_true": y})
    overall = {}

    for name, estimator in models.items():
        probability = cross_val_predict(
            clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=None
        )[:, 1]
        all_predictions[f"{name}_probability"] = probability
        metrics = classification_metrics(y, probability)
        intercept, slope = calibration_intercept_slope(y, probability)
        overall[name] = {
            **metrics,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }

    all_predictions.round(8).to_csv(METRIC_DIR / "05_oof_predictions.csv", index=False)
    rounded_overall = round_metrics(overall)
    write_json(rounded_overall, METRIC_DIR / "05_overall_oof_metrics.json")
    make_calibration_summary_figure(rounded_overall)

    rf_prob = all_predictions["selected_random_forest_probability"].to_numpy()
    subgroup = subgroup_rows(df, y, rf_prob)
    subgroup.round(6).to_csv(METRIC_DIR / "05_subgroup_metrics.csv", index=False)
    print(json.dumps(rounded_overall, indent=2))
    print(subgroup.to_string(index=False))


if __name__ == "__main__":
    main()
