"""Shared data-loading, metric, and JSON-persistence utilities."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def classification_metrics(y_true, probability, threshold=0.5):
    """Compute discrimination, classification, and probability-quality metrics."""
    probability = np.asarray(probability)
    prediction = (probability >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier": brier_score_loss(y_true, probability),
    }


def write_json(obj, path: Path):
    """Serialize an object as indented JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=float)
        handle.write("\n")
