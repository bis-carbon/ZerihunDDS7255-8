"""Prepare leakage-controlled baseline and selected-feature datasets."""

import pandas as pd

from config import (
    RAW_DATA, PROCESSED_DIR, BASELINE_FEATURES, SELECTED_FEATURES,
    TARGET, FOLLOW_UP,
)
from utils import write_json


def main():
    """Create processed datasets and persist a data-contract summary."""
    df = pd.read_csv(RAW_DATA)

    # Explicit leakage controls.
    assert TARGET not in BASELINE_FEATURES
    assert FOLLOW_UP not in BASELINE_FEATURES
    assert TARGET not in SELECTED_FEATURES
    assert FOLLOW_UP not in SELECTED_FEATURES

    baseline = df[BASELINE_FEATURES + [TARGET]].copy()
    selected = df[SELECTED_FEATURES + [TARGET]].copy()

    baseline.to_csv(PROCESSED_DIR / "heart_failure_baseline.csv", index=False)
    selected.to_csv(PROCESSED_DIR / "heart_failure_selected.csv", index=False)

    contract = {
        "crisp_dm_phase": "Data Preparation",
        "raw_rows": int(len(df)),
        "target": TARGET,
        "excluded_post_baseline_variable": FOLLOW_UP,
        "baseline_features": BASELINE_FEATURES,
        "selected_features": SELECTED_FEATURES,
        "missing_values": int(df.isna().sum().sum()),
        "baseline_shape": list(baseline.shape),
        "selected_shape": list(selected.shape),
    }
    write_json(contract, PROCESSED_DIR / "data_contract.json")
    print("Processed data and data contract written.")


if __name__ == "__main__":
    main()
