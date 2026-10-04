import joblib
import pandas as pd
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from config import RAW_DATA, SELECTED_FEATURES, TARGET, MODEL_DIR, METRIC_DIR

modeling_path = Path(__file__).with_name("04_modeling.py")
spec = spec_from_file_location("modeling_module", modeling_path)
modeling = module_from_spec(spec)
spec.loader.exec_module(modeling)

def main():
    df = pd.read_csv(RAW_DATA)
    X = df[SELECTED_FEATURES]
    y = df[TARGET]

    logistic = modeling.logistic_pipeline().fit(X, y)
    forest = modeling.random_forest().fit(X, y)

    joblib.dump(logistic, MODEL_DIR / "selected_logistic_regression.joblib")
    joblib.dump(forest, MODEL_DIR / "selected_random_forest.joblib")

    example = X.head(5).copy()
    example["predicted_probability_random_forest"] = forest.predict_proba(X.head(5))[:, 1]
    example.to_csv(METRIC_DIR / "06_example_predictions.csv", index=False)

    model_card = '''# Model Card

## Intended use
Educational, reproducible research demonstration of heart-failure mortality prediction.

## Outcome
`DEATH_EVENT`: whether death was recorded during each patient's observed follow-up.
This is **not** a fixed-horizon 30-day or 1-year mortality probability.

## Predictors
- age
- ejection_fraction
- serum_creatinine

## Important exclusions
`time` is excluded because it is follow-up duration and is unavailable at the
intended baseline prediction point.

## Validation
Internal repeated stratified cross-validation and separate 5-fold out-of-fold
calibration. No external or prospective validation.

## Limitations
- n = 299
- single historical clinical cohort
- cohort-informed feature selection
- no fixed mortality horizon
- no causal interpretation
- no clinical-utility or treatment-impact evaluation
- subgroup results are descriptive, not fairness certification

## Deployment status
Research/education only. Not for clinical decision-making.
'''
    (MODEL_DIR / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print("Models and deployment documentation written.")

if __name__ == "__main__":
    main()
