from config import METRIC_DIR
from utils import write_json

def main():
    charter = {
        "project_title": "Feature Selection and Machine Learning for Mortality Prediction in Heart Failure",
        "crisp_dm_phase": "Business Understanding",
        "problem_statement": (
            "Determine whether a parsimonious, clinically supported baseline predictor set "
            "achieves better internally cross-validated mortality-prediction performance than "
            "the full baseline clinical feature set without using post-baseline follow-up time."
        ),
        "purpose_statement": (
            "Compare 11 baseline predictors with age, ejection fraction, and serum creatinine "
            "using logistic regression and random forest under the same repeated validation design."
        ),
        "research_questions": [
            "RQ1: How does literature-informed feature selection alter internally cross-validated performance?",
            "RQ2: How do logistic regression and random forest compare on the same selected features?",
            "RQ3: How well do selected models discriminate and calibrate the recorded mortality outcome, "
            "and how stable is the best selected model across sex and age subgroups?",
        ],
        "prediction_point": "Baseline clinical assessment",
        "outcome_interpretation": (
            "DEATH_EVENT indicates whether death was recorded during each patient's observed follow-up; "
            "it is not a fixed-horizon 30-day or 1-year mortality outcome."
        ),
        "success_criteria": [
            "Report multiple predictive metrics rather than accuracy alone.",
            "Prevent post-baseline leakage by excluding follow-up time.",
            "Use repeated stratified cross-validation for model comparison.",
            "Assess calibration with independent out-of-fold probabilities.",
            "Document subgroup variation without claiming fairness from small samples.",
        ],
    }
    write_json(charter, METRIC_DIR / "01_project_charter.json")
    print("Business-understanding artifact written.")

if __name__ == "__main__":
    main()
