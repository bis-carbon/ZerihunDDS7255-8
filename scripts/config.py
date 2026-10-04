from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "heart_failure_clinical_records_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"

TARGET = "DEATH_EVENT"
FOLLOW_UP = "time"

BASELINE_FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]

SELECTED_FEATURES = [
    "age",
    "ejection_fraction",
    "serum_creatinine",
]

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 20

for path in [PROCESSED_DIR, FIGURE_DIR, METRIC_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
