import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from config import RAW_DATA, BASELINE_FEATURES, SELECTED_FEATURES, TARGET, FOLLOW_UP

def test_raw_dataset_contract():
    df = pd.read_csv(RAW_DATA)
    assert df.shape == (299, 13)
    assert int(df[TARGET].sum()) == 96
    assert df.isna().sum().sum() == 0

def test_no_post_baseline_leakage():
    assert TARGET not in BASELINE_FEATURES
    assert FOLLOW_UP not in BASELINE_FEATURES
    assert TARGET not in SELECTED_FEATURES
    assert FOLLOW_UP not in SELECTED_FEATURES

def test_selected_feature_definition():
    assert SELECTED_FEATURES == ["age", "ejection_fraction", "serum_creatinine"]
