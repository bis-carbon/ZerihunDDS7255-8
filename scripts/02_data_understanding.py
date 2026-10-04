"""Inspect the raw heart-failure dataset and produce descriptive artifacts."""

import json
import pandas as pd

from config import RAW_DATA, TARGET, FOLLOW_UP, FIGURE_DIR, METRIC_DIR
from utils import write_json

EXPECTED_COLUMNS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking",
    "time", "DEATH_EVENT",
]


def validate_raw_data(df):
    """Validate the expected schema, sample size, outcome count, and missingness."""
    assert list(df.columns) == EXPECTED_COLUMNS, "Unexpected column order or names."
    assert df.shape == (299, 13), f"Unexpected shape: {df.shape}"
    assert int(df[TARGET].sum()) == 96, "Unexpected death-event count."
    assert df.isna().sum().sum() == 0, "Unexpected missing values."


def mortality_strata(df):
    """Return observed mortality proportions for prespecified descriptive strata."""
    return {
        "Age <65": df.loc[df["age"] < 65, TARGET].mean(),
        "Age >=65": df.loc[df["age"] >= 65, TARGET].mean(),
        "EF <40%": df.loc[df["ejection_fraction"] < 40, TARGET].mean(),
        "EF >=40%": df.loc[df["ejection_fraction"] >= 40, TARGET].mean(),
        "Creatinine <=1.5": df.loc[df["serum_creatinine"] <= 1.5, TARGET].mean(),
        "Creatinine >1.5": df.loc[df["serum_creatinine"] > 1.5, TARGET].mean(),
    }


def make_mortality_figure(groups):
    """Write an SVG bar chart of observed mortality for the supplied strata."""
    bars = []
    for i, (label, value) in enumerate(groups.items()):
        x = 110 + i * 120
        height = value * 500
        y = 420 - height
        safe_label = label.replace("<=", "≤").replace(">=", "≥").replace("<", "&lt;").replace(">", "&gt;")
        bars.append(
            f'<rect x="{x}" y="{y:.3f}" width="70" height="{height:.3f}" fill="#777"/>'
            f'<text x="{x+35}" y="{y-8:.3f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.3f}</text>'
            f'<text x="{x+35}" y="448" text-anchor="middle" font-family="Arial" font-size="12">{safe_label}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="450" y="34" text-anchor="middle" font-family="Arial" font-size="22">Observed Mortality by Clinically Relevant Strata</text>\n'
        '<line x1="80" y1="420" x2="860" y2="420" stroke="black"/>\n'
        '<line x1="80" y1="70" x2="80" y2="420" stroke="black"/>\n'
        '<text x="24" y="250" transform="rotate(-90 24 250)" text-anchor="middle" font-family="Arial" font-size="16">Observed mortality proportion</text>\n'
        + ''.join(bars) + '\n</svg>'
    )
    (FIGURE_DIR / "figure_01_mortality_strata.svg").write_text(svg, encoding="utf-8")


def main():
    """Run data validation, descriptive profiling, and figure generation."""
    df = pd.read_csv(RAW_DATA)
    validate_raw_data(df)
    strata = mortality_strata(df)
    make_mortality_figure(strata)
    profile = {
        "crisp_dm_phase": "Data Understanding",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "event_count": int(df[TARGET].sum()),
        "event_prevalence": float(df[TARGET].mean()),
        "missing_values_total": int(df.isna().sum().sum()),
        "follow_up_days": {
            "min": int(df[FOLLOW_UP].min()),
            "max": int(df[FOLLOW_UP].max()),
            "median": int(df[FOLLOW_UP].median()),
        },
        "mortality_strata": strata,
    }
    write_json(profile, METRIC_DIR / "02_data_profile.json")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
