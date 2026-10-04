"""Execute all CRISP-DM phase scripts in sequence."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_business_understanding.py",
    "02_data_understanding.py",
    "03_data_preparation.py",
    "04_modeling.py",
    "05_evaluation.py",
    "06_deployment.py",
]


def main():
    """Run each CRISP-DM phase script and stop immediately if any phase fails."""
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, str(HERE / step)], check=True)
    print("\nAll CRISP-DM phases completed successfully.")


if __name__ == "__main__":
    main()
