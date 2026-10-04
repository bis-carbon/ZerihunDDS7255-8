# Feature Selection and Machine Learning for Mortality Prediction in Heart Failure

## Project Overview

This public repository implements **Project 1: Best Practices for Reproducible Research in Data Science** for DDS7255. It operationalizes the Signature Assignment study on healthcare AI for predicting patient outcomes and follows the **CRISP-DM** methodology.

The project asks whether a literature-informed, parsimonious feature set improves internally cross-validated prediction of the recorded mortality event among patients with heart failure.

The study compares:

- **All-feature baseline:** 11 baseline clinical variables
- **Selected representation:** `age`, `ejection_fraction`, and `serum_creatinine`
- **Models:** logistic regression and random forest
- **Validation:** 5-fold stratified cross-validation repeated 20 times
- **Additional evaluation:** 5-fold out-of-fold calibration and descriptive subgroup analysis

The variable `time` is deliberately excluded because it is follow-up duration and is not available at the intended baseline prediction point.

> **Important interpretation:** `DEATH_EVENT` indicates whether death was recorded during each patient's observed follow-up. It is not a fixed-horizon 30-day or 1-year mortality outcome.

## Dataset Information

**Dataset:** Heart Failure Clinical Records  
**Source:** UCI Machine Learning Repository  
**DOI:** https://doi.org/10.24432/C5Z89R  
**License:** CC BY 4.0  
**Rows:** 299  
**Columns:** 13  
**Recorded death events:** 96 (32.1%)

The exact source CSV used in the project is stored in `data/raw/`. The dataset retains its original CC BY 4.0 terms. Project code is licensed separately under MIT.

## CRISP-DM Process

### 1. Business Understanding
`scripts/01_business_understanding.py`

Defines the problem statement, purpose statement, research questions, prediction point, outcome interpretation, and success criteria.

### 2. Data Understanding
`scripts/02_data_understanding.py`

Validates the source dataset, summarizes event prevalence and follow-up, and generates the descriptive mortality-strata figure.

### 3. Data Preparation
`scripts/03_data_preparation.py`

Creates reproducible baseline and selected predictor datasets and enforces leakage controls so `time` and `DEATH_EVENT` never enter predictor matrices.

### 4. Modeling
`scripts/04_modeling.py`

Evaluates four fixed configurations:

1. all-feature logistic regression
2. selected-feature logistic regression
3. all-feature random forest
4. selected-feature random forest

No grid search or random search is performed on the repeated-cross-validation results.

### 5. Evaluation
`scripts/05_evaluation.py`

Creates out-of-fold probabilities, calibration curves, calibration intercept/slope estimates, and descriptive age/sex subgroup metrics.

### 6. Deployment
`scripts/06_deployment.py`

Fits the two selected-feature models on the full course dataset, writes research-only model artifacts, creates example predictions, and publishes a model card documenting limitations.

## Repository Structure

```text
ZerihunDDS7255-8/
├── data/
│   ├── raw/
│   │   └── heart_failure_clinical_records_dataset.csv
│   ├── processed/
│   │   ├── heart_failure_baseline.csv
│   │   ├── heart_failure_selected.csv
│   │   └── data_contract.json
│   └── README.md
├── scripts/
│   ├── 01_business_understanding.py
│   ├── 02_data_understanding.py
│   ├── 03_data_preparation.py
│   ├── 04_modeling.py
│   ├── 05_evaluation.py
│   ├── 06_deployment.py
│   ├── config.py
│   ├── utils.py
│   └── run_all.py
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── tests/
│   └── test_data_contract.py
├── .github/workflows/reproducibility.yml
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/bis-carbon/ZerihunDDS7255-8.git
cd ZerihunDDS7255-8
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Validate the data contract

```bash
pytest -q
```

### 5. Reproduce the full analysis

```bash
python scripts/run_all.py
```

This executes all six CRISP-DM phases and regenerates processed data, metrics, visualizations, and research-only model artifacts.

Serialized `.joblib` models and fold-level/out-of-fold intermediate files are generated artifacts and are intentionally excluded from version control; rerunning the workflow recreates them.

### 6. Run one CRISP-DM phase

```bash
python scripts/02_data_understanding.py
python scripts/04_modeling.py
python scripts/05_evaluation.py
```

### 7. Optional notebook workflow

Start Jupyter from the repository root and open any notebook in `notebooks/`. Each notebook calls the corresponding modular script.

## Dependencies

The pinned environment is recorded in `requirements.txt`:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- matplotlib 3.10.8
- joblib 1.5.3
- pytest 8.4.2

## Modeling Configuration

### Logistic Regression

- standardized predictors
- solver: `lbfgs`
- penalty: `l2`
- `C = 1.0`
- `max_iter = 3000`
- no class weighting
- `random_state = 42`

### Random Forest

- 500 trees
- `max_features = "sqrt"`
- `min_samples_leaf = 3`
- no class weighting
- `random_state = 42`

### Validation

- `RepeatedStratifiedKFold`
- 5 folds
- 20 repeats
- `random_state = 42`
- Average Precision (AP) is computed with scikit-learn's `average_precision` scorer

## Results and Insights

| Model | ROC-AUC | AP | Accuracy | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| All-feature logistic regression | .764 | .635 | .738 | .438 | .514 | .178 |
| Selected logistic regression | .777 | .658 | .758 | .464 | .547 | .171 |
| All-feature random forest | .791 | .668 | .733 | .445 | .513 | .168 |
| Selected random forest | **.819** | **.694** | .752 | **.553** | **.584** | **.158** |

### Main Findings

- The three-feature representation achieved better **internally cross-validated** mean performance than the full baseline set for both model families.
- The selected random forest had the strongest overall discrimination and Brier score.
- Logistic regression remained the more transparent model.
- Internal calibration was broadly reasonable, but this is not evidence of external transportability.
- Performance varied descriptively across age and sex subgroups.
- The dataset is small and from one historical clinical cohort, so the models are **not for clinical use**.

## Reproducibility and Generalizability Notes

The selected features are literature-informed but not independent of this cohort because prior publications analyzed the same underlying 299 records. This reduces within-workflow feature-search bias but does not replace external validation.

The binary outcome lacks a common fixed prediction horizon because follow-up duration varies across patients. Predicted probabilities therefore refer to the cohort-specific recorded event during observed follow-up rather than a defined 30-day or 1-year risk.

## Continuous Reproducibility

The GitHub Actions workflow at `.github/workflows/reproducibility.yml` installs the pinned environment, runs the data-contract tests, executes all CRISP-DM scripts, and verifies that committed reproducibility artifacts remain synchronized with regenerated outputs on every push and pull request.

## Citation

Dataset:

> Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R

Associated article:

> Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making, 20*, 16. https://doi.org/10.1186/s12911-020-1023-5

## License

- Project code: MIT
- UCI dataset: CC BY 4.0

See `data/README.md` for dataset provenance and licensing details.

## Clinical Use Disclaimer

This repository is an academic reproducibility exercise. The models have not undergone external, prospective, or clinical-impact validation and must not be used for patient care.
