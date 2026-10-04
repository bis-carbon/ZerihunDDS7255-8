# Data

## Source

**Dataset:** Heart Failure Clinical Records  
**Repository:** UCI Machine Learning Repository  
**DOI:** https://doi.org/10.24432/C5Z89R  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The raw dataset contains 299 patient records and 13 columns, including the
binary outcome `DEATH_EVENT`.

## Provenance

The exact file used for this project is:

`data/raw/heart_failure_clinical_records_dataset.csv`

SHA-256:

`8e73886de77e8ccab902a616352ed596ef0b1db43c37bef14db0e68af1885037`

## Prediction-time decision

`time` is follow-up duration and is deliberately excluded from the baseline
predictor set because it is not available at the intended baseline prediction
point. `DEATH_EVENT` is the outcome and is never used as a predictor.

## Processed files

Running `python scripts/03_data_preparation.py` creates:

- `data/processed/heart_failure_baseline.csv`
- `data/processed/heart_failure_selected.csv`
- `data/processed/data_contract.json`

## Required attribution

Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning
Repository. https://doi.org/10.24432/C5Z89R
