# Model Card

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
