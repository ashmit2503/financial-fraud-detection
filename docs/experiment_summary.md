# Experiment and validation handoff

## What is verified in this repository

- Deterministic IEEE-like data passes preparation, temporal training, calibration, weekly replay,
  delayed-label monitoring, public export, and all four Streamlit pages.
- Controlled numeric, categorical, missingness, prediction, and prevalence changes exercise alert
  transitions; stable synthetic batches remain healthy.
- The replay contains at least seven production windows plus a later unlabeled shadow stream.
- The last two production windows remain pending, while deliberately stale labels are surfaced as
  investigation evidence in the public demo.
- Challenger evaluation respects chronological train/calibration/evaluation cutoffs and records a
  rejection when promotion criteria are not met.

## Full IEEE-CIS run handoff

Run `notebooks/kaggle_pipeline.ipynb` with the competition dataset attached and full execution
enabled. Preserve these private outputs:

- `processed/manifest.json`
- `model/training_summary.json`
- `model/acceptance_review_budgets.parquet`
- `model/acceptance_reliability.parquet`
- `monitoring/monitoring_manifest.json`
- `retraining/retraining_evaluation.json`, when manually triggered

Record the following without copying transaction rows into Git:

| Question | Source |
|---|---|
| Did LightGBM beat logistic PR-AUC? | `acceptance_metrics.pr_auc` vs `baseline_pr_auc.logistic` |
| Is the improvement reliable? | `acceptance_intervals.pr_auc_improvement_over_logistic` |
| What recall does 2% capacity achieve? | `acceptance_metrics.recall` and its interval |
| Is calibration usable? | Brier, ECE, slope/intercept, and reliability table |
| Are seven mature production windows available? | monitoring manifest and batch metrics |
| Did controlled or observed drift lead to the expected policy state? | investigations and recommendations |

Do not promote a result if the paired improvement interval includes zero, if 2%-capacity recall is
operationally inadequate, or if data-contract failures make the reference unreliable.

## Public demo interpretation

The committed aggregate demo is a deterministic scenario generator. Its values demonstrate UI and
policy behavior, not model quality. In particular, its incident at production batches 7–8 and stale
label at batch 10 are intentional diagnostic cases.
