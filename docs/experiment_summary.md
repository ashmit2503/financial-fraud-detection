# Experiment and validation handoff

## Verified full IEEE-CIS run

Kaggle notebook Version 5 completed on 2026-08-29 from repository commit `015c1f1`. The private
run produced data version `805c429ec247` and model version `fe47c8a821b3`. Only the allow-listed,
aggregate dashboard export was copied into `artifacts/demo/`; competition rows, transaction IDs,
the model bundle, MLflow runtime data, and other private artifacts were not imported.

### Model development

- The selected LightGBM used 449 features: 47 native categoricals, two past-only frequency
  encodings, and the remaining numeric and derived features.
- A constrained 20-trial search selected an unweighted estimator with 447 final trees.
- Expanding-fold PR-AUC values were 0.6331, 0.6501, and 0.6240; their mean was 0.6357.
- Isotonic calibration was selected on the disjoint calibration period. Its calibration-window
  Brier score was 0.0212, compared with 0.0214 for sigmoid.
- Frozen review thresholds were 0.9787, 0.7600, 0.3657, and 0.1102 for the 0.5%, 1%, 2%, and 5%
  calibration-period capacity targets.

### Locked acceptance result

| Measure | Verified result |
|---|---:|
| Rows / fraud prevalence | 57,310 / 4.11% |
| Dummy prior PR-AUC | 0.0411 |
| Logistic regression PR-AUC | 0.1616 |
| LightGBM PR-AUC | 0.5826 (95% CI 0.5638–0.6002) |
| Paired improvement over logistic | 0.4210 (95% CI 0.4026–0.4380) |
| ROC-AUC | 0.9080 |
| Precision / recall | 0.7397 / 0.4662 |
| Recall 95% bootstrap CI | 0.4452–0.4851 |
| Actual review rate at frozen threshold | 2.59% |
| Captured fraud amount rate | 30.08% |
| Brier score / ECE | 0.0238 / 0.0015 |
| Calibration slope / intercept | 0.9186 / -0.1310 |

The acceptance criterion is met: LightGBM materially outperformed logistic regression, and the
paired improvement interval excludes zero. The frozen threshold's later review rate differs from
the 2% calibration target because score distributions changed between the two disjoint windows.

### Production and shadow replay

- Eight consecutive production batches were replayed; six have mature labels and the last two are
  pending under the configured two-batch delay.
- The official test set produced 27 later unlabeled shadow batches. Their performance state is
  unavailable, while data and prediction drift remain measurable.
- The public export contains 35 batch rows, 3,570 feature-drift rows, 64 performance-metric rows,
  232 segment rows, and aggregate investigation and SHAP summaries.
- The policy recorded 32 `investigate` states and three `retrain_evaluation_required` states.
  Challenger evaluation remained manual and was not run, so no replacement was recommended.

Observed production PR-AUC across the six mature batches ranged from 0.4011 to 0.4972. These values
are monitoring history, not additional acceptance scores. They provide evidence that the frozen
model encountered meaningful temporal change and that the monitoring policy escalated rather than
silently presenting the stream as healthy.

## Repository validation

- Deterministic IEEE-like fixtures still exercise preparation, temporal training, calibration,
  replay, delayed labels, drift injection, public export, and all four dashboard pages.
- Public Parquet schemas are checked against an explicit allow-list and contain no transaction ID.
- The full-run demo manifest is marked non-synthetic and references only the eight approved
  aggregate tables.
- Tests cover leakage, joins, category handling, calibration separation, label maturity, monitoring
  limits, segment support, replay, retraining gates, and Streamlit loading.

## Interpretation boundaries

The run demonstrates an offline model-lifecycle design, not approval for live payment decisions.
The anonymized data cannot support protected-class fairness claims. Review capacity, two-week label
delay, simulated weekly replay, and observed control-limit breaches need operational validation
before production use. Drift alone did not deploy a challenger or replace the champion.
