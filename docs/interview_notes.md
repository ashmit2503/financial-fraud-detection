# CV and interview talking points

## Concise CV bullets

- Built an offline-first IEEE-CIS fraud lifecycle with elapsed-time cross-validation, past-only
  velocity features, calibrated LightGBM scoring, and fixed review-capacity thresholds.
- Simulated weekly production with 14-day delayed labels and empirical controls for data,
  prediction, calibration, performance, and segment drift.
- Added aggregate TreeSHAP diagnosis and a conservative challenger gate using paired bootstrap
  improvement, recall non-inferiority, and segment-regression checks.
- Shipped a public Streamlit monitor backed only by allow-listed aggregate Parquet artifacts, with
  reproducible CLI pipelines, MLflow tracking, tests, and Kaggle orchestration.

## Design decisions to explain

**Why elapsed-time splits?** Row quantiles can hide long sparse or dense periods. Percentages of the
observed time range better represent chronological deployment boundaries.

**Why no SMOTE?** Synthetic neighbors are difficult to justify for anonymized mixed-type temporal
transactions and can blur time. Fold-local class weighting is simpler and leakage-safe.

**Why calibrate separately?** Ranking quality and probability quality are different. A disjoint
window prevents threshold selection from contaminating model fitting or acceptance.

**Why bootstrap control limits?** Fixed folklore thresholds ignore natural feature scale and batch
variation. Resampled reference day blocks estimate the behavior of a normal seven-day window.

**Why doesn't drift trigger retraining?** Covariate movement can be harmless, temporary, or caused
by a data-quality issue. Mature performance and a better untouched challenger are stronger evidence.

**What is the strongest leakage test?** Causal entity features are computed before updates, and all
rows sharing a timestamp see the same pre-timestamp state.

**What would v2 add?** Live data contracts, reviewer outcomes, cost-sensitive objectives, governance
workflows, authenticated serving, orchestration, and a protected-attribute fairness plan where
legally and ethically appropriate data exists.
