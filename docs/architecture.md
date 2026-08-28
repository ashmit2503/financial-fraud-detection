# Architecture and artifact contracts

## Boundaries

The project has three deliberate security and lifecycle boundaries:

1. **Private row-level boundary:** raw CSVs, processed Parquet rows, fitted models, aggregate state,
   and MLflow runs remain under ignored data or `artifacts/private/` paths.
2. **Immutable deployment boundary:** the champion bundle contains everything needed to reproduce
   features and calibrated scores, including causal state through acceptance, but production replay
   cannot refit it.
3. **Public aggregate boundary:** `build-demo` reads monitoring outputs through an explicit column
   allow-list. The Streamlit app imports only the aggregate loader.

## ModelBundle

`ModelBundle` is a versioned joblib contract containing:

- fitted `FeaturePreprocessor` and feature schema;
- causal entity state through the deployment cutoff;
- fitted LightGBM estimator and selected probability calibrator;
- review-rate thresholds and default capacity;
- source-data and model versions;
- training parameters, creation time, and temporal cutoffs.

The bundle is private and intentionally excluded from Git.

## Monitoring tables

| Artifact | Granularity | Purpose |
|---|---|---|
| `batch_metrics.parquet` | batch | Label maturity, score movement, performance, health, and action |
| `feature_drift.parquet` | batch × feature × statistic | Observed drift, empirical limits, and severity |
| `performance_metrics.parquet` | mature batch × metric | Reference comparison and control status |
| `segment_metrics.parquet` | batch × segment value | Support, prevalence, errors, precision, and recall |
| `shap_summary.parquet` | batch × feature | Acceptance/current TreeSHAP importance change |
| `investigations.parquet` | alerted batch | Diagnosis, likely drivers, segment evidence, and next action |
| `recommendations.parquet` | batch | Policy state and optional challenger outcome |

Unmatured performance records use `unavailable`; they are not imputed and are excluded from
consecutive mature-breach logic.

## Stateful replay

The causal builder processes development, calibration, acceptance, production, and shadow rows in
strict time order. Each batch is transformed before entity state moves forward. Model preprocessing,
calibration, thresholds, and reference limits stay frozen. Only causal counters and timestamps
advance.

## Failure behavior

Preparation fails closed on missing files, duplicate IDs, invalid target values, unexpected join
cardinality, overlapping temporal periods, missing development blocks, or a shadow period that is
not strictly later. Feature transformation fails on missing fitted inputs. Monitoring reports
non-finite or structurally missing drift evidence as critical rather than silently passing it.
