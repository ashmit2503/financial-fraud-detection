# Financial fraud detection and model monitoring

An offline-first machine-learning project that uses the IEEE-CIS Fraud Detection dataset to
demonstrate temporal validation, calibrated fraud scoring, delayed-label production replay,
distribution monitoring, diagnosis, and retraining decisions.

The implementation is intentionally organized as reusable pipelines rather than a collection of
notebooks. Full-data execution is designed for Kaggle; tests and lightweight validation use a
deterministic synthetic fixture and do not require competition data.

## Development setup

Python 3.12 is the reference runtime.

```bash
uv sync --extra dev --extra train --extra app
uv run fraud-monitor show-config
uv run pytest
```

Download the four competition files into `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

Prepare validated, chronologically partitioned Parquet tables with:

```bash
uv run fraud-monitor prepare
```

The command verifies schemas, join cardinality, targets, and temporal ordering before writing
`data/processed/train.parquet`, `data/processed/test.parquet`, and a reproducibility manifest.
Raw and processed row-level data are never committed.

Train the temporal LightGBM champion, calibrator, and fixed review thresholds with:

```bash
uv run fraud-monitor train
```

For a fast end-to-end pipeline check, use `uv run fraud-monitor train --quick --no-mlflow`.
Private model artifacts and local MLflow runs are written below `artifacts/private/` and ignored.

Replay weekly production and the later unlabeled test stream with:

```bash
uv run fraud-monitor replay --bundle artifacts/private/model/model_bundle.joblib
```

The replay writes aggregate batch, feature-drift, performance, segment, TreeSHAP, investigation,
and action tables below `artifacts/private/monitoring/`. The final two production batches remain
label-pending by design.

Manually evaluate a challenger on untouched, later mature batches with:

```bash
uv run fraud-monitor retrain-eval --bundle artifacts/private/model/model_bundle.joblib
```

The command records paired PR-AUC and recall uncertainty plus eligible segment recall checks. It
updates the recommendation artifact but never replaces or deploys the champion automatically.
