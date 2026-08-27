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

The raw Kaggle files belong in `data/raw/` and are never committed. Detailed data instructions,
pipeline commands, architecture, results, and monitoring interpretation will be added as each
capability becomes usable.

