import json

import numpy as np

from fraud_monitor.config import load_config
from fraud_monitor.data import prepare_dataset
from fraud_monitor.modeling import (
    fit_probability_calibrators,
    load_model_bundle,
    train_from_prepared,
)
from tests.factories import make_ieee_cis_tables


def test_calibrator_selection_produces_valid_probabilities() -> None:
    raw = np.array([0.01, 0.05, 0.10, 0.30, 0.70, 0.90])
    target = np.array([0, 0, 0, 1, 1, 1])

    calibrator, candidates = fit_probability_calibrators(raw, target)
    calibrated = calibrator.predict(raw)

    assert {candidate.method for candidate in candidates} == {"sigmoid", "isotonic"}
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_quick_temporal_training_writes_loadable_bundle(tmp_path) -> None:
    config = load_config()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    make_ieee_cis_tables(rows=600).write_csvs(raw_dir)
    prepare_dataset(config, raw_dir=raw_dir, output_dir=processed_dir)

    result = train_from_prepared(
        config,
        processed_dir=processed_dir,
        output_dir=model_dir,
        trials=1,
        max_estimators=30,
        early_stopping_rounds=5,
        n_jobs=1,
        enable_mlflow=False,
    )

    bundle = load_model_bundle(result.bundle_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert bundle.model_version == result.model_version
    assert bundle.default_threshold == bundle.thresholds[0.02]
    assert summary["acceptance_metrics"]["rows"] > 0
    assert summary["tuning"]["trials"] == 1
    assert result.budget_path.is_file()

