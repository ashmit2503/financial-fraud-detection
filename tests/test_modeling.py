import json
from typing import Any

import numpy as np
import pandas as pd

from fraud_monitor import modeling
from fraud_monitor.config import load_config
from fraud_monitor.data import prepare_dataset
from fraud_monitor.modeling import (
    fit_probability_calibrators,
    load_model_bundle,
    train_from_prepared,
)
from tests.factories import make_ieee_cis_tables


def test_lightgbm_validation_is_passed_as_eval_set(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class StubLGBMClassifier:
        def __init__(self, **parameters: Any) -> None:
            captured["parameters"] = parameters

        def fit(
            self,
            features: pd.DataFrame,
            target: np.ndarray,
            **fit_arguments: Any,
        ) -> "StubLGBMClassifier":
            captured["features"] = features
            captured["target"] = target
            captured["fit_arguments"] = fit_arguments
            return self

    monkeypatch.setattr(modeling.lgb, "LGBMClassifier", StubLGBMClassifier)
    train_features = pd.DataFrame({"amount": [10.0, 20.0, 30.0, 40.0]})
    train_target = np.array([0, 0, 1, 1])
    validation_features = pd.DataFrame({"amount": [15.0, 35.0]})
    validation_target = np.array([0, 1])

    modeling._fit_lgbm(
        {},
        train_features,
        train_target,
        validation_features=validation_features,
        validation_target=validation_target,
        estimators=20,
        early_stopping_rounds=5,
        random_seed=42,
        n_jobs=1,
    )

    fit_arguments = captured["fit_arguments"]
    assert "eval_X" not in fit_arguments
    assert "eval_y" not in fit_arguments
    assert len(fit_arguments["eval_set"]) == 1
    eval_features, eval_target = fit_arguments["eval_set"][0]
    assert eval_features is validation_features
    assert eval_target is validation_target


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
    assert bundle.temporal_cutoffs["acceptance_cutoff"] > 0
    assert summary["acceptance_metrics"]["rows"] > 0
    assert summary["tuning"]["trials"] == 1
    assert result.budget_path.is_file()
    assert result.reliability_path.is_file()
    assert "pr_auc_improvement_over_logistic" in summary["acceptance_intervals"]
