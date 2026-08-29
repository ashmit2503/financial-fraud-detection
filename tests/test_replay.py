import json
from dataclasses import replace

import pandas as pd

from fraud_monitor.config import load_config
from fraud_monitor.data import prepare_dataset
from fraud_monitor.modeling import train_from_prepared
from fraud_monitor.replay import run_replay
from fraud_monitor.retraining import run_retraining_evaluation
from tests.factories import make_ieee_cis_tables


def test_replay_builds_labeled_and_shadow_monitoring_artifacts(tmp_path) -> None:
    base_config = load_config()
    config = replace(
        base_config,
        monitoring=replace(
            base_config.monitoring,
            bootstrap_iterations=20,
            minimum_segment_positive=2,
            minimum_segment_negative=2,
        ),
    )
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    monitoring_dir = tmp_path / "monitoring"
    make_ieee_cis_tables(rows=600).write_csvs(raw_dir)
    prepare_dataset(config, raw_dir=raw_dir, output_dir=processed_dir)
    training = train_from_prepared(
        config,
        processed_dir=processed_dir,
        output_dir=model_dir,
        trials=1,
        max_estimators=30,
        early_stopping_rounds=5,
        n_jobs=1,
        enable_mlflow=False,
    )

    replay = run_replay(
        config,
        bundle_path=training.bundle_path,
        processed_dir=processed_dir,
        output_dir=monitoring_dir,
        bootstrap_iterations=20,
    )

    batch_metrics = pd.read_parquet(replay.batch_metrics_path)
    production = batch_metrics[batch_metrics["stream"] == "production"]
    shadow = batch_metrics[batch_metrics["stream"] == "shadow"]
    assert replay.production_batches >= 7
    assert replay.shadow_batches >= 1
    assert set(production.tail(2)["label_status"]) == {"pending"}
    assert "mature" in set(production["label_status"])
    assert set(shadow["label_status"]) == {"unavailable"}
    assert set(batch_metrics["action"]) <= {
        "continue_monitoring",
        "investigate",
        "retrain_evaluation_required",
    }
    assert replay.feature_drift_path.is_file()
    assert replay.performance_metrics_path.is_file()
    assert replay.segment_metrics_path.is_file()
    assert replay.shap_summary_path.is_file()
    assert replay.investigations_path.is_file()
    recommendations = pd.read_parquet(replay.recommendations_path)
    assert "challenger_outcome" in recommendations
    assert recommendations["challenger_outcome"].isna().all()
    manifest = json.loads(replay.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_version"] == training.model_version
    assert "TransactionID" not in pd.read_parquet(replay.feature_drift_path).columns

    retraining = run_retraining_evaluation(
        config,
        bundle_path=training.bundle_path,
        processed_dir=processed_dir,
        monitoring_dir=monitoring_dir,
        output_dir=tmp_path / "retraining",
        bootstrap_iterations=20,
        n_jobs=1,
    )
    assert retraining.summary_path.is_file()
    assert max(retraining.calibration_batches) < min(retraining.evaluation_batches)
    recommendations = pd.read_parquet(replay.recommendations_path)
    assert recommendations["challenger_evaluated"].any()
