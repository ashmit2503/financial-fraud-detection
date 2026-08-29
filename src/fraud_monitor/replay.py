"""Chronological production replay and aggregate monitoring artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from fraud_monitor.config import ProjectConfig
from fraud_monitor.diagnostics import (
    build_investigation_records,
    compare_shap_summaries,
    tree_shap_summary,
)
from fraud_monitor.drift import (
    ReferenceDriftMonitor,
    numeric_jensen_shannon,
)
from fraud_monitor.features import CausalFeatureBuilder
from fraud_monitor.modeling import ModelBundle, load_model_bundle
from fraud_monitor.monitoring import (
    PerformanceMonitor,
    SegmentProfiler,
    compute_segment_metrics,
    derive_monitoring_actions,
)
from fraud_monitor.splits import PERIOD_COLUMN, PRODUCTION_BATCH_COLUMN, SHADOW_BATCH_COLUMN

CRITICAL_MONITOR_FEATURES = (
    "TransactionAmt",
    "amount_log1p",
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "identity_available",
    "missing_count_identity",
    "missing_count_transaction",
    "missing_count_vesta",
    "card_addr_prior_count",
    "card_email_prior_count",
)


@dataclass(frozen=True)
class ReplayResult:
    batch_metrics_path: Path
    feature_drift_path: Path
    performance_metrics_path: Path
    segment_metrics_path: Path
    recommendations_path: Path
    shap_summary_path: Path
    investigations_path: Path
    manifest_path: Path
    production_batches: int
    shadow_batches: int


def _engineer_and_score(
    bundle: ModelBundle,
    builder: CausalFeatureBuilder,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    engineered = builder.transform_batch(frame)
    model_features = bundle.preprocessor.transform(engineered)
    raw = bundle.model.predict_proba(model_features)[:, 1]
    scores = bundle.calibrator.predict(raw)
    return engineered, model_features, scores


def _monitor_feature_columns(
    bundle: ModelBundle, reference: pd.DataFrame, top_features: int
) -> list[str]:
    schema = bundle.preprocessor.schema_
    if schema is None:
        raise ValueError("Model bundle does not contain a fitted feature schema.")
    importance = pd.Series(bundle.model.feature_importances_, index=schema.feature_columns)
    ranked = list(importance.sort_values(ascending=False).head(top_features).index)
    selected: list[str] = []
    for feature in [*ranked, *CRITICAL_MONITOR_FEATURES]:
        if feature in reference and feature not in selected:
            selected.append(feature)
    return selected


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _base_batch_record(
    *,
    batch_id: str,
    stream: str,
    batch_number: int,
    raw: pd.DataFrame,
    scores: np.ndarray,
    bundle: ModelBundle,
    label_status: str,
) -> dict[str, Any]:
    predicted = scores >= bundle.default_threshold
    return {
        "batch_id": batch_id,
        "stream": stream,
        "batch_number": batch_number,
        "rows": len(raw),
        "elapsed_day_start": float(raw["TransactionDT"].min() / 86_400),
        "elapsed_day_end": float(raw["TransactionDT"].max() / 86_400),
        "label_status": label_status,
        "score_mean": float(scores.mean()),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "review_rate": float(predicted.mean()),
        "review_threshold": bundle.default_threshold,
        "model_version": bundle.model_version,
        "data_version": bundle.data_version,
    }


def run_replay(
    config: ProjectConfig,
    *,
    bundle_path: Path,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int | None = None,
) -> ReplayResult:
    """Replay labeled production and the later unlabeled shadow stream."""

    source = (processed_dir or config.paths.processed_dir).resolve()
    destination = (output_dir or (config.paths.artifact_dir / "private" / "monitoring")).resolve()
    train_path = source / "train.parquet"
    test_path = source / "test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Prepared train.parquet and test.parquet are required.")

    bundle = load_model_bundle(bundle_path)
    train = (
        pl.read_parquet(train_path)
        .to_pandas()
        .sort_values([config.data.time_column, config.data.transaction_id_column], kind="stable")
    )
    shadow = (
        pl.read_parquet(test_path)
        .to_pandas()
        .sort_values([config.data.time_column, config.data.transaction_id_column], kind="stable")
    )
    builder = CausalFeatureBuilder()
    reference_engineered: pd.DataFrame | None = None
    reference_features: pd.DataFrame | None = None
    reference_scores: np.ndarray | None = None
    for period in ("development", "calibration", "acceptance"):
        subset = train[train[PERIOD_COLUMN] == period].copy()
        engineered, model_features, scores = _engineer_and_score(bundle, builder, subset)
        if period == "acceptance":
            reference_engineered = engineered
            reference_features = model_features
            reference_scores = scores
    if reference_engineered is None or reference_features is None or reference_scores is None:
        raise ValueError("Acceptance reference could not be reconstructed.")

    monitored_features = _monitor_feature_columns(
        bundle, reference_features, config.monitoring.top_shap_features
    )
    schema = bundle.preprocessor.schema_
    categorical_columns = set(schema.native_categorical_columns) & set(monitored_features)
    iterations = bootstrap_iterations or config.monitoring.bootstrap_iterations
    drift_monitor = ReferenceDriftMonitor.fit(
        reference_features,
        reference_scores,
        reference_engineered[config.data.time_column],
        feature_columns=monitored_features,
        categorical_columns=categorical_columns,
        warning_quantile=config.monitoring.warning_quantile,
        critical_quantile=config.monitoring.critical_quantile,
        bootstrap_iterations=iterations,
        random_seed=config.monitoring.random_seed,
    )
    reference_target = reference_engineered[config.data.target_column].to_numpy(dtype=int)
    performance_monitor = PerformanceMonitor.fit(
        reference_target,
        reference_scores,
        reference_engineered["TransactionAmt"].to_numpy(dtype=float),
        reference_engineered[config.data.time_column],
        threshold=bundle.default_threshold,
        warning_quantile=config.monitoring.warning_quantile,
        critical_quantile=config.monitoring.critical_quantile,
        bootstrap_iterations=iterations,
        random_seed=config.monitoring.random_seed,
    )
    segment_profiler = SegmentProfiler.fit(reference_engineered)
    reference_shap = tree_shap_summary(
        bundle.model,
        reference_features,
        target=reference_target,
        random_seed=config.monitoring.random_seed,
    )

    batch_records: list[dict[str, Any]] = []
    drift_records: list[dict[str, Any]] = []
    performance_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    shap_records: list[dict[str, Any]] = []
    production = train[train[PERIOD_COLUMN] == "production"].copy()
    maximum_production_batch = int(production[PRODUCTION_BATCH_COLUMN].max())
    previous_segment_frame: pd.DataFrame | None = None

    for batch_value, raw_batch in production.groupby(PRODUCTION_BATCH_COLUMN, sort=True):
        batch_number = int(batch_value)
        batch_id = f"production_{batch_number:03d}"
        engineered, model_features, scores = _engineer_and_score(bundle, builder, raw_batch)
        feature_records, drift_summary = drift_monitor.compare(model_features, scores)
        current_shap = tree_shap_summary(
            bundle.model,
            model_features,
            target=engineered[config.data.target_column].to_numpy(dtype=int),
            random_seed=config.monitoring.random_seed,
        )
        shap_records.extend(
            {"batch_id": batch_id, "stream": "production", **record}
            for record in compare_shap_summaries(reference_shap, current_shap).to_dict(
                orient="records"
            )
        )
        labels_mature = batch_number + config.split.label_delay_batches <= maximum_production_batch
        label_status = "mature" if labels_mature else "pending"
        perf_records, perf_summary = performance_monitor.compare(
            engineered[config.data.target_column].to_numpy(dtype=int) if labels_mature else None,
            scores,
            engineered["TransactionAmt"].to_numpy(dtype=float),
            labels_available=labels_mature,
        )
        base = _base_batch_record(
            batch_id=batch_id,
            stream="production",
            batch_number=batch_number,
            raw=engineered,
            scores=scores,
            bundle=bundle,
            label_status=label_status,
        )
        base.update(
            {
                "label_available_at_batch": batch_number + config.split.label_delay_batches,
                "drift_severity": drift_summary.global_severity,
                "prediction_drift_severity": drift_summary.prediction_severity,
                "warning_features": drift_summary.warning_features,
                "critical_features": drift_summary.critical_features,
                "score_jensen_shannon": numeric_jensen_shannon(reference_scores, scores),
                "score_mean_shift": abs(float(scores.mean() - reference_scores.mean())),
                "performance_severity": perf_summary.status,
                "primary_performance_critical": perf_summary.primary_critical,
                **perf_summary.metrics,
            }
        )
        batch_records.append(base)
        drift_records.extend(
            {"batch_id": batch_id, "stream": "production", **asdict(record)}
            for record in feature_records
        )
        performance_records.extend(
            {"batch_id": batch_id, "stream": "production", **asdict(record)}
            for record in perf_records
        )

        if labels_mature:
            segment_frame = segment_profiler.add_segments(engineered)
            segment_frame["fraud_probability"] = scores
            segment_table = compute_segment_metrics(
                segment_frame,
                previous=previous_segment_frame,
                threshold=bundle.default_threshold,
                minimum_positive=config.monitoring.minimum_segment_positive,
                minimum_negative=config.monitoring.minimum_segment_negative,
                target_column=config.data.target_column,
            )
            if not segment_table.empty:
                segment_table.insert(0, "batch_id", batch_id)
                segment_table.insert(1, "stream", "production")
                segment_records.extend(segment_table.to_dict(orient="records"))
            previous_segment_frame = segment_frame

    production_metrics = derive_monitoring_actions(pd.DataFrame(batch_records))
    batch_records = production_metrics.to_dict(orient="records")

    shadow_records: list[dict[str, Any]] = []
    for batch_value, raw_batch in shadow.groupby(SHADOW_BATCH_COLUMN, sort=True):
        batch_number = int(batch_value)
        batch_id = f"shadow_{batch_number:03d}"
        engineered, model_features, scores = _engineer_and_score(bundle, builder, raw_batch)
        feature_records, drift_summary = drift_monitor.compare(model_features, scores)
        current_shap = tree_shap_summary(
            bundle.model,
            model_features,
            random_seed=config.monitoring.random_seed,
        )
        shap_records.extend(
            {"batch_id": batch_id, "stream": "shadow", **record}
            for record in compare_shap_summaries(reference_shap, current_shap).to_dict(
                orient="records"
            )
        )
        base = _base_batch_record(
            batch_id=batch_id,
            stream="shadow",
            batch_number=batch_number,
            raw=engineered,
            scores=scores,
            bundle=bundle,
            label_status="unavailable",
        )
        base.update(
            {
                "label_available_at_batch": None,
                "drift_severity": drift_summary.global_severity,
                "prediction_drift_severity": drift_summary.prediction_severity,
                "warning_features": drift_summary.warning_features,
                "critical_features": drift_summary.critical_features,
                "score_jensen_shannon": numeric_jensen_shannon(reference_scores, scores),
                "score_mean_shift": abs(float(scores.mean() - reference_scores.mean())),
                "performance_severity": "unavailable",
                "primary_performance_critical": False,
            }
        )
        shadow_records.append(base)
        drift_records.extend(
            {"batch_id": batch_id, "stream": "shadow", **asdict(record)}
            for record in feature_records
        )
    shadow_metrics = derive_monitoring_actions(pd.DataFrame(shadow_records))
    all_batch_metrics = pd.concat([production_metrics, shadow_metrics], ignore_index=True)
    recommendations = all_batch_metrics[
        [
            "batch_id",
            "stream",
            "batch_number",
            "action",
            "action_evidence",
            "model_version",
            "data_version",
        ]
    ].copy()
    recommendations["challenger_evaluated"] = False
    recommendations["retrain_recommended"] = False
    recommendations["challenger_outcome"] = pd.Series(
        pd.NA,
        index=recommendations.index,
        dtype="string",
    )
    feature_drift = pd.DataFrame(drift_records)
    performance_metrics = pd.DataFrame(performance_records)
    segment_metrics = pd.DataFrame(segment_records)
    shap_summaries = pd.DataFrame(shap_records)
    investigations = build_investigation_records(
        all_batch_metrics,
        feature_drift,
        performance_metrics,
        shap_summaries,
        segment_metrics,
    )

    destination.mkdir(parents=True, exist_ok=True)
    batch_metrics_path = destination / "batch_metrics.parquet"
    feature_drift_path = destination / "feature_drift.parquet"
    performance_metrics_path = destination / "performance_metrics.parquet"
    segment_metrics_path = destination / "segment_metrics.parquet"
    recommendations_path = destination / "recommendations.parquet"
    shap_summary_path = destination / "shap_summary.parquet"
    investigations_path = destination / "investigations.parquet"
    manifest_path = destination / "monitoring_manifest.json"
    profile_path = destination / "reference_profiles.joblib"
    all_batch_metrics.to_parquet(batch_metrics_path, index=False)
    feature_drift.to_parquet(feature_drift_path, index=False)
    performance_metrics.to_parquet(performance_metrics_path, index=False)
    segment_metrics.to_parquet(segment_metrics_path, index=False)
    recommendations.to_parquet(recommendations_path, index=False)
    shap_summaries.to_parquet(shap_summary_path, index=False)
    investigations.to_parquet(investigations_path, index=False)
    joblib.dump(
        {
            "drift_monitor": drift_monitor,
            "performance_monitor": performance_monitor,
            "segment_profiler": segment_profiler,
        },
        profile_path,
        compress=3,
    )
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": bundle.model_version,
        "data_version": bundle.data_version,
        "production_batches": int(production[PRODUCTION_BATCH_COLUMN].nunique()),
        "shadow_batches": int(shadow[SHADOW_BATCH_COLUMN].nunique()),
        "label_delay_batches": config.split.label_delay_batches,
        "monitored_features": monitored_features,
        "categorical_monitor_features": sorted(categorical_columns),
        "reference_rows": len(reference_engineered),
        "outputs": {
            "batch_metrics": batch_metrics_path.name,
            "feature_drift": feature_drift_path.name,
            "performance_metrics": performance_metrics_path.name,
            "segment_metrics": segment_metrics_path.name,
            "recommendations": recommendations_path.name,
            "shap_summary": shap_summary_path.name,
            "investigations": investigations_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return ReplayResult(
        batch_metrics_path=batch_metrics_path,
        feature_drift_path=feature_drift_path,
        performance_metrics_path=performance_metrics_path,
        segment_metrics_path=segment_metrics_path,
        recommendations_path=recommendations_path,
        shap_summary_path=shap_summary_path,
        investigations_path=investigations_path,
        manifest_path=manifest_path,
        production_batches=manifest["production_batches"],
        shadow_batches=manifest["shadow_batches"],
    )
