"""Public-safe aggregate demo artifact contracts and deterministic fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoBuildResult:
    output_dir: Path
    files: tuple[str, ...]
    batches: int
    synthetic: bool


PUBLIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "batch_metrics.parquet": (
        "batch_id",
        "stream",
        "batch_number",
        "rows",
        "elapsed_day_start",
        "elapsed_day_end",
        "label_status",
        "score_mean",
        "score_p50",
        "score_p95",
        "review_rate",
        "review_threshold",
        "model_version",
        "data_version",
        "drift_severity",
        "prediction_drift_severity",
        "warning_features",
        "critical_features",
        "score_jensen_shannon",
        "score_mean_shift",
        "performance_severity",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "fraud_prevalence",
        "captured_fraud_amount_rate",
        "brier_score",
        "expected_calibration_error",
        "action",
        "action_evidence",
    ),
    "feature_drift.parquet": (
        "batch_id",
        "stream",
        "feature",
        "feature_kind",
        "metric",
        "value",
        "warning_limit",
        "critical_limit",
        "severity",
    ),
    "performance_metrics.parquet": (
        "batch_id",
        "stream",
        "metric",
        "value",
        "reference_value",
        "warning_limit",
        "critical_limit",
        "direction",
        "status",
    ),
    "segment_metrics.parquet": (
        "batch_id",
        "stream",
        "segment",
        "segment_value",
        "window_batches",
        "rows",
        "positives",
        "negatives",
        "status",
        "fraud_prevalence",
        "precision",
        "recall",
        "false_positive",
        "false_negative",
        "captured_fraud_amount_rate",
    ),
    "recommendations.parquet": (
        "batch_id",
        "stream",
        "batch_number",
        "action",
        "action_evidence",
        "model_version",
        "data_version",
        "challenger_evaluated",
        "retrain_recommended",
        "challenger_outcome",
    ),
    "shap_summary.parquet": (
        "batch_id",
        "stream",
        "feature",
        "reference_mean_abs_shap",
        "reference_mean_shap",
        "current_mean_abs_shap",
        "current_mean_shap",
        "importance_change",
        "absolute_importance_change",
        "reference_rank",
        "current_rank",
    ),
    "investigations.parquet": (
        "batch_id",
        "stream",
        "batch_number",
        "label_status",
        "action",
        "classification",
        "likely_driver",
        "driver_evidence",
        "top_false_negative_segment",
        "top_false_positive_segment",
        "top_prevalence_segment",
        "recommended_action",
    ),
}


def _validate_public_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    allowed = PUBLIC_COLUMNS[name]
    selected = frame.loc[:, [column for column in allowed if column in frame]].copy()
    forbidden = [column for column in selected if "transactionid" in column.lower()]
    if forbidden:
        raise ValueError(f"Public artifact {name} contains row identifiers: {forbidden}")
    if "batch_id" not in selected:
        raise ValueError(f"Public artifact {name} must contain batch_id.")
    return selected


def _write_tables(
    tables: dict[str, pd.DataFrame],
    destination: Path,
    manifest: dict[str, object],
) -> DemoBuildResult:
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        _validate_public_table(name, frame).to_parquet(destination / name, index=False)
    (destination / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    batches = int(tables["batch_metrics.parquet"]["batch_id"].nunique())
    return DemoBuildResult(
        output_dir=destination,
        files=tuple(sorted([*tables, "demo_manifest.json"])),
        batches=batches,
        synthetic=bool(manifest.get("synthetic", False)),
    )


def export_demo_artifacts(
    source_dir: Path,
    destination: Path,
    *,
    review_budget_path: Path | None = None,
) -> DemoBuildResult:
    """Copy only allow-listed aggregate columns into a public dashboard directory."""

    tables: dict[str, pd.DataFrame] = {}
    for name in PUBLIC_COLUMNS:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required monitoring artifact is missing: {path}")
        tables[name] = pd.read_parquet(path)
    manifest = {
        "manifest_version": 1,
        "synthetic": False,
        "source": "aggregate_monitoring_export",
        "model_version": str(tables["batch_metrics.parquet"]["model_version"].iloc[-1]),
        "data_version": str(tables["batch_metrics.parquet"]["data_version"].iloc[-1]),
        "files": sorted(PUBLIC_COLUMNS),
    }
    result = _write_tables(tables, destination.resolve(), manifest)
    if review_budget_path is not None:
        if not review_budget_path.is_file():
            raise FileNotFoundError(review_budget_path)
        budget = pd.read_parquet(review_budget_path)
        budget.loc[
            :,
            [
                column
                for column in (
                    "target_review_rate",
                    "review_rate",
                    "precision",
                    "recall",
                    "captured_fraud_amount_rate",
                    "threshold",
                )
                if column in budget
            ],
        ].to_parquet(destination / "acceptance_review_budgets.parquet", index=False)
    return result


def _synthetic_batch_metrics() -> pd.DataFrame:
    pr_auc = [0.51, 0.53, 0.52, 0.50, 0.49, 0.48, 0.38, 0.35, 0.46, np.nan, np.nan, np.nan]
    recall = [0.46, 0.49, 0.48, 0.45, 0.44, 0.43, 0.30, 0.26, 0.41, np.nan, np.nan, np.nan]
    records = []
    for index in range(1, 13):
        degraded = index in {7, 8}
        stale = index == 10
        pending = index in {11, 12}
        label_status = "stale" if stale else "pending" if pending else "mature"
        drift = "critical" if index == 7 else "warning" if index in {6, 8} else "healthy"
        performance = (
            "unavailable"
            if stale or pending
            else "critical"
            if degraded
            else "warning"
            if index == 6
            else "healthy"
        )
        action = (
            "retrain_evaluation_required"
            if index == 8
            else "investigate"
            if index in {7, 10}
            else "continue_monitoring"
        )
        records.append(
            {
                "batch_id": f"production_{index:03d}",
                "stream": "production",
                "batch_number": index,
                "rows": 15_000 + index * 230,
                "elapsed_day_start": 120 + (index - 1) * 7,
                "elapsed_day_end": 126 + (index - 1) * 7,
                "label_status": label_status,
                "score_mean": 0.037 + index * 0.0008 + (0.012 if degraded else 0),
                "score_p50": 0.012 + index * 0.0002,
                "score_p95": 0.19 + index * 0.004 + (0.06 if degraded else 0),
                "review_rate": 0.02 + (0.014 if degraded else index * 0.0002),
                "review_threshold": 0.214,
                "model_version": "m-6f9d2c1a",
                "data_version": "d-81c4e7b2",
                "drift_severity": drift,
                "prediction_drift_severity": "warning" if degraded else "healthy",
                "warning_features": 4 if index in {6, 8} else 0,
                "critical_features": 3 if index == 7 else 0,
                "score_jensen_shannon": 0.19 if degraded else 0.04 + index * 0.002,
                "score_mean_shift": 0.018 if degraded else 0.003,
                "performance_severity": performance,
                "pr_auc": pr_auc[index - 1],
                "roc_auc": np.nan if stale or pending else 0.87 - (0.08 if degraded else 0),
                "precision": np.nan if stale or pending else 0.31 - (0.09 if degraded else 0),
                "recall": recall[index - 1],
                "f1": np.nan if stale or pending else 0.37 - (0.10 if degraded else 0),
                "false_positive_rate": np.nan if stale or pending else 0.014,
                "false_negative_rate": np.nan if stale or pending else 1 - recall[index - 1],
                "true_negative": np.nan if stale or pending else 14_000,
                "false_positive": np.nan if stale or pending else 210,
                "false_negative": np.nan if stale or pending else 120 + index * 3,
                "true_positive": np.nan if stale or pending else 105 - index * 2,
                "fraud_prevalence": np.nan if stale or pending else 0.034 + index * 0.0004,
                "captured_fraud_amount_rate": (
                    np.nan if stale or pending else 0.58 - (0.18 if degraded else 0)
                ),
                "brier_score": np.nan if stale or pending else 0.029 + (0.012 if degraded else 0),
                "expected_calibration_error": (
                    np.nan if stale or pending else 0.018 + (0.026 if degraded else 0)
                ),
                "action": action,
                "action_evidence": (
                    "Labels are stale beyond the configured delay"
                    if stale
                    else "Primary performance guardrail breached in two mature batches"
                    if index == 8
                    else "Critical data and performance breach"
                    if index == 7
                    else "No sustained or critical guardrail breach"
                ),
            }
        )
    for index in range(1, 3):
        records.append(
            {
                **records[-1],
                "batch_id": f"shadow_{index:03d}",
                "stream": "shadow",
                "batch_number": index,
                "rows": 18_000 + index * 500,
                "elapsed_day_start": 204 + (index - 1) * 7,
                "elapsed_day_end": 210 + (index - 1) * 7,
                "label_status": "unavailable",
                "drift_severity": "warning" if index == 2 else "healthy",
                "performance_severity": "unavailable",
                "action": "continue_monitoring",
                "action_evidence": "Performance unavailable for unlabeled shadow data",
            }
        )
    return pd.DataFrame(records)


def _synthetic_feature_drift(batches: pd.DataFrame) -> pd.DataFrame:
    features = [
        ("TransactionAmt", "numeric", "normalized_wasserstein"),
        ("amount_log1p", "numeric", "normalized_wasserstein"),
        ("ProductCD", "categorical", "jensen_shannon"),
        ("card4", "categorical", "jensen_shannon"),
        ("DeviceType", "categorical", "unseen_category_rate"),
        ("identity_available", "numeric", "missingness_shift"),
    ]
    records = []
    for batch in batches.itertuples(index=False):
        for position, (feature, kind, metric) in enumerate(features):
            critical = batch.batch_id == "production_007" and position < 3
            warning = batch.batch_id in {"production_006", "production_008", "shadow_002"}
            value = (
                0.41 + position * 0.03 if critical else 0.24 if warning else 0.08 + position * 0.005
            )
            records.append(
                {
                    "batch_id": batch.batch_id,
                    "stream": batch.stream,
                    "feature": feature,
                    "feature_kind": kind,
                    "metric": metric,
                    "value": value,
                    "warning_limit": 0.18,
                    "critical_limit": 0.34,
                    "severity": "critical" if critical else "warning" if warning else "healthy",
                }
            )
    return pd.DataFrame(records)


def _synthetic_performance(batches: pd.DataFrame) -> pd.DataFrame:
    limits = {
        "pr_auc": (0.51, 0.43, 0.39, "lower"),
        "recall": (0.47, 0.39, 0.34, "lower"),
        "brier_score": (0.029, 0.036, 0.041, "higher"),
        "expected_calibration_error": (0.018, 0.032, 0.041, "higher"),
    }
    records = []
    for batch in batches[batches["stream"] == "production"].itertuples(index=False):
        for metric, (reference, warning, critical, direction) in limits.items():
            value = getattr(batch, metric)
            if pd.isna(value):
                status = "unavailable"
            elif direction == "lower":
                status = (
                    "critical" if value < critical else "warning" if value < warning else "healthy"
                )
            else:
                status = (
                    "critical" if value > critical else "warning" if value > warning else "healthy"
                )
            records.append(
                {
                    "batch_id": batch.batch_id,
                    "stream": batch.stream,
                    "metric": metric,
                    "value": value,
                    "reference_value": reference,
                    "warning_limit": warning,
                    "critical_limit": critical,
                    "direction": direction,
                    "status": status,
                }
            )
    return pd.DataFrame(records)


def _synthetic_segments(batches: pd.DataFrame) -> pd.DataFrame:
    values = [("ProductCD", "W"), ("ProductCD", "C"), ("card4", "visa"), ("card4", "mastercard")]
    records = []
    for batch in batches[
        (batches["stream"] == "production") & (batches["label_status"] == "mature")
    ].itertuples(index=False):
        for position, (segment, value) in enumerate(values):
            degraded = batch.batch_number in {7, 8} and value in {"C", "mastercard"}
            records.append(
                {
                    "batch_id": batch.batch_id,
                    "stream": "production",
                    "segment": segment,
                    "segment_value": value,
                    "window_batches": 1,
                    "rows": 2_400 + position * 300,
                    "positives": 48 + position * 5,
                    "negatives": 2_300 + position * 250,
                    "status": "reported",
                    "fraud_prevalence": 0.035 + position * 0.004 + (0.018 if degraded else 0),
                    "precision": 0.29 - (0.11 if degraded else 0),
                    "recall": 0.46 - (0.21 if degraded else 0),
                    "false_positive": 42 + position * 5,
                    "false_negative": 20 + position * 2 + (18 if degraded else 0),
                    "captured_fraud_amount_rate": 0.56 - (0.19 if degraded else 0),
                }
            )
    return pd.DataFrame(records)


def _synthetic_shap(batches: pd.DataFrame) -> pd.DataFrame:
    features = [
        "TransactionAmt",
        "ProductCD",
        "card_addr_prior_count",
        "card4",
        "DeviceType",
        "amount_log1p",
    ]
    records = []
    for batch in batches.itertuples(index=False):
        for rank, feature in enumerate(features, start=1):
            reference = 0.34 / rank
            change = (
                (0.12 / rank)
                if batch.batch_id in {"production_007", "production_008"}
                else 0.008 * np.sin(batch.batch_number + rank)
            )
            records.append(
                {
                    "batch_id": batch.batch_id,
                    "stream": batch.stream,
                    "feature": feature,
                    "reference_mean_abs_shap": reference,
                    "reference_mean_shap": 0.01 / rank,
                    "current_mean_abs_shap": reference + change,
                    "current_mean_shap": 0.01 / rank + change / 4,
                    "importance_change": change,
                    "absolute_importance_change": abs(change),
                    "reference_rank": rank,
                    "current_rank": rank,
                }
            )
    return pd.DataFrame(records)


def generate_synthetic_demo(destination: Path) -> DemoBuildResult:
    """Build deterministic, aggregate-only dashboard data with controlled incidents."""

    batches = _synthetic_batch_metrics()
    recommendations = batches[
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
    recommendations["challenger_evaluated"] = recommendations["batch_id"] == "production_008"
    recommendations["retrain_recommended"] = False
    recommendations["challenger_outcome"] = np.where(
        recommendations["challenger_evaluated"], "challenger_rejected", None
    )
    investigations = pd.DataFrame(
        [
            {
                "batch_id": "production_007",
                "stream": "production",
                "batch_number": 7,
                "label_status": "mature",
                "action": "investigate",
                "classification": (
                    "covariate_drift, prediction_drift, confirmed_performance_degradation"
                ),
                "likely_driver": "TransactionAmt",
                "driver_evidence": json.dumps(
                    [{"feature": "TransactionAmt", "severity": "critical", "driver_score": 7.4}]
                ),
                "top_false_negative_segment": "ProductCD=C",
                "top_false_positive_segment": "card4=mastercard",
                "top_prevalence_segment": "ProductCD=C",
                "recommended_action": (
                    "Investigate amount mix and the ProductCD=C error concentration."
                ),
            },
            {
                "batch_id": "production_008",
                "stream": "production",
                "batch_number": 8,
                "label_status": "mature",
                "action": "retrain_evaluation_required",
                "classification": "confirmed_performance_degradation, calibration_drift",
                "likely_driver": "ProductCD",
                "driver_evidence": json.dumps(
                    [{"feature": "ProductCD", "severity": "warning", "driver_score": 4.9}]
                ),
                "top_false_negative_segment": "ProductCD=C",
                "top_false_positive_segment": "card4=mastercard",
                "top_prevalence_segment": "ProductCD=C",
                "recommended_action": (
                    "Evaluate a challenger; do not replace the champion without paired evidence."
                ),
            },
            {
                "batch_id": "production_010",
                "stream": "production",
                "batch_number": 10,
                "label_status": "stale",
                "action": "investigate",
                "classification": "data_quality_failure",
                "likely_driver": "label_delivery",
                "driver_evidence": "[]",
                "top_false_negative_segment": None,
                "top_false_positive_segment": None,
                "top_prevalence_segment": None,
                "recommended_action": (
                    "Repair label delivery before drawing performance conclusions."
                ),
            },
        ]
    )
    tables = {
        "batch_metrics.parquet": batches,
        "feature_drift.parquet": _synthetic_feature_drift(batches),
        "performance_metrics.parquet": _synthetic_performance(batches),
        "segment_metrics.parquet": _synthetic_segments(batches),
        "recommendations.parquet": recommendations,
        "shap_summary.parquet": _synthetic_shap(batches),
        "investigations.parquet": investigations,
    }
    result = _write_tables(
        tables,
        destination.resolve(),
        {
            "manifest_version": 1,
            "synthetic": True,
            "source": "deterministic_controlled_scenarios",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "model_version": "m-6f9d2c1a",
            "data_version": "d-81c4e7b2",
            "description": "Aggregate-only portfolio demo; not IEEE-CIS results.",
            "files": sorted(PUBLIC_COLUMNS),
        },
    )
    budget = pd.DataFrame(
        {
            "target_review_rate": [0.005, 0.01, 0.02, 0.05],
            "review_rate": [0.005, 0.01, 0.02, 0.05],
            "precision": [0.48, 0.40, 0.31, 0.21],
            "recall": [0.19, 0.31, 0.47, 0.69],
            "captured_fraud_amount_rate": [0.24, 0.38, 0.58, 0.77],
            "threshold": [0.52, 0.37, 0.214, 0.096],
        }
    )
    budget.to_parquet(destination / "acceptance_review_budgets.parquet", index=False)
    return result
