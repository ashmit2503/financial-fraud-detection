"""TreeSHAP summaries and evidence-backed monitoring investigations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _sample_positions(
    rows: int,
    target: np.ndarray | None,
    *,
    maximum_rows: int,
    random_seed: int,
) -> np.ndarray:
    if rows <= maximum_rows:
        return np.arange(rows)
    rng = np.random.default_rng(random_seed)
    if target is None or np.unique(target).size < 2:
        return np.sort(rng.choice(rows, size=maximum_rows, replace=False))

    selected: list[np.ndarray] = []
    labels, counts = np.unique(target, return_counts=True)
    allocations = np.maximum(1, np.floor(maximum_rows * counts / counts.sum()).astype(int))
    while allocations.sum() > maximum_rows:
        allocations[int(np.argmax(allocations))] -= 1
    while allocations.sum() < maximum_rows:
        candidates = counts - allocations
        allocations[int(np.argmax(candidates))] += 1
    for label, allocation in zip(labels, allocations, strict=True):
        positions = np.flatnonzero(target == label)
        selected.append(rng.choice(positions, size=min(allocation, len(positions)), replace=False))
    combined = np.concatenate(selected)
    if len(combined) < maximum_rows:
        remaining = np.setdiff1d(np.arange(rows), combined, assume_unique=False)
        fill = rng.choice(remaining, size=maximum_rows - len(combined), replace=False)
        combined = np.concatenate([combined, fill])
    return np.sort(combined)


def tree_shap_summary(
    model: Any,
    features: pd.DataFrame,
    *,
    target: np.ndarray | None = None,
    maximum_rows: int = 5_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Calculate a compact TreeSHAP summary using LightGBM contributions."""

    if features.empty:
        raise ValueError("TreeSHAP requires at least one feature row.")
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive.")
    labels = None if target is None else np.asarray(target, dtype=int)
    if labels is not None and len(labels) != len(features):
        raise ValueError("TreeSHAP features and target must align.")
    positions = _sample_positions(
        len(features), labels, maximum_rows=maximum_rows, random_seed=random_seed
    )
    sampled = features.iloc[positions]
    sampled_target = None if labels is None else labels[positions]
    contributions = np.asarray(model.booster_.predict(sampled, pred_contrib=True), dtype=float)
    if contributions.shape != (len(sampled), len(features.columns) + 1):
        raise ValueError("LightGBM returned an unexpected contribution matrix.")
    values = contributions[:, :-1]
    summary = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "sample_rows": len(sampled),
        }
    )
    summary["fraud_mean_shap"] = np.nan
    summary["legitimate_mean_shap"] = np.nan
    if sampled_target is not None:
        if np.any(sampled_target == 1):
            summary["fraud_mean_shap"] = values[sampled_target == 1].mean(axis=0)
        if np.any(sampled_target == 0):
            summary["legitimate_mean_shap"] = values[sampled_target == 0].mean(axis=0)
    return summary.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def compare_shap_summaries(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Compare global TreeSHAP importance without retaining row-level explanations."""

    required = {"feature", "mean_abs_shap", "mean_shap"}
    if not required <= set(reference) or not required <= set(current):
        raise ValueError(f"SHAP summaries must contain {sorted(required)}.")
    baseline = reference.rename(
        columns={
            "mean_abs_shap": "reference_mean_abs_shap",
            "mean_shap": "reference_mean_shap",
        }
    )[["feature", "reference_mean_abs_shap", "reference_mean_shap"]]
    observed = current.rename(
        columns={
            "mean_abs_shap": "current_mean_abs_shap",
            "mean_shap": "current_mean_shap",
        }
    )[["feature", "current_mean_abs_shap", "current_mean_shap"]]
    comparison = baseline.merge(observed, on="feature", validate="one_to_one")
    comparison["importance_change"] = (
        comparison["current_mean_abs_shap"] - comparison["reference_mean_abs_shap"]
    )
    comparison["absolute_importance_change"] = comparison["importance_change"].abs()
    comparison["reference_rank"] = comparison["reference_mean_abs_shap"].rank(
        method="min", ascending=False
    )
    comparison["current_rank"] = comparison["current_mean_abs_shap"].rank(
        method="min", ascending=False
    )
    return comparison.sort_values("absolute_importance_change", ascending=False, ignore_index=True)


def rank_potential_drivers(
    drift_records: pd.DataFrame,
    shap_comparison: pd.DataFrame,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    """Rank alert drivers from drift severity, missingness, and SHAP change."""

    alerts = drift_records[drift_records["severity"].isin(["warning", "critical"])].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "driver_score",
                "severity",
                "drift_excess",
                "missingness_change",
                "absolute_importance_change",
                "alert_reasons",
            ]
        )
    denominator = (alerts["critical_limit"] - alerts["warning_limit"]).abs().clip(1e-9)
    alerts["drift_excess"] = ((alerts["value"] - alerts["warning_limit"]) / denominator).clip(
        lower=0, upper=5
    )
    alerts["missingness_change"] = np.where(
        alerts["metric"] == "missingness_shift", alerts["value"], 0.0
    )
    alerts["severity_score"] = alerts["severity"].map({"warning": 1.0, "critical": 2.0})
    aggregated = (
        alerts.groupby("feature", as_index=False)
        .agg(
            severity_score=("severity_score", "max"),
            drift_excess=("drift_excess", "max"),
            missingness_change=("missingness_change", "max"),
            alert_reasons=("metric", lambda values: ", ".join(sorted(set(values)))),
        )
        .merge(
            shap_comparison[["feature", "absolute_importance_change"]],
            on="feature",
            how="left",
        )
    )
    aggregated["absolute_importance_change"] = aggregated["absolute_importance_change"].fillna(0.0)
    scale = max(float(aggregated["absolute_importance_change"].max()), 1e-9)
    aggregated["driver_score"] = (
        2 * aggregated["severity_score"]
        + aggregated["drift_excess"]
        + 2 * aggregated["missingness_change"]
        + aggregated["absolute_importance_change"] / scale
    )
    aggregated["severity"] = aggregated["severity_score"].map({1.0: "warning", 2.0: "critical"})
    return (
        aggregated.sort_values("driver_score", ascending=False).head(limit).reset_index(drop=True)
    )


def _top_segment(segment_records: pd.DataFrame, metric: str) -> str | None:
    reported = segment_records[segment_records["status"] == "reported"].dropna(subset=[metric])
    if reported.empty:
        return None
    row = reported.sort_values(metric, ascending=False).iloc[0]
    return f"{row['segment']}={row['segment_value']}"


def build_investigation_records(
    batch_metrics: pd.DataFrame,
    drift_records: pd.DataFrame,
    performance_records: pd.DataFrame,
    shap_comparisons: pd.DataFrame,
    segment_records: pd.DataFrame,
) -> pd.DataFrame:
    """Create one aggregate investigation record for every warning or critical batch."""

    records: list[dict[str, Any]] = []
    for batch in batch_metrics.itertuples(index=False):
        if batch.drift_severity == "healthy" and batch.performance_severity in {
            "healthy",
            "unavailable",
        }:
            continue
        batch_drift = drift_records[drift_records["batch_id"] == batch.batch_id]
        batch_performance = performance_records[performance_records["batch_id"] == batch.batch_id]
        batch_shap = shap_comparisons[shap_comparisons["batch_id"] == batch.batch_id]
        batch_segments = segment_records[segment_records["batch_id"] == batch.batch_id]
        drivers = rank_potential_drivers(batch_drift, batch_shap)
        alerted_metrics = set(
            batch_performance.loc[
                batch_performance["status"].isin(["warning", "critical"]), "metric"
            ]
        )
        alerted_drift_metrics = set(
            batch_drift.loc[batch_drift["severity"].isin(["warning", "critical"]), "metric"]
        )
        classifications: list[str] = []
        if "missingness_shift" in alerted_drift_metrics:
            classifications.append("data_quality_failure")
        if alerted_drift_metrics - {"missingness_shift"}:
            classifications.append("covariate_drift")
        if batch.prediction_drift_severity in {"warning", "critical"}:
            classifications.append("prediction_drift")
        if "fraud_prevalence_shift" in alerted_metrics:
            classifications.append("prevalence_shift")
        if alerted_metrics & {"brier_score", "expected_calibration_error"}:
            classifications.append("calibration_drift")
        if alerted_metrics & {
            "pr_auc",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        }:
            classifications.append("confirmed_performance_degradation")
        if not classifications:
            classifications.append("monitoring_signal")

        driver_payload = drivers.to_dict(orient="records")
        records.append(
            {
                "batch_id": batch.batch_id,
                "stream": batch.stream,
                "batch_number": batch.batch_number,
                "label_status": batch.label_status,
                "action": batch.action,
                "classification": ", ".join(classifications),
                "likely_driver": None if drivers.empty else drivers.iloc[0]["feature"],
                "driver_evidence": json.dumps(driver_payload, default=float),
                "top_false_negative_segment": _top_segment(batch_segments, "false_negative"),
                "top_false_positive_segment": _top_segment(batch_segments, "false_positive"),
                "top_prevalence_segment": _top_segment(batch_segments, "fraud_prevalence"),
                "recommended_action": batch.action_evidence,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
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
        ],
    )
