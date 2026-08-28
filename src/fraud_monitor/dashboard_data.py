"""Validated aggregate-only data access for the public dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_monitor.demo import PUBLIC_COLUMNS, REVIEW_BUDGET_COLUMNS, REVIEW_BUDGET_FILE


@dataclass(frozen=True)
class DashboardData:
    manifest: dict[str, object]
    batches: pd.DataFrame
    feature_drift: pd.DataFrame
    performance: pd.DataFrame
    segments: pd.DataFrame
    recommendations: pd.DataFrame
    shap: pd.DataFrame
    investigations: pd.DataFrame
    review_budgets: pd.DataFrame


def load_dashboard_data(directory: str | Path) -> DashboardData:
    """Load and validate precomputed public artifacts without accessing raw data."""

    root = Path(directory).resolve()
    manifest_path = root / "demo_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dashboard manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported dashboard manifest version.")

    frames: dict[str, pd.DataFrame] = {}
    for name, allowed_columns in PUBLIC_COLUMNS.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Dashboard artifact is missing: {path}")
        frame = pd.read_parquet(path)
        if set(frame) != set(allowed_columns):
            raise ValueError(
                f"Dashboard artifact {name} must contain exactly {sorted(allowed_columns)}."
            )
        frames[name] = frame

    budget_path = root / REVIEW_BUDGET_FILE
    expected_files = set(PUBLIC_COLUMNS)
    if budget_path.is_file():
        expected_files.add(REVIEW_BUDGET_FILE)
    if set(manifest.get("files", ())) != expected_files:
        raise ValueError("Dashboard manifest file inventory does not match available artifacts.")
    review_budgets = (
        pd.read_parquet(budget_path)
        if budget_path.is_file()
        else pd.DataFrame(columns=REVIEW_BUDGET_COLUMNS)
    )
    if budget_path.is_file() and set(review_budgets) != set(REVIEW_BUDGET_COLUMNS):
        raise ValueError(
            f"Dashboard artifact {REVIEW_BUDGET_FILE} must contain exactly "
            f"{sorted(REVIEW_BUDGET_COLUMNS)}."
        )
    return DashboardData(
        manifest=manifest,
        batches=frames["batch_metrics.parquet"],
        feature_drift=frames["feature_drift.parquet"],
        performance=frames["performance_metrics.parquet"],
        segments=frames["segment_metrics.parquet"],
        recommendations=frames["recommendations.parquet"],
        shap=frames["shap_summary.parquet"],
        investigations=frames["investigations.parquet"],
        review_budgets=review_budgets,
    )
