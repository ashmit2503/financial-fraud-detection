import pandas as pd
import pytest

from fraud_monitor.demo import (
    PUBLIC_COLUMNS,
    REVIEW_BUDGET_FILE,
    export_demo_artifacts,
    generate_synthetic_demo,
)


def test_synthetic_demo_contains_only_aggregate_monitoring_data(tmp_path) -> None:
    result = generate_synthetic_demo(tmp_path / "demo")

    assert result.synthetic is True
    assert result.batches >= 14
    assert REVIEW_BUDGET_FILE in result.files
    for name in PUBLIC_COLUMNS:
        frame = pd.read_parquet(result.output_dir / name)
        assert "batch_id" in frame
        assert all("transactionid" not in column.lower() for column in frame)
    batches = pd.read_parquet(result.output_dir / "batch_metrics.parquet")
    assert {"healthy", "critical"} <= set(batches["drift_severity"])
    assert {"mature", "stale", "pending", "unavailable"} <= set(batches["label_status"])


def test_export_demo_strips_non_allowlisted_columns(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_demo(source)
    batches = pd.read_parquet(source / "batch_metrics.parquet")
    batches["private_note"] = "remove me"
    batches.to_parquet(source / "batch_metrics.parquet", index=False)

    result = export_demo_artifacts(source, tmp_path / "public")

    public_batches = pd.read_parquet(result.output_dir / "batch_metrics.parquet")
    assert "private_note" not in public_batches


def test_export_demo_rejects_incomplete_review_budget(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_demo(source)
    incomplete_budget = tmp_path / "incomplete_budget.parquet"
    destination = tmp_path / "public"
    pd.DataFrame({"target_review_rate": [0.02]}).to_parquet(incomplete_budget, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        export_demo_artifacts(
            source,
            destination,
            review_budget_path=incomplete_budget,
        )
    assert not destination.exists()


def test_export_demo_rejects_missing_required_monitoring_column(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_demo(source)
    destination = tmp_path / "public"
    batches_path = source / "batch_metrics.parquet"
    batches = pd.read_parquet(batches_path).drop(columns="precision")
    batches.to_parquet(batches_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        export_demo_artifacts(source, destination)
    assert not destination.exists()
