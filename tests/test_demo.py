import pandas as pd

from fraud_monitor.demo import PUBLIC_COLUMNS, export_demo_artifacts, generate_synthetic_demo


def test_synthetic_demo_contains_only_aggregate_monitoring_data(tmp_path) -> None:
    result = generate_synthetic_demo(tmp_path / "demo")

    assert result.synthetic is True
    assert result.batches >= 14
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
