import json

import pandas as pd
import pytest

from fraud_monitor.dashboard_data import load_dashboard_data
from fraud_monitor.demo import REVIEW_BUDGET_FILE, generate_synthetic_demo


def test_dashboard_loader_reads_public_contract(tmp_path) -> None:
    directory = tmp_path / "demo"
    generate_synthetic_demo(directory)

    data = load_dashboard_data(directory)

    assert data.manifest["synthetic"] is True
    assert not data.batches.empty
    assert not data.investigations.empty
    assert set(data.batches["stream"]) == {"production", "shadow"}


def test_dashboard_loader_rejects_invalid_review_budget_schema(tmp_path) -> None:
    directory = tmp_path / "demo"
    generate_synthetic_demo(directory)
    pd.DataFrame({"target_review_rate": [0.02]}).to_parquet(
        directory / REVIEW_BUDGET_FILE, index=False
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        load_dashboard_data(directory)


def test_dashboard_loader_rejects_stale_manifest_inventory(tmp_path) -> None:
    directory = tmp_path / "demo"
    generate_synthetic_demo(directory)
    manifest_path = directory / "demo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].remove(REVIEW_BUDGET_FILE)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory"):
        load_dashboard_data(directory)


def test_dashboard_loader_rejects_incomplete_public_table(tmp_path) -> None:
    directory = tmp_path / "demo"
    generate_synthetic_demo(directory)
    batches_path = directory / "batch_metrics.parquet"
    batches = pd.read_parquet(batches_path).drop(columns="precision")
    batches.to_parquet(batches_path, index=False)

    with pytest.raises(ValueError, match="must contain exactly"):
        load_dashboard_data(directory)
