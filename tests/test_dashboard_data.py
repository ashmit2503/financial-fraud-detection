from fraud_monitor.dashboard_data import load_dashboard_data
from fraud_monitor.demo import generate_synthetic_demo


def test_dashboard_loader_reads_public_contract(tmp_path) -> None:
    directory = tmp_path / "demo"
    generate_synthetic_demo(directory)

    data = load_dashboard_data(directory)

    assert data.manifest["synthetic"] is True
    assert not data.batches.empty
    assert not data.investigations.empty
    assert set(data.batches["stream"]) == {"production", "shadow"}
