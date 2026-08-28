import json
from pathlib import Path

from fraud_monitor.cli import main
from tests.factories import make_ieee_cis_tables

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def test_show_config_command_prints_valid_json(capsys) -> None:
    exit_code = main(["show-config", "--config", str(CONFIG_PATH)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["name"] == "ieee-cis-fraud-monitor"
    assert payload["model"]["default_review_rate"] == 0.02


def test_prepare_command_writes_validated_outputs(tmp_path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    make_ieee_cis_tables().write_csvs(raw_dir)

    exit_code = main(
        [
            "prepare",
            "--config",
            str(CONFIG_PATH),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(processed_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["train_rows"] == 240
    assert payload["test_rows"] == 80
    assert Path(payload["train_path"]).is_file()
    assert Path(payload["test_path"]).is_file()
    assert Path(payload["manifest_path"]).is_file()


def test_synthetic_demo_command_writes_public_artifacts(tmp_path, capsys) -> None:
    output_dir = tmp_path / "demo"

    exit_code = main(
        [
            "build-demo",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--synthetic",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload["files"]) == 9
    assert payload["batches"] >= 14
    assert Path(payload["output_dir"], "demo_manifest.json").is_file()
