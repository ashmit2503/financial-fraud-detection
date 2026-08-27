from dataclasses import replace
from pathlib import Path

import pytest

from fraud_monitor.config import SplitConfig, load_config


def test_base_configuration_loads_from_repository() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.name == "ieee-cis-fraud-monitor"
    assert config.split.acceptance_end == 0.70
    assert config.model.default_review_rate == 0.02
    assert config.paths.raw_dir.is_absolute()


def test_split_boundaries_must_be_strictly_increasing() -> None:
    config = load_config(Path("configs/base.yaml"))

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config.split, calibration_end=config.split.development_end)


def test_split_requires_complete_production_range() -> None:
    with pytest.raises(ValueError, match="must end at 1.0"):
        SplitConfig(
            development_end=0.5,
            calibration_end=0.6,
            acceptance_end=0.7,
            production_end=0.9,
            development_blocks=5,
            batch_seconds=604_800,
            label_delay_batches=2,
        )
