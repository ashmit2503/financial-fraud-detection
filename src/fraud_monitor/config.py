"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    raw_dir: Path
    processed_dir: Path
    artifact_dir: Path
    demo_dir: Path


@dataclass(frozen=True)
class DataConfig:
    transaction_id_column: str
    time_column: str
    target_column: str
    expected_files: tuple[str, ...]


@dataclass(frozen=True)
class SplitConfig:
    development_end: float
    calibration_end: float
    acceptance_end: float
    production_end: float
    development_blocks: int
    batch_seconds: int
    label_delay_batches: int

    def __post_init__(self) -> None:
        boundaries = (
            self.development_end,
            self.calibration_end,
            self.acceptance_end,
            self.production_end,
        )
        if not all(0.0 < value <= 1.0 for value in boundaries):
            raise ValueError("Split boundaries must be in (0, 1].")
        if tuple(sorted(boundaries)) != boundaries or len(set(boundaries)) != len(boundaries):
            raise ValueError("Split boundaries must be strictly increasing.")
        if self.production_end != 1.0:
            raise ValueError("The production boundary must end at 1.0.")
        if self.development_blocks < 3:
            raise ValueError("At least three development blocks are required.")
        if self.batch_seconds <= 0 or self.label_delay_batches < 0:
            raise ValueError("Batch duration must be positive and label delay non-negative.")


@dataclass(frozen=True)
class ModelConfig:
    review_rates: tuple[float, ...]
    default_review_rate: float
    categorical_cardinality_limit: int
    missingness_drop_threshold: float
    optuna_trials: int

    def __post_init__(self) -> None:
        if not self.review_rates or not all(0.0 < rate < 1.0 for rate in self.review_rates):
            raise ValueError("Review rates must be non-empty values in (0, 1).")
        if self.default_review_rate not in self.review_rates:
            raise ValueError("The default review rate must be one of review_rates.")
        if self.categorical_cardinality_limit < 2:
            raise ValueError("Categorical cardinality limit must be at least two.")
        if not 0.0 < self.missingness_drop_threshold <= 1.0:
            raise ValueError("Missingness threshold must be in (0, 1].")
        if self.optuna_trials < 1:
            raise ValueError("At least one optimization trial is required.")


@dataclass(frozen=True)
class MonitoringConfig:
    warning_quantile: float
    critical_quantile: float
    bootstrap_iterations: int
    random_seed: int
    minimum_segment_positive: int
    minimum_segment_negative: int
    top_shap_features: int

    def __post_init__(self) -> None:
        if not 0.0 < self.warning_quantile < self.critical_quantile < 1.0:
            raise ValueError("Monitoring quantiles must satisfy 0 < warning < critical < 1.")
        if self.bootstrap_iterations < 1:
            raise ValueError("Bootstrap iterations must be positive.")
        if min(self.minimum_segment_positive, self.minimum_segment_negative) < 1:
            raise ValueError("Segment support requirements must be positive.")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int
    paths: PathConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    monitoring: MonitoringConfig


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration section: {key}")
    return mapping[key]


def load_config(path: str | Path = "configs/base.yaml") -> ProjectConfig:
    """Load and validate project configuration.

    Relative data paths are resolved from the repository root, assumed to be the
    parent of the configuration directory.
    """

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    project = _required(raw, "project")
    paths = _required(raw, "paths")
    data = _required(raw, "data")
    split = _required(raw, "split")
    model = _required(raw, "model")
    monitoring = _required(raw, "monitoring")
    root = config_path.parent.parent

    return ProjectConfig(
        name=str(_required(project, "name")),
        random_seed=int(_required(project, "random_seed")),
        paths=PathConfig(
            raw_dir=_resolve_path(_required(paths, "raw_dir"), root),
            processed_dir=_resolve_path(_required(paths, "processed_dir"), root),
            artifact_dir=_resolve_path(_required(paths, "artifact_dir"), root),
            demo_dir=_resolve_path(_required(paths, "demo_dir"), root),
        ),
        data=DataConfig(
            transaction_id_column=str(_required(data, "transaction_id_column")),
            time_column=str(_required(data, "time_column")),
            target_column=str(_required(data, "target_column")),
            expected_files=tuple(str(value) for value in _required(data, "expected_files")),
        ),
        split=SplitConfig(**split),
        model=ModelConfig(
            review_rates=tuple(float(value) for value in _required(model, "review_rates")),
            default_review_rate=float(_required(model, "default_review_rate")),
            categorical_cardinality_limit=int(_required(model, "categorical_cardinality_limit")),
            missingness_drop_threshold=float(_required(model, "missingness_drop_threshold")),
            optuna_trials=int(_required(model, "optuna_trials")),
        ),
        monitoring=MonitoringConfig(**monitoring),
    )
