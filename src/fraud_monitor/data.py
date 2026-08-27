"""IEEE-CIS ingestion, validation, Parquet preparation, and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from fraud_monitor.config import ProjectConfig
from fraud_monitor.splits import (
    PERIOD_COLUMN,
    assign_shadow_batches,
    assign_temporal_partitions,
    expanding_development_folds,
    validate_shadow_period,
)


class DataValidationError(ValueError):
    """Raised when input data violates the project data contract."""


@dataclass(frozen=True)
class PreparedDataset:
    train_path: Path
    test_path: Path
    manifest_path: Path
    train_rows: int
    test_rows: int


def normalize_identity_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize Kaggle test identity names such as ``id-12`` to ``id_12``."""

    mapping = {
        name: name.replace("-", "_")
        for name in frame.columns
        if name.startswith("id-")
    }
    resulting_names = [mapping.get(name, name) for name in frame.columns]
    if len(set(resulting_names)) != len(resulting_names):
        raise DataValidationError("Identity column normalization creates duplicate names.")
    return frame.rename(mapping)


def _require_columns(frame: pl.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{table_name} is missing required columns: {missing}")


def _validate_unique_id(frame: pl.DataFrame, id_column: str, table_name: str) -> None:
    if frame[id_column].null_count() > 0:
        raise DataValidationError(f"{table_name}.{id_column} must not contain nulls.")
    if frame[id_column].n_unique() != frame.height:
        raise DataValidationError(f"{table_name}.{id_column} must be unique.")


def validate_transaction_table(
    frame: pl.DataFrame,
    config: ProjectConfig,
    *,
    labeled: bool,
    table_name: str,
) -> None:
    required = {
        config.data.transaction_id_column,
        config.data.time_column,
        "TransactionAmt",
    }
    if labeled:
        required.add(config.data.target_column)
    _require_columns(frame, required, table_name)
    _validate_unique_id(frame, config.data.transaction_id_column, table_name)
    if frame[config.data.time_column].null_count() > 0:
        raise DataValidationError(f"{table_name}.{config.data.time_column} must not contain nulls.")
    if labeled:
        values = set(frame[config.data.target_column].drop_nulls().unique().to_list())
        if values - {0, 1} or frame[config.data.target_column].null_count() > 0:
            raise DataValidationError("isFraud must contain only non-null binary values.")


def join_transaction_identity(
    transaction: pl.DataFrame,
    identity: pl.DataFrame,
    config: ProjectConfig,
    *,
    labeled: bool,
    table_name: str,
) -> pl.DataFrame:
    """Validate and left-join transaction and optional identity data."""

    identity = normalize_identity_columns(identity)
    id_column = config.data.transaction_id_column
    validate_transaction_table(transaction, config, labeled=labeled, table_name=table_name)
    _require_columns(identity, {id_column}, f"{table_name}_identity")
    _validate_unique_id(identity, id_column, f"{table_name}_identity")
    if "identity_available" in transaction.columns or "identity_available" in identity.columns:
        raise DataValidationError("identity_available is a reserved derived column.")

    identity = identity.with_columns(pl.lit(True).alias("identity_available"))
    joined = transaction.join(identity, on=id_column, how="left", validate="1:1").with_columns(
        pl.col("identity_available").fill_null(False)
    )
    if joined.height != transaction.height:
        raise DataValidationError("Identity join changed transaction row count.")
    return joined.sort(config.data.time_column)


def read_ieee_csv(path: Path) -> pl.DataFrame:
    """Read a competition CSV with stable identifier types and strict parsing."""

    try:
        return pl.read_csv(
            path,
            infer_schema_length=50_000,
            schema_overrides={"TransactionID": pl.Int64, "TransactionDT": pl.Int64},
            null_values=["", "NA", "NaN", "null"],
            try_parse_dates=False,
            ignore_errors=False,
        )
    except Exception as exc:  # pragma: no cover - Polars emits several concrete parser errors
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_profile(
    frame: pl.DataFrame,
    config: ProjectConfig,
    *,
    include_target: bool,
) -> dict[str, Any]:
    time_column = config.data.time_column
    profile: dict[str, Any] = {
        "rows": frame.height,
        "columns": frame.width,
        "minimum_time": int(frame[time_column].min()),
        "maximum_time": int(frame[time_column].max()),
        "identity_coverage": float(frame["identity_available"].mean()),
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "missing_fraction": {
            name: frame[name].null_count() / frame.height for name in frame.columns
        },
    }
    if include_target:
        profile["fraud_prevalence"] = float(frame[config.data.target_column].mean())
        profile["period_rows"] = {
            row[PERIOD_COLUMN]: row["len"]
            for row in frame.group_by(PERIOD_COLUMN).len().sort(PERIOD_COLUMN).to_dicts()
        }
    return profile


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def prepare_dataset(
    config: ProjectConfig,
    *,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> PreparedDataset:
    """Build validated, partitioned Parquet tables and a reproducibility manifest."""

    source_dir = (raw_dir or config.paths.raw_dir).resolve()
    destination = (output_dir or config.paths.processed_dir).resolve()
    files = {name: source_dir / name for name in config.data.expected_files}
    missing_files = sorted(name for name, path in files.items() if not path.is_file())
    if missing_files:
        raise FileNotFoundError(f"Missing IEEE-CIS input files in {source_dir}: {missing_files}")

    train_transaction = read_ieee_csv(files["train_transaction.csv"])
    train_identity = read_ieee_csv(files["train_identity.csv"])
    test_transaction = read_ieee_csv(files["test_transaction.csv"])
    test_identity = read_ieee_csv(files["test_identity.csv"])

    train = join_transaction_identity(
        train_transaction,
        train_identity,
        config,
        labeled=True,
        table_name="train_transaction",
    )
    test = join_transaction_identity(
        test_transaction,
        test_identity,
        config,
        labeled=False,
        table_name="test_transaction",
    )
    validate_shadow_period(train, test, time_column=config.data.time_column)
    train, boundaries = assign_temporal_partitions(
        train,
        config.split,
        time_column=config.data.time_column,
    )
    test = assign_shadow_batches(test, config.split, time_column=config.data.time_column)

    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.parquet"
    test_path = destination / "test.parquet"
    manifest_path = destination / "manifest.json"
    train.write_parquet(train_path, compression="zstd", statistics=True)
    test.write_parquet(test_path, compression="zstd", statistics=True)

    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project": config.name,
        "source_files": {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in files.items()
        },
        "train": _table_profile(train, config, include_target=True),
        "test": _table_profile(test, config, include_target=False),
        "temporal_boundaries": boundaries.to_dict(),
        "development_folds": [
            asdict(fold) for fold in expanding_development_folds(config.split.development_blocks)
        ],
        "outputs": {"train": train_path.name, "test": test_path.name},
    }
    _write_json(manifest_path, manifest)
    return PreparedDataset(
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        train_rows=train.height,
        test_rows=test.height,
    )

