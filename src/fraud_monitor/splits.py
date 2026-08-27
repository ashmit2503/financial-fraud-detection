"""Chronological dataset partitioning and replay batch assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from fraud_monitor.config import SplitConfig

PERIOD_COLUMN = "dataset_period"
DEVELOPMENT_BLOCK_COLUMN = "development_block"
PRODUCTION_BATCH_COLUMN = "production_batch"
SHADOW_BATCH_COLUMN = "shadow_batch"


class TemporalSplitError(ValueError):
    """Raised when chronological partition assumptions are violated."""


@dataclass(frozen=True)
class TemporalBoundaries:
    minimum_time: int
    maximum_time: int
    development_cutoff: float
    calibration_cutoff: float
    acceptance_cutoff: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_blocks: tuple[int, ...]
    validation_block: int


def expanding_development_folds(blocks: int = 5) -> tuple[TemporalFold, ...]:
    """Return expanding folds that reserve each block from three onward for validation."""

    if blocks < 3:
        raise TemporalSplitError("At least three development blocks are required.")
    return tuple(
        TemporalFold(
            name=f"blocks_1_{validation - 1}_to_{validation}",
            train_blocks=tuple(range(1, validation)),
            validation_block=validation,
        )
        for validation in range(3, blocks + 1)
    )


def _validated_time_values(frame: pl.DataFrame, time_column: str) -> np.ndarray:
    if time_column not in frame.columns:
        raise TemporalSplitError(f"Missing required time column: {time_column}")
    if frame.height == 0:
        raise TemporalSplitError("Cannot partition an empty dataset.")
    if frame[time_column].null_count() > 0:
        raise TemporalSplitError(f"{time_column} must not contain null values.")

    values = frame[time_column].cast(pl.Int64).to_numpy()
    if values.max() <= values.min():
        raise TemporalSplitError(f"{time_column} must span more than one instant.")
    return values


def assign_temporal_partitions(
    frame: pl.DataFrame,
    split: SplitConfig,
    *,
    time_column: str = "TransactionDT",
) -> tuple[pl.DataFrame, TemporalBoundaries]:
    """Sort labeled data and assign development, calibration, acceptance, and production."""

    ordered = frame.sort(time_column)
    times = _validated_time_values(ordered, time_column)
    minimum = int(times.min())
    maximum = int(times.max())
    span = maximum - minimum
    position = (times - minimum) / span

    periods = np.select(
        [
            position < split.development_end,
            position < split.calibration_end,
            position < split.acceptance_end,
        ],
        ["development", "calibration", "acceptance"],
        default="production",
    )

    development_mask = position < split.development_end
    raw_blocks = np.floor(
        position[development_mask] / split.development_end * split.development_blocks
    ).astype(int)
    clipped_blocks = np.clip(raw_blocks + 1, 1, split.development_blocks)
    development_blocks: list[int | None] = [None] * ordered.height
    for row_index, block in zip(np.flatnonzero(development_mask), clipped_blocks, strict=True):
        development_blocks[int(row_index)] = int(block)

    acceptance_cutoff = minimum + split.acceptance_end * span
    production_mask = periods == "production"
    calculated_batches = (
        np.floor((times[production_mask] - acceptance_cutoff) / split.batch_seconds).astype(int)
        + 1
    )
    production_batches: list[int | None] = [None] * ordered.height
    for row_index, batch in zip(
        np.flatnonzero(production_mask), calculated_batches, strict=True
    ):
        production_batches[int(row_index)] = int(batch)

    result = ordered.with_columns(
        pl.Series(PERIOD_COLUMN, periods, dtype=pl.String),
        pl.Series(DEVELOPMENT_BLOCK_COLUMN, development_blocks, dtype=pl.Int16),
        pl.Series(PRODUCTION_BATCH_COLUMN, production_batches, dtype=pl.Int32),
    )

    period_counts = result.group_by(PERIOD_COLUMN).len().to_dict(as_series=False)
    observed = set(period_counts[PERIOD_COLUMN])
    expected = {"development", "calibration", "acceptance", "production"}
    if observed != expected:
        raise TemporalSplitError(
            f"Every temporal period must contain rows; observed {sorted(observed)}."
        )
    observed_blocks = set(
        result.filter(pl.col(PERIOD_COLUMN) == "development")
        .get_column(DEVELOPMENT_BLOCK_COLUMN)
        .drop_nulls()
        .to_list()
    )
    expected_blocks = set(range(1, split.development_blocks + 1))
    if observed_blocks != expected_blocks:
        raise TemporalSplitError(
            f"Every development block must contain rows; observed {sorted(observed_blocks)}."
        )

    return result, TemporalBoundaries(
        minimum_time=minimum,
        maximum_time=maximum,
        development_cutoff=minimum + split.development_end * span,
        calibration_cutoff=minimum + split.calibration_end * span,
        acceptance_cutoff=acceptance_cutoff,
    )


def validate_shadow_period(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    time_column: str = "TransactionDT",
) -> None:
    """Require the unlabeled shadow stream to begin strictly after labeled history."""

    train_times = _validated_time_values(train, time_column)
    test_times = _validated_time_values(test, time_column)
    if int(test_times.min()) <= int(train_times.max()):
        raise TemporalSplitError(
            "Shadow/test transactions must begin strictly after labeled training transactions."
        )


def assign_shadow_batches(
    frame: pl.DataFrame,
    split: SplitConfig,
    *,
    time_column: str = "TransactionDT",
) -> pl.DataFrame:
    """Assign consecutive replay batches to the unlabeled shadow stream."""

    ordered = frame.sort(time_column)
    times = _validated_time_values(ordered, time_column)
    batch = ((times - times.min()) // split.batch_seconds + 1).astype(np.int32)
    return ordered.with_columns(pl.Series(SHADOW_BATCH_COLUMN, batch, dtype=pl.Int32))
