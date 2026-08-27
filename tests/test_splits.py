import polars as pl
import pytest

from fraud_monitor.config import load_config
from fraud_monitor.splits import (
    DEVELOPMENT_BLOCK_COLUMN,
    PERIOD_COLUMN,
    PRODUCTION_BATCH_COLUMN,
    TemporalSplitError,
    assign_temporal_partitions,
    expanding_development_folds,
    validate_shadow_period,
)
from tests.factories import make_ieee_cis_tables


def test_temporal_partitioning_uses_elapsed_time_and_builds_all_blocks() -> None:
    config = load_config()
    train = pl.from_pandas(make_ieee_cis_tables(rows=240).train_transaction)

    partitioned, boundaries = assign_temporal_partitions(train, config.split)

    assert partitioned["TransactionDT"].is_sorted()
    assert set(partitioned[PERIOD_COLUMN].unique()) == {
        "development",
        "calibration",
        "acceptance",
        "production",
    }
    assert set(
        partitioned.filter(pl.col(PERIOD_COLUMN) == "development")
        .get_column(DEVELOPMENT_BLOCK_COLUMN)
        .unique()
    ) == {1, 2, 3, 4, 5}
    assert (
        partitioned.filter(pl.col(PERIOD_COLUMN) == "production")
        .get_column(PRODUCTION_BATCH_COLUMN)
        .min()
        == 1
    )
    assert boundaries.minimum_time == partitioned["TransactionDT"].min()


def test_expanding_fold_contract() -> None:
    folds = expanding_development_folds(5)

    assert [(fold.train_blocks, fold.validation_block) for fold in folds] == [
        ((1, 2), 3),
        ((1, 2, 3), 4),
        ((1, 2, 3, 4), 5),
    ]


def test_shadow_data_must_follow_training() -> None:
    train = pl.DataFrame({"TransactionDT": [10, 20, 30]})
    overlapping = pl.DataFrame({"TransactionDT": [30, 40]})

    with pytest.raises(TemporalSplitError, match="strictly after"):
        validate_shadow_period(train, overlapping)
