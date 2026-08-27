import json

import polars as pl
import pytest

from fraud_monitor.config import load_config
from fraud_monitor.data import (
    DataValidationError,
    join_transaction_identity,
    normalize_identity_columns,
    prepare_dataset,
)
from tests.factories import make_ieee_cis_tables


def test_identity_columns_are_normalized_without_touching_other_names() -> None:
    frame = pl.DataFrame({"TransactionID": [1], "id-12": ["Found"], "card-raw": [1]})

    normalized = normalize_identity_columns(frame)

    assert normalized.columns == ["TransactionID", "id_12", "card-raw"]


def test_join_preserves_transactions_and_marks_identity_coverage() -> None:
    config = load_config()
    tables = make_ieee_cis_tables(rows=120)

    joined = join_transaction_identity(
        pl.from_pandas(tables.train_transaction),
        pl.from_pandas(tables.train_identity),
        config,
        labeled=True,
        table_name="train",
    )

    assert joined.height == len(tables.train_transaction)
    assert joined["identity_available"].sum() == len(tables.train_identity)
    assert "id_12" in joined.columns


def test_duplicate_identity_rows_fail_validation() -> None:
    config = load_config()
    tables = make_ieee_cis_tables(rows=80)
    duplicate = pl.concat(
        [pl.from_pandas(tables.train_identity), pl.from_pandas(tables.train_identity.iloc[[0]])]
    )

    with pytest.raises(DataValidationError, match="must be unique"):
        join_transaction_identity(
            pl.from_pandas(tables.train_transaction),
            duplicate,
            config,
            labeled=True,
            table_name="train",
        )


def test_prepare_dataset_writes_validated_parquet_and_manifest(tmp_path) -> None:
    config = load_config()
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    tables = make_ieee_cis_tables(rows=240)
    tables.write_csvs(raw_dir)

    prepared = prepare_dataset(config, raw_dir=raw_dir, output_dir=output_dir)

    assert prepared.train_rows == 240
    assert prepared.test_rows == 80
    assert prepared.train_path.is_file()
    assert prepared.test_path.is_file()
    train = pl.read_parquet(prepared.train_path)
    test = pl.read_parquet(prepared.test_path)
    assert "dataset_period" in train.columns
    assert "shadow_batch" in test.columns
    assert "isFraud" not in test.columns
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["train"]["rows"] == 240
    assert len(manifest["source_files"]["train_transaction.csv"]["sha256"]) == 64

