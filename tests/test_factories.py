from tests.factories import make_ieee_cis_tables


def test_synthetic_tables_match_expected_temporal_contract() -> None:
    tables = make_ieee_cis_tables(rows=120)

    assert tables.train_transaction["TransactionID"].is_unique
    assert set(tables.train_transaction["isFraud"].unique()) <= {0, 1}
    assert tables.train_identity["TransactionID"].is_unique
    assert tables.test_transaction["TransactionDT"].min() > tables.train_transaction[
        "TransactionDT"
    ].max()
    assert "id-12" in tables.test_identity.columns

