import numpy as np
import pandas as pd
import pytest

from fraud_monitor.features import (
    UNKNOWN_CATEGORY,
    CausalFeatureBuilder,
    FeaturePreprocessor,
    FeatureValidationError,
    add_stateless_features,
)


def _causal_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 100, 200, 300],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": [1, 1, 1, 2],
            "card2": [10, 10, 10, 20],
            "addr1": [100, 100, 100, 200],
            "P_emaildomain": ["a.com", "a.com", "a.com", "b.com"],
            "ProductCD": ["W", "W", "C", "H"],
            "identity_available": [False, False, True, True],
        }
    )


def test_rows_at_same_timestamp_do_not_see_each_other() -> None:
    transformed = CausalFeatureBuilder().transform_batch(_causal_rows().iloc[:3])

    assert transformed["card_addr_prior_count"].tolist() == [0, 0, 2]
    assert np.isnan(transformed.loc[0, "card_addr_prior_amount_mean"])
    assert transformed.loc[2, "card_addr_prior_amount_mean"] == pytest.approx(15.0)
    assert transformed.loc[2, "card_addr_amount_to_prior_mean"] == pytest.approx(2.0)


def test_causal_state_persists_across_ordered_batches() -> None:
    builder = CausalFeatureBuilder()
    first = _causal_rows().iloc[:2]
    second = _causal_rows().iloc[2:]

    builder.transform_batch(first)
    transformed = builder.transform_batch(second)

    assert transformed.iloc[0]["card_email_prior_count"] == 2
    assert transformed.iloc[0]["card_email_hours_since_previous"] == pytest.approx(100 / 3600)


def test_causal_builder_rejects_time_regression() -> None:
    builder = CausalFeatureBuilder()
    builder.transform_batch(_causal_rows().iloc[2:])

    with pytest.raises(FeatureValidationError, match="non-decreasing"):
        builder.transform_batch(_causal_rows().iloc[:1])


def test_stateless_features_add_time_amount_and_missingness() -> None:
    frame = _causal_rows().copy()
    frame["id_12"] = [None, "Found", None, "Found"]
    frame["V1"] = [1.0, None, 2.0, None]

    transformed = add_stateless_features(frame)

    assert transformed["amount_log1p"].iloc[0] == pytest.approx(np.log1p(10.0))
    assert {"hour_sin", "day_of_week_cos", "missing_count_identity"} <= set(
        transformed.columns
    )
    assert transformed["identity_available"].dtype == "int8"


def test_preprocessor_freezes_category_and_frequency_mappings() -> None:
    train = add_stateless_features(_causal_rows())
    train["DeviceInfo"] = [f"device-{index}" for index in range(len(train))]
    preprocessor = FeaturePreprocessor(categorical_cardinality_limit=3)

    transformed_train = preprocessor.fit_transform(train)
    future = train.iloc[[0]].copy()
    future["ProductCD"] = "unseen"
    future["DeviceInfo"] = "new-device"
    transformed_future = preprocessor.transform(future)

    assert "DeviceInfo__frequency" in transformed_train.columns
    assert transformed_future["DeviceInfo__frequency"].iloc[0] == 0.0
    assert str(transformed_future["ProductCD"].iloc[0]) == UNKNOWN_CATEGORY
    assert tuple(transformed_future.columns) == preprocessor.schema_.feature_columns


def test_preprocessor_drops_constant_and_extremely_sparse_features() -> None:
    frame = add_stateless_features(_causal_rows())
    frame["constant"] = 1
    frame["sparse"] = [None, None, None, 1.0]
    preprocessor = FeaturePreprocessor(missingness_drop_threshold=0.70)

    transformed = preprocessor.fit_transform(frame)

    assert "constant" not in transformed.columns
    assert "sparse" not in transformed.columns
