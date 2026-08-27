"""Leakage-safe feature construction for chronological fraud scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_monitor.splits import (
    DEVELOPMENT_BLOCK_COLUMN,
    PERIOD_COLUMN,
    PRODUCTION_BATCH_COLUMN,
    SHADOW_BATCH_COLUMN,
)

UNKNOWN_CATEGORY = "__UNKNOWN__"
MISSING_CATEGORY = "__MISSING__"

TRANSACTION_CATEGORICAL_COLUMNS = {
    "ProductCD",
    *(f"card{index}" for index in range(1, 7)),
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{index}" for index in range(1, 10)),
}
IDENTITY_CATEGORICAL_COLUMNS = {
    "DeviceType",
    "DeviceInfo",
    *(f"id_{index:02d}" for index in range(12, 39)),
}
MODEL_EXCLUDED_COLUMNS = {
    "TransactionID",
    "isFraud",
    "TransactionDT",
    PERIOD_COLUMN,
    DEVELOPMENT_BLOCK_COLUMN,
    PRODUCTION_BATCH_COLUMN,
    SHADOW_BATCH_COLUMN,
}


class FeatureValidationError(ValueError):
    """Raised when features cannot be produced without violating temporal assumptions."""


def _missing_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0, index=frame.index, dtype="int16")
    return frame[columns].isna().sum(axis=1).astype("int16")


def add_stateless_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic row-level features without fitting any data-dependent state."""

    required = {"TransactionDT", "TransactionAmt"}
    missing = required - set(frame.columns)
    if missing:
        raise FeatureValidationError(f"Missing stateless feature inputs: {sorted(missing)}")

    result = frame.copy()
    transaction_dt = pd.to_numeric(result["TransactionDT"], errors="raise").astype("int64")
    amount = pd.to_numeric(result["TransactionAmt"], errors="coerce")
    elapsed_days = transaction_dt / 86_400.0
    hour = (transaction_dt % 86_400) / 3_600.0
    day_of_week = (transaction_dt % 604_800) / 86_400.0

    result["amount_log1p"] = np.log1p(amount.clip(lower=0))
    result["amount_fraction"] = amount - np.floor(amount)
    result["elapsed_day"] = elapsed_days.astype("float32")
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    result["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7).astype("float32")
    result["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7).astype("float32")

    identity_columns = [
        column
        for column in result.columns
        if column.startswith("id_") or column in {"DeviceType", "DeviceInfo"}
    ]
    vesta_columns = [
        column
        for column in result.columns
        if len(column) > 1 and column[0] in {"C", "D", "M", "V"} and column[1:].isdigit()
    ]
    transaction_columns = [
        column
        for column in result.columns
        if column not in identity_columns
        and column not in MODEL_EXCLUDED_COLUMNS
        and column not in {"identity_available"}
    ]
    result["missing_count_identity"] = _missing_count(result, identity_columns)
    result["missing_count_vesta"] = _missing_count(result, vesta_columns)
    result["missing_count_transaction"] = _missing_count(result, transaction_columns)
    if "identity_available" in result:
        result["identity_available"] = result["identity_available"].fillna(False).astype("int8")
    return result


def _key_value(value: Any) -> Any:
    return MISSING_CATEGORY if pd.isna(value) else value


@dataclass
class EntityState:
    count: int = 0
    amount_sum: float = 0.0
    last_time: int | None = None


@dataclass
class CausalFeatureBuilder:
    """Maintain prior-only aggregate state across chronological batches.

    Rows sharing the same timestamp are all transformed before any of them update
    state, preventing arbitrary row order within a timestamp from leaking signal.
    """

    entity_keys: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "card_addr": ("card1", "card2", "addr1"),
            "card_email": ("card1", "P_emaildomain"),
        }
    )
    time_column: str = "TransactionDT"
    amount_column: str = "TransactionAmt"
    id_column: str = "TransactionID"
    _state: dict[str, dict[tuple[Any, ...], EntityState]] = field(default_factory=dict)
    _latest_time: int | None = None

    def reset(self) -> None:
        self._state = {name: {} for name in self.entity_keys}
        self._latest_time = None

    def __post_init__(self) -> None:
        if not self._state:
            self.reset()

    def transform_batch(self, frame: pd.DataFrame, *, update_state: bool = True) -> pd.DataFrame:
        """Build causal features for one ordered batch and optionally advance state."""

        required = {self.time_column, self.amount_column, *self.entity_keys.values()}
        flattened_required = {
            value
            for item in required
            for value in (item if isinstance(item, tuple) else (item,))
        }
        missing = flattened_required - set(frame.columns)
        if missing:
            raise FeatureValidationError(f"Missing causal feature inputs: {sorted(missing)}")
        if frame.empty:
            return add_stateless_features(frame)

        sort_columns = [self.time_column]
        if self.id_column in frame:
            sort_columns.append(self.id_column)
        ordered = frame.sort_values(sort_columns, kind="stable").copy()
        times = pd.to_numeric(ordered[self.time_column], errors="raise").astype("int64")
        if self._latest_time is not None and int(times.min()) < self._latest_time:
            raise FeatureValidationError(
                "Causal batches must arrive in non-decreasing TransactionDT order."
            )

        feature_values: dict[str, np.ndarray] = {}
        for entity_name in self.entity_keys:
            feature_values[f"{entity_name}_prior_count"] = np.zeros(len(ordered), dtype=np.int32)
            feature_values[f"{entity_name}_hours_since_previous"] = np.full(
                len(ordered), np.nan, dtype=np.float32
            )
            feature_values[f"{entity_name}_prior_amount_mean"] = np.full(
                len(ordered), np.nan, dtype=np.float32
            )
            feature_values[f"{entity_name}_amount_to_prior_mean"] = np.full(
                len(ordered), np.nan, dtype=np.float32
            )

        grouped_positions = pd.Series(np.arange(len(ordered)), index=ordered.index).groupby(
            times, sort=True
        )
        for timestamp, position_series in grouped_positions:
            positions = position_series.to_numpy()
            pending_updates: list[tuple[str, tuple[Any, ...], float]] = []
            for position in positions:
                row = ordered.iloc[int(position)]
                amount = (
                    float(row[self.amount_column])
                    if pd.notna(row[self.amount_column])
                    else math.nan
                )
                for entity_name, columns in self.entity_keys.items():
                    key = tuple(_key_value(row[column]) for column in columns)
                    state = self._state[entity_name].get(key, EntityState())
                    feature_values[f"{entity_name}_prior_count"][position] = state.count
                    if state.last_time is not None:
                        feature_values[f"{entity_name}_hours_since_previous"][position] = (
                            int(timestamp) - state.last_time
                        ) / 3_600.0
                    if state.count > 0:
                        prior_mean = state.amount_sum / state.count
                        feature_values[f"{entity_name}_prior_amount_mean"][position] = prior_mean
                        if prior_mean != 0 and not math.isnan(amount):
                            feature_values[f"{entity_name}_amount_to_prior_mean"][position] = (
                                amount / prior_mean
                            )
                    pending_updates.append((entity_name, key, amount))

            if update_state:
                for entity_name, key, amount in pending_updates:
                    state = self._state[entity_name].setdefault(key, EntityState())
                    state.count += 1
                    if not math.isnan(amount):
                        state.amount_sum += amount
                    state.last_time = int(timestamp)

        for name, values in feature_values.items():
            ordered[name] = values
        if update_state:
            self._latest_time = int(times.max())
        return add_stateless_features(ordered)


@dataclass
class FeatureSchema:
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    native_categorical_columns: tuple[str, ...]
    frequency_encoded_columns: tuple[str, ...]
    category_levels: dict[str, tuple[str, ...]]
    frequency_maps: dict[str, dict[str, float]]


class FeaturePreprocessor:
    """Fit a stable LightGBM-ready schema on historical features only."""

    def __init__(
        self,
        *,
        categorical_cardinality_limit: int = 500,
        missingness_drop_threshold: float = 0.995,
    ) -> None:
        self.categorical_cardinality_limit = categorical_cardinality_limit
        self.missingness_drop_threshold = missingness_drop_threshold
        self.schema_: FeatureSchema | None = None

    @staticmethod
    def _is_categorical(column: str) -> bool:
        return column in TRANSACTION_CATEGORICAL_COLUMNS | IDENTITY_CATEGORICAL_COLUMNS

    @staticmethod
    def _normalized_category(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(MISSING_CATEGORY)

    def fit(self, frame: pd.DataFrame) -> FeaturePreprocessor:
        candidates = [column for column in frame.columns if column not in MODEL_EXCLUDED_COLUMNS]
        dropped: list[str] = []
        native: list[str] = []
        frequency_encoded: list[str] = []
        category_levels: dict[str, tuple[str, ...]] = {}
        frequency_maps: dict[str, dict[str, float]] = {}

        for column in candidates:
            missing_fraction = float(frame[column].isna().mean())
            if missing_fraction > self.missingness_drop_threshold or frame[column].nunique(
                dropna=False
            ) <= 1:
                dropped.append(column)
                continue
            if not self._is_categorical(column):
                continue
            normalized = self._normalized_category(frame[column])
            cardinality = int(normalized.nunique(dropna=False))
            if cardinality > self.categorical_cardinality_limit:
                frequency_encoded.append(column)
                frequencies = normalized.value_counts(normalize=True)
                frequency_maps[column] = {
                    str(key): float(value) for key, value in frequencies.items()
                }
            else:
                native.append(column)
                levels = tuple(sorted(str(value) for value in normalized.unique()))
                category_levels[column] = (*levels, UNKNOWN_CATEGORY)

        output_columns = [column for column in candidates if column not in dropped]
        output_columns = [
            f"{column}__frequency" if column in frequency_encoded else column
            for column in output_columns
        ]
        active_inputs = tuple(column for column in candidates if column not in dropped)
        self.schema_ = FeatureSchema(
            input_columns=active_inputs,
            feature_columns=tuple(output_columns),
            dropped_columns=tuple(dropped),
            native_categorical_columns=tuple(native),
            frequency_encoded_columns=tuple(frequency_encoded),
            category_levels=category_levels,
            frequency_maps=frequency_maps,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.schema_ is None:
            raise FeatureValidationError("FeaturePreprocessor must be fitted before transform.")
        missing_inputs = set(self.schema_.input_columns) - set(frame.columns)
        if missing_inputs:
            raise FeatureValidationError(f"Missing fitted input columns: {sorted(missing_inputs)}")

        result = pd.DataFrame(index=frame.index)
        frequency_set = set(self.schema_.frequency_encoded_columns)
        native_set = set(self.schema_.native_categorical_columns)
        dropped_set = set(self.schema_.dropped_columns)
        for column in self.schema_.input_columns:
            if column in dropped_set:
                continue
            if column in frequency_set:
                normalized = self._normalized_category(frame[column])
                result[f"{column}__frequency"] = (
                    normalized.map(self.schema_.frequency_maps[column]).fillna(0.0).astype("float32")
                )
            elif column in native_set:
                normalized = self._normalized_category(frame[column])
                known = set(self.schema_.category_levels[column]) - {UNKNOWN_CATEGORY}
                normalized = normalized.where(normalized.isin(known), UNKNOWN_CATEGORY)
                result[column] = pd.Categorical(
                    normalized,
                    categories=self.schema_.category_levels[column],
                )
            else:
                result[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        return result.loc[:, self.schema_.feature_columns]

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
