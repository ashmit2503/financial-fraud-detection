"""Simple, interpretable baselines for temporal fraud experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_CATEGORICAL_FEATURES = (
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
)
BASELINE_NUMERIC_FEATURES = (
    "TransactionAmt",
    "amount_log1p",
    "amount_fraction",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "identity_available",
    "missing_count_identity",
    "missing_count_transaction",
    "C1",
    "D1",
    "V1",
)


@dataclass
class BaselineModels:
    dummy: DummyClassifier
    logistic: Pipeline
    feature_columns: tuple[str, ...]

    def predict_probabilities(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        features = frame.loc[:, self.feature_columns]
        return {
            "dummy": self.dummy.predict_proba(features)[:, 1],
            "logistic": self.logistic.predict_proba(features)[:, 1],
        }


def fit_baselines(frame: pd.DataFrame, target: np.ndarray) -> BaselineModels:
    """Fit a prior baseline and a compact class-weighted logistic model."""

    categorical = [column for column in BASELINE_CATEGORICAL_FEATURES if column in frame]
    numeric = [column for column in BASELINE_NUMERIC_FEATURES if column in frame]
    feature_columns = tuple([*categorical, *numeric])
    if not feature_columns:
        raise ValueError("No baseline features are available.")

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    transform = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )
    logistic = Pipeline(
        [
            ("transform", transform),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    dummy = DummyClassifier(strategy="prior")
    features = frame.loc[:, feature_columns]
    dummy.fit(features, target)
    logistic.fit(features, target)
    return BaselineModels(dummy=dummy, logistic=logistic, feature_columns=feature_columns)

