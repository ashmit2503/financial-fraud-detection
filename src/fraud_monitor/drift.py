"""Transparent feature and prediction drift metrics with empirical limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from fraud_monitor.features import MISSING_CATEGORY

Severity = Literal["healthy", "warning", "critical"]


def _finite_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def normalized_wasserstein(reference: pd.Series, current: pd.Series) -> float:
    """Wasserstein distance normalized by robust reference scale."""

    reference_values = _finite_numeric(reference)
    current_values = _finite_numeric(current)
    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")
    q25, q75 = np.quantile(reference_values, [0.25, 0.75])
    scale = max(float(q75 - q25), float(np.std(reference_values)), 1e-6)
    return float(wasserstein_distance(reference_values, current_values) / scale)


def _quantile_edges(reference_values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        value = float(reference_values[0])
        return np.array([-np.inf, value, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _smoothed_probabilities(counts: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = counts.astype(float) + epsilon
    return values / values.sum()


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    reference_values = _finite_numeric(reference)
    current_values = _finite_numeric(current)
    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")
    edges = _quantile_edges(reference_values, bins)
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    expected = _smoothed_probabilities(reference_counts)
    actual = _smoothed_probabilities(current_counts)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_jensen_shannon(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    reference_values = _finite_numeric(reference)
    current_values = _finite_numeric(current)
    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")
    edges = _quantile_edges(reference_values, bins)
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    return float(
        jensenshannon(
            _smoothed_probabilities(reference_counts),
            _smoothed_probabilities(current_counts),
            base=2,
        )
    )


def _normalized_categories(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype="string").fillna(MISSING_CATEGORY)


def categorical_jensen_shannon(reference: pd.Series, current: pd.Series) -> float:
    expected = _normalized_categories(reference)
    actual = _normalized_categories(current)
    levels = sorted(set(expected.unique()) | set(actual.unique()))
    expected_counts = expected.value_counts().reindex(levels, fill_value=0).to_numpy()
    actual_counts = actual.value_counts().reindex(levels, fill_value=0).to_numpy()
    return float(
        jensenshannon(
            _smoothed_probabilities(expected_counts),
            _smoothed_probabilities(actual_counts),
            base=2,
        )
    )


def unseen_category_rate(reference: pd.Series, current: pd.Series) -> float:
    expected_levels = set(_normalized_categories(reference).unique())
    actual = _normalized_categories(current)
    return float((~actual.isin(expected_levels)).mean())


def missingness_shift(reference: pd.Series, current: pd.Series) -> float:
    return float(abs(reference.isna().mean() - current.isna().mean()))


@dataclass(frozen=True)
class MetricLimit:
    warning: float
    critical: float


@dataclass(frozen=True)
class DriftRecord:
    feature: str
    feature_kind: str
    metric: str
    value: float
    warning_limit: float
    critical_limit: float
    severity: Severity


@dataclass(frozen=True)
class DriftSummary:
    warning_features: int
    critical_features: int
    warning_feature_limit: int
    critical_feature_limit: int
    prediction_severity: Severity
    global_severity: Severity


@dataclass
class ReferenceDriftMonitor:
    """Reference data, bootstrapped limits, and comparison behavior."""

    feature_columns: tuple[str, ...]
    categorical_columns: frozenset[str]
    warning_quantile: float
    critical_quantile: float
    reference_features: pd.DataFrame
    reference_scores: np.ndarray
    feature_limits: dict[str, dict[str, MetricLimit]] = field(default_factory=dict)
    prediction_limits: dict[str, MetricLimit] = field(default_factory=dict)
    warning_feature_limit: int = 0
    critical_feature_limit: int = 0

    @staticmethod
    def _feature_metrics(
        reference: pd.Series,
        current: pd.Series,
        *,
        categorical: bool,
    ) -> dict[str, float]:
        metrics = {"missingness_shift": missingness_shift(reference, current)}
        if categorical:
            metrics.update(
                {
                    "jensen_shannon": categorical_jensen_shannon(reference, current),
                    "unseen_category_rate": unseen_category_rate(reference, current),
                }
            )
        else:
            metrics.update(
                {
                    "normalized_wasserstein": normalized_wasserstein(reference, current),
                    "psi": population_stability_index(reference, current),
                }
            )
        return metrics

    @staticmethod
    def _severity(value: float, limit: MetricLimit) -> Severity:
        if not np.isfinite(value):
            return "critical"
        if value > limit.critical:
            return "critical"
        if value > limit.warning:
            return "warning"
        return "healthy"

    @classmethod
    def fit(
        cls,
        reference_features: pd.DataFrame,
        reference_scores: np.ndarray,
        transaction_time: pd.Series,
        *,
        feature_columns: list[str],
        categorical_columns: set[str],
        warning_quantile: float = 0.95,
        critical_quantile: float = 0.99,
        bootstrap_iterations: int = 500,
        window_days: int = 7,
        random_seed: int = 42,
    ) -> ReferenceDriftMonitor:
        """Fit feature-specific limits from one-day block bootstrap windows."""

        missing = sorted(set(feature_columns) - set(reference_features.columns))
        if missing:
            raise ValueError(f"Reference is missing monitored features: {missing}")
        scores = np.asarray(reference_scores, dtype=float)
        if len(reference_features) != len(scores) or len(scores) != len(transaction_time):
            raise ValueError("Reference features, scores, and transaction time must align.")
        if not 0 < warning_quantile < critical_quantile < 1:
            raise ValueError("Control quantiles must satisfy 0 < warning < critical < 1.")
        if bootstrap_iterations < 1 or window_days < 1:
            raise ValueError("Bootstrap iterations and window days must be positive.")

        selected = reference_features.loc[:, feature_columns].reset_index(drop=True).copy()
        time_values = pd.to_numeric(transaction_time, errors="raise").to_numpy(dtype=np.int64)
        day = ((time_values - time_values.min()) // 86_400).astype(int)
        day_positions = [np.flatnonzero(day == value) for value in np.unique(day)]
        if not day_positions:
            raise ValueError("Reference must contain at least one elapsed day.")

        rng = np.random.default_rng(random_seed)
        feature_samples: dict[str, dict[str, list[float]]] = {
            feature: {} for feature in feature_columns
        }
        prediction_samples = {"score_jensen_shannon": [], "score_mean_shift": []}
        sampled_feature_metrics: list[dict[str, dict[str, float]]] = []
        for _ in range(bootstrap_iterations):
            chosen_days = rng.choice(len(day_positions), size=window_days, replace=True)
            positions = np.concatenate([day_positions[int(index)] for index in chosen_days])
            current = selected.iloc[positions]
            iteration_metrics: dict[str, dict[str, float]] = {}
            for feature in feature_columns:
                metrics = cls._feature_metrics(
                    selected[feature],
                    current[feature],
                    categorical=feature in categorical_columns,
                )
                iteration_metrics[feature] = metrics
                for metric, value in metrics.items():
                    feature_samples[feature].setdefault(metric, []).append(value)
            sampled_feature_metrics.append(iteration_metrics)
            sampled_scores = scores[positions]
            prediction_samples["score_jensen_shannon"].append(
                numeric_jensen_shannon(scores, sampled_scores)
            )
            prediction_samples["score_mean_shift"].append(
                abs(float(sampled_scores.mean() - scores.mean()))
            )

        feature_limits = {
            feature: {
                metric: MetricLimit(
                    warning=float(np.nanquantile(values, warning_quantile)),
                    critical=float(np.nanquantile(values, critical_quantile)),
                )
                for metric, values in metrics.items()
            }
            for feature, metrics in feature_samples.items()
        }
        prediction_limits = {
            metric: MetricLimit(
                warning=float(np.nanquantile(values, warning_quantile)),
                critical=float(np.nanquantile(values, critical_quantile)),
            )
            for metric, values in prediction_samples.items()
        }

        warning_counts: list[int] = []
        critical_counts: list[int] = []
        for iteration_metrics in sampled_feature_metrics:
            feature_severities: list[Severity] = []
            for feature, metrics in iteration_metrics.items():
                severities = [
                    cls._severity(value, feature_limits[feature][metric])
                    for metric, value in metrics.items()
                ]
                feature_severities.append(
                    "critical"
                    if "critical" in severities
                    else "warning"
                    if "warning" in severities
                    else "healthy"
                )
            warning_counts.append(
                sum(value in {"warning", "critical"} for value in feature_severities)
            )
            critical_counts.append(sum(value == "critical" for value in feature_severities))

        return cls(
            feature_columns=tuple(feature_columns),
            categorical_columns=frozenset(categorical_columns),
            warning_quantile=warning_quantile,
            critical_quantile=critical_quantile,
            reference_features=selected,
            reference_scores=scores,
            feature_limits=feature_limits,
            prediction_limits=prediction_limits,
            warning_feature_limit=int(
                np.quantile(warning_counts, warning_quantile, method="higher")
            ),
            critical_feature_limit=int(
                np.quantile(critical_counts, critical_quantile, method="higher")
            ),
        )

    def compare(
        self,
        current_features: pd.DataFrame,
        current_scores: np.ndarray,
    ) -> tuple[list[DriftRecord], DriftSummary]:
        """Compare an incoming batch with the fitted reference."""

        scores = np.asarray(current_scores, dtype=float)
        if len(current_features) != len(scores):
            raise ValueError("Current features and scores must align.")
        records: list[DriftRecord] = []
        feature_severity: dict[str, Severity] = {}
        for feature in self.feature_columns:
            if feature not in current_features:
                feature_severity[feature] = "critical"
                records.append(
                    DriftRecord(
                        feature=feature,
                        feature_kind=(
                            "categorical" if feature in self.categorical_columns else "numeric"
                        ),
                        metric="schema_missing",
                        value=1.0,
                        warning_limit=0.0,
                        critical_limit=0.0,
                        severity="critical",
                    )
                )
                continue
            metrics = self._feature_metrics(
                self.reference_features[feature],
                current_features[feature],
                categorical=feature in self.categorical_columns,
            )
            severities: list[Severity] = []
            for metric, value in metrics.items():
                limit = self.feature_limits[feature][metric]
                severity = self._severity(value, limit)
                severities.append(severity)
                records.append(
                    DriftRecord(
                        feature=feature,
                        feature_kind=(
                            "categorical" if feature in self.categorical_columns else "numeric"
                        ),
                        metric=metric,
                        value=float(value),
                        warning_limit=limit.warning,
                        critical_limit=limit.critical,
                        severity=severity,
                    )
                )
            feature_severity[feature] = (
                "critical"
                if "critical" in severities
                else "warning"
                if "warning" in severities
                else "healthy"
            )

        prediction_values = {
            "score_jensen_shannon": numeric_jensen_shannon(self.reference_scores, scores),
            "score_mean_shift": abs(float(scores.mean() - self.reference_scores.mean())),
        }
        prediction_severities = [
            self._severity(value, self.prediction_limits[metric])
            for metric, value in prediction_values.items()
        ]
        prediction_severity: Severity = (
            "critical"
            if "critical" in prediction_severities
            else "warning"
            if "warning" in prediction_severities
            else "healthy"
        )
        warning_features = sum(
            severity in {"warning", "critical"} for severity in feature_severity.values()
        )
        critical_features = sum(severity == "critical" for severity in feature_severity.values())
        global_severity: Severity = "healthy"
        if critical_features > self.critical_feature_limit or prediction_severity == "critical":
            global_severity = "critical"
        elif warning_features > self.warning_feature_limit or prediction_severity == "warning":
            global_severity = "warning"

        return records, DriftSummary(
            warning_features=warning_features,
            critical_features=critical_features,
            warning_feature_limit=self.warning_feature_limit,
            critical_feature_limit=self.critical_feature_limit,
            prediction_severity=prediction_severity,
            global_severity=global_severity,
        )
