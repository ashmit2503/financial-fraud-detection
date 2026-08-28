"""Performance controls, segment metrics, and monitoring action policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fraud_monitor.evaluation import binary_classification_metrics
from fraud_monitor.features import MISSING_CATEGORY

MetricStatus = Literal["healthy", "warning", "critical", "unavailable"]

LOWER_IS_WORSE = ("pr_auc", "precision", "recall")
HIGHER_IS_WORSE = (
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
    "expected_calibration_error",
)
PRIMARY_PERFORMANCE_METRICS = {"pr_auc", "recall"}


@dataclass(frozen=True)
class PerformanceLimit:
    direction: Literal["lower", "higher"]
    warning: float
    critical: float


@dataclass(frozen=True)
class PerformanceRecord:
    metric: str
    value: float
    reference_value: float
    warning_limit: float
    critical_limit: float
    direction: str
    status: MetricStatus


@dataclass(frozen=True)
class PerformanceSummary:
    status: MetricStatus
    primary_critical: bool
    metrics: dict[str, float | int]


@dataclass
class PerformanceMonitor:
    threshold: float
    reference_metrics: dict[str, float | int]
    limits: dict[str, PerformanceLimit]
    calibration_bins: int = 10

    @staticmethod
    def _status(value: float, limit: PerformanceLimit) -> MetricStatus:
        if not np.isfinite(value):
            return "unavailable"
        if limit.direction == "lower":
            if value < limit.critical:
                return "critical"
            if value < limit.warning:
                return "warning"
        else:
            if value > limit.critical:
                return "critical"
            if value > limit.warning:
                return "warning"
        return "healthy"

    @classmethod
    def fit(
        cls,
        target: np.ndarray,
        probabilities: np.ndarray,
        amounts: np.ndarray,
        transaction_time: pd.Series,
        *,
        threshold: float,
        warning_quantile: float = 0.95,
        critical_quantile: float = 0.99,
        bootstrap_iterations: int = 500,
        window_days: int = 7,
        random_seed: int = 42,
        calibration_bins: int = 10,
    ) -> PerformanceMonitor:
        """Fit one-sided empirical limits from reference day-block resamples."""

        y = np.asarray(target, dtype=int)
        scores = np.asarray(probabilities, dtype=float)
        amount_values = np.asarray(amounts, dtype=float)
        if not (len(y) == len(scores) == len(amount_values) == len(transaction_time)):
            raise ValueError("Reference labels, scores, amounts, and time must align.")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Performance reference must contain both target classes.")
        if not 0 < warning_quantile < critical_quantile < 1:
            raise ValueError("Control quantiles must satisfy 0 < warning < critical < 1.")

        reference = binary_classification_metrics(
            y,
            scores,
            threshold=threshold,
            amounts=amount_values,
            calibration_bins=calibration_bins,
        )
        times = pd.to_numeric(transaction_time, errors="raise").to_numpy(dtype=np.int64)
        day = ((times - times.min()) // 86_400).astype(int)
        day_positions = [np.flatnonzero(day == value) for value in np.unique(day)]
        rng = np.random.default_rng(random_seed)
        metric_samples: dict[str, list[float]] = {
            metric: [] for metric in (*LOWER_IS_WORSE, *HIGHER_IS_WORSE)
        }
        metric_samples["fraud_prevalence_shift"] = []

        for _ in range(bootstrap_iterations):
            selected_days = rng.choice(len(day_positions), size=window_days, replace=True)
            positions = np.concatenate([day_positions[int(index)] for index in selected_days])
            sampled_y = y[positions]
            if np.unique(sampled_y).size < 2:
                continue
            sampled = binary_classification_metrics(
                sampled_y,
                scores[positions],
                threshold=threshold,
                amounts=amount_values[positions],
                calibration_bins=calibration_bins,
            )
            for metric in (*LOWER_IS_WORSE, *HIGHER_IS_WORSE):
                metric_samples[metric].append(float(sampled[metric]))
            metric_samples["fraud_prevalence_shift"].append(
                abs(float(sampled["fraud_prevalence"]) - float(reference["fraud_prevalence"]))
            )
        if any(not values for values in metric_samples.values()):
            raise ValueError("Bootstrap windows did not contain enough target variation.")

        limits: dict[str, PerformanceLimit] = {}
        for metric in LOWER_IS_WORSE:
            values = metric_samples[metric]
            limits[metric] = PerformanceLimit(
                direction="lower",
                warning=float(np.nanquantile(values, 1 - warning_quantile)),
                critical=float(np.nanquantile(values, 1 - critical_quantile)),
            )
        for metric in (*HIGHER_IS_WORSE, "fraud_prevalence_shift"):
            values = metric_samples[metric]
            limits[metric] = PerformanceLimit(
                direction="higher",
                warning=float(np.nanquantile(values, warning_quantile)),
                critical=float(np.nanquantile(values, critical_quantile)),
            )
        return cls(
            threshold=threshold,
            reference_metrics=reference,
            limits=limits,
            calibration_bins=calibration_bins,
        )

    def compare(
        self,
        target: np.ndarray | None,
        probabilities: np.ndarray,
        amounts: np.ndarray,
        *,
        labels_available: bool,
    ) -> tuple[list[PerformanceRecord], PerformanceSummary]:
        """Score a mature batch or return an explicit unavailable state."""

        if not labels_available or target is None:
            records = [
                PerformanceRecord(
                    metric=metric,
                    value=float("nan"),
                    reference_value=float(self.reference_metrics.get(metric, float("nan"))),
                    warning_limit=limit.warning,
                    critical_limit=limit.critical,
                    direction=limit.direction,
                    status="unavailable",
                )
                for metric, limit in self.limits.items()
            ]
            return records, PerformanceSummary(
                status="unavailable",
                primary_critical=False,
                metrics={},
            )

        y = np.asarray(target, dtype=int)
        if np.unique(y).size < 2:
            return self.compare(None, probabilities, amounts, labels_available=False)
        metrics = binary_classification_metrics(
            y,
            probabilities,
            threshold=self.threshold,
            amounts=amounts,
            calibration_bins=self.calibration_bins,
        )
        monitored_values = {
            metric: float(metrics[metric]) for metric in (*LOWER_IS_WORSE, *HIGHER_IS_WORSE)
        }
        monitored_values["fraud_prevalence_shift"] = abs(
            float(metrics["fraud_prevalence"]) - float(self.reference_metrics["fraud_prevalence"])
        )
        records = []
        for metric, value in monitored_values.items():
            limit = self.limits[metric]
            reference_value = (
                0.0 if metric == "fraud_prevalence_shift" else float(self.reference_metrics[metric])
            )
            records.append(
                PerformanceRecord(
                    metric=metric,
                    value=value,
                    reference_value=reference_value,
                    warning_limit=limit.warning,
                    critical_limit=limit.critical,
                    direction=limit.direction,
                    status=self._status(value, limit),
                )
            )
        statuses = [record.status for record in records]
        overall: MetricStatus = (
            "critical"
            if "critical" in statuses
            else "warning"
            if "warning" in statuses
            else "healthy"
        )
        return records, PerformanceSummary(
            status=overall,
            primary_critical=any(
                record.metric in PRIMARY_PERFORMANCE_METRICS and record.status == "critical"
                for record in records
            ),
            metrics=metrics,
        )


@dataclass(frozen=True)
class SegmentProfiler:
    top_email_domains: tuple[str, ...]
    top_addresses: tuple[str, ...]

    @classmethod
    def fit(cls, reference: pd.DataFrame, *, top_values: int = 10) -> SegmentProfiler:
        def most_common(column: str) -> tuple[str, ...]:
            if column not in reference:
                return ()
            values = reference[column].astype("string").fillna(MISSING_CATEGORY)
            return tuple(str(value) for value in values.value_counts().head(top_values).index)

        return cls(
            top_email_domains=most_common("P_emaildomain"),
            top_addresses=most_common("addr1"),
        )

    @staticmethod
    def _grouped(values: pd.Series, top: tuple[str, ...] | None = None) -> pd.Series:
        normalized = values.astype("string").fillna(MISSING_CATEGORY)
        if top is not None:
            normalized = normalized.where(normalized.isin(top), "__OTHER__")
        return normalized

    def add_segments(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        definitions: dict[str, tuple[str, ...] | None] = {
            "ProductCD": None,
            "card4": None,
            "card6": None,
            "DeviceType": None,
            "identity_available": None,
            "P_emaildomain": self.top_email_domains,
            "addr1": self.top_addresses,
        }
        for column, top in definitions.items():
            if column in result:
                result[f"segment__{column}"] = self._grouped(result[column], top)
        return result


def compute_segment_metrics(
    current: pd.DataFrame,
    *,
    threshold: float,
    minimum_positive: int,
    minimum_negative: int,
    previous: pd.DataFrame | None = None,
    target_column: str = "isFraud",
    score_column: str = "fraud_probability",
) -> pd.DataFrame:
    """Compute segment errors, falling back to a two-batch pool when support is thin."""

    segment_columns = [column for column in current if column.startswith("segment__")]
    records: list[dict[str, object]] = []
    for segment_column in segment_columns:
        segment_name = segment_column.removeprefix("segment__")
        for value in sorted(current[segment_column].dropna().unique()):
            current_group = current[current[segment_column] == value]
            evaluation_group = current_group
            window_batches = 1
            positives = int(current_group[target_column].sum())
            negatives = int((current_group[target_column] == 0).sum())
            if (
                positives < minimum_positive or negatives < minimum_negative
            ) and previous is not None:
                previous_group = previous[previous[segment_column] == value]
                evaluation_group = pd.concat([previous_group, current_group], ignore_index=True)
                window_batches = 2
                positives = int(evaluation_group[target_column].sum())
                negatives = int((evaluation_group[target_column] == 0).sum())

            base_record: dict[str, object] = {
                "segment": segment_name,
                "segment_value": str(value),
                "window_batches": window_batches,
                "rows": len(evaluation_group),
                "positives": positives,
                "negatives": negatives,
            }
            if positives < minimum_positive or negatives < minimum_negative:
                records.append({**base_record, "status": "suppressed"})
                continue

            metrics = binary_classification_metrics(
                evaluation_group[target_column].to_numpy(dtype=int),
                evaluation_group[score_column].to_numpy(dtype=float),
                threshold=threshold,
                amounts=evaluation_group["TransactionAmt"].to_numpy(dtype=float),
            )
            records.append(
                {
                    **base_record,
                    "status": "reported",
                    "fraud_prevalence": metrics["fraud_prevalence"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                    "captured_fraud_amount_rate": metrics["captured_fraud_amount_rate"],
                }
            )
    return pd.DataFrame(records)


def derive_monitoring_actions(batch_metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply hysteresis and mature-label rules to chronological production batches."""

    ordered = batch_metrics.sort_values("batch_number").copy()
    actions: list[str] = []
    evidence: list[str] = []
    previous_warning = False
    previous_primary_critical = False
    for row in ordered.itertuples(index=False):
        drift = row.drift_severity
        performance = row.performance_severity
        primary_critical = bool(row.primary_performance_critical)
        current_warning = drift == "warning" or performance == "warning"
        reasons: list[str] = []
        action = "continue_monitoring"
        if row.label_status == "stale":
            action = "investigate"
            reasons.append("Labels are stale beyond the configured maturity delay")
        elif primary_critical and previous_primary_critical:
            action = "retrain_evaluation_required"
            reasons.append("Primary performance guardrail breached in two mature batches")
        elif drift == "critical" or performance == "critical":
            action = "investigate"
            if drift == "critical":
                reasons.append("Critical data or prediction drift")
            if performance == "critical":
                reasons.append("Critical mature-label performance breach")
        elif current_warning and previous_warning:
            action = "investigate"
            reasons.append("Warning persisted for two consecutive batches")
        else:
            reasons.append("No sustained or critical guardrail breach")

        actions.append(action)
        evidence.append("; ".join(reasons))
        previous_warning = current_warning
        if row.label_status == "mature":
            previous_primary_critical = primary_critical
    ordered["action"] = actions
    ordered["action_evidence"] = evidence
    return ordered
