"""Fraud-specific ranking, threshold, calibration, and uncertainty metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricInterval:
    estimate: float
    lower: float
    upper: float


def threshold_for_review_rate(scores: np.ndarray, review_rate: float) -> float:
    """Return a fixed threshold selected from the historical score distribution."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a non-empty finite one-dimensional array.")
    if not 0.0 < review_rate < 1.0:
        raise ValueError("Review rate must be in (0, 1).")
    return float(np.quantile(scores, 1.0 - review_rate, method="higher"))


def thresholds_for_review_rates(
    scores: np.ndarray, review_rates: Iterable[float]
) -> dict[float, float]:
    return {float(rate): threshold_for_review_rate(scores, float(rate)) for rate in review_rates}


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute weighted absolute calibration error using equal-frequency bins."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if y.size == 0 or y.shape != probability.shape:
        raise ValueError("Labels and probabilities must be non-empty arrays with equal shape.")
    if bins < 2:
        raise ValueError("At least two calibration bins are required.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1].")

    unique_quantiles = np.unique(np.quantile(probability, np.linspace(0, 1, bins + 1)))
    if unique_quantiles.size < 2:
        return float(abs(y.mean() - probability.mean()))
    assignments = np.digitize(probability, unique_quantiles[1:-1], right=True)
    error = 0.0
    for index in range(unique_quantiles.size - 1):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(error)


def calibration_slope_intercept(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Fit outcome against log-odds to summarize calibration slope and intercept."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if np.unique(y).size < 2:
        return math_nan_pair()
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    model.fit(logit(clipped).reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def math_nan_pair() -> tuple[float, float]:
    return float("nan"), float("nan")


def binary_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    amounts: np.ndarray | None = None,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Evaluate ranking, decisions, calibration, and captured fraud value."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if y.size == 0 or y.shape != probability.shape:
        raise ValueError("Labels and probabilities must be non-empty arrays with equal shape.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")
    if np.any((probability < 0) | (probability > 1)) or not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite values in [0, 1].")

    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    has_both_classes = positives > 0 and negatives > 0
    slope, intercept = calibration_slope_intercept(y, probability)

    metrics: dict[str, float | int] = {
        "rows": int(y.size),
        "positives": positives,
        "fraud_prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, probability)) if positives else float("nan"),
        "roc_auc": float(roc_auc_score(y, probability)) if has_both_classes else float("nan"),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "false_positive_rate": float(fp / negatives) if negatives else float("nan"),
        "false_negative_rate": float(fn / positives) if positives else float("nan"),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "review_rate": float(predicted.mean()),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            y, probability, bins=calibration_bins
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "score_mean": float(probability.mean()),
        "score_p50": float(np.quantile(probability, 0.50)),
        "score_p95": float(np.quantile(probability, 0.95)),
        "threshold": float(threshold),
    }
    if amounts is not None:
        amount_values = np.asarray(amounts, dtype=float)
        if amount_values.shape != y.shape:
            raise ValueError("Amounts must have the same shape as labels.")
        fraud_amount = float(np.nansum(amount_values[y == 1]))
        captured = float(np.nansum(amount_values[(y == 1) & predicted]))
        metrics["fraud_amount"] = fraud_amount
        metrics["captured_fraud_amount"] = captured
        metrics["captured_fraud_amount_rate"] = (
            captured / fraud_amount if fraud_amount > 0 else float("nan")
        )
    return metrics


def review_budget_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: dict[float, float],
    amounts: np.ndarray | None = None,
) -> pd.DataFrame:
    records = []
    for review_rate, threshold in sorted(thresholds.items()):
        metrics = binary_classification_metrics(
            y_true,
            probabilities,
            threshold=threshold,
            amounts=amounts,
        )
        records.append({"target_review_rate": review_rate, **metrics})
    return pd.DataFrame(records)


def stratified_bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int = 500,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> MetricInterval:
    """Estimate a metric interval while retaining both outcome classes per sample."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if np.unique(y).size < 2:
        estimate = float(metric(y, probability))
        return MetricInterval(estimate=estimate, lower=float("nan"), upper=float("nan"))
    if iterations < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Iterations must be positive and confidence must be in (0, 1).")

    rng = np.random.default_rng(random_seed)
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        values[iteration] = metric(y[sampled], probability[sampled])
    alpha = 1.0 - confidence
    return MetricInterval(
        estimate=float(metric(y, probability)),
        lower=float(np.nanquantile(values, alpha / 2)),
        upper=float(np.nanquantile(values, 1 - alpha / 2)),
    )

