import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud_monitor.evaluation import (
    binary_classification_metrics,
    expected_calibration_error,
    review_budget_table,
    stratified_bootstrap_interval,
    threshold_for_review_rate,
)


def test_perfect_ranking_has_perfect_pr_auc_and_recall() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.1, 0.8, 0.9])

    metrics = binary_classification_metrics(y, scores, threshold=0.5)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["false_positive"] == 0


def test_review_threshold_is_derived_from_reference_scores() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    threshold = threshold_for_review_rate(scores, 0.02)

    assert (scores >= threshold).mean() <= 0.03
    assert threshold >= 0.98


def test_budget_table_reports_captured_fraud_amount() -> None:
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.4, 0.2])
    amounts = np.array([10.0, 50.0, 100.0, 20.0])

    table = review_budget_table(y, scores, thresholds={0.25: 0.8}, amounts=amounts)

    assert table.loc[0, "captured_fraud_amount"] == pytest.approx(50.0)
    assert table.loc[0, "captured_fraud_amount_rate"] == pytest.approx(1 / 3)


def test_no_positive_batch_returns_nan_ranking_metrics_without_crashing() -> None:
    metrics = binary_classification_metrics(
        np.zeros(5, dtype=int), np.linspace(0.1, 0.5, 5), threshold=0.4
    )

    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["false_negative_rate"])
    assert metrics["recall"] == 0.0


def test_equal_predictions_have_calibration_error_against_prevalence() -> None:
    y = np.array([0, 0, 0, 1])
    scores = np.full(4, 0.5)

    assert expected_calibration_error(y, scores) == pytest.approx(0.25)


def test_stratified_bootstrap_is_reproducible() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    first = stratified_bootstrap_interval(
        y, scores, average_precision_score, iterations=30, random_seed=7
    )
    second = stratified_bootstrap_interval(
        y, scores, average_precision_score, iterations=30, random_seed=7
    )

    assert first == second
    assert first.estimate == pytest.approx(1.0)

