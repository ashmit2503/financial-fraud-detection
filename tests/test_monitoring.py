import numpy as np
import pandas as pd

from fraud_monitor.monitoring import (
    PerformanceMonitor,
    SegmentProfiler,
    compute_segment_metrics,
    derive_monitoring_actions,
)


def _reference_performance_data(
    rows: int = 280,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    rng = np.random.default_rng(11)
    target = rng.binomial(1, 0.2, rows)
    scores = np.clip(0.05 + 0.7 * target + rng.normal(0, 0.08, rows), 0.001, 0.999)
    amounts = rng.lognormal(4, 0.5, rows)
    time = pd.Series(86_400 + np.arange(rows) * 21_600)
    return target, scores, amounts, time


def test_performance_monitor_marks_unavailable_labels_explicitly() -> None:
    target, scores, amounts, time = _reference_performance_data()
    monitor = PerformanceMonitor.fit(
        target,
        scores,
        amounts,
        time,
        threshold=0.5,
        bootstrap_iterations=30,
    )

    records, summary = monitor.compare(None, scores[:20], amounts[:20], labels_available=False)

    assert summary.status == "unavailable"
    assert all(record.status == "unavailable" for record in records)


def test_performance_monitor_flags_degraded_mature_batch() -> None:
    target, scores, amounts, time = _reference_performance_data()
    monitor = PerformanceMonitor.fit(
        target,
        scores,
        amounts,
        time,
        threshold=0.5,
        bootstrap_iterations=40,
        random_seed=4,
    )
    degraded_scores = 1 - scores

    _, summary = monitor.compare(target, degraded_scores, amounts, labels_available=True)

    assert summary.status == "critical"
    assert summary.primary_critical


def test_segment_metrics_pool_previous_batch_before_suppressing() -> None:
    reference = pd.DataFrame(
        {
            "ProductCD": ["W", "C", "W", "C"],
            "P_emaildomain": ["a.com", "b.com", "a.com", "b.com"],
            "addr1": [100, 200, 100, 200],
        }
    )
    profiler = SegmentProfiler.fit(reference)
    current = profiler.add_segments(
        pd.DataFrame(
            {
                "ProductCD": ["W", "W", "W", "W"],
                "P_emaildomain": ["a.com"] * 4,
                "addr1": [100] * 4,
                "isFraud": [0, 0, 1, 1],
                "fraud_probability": [0.1, 0.2, 0.8, 0.9],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            }
        )
    )
    previous = current.copy()

    metrics = compute_segment_metrics(
        current,
        previous=previous,
        threshold=0.5,
        minimum_positive=3,
        minimum_negative=3,
    )

    product = metrics[(metrics["segment"] == "ProductCD") & (metrics["segment_value"] == "W")]
    assert product.iloc[0]["status"] == "reported"
    assert product.iloc[0]["window_batches"] == 2
    assert product.iloc[0]["recall"] == 1.0


def test_action_policy_requires_persistence_and_two_primary_breaches() -> None:
    batches = pd.DataFrame(
        {
            "batch_number": [1, 2, 3, 4],
            "drift_severity": ["warning", "warning", "healthy", "healthy"],
            "performance_severity": ["healthy", "healthy", "critical", "critical"],
            "primary_performance_critical": [False, False, True, True],
            "label_status": ["mature"] * 4,
        }
    )

    actions = derive_monitoring_actions(batches)

    assert actions["action"].tolist() == [
        "continue_monitoring",
        "investigate",
        "investigate",
        "retrain_evaluation_required",
    ]


def test_action_policy_investigates_stale_labels_without_performance_claim() -> None:
    batch = pd.DataFrame(
        {
            "batch_number": [5],
            "drift_severity": ["healthy"],
            "performance_severity": ["unavailable"],
            "primary_performance_critical": [False],
            "label_status": ["stale"],
        }
    )

    result = derive_monitoring_actions(batch)

    assert result.iloc[0]["action"] == "investigate"
    assert "stale" in result.iloc[0]["action_evidence"].lower()
