import numpy as np
import pandas as pd

from fraud_monitor.drift import (
    ReferenceDriftMonitor,
    categorical_jensen_shannon,
    normalized_wasserstein,
    population_stability_index,
    unseen_category_rate,
)


def test_numeric_drift_metrics_distinguish_shifted_distribution() -> None:
    reference = pd.Series(np.linspace(0, 1, 200))
    stable = pd.Series(np.linspace(0, 1, 200))
    shifted = pd.Series(np.linspace(3, 4, 200))

    assert normalized_wasserstein(reference, stable) == 0.0
    assert population_stability_index(reference, stable) == 0.0
    assert normalized_wasserstein(reference, shifted) > 1.0
    assert population_stability_index(reference, shifted) > 1.0


def test_categorical_drift_tracks_new_levels() -> None:
    reference = pd.Series(["a", "a", "b", None])
    current = pd.Series(["c", "c", "a", None])

    assert categorical_jensen_shannon(reference, current) > 0
    assert unseen_category_rate(reference, current) == 0.5


def test_bootstrapped_monitor_is_reproducible_and_flags_shift() -> None:
    rng = np.random.default_rng(9)
    rows = 280
    reference = pd.DataFrame(
        {
            "amount": rng.normal(100, 10, rows),
            "product": rng.choice(["W", "C"], rows, p=[0.8, 0.2]),
        }
    )
    scores = np.clip(rng.normal(0.1, 0.03, rows), 0, 1)
    transaction_time = pd.Series(86_400 + np.arange(rows) * 21_600)
    monitor = ReferenceDriftMonitor.fit(
        reference,
        scores,
        transaction_time,
        feature_columns=["amount", "product"],
        categorical_columns={"product"},
        bootstrap_iterations=40,
        random_seed=3,
    )
    _, stable_summary = monitor.compare(reference, scores)
    shifted = reference.copy()
    shifted["amount"] += 100
    shifted["product"] = "NEW"
    shifted_scores = np.clip(scores + 0.5, 0, 1)

    records, summary = monitor.compare(shifted, shifted_scores)

    assert stable_summary.global_severity == "healthy"
    assert summary.global_severity == "critical"
    assert summary.prediction_severity == "critical"
    assert any(record.feature == "amount" and record.severity == "critical" for record in records)
    assert any(record.metric == "unseen_category_rate" for record in records)


def test_missing_monitored_column_is_a_critical_schema_failure() -> None:
    reference = pd.DataFrame({"amount": np.arange(30, dtype=float)})
    scores = np.linspace(0.01, 0.2, 30)
    monitor = ReferenceDriftMonitor.fit(
        reference,
        scores,
        pd.Series(np.arange(30) * 86_400),
        feature_columns=["amount"],
        categorical_columns=set(),
        bootstrap_iterations=10,
    )

    records, summary = monitor.compare(pd.DataFrame(index=range(5)), np.full(5, 0.1))

    assert records[0].metric == "schema_missing"
    assert records[0].severity == "critical"
    assert summary.global_severity == "critical"
