import json

import numpy as np
import pandas as pd

from fraud_monitor.diagnostics import (
    build_investigation_records,
    compare_shap_summaries,
    rank_potential_drivers,
    tree_shap_summary,
)


class _FakeBooster:
    def predict(self, features, *, pred_contrib):
        assert pred_contrib is True
        values = features.to_numpy(dtype=float)
        return np.column_stack([values, np.zeros(len(features))])


class _FakeModel:
    booster_ = _FakeBooster()


def test_tree_shap_summary_is_stratified_and_compares_importance() -> None:
    features = pd.DataFrame({"stable": np.ones(20), "moving": np.arange(20)})
    target = np.array([0] * 18 + [1] * 2)
    reference = tree_shap_summary(
        _FakeModel(), features, target=target, maximum_rows=10, random_seed=7
    )
    current = tree_shap_summary(
        _FakeModel(), features.assign(moving=features["moving"] * 2), target=target
    )

    comparison = compare_shap_summaries(reference, current)

    assert set(reference["sample_rows"]) == {10}
    assert reference["fraud_mean_shap"].notna().all()
    assert comparison.iloc[0]["feature"] == "moving"
    assert comparison.iloc[0]["importance_change"] > 0


def test_investigation_combines_alerts_shap_and_segment_evidence() -> None:
    drift = pd.DataFrame(
        [
            {
                "batch_id": "production_001",
                "feature": "TransactionAmt",
                "metric": "normalized_wasserstein",
                "value": 0.8,
                "warning_limit": 0.2,
                "critical_limit": 0.4,
                "severity": "critical",
            },
            {
                "batch_id": "production_001",
                "feature": "card4",
                "metric": "missingness_shift",
                "value": 0.3,
                "warning_limit": 0.1,
                "critical_limit": 0.2,
                "severity": "critical",
            },
        ]
    )
    shap = pd.DataFrame(
        [
            {
                "batch_id": "production_001",
                "feature": "TransactionAmt",
                "absolute_importance_change": 0.5,
            },
            {
                "batch_id": "production_001",
                "feature": "card4",
                "absolute_importance_change": 0.1,
            },
        ]
    )
    ranked = rank_potential_drivers(drift, shap)
    assert ranked.iloc[0]["feature"] == "TransactionAmt"

    batches = pd.DataFrame(
        [
            {
                "batch_id": "production_001",
                "stream": "production",
                "batch_number": 1,
                "label_status": "mature",
                "action": "investigate",
                "action_evidence": "Critical data or prediction drift",
                "drift_severity": "critical",
                "prediction_drift_severity": "warning",
                "performance_severity": "critical",
            }
        ]
    )
    performance = pd.DataFrame(
        [
            {
                "batch_id": "production_001",
                "metric": "recall",
                "status": "critical",
            }
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "batch_id": "production_001",
                "segment": "card4",
                "segment_value": "visa",
                "status": "reported",
                "false_negative": 4,
                "false_positive": 1,
                "fraud_prevalence": 0.2,
            }
        ]
    )

    investigations = build_investigation_records(batches, drift, performance, shap, segments)

    record = investigations.iloc[0]
    assert "data_quality_failure" in record["classification"]
    assert "confirmed_performance_degradation" in record["classification"]
    assert record["top_false_negative_segment"] == "card4=visa"
    assert json.loads(record["driver_evidence"])[0]["feature"] == "TransactionAmt"
