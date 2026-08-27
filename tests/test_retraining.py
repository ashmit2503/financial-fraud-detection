import numpy as np
from sklearn.metrics import average_precision_score

from fraud_monitor.retraining import paired_stratified_difference


def test_paired_bootstrap_detects_reliable_challenger_improvement() -> None:
    target = np.array([0] * 80 + [1] * 20)
    champion = np.linspace(0.9, 0.1, 100)
    challenger = np.concatenate([np.linspace(0.1, 0.3, 80), np.linspace(0.7, 0.9, 20)])

    interval = paired_stratified_difference(
        target,
        champion,
        challenger,
        average_precision_score,
        iterations=100,
        random_seed=7,
    )

    assert interval.estimate > 0
    assert interval.lower > 0
