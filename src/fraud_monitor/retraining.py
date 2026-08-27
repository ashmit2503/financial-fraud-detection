"""Leakage-safe manual challenger training and paired evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, recall_score

from fraud_monitor.config import ProjectConfig
from fraud_monitor.evaluation import (
    binary_classification_metrics,
    paired_stratified_difference,
    threshold_for_review_rate,
)
from fraud_monitor.features import CausalFeatureBuilder, FeaturePreprocessor
from fraud_monitor.modeling import (
    _fit_lgbm,
    fit_probability_calibrators,
    load_model_bundle,
)
from fraud_monitor.monitoring import SegmentProfiler
from fraud_monitor.splits import PERIOD_COLUMN, PRODUCTION_BATCH_COLUMN


@dataclass(frozen=True)
class RetrainingEvaluationResult:
    summary_path: Path
    segment_comparison_path: Path
    retrain_recommended: bool
    training_rows: int
    calibration_batches: tuple[int, ...]
    evaluation_batches: tuple[int, ...]


def _select_windows(
    production: pd.DataFrame,
    *,
    maximum_batch: int,
    label_delay_batches: int,
    target_column: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mature = sorted(
        int(value)
        for value in production[PRODUCTION_BATCH_COLUMN].dropna().unique()
        if int(value) + label_delay_batches <= maximum_batch
    )
    if len(mature) < 4:
        raise ValueError("Retraining evaluation requires at least four mature production batches.")
    evaluation = tuple(mature[-2:])
    candidates = mature[:-2]
    calibration: list[int] = []
    for batch in reversed(candidates):
        calibration.insert(0, batch)
        labels = production[production[PRODUCTION_BATCH_COLUMN].isin(calibration)][target_column]
        if labels.nunique() == 2:
            break
    if not calibration:
        raise ValueError("Mature history cannot provide a challenger calibration window.")
    calibration_labels = production[production[PRODUCTION_BATCH_COLUMN].isin(calibration)][
        target_column
    ]
    if calibration_labels.nunique() < 2:
        raise ValueError("Challenger calibration history must contain both target classes.")
    return tuple(calibration), evaluation


def _score_with_builder(
    model,
    calibrator,
    preprocessor,
    builder: CausalFeatureBuilder,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    engineered = builder.transform_batch(frame)
    features = preprocessor.transform(engineered)
    raw_scores = model.predict_proba(features)[:, 1]
    return engineered, calibrator.predict(raw_scores)


def _segment_recall_comparison(
    profiler: SegmentProfiler,
    evaluation: pd.DataFrame,
    target: np.ndarray,
    champion_scores: np.ndarray,
    challenger_scores: np.ndarray,
    *,
    champion_threshold: float,
    challenger_threshold: float,
    minimum_positive: int,
    minimum_negative: int,
    iterations: int,
    random_seed: int,
) -> pd.DataFrame:
    segmented = profiler.add_segments(evaluation).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for column in [name for name in segmented if name.startswith("segment__")]:
        for value in sorted(segmented[column].dropna().unique()):
            mask = (segmented[column] == value).to_numpy()
            segment_target = target[mask]
            positives = int(segment_target.sum())
            negatives = int((segment_target == 0).sum())
            if positives < minimum_positive or negatives < minimum_negative:
                continue

            def champion_recall(y: np.ndarray, scores: np.ndarray) -> float:
                return float(recall_score(y, scores >= champion_threshold, zero_division=0))

            def challenger_recall(y: np.ndarray, scores: np.ndarray) -> float:
                return float(recall_score(y, scores >= challenger_threshold, zero_division=0))

            champion_metric = champion_recall(segment_target, champion_scores[mask])
            challenger_metric = challenger_recall(segment_target, challenger_scores[mask])
            rng = np.random.default_rng(random_seed)
            class_positions = [np.flatnonzero(segment_target == item) for item in (0, 1)]
            differences = []
            for _ in range(iterations):
                sampled = np.concatenate(
                    [rng.choice(rows, size=len(rows), replace=True) for rows in class_positions]
                )
                differences.append(
                    challenger_recall(segment_target[sampled], challenger_scores[mask][sampled])
                    - champion_recall(segment_target[sampled], champion_scores[mask][sampled])
                )
            records.append(
                {
                    "segment": column.removeprefix("segment__"),
                    "segment_value": str(value),
                    "rows": int(mask.sum()),
                    "positives": positives,
                    "negatives": negatives,
                    "champion_recall": champion_metric,
                    "challenger_recall": challenger_metric,
                    "recall_difference": challenger_metric - champion_metric,
                    "recall_difference_lower": float(np.quantile(differences, 0.025)),
                    "recall_difference_upper": float(np.quantile(differences, 0.975)),
                    "reliable_regression": float(np.quantile(differences, 0.975)) < 0,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "segment",
            "segment_value",
            "rows",
            "positives",
            "negatives",
            "champion_recall",
            "challenger_recall",
            "recall_difference",
            "recall_difference_lower",
            "recall_difference_upper",
            "reliable_regression",
        ],
    )


def run_retraining_evaluation(
    config: ProjectConfig,
    *,
    bundle_path: Path,
    processed_dir: Path | None = None,
    monitoring_dir: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int | None = None,
    n_jobs: int = -1,
) -> RetrainingEvaluationResult:
    """Train a challenger on matured history and evaluate it on untouched later batches."""

    source = (processed_dir or config.paths.processed_dir).resolve()
    monitoring = (
        monitoring_dir or (config.paths.artifact_dir / "private" / "monitoring")
    ).resolve()
    destination = (output_dir or (config.paths.artifact_dir / "private" / "retraining")).resolve()
    train_path = source / "train.parquet"
    recommendations_path = monitoring / "recommendations.parquet"
    if not train_path.is_file() or not recommendations_path.is_file():
        raise FileNotFoundError(
            "Prepared train data and replay recommendations are required before retraining."
        )
    champion = load_model_bundle(bundle_path)
    frame = (
        pl.read_parquet(train_path)
        .to_pandas()
        .sort_values([config.data.time_column, config.data.transaction_id_column], kind="stable")
        .reset_index(drop=True)
    )
    production = frame[frame[PERIOD_COLUMN] == "production"].copy()
    maximum_batch = int(production[PRODUCTION_BATCH_COLUMN].max())
    calibration_batches, evaluation_batches = _select_windows(
        production,
        maximum_batch=maximum_batch,
        label_delay_batches=config.split.label_delay_batches,
        target_column=config.data.target_column,
    )
    calibration_start = min(calibration_batches)
    training = frame[
        (frame[PERIOD_COLUMN] != "production")
        | (frame[PRODUCTION_BATCH_COLUMN] < calibration_start)
    ].copy()
    calibration = production[production[PRODUCTION_BATCH_COLUMN].isin(calibration_batches)].copy()
    evaluation = production[production[PRODUCTION_BATCH_COLUMN].isin(evaluation_batches)].copy()
    training_target = training[config.data.target_column].to_numpy(dtype=int)
    calibration_target = calibration[config.data.target_column].to_numpy(dtype=int)
    evaluation_target = evaluation[config.data.target_column].to_numpy(dtype=int)
    if np.unique(training_target).size < 2 or np.unique(evaluation_target).size < 2:
        raise ValueError("Challenger training and evaluation must each contain both classes.")

    challenger_builder = CausalFeatureBuilder()
    training_engineered = challenger_builder.transform_batch(training)
    calibration_engineered = challenger_builder.transform_batch(calibration)
    evaluation_engineered = challenger_builder.transform_batch(evaluation)
    challenger_preprocessor = FeaturePreprocessor(
        categorical_cardinality_limit=config.model.categorical_cardinality_limit,
        missingness_drop_threshold=config.model.missingness_drop_threshold,
    )
    training_features = challenger_preprocessor.fit_transform(training_engineered)
    calibration_features = challenger_preprocessor.transform(calibration_engineered)
    evaluation_features = challenger_preprocessor.transform(evaluation_engineered)
    parameters = dict(champion.training_parameters)
    estimators = int(parameters.pop("n_estimators"))
    challenger_model = _fit_lgbm(
        parameters,
        training_features,
        training_target,
        validation_features=None,
        validation_target=None,
        estimators=estimators,
        early_stopping_rounds=1,
        random_seed=config.random_seed,
        n_jobs=n_jobs,
    )
    challenger_calibrator, _ = fit_probability_calibrators(
        challenger_model.predict_proba(calibration_features)[:, 1], calibration_target
    )
    challenger_calibration_scores = challenger_calibrator.predict(
        challenger_model.predict_proba(calibration_features)[:, 1]
    )
    challenger_threshold = threshold_for_review_rate(
        challenger_calibration_scores, config.model.default_review_rate
    )
    challenger_scores = challenger_calibrator.predict(
        challenger_model.predict_proba(evaluation_features)[:, 1]
    )

    champion_builder = CausalFeatureBuilder()
    history = frame[frame[config.data.time_column] < evaluation[config.data.time_column].min()]
    champion_builder.transform_batch(history)
    champion_evaluation, champion_scores = _score_with_builder(
        champion.model,
        champion.calibrator,
        champion.preprocessor,
        champion_builder,
        evaluation,
    )
    iterations = bootstrap_iterations or config.monitoring.bootstrap_iterations
    pr_auc_difference = paired_stratified_difference(
        evaluation_target,
        champion_scores,
        challenger_scores,
        average_precision_score,
        iterations=iterations,
        random_seed=config.monitoring.random_seed,
    )

    # The generic paired helper applies the same callable to both score arrays, so use
    # binary review decisions here to preserve each model's independently frozen threshold.
    recall_difference = paired_stratified_difference(
        evaluation_target,
        (champion_scores >= champion.default_threshold).astype(float),
        (challenger_scores >= challenger_threshold).astype(float),
        lambda y, decisions: float(recall_score(y, decisions.astype(bool), zero_division=0)),
        iterations=iterations,
        random_seed=config.monitoring.random_seed,
    )
    champion_metrics = binary_classification_metrics(
        evaluation_target,
        champion_scores,
        threshold=champion.default_threshold,
        amounts=evaluation["TransactionAmt"].to_numpy(dtype=float),
    )
    challenger_metrics = binary_classification_metrics(
        evaluation_target,
        challenger_scores,
        threshold=challenger_threshold,
        amounts=evaluation["TransactionAmt"].to_numpy(dtype=float),
    )
    profiler = SegmentProfiler.fit(training_engineered)
    segment_comparison = _segment_recall_comparison(
        profiler,
        champion_evaluation,
        evaluation_target,
        champion_scores,
        challenger_scores,
        champion_threshold=champion.default_threshold,
        challenger_threshold=challenger_threshold,
        minimum_positive=config.monitoring.minimum_segment_positive,
        minimum_negative=config.monitoring.minimum_segment_negative,
        iterations=iterations,
        random_seed=config.monitoring.random_seed,
    )
    reliable_improvement = pr_auc_difference.lower > 0
    recall_non_inferior = recall_difference.lower >= 0
    no_segment_regression = not (
        not segment_comparison.empty and segment_comparison["reliable_regression"].any()
    )
    retrain_recommended = bool(
        reliable_improvement and recall_non_inferior and no_segment_regression
    )

    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "retraining_evaluation.json"
    segment_comparison_path = destination / "challenger_segment_comparison.parquet"
    segment_comparison.to_parquet(segment_comparison_path, index=False)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "champion_model_version": champion.model_version,
        "training_rows": len(training),
        "training_elapsed_day_end": float(training[config.data.time_column].max() / 86_400),
        "calibration_batches": list(calibration_batches),
        "evaluation_batches": list(evaluation_batches),
        "evaluation_rows": len(evaluation),
        "challenger_threshold": challenger_threshold,
        "champion_threshold": champion.default_threshold,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "pr_auc_difference": asdict(pr_auc_difference),
        "recall_difference": asdict(recall_difference),
        "eligible_segments": len(segment_comparison),
        "criteria": {
            "statistically_reliable_pr_auc_improvement": reliable_improvement,
            "recall_non_inferior": recall_non_inferior,
            "no_reliable_segment_recall_regression": no_segment_regression,
        },
        "retrain_recommended": retrain_recommended,
        "outcome": "retrain_recommended" if retrain_recommended else "challenger_rejected",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    recommendations = pd.read_parquet(recommendations_path)
    mature = recommendations[
        (recommendations["stream"] == "production")
        & (recommendations["batch_number"].isin(evaluation_batches))
    ]
    if mature.empty:
        raise ValueError("Replay recommendations do not include the evaluation batches.")
    selected_index = mature.sort_values("batch_number").index[-1]
    recommendations.loc[selected_index, "challenger_evaluated"] = True
    recommendations.loc[selected_index, "retrain_recommended"] = retrain_recommended
    recommendations.loc[selected_index, "challenger_outcome"] = summary["outcome"]
    recommendations.to_parquet(recommendations_path, index=False)
    return RetrainingEvaluationResult(
        summary_path=summary_path,
        segment_comparison_path=segment_comparison_path,
        retrain_recommended=retrain_recommended,
        training_rows=len(training),
        calibration_batches=calibration_batches,
        evaluation_batches=evaluation_batches,
    )
