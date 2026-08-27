"""Temporal LightGBM training, calibration, versioning, and model artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from fraud_monitor.baselines import fit_baselines
from fraud_monitor.config import ProjectConfig
from fraud_monitor.evaluation import (
    binary_classification_metrics,
    expected_calibration_error,
    review_budget_table,
    thresholds_for_review_rates,
)
from fraud_monitor.features import CausalFeatureBuilder, FeaturePreprocessor
from fraud_monitor.splits import (
    DEVELOPMENT_BLOCK_COLUMN,
    PERIOD_COLUMN,
    expanding_development_folds,
)


class TrainingError(RuntimeError):
    """Raised when chronological model training cannot produce a valid model."""


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: Any

    def predict(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float)
        if self.method == "sigmoid":
            values = logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            return self.estimator.predict_proba(values)[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.estimator.predict(raw), dtype=float)
        raise TrainingError(f"Unsupported calibration method: {self.method}")


@dataclass(frozen=True)
class CalibrationCandidate:
    method: str
    brier_score: float
    expected_calibration_error: float


@dataclass
class ModelBundle:
    """Serializable deployment contract for chronological batch scoring."""

    model: lgb.LGBMClassifier
    preprocessor: FeaturePreprocessor
    causal_builder: CausalFeatureBuilder
    calibrator: ProbabilityCalibrator
    thresholds: dict[float, float]
    default_review_rate: float
    model_version: str
    data_version: str
    created_at_utc: str
    training_parameters: dict[str, Any]

    def score_batch(self, frame: pd.DataFrame, *, update_state: bool = True) -> np.ndarray:
        engineered = self.causal_builder.transform_batch(frame, update_state=update_state)
        features = self.preprocessor.transform(engineered)
        raw = self.model.predict_proba(features)[:, 1]
        return self.calibrator.predict(raw)

    @property
    def default_threshold(self) -> float:
        return self.thresholds[self.default_review_rate]


@dataclass(frozen=True)
class FoldData:
    name: str
    train_features: pd.DataFrame
    train_target: np.ndarray
    validation_features: pd.DataFrame
    validation_target: np.ndarray


@dataclass(frozen=True)
class TuningResult:
    parameters: dict[str, Any]
    mean_pr_auc: float
    fold_pr_auc: tuple[float, ...]
    final_estimators: int
    trials: int


@dataclass(frozen=True)
class TrainingResult:
    bundle_path: Path
    summary_path: Path
    budget_path: Path
    model_version: str
    data_version: str
    acceptance_metrics: dict[str, float | int]


def fit_probability_calibrators(
    raw_probabilities: np.ndarray,
    target: np.ndarray,
) -> tuple[ProbabilityCalibrator, tuple[CalibrationCandidate, ...]]:
    """Fit sigmoid and isotonic candidates and select by Brier score then ECE."""

    raw = np.asarray(raw_probabilities, dtype=float)
    y = np.asarray(target, dtype=int)
    if raw.shape != y.shape or raw.size == 0 or np.unique(y).size < 2:
        raise TrainingError("Calibration requires aligned scores with both target classes.")

    sigmoid_estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    sigmoid_estimator.fit(logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
    sigmoid = ProbabilityCalibrator("sigmoid", sigmoid_estimator)
    isotonic = ProbabilityCalibrator(
        "isotonic",
        IsotonicRegression(out_of_bounds="clip").fit(raw, y),
    )
    calibrators = (sigmoid, isotonic)
    candidates = tuple(
        CalibrationCandidate(
            method=calibrator.method,
            brier_score=float(brier_score_loss(y, calibrator.predict(raw))),
            expected_calibration_error=expected_calibration_error(y, calibrator.predict(raw)),
        )
        for calibrator in calibrators
    )
    ordering = sorted(
        zip(calibrators, candidates, strict=True),
        key=lambda item: (
            item[1].brier_score,
            item[1].expected_calibration_error,
            item[1].method != "sigmoid",
        ),
    )
    return ordering[0][0], candidates


def _require_binary_target(target: np.ndarray, context: str) -> None:
    if set(np.unique(target)) != {0, 1}:
        raise TrainingError(f"{context} must contain both fraud and legitimate examples.")


def _prepare_temporal_folds(
    development: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[FoldData, ...]:
    folds: list[FoldData] = []
    target_column = config.data.target_column
    for fold in expanding_development_folds(config.split.development_blocks):
        train = development[development[DEVELOPMENT_BLOCK_COLUMN].isin(fold.train_blocks)].copy()
        validation = development[
            development[DEVELOPMENT_BLOCK_COLUMN] == fold.validation_block
        ].copy()
        builder = CausalFeatureBuilder()
        train_engineered = builder.transform_batch(train)
        validation_engineered = builder.transform_batch(validation)
        preprocessor = FeaturePreprocessor(
            categorical_cardinality_limit=config.model.categorical_cardinality_limit,
            missingness_drop_threshold=config.model.missingness_drop_threshold,
        )
        train_features = preprocessor.fit_transform(train_engineered)
        validation_features = preprocessor.transform(validation_engineered)
        train_target = train[target_column].to_numpy(dtype=int)
        validation_target = validation[target_column].to_numpy(dtype=int)
        _require_binary_target(train_target, f"{fold.name} training target")
        _require_binary_target(validation_target, f"{fold.name} validation target")
        folds.append(
            FoldData(
                name=fold.name,
                train_features=train_features,
                train_target=train_target,
                validation_features=validation_features,
                validation_target=validation_target,
            )
        )
    return tuple(folds)


def _trial_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 128, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 12]),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.70, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight_mode": trial.suggest_categorical("class_weight_mode", ["none", "balanced"]),
    }


def _fit_lgbm(
    parameters: dict[str, Any],
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    *,
    validation_features: pd.DataFrame | None,
    validation_target: np.ndarray | None,
    estimators: int,
    early_stopping_rounds: int,
    random_seed: int,
    n_jobs: int,
) -> lgb.LGBMClassifier:
    model_parameters = dict(parameters)
    weight_mode = model_parameters.pop("class_weight_mode", "none")
    if weight_mode == "balanced":
        positives = int(train_target.sum())
        negatives = int((train_target == 0).sum())
        model_parameters["scale_pos_weight"] = negatives / positives
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **model_parameters,
    )
    fit_arguments: dict[str, Any] = {}
    if validation_features is not None and validation_target is not None:
        fit_arguments = {
            "eval_X": validation_features,
            "eval_y": validation_target,
            "eval_metric": "average_precision",
            "callbacks": [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        }
    model.fit(train_features, train_target, **fit_arguments)
    return model


def tune_lightgbm(
    folds: tuple[FoldData, ...],
    *,
    trials: int,
    max_estimators: int,
    early_stopping_rounds: int,
    random_seed: int,
    n_jobs: int,
) -> TuningResult:
    """Tune a compact parameter space against mean expanding-fold PR-AUC."""

    if not folds:
        raise TrainingError("At least one temporal fold is required for tuning.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        parameters = _trial_parameters(trial)
        scores: list[float] = []
        iterations: list[int] = []
        for fold in folds:
            model = _fit_lgbm(
                parameters,
                fold.train_features,
                fold.train_target,
                validation_features=fold.validation_features,
                validation_target=fold.validation_target,
                estimators=max_estimators,
                early_stopping_rounds=early_stopping_rounds,
                random_seed=random_seed,
                n_jobs=n_jobs,
            )
            probability = model.predict_proba(fold.validation_features)[:, 1]
            scores.append(float(average_precision_score(fold.validation_target, probability)))
            iterations.append(int(model.best_iteration_ or max_estimators))
        trial.set_user_attr("fold_pr_auc", scores)
        trial.set_user_attr("best_iterations", iterations)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_trial
    iteration_values = [int(value) for value in best.user_attrs["best_iterations"]]
    return TuningResult(
        parameters=dict(best.params),
        mean_pr_auc=float(best.value),
        fold_pr_auc=tuple(float(value) for value in best.user_attrs["fold_pr_auc"]),
        final_estimators=max(1, int(np.median(iteration_values))),
        trials=trials,
    )


def _data_version(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest["source_files"], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _model_version(data_version: str, tuning: TuningResult) -> str:
    payload = json.dumps(
        {"data_version": data_version, "parameters": tuning.parameters}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def train_from_prepared(
    config: ProjectConfig,
    *,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
    trials: int | None = None,
    max_estimators: int = 2_000,
    early_stopping_rounds: int = 100,
    n_jobs: int = -1,
    enable_mlflow: bool = True,
) -> TrainingResult:
    """Train, calibrate, evaluate, version, and save the deployment bundle."""

    source = (processed_dir or config.paths.processed_dir).resolve()
    destination = (output_dir or (config.paths.artifact_dir / "private" / "model")).resolve()
    train_path = source / "train.parquet"
    manifest_path = source / "manifest.json"
    if not train_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Prepared train.parquet and manifest.json are required.")

    frame = pl.read_parquet(train_path).to_pandas()
    frame = frame.sort_values(
        [config.data.time_column, config.data.transaction_id_column], kind="stable"
    ).reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development = frame[frame[PERIOD_COLUMN] == "development"].copy()
    calibration = frame[frame[PERIOD_COLUMN] == "calibration"].copy()
    acceptance = frame[frame[PERIOD_COLUMN] == "acceptance"].copy()
    for name, subset in (
        ("development", development),
        ("calibration", calibration),
        ("acceptance", acceptance),
    ):
        _require_binary_target(subset[config.data.target_column].to_numpy(dtype=int), name)

    folds = _prepare_temporal_folds(development, config)
    tuning = tune_lightgbm(
        folds,
        trials=trials or config.model.optuna_trials,
        max_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=config.random_seed,
        n_jobs=n_jobs,
    )

    causal_builder = CausalFeatureBuilder()
    development_engineered = causal_builder.transform_batch(development)
    calibration_engineered = causal_builder.transform_batch(calibration)
    acceptance_engineered = causal_builder.transform_batch(acceptance)
    preprocessor = FeaturePreprocessor(
        categorical_cardinality_limit=config.model.categorical_cardinality_limit,
        missingness_drop_threshold=config.model.missingness_drop_threshold,
    )
    development_features = preprocessor.fit_transform(development_engineered)
    calibration_features = preprocessor.transform(calibration_engineered)
    acceptance_features = preprocessor.transform(acceptance_engineered)
    development_target = development[config.data.target_column].to_numpy(dtype=int)
    calibration_target = calibration[config.data.target_column].to_numpy(dtype=int)
    acceptance_target = acceptance[config.data.target_column].to_numpy(dtype=int)

    model = _fit_lgbm(
        tuning.parameters,
        development_features,
        development_target,
        validation_features=None,
        validation_target=None,
        estimators=tuning.final_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=config.random_seed,
        n_jobs=n_jobs,
    )
    raw_calibration = model.predict_proba(calibration_features)[:, 1]
    calibrator, calibration_candidates = fit_probability_calibrators(
        raw_calibration, calibration_target
    )
    calibrated_calibration = calibrator.predict(raw_calibration)
    thresholds = thresholds_for_review_rates(calibrated_calibration, config.model.review_rates)
    raw_acceptance = model.predict_proba(acceptance_features)[:, 1]
    acceptance_probability = calibrator.predict(raw_acceptance)
    acceptance_metrics = binary_classification_metrics(
        acceptance_target,
        acceptance_probability,
        threshold=thresholds[config.model.default_review_rate],
        amounts=acceptance["TransactionAmt"].to_numpy(dtype=float),
    )
    budget_table = review_budget_table(
        acceptance_target,
        acceptance_probability,
        thresholds=thresholds,
        amounts=acceptance["TransactionAmt"].to_numpy(dtype=float),
    )

    baselines = fit_baselines(development_engineered, development_target)
    baseline_probabilities = baselines.predict_probabilities(acceptance_engineered)
    baseline_pr_auc = {
        name: float(average_precision_score(acceptance_target, probability))
        for name, probability in baseline_probabilities.items()
    }

    data_version = _data_version(manifest)
    model_version = _model_version(data_version, tuning)
    created_at = datetime.now(UTC).isoformat()
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        causal_builder=causal_builder,
        calibrator=calibrator,
        thresholds=thresholds,
        default_review_rate=config.model.default_review_rate,
        model_version=model_version,
        data_version=data_version,
        created_at_utc=created_at,
        training_parameters={**tuning.parameters, "n_estimators": tuning.final_estimators},
    )

    destination.mkdir(parents=True, exist_ok=True)
    bundle_path = destination / "model_bundle.joblib"
    summary_path = destination / "training_summary.json"
    budget_path = destination / "acceptance_review_budgets.parquet"
    joblib.dump(bundle, bundle_path, compress=3)
    budget_table.to_parquet(budget_path, index=False)
    summary = {
        "model_version": model_version,
        "data_version": data_version,
        "created_at_utc": created_at,
        "tuning": asdict(tuning),
        "calibration_candidates": [asdict(candidate) for candidate in calibration_candidates],
        "selected_calibrator": calibrator.method,
        "thresholds": thresholds,
        "acceptance_metrics": acceptance_metrics,
        "baseline_pr_auc": baseline_pr_auc,
        "feature_count": len(preprocessor.schema_.feature_columns),
        "native_categorical_count": len(preprocessor.schema_.native_categorical_columns),
        "frequency_encoded_count": len(preprocessor.schema_.frequency_encoded_columns),
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    if enable_mlflow:
        tracking_path = (config.paths.artifact_dir / "private" / "mlruns").resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_path.as_uri())
        mlflow.set_experiment(config.name)
        with mlflow.start_run(run_name=model_version):
            mlflow.log_params(_json_safe(bundle.training_parameters))
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("calibrator", calibrator.method)
            mlflow.log_metric("temporal_cv_pr_auc", tuning.mean_pr_auc)
            for name, value in acceptance_metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(f"acceptance_{name}", float(value))
            mlflow.log_artifact(str(summary_path))

    return TrainingResult(
        bundle_path=bundle_path,
        summary_path=summary_path,
        budget_path=budget_path,
        model_version=model_version,
        data_version=data_version,
        acceptance_metrics=acceptance_metrics,
    )


def load_model_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TrainingError(f"Artifact at {path} is not a ModelBundle.")
    return bundle
