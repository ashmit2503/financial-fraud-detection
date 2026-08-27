"""Command-line entry point for the project pipelines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from fraud_monitor import __version__
from fraud_monitor.config import load_config


def _path_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-monitor",
        description="Temporal fraud detection and monitoring pipelines.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config", help="Validate and print configuration.")
    show_config.add_argument("--config", default="configs/base.yaml", type=Path)

    prepare = subparsers.add_parser("prepare", help="Validate and prepare IEEE-CIS data.")
    prepare.add_argument("--config", default="configs/base.yaml", type=Path)
    prepare.add_argument("--raw-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)

    train = subparsers.add_parser("train", help="Tune, calibrate, and save the champion model.")
    train.add_argument("--config", default="configs/base.yaml", type=Path)
    train.add_argument("--processed-dir", type=Path)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--trials", type=int)
    train.add_argument(
        "--quick",
        action="store_true",
        help="Use one short trial for pipeline verification rather than model selection.",
    )
    train.add_argument("--no-mlflow", action="store_true")
    train.add_argument("--bootstrap-iterations", type=int)

    replay = subparsers.add_parser("replay", help="Replay production and build monitoring tables.")
    replay.add_argument("--config", default="configs/base.yaml", type=Path)
    replay.add_argument("--processed-dir", type=Path)
    replay.add_argument("--bundle", required=True, type=Path)
    replay.add_argument("--output-dir", type=Path)
    replay.add_argument("--bootstrap-iterations", type=int)

    retrain = subparsers.add_parser(
        "retrain-eval", help="Train and evaluate a leakage-safe manual challenger."
    )
    retrain.add_argument("--config", default="configs/base.yaml", type=Path)
    retrain.add_argument("--processed-dir", type=Path)
    retrain.add_argument("--monitoring-dir", type=Path)
    retrain.add_argument("--bundle", required=True, type=Path)
    retrain.add_argument("--output-dir", type=Path)
    retrain.add_argument("--bootstrap-iterations", type=int)
    retrain.add_argument("--n-jobs", type=int, default=-1)

    demo = subparsers.add_parser(
        "build-demo", help="Export public-safe aggregate dashboard artifacts."
    )
    demo.add_argument("--config", default="configs/base.yaml", type=Path)
    demo.add_argument("--source-dir", type=Path)
    demo.add_argument("--output-dir", type=Path)
    demo.add_argument("--review-budget", type=Path)
    demo.add_argument("--synthetic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show-config":
        config = load_config(args.config)
        print(json.dumps(asdict(config), indent=2, default=_path_default))
        return 0
    if args.command == "prepare":
        from fraud_monitor.data import prepare_dataset

        config = load_config(args.config)
        result = prepare_dataset(config, raw_dir=args.raw_dir, output_dir=args.output_dir)
        print(
            json.dumps(
                {
                    "train_path": str(result.train_path),
                    "test_path": str(result.test_path),
                    "manifest_path": str(result.manifest_path),
                    "train_rows": result.train_rows,
                    "test_rows": result.test_rows,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "replay":
        from fraud_monitor.replay import run_replay

        config = load_config(args.config)
        result = run_replay(
            config,
            bundle_path=args.bundle,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(asdict(result), indent=2, default=_path_default))
        return 0
    if args.command == "retrain-eval":
        from fraud_monitor.retraining import run_retraining_evaluation

        config = load_config(args.config)
        result = run_retraining_evaluation(
            config,
            bundle_path=args.bundle,
            processed_dir=args.processed_dir,
            monitoring_dir=args.monitoring_dir,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(asdict(result), indent=2, default=_path_default))
        return 0
    if args.command == "build-demo":
        from fraud_monitor.demo import export_demo_artifacts, generate_synthetic_demo

        config = load_config(args.config)
        output_dir = args.output_dir or config.paths.demo_dir
        if args.synthetic:
            result = generate_synthetic_demo(output_dir)
        else:
            source_dir = args.source_dir or (config.paths.artifact_dir / "private" / "monitoring")
            result = export_demo_artifacts(
                source_dir,
                output_dir,
                review_budget_path=args.review_budget,
            )
        print(json.dumps(asdict(result), indent=2, default=_path_default))
        return 0
    if args.command == "train":
        from fraud_monitor.modeling import train_from_prepared

        config = load_config(args.config)
        result = train_from_prepared(
            config,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            trials=1 if args.quick else args.trials,
            max_estimators=80 if args.quick else 2_000,
            early_stopping_rounds=10 if args.quick else 100,
            n_jobs=1 if args.quick else -1,
            enable_mlflow=not args.no_mlflow,
            bootstrap_iterations=(
                args.bootstrap_iterations
                if args.bootstrap_iterations is not None
                else 100
                if args.quick
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "bundle_path": str(result.bundle_path),
                    "summary_path": str(result.summary_path),
                    "budget_path": str(result.budget_path),
                    "reliability_path": str(result.reliability_path),
                    "model_version": result.model_version,
                    "data_version": result.data_version,
                    "acceptance_metrics": result.acceptance_metrics,
                },
                indent=2,
                default=_path_default,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
