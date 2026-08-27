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
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
