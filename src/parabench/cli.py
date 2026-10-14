import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from parabench import __version__
from parabench.config import load_yaml, model_config, project_root, resolve_dataset
from parabench.datasets import validate_release
from parabench.evaluators import validate_config
from parabench.models.base import ModelError
from parabench.reporting import build_report
from parabench.runner import predict_run
from parabench.runner.evaluation import evaluate_run
from parabench.schemas import DIMENSIONS


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="parabench", description="Paralinguistic audio understanding benchmark"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset").add_subparsers(dest="action", required=True)
    dataset.add_parser("validate").add_argument(
        "--config", type=Path, default=Path("configs/dataset.yaml")
    )
    predict = commands.add_parser("predict")
    predict.add_argument("--config", type=Path, required=True, help="Model YAML configuration")
    predict.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    predict.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"))
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--results-root", type=Path)
    predict.add_argument("--resume", action="store_true")
    predict.add_argument("--language", choices=["zh", "en"])
    predict.add_argument("--limit", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    commands.add_parser("report").add_argument("--run", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "dataset":
            result = validate_release(*resolve_dataset(args.config))
        elif args.command == "predict":
            release, audio = resolve_dataset(args.dataset_config)
            evaluation = load_yaml(args.evaluation_config)
            validate_config(evaluation)
            dimensions = tuple(k for k in DIMENSIONS if evaluation["dimensions"][k]["enabled"])
            result = asyncio.run(
                predict_run(
                    release,
                    audio,
                    model_config(args.config),
                    args.results_root or project_root(args.dataset_config) / "results",
                    args.run_id,
                    resume=args.resume,
                    language=args.language,
                    limit=args.limit,
                    model_root=project_root(args.config),
                    dimensions=dimensions,
                )
            )
        elif args.command == "evaluate":
            result = evaluate_run(args.run, load_yaml(args.config))
        else:
            result = build_report(args.run)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "scores"},
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 2 if result.get("failed", 0) or result.get("errors", 0) else 0
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, ModelError) as exc:
        print(f"parabench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
