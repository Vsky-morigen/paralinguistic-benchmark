import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from parabench import __version__
from parabench.config import load_yaml, model_config, project_root, resolve_dataset
from parabench.datasets import import_release, validate_release
from parabench.models.base import ModelError
from parabench.reporting import build_report
from parabench.runner import generate_run
from parabench.runner.evaluation import evaluate_run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="parabench", description="Paralinguistic speech expression benchmark"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset").add_subparsers(dest="action", required=True)
    imp = dataset.add_parser("import")
    imp.add_argument("--annotations", type=Path, required=True)
    imp.add_argument("--originals", type=Path, required=True)
    imp.add_argument(
        "--review-workbook", type=Path, help="Use the latest answers directly from the review XLSX"
    )
    imp.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    val = dataset.add_parser("validate")
    val.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    gen = commands.add_parser("generate")
    gen.add_argument("--config", type=Path, required=True, help="Model YAML configuration")
    gen.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    gen.add_argument("--run-id", required=True)
    gen.add_argument("--results-root", type=Path)
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--language", choices=["zh", "en"])
    gen.add_argument("--limit", type=int)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    ev.add_argument("--evidence", type=Path)
    rep = commands.add_parser("report")
    rep.add_argument("--run", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "dataset":
            release, references = resolve_dataset(args.config)
            if args.action == "import":
                result = import_release(
                    args.annotations, args.originals, release, references, args.review_workbook
                )
            else:
                result = validate_release(release, references)
        elif args.command == "generate":
            release, _ = resolve_dataset(args.dataset_config)
            config = model_config(args.config)
            results_root = args.results_root or project_root(args.dataset_config) / "results"
            result = asyncio.run(
                generate_run(
                    release,
                    config,
                    results_root,
                    args.run_id,
                    resume=args.resume,
                    language=args.language,
                    limit=args.limit,
                    model_root=project_root(args.config),
                )
            )
        elif args.command == "evaluate":
            result = evaluate_run(args.run, load_yaml(args.config), args.evidence)
        else:
            result = build_report(args.run)
        # Keep reports in files; stdout gives a bounded summary.
        display = {k: v for k, v in result.items() if k not in {"scores", "source_hashes"}}
        print(json.dumps(display, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("failed", 0) or result.get("errors", 0) else 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        json.JSONDecodeError,
        yaml.YAMLError,
        ModelError,
    ) as exc:
        print(f"parabench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
