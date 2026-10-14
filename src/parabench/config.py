import math
import os
import re
from pathlib import Path

import yaml

from parabench.io import fingerprint, sha256_file


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Expected schema_version: 1 in {path}")
    return value


def project_root(config_path: Path) -> Path:
    """Configs live under <project>/configs, regardless of the current directory."""
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    raise ValueError("Configuration must be inside the project's configs directory")


def resolve_dataset(config_path: Path) -> tuple[Path, Path]:
    value = load_yaml(config_path)
    root = project_root(config_path)
    release = (root / value["release"]).resolve()
    reference = os.environ.get("PARABENCH_REFERENCE_ROOT") or value["reference_root"]
    return release, (root / reference).resolve()


def positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def model_config(path: Path) -> dict:
    value = load_yaml(path)
    if value.get("adapter") not in {"mock", "http", "local"}:
        raise ValueError("Unknown model adapter; register custom adapters explicitly")
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError("model must be a nonempty name")
    for key, default, maximum in [("concurrency", 1, 64), ("max_attempts", 1, 10)]:
        v = value.setdefault(key, default)
        if type(v) is not int or not 1 <= v <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    for key, default in [("timeout_seconds", 120), ("retry_backoff_seconds", 1)]:
        value[key] = positive_number(value.get(key, default), key)
    if not isinstance(value.setdefault("generation", {}), dict):
        raise ValueError("generation must be a mapping")
    # API credentials must be supplied by environment, never configuration snapshots.
    if any(k.lower() in {"api_key", "token", "authorization", "headers"} for k in value):
        raise ValueError("Use api_key_env; literal credentials/custom headers are not persisted")
    if value["adapter"] == "local":
        if value["concurrency"] != 1:
            raise ValueError("Local models require concurrency: 1")
        root = project_root(path)
        checkpoint = value.get("model_path")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("Local models require model_path")
        checkpoint = (root / checkpoint).resolve()
        if not checkpoint.exists():
            raise ValueError(f"Local model weights do not exist: {checkpoint}")
        files = (
            [checkpoint]
            if checkpoint.is_file()
            else sorted(p for p in checkpoint.rglob("*") if p.is_file())
        )
        if not files:
            raise ValueError("Local model weight directory is empty")
        value["checkpoint_metadata_fingerprint"] = fingerprint(
            [
                (
                    p.name if checkpoint.is_file() else p.relative_to(checkpoint).as_posix(),
                    p.stat().st_size,
                    p.stat().st_mtime_ns,
                )
                for p in files
            ]
        )
        loader = value.get("loader")
        if not isinstance(loader, str) or ":" not in loader:
            raise ValueError("Local models require loader: module:function or file.py:function")
        source, function = loader.rsplit(":", 1)
        if not re.fullmatch(r"[A-Za-z_]\w*", function):
            raise ValueError("Invalid local loader function name")
        if source.endswith(".py"):
            source_path = (root / source).resolve()
            if not source_path.is_file():
                raise ValueError(f"Local loader file does not exist: {source_path}")
            value["loader_sha256"] = sha256_file(source_path)
        elif not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", source):
            raise ValueError("Invalid local loader module")
        if not isinstance(value.setdefault("load_kwargs", {}), dict):
            raise ValueError("load_kwargs must be a mapping")
        if {"model_path", "device"} & value["load_kwargs"].keys():
            raise ValueError("model_path/device must not be repeated in load_kwargs")
        if value.setdefault("device", "auto") not in {
            "auto",
            "cpu",
            "cuda",
            "mps",
        } and not re.fullmatch(r"cuda:\d+", str(value["device"])):
            raise ValueError("Invalid local model device")
    return value
