from pathlib import Path

from parabench.models.http_adapter import HttpAdapter
from parabench.models.local import LocalAdapter
from parabench.models.mock import MockAdapter

ADAPTERS = {"mock": MockAdapter, "http": HttpAdapter, "local": LocalAdapter}


def create_adapter(config: dict, *, model_root: Path | None = None):
    try:
        adapter = ADAPTERS[config["adapter"]]
    except KeyError as exc:
        raise ValueError("Adapter not registered") from exc
    if config["adapter"] == "local":
        # Resolve paths only for execution; keep saved configurations portable.
        root = model_root or Path.cwd()
        config = dict(config)
        config["model_path"] = str((root / config["model_path"]).resolve())
        source, function = config["loader"].rsplit(":", 1)
        if source.endswith(".py"):
            config["loader"] = str((root / source).resolve()) + ":" + function
    return adapter(config)
