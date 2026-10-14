"""Load a user-supplied audio understanding model once per run."""

import asyncio
import importlib
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from parabench.io import fingerprint
from parabench.models.base import AudioModelAdapter, ModelError, validate_input
from parabench.schemas import PredictionResult, UnderstandingRequest


class LocalUnderstandingModel(Protocol):
    def predict(self, request: UnderstandingRequest) -> object:
        """Read request.audio_path; return a label or a list of positions."""


def load_factory(reference: str):
    source, name = reference.rsplit(":", 1)
    if source.endswith(".py"):
        path = Path(source)
        module_name = "_parabench_local_" + fingerprint(str(path.resolve()))[:16]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the local loader file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = importlib.import_module(source)
    factory = getattr(module, name)
    if not callable(factory):
        raise ValueError("Local loader is not callable")
    return factory


class LocalAdapter(AudioModelAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        self._model = None
        self._load_failure = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parabench-local")

    def _predict(self, request: UnderstandingRequest) -> PredictionResult:
        if self._closed:
            raise ModelError("Local adapter has been closed")
        validate_input(request)
        if self._load_failure:
            raise ModelError(self._load_failure)
        started = time.perf_counter()
        if self._model is None:
            try:
                self._model = load_factory(self.config["loader"])(
                    model_path=Path(self.config["model_path"]),
                    device=self.config.get("device", "auto"),
                    **self.config.get("load_kwargs", {}),
                )
                if not callable(getattr(self._model, "predict", None)):
                    raise TypeError("Local model has no predict method")
            except Exception as exc:
                self._model = None
                self._load_failure = f"Local model loading failed ({type(exc).__name__}); check loader and dependencies"
                raise ModelError(self._load_failure) from None
        try:
            answer = self._model.predict(request)
        except Exception as exc:
            raise ModelError(f"Local inference failed ({type(exc).__name__})") from None
        return PredictionResult(
            answer,
            time.perf_counter() - started,
            self.config["model"],
            simulated=self.config.get("simulated", False),
        )

    async def predict(self, request: UnderstandingRequest) -> PredictionResult:
        if self._closed:
            raise ModelError("Local adapter has been closed")
        return await asyncio.wrap_future(self._executor.submit(self._predict, request))

    def _close(self):
        if self._closed:
            return
        self._closed = True
        model, self._model = self._model, None
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                raise ModelError(f"Local cleanup failed ({type(exc).__name__})") from None

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await asyncio.wrap_future(self._executor.submit(self._close))
        finally:
            self._executor.shutdown(wait=True)
