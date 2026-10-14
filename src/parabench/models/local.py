"""Load a user-supplied speech model once and infer directly in this process."""

import asyncio
import importlib
import importlib.util
import io
import math
import struct
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from parabench.audio import inspect_wav
from parabench.io import atomic_bytes, fingerprint
from parabench.models.base import ModelError, SpeechModelAdapter
from parabench.schemas import GenerationRequest, GenerationResult


@dataclass
class LocalAudio:
    """Return exactly one of mono normalized float samples or an existing WAV."""

    sample_rate: int
    samples: Sequence[float] | None = None
    audio_path: str | Path | None = None
    simulated: bool = False


class LocalSpeechModel(Protocol):
    def generate(self, request: GenerationRequest) -> LocalAudio:
        """Use request.text, language, instructions, and generation options."""


def load_factory(reference: str):
    source, name = reference.rsplit(":", 1)
    if source.endswith(".py"):
        path = Path(source)
        module_name = "_parabench_local_" + fingerprint(str(path.resolve()))[:16]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the local loader file")
        module = importlib.util.module_from_spec(spec)
        # Some model/dataclass modules need registration during execution.
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


def save_local_audio(value: LocalAudio, output_path: Path) -> dict:
    if not isinstance(value, LocalAudio):
        raise ValueError("Local model must return LocalAudio")
    if (value.samples is None) == (value.audio_path is None):
        raise ValueError("Return exactly one of samples or audio_path")
    if type(value.sample_rate) is not int or not 8000 <= value.sample_rate <= 192000:
        raise ValueError("Invalid local audio sample rate")
    if type(value.simulated) is not bool:
        raise ValueError("simulated must be boolean")
    if value.audio_path is not None:
        # Validate before copying; never replace a source file returned by a model.
        info = inspect_wav(Path(value.audio_path))
        if info["sample_rate"] != value.sample_rate:
            raise ValueError("Local audio sample rate mismatch")
        atomic_bytes(output_path, Path(value.audio_path).read_bytes())
        return info
    samples = value.samples
    if hasattr(samples, "tolist"):
        samples = samples.tolist()
    if (
        not isinstance(samples, (list, tuple))
        or not samples
        or len(samples) / value.sample_rate > 600
    ):
        raise ValueError("Expected a nonempty mono sample sequence, at most 600 seconds")
    pcm = bytearray()
    for item in samples:
        if type(item) not in {int, float} or not math.isfinite(item) or not -1 <= item <= 1:
            raise ValueError("Local samples must be finite mono floats in [-1, 1]")
        pcm.extend(struct.pack("<h", max(-32768, min(32767, round(item * 32767)))))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setparams((1, 2, value.sample_rate, 0, "NONE", "not compressed"))
        handle.writeframes(bytes(pcm))
    atomic_bytes(output_path, buf.getvalue())
    return inspect_wav(output_path)


class LocalAdapter(SpeechModelAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        self._model = None
        self._load_failure = None
        self._lock = threading.Lock()
        self._closed = False
        self._load_seconds = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parabench-local")

    def _generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        # Serialize model access even when the adapter is used outside the CLI.
        with self._lock:
            if self._closed:
                raise ModelError("Local adapter has been closed")
            if self._load_failure:
                raise ModelError(self._load_failure)
            started = time.perf_counter()
            if self._model is None:
                load_start = time.perf_counter()
                try:
                    factory = load_factory(self.config["loader"])
                    self._model = factory(
                        model_path=Path(self.config["model_path"]),
                        device=self.config.get("device", "auto"),
                        **self.config.get("load_kwargs", {}),
                    )
                    if not callable(getattr(self._model, "generate", None)):
                        raise TypeError("Local model has no generate method")
                except Exception as exc:
                    self._model = None
                    self._load_failure = f"Local model loading failed ({type(exc).__name__}); check loader and model dependencies"
                    raise ModelError(self._load_failure) from None
                self._load_seconds = time.perf_counter() - load_start
            try:
                audio = self._model.generate(request)
                save_local_audio(audio, output_path)
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                raise ModelError(f"Local generation failed ({type(exc).__name__})") from None
            return GenerationResult(
                str(output_path),
                audio.sample_rate,
                time.perf_counter() - started,
                self.config["model"],
                simulated=audio.simulated,
                metadata={
                    "backend": "local_python",
                    "device": self.config.get("device", "auto"),
                    "load_seconds": self._load_seconds,
                },
            )

    async def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        if self._closed:
            raise ModelError("Local adapter has been closed")
        return await asyncio.wrap_future(
            self._executor.submit(self._generate, request, output_path)
        )

    def _close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            model, self._model = self._model, None
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    raise ModelError(f"Local model cleanup failed ({type(exc).__name__})") from None

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await asyncio.wrap_future(self._executor.submit(self._close))
        finally:
            self._executor.shutdown(wait=True)
