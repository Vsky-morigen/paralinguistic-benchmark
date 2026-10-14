from abc import ABC, abstractmethod
from pathlib import Path

from parabench.schemas import GenerationRequest, GenerationResult


class ModelError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class SpeechModelAdapter(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        """Save one complete PCM WAV and return metadata. Never expose reference audio."""

    async def close(self) -> None:
        """Override for adapters with persistent network sessions."""
