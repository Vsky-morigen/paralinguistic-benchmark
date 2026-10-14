from abc import ABC, abstractmethod

from parabench.io import sha256_file
from parabench.schemas import PredictionResult, UnderstandingRequest


class ModelError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def validate_input(request: UnderstandingRequest) -> None:
    if sha256_file(request.audio_path) != request.audio_sha256:
        raise ModelError("Input audio hash mismatch")


class AudioModelAdapter(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def predict(self, request: UnderstandingRequest) -> PredictionResult:
        """Read the supplied audio and return this dimension's answer."""

    async def close(self) -> None:
        """Release persistent resources."""
