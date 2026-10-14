from pathlib import Path

from parabench.schemas import UnderstandingRequest


def load_model(*, model_path: Path, device: str, **load_kwargs):
    """Replace with your model SDK's weight-loading code."""
    raise NotImplementedError("Connect your audio understanding model SDK here")


class YourModel:
    def predict(self, request: UnderstandingRequest) -> object:
        # Send audio_path, question, choices and optional transcript to your SDK.
        # Return the selected label, stress spans [[start,end]], or pause positions [n].
        raise NotImplementedError("Connect your SDK's audio inference here")
