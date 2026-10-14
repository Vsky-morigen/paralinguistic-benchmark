"""Copy and adapt this loader to the speech model SDK you use.

The framework calls load_model once, then model.generate for every request.
This is a template, not a preinstalled speech model or a synthetic substitute.
"""

from pathlib import Path

from parabench.models.local import LocalAudio
from parabench.schemas import GenerationRequest


def load_model(*, model_path: Path, device: str, **load_kwargs):
    # Import your model's SDK here, not at benchmark startup, for example:
    # from your_speech_project import SpeechModel
    # native_model = SpeechModel.load(model_path, device=device, **load_kwargs)
    # return YourModelWrapper(native_model)
    raise NotImplementedError("Implement this loader with your speech model SDK")


class YourModelWrapper:
    def __init__(self, native_model):
        self.native_model = native_model

    def generate(self, request: GenerationRequest) -> LocalAudio:
        # Map ALL supported inputs to your model's inference function:
        # result = self.native_model.synthesize(
        #     text=request.text,
        #     instructions=request.instructions,
        #     language=request.language,
        #     **request.generation,
        # )
        # If the SDK returns a tensor, convert it to mono CPU floats, then:
        # return LocalAudio(sample_rate=result.sample_rate, samples=result.audio.tolist())
        # If the SDK writes a PCM WAV:
        # return LocalAudio(sample_rate=16000, audio_path=result.wav_path)
        raise NotImplementedError("Map this request to your model's inference function")

    def close(self):
        # Release optional sessions/model resources here.
        pass
