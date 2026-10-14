from parabench.models.base import AudioModelAdapter, validate_input
from parabench.schemas import PredictionResult, UnderstandingRequest


class MockAdapter(AudioModelAdapter):
    async def predict(self, request: UnderstandingRequest) -> PredictionResult:
        validate_input(request)
        answer = [] if request.dimension in {"stress", "pause"} else request.choices[0]
        return PredictionResult(answer, 0.0, self.config["model"], simulated=True)
