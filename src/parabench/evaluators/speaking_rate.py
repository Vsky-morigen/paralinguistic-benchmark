import math

from parabench.evaluators.base import Evaluator
from parabench.schemas import EvaluationResult, Sample


class SpeakingRateEvaluator(Evaluator):
    dimension = "speaking_rate"

    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        if not sample.targets["speed_verified"]:
            return self.skip("Reference speaking rate has not been human validated")
        item = self.evidence_for(evidence)
        if item is None:
            return self.skip("No calibrated speaking-rate evidence")
        if item["method"] != "acoustic":
            raise ValueError("Speaking-rate evidence must use calibrated alignment/timing")
        bounds = sample.targets.get("speed_range")
        rate = item.get("rate")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(type(v) in {int, float} and math.isfinite(v) for v in bounds)
            or not 0 < bounds[0] <= bounds[1]
        ):
            return self.skip("No validated numeric target range")
        if type(rate) not in {int, float} or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Observed rate must be positive and finite")
        expected_unit = "characters/second" if sample.language == "zh" else "words/second"
        if item.get("unit") != expected_unit or sample.targets.get("speed_unit") != expected_unit:
            raise ValueError("Speaking-rate unit mismatch")
        passed = bounds[0] <= rate <= bounds[1]
        return EvaluationResult(
            "speaking_rate", "scored", float(passed), passed, "acoustic", evidence=item
        )
