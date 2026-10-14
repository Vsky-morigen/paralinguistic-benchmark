import math

from parabench.evaluators.content import ContentEvaluator
from parabench.evaluators.context import ContextEvaluator
from parabench.evaluators.emotion import EmotionEvaluator
from parabench.evaluators.gender import GenderEvaluator
from parabench.evaluators.pause import PauseEvaluator
from parabench.evaluators.speaking_rate import SpeakingRateEvaluator
from parabench.evaluators.stress import StressEvaluator
from parabench.schemas import DIMENSIONS, EvaluationResult

EVALUATORS = {
    "content": ContentEvaluator,
    "gender": GenderEvaluator,
    "emotion": EmotionEvaluator,
    "stress": StressEvaluator,
    "pause": PauseEvaluator,
    "speaking_rate": SpeakingRateEvaluator,
    "context": ContextEvaluator,
}


def validate_config(config: dict) -> None:
    if config.setdefault("prosody_metric", "position_f1_v2") != "position_f1_v2":
        raise ValueError("This version supports prosody_metric: position_f1_v2 only")
    dimensions = config.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
        raise ValueError("Evaluation config must explicitly configure all seven dimensions")
    if any(
        not isinstance(v, dict) or v.get("mode") not in {"disabled", "evidence"}
        for v in dimensions.values()
    ):
        raise ValueError("Evaluation mode must be disabled or evidence")
    threshold = config.get("content_max_error_rate", 0.1)
    if (
        type(threshold) not in {int, float}
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid content_max_error_rate")


def evaluate_sample(sample, evidence: dict, config: dict, *, simulated: bool = False) -> list[dict]:
    result = []
    for name in DIMENSIONS:
        if simulated:
            value = EvaluationResult(
                name, "skipped", reason="Simulated tone; not a benchmark result"
            )
        elif config["dimensions"][name]["mode"] == "disabled":
            value = EvaluationResult(
                name, "skipped", reason="Dimension disabled; metric not calibrated/configured"
            )
        else:
            try:
                value = EVALUATORS[name](config).evaluate(sample, evidence)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                value = EvaluationResult(name, "error", reason=str(exc))
        result.append(value.to_dict())
    return result
