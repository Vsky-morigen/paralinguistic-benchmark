from parabench.datasets.annotations import canonical
from parabench.evaluators.position_f1 import expand_spans, position_metrics
from parabench.schemas import DIMENSIONS, EvaluationResult
from parabench.tasks import label_in_language


def validate_config(config: dict) -> None:
    if config.get("prosody_metric", "position_f1_v2") != "position_f1_v2":
        raise ValueError("Only position_f1_v2 is supported")
    dimensions = config.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
        raise ValueError("Configure all six dimensions explicitly")
    if any(
        not isinstance(v, dict) or type(v.get("enabled")) is not bool for v in dimensions.values()
    ):
        raise ValueError("Each dimension needs enabled: true or false")


def unavailable_reason(sample, name: str) -> str | None:
    if name in {"stress", "pause"} and sample.targets[name]["status"] == "unknown":
        return "Reference annotation is unknown"
    if name == "speaking_rate" and not sample.targets["speed_verified"]:
        return "Reference speaking rate is not validated"
    if name in {"gender", "emotion", "context"} and sample.targets[name] in {
        "无法判断",
        "听不清",
        "不确定",
    }:
        return "Reference label is unknown"
    return None


def evaluate_answer(sample, name: str, answer, choices: list[str]) -> EvaluationResult:
    method = "position_f1" if name in {"stress", "pause"} else "label_match"
    if name not in {"stress", "pause"}:
        target = label_in_language(
            sample.targets["speed" if name == "speaking_rate" else name], sample.language
        )
        valid = isinstance(answer, str) and answer.strip() in choices
        matched = valid and answer.strip() == target
        return EvaluationResult(
            name,
            "scored",
            float(matched),
            bool(matched),
            method,
            None if valid else "Missing answer or label outside the supplied choices",
            {"prediction": answer, "reference": target, "valid_answer": valid},
        )
    n = len(canonical(sample.text))
    if name == "stress":
        valid = isinstance(answer, list) and all(
            isinstance(span, list)
            and len(span) == 2
            and all(type(v) is int for v in span)
            and 0 <= span[0] < span[1] <= n
            for span in answer
        )
        predicted = expand_spans(answer) if valid else set()
        gold = expand_spans(sample.targets[name]["spans"])
    else:
        valid = isinstance(answer, list) and all(type(v) is int and 0 <= v <= n for v in answer)
        excluded = set(sample.provenance.get("sentence_boundaries", [])) | {0, n}
        predicted = set(answer) - excluded if valid else set()
        gold = set(sample.targets[name]["positions"]) - excluded
    metrics = position_metrics(gold, predicted)
    # Invalid output is not equivalent to a valid empty prediction.
    if not valid:
        metrics.update(precision=0.0, recall=0.0, f1=0.0, exact_match=False)
    return EvaluationResult(
        name,
        "scored",
        metrics["f1"],
        metrics["exact_match"],
        method,
        None if valid else "Missing answer or invalid position format",
        {
            **metrics,
            "prediction": answer,
            "valid_answer": valid,
            "included_positions": sorted(predicted),
            "position_unit": "canonical_character" if name == "stress" else "canonical_boundary",
        },
    )
