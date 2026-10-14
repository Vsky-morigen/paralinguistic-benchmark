from parabench.datasets.annotations import canonical
from parabench.evaluators.base import Evaluator
from parabench.evaluators.position_f1 import expand_spans, position_metrics
from parabench.schemas import EvaluationResult, Sample


class StressEvaluator(Evaluator):
    dimension = "stress"

    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        target = sample.targets["stress"]
        if target["status"] == "unknown":
            return self.skip("Reference stress annotation is unknown")
        item = self.evidence_for(evidence)
        if item is None:
            return self.skip("Calibrated acoustic stress extraction/alignment is not configured")
        if item["method"] != "acoustic":
            raise ValueError("Stress scorer accepts calibrated acoustic evidence")
        observed = item.get("spans")
        n = len(canonical(sample.text))
        if not isinstance(observed, list) or any(
            not isinstance(s, list)
            or len(s) != 2
            or any(type(v) is not int for v in s)
            or not 0 <= s[0] < s[1] <= n
            for s in observed
        ):
            raise ValueError("Invalid aligned stress spans")
        # Expand every span to canonical character positions. One full-sentence
        # interval incurs the same false positives as marking every character.
        metrics = position_metrics(expand_spans(target["spans"]), expand_spans(observed))
        return EvaluationResult(
            "stress",
            "scored",
            metrics["f1"],
            metrics["exact_match"],
            "acoustic",
            evidence={
                **item,
                **metrics,
                "position_unit": "canonical_character",
                "extra_positions_penalized": True,
            },
        )
