from parabench.datasets.annotations import canonical
from parabench.evaluators.base import Evaluator
from parabench.evaluators.position_f1 import position_metrics
from parabench.schemas import EvaluationResult, Sample


class PauseEvaluator(Evaluator):
    dimension = "pause"

    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        target = sample.targets["pause"]
        if target["status"] == "unknown":
            return self.skip("Reference pause annotation is unknown")
        item = self.evidence_for(evidence)
        if item is None:
            return self.skip("Calibrated silence extraction/forced alignment is not configured")
        if item["method"] != "acoustic":
            raise ValueError("Pause scorer accepts calibrated acoustic evidence")
        observed = item.get("positions")
        n = len(canonical(sample.text))
        if not isinstance(observed, list) or any(
            type(p) is not int or not 0 <= p <= n for p in observed
        ):
            raise ValueError("Invalid aligned pause positions")
        excluded = set(sample.provenance.get("sentence_boundaries", [])) | {0, n}
        predicted = set(observed) - excluded
        gold = set(target["positions"])
        metrics = position_metrics(gold, predicted)
        return EvaluationResult(
            "pause",
            "scored",
            metrics["f1"],
            metrics["exact_match"],
            "acoustic",
            evidence={
                **item,
                **metrics,
                "included_positions": sorted(predicted),
                "position_unit": "canonical_boundary",
                "extra_positions_penalized": True,
            },
        )
