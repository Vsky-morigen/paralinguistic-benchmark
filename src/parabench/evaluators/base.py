from abc import ABC, abstractmethod

from parabench.schemas import EvaluationResult, Sample

METHODS = {"independent_asr", "classifier", "human_blind", "audio_model_judge", "acoustic"}


class Evaluator(ABC):
    dimension: str

    def __init__(self, config: dict):
        self.config = config

    def skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(self.dimension, "skipped", reason=reason)

    def evidence_for(self, evidence: dict) -> dict | None:
        item = evidence.get("dimensions", {}).get(self.dimension)
        if item is None:
            return None
        if not isinstance(item, dict) or item.get("method") not in METHODS:
            raise ValueError("Unknown evidence method")
        if not isinstance(item.get("evaluator_id"), str) or not item["evaluator_id"].strip():
            raise ValueError("External evidence needs evaluator_id/version")
        if item["method"] in {"classifier", "acoustic"} and not item.get("calibration_id"):
            raise ValueError("Objective proxy evidence needs calibration_id")
        return item

    @abstractmethod
    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        """Return scored/skipped/error; missing evidence is never a zero score."""


class LabelEvaluator(Evaluator):
    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        item = self.evidence_for(evidence)
        if item is None:
            return self.skip("No external audio classification/blind-listening evidence")
        target = sample.targets[self.dimension]
        if target in {"无法判断", "听不清", "不确定"}:
            return self.skip("Reference target is unknown")
        if item["method"] == "human_blind":
            votes = item.get("votes")
            if not isinstance(votes, list) or len(votes) < 2:
                raise ValueError("human_blind needs at least two independent votes")
            ids = [v["rater_id"] for v in votes]
            if len(set(ids)) != len(ids) or not all(isinstance(x, str) and x for x in ids):
                raise ValueError("Rater IDs must be nonempty and unique")
            matches = sum(v["label"] == target for v in votes)
            score, passed = matches / len(votes), matches >= 2
        else:
            if item["method"] not in {"classifier", "audio_model_judge"}:
                raise ValueError("This label dimension needs classifier/judge/human_blind evidence")
            if not isinstance(item.get("label"), str):
                raise ValueError("Missing observed label")
            passed = item["label"] == target
            score = float(passed)
        return EvaluationResult(
            self.dimension, "scored", score, passed, item["method"], evidence=item
        )
