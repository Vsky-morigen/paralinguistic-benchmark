import re

from parabench.datasets.annotations import canonical
from parabench.evaluators.base import Evaluator
from parabench.schemas import EvaluationResult, Sample


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class ContentEvaluator(Evaluator):
    dimension = "content"

    def evaluate(self, sample: Sample, evidence: dict) -> EvaluationResult:
        item = self.evidence_for(evidence)
        if item is None:
            return self.skip("An independent ASR transcript of the generated audio is required")
        if item["method"] != "independent_asr":
            raise ValueError(
                "Content scoring requires independent_asr, not the model's own transcript"
            )
        text = item.get("transcript")
        if not isinstance(text, str) or len(text) > 20000:
            raise ValueError("Invalid independent transcript")
        if sample.language == "zh":
            reference, observed = list(canonical(sample.text)), list(canonical(text))
            metric = "cer"
        else:
            reference = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", sample.text.lower())
            observed = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
            metric = "wer"
        if not reference:
            raise ValueError("Reference transcript has no tokens")
        rate = edit_distance(reference, observed) / len(reference)
        threshold = self.config.get("content_max_error_rate", 0.1)
        return EvaluationResult(
            "content",
            "scored",
            max(0.0, 1 - rate),
            rate <= threshold,
            "independent_asr",
            evidence={**item, metric: rate, "threshold": threshold},
        )
