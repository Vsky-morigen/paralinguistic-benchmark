"""Set-based position scoring. Prediction grouping never changes the score."""

METRIC_VERSION = "prosody-position-f1-v2"


def position_metrics(gold: set[int], predicted: set[int]) -> dict:
    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    if not gold and not predicted:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "metric_version": METRIC_VERSION,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "exact_match": gold == predicted,
    }


def expand_spans(spans: list[list[int]]) -> set[int]:
    return {i for start, end in spans for i in range(start, end)}
