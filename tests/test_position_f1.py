import pytest

from parabench.datasets.precollection import load_samples
from parabench.evaluators import evaluate_sample
from parabench.evaluators.position_f1 import position_metrics


@pytest.mark.parametrize(
    "gold,predicted,score",
    [
        (set(), set(), 1),
        ({1}, set(), 0),
        (set(), {1}, 0),
        ({1, 2}, {1, 2}, 1),
        ({1, 2}, {1, 3}, 0.5),
        ({1, 2}, set(range(10)), 1 / 3),
    ],
)
def test_position_counts(gold, predicted, score):
    value = position_metrics(gold, predicted)
    assert value["f1"] == pytest.approx(score)
    assert value["exact_match"] == (gold == predicted)


def score(sample, config, dimension, **prediction):
    evidence = {
        "dimensions": {
            dimension: {
                "method": "acoustic",
                "evaluator_id": "test-v2",
                "calibration_id": "fixture-v2",
                **prediction,
            }
        }
    }
    return next(r for r in evaluate_sample(sample, evidence, config) if r["dimension"] == dimension)


def test_whole_sentence_stress_is_not_perfect(release, eval_config):
    sample = load_samples(release[0])[0]  # canonical 你好再见; gold 你好.
    r = score(sample, eval_config, "stress", spans=[[0, 4]])
    assert r["score"] == pytest.approx(2 / 3)
    assert r["evidence"]["false_positive"] == 2
    assert not r["passed"]


def test_stress_grouping_and_duplicates_cannot_game_f1(release, eval_config):
    sample = load_samples(release[0])[0]
    one = score(sample, eval_config, "stress", spans=[[0, 2]])
    split = score(sample, eval_config, "stress", spans=[[0, 1], [1, 2], [0, 2], [0, 2]])
    assert one["score"] == split["score"] == 1
    partial = score(sample, eval_config, "stress", spans=[[0, 1]])
    assert partial["score"] == pytest.approx(2 / 3)
    assert not partial["passed"]


def test_extra_pause_and_missing_pause_both_penalized(release, eval_config):
    sample = load_samples(release[0])[0]
    sample.targets["pause"]["positions"] = [1]
    r = score(sample, eval_config, "pause", positions=[0, 1, 2, 3, 4, 3])
    assert r["score"] == pytest.approx(2 / 3)
    assert r["evidence"]["included_positions"] == [1, 3]
    assert r["evidence"]["false_positive"] == 1
    sample.targets["pause"]["positions"] = [1, 3]
    missing = score(sample, eval_config, "pause", positions=[1])
    assert missing["score"] == pytest.approx(2 / 3)
    assert missing["evidence"]["false_negative"] == 1


def test_explicit_none_does_not_get_credit_for_extras(release, eval_config):
    sample = load_samples(release[0])[0]
    sample.targets["stress"]["spans"] = []
    assert score(sample, eval_config, "stress", spans=[])["score"] == 1
    assert score(sample, eval_config, "stress", spans=[[0, 4]])["score"] == 0


def test_report_exports_precision_recall_and_metric_version(tmp_path):
    import csv

    from parabench.io import sha256_file, write_json, write_jsonl
    from parabench.reporting import build_report
    from parabench.schemas import DIMENSIONS

    run = tmp_path / "report-fixture"
    run.mkdir()
    metrics = position_metrics({0, 1}, {0, 1, 2, 3})
    results = [
        {
            "dimension": k,
            "status": "scored" if k == "stress" else "skipped",
            "score": metrics["f1"] if k == "stress" else None,
            "passed": False if k == "stress" else None,
            "method": "acoustic" if k == "stress" else None,
            "reason": None,
            "evidence": metrics if k == "stress" else {},
        }
        for k in DIMENSIONS
    ]
    write_jsonl(
        run / "evaluations.jsonl",
        [
            {
                "sample_id": "fixture",
                "language": "zh",
                "generation_status": "success",
                "simulated": False,
                "results": results,
            }
        ],
    )
    write_json(
        run / "evaluation_provenance.json",
        {
            "evaluations_sha256": sha256_file(run / "evaluations.jsonl"),
            "generation_record_hashes": {},
        },
    )
    report = build_report(run)
    summary = next(r for r in report["scores"] if r["scope"] == "zh" and r["dimension"] == "stress")
    assert summary["mean_precision"] == 0.5 and summary["mean_recall"] == 1
    assert summary["mean_score"] == pytest.approx(2 / 3)
    with (run / "details.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["dimension"] == "stress")
    assert row["precision"] == "0.5" and row["false_positive"] == "2"
    assert row["metric_version"] == "prosody-position-f1-v2"
