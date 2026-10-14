import asyncio
import copy

import pytest

from parabench.config import model_config
from parabench.datasets.precollection import load_samples
from parabench.evaluators import evaluate_answer
from parabench.evaluators.position_f1 import position_metrics
from parabench.io import read_json, write_json
from parabench.reporting import build_report
from parabench.runner.evaluation import evaluate_run
from parabench.runner.execution import predict_run


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
    assert position_metrics(gold, predicted)["f1"] == pytest.approx(score)


def test_full_sentence_stress_and_partial_match(release):
    sample = load_samples(release[0])[0]
    result = evaluate_answer(sample, "stress", [[0, 4]], [])
    assert result.score == pytest.approx(2 / 3)
    assert result.evidence["false_positive"] == 2 and not result.passed
    assert evaluate_answer(sample, "stress", [[0, 1]], []).score == pytest.approx(2 / 3)
    assert evaluate_answer(sample, "stress", [[0, 1], [1, 2], [0, 2]], []).score == 1


def test_extra_pause_and_sentence_boundaries(release):
    sample = load_samples(release[0])[0]
    sample.targets["pause"]["positions"] = [1]
    result = evaluate_answer(sample, "pause", [0, 1, 2, 3, 4, 3], [])
    assert result.score == pytest.approx(2 / 3)
    assert result.evidence["included_positions"] == [1, 3]
    assert result.evidence["false_positive"] == 1


@pytest.mark.parametrize("answer", [None, "[]", {}, [True], [[0, 99]]])
def test_invalid_stress_is_zero_even_when_reference_empty(release, answer):
    sample = load_samples(release[0])[0]
    sample.targets["stress"]["spans"] = []
    assert evaluate_answer(sample, "stress", answer, []).score == 0
    assert evaluate_answer(sample, "stress", [], []).score == 1


@pytest.mark.parametrize("answer", [None, [], ["男声", "女声"], "unknown"])
def test_missing_or_all_labels_cannot_score(release, answer):
    sample = load_samples(release[0])[0]
    assert evaluate_answer(sample, "gender", answer, ["男声", "女声"]).score == 0
    assert evaluate_answer(sample, "gender", "男声", ["男声", "女声"]).score == 1


def test_english_label_mapping(release):
    sample = copy.deepcopy(load_samples(release[0])[0])
    object.__setattr__(sample, "language", "en")
    result = evaluate_answer(
        sample, "gender", "male-sounding voice", ["male-sounding voice", "female-sounding voice"]
    )
    assert result.score == 1


def test_scoring_uses_saved_model_answers_without_external_evidence(
    release, local_setup, eval_config, tmp_path
):
    root, path, _ = local_setup
    asyncio.run(predict_run(*release[:2], model_config(path), tmp_path, "local", model_root=root))
    run = tmp_path / "local"
    result = evaluate_run(run, eval_config)
    assert result["scored"] == 15 and result["errors"] == 0
    summary = build_report(run)
    gender = next(
        g for g in summary["scores"] if g["scope"] == "all" and g["dimension"] == "gender"
    )
    assert gender["mean_score"] == 1 and gender["eligible"] == 3
    assert "副语言理解" in (run / "report.md").read_text(encoding="utf-8")
    assert (run / "details.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_failed_inference_stays_in_main_denominator(release, local_setup, eval_config, tmp_path):
    root, path, _ = local_setup
    asyncio.run(predict_run(*release[:2], model_config(path), tmp_path, "local", model_root=root))
    run = tmp_path / "local"
    record_path = run / "records/AUDIO_001.json"
    record = read_json(record_path)
    record["dimensions"]["gender"] = {"status": "failed"}
    write_json(record_path, record)
    assert evaluate_run(run, eval_config)["errors"] == 1
    gender = next(
        g for g in build_report(run)["scores"] if g["scope"] == "all" and g["dimension"] == "gender"
    )
    assert gender["eligible"] == 3 and gender["scored"] == 2
    assert gender["mean_score"] == pytest.approx(2 / 3) and gender["completed_mean_score"] == 1


def test_tampered_prediction_or_audio_is_rejected(release, local_setup, eval_config, tmp_path):
    root, path, _ = local_setup
    asyncio.run(predict_run(*release[:2], model_config(path), tmp_path, "local", model_root=root))
    run = tmp_path / "local"
    evaluate_run(run, eval_config)
    record_path = run / "records/AUDIO_001.json"
    record = read_json(record_path)
    record["dimensions"]["gender"]["prediction"]["answer"] = "女声"
    write_json(record_path, record)
    with pytest.raises(ValueError, match="Prediction changed"):
        build_report(run)
    assert evaluate_run(run, eval_config)["errors"] == 1
    (run / "inputs/AUDIO_001.wav").write_bytes(b"tamper")
    assert evaluate_run(run, eval_config)["errors"] == 5


def test_generation_runs_are_not_scored_as_understanding(tmp_path, eval_config):
    write_json(tmp_path / "run.json", {"schema_version": 1, "spec": {"task": "generation"}})
    with pytest.raises(ValueError, match="old generation"):
        evaluate_run(tmp_path, eval_config)
