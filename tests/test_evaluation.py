import asyncio

import pytest

from parabench.datasets.precollection import load_samples
from parabench.evaluators import evaluate_sample
from parabench.evaluators.content import edit_distance
from parabench.io import read_json, read_jsonl, write_json, write_jsonl
from parabench.reporting import build_report
from parabench.runner.evaluation import evaluate_run
from parabench.runner.execution import generate_run


def test_mock_never_gets_a_benchmark_score(release, mock_config, eval_config, tmp_path):
    asyncio.run(generate_run(release[0], mock_config, tmp_path, "mock"))
    run = tmp_path / "mock"
    assert evaluate_run(run, eval_config)["scored"] == 0
    report = build_report(run)
    assert report["simulated_samples"] == 3 and report["overall_score"] is None
    assert all(x["mean_score"] is None for x in report["scores"])


def test_content_and_missing_evidence(release, eval_config):
    sample = load_samples(release[0])[0]
    assert edit_distance(list("abc"), list("axc")) == 1
    ev = {
        "dimensions": {
            "content": {
                "method": "independent_asr",
                "evaluator_id": "asr-test-v1",
                "transcript": sample.text,
            }
        }
    }
    result = evaluate_sample(sample, ev, eval_config)
    assert result[0]["score"] == 1 and result[0]["passed"]
    assert all(r["status"] == "skipped" for r in result[1:])


def test_pause_boundary_exclusion_and_original_none(release, eval_config):
    sample = load_samples(release[0])[0]
    ev = {
        "dimensions": {
            "pause": {
                "method": "acoustic",
                "evaluator_id": "aligner-v1",
                "calibration_id": "calibration-v1",
                "positions": [0, 2, 4],
            }
        }
    }
    result = evaluate_sample(sample, ev, eval_config)
    pause = next(r for r in result if r["dimension"] == "pause")
    assert pause["passed"] and pause["score"] == 1
    ev["dimensions"]["pause"]["positions"].append(1)
    pause = next(r for r in evaluate_sample(sample, ev, eval_config) if r["dimension"] == "pause")
    assert not pause["passed"]


def test_stress_extra_positions_reduce_precision_and_f1(release, eval_config):
    sample = load_samples(release[0])[0]
    ev = {
        "dimensions": {
            "stress": {
                "method": "acoustic",
                "evaluator_id": "extractor-v1",
                "calibration_id": "calibration-v1",
                "spans": [[0, 2], [2, 4]],
            }
        }
    }
    stress = next(r for r in evaluate_sample(sample, ev, eval_config) if r["dimension"] == "stress")
    assert not stress["passed"]
    assert stress["score"] == pytest.approx(2 / 3)
    assert stress["evidence"]["precision"] == 0.5
    assert stress["evidence"]["recall"] == 1
    ev["dimensions"]["stress"].pop("calibration_id")
    stress = next(r for r in evaluate_sample(sample, ev, eval_config) if r["dimension"] == "stress")
    assert stress["status"] == "error" and stress["score"] is None


def test_human_votes_need_distinct_raters(release, eval_config):
    sample = load_samples(release[0])[0]
    ev = {
        "dimensions": {
            "emotion": {
                "method": "human_blind",
                "evaluator_id": "study-v1",
                "votes": [{"rater_id": "a", "label": "中性"}, {"rater_id": "a", "label": "中性"}],
            }
        }
    }
    emotion = next(
        r for r in evaluate_sample(sample, ev, eval_config) if r["dimension"] == "emotion"
    )
    assert emotion["status"] == "error"


def test_evidence_must_match_generated_audio(release, mock_config, eval_config, tmp_path):
    asyncio.run(generate_run(release[0], mock_config, tmp_path, "run"))
    evidence = tmp_path / "evidence.jsonl"
    write_jsonl(evidence, [{"sample_id": "AUDIO_001", "audio_sha256": "wrong", "dimensions": {}}])
    result = evaluate_run(tmp_path / "run", eval_config, evidence)
    assert result["errors"] == 7


def test_reporting_rejects_stale_generation(release, mock_config, eval_config, tmp_path):
    asyncio.run(generate_run(release[0], mock_config, tmp_path, "run"))
    run = tmp_path / "run"
    evaluate_run(run, eval_config)
    path = run / "records/AUDIO_001.json"
    value = read_json(path)
    value["status"] = "failed"
    write_json(path, value)
    with pytest.raises(ValueError, match="Generation changed"):
        build_report(run)


def test_modified_audio_is_an_error_not_a_skip(release, mock_config, eval_config, tmp_path):
    asyncio.run(generate_run(release[0], mock_config, tmp_path, "run"))
    run = tmp_path / "run"
    (run / "audio/AUDIO_001.wav").write_bytes(b"broken")
    assert evaluate_run(run, eval_config)["errors"] == 7
    first = read_jsonl(run / "evaluations.jsonl")[0]
    assert all(r["status"] == "error" for r in first["results"])
