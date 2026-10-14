import asyncio
import copy
import shutil

import pytest

from parabench.config import load_yaml, model_config, resolve_dataset
from parabench.datasets.precollection import load_samples
from parabench.io import read_json, sha256_file, write_json
from parabench.models.base import AudioModelAdapter, ModelError
from parabench.models.registry import ADAPTERS
from parabench.reporting import build_report
from parabench.runner.evaluation import evaluate_run
from parabench.runner.execution import predict_run
from parabench.schemas import DIMENSIONS, PredictionResult
from parabench.tasks import build_request, load_choices, plain_transcript


def test_only_position_tasks_receive_plain_transcript(release):
    labels, audio, *_ = release
    sample = load_samples(labels)[0]
    catalog = load_choices(labels)
    for name in DIMENSIONS:
        request = build_request(sample, name, audio / sample.reference_audio, catalog, {})
        payload = request.to_payload()
        assert ("transcript" in payload) == (name in {"stress", "pause"})
        if "transcript" in payload:
            assert payload["transcript"] == "你好再见"
        assert (
            not {"targets", "reference_audio", "audio_path", "provenance", "annotation_decision"}
            & payload.keys()
        )
        prior = copy.deepcopy(sample.targets)
        sample.targets.update(
            gender="other",
            emotion="other",
            context="other",
            stress={"spans": [[1, 2]]},
            pause={"positions": [1]},
        )
        assert build_request(sample, name, request.audio_path, catalog, {}).to_payload() == payload
        sample.targets.clear()
        sample.targets.update(prior)


def test_plain_transcript_removes_punctuation_and_marks():
    assert plain_transcript("Ａ【test】｜，Next！") == "AtestNext"
    assert plain_transcript("I 【want】 this｜ now.") == "I want this now"


def test_mock_prediction_and_resume_use_existing_audio(release, mock_config, eval_config, tmp_path):
    labels, audio, *_ = release
    before = {p.name: sha256_file(p) for p in audio.glob("*.wav")}
    result = asyncio.run(predict_run(labels, audio, mock_config, tmp_path / "runs", "mock"))
    assert result["predicted"] == 18 and result["failed"] == 0
    run = tmp_path / "runs/mock"
    assert sha256_file(run / "inputs/AUDIO_001.wav") == before["AUDIO_001.wav"]
    second = asyncio.run(
        predict_run(labels, audio, mock_config, tmp_path / "runs", "mock", resume=True)
    )
    assert second["resumed"] == 18 and second["predicted"] == 0
    assert evaluate_run(run, eval_config)["scored"] == 0
    assert all(g["mean_score"] is None for g in build_report(run)["scores"])
    assert before == {p.name: sha256_file(p) for p in audio.glob("*.wav")}


def test_resume_repairs_only_changed_prediction(release, mock_config, tmp_path):
    labels, audio, *_ = release
    asyncio.run(predict_run(labels, audio, mock_config, tmp_path, "run"))
    record_path = tmp_path / "run/records/AUDIO_001.json"
    record = read_json(record_path)
    record["dimensions"]["gender"]["prediction"]["answer"] = "changed"
    write_json(record_path, record)
    result = asyncio.run(predict_run(labels, audio, mock_config, tmp_path, "run", resume=True))
    assert result["predicted"] == 1 and result["resumed"] == 17


def test_changed_source_audio_rejected_before_inference(release, mock_config, tmp_path):
    labels, audio, *_ = release
    (audio / "AUDIO_001.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        asyncio.run(predict_run(labels, audio, mock_config, tmp_path, "run"))


def test_local_reads_audio_and_loads_once(release, local_setup, tmp_path):
    root, config_path, log = local_setup
    config = model_config(config_path)
    result = asyncio.run(predict_run(*release[:2], config, tmp_path, "local", model_root=root))
    assert result["predicted"] == 18
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls.count("load:cpu") == calls.count("close") == 1
    assert calls.count("predict:stress:True") == calls.count("predict:pause:True") == 3
    assert calls.count("predict:gender:False") == calls.count("predict:context:False") == 3
    saved = read_json(tmp_path / "local/run.json")["spec"]["model_config"]
    assert saved["model_path"] == "weights" and saved["loader"] == "loader.py:load_model"


def test_local_changes_reject_resume(release, local_setup, tmp_path):
    root, path, _ = local_setup
    asyncio.run(predict_run(*release[:2], model_config(path), tmp_path, "local", model_root=root))
    (root / "weights/weights.json").write_text('{"label":"女声"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Resume rejected"):
        asyncio.run(
            predict_run(
                *release[:2], model_config(path), tmp_path, "local", model_root=root, resume=True
            )
        )


def test_local_module_loader(release, local_setup, tmp_path, monkeypatch):
    root, path, log = local_setup
    shutil.copyfile(root / "loader.py", root / "fixture_loader.py")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "loader.py:load_model", "fixture_loader:load_model"
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(root))
    result = asyncio.run(
        predict_run(*release[:2], model_config(path), tmp_path, "module", model_root=root)
    )
    assert result["predicted"] == 18 and log.read_text().count("load:cpu") == 1


def test_local_failure_does_not_leak_or_fallback(release, local_setup, tmp_path):
    root, path, _ = local_setup
    (root / "loader.py").write_text(
        "def load_model(**kwargs):\n    raise RuntimeError('private-token')\n", encoding="utf-8"
    )
    result = asyncio.run(
        predict_run(*release[:2], model_config(path), tmp_path, "bad", model_root=root)
    )
    assert result["failed"] == 18 and result["predicted"] == 0
    assert "private-token" not in (tmp_path / "bad/records/AUDIO_001.json").read_text()


def test_transient_retry_and_permanent_error(release, mock_config, tmp_path, monkeypatch):
    class Flaky(AudioModelAdapter):
        attempts = 0

        async def predict(self, request):
            Flaky.attempts += 1
            if Flaky.attempts == 1:
                raise ModelError("HTTP 429", retryable=True)
            return PredictionResult("男声", 0, self.config["model"])

    monkeypatch.setitem(ADAPTERS, "http", Flaky)
    config = {**mock_config, "adapter": "http"}
    result = asyncio.run(
        predict_run(*release[:2], config, tmp_path, "retry", dimensions=("gender",))
    )
    assert result["predicted"] == 3 and Flaky.attempts == 4


def test_relocated_run_can_be_scored(release, local_setup, eval_config, tmp_path):
    root, path, _ = local_setup
    asyncio.run(
        predict_run(*release[:2], model_config(path), tmp_path / "before", "local", model_root=root)
    )
    moved = tmp_path / "other machine" / "local"
    shutil.copytree(tmp_path / "before/local", moved)
    assert evaluate_run(moved, eval_config)["scored"] == 15
    assert build_report(moved)["samples"] == 3


def test_disabled_dimensions_do_not_call_model(release, mock_config, tmp_path):
    result = asyncio.run(
        predict_run(*release[:2], mock_config, tmp_path, "one", dimensions=("gender",))
    )
    assert result["predicted"] == 3 and result["requests"] == 3


def test_config_rejects_old_generation_mode_and_parallel_local(local_setup):
    root, path, _ = local_setup
    value = load_yaml(path)
    import yaml

    value["concurrency"] = 2
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        model_config(path)
    value["concurrency"] = 1
    value["generation"] = {}
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="old generation"):
        model_config(path)


def test_dataset_paths_follow_config_after_move(release, tmp_path, monkeypatch):
    moved = tmp_path / "moved"
    (moved / "configs").mkdir(parents=True)
    shutil.copytree(release[0], moved / "data/metadata")
    shutil.copytree(release[1], moved / "data/audio")
    path = moved / "configs/dataset.yaml"
    path.write_text(
        "schema_version: 1\nrelease: data/metadata\nreference_root: data/audio\n", encoding="utf-8"
    )
    monkeypatch.delenv("PARABENCH_REFERENCE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_dataset(path) == (moved / "data/metadata", moved / "data/audio")
