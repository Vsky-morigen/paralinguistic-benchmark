import asyncio
import json
import math

import pytest

from parabench.audio import inspect_wav
from parabench.config import model_config
from parabench.models.local import LocalAudio, save_local_audio
from parabench.runner.execution import generate_run


def test_local_loads_once_and_reuses_checkpoint(release, local_setup, tmp_path):
    root, path, log = local_setup
    config = model_config(path)
    result = asyncio.run(
        generate_run(release[0], config, tmp_path / "runs", "local", model_root=root)
    )
    assert result["generated"] == 3 and result["failed"] == 0
    calls = log.read_text().splitlines()
    assert calls.count("load:cpu") == 1 and calls.count("close") == 1
    assert len([x for x in calls if x.startswith("generate:")]) == 3
    assert all(x.endswith(":True") for x in calls if x.startswith("generate:"))
    assert inspect_wav(tmp_path / "runs/local/audio/AUDIO_001.wav")["sample_rate"] == 16000


def test_local_weight_change_rejects_resume(release, local_setup, tmp_path):
    root, path, _ = local_setup
    config = model_config(path)
    asyncio.run(generate_run(release[0], config, tmp_path / "runs", "local", model_root=root))
    (root / "weights/weights.json").write_text('{"amplitude":0.123456}', encoding="utf-8")
    with pytest.raises(ValueError, match="Resume rejected"):
        asyncio.run(
            generate_run(
                release[0],
                model_config(path),
                tmp_path / "runs",
                "local",
                resume=True,
                model_root=root,
            )
        )


def test_local_rejects_missing_weights_and_parallel_inference(local_setup):
    root, path, _ = local_setup
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("concurrency: 1", "concurrency: 2"), encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        model_config(path)
    path.write_text(text.replace("model_path: weights", "model_path: missing"), encoding="utf-8")
    with pytest.raises(ValueError, match="weights do not exist"):
        model_config(path)


def test_local_module_loader_does_not_require_registry_edit(
    release, local_setup, tmp_path, monkeypatch
):
    root, path, log = local_setup
    (root / "public_fixture_loader.py").write_text(
        (root / "loader.py").read_text(), encoding="utf-8"
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "loader.py:load_model", "public_fixture_loader:load_model"
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(root))
    result = asyncio.run(
        generate_run(
            release[0], model_config(path), tmp_path / "module_runs", "module", model_root=root
        )
    )
    assert result["generated"] == 3 and log.read_text().count("load:cpu") == 1


@pytest.mark.parametrize("samples", [[math.nan], [math.inf], [1.1], [[0.1]], []])
def test_bad_local_samples_do_not_get_written(samples, tmp_path):
    with pytest.raises(ValueError):
        save_local_audio(LocalAudio(sample_rate=16000, samples=samples), tmp_path / "bad.wav")


def test_local_wav_return_and_sample_rate_check(tmp_path):
    source = tmp_path / "source.wav"
    save_local_audio(LocalAudio(sample_rate=16000, samples=[0.1] * 160), source)
    output = tmp_path / "copy.wav"
    save_local_audio(LocalAudio(sample_rate=16000, audio_path=source), output)
    assert source.read_bytes() == output.read_bytes()
    with pytest.raises(ValueError, match="rate mismatch"):
        save_local_audio(LocalAudio(sample_rate=8000, audio_path=source), output)


def test_loader_error_is_permanent_and_no_mock_fallback(release, local_setup, tmp_path):
    root, path, _ = local_setup
    (root / "loader.py").write_text(
        "def load_model(**kwargs):\n    raise RuntimeError('do-not-leak-secret')\n",
        encoding="utf-8",
    )
    result = asyncio.run(
        generate_run(release[0], model_config(path), tmp_path / "runs", "failure", model_root=root)
    )
    assert result["failed"] == 3 and result["generated"] == 0
    record = json.loads((tmp_path / "runs/failure/records/AUDIO_001.json").read_text())
    assert len(record["attempts"]) == 1 and "do-not-leak-secret" not in str(record)
