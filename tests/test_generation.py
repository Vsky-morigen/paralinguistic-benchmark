import asyncio
import base64
import io
import json
import urllib.error
from dataclasses import asdict

import pytest
from conftest import wav_bytes

from parabench.datasets.precollection import load_samples
from parabench.io import read_json
from parabench.models.base import ModelError
from parabench.models.http_adapter import HttpAdapter
from parabench.runner.execution import generate_run
from parabench.tasks import build_request


def test_resume_regenerates_missing_audio_and_rejects_config_change(release, mock_config, tmp_path):
    root = tmp_path / "results"
    first = asyncio.run(generate_run(release[0], mock_config, root, "test"))
    assert first["generated"] == 3
    second = asyncio.run(generate_run(release[0], mock_config, root, "test", resume=True))
    assert second["resumed"] == 3
    (root / "test/audio/AUDIO_001.wav").unlink()
    third = asyncio.run(generate_run(release[0], mock_config, root, "test", resume=True))
    assert third["generated"] == 1 and third["resumed"] == 2
    with pytest.raises(ValueError, match="Resume rejected"):
        asyncio.run(
            generate_run(
                release[0], {**mock_config, "model": "different"}, root, "test", resume=True
            )
        )


def test_retry_only_transient_errors(release, mock_config, tmp_path, monkeypatch):
    from parabench.models.mock import MockAdapter

    original = MockAdapter.generate
    calls = {}

    async def flaky(self, request, path):
        calls[request.sample_id] = calls.get(request.sample_id, 0) + 1
        if calls[request.sample_id] == 1:
            raise ModelError("temporary", retryable=True)
        return await original(self, request, path)

    monkeypatch.setattr(MockAdapter, "generate", flaky)
    result = asyncio.run(generate_run(release[0], mock_config, tmp_path / "runs", "retry"))
    assert result["generated"] == 3 and all(v == 2 for v in calls.values())

    async def invalid(self, request, path):
        raise ModelError("permanent", retryable=False)

    monkeypatch.setattr(MockAdapter, "generate", invalid)
    result = asyncio.run(generate_run(release[0], mock_config, tmp_path / "runs", "fail"))
    assert result["failed"] == 3
    record = read_json(tmp_path / "runs/fail/records/AUDIO_001.json")
    assert len(record["attempts"]) == 1


def test_http_contract_and_secret_isolation(release, tmp_path, monkeypatch):
    config = {
        "endpoint": "http://localhost:8000/generate",
        "model": "test",
        "timeout_seconds": 1,
        "api_key_env": "TEST_PARABENCH_KEY",
    }
    monkeypatch.setenv("TEST_PARABENCH_KEY", "secret-fixture-value")
    adapter = HttpAdapter(config)
    request = build_request(load_samples(release[0])[0], {})
    response = json.dumps(
        {
            "format": "wav",
            "sample_rate": 16000,
            "audio_base64": base64.b64encode(wav_bytes()).decode(),
            "usage": {"input_tokens": 12, "api_key": "do-not-store"},
        }
    ).encode()

    def fake_open(req, timeout):
        body = json.loads(req.data)
        assert body["request_id"] == request.sample_id
        assert body["input"]["text"] == request.text
        assert set(body) == {"request_id", "model", "input", "generation"}
        assert req.get_header("Authorization") == "Bearer secret-fixture-value"
        return io.BytesIO(response)

    monkeypatch.setattr(adapter.opener, "open", fake_open)
    result = asyncio.run(adapter.generate(request, tmp_path / "out.wav"))
    assert result.sample_rate == 16000
    assert result.usage == {"input_tokens": 12}
    assert "secret-fixture-value" not in json.dumps(asdict(result))


@pytest.mark.parametrize("status,retryable", [(401, False), (429, True), (503, True), (302, False)])
def test_http_error_codes_do_not_leak_body(status, retryable, release, tmp_path, monkeypatch):
    adapter = HttpAdapter(
        {"endpoint": "http://localhost/generate", "model": "test", "timeout_seconds": 1}
    )
    request = build_request(load_samples(release[0])[0], {})

    def failing(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://localhost", status, "secret", {}, io.BytesIO(b"secret")
        )

    monkeypatch.setattr(adapter.opener, "open", failing)
    with pytest.raises(ModelError) as exc:
        asyncio.run(adapter.generate(request, tmp_path / "out.wav"))
    assert exc.value.retryable == retryable and "secret" not in str(exc.value)


def test_changed_sample_snapshot_blocks_resume(release, mock_config, tmp_path):
    root = tmp_path / "runs"
    asyncio.run(generate_run(release[0], mock_config, root, "test"))
    path = root / "test/samples.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        asyncio.run(generate_run(release[0], mock_config, root, "test", resume=True))


def test_empty_selection_and_invalid_run_id(release, mock_config, tmp_path):
    with pytest.raises(ValueError, match="run ID"):
        asyncio.run(generate_run(release[0], mock_config, tmp_path, "../escape"))
    with pytest.raises(ValueError, match="No samples"):
        asyncio.run(generate_run(release[0], mock_config, tmp_path, "test", language="en"))
