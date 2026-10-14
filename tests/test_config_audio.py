import math

import pytest
from conftest import wav_bytes

from parabench.audio import inspect_wav
from parabench.config import load_yaml, model_config
from parabench.models.http_adapter import HttpAdapter


def test_yaml_safe_loader_and_unknown_adapter(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    import yaml

    with pytest.raises(yaml.YAMLError):
        load_yaml(p)
    p.write_text("schema_version: 1\nadapter: typo\nmodel: test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        model_config(p)


@pytest.mark.parametrize(
    "url", ["file:///tmp/out", "http://user:secret@example.com", "http://x/?key=secret"]
)
def test_endpoint_credentials_and_non_http_urls_are_rejected(url):
    with pytest.raises(ValueError, match="endpoint"):
        HttpAdapter({"endpoint": url})


def test_wav_validation_detects_truncation(tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(wav_bytes())
    assert math.isclose(inspect_wav(p)["duration_seconds"], 0.1)
    p.write_bytes(p.read_bytes()[:-10])
    with pytest.raises(ValueError, match="Truncated"):
        inspect_wav(p)
