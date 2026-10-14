import asyncio
import base64
import io
import json
import urllib.error

import pytest

from parabench.datasets.precollection import load_samples
from parabench.models.base import ModelError
from parabench.models.http_adapter import HttpAdapter
from parabench.tasks import build_request, load_choices


def setup(release, dimension="gender"):
    sample = load_samples(release[0])[0]
    request = build_request(
        sample, dimension, release[1] / sample.reference_audio, load_choices(release[0]), {}
    )
    adapter = HttpAdapter(
        {"endpoint": "http://localhost/predict", "model": "fixture", "timeout_seconds": 1}
    )
    return request, adapter


@pytest.mark.parametrize(
    "dimension", ["gender", "emotion", "context", "speaking_rate", "stress", "pause"]
)
def test_http_sends_exact_audio_and_only_allowed_text(release, dimension):
    request, adapter = setup(release, dimension)

    class Opener:
        def open(self, req, timeout):
            payload = json.loads(req.data)
            body = payload["input"]
            assert base64.b64decode(body["audio"]["data_base64"]) == request.audio_path.read_bytes()
            assert ("transcript" in body) == (dimension in {"stress", "pause"})
            assert not {"targets", "audio_path", "reference_audio", "provenance"} & body.keys()
            return io.BytesIO(
                json.dumps(
                    {"answer": [] if dimension in {"stress", "pause"} else request.choices[0]}
                ).encode()
            )

    adapter.opener = Opener()
    assert asyncio.run(adapter.predict(request)).model == "fixture"


@pytest.mark.parametrize("status,retryable", [(401, False), (400, False), (429, True), (503, True)])
def test_http_retry_classification_and_secret_redaction(release, status, retryable):
    request, adapter = setup(release)

    class Opener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(
                req.full_url, status, "private-token", {}, io.BytesIO(b"private-token")
            )

    adapter.opener = Opener()
    with pytest.raises(ModelError) as error:
        asyncio.run(adapter.predict(request))
    assert error.value.retryable == retryable and "private-token" not in str(error.value)


@pytest.mark.parametrize("payload", [b"invalid", b"{}", b"[]", b'{"answer":1,"usage":[]}'])
def test_malformed_http_responses_fail(release, payload):
    request, adapter = setup(release)

    class Opener:
        def open(self, req, timeout):
            return io.BytesIO(payload)

    adapter.opener = Opener()
    with pytest.raises(ModelError, match="Malformed"):
        asyncio.run(adapter.predict(request))


def test_changed_audio_is_not_sent(release):
    request, adapter = setup(release)
    request.audio_path.write_bytes(b"tamper")
    with pytest.raises(ModelError, match="hash mismatch"):
        asyncio.run(adapter.predict(request))
