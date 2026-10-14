import asyncio
import base64
import binascii
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from parabench.audio import inspect_wav
from parabench.io import atomic_bytes
from parabench.models.base import ModelError, SpeechModelAdapter
from parabench.schemas import GenerationRequest, GenerationResult


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward an Authorization header to a redirected endpoint.
        return None


class HttpAdapter(SpeechModelAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        parsed = urllib.parse.urlsplit(config.get("endpoint", ""))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("endpoint must be an HTTP(S) URL without credentials/query/fragment")
        self.limit = config.get("max_response_bytes", 32 * 1024 * 1024)
        if type(self.limit) is not int or not 1024 <= self.limit <= 128 * 1024 * 1024:
            raise ValueError("Invalid max_response_bytes")
        self.opener = urllib.request.build_opener(NoRedirect())

    def _request(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        start = time.perf_counter()
        body = {
            "request_id": request.sample_id,
            "model": self.config["model"],
            "input": {
                "text": request.text,
                "language": request.language,
                "instructions": request.instructions,
            },
            "generation": request.generation,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key_name = self.config.get("api_key_env")
        if key_name:
            key = os.environ.get(key_name)
            if not key:
                raise ModelError(f"Missing environment variable: {key_name}")
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            self.config["endpoint"], json.dumps(body, ensure_ascii=False).encode("utf-8"), headers
        )
        try:
            with self.opener.open(req, timeout=self.config["timeout_seconds"]) as response:
                payload = response.read(self.limit + 1)
        except urllib.error.HTTPError as exc:
            # Provider response bodies can contain credentials; persist only the code.
            raise ModelError(
                f"HTTP {exc.code}", retryable=exc.code in {408, 429} or exc.code >= 500
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ModelError("Network request failed or timed out", retryable=True) from None
        if len(payload) > self.limit:
            raise ModelError("HTTP response exceeds configured size limit")
        try:
            value = json.loads(payload)
            if value.get("format") != "wav" or not isinstance(value.get("audio_base64"), str):
                raise ValueError("Expected WAV audio_base64")
            audio = base64.b64decode(value["audio_base64"], validate=True)
            atomic_bytes(output_path, audio)
            info = inspect_wav(output_path)
            if (
                type(value.get("sample_rate")) is not int
                or value["sample_rate"] != info["sample_rate"]
            ):
                raise ValueError("Declared sample_rate does not match WAV header")
            usage_raw = value.get("usage", {})
            if not isinstance(usage_raw, dict):
                raise ValueError("usage must be an object")
            usage = {
                k: v
                for k, v in usage_raw.items()
                if k in {"input_tokens", "output_tokens", "audio_seconds"}
                and type(v) in {int, float}
                and 0 <= v < 10**12
            }
        except (ValueError, TypeError, AttributeError, binascii.Error):
            output_path.unlink(missing_ok=True)
            raise ModelError("Malformed response, invalid WAV, or mismatched sample rate") from None
        return GenerationResult(
            str(output_path),
            info["sample_rate"],
            time.perf_counter() - start,
            self.config["model"],
            usage=usage,
        )

    async def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        return await asyncio.to_thread(self._request, request, output_path)
