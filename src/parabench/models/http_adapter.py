import asyncio
import base64
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from parabench.models.base import AudioModelAdapter, ModelError, validate_input
from parabench.schemas import PredictionResult, UnderstandingRequest


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpAdapter(AudioModelAdapter):
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
        self.limit = config.get("max_response_bytes", 1024 * 1024)
        if type(self.limit) is not int or not 1024 <= self.limit <= 16 * 1024 * 1024:
            raise ValueError("Invalid max_response_bytes")
        self.opener = urllib.request.build_opener(NoRedirect())

    def _request(self, request: UnderstandingRequest) -> PredictionResult:
        validate_input(request)
        started = time.perf_counter()
        value = request.to_payload()
        value["audio"] = {
            "format": "wav",
            "data_base64": base64.b64encode(request.audio_path.read_bytes()).decode("ascii"),
        }
        body = {"model": self.config["model"], "input": value}
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
            raise ModelError(
                f"HTTP {exc.code}", retryable=exc.code in {408, 429} or exc.code >= 500
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ModelError("Network request failed or timed out", retryable=True) from None
        if len(payload) > self.limit:
            raise ModelError("HTTP response exceeds configured size limit")
        try:
            result = json.loads(payload)
            if not isinstance(result, dict) or "answer" not in result:
                raise ValueError("Expected answer")
            usage_raw = result.get("usage", {})
            if not isinstance(usage_raw, dict):
                raise ValueError("Invalid usage")
            usage = {
                k: v
                for k, v in usage_raw.items()
                if k in {"input_tokens", "output_tokens", "audio_seconds"}
                and type(v) in {int, float}
                and math.isfinite(v)
                and 0 <= v < 10**12
            }
        except (ValueError, TypeError):
            raise ModelError("Malformed JSON response or missing answer") from None
        return PredictionResult(
            result["answer"], time.perf_counter() - started, self.config["model"], usage=usage
        )

    async def predict(self, request: UnderstandingRequest) -> PredictionResult:
        return await asyncio.to_thread(self._request, request)
