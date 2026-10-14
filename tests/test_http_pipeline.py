import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from parabench.io import sha256_file
from parabench.reporting import build_report
from parabench.runner.evaluation import evaluate_run
from parabench.runner.execution import predict_run


def test_real_http_transport_input_prediction_and_scoring(release, eval_config, tmp_path):
    received = []
    answers = {
        "gender": "男声",
        "emotion": "中性",
        "context": "平常地打招呼",
        "speaking_rate": "中速",
        "stress": [[0, 2]],
        "pause": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request = body["input"]
            received.append(request)
            reply = json.dumps({"answer": answers[request["dimension"]]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "schema_version": 1,
        "adapter": "http",
        "model": "http-fixture",
        "endpoint": f"http://127.0.0.1:{server.server_port}/predict",
        "timeout_seconds": 5,
        "concurrency": 2,
        "max_attempts": 1,
        "retry_backoff_seconds": 0.001,
        "parameters": {},
    }
    try:
        result = asyncio.run(predict_run(*release[:2], config, tmp_path, "http"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["predicted"] == len(received) == 18
    for request in received:
        raw = base64.b64decode(request["audio"]["data_base64"])
        sample_id = request["request_id"].split(".")[0]
        audio = release[1] / (sample_id + ".wav")
        assert raw == audio.read_bytes() and request["audio_sha256"] == sha256_file(audio)
        assert ("transcript" in request) == (request["dimension"] in {"stress", "pause"})
        assert "targets" not in request
    run = tmp_path / "http"
    assert evaluate_run(run, eval_config)["scored"] == 15
    summary = build_report(run)
    emotion = next(
        g for g in summary["scores"] if g["scope"] == "all" and g["dimension"] == "emotion"
    )
    assert emotion["mean_score"] == pytest.approx(2 / 3)
