import io
import json
import wave

import pytest

from parabench.datasets.precollection import import_release
from parabench.io import sha256_file, write_json


def wav_bytes(rate=16000, frames=1600):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "references"
    root.mkdir()
    refs, rows = [], []
    for i, decision in enumerate(["保留原标注", "修正录后自标", "原五项通过", "需要重录"], 1):
        audio = f"AUDIO_{i:03}.wav"
        (root / audio).write_bytes(wav_bytes())
        refs.append(
            {
                "audio": audio,
                "text": "你好。再见。",
                "annotation": "【你好】。｜再见。",
                "gender": "男声",
                "emotion": "中性",
                "speed": "中速",
                "expected_context": "平常地打招呼",
                "original": audio,
                "speaker": "test",
                "sid": "ZH-TEST",
                "version": "A",
            }
        )
        rows.append(
            {
                "audio": audio,
                "language": "中文",
                "decision": decision,
                "review_fields": {"emotion": "悲伤", "stress": "你好。再【见】。", "pause": "无"},
            }
        )
    originals, review = tmp_path / "originals.json", tmp_path / "review.json"
    write_json(
        originals,
        {
            "source": {"refs": refs},
            "sentence_boundaries": [{"sid": "ZH-TEST", "canonical_positions": [2]}],
        },
    )
    write_json(review, {"rows": rows, "review_sha256": sha256_file(originals)})
    output = tmp_path / "release_v1"
    import_release(review, originals, output, root)
    return output, root, originals, review


@pytest.fixture
def mock_config():
    return {
        "schema_version": 1,
        "adapter": "mock",
        "model": "test-mock",
        "concurrency": 2,
        "max_attempts": 3,
        "retry_backoff_seconds": 0.001,
        "timeout_seconds": 1,
        "generation": {"sample_rate": 16000, "duration_seconds": 0.02},
    }


@pytest.fixture
def eval_config():
    from parabench.schemas import DIMENSIONS

    return {
        "schema_version": 1,
        "content_max_error_rate": 0.1,
        "dimensions": {k: {"mode": "evidence"} for k in DIMENSIONS},
    }


@pytest.fixture
def local_setup(tmp_path):
    root = tmp_path / "project"
    (root / "configs/models").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "weights/weights.json").write_text('{"amplitude": 0.1}', encoding="utf-8")
    loader = root / "loader.py"
    loader.write_text(
        """
import json
from pathlib import Path
from parabench.models.local import LocalAudio

def load_model(*, model_path, device, **kwargs):
    amplitude = json.loads((model_path/'weights.json').read_text())['amplitude']
    log = Path(kwargs['log'])
    with log.open('a', encoding='utf-8') as f: f.write('load:'+device+'\\n')
    class Model:
        def generate(self, request):
            with log.open('a', encoding='utf-8') as f:
                f.write('generate:'+request.sample_id+':'+str(bool(request.instructions))+'\\n')
            return LocalAudio(sample_rate=16000, samples=[amplitude]*160, simulated=True)
        def close(self):
            with log.open('a', encoding='utf-8') as f: f.write('close\\n')
    return Model()
""",
        encoding="utf-8",
    )
    log = root / "calls.txt"
    path = root / "configs/models/local.yaml"
    path.write_text(
        "schema_version: 1\nadapter: local\nmodel: fixture-local\nmodel_path: weights\nloader: loader.py:load_model\ndevice: cpu\nconcurrency: 1\nmax_attempts: 1\nload_kwargs:\n  log: "
        + json.dumps(str(log))
        + "\ngeneration: {}\n",
        encoding="utf-8",
    )
    return root, path, log
