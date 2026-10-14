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
    write_json(
        output / "choices.json",
        {
            "schema_version": 1,
            "labels": {
                "gender": ["男声", "女声"],
                "emotion": ["中性", "悲伤"],
                "speaking_rate": ["偏慢", "中速", "偏快"],
            },
            "context": {"ZH-TEST": ["平常地打招呼", "表达不满"]},
        },
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["choices_sha256"] = sha256_file(output / "choices.json")
    write_json(output / "manifest.json", manifest)
    return output, root, originals, review


@pytest.fixture
def mock_config():
    return {
        "schema_version": 1,
        "adapter": "mock",
        "model": "test-mock",
        "concurrency": 2,
        "max_attempts": 2,
        "retry_backoff_seconds": 0.001,
        "timeout_seconds": 1,
        "seed": 0,
        "parameters": {},
    }


@pytest.fixture
def eval_config():
    from parabench.schemas import DIMENSIONS

    return {
        "schema_version": 1,
        "prosody_metric": "position_f1_v2",
        "dimensions": {name: {"enabled": True} for name in DIMENSIONS},
    }


@pytest.fixture
def local_setup(tmp_path):
    root = tmp_path / "project"
    (root / "configs/models").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "weights/weights.json").write_text('{"label":"男声"}', encoding="utf-8")
    log = root / "calls.txt"
    (root / "loader.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def load_model(*, model_path, device, **kwargs):\n"
        "    label=json.loads((model_path/'weights.json').read_text(encoding='utf-8'))['label']\n"
        "    log=Path(kwargs['log'])\n"
        "    def note(text):\n"
        "        with log.open('a',encoding='utf-8') as f: f.write(text+'\\n')\n"
        "    note('load:'+device)\n"
        "    class Model:\n"
        "        def predict(self, request):\n"
        "            assert request.audio_path.read_bytes().startswith(b'RIFF')\n"
        "            note('predict:'+request.dimension+':'+str(request.transcript is not None))\n"
        "            return [] if request.dimension in {'stress','pause'} else label if request.dimension=='gender' else request.choices[0]\n"
        "        def close(self): note('close')\n"
        "    return Model()\n",
        encoding="utf-8",
    )
    path = root / "configs/models/local.yaml"
    import yaml

    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "adapter": "local",
                "model": "fixture-local",
                "model_path": "weights",
                "loader": "loader.py:load_model",
                "device": "cpu",
                "concurrency": 1,
                "max_attempts": 1,
                "parameters": {},
                "load_kwargs": {"log": str(log)},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root, path, log
