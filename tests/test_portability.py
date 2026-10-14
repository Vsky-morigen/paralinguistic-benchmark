import asyncio
import json
import shutil
from pathlib import Path

import pytest

from parabench.config import model_config, resolve_dataset
from parabench.datasets.precollection import validate_release
from parabench.io import read_json, read_jsonl, sha256_file, write_json
from parabench.runner.execution import generate_run


def test_dataset_paths_follow_project_after_relocation(release, tmp_path, monkeypatch):
    original = tmp_path / "original"
    (original / "configs").mkdir(parents=True)
    shutil.copytree(release[0], original / "data/metadata")
    shutil.copytree(release[1], original / "data/audio")
    (original / "configs/dataset.yaml").write_text(
        "schema_version: 1\nrelease: data/metadata\nreference_root: data/audio\n",
        encoding="utf-8",
    )
    moved = tmp_path / "another machine" / "benchmark"
    shutil.copytree(original, moved)
    monkeypatch.delenv("PARABENCH_REFERENCE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    labels, audio = resolve_dataset(moved / "configs/dataset.yaml")
    assert labels == moved / "data/metadata"
    assert audio == moved / "data/audio"
    assert validate_release(labels, audio)["samples"] == 3


def test_local_run_preserves_relative_paths(release, local_setup, tmp_path):
    root, path, _ = local_setup
    config = model_config(path)
    assert config["model_path"] == "weights"
    assert config["loader"] == "loader.py:load_model"
    asyncio.run(generate_run(release[0], config, tmp_path / "runs", "portable", model_root=root))
    saved = read_json(tmp_path / "runs/portable/run.json")["spec"]["model_config"]
    assert saved["model_path"] == "weights"
    assert saved["loader"] == "loader.py:load_model"


def test_public_manifest_does_not_require_review_history(release):
    labels, audio, *_ = release
    manifest = read_json(labels / "manifest.json")
    for key in ("excluded_sha256", "excluded_count"):
        manifest.pop(key)
    (labels / "excluded.jsonl").unlink()
    write_json(labels / "manifest.json", manifest)
    result = validate_release(labels, audio)
    assert result["samples"] == 3 and "excluded" not in result


def test_reference_format_is_checked(release):
    labels, audio, *_ = release
    manifest = read_json(labels / "manifest.json")
    manifest["reference_audio_format"] = {"codec": "pcm_s16le", "sample_rate": 48000, "channels": 1}
    write_json(labels / "manifest.json", manifest)
    with pytest.raises(ValueError, match="format mismatch"):
        validate_release(labels, audio)


def test_shipped_metadata_is_portable_and_complete():
    root = Path(__file__).resolve().parents[1]
    metadata = root / "data/metadata"
    rows = read_jsonl(metadata / "samples.jsonl")
    manifest = read_json(metadata / "manifest.json")
    assert len(rows) == manifest["sample_count"] == 44
    assert sha256_file(metadata / "samples.jsonl") == manifest["samples_sha256"]
    for row in rows:
        assert not Path(row["reference_audio"]).is_absolute()
        assert set(row["provenance"]) == {"text_id", "version", "sentence_boundaries"}
        assert "annotation_decision" not in row
    assert "Users" not in json.dumps(rows, ensure_ascii=False)
