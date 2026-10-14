import pytest

from parabench.datasets.precollection import derive_sample
from parabench.io import sha256_file, write_json


def test_original_filename_maps_to_verified_normalized_wav(release, tmp_path):
    old_release, original_root, originals_path, _ = release
    from parabench.io import read_json

    original = read_json(originals_path)["source"]["refs"][0]
    original["audio"] = "AUDIO_001.m4a"
    wav_root = tmp_path / "wav"
    wav_root.mkdir()
    output = wav_root / "AUDIO_001.wav"
    output.write_bytes((original_root / "AUDIO_001.wav").read_bytes())
    write_json(
        wav_root / "conversion_manifest.json",
        {
            "schema_version": 1,
            "sample_rate": 16000,
            "files": [
                {
                    "source": "AUDIO_001.m4a",
                    "source_sha256": "a" * 64,
                    "output": "AUDIO_001.wav",
                    "output_sha256": sha256_file(output),
                }
            ],
        },
    )
    row = {"audio": "AUDIO_001.m4a", "language": "中文", "decision": "保留原标注"}
    sample = derive_sample(row, original, {2}, wav_root)
    assert sample.reference_audio == "AUDIO_001.wav"
    assert sample.provenance["source_reference_audio"] == "AUDIO_001.m4a"
    assert sample.targets["emotion"] == "中性"
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        derive_sample(row, original, {2}, wav_root)


def test_wav_without_source_mapping_is_not_silently_adopted(release, tmp_path):
    _, original_root, originals_path, _ = release
    from parabench.io import read_json

    original = read_json(originals_path)["source"]["refs"][0]
    root = tmp_path / "wav"
    root.mkdir()
    (root / "AUDIO_001.wav").write_bytes((original_root / "AUDIO_001.wav").read_bytes())
    row = {"audio": "AUDIO_001.m4a", "language": "中文", "decision": "保留原标注"}
    with pytest.raises(ValueError, match="Missing"):
        derive_sample(row, original, {2}, root)
