import json

import pytest

from parabench.datasets.annotations import pause_annotation, stress_annotation
from parabench.datasets.precollection import import_release, load_samples, validate_release
from parabench.io import read_json, write_json
from parabench.tasks import build_request


def test_decision_overrides_populated_review_cells(release):
    path, root, _, _ = release
    samples = load_samples(path)
    keep, corrected, original = samples
    assert keep.targets["emotion"] == original.targets["emotion"] == "中性"
    assert keep.targets["stress"]["spans"] == [[0, 2]]
    assert corrected.targets["emotion"] == "悲伤"
    assert corrected.targets["stress"]["spans"] == [[3, 4]]
    assert corrected.targets["pause"]["positions"] == []
    assert validate_release(path, root)["excluded"] == 1


def test_frozen_release_and_reference_hashes(release):
    path, root, originals, review = release
    with pytest.raises(ValueError, match="already exists"):
        import_release(review, originals, path, root)
    (root / "AUDIO_001.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_release(path, root)


def test_duplicate_review_and_unknown_decisions(release, tmp_path):
    _, root, originals, review = release
    source = read_json(review)
    source["rows"].append(source["rows"][0])
    write_json(review, source)
    with pytest.raises(ValueError, match="uniquely"):
        import_release(review, originals, tmp_path / "v2", root)
    source["rows"].pop()
    source["rows"][0]["decision"] = "invented"
    write_json(review, source)
    with pytest.raises(ValueError, match="decision"):
        import_release(review, originals, tmp_path / "v2", root)


def test_sentence_boundaries_and_explicit_none():
    text = "先看。再听。"
    assert pause_annotation("先｜看。｜再听。", text, {2})["positions"] == [1]
    assert pause_annotation("无", text, {2})["positions"] == []
    assert stress_annotation("无", text)["spans"] == []
    assert stress_annotation("无法判断", text)["status"] == "unknown"
    with pytest.raises(ValueError, match="brackets"):
        stress_annotation("【先看。再听。", text)


def test_model_request_does_not_include_reference_or_raters(release):
    sample = load_samples(release[0])[0]
    from dataclasses import asdict

    payload = json.dumps(asdict(build_request(sample, {})), ensure_ascii=False)
    assert "reference_audio" not in payload and "rater" not in payload
    assert "AUDIO_001.wav" not in payload


def test_blank_correction_is_not_explicit_none(release, tmp_path):
    _, root, originals, review = release
    data = read_json(review)
    data["rows"][1]["review_fields"]["stress"] = None
    write_json(review, data)
    output = tmp_path / "v2"
    import_release(review, originals, output, root)
    assert load_samples(output)[1].targets["stress"]["spans"] == [[0, 2]]


def test_latest_workbook_decision_is_authoritative(release, tmp_path):
    import openpyxl

    _, root, originals, review = release
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "逐条复核"
    for audio, decision in [
        ("AUDIO_001.wav", "维持原标注"),
        ("AUDIO_002.wav", "需要重录"),
        ("AUDIO_004.wav", "修正录后自标"),
    ]:
        sheet.append(["音频", audio, "原录音", None])
        sheet.append(["情绪", "原始值", "我的情绪（必填）", "担忧"])
        sheet.append(["原标", "原始值", "我的处理意见（必填）", decision])
    path = tmp_path / "review.xlsx"
    workbook.save(path)
    output = tmp_path / "workbook_release"
    import_release(review, originals, output, root, path)
    samples = {s.sample_id: s for s in load_samples(output)}
    assert samples["AUDIO_001"].targets["emotion"] == "中性"
    assert "AUDIO_002" not in samples
    assert samples["AUDIO_004"].targets["emotion"] == "担忧"
