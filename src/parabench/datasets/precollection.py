from collections import Counter
from pathlib import Path

from parabench.audio import inspect_wav
from parabench.datasets.annotations import canonical, pause_annotation, stress_annotation
from parabench.io import (
    exclusive_lock,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from parabench.schemas import Sample

KEEP = {"保留原标注", "维持原标注", "原五项通过"}
CORRECT = "修正录后自标"


def derive_sample(row: dict, original: dict, boundaries: set[int], root: Path) -> Sample:
    decision = row["decision"]
    if decision not in KEEP | {CORRECT}:
        raise ValueError(f"Not a retained decision: {decision}")
    # Decision takes precedence over populated review cells. Blank corrected dimensions
    # retain the original annotation; absence is never interpreted as explicit 'none'.
    fields = row.get("review_fields", {}) if decision == CORRECT else {}
    targets = {
        "gender": fields.get("gender") or original["gender"],
        "emotion": fields.get("emotion") or original["emotion"],
        "context": fields.get("context") or original["expected_context"],
        "speed": fields.get("speed") or original["speed"],
        "speed_verified": False,
        "stress": stress_annotation(
            fields.get("stress") or original["annotation"], original["text"]
        ),
        "pause": pause_annotation(
            fields.get("pause") or original["annotation"], original["text"], boundaries
        ),
    }
    audio = root / row["audio"]
    conversion = None
    if not audio.is_file():
        mapping_path = root / "conversion_manifest.json"
        candidate = root / (Path(row["audio"]).stem + ".wav")
        if mapping_path.is_file() and candidate.is_file():
            mapping = read_json(mapping_path)
            matches = [
                r
                for r in mapping["files"]
                if r["source"] == row["audio"] and r["output"] == candidate.name
            ]
            if mapping.get("schema_version") != 1 or len(matches) != 1:
                raise ValueError("Invalid converted reference mapping")
            entry = matches[0]
            if sha256_file(candidate) != entry["output_sha256"]:
                raise ValueError("Converted reference audio hash mismatch")
            info = inspect_wav(candidate)
            if (info["sample_rate"], info["channels"], info["sample_width_bytes"]) != (
                mapping["sample_rate"],
                1,
                2,
            ):
                raise ValueError("Converted reference WAV format mismatch")
            audio = candidate
            conversion = {
                "source_reference_audio": row["audio"],
                "source_reference_sha256": entry["source_sha256"],
                "conversion_manifest_sha256": sha256_file(mapping_path),
            }
    if not audio.is_file() or audio.resolve().parent != root.resolve():
        raise ValueError(f"Missing or invalid reference audio: {row['audio']}")
    return Sample.from_dict(
        {
            "sample_id": Path(row["audio"]).stem,
            "language": {"中文": "zh", "英文": "en"}[row["language"]],
            "text": original["text"],
            "targets": targets,
            "reference_audio": audio.name,
            "reference_sha256": sha256_file(audio),
            "annotation_decision": decision,
            "provenance": {
                "original_filename": original["original"],
                "speaker": original["speaker"],
                "text_id": original["sid"],
                "version": original["version"],
                "original_annotation": original["annotation"],
                "original_emotion": original["emotion"],
                "original_context": original["expected_context"],
                "sentence_boundaries": sorted(boundaries),
                "original_three_listener_support": row.get("support", {}),
                "quality_notes": row.get("notes", ""),
                **(conversion or {}),
            },
        }
    )


def import_release(
    annotations: Path,
    originals: Path,
    release: Path,
    reference_root: Path,
    review_workbook: Path | None = None,
) -> dict:
    review = read_json(annotations)
    source = read_json(originals)
    refs = {x["audio"]: x for x in source["source"]["refs"]}
    boundaries = {x["sid"]: set(x["canonical_positions"]) for x in source["sentence_boundaries"]}
    rows = review["rows"]
    if review_workbook:
        from parabench.datasets.workbook import read_review_workbook

        answers = read_review_workbook(review_workbook)
        expected = {x["audio"] for x in rows if x["decision"] != "原五项通过"}
        if set(answers) != expected:
            raise ValueError("Review workbook must cover exactly the pending review audio IDs")
        for row in rows:
            if row["audio"] in answers:
                row["review_fields"] = answers[row["audio"]]
                row["decision"] = answers[row["audio"]].get("decision")
    ids = [x["audio"] for x in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(refs):
        raise ValueError("Review rows must uniquely cover the original audio IDs")
    samples, excluded = [], []
    for row in rows:
        decision = row["decision"]
        if decision == "需要重录":
            excluded.append(
                {"audio": row["audio"], "language": row["language"], "reason": decision}
            )
        else:
            ref = refs[row["audio"]]
            samples.append(derive_sample(row, ref, boundaries[ref["sid"]], reference_root))
    samples.sort(key=lambda x: x.sample_id)
    files = {
        "review_result": sha256_file(annotations),
        "original_analysis": sha256_file(originals),
    }
    if review_workbook:
        files["review_workbook"] = sha256_file(review_workbook)
    with exclusive_lock(release.parent / (release.name + ".import.lock")):
        if release.exists() and any(release.iterdir()):
            raise ValueError(
                "Release already exists; use a new version directory, do not overwrite"
            )
        release.mkdir(parents=True, exist_ok=True)
        write_jsonl(release / "samples.jsonl", [s.to_dict() for s in samples])
        write_jsonl(release / "excluded.jsonl", excluded)
        manifest = {
            "schema_version": 1,
            "dataset_version": release.name,
            "created_at": utc_now(),
            "annotation_rule": "review_decision_precedence_v2",
            "source_hashes": files,
            "review_workbook_sha256": sha256_file(review_workbook)
            if review_workbook
            else review.get("review_sha256"),
            "sample_count": len(samples),
            "excluded_count": len(excluded),
            "languages": dict(Counter(s.language for s in samples)),
            "decisions": dict(Counter(s.annotation_decision for s in samples)),
            "samples_sha256": sha256_file(release / "samples.jsonl"),
            "excluded_sha256": sha256_file(release / "excluded.jsonl"),
            "purpose": "paralinguistic audio understanding",
            "speed_verified": False,
        }
        write_json(release / "manifest.json", manifest)
    validate_release(release, reference_root)
    return manifest


def load_samples(release: Path) -> list[Sample]:
    manifest = read_json(release / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported release schema")
    if sha256_file(release / "samples.jsonl") != manifest["samples_sha256"]:
        raise ValueError("samples.jsonl hash mismatch; create a new release for edits")
    samples = [Sample.from_dict(x) for x in read_jsonl(release / "samples.jsonl")]
    if len(samples) != manifest["sample_count"] or len({s.sample_id for s in samples}) != len(
        samples
    ):
        raise ValueError("Manifest count/unique sample IDs mismatch")
    return samples


def validate_release(release: Path, reference_root: Path) -> dict:
    samples = load_samples(release)
    manifest = read_json(release / "manifest.json")
    if "choices_sha256" in manifest:
        from parabench.tasks import load_choices

        catalog = load_choices(release)
        for sample in samples:
            context_options = catalog["context"].get(sample.provenance["text_id"], [])
            if sample.targets["context"] not in context_options:
                raise ValueError(f"Reference context is not in the choice set: {sample.sample_id}")
            for dimension, target in [
                ("gender", "gender"),
                ("emotion", "emotion"),
                ("speaking_rate", "speed"),
            ]:
                if dimension == "speaking_rate" and not sample.targets["speed_verified"]:
                    continue
                if sample.targets[target] not in catalog["labels"][dimension]:
                    raise ValueError(
                        f"Reference label is not in the choice set: {sample.sample_id}"
                    )
    excluded = None
    if "excluded_sha256" in manifest:
        if sha256_file(release / "excluded.jsonl") != manifest["excluded_sha256"]:
            raise ValueError("Excluded list hash mismatch")
        excluded = read_jsonl(release / "excluded.jsonl")
        if len(excluded) != manifest["excluded_count"]:
            raise ValueError("Excluded list count mismatch")
    for s in samples:
        audio = (reference_root / s.reference_audio).resolve()
        if not audio.is_relative_to(reference_root.resolve()) or not audio.is_file():
            raise ValueError(f"Missing reference audio: {s.sample_id}")
        if sha256_file(audio) != s.reference_sha256:
            raise ValueError(f"Reference audio hash mismatch: {s.sample_id}")
        info = inspect_wav(audio)
        expected_format = manifest.get("reference_audio_format")
        if expected_format:
            actual = (info["sample_rate"], info["channels"], info["sample_width_bytes"])
            expected = (expected_format["sample_rate"], expected_format["channels"], 2)
            if expected_format["codec"] != "pcm_s16le" or actual != expected:
                raise ValueError(f"Reference audio format mismatch: {s.sample_id}")
        conversion_hash = s.provenance.get("conversion_manifest_sha256")
        if (
            conversion_hash
            and sha256_file(reference_root / "conversion_manifest.json") != conversion_hash
        ):
            raise ValueError("Reference conversion manifest changed")
        n = len(canonical(s.text))
        for start, end in s.targets["stress"]["spans"]:
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= n:
                raise ValueError(f"Invalid stress span: {s.sample_id}")
        for offset in s.targets["pause"]["positions"]:
            if type(offset) is not int or not 0 < offset < n:
                raise ValueError(f"Invalid pause offset: {s.sample_id}")
        if set(s.targets["pause"]["positions"]) & set(s.provenance["sentence_boundaries"]):
            raise ValueError("Intersentence pause was not removed")
    if dict(Counter(s.language for s in samples)) != manifest["languages"]:
        raise ValueError("Language count mismatch")
    result = {
        "valid": True,
        "samples": len(samples),
        "languages": manifest["languages"],
    }
    if excluded is not None:
        result["excluded"] = len(excluded)
    return result
