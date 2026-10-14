from pathlib import Path

from parabench.audio import inspect_wav
from parabench.evaluators import evaluate_sample, validate_config
from parabench.io import (
    exclusive_lock,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from parabench.schemas import DIMENSIONS, EvaluationResult, Sample


def evaluate_run(run: Path, config: dict, evidence_path: Path | None = None) -> dict:
    validate_config(config)
    items = read_jsonl(evidence_path) if evidence_path else []
    evidence = {x["sample_id"]: x for x in items}
    if len(evidence) != len(items):
        raise ValueError("Duplicate sample_id in external evidence")
    with exclusive_lock(run.parent / (run.name + ".lock")):
        info = read_json(run / "run.json")
        if sha256_file(run / "samples.json") != info["samples_sha256"]:
            raise ValueError("Run sample snapshot changed")
        samples = [Sample.from_dict(x) for x in read_json(run / "samples.json")]
        if set(evidence) - {s.sample_id for s in samples}:
            raise ValueError("Evidence contains sample IDs outside this run")
        output = []
        for sample in samples:
            path = run / "records" / (sample.sample_id + ".json")
            record = read_json(path) if path.exists() else {"status": "not_generated"}
            ev = evidence.get(sample.sample_id, {})
            error = None
            if record["status"] == "success":
                audio = run / record["generation"]["audio_path"]
                if not audio.resolve().is_relative_to(run.resolve()):
                    raise ValueError("Generated audio path escapes the run folder")
                try:
                    inspect_wav(audio)
                    if sha256_file(audio) != record["audio_sha256"]:
                        error = "Generated audio hash mismatch"
                except (OSError, ValueError):
                    error = "Generated audio missing or invalid"
                if ev and ev.get("audio_sha256") != record["audio_sha256"]:
                    error = "External evidence is not bound to this generated audio hash"
                if error:
                    results = [
                        EvaluationResult(k, "error", reason=error).to_dict() for k in DIMENSIONS
                    ]
                else:
                    results = evaluate_sample(
                        sample, ev, config, simulated=record["generation"]["simulated"]
                    )
            else:
                results = [
                    EvaluationResult(
                        k, "skipped", reason="Generation failed or is missing"
                    ).to_dict()
                    for k in DIMENSIONS
                ]
            output.append(
                {
                    "sample_id": sample.sample_id,
                    "language": sample.language,
                    "generation_status": record["status"],
                    "simulated": record.get("generation", {}).get("simulated", False),
                    "results": results,
                }
            )
        write_jsonl(run / "evaluations.jsonl", output)
        write_json(run / "evaluation_config.json", config)
        write_json(
            run / "evaluation_provenance.json",
            {
                "created_at": utc_now(),
                "evidence_sha256": sha256_file(evidence_path) if evidence_path else None,
                "run_fingerprint": info["fingerprint"],
                "evaluations_sha256": sha256_file(run / "evaluations.jsonl"),
                "generation_record_hashes": {
                    s.sample_id: sha256_file(run / "records" / (s.sample_id + ".json"))
                    for s in samples
                    if (run / "records" / (s.sample_id + ".json")).exists()
                },
            },
        )
        return {
            "samples": len(output),
            "scored": sum(r["status"] == "scored" for x in output for r in x["results"]),
            "errors": sum(r["status"] == "error" for x in output for r in x["results"]),
        }
