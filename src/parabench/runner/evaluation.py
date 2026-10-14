from pathlib import Path

from parabench.evaluators import evaluate_answer, unavailable_reason, validate_config
from parabench.io import (
    exclusive_lock,
    fingerprint,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from parabench.schemas import DIMENSIONS, EvaluationResult, Sample
from parabench.tasks import build_request


def checked_snapshot(run: Path) -> tuple[dict, list[Sample], dict]:
    info = read_json(run / "run.json")
    if (
        info.get("schema_version") != 2
        or info["spec"].get("task") != "paralinguistic_understanding"
    ):
        raise ValueError(
            "Expected an audio understanding run; old generation runs are incompatible"
        )
    if fingerprint(info["spec"]) != info["fingerprint"]:
        raise ValueError("Run specification changed")
    for filename, key in [("samples.json", "samples_sha256"), ("choices.json", "choices_sha256")]:
        if sha256_file(run / filename) != info[key]:
            raise ValueError("Run snapshot changed")
    return (
        info,
        [Sample.from_dict(s) for s in read_json(run / "samples.json")],
        read_json(run / "choices.json"),
    )


def evaluate_run(run: Path, config: dict) -> dict:
    validate_config(config)
    with exclusive_lock(run.parent / (run.name + ".lock")):
        info, samples, catalog = checked_snapshot(run)
        output, record_hashes = [], {}
        model = info["spec"]["model_config"]
        for sample in samples:
            path = run / "records" / (sample.sample_id + ".json")
            record = read_json(path) if path.is_file() else {"dimensions": {}}
            if path.is_file():
                record_hashes[sample.sample_id] = sha256_file(path)
            audio = run / "inputs" / (sample.sample_id + ".wav")
            audio_error = not audio.is_file() or sha256_file(audio) != sample.reference_sha256
            results = []
            for name in DIMENSIONS:
                task = record.get("dimensions", {}).get(name, {})
                prediction = task.get("prediction", {})
                skip = unavailable_reason(sample, name)
                if (
                    not config["dimensions"][name]["enabled"]
                    or name not in info["spec"]["dimensions"]
                ):
                    value = EvaluationResult(name, "skipped", reason="Dimension disabled")
                elif (
                    prediction.get("simulated")
                    or model["adapter"] == "mock"
                    or model.get("simulated", False)
                ):
                    value = EvaluationResult(
                        name, "skipped", reason="Simulated prediction; not a benchmark score"
                    )
                elif skip:
                    value = EvaluationResult(name, "skipped", reason=skip)
                elif audio_error:
                    value = EvaluationResult(
                        name, "error", reason="Input audio changed or is missing"
                    )
                elif task.get("status") != "success":
                    value = EvaluationResult(
                        name, "error", reason="Model inference failed or is missing"
                    )
                else:
                    request = build_request(
                        sample, name, audio, catalog, model["parameters"], model.get("seed", 0)
                    )
                    if task.get("request_hash") != fingerprint(request.to_payload()):
                        value = EvaluationResult(
                            name,
                            "error",
                            reason="Prediction is bound to a different input or question",
                        )
                    elif task.get("prediction_hash") != fingerprint(prediction):
                        value = EvaluationResult(
                            name, "error", reason="Prediction changed after inference"
                        )
                    else:
                        value = evaluate_answer(
                            sample, name, prediction.get("answer"), request.choices
                        )
                results.append(value.to_dict())
            output.append(
                {
                    "sample_id": sample.sample_id,
                    "language": sample.language,
                    "input_audio_sha256": sample.reference_sha256,
                    "results": results,
                }
            )
        write_jsonl(run / "evaluations.jsonl", output)
        write_json(run / "evaluation_config.json", config)
        write_json(
            run / "evaluation_provenance.json",
            {
                "created_at": utc_now(),
                "run_fingerprint": info["fingerprint"],
                "evaluations_sha256": sha256_file(run / "evaluations.jsonl"),
                "evaluation_config_sha256": sha256_file(run / "evaluation_config.json"),
                "record_hashes": record_hashes,
            },
        )
        return {
            "samples": len(output),
            "scored": sum(v["status"] == "scored" for row in output for v in row["results"]),
            "errors": sum(v["status"] == "error" for row in output for v in row["results"]),
        }
