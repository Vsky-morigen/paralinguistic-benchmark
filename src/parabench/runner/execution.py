import asyncio
import json
import math
from pathlib import Path

from parabench import __version__
from parabench.datasets.precollection import load_samples, validate_release
from parabench.io import (
    atomic_bytes,
    exclusive_lock,
    fingerprint,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from parabench.models import create_adapter
from parabench.models.base import ModelError
from parabench.schemas import DIMENSIONS, ID_PATTERN, PredictionResult
from parabench.tasks import PROMPT_VERSION, build_request, load_choices


async def predict_run(
    release: Path,
    reference_root: Path,
    config: dict,
    results_root: Path,
    run_id: str,
    *,
    resume: bool = False,
    language: str | None = None,
    limit: int | None = None,
    model_root: Path | None = None,
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict:
    if not ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or set(dimensions) - set(DIMENSIONS)
    ):
        raise ValueError("Select distinct supported dimensions")
    validate_release(release, reference_root)
    catalog = load_choices(release)
    samples = [s for s in load_samples(release) if language is None or s.language == language]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples selected")
    run = results_root / run_id
    spec = {
        "task": "paralinguistic_understanding",
        "package_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "manifest_sha256": sha256_file(release / "manifest.json"),
        "model_config": config,
        "sample_ids": [s.sample_id for s in samples],
        "dimensions": list(dimensions),
    }
    signature = fingerprint(spec)
    with exclusive_lock(results_root / (run_id + ".lock")):
        run_file = run / "run.json"
        if run_file.exists():
            info = read_json(run_file)
            if not resume:
                raise ValueError("Run exists; use --resume or a new run ID")
            if info["fingerprint"] != signature:
                raise ValueError("Resume rejected: dataset, model, tasks or prompt changed")
            if fingerprint(info["spec"]) != info["fingerprint"]:
                raise ValueError("Run specification changed")
            if sha256_file(run / "samples.json") != info["samples_sha256"]:
                raise ValueError("Run sample snapshot changed")
            if sha256_file(run / "choices.json") != info["choices_sha256"]:
                raise ValueError("Run choice snapshot changed")
        else:
            if run.exists() and any(run.iterdir()):
                raise ValueError("Run folder is nonempty but has no run.json")
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / "samples.json", [s.to_dict() for s in samples])
            write_json(run / "choices.json", catalog)
            info = {
                "schema_version": 2,
                "run_id": run_id,
                "created_at": utc_now(),
                "fingerprint": signature,
                "spec": spec,
                "samples_sha256": sha256_file(run / "samples.json"),
                "choices_sha256": sha256_file(run / "choices.json"),
            }
            write_json(run_file, info)
        adapter = create_adapter(config, model_root=model_root)
        semaphore = asyncio.Semaphore(config["concurrency"])
        counts = {"predicted": 0, "resumed": 0, "failed": 0}

        async def one(sample):
            async with semaphore:
                audio = run / "inputs" / (sample.sample_id + ".wav")
                if not audio.is_file() or sha256_file(audio) != sample.reference_sha256:
                    source = reference_root / sample.reference_audio
                    payload = source.read_bytes()
                    import hashlib

                    if hashlib.sha256(payload).hexdigest() != sample.reference_sha256:
                        raise ValueError("Source audio changed during snapshot")
                    atomic_bytes(audio, payload)
                path = run / "records" / (sample.sample_id + ".json")
                record = (
                    read_json(path)
                    if resume and path.is_file()
                    else {
                        "sample_id": sample.sample_id,
                        "audio_path": audio.relative_to(run).as_posix(),
                        "audio_sha256": sample.reference_sha256,
                        "dimensions": {},
                    }
                )
                for name in dimensions:
                    request = build_request(
                        sample, name, audio, catalog, config["parameters"], config.get("seed", 0)
                    )
                    portable = request.to_payload()
                    request_hash = fingerprint(portable)
                    request_path = run / "requests" / (request.request_id + ".json")
                    prior = record["dimensions"].get(name, {})
                    if (
                        resume
                        and prior.get("status") == "success"
                        and prior.get("request_hash") == request_hash
                        and prior.get("prediction_hash") == fingerprint(prior.get("prediction"))
                        and request_path.is_file()
                        and read_json(request_path) == portable
                    ):
                        counts["resumed"] += 1
                        continue
                    write_json(request_path, portable)
                    attempts = []
                    for number in range(1, config["max_attempts"] + 1):
                        try:
                            prediction = await adapter.predict(request)
                            if not isinstance(prediction, PredictionResult):
                                raise ValueError("Adapter must return PredictionResult")
                            if (
                                prediction.model != config["model"]
                                or type(prediction.latency_seconds) not in {int, float}
                                or not math.isfinite(prediction.latency_seconds)
                                or prediction.latency_seconds < 0
                            ):
                                raise ValueError("Invalid prediction metadata")
                            value = prediction.to_dict()
                            json.dumps(value, allow_nan=False)
                            record["dimensions"][name] = {
                                "status": "success",
                                "request_hash": request_hash,
                                "prediction": value,
                                "prediction_hash": fingerprint(value),
                                "attempts": attempts,
                            }
                            counts["predicted"] += 1
                            break
                        except (ModelError, ValueError, TypeError, OSError) as exc:
                            reason = str(exc) if isinstance(exc, ModelError) else type(exc).__name__
                            attempts.append({"attempt": number, "error": reason})
                            retryable = isinstance(exc, ModelError) and exc.retryable
                            if not retryable or number == config["max_attempts"]:
                                record["dimensions"][name] = {
                                    "status": "failed",
                                    "request_hash": request_hash,
                                    "attempts": attempts,
                                }
                                counts["failed"] += 1
                                break
                            await asyncio.sleep(config["retry_backoff_seconds"] * 2 ** (number - 1))
                    write_json(path, record)

        try:
            await asyncio.gather(*(one(s) for s in samples))
        finally:
            await adapter.close()
        info["last_prediction"] = {**counts, "completed_at": utc_now()}
        write_json(run_file, info)
        return {
            "run_id": run_id,
            "samples": len(samples),
            "requests": len(samples) * len(dimensions),
            **counts,
        }
