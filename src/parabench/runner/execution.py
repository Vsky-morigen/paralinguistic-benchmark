import asyncio
from dataclasses import asdict
from pathlib import Path

from parabench import __version__
from parabench.audio import inspect_wav
from parabench.datasets.precollection import load_samples
from parabench.io import exclusive_lock, fingerprint, read_json, sha256_file, utc_now, write_json
from parabench.models import create_adapter
from parabench.models.base import ModelError
from parabench.schemas import ID_PATTERN
from parabench.tasks import PROMPT_VERSION, build_request


async def generate_run(
    release: Path,
    config: dict,
    results_root: Path,
    run_id: str,
    *,
    resume: bool = False,
    language: str | None = None,
    limit: int | None = None,
    model_root: Path | None = None,
) -> dict:
    if not ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    samples = [s for s in load_samples(release) if language is None or s.language == language]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples selected")
    run = results_root / run_id
    spec = {
        "package_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "manifest_sha256": sha256_file(release / "manifest.json"),
        "model_config": config,
        "sample_ids": [s.sample_id for s in samples],
    }
    signature = fingerprint(spec)
    with exclusive_lock(results_root / (run_id + ".lock")):
        run_file = run / "run.json"
        if run_file.exists():
            if not resume:
                raise ValueError("Run exists; use --resume or choose a new run ID")
            info = read_json(run_file)
            if info["fingerprint"] != signature:
                raise ValueError("Resume rejected: dataset/model/selection/prompt version changed")
            if sha256_file(run / "samples.json") != info["samples_sha256"]:
                raise ValueError("Run sample snapshot changed")
        else:
            if run.exists() and any(run.iterdir()):
                raise ValueError("Run folder has files but no run.json; use a new run ID")
            info = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
                "fingerprint": signature,
                "spec": spec,
            }
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / "samples.json", [s.to_dict() for s in samples])
            info["samples_sha256"] = sha256_file(run / "samples.json")
            write_json(run_file, info)
        adapter = create_adapter(config, model_root=model_root)
        semaphore = asyncio.Semaphore(config["concurrency"])
        counts = {"generated": 0, "resumed": 0, "failed": 0}

        async def one(sample):
            async with semaphore:
                record_path = run / "records" / (sample.sample_id + ".json")
                audio_path = run / "audio" / (sample.sample_id + ".wav")
                request = build_request(sample, config["generation"])
                request_hash = fingerprint(asdict(request))
                if resume and record_path.is_file():
                    prior = read_json(record_path)
                    if prior.get("status") == "success" and audio_path.is_file():
                        try:
                            inspect_wav(audio_path)
                            intact = sha256_file(audio_path) == prior["audio_sha256"]
                        except ValueError:
                            intact = False
                        if intact and prior.get("request_hash") == request_hash:
                            counts["resumed"] += 1
                            return
                write_json(run / "requests" / (sample.sample_id + ".json"), asdict(request))
                attempts = []
                for number in range(1, config["max_attempts"] + 1):
                    try:
                        result = await adapter.generate(request, audio_path)
                        if Path(result.audio_path).resolve() != audio_path.resolve():
                            raise ValueError("Adapter wrote to an unexpected output path")
                        audio_info = inspect_wav(audio_path)
                        if result.sample_rate != audio_info["sample_rate"]:
                            raise ValueError("Adapter returned mismatched sample rate")
                        value = result.to_dict()
                        value["audio_path"] = audio_path.relative_to(run).as_posix()
                        write_json(
                            record_path,
                            {
                                "sample_id": sample.sample_id,
                                "status": "success",
                                "request_hash": request_hash,
                                "completed_at": utc_now(),
                                "generation": value,
                                "audio_sha256": sha256_file(audio_path),
                                "audio_diagnostics": audio_info,
                                "attempts": attempts,
                            },
                        )
                        counts["generated"] += 1
                        return
                    except (ModelError, ValueError, OSError) as exc:
                        # Arbitrary adapter exception messages may leak secrets. Only known
                        # ModelError messages are retained; custom adapters must sanitize them.
                        reason = str(exc) if isinstance(exc, ModelError) else type(exc).__name__
                        attempts.append({"attempt": number, "error": reason, "at": utc_now()})
                        retryable = isinstance(exc, ModelError) and exc.retryable
                        if not retryable or number == config["max_attempts"]:
                            break
                        await asyncio.sleep(config["retry_backoff_seconds"] * 2 ** (number - 1))
                audio_path.unlink(missing_ok=True)
                write_json(
                    record_path,
                    {"sample_id": sample.sample_id, "status": "failed", "attempts": attempts},
                )
                counts["failed"] += 1

        try:
            await asyncio.gather(*(one(s) for s in samples))
        finally:
            await adapter.close()
        info["last_generation"] = {**counts, "completed_at": utc_now()}
        write_json(run_file, info)
        return {"run_id": run_id, "selected": len(samples), **counts}
