"""Create uniform reference WAV copies; keep originals and their timing intact."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable():
    tool = shutil.which("ffmpeg")
    if tool:
        return tool
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ValueError("Install FFmpeg or run: python -m pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args):
    result = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    if result.returncode:
        raise ValueError("FFmpeg failed: " + result.stderr[-1000:])
    return result


def normalize(input_dir: Path, output_dir: Path, sample_rate=48000):
    if sample_rate not in {16000, 24000, 44100, 48000}:
        raise ValueError("Unsupported target sample rate")
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Output must be a different directory from the originals")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output folder is not empty; refuse to overwrite audio")
    files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac"}
    )
    if not files or len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("No audio found or duplicate output filenames")
    ffmpeg = executable()
    profiles = []
    for source in files:
        probe = run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-t",
                "0",
                "-f",
                "null",
                "-",
            ]
        )
        match = re.search(r"Stream #0:\d+.*?Audio: ([^,]+), (\d+) Hz, ([^,]+),", probe.stderr)
        if not match or match[3] not in {"mono", "stereo"}:
            raise ValueError(f"Unsupported channel layout: {source.name}")
        profiles.append((source, match[1], int(match[2]), match[3], sha256(source)))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source, codec, rate, layout, original_hash in profiles:
        output = output_dir / (source.stem + ".wav")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
        ]
        if layout == "stereo":
            command += ["-af", "pan=mono|c0=0.5*c0+0.5*c1"]
        command += [
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            str(output),
        ]
        run(command)
        with wave.open(str(output), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                sample_rate,
                1,
                2,
            ):
                raise ValueError("Unexpected converted WAV format")
            frames = audio.getnframes()
            data = audio.readframes(frames)
            if frames <= 0 or len(data) != frames * 2:
                raise ValueError("Empty or truncated converted WAV")
        if sha256(source) != original_hash:
            raise ValueError("Original audio changed during conversion")
        records.append(
            {
                "source": source.name,
                "source_sha256": original_hash,
                "source_codec": codec,
                "source_sample_rate": rate,
                "source_channels": 2 if layout == "stereo" else 1,
                "output": output.name,
                "output_sha256": sha256(output),
                "sample_rate": sample_rate,
                "channels": 1,
                "sample_width_bytes": 2,
                "frames": frames,
                "duration_seconds": frames / sample_rate,
                "mono_method": "mean_stereo" if layout == "stereo" else "mono_passthrough",
            }
        )
    version = run([ffmpeg, "-version"]).stdout.splitlines()[0]
    manifest = {
        "schema_version": 1,
        "format": "wav_pcm_s16le",
        "sample_rate": sample_rate,
        "channels": 1,
        "ffmpeg_version": version,
        "source_count": len(records),
        "files": records,
        "processing": "decode; stereo mean if needed; optional resampling; PCM16 encoding; no trimming, denoising, or loudness normalization",
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "converted": len(records),
        "sample_rate": sample_rate,
        "channels": 1,
        "format": "PCM16 WAV",
        "output": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                normalize(args.input_dir, args.output_dir, args.sample_rate), ensure_ascii=False
            )
        )
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
