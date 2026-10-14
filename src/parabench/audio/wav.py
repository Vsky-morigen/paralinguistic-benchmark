import math
import wave
from pathlib import Path


def inspect_wav(path: Path) -> dict:
    """Validate a generated PCM WAV; compressed reference recordings stay untouched."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels, width, rate, frames, compression, _ = handle.getparams()
            if compression != "NONE" or channels not in {1, 2} or width not in {1, 2, 3, 4}:
                raise ValueError("Expected mono/stereo PCM WAV with 8/16/24/32-bit samples")
            if not 8000 <= rate <= 192000 or frames <= 0 or frames / rate > 600:
                raise ValueError(
                    "Invalid sample rate, empty audio, or audio longer than 600 seconds"
                )
            payload = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid or unsupported WAV") from exc
    if len(payload) != frames * width * channels:
        raise ValueError("Truncated WAV data")
    scale = 2 ** (8 * width - 1)
    count, squares, peak, clipped = 0, 0.0, 0.0, 0
    for i in range(0, len(payload), width):
        chunk = payload[i : i + width]
        value = (chunk[0] - 128) if width == 1 else int.from_bytes(chunk, "little", signed=True)
        level = value / scale
        squares += level * level
        peak = max(peak, abs(level))
        clipped += abs(level) >= 0.999
        count += 1
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_seconds": frames / rate,
        "rms": math.sqrt(squares / count),
        "peak": peak,
        "clipped_fraction": clipped / count,
        "silent": peak == 0,
    }
