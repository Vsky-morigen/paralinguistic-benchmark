import io
import math
import struct
import time
import wave
from pathlib import Path

from parabench.io import atomic_bytes
from parabench.models.base import SpeechModelAdapter
from parabench.schemas import GenerationRequest, GenerationResult


class MockAdapter(SpeechModelAdapter):
    async def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult:
        start = time.perf_counter()
        rate = self.config["generation"].get("sample_rate", 16000)
        duration = self.config["generation"].get("duration_seconds", 0.25)
        if type(rate) is not int or not 8000 <= rate <= 48000 or not 0 < duration <= 10:
            raise ValueError("Invalid mock audio settings")
        payload = b"".join(
            struct.pack("<h", int(4000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(rate * duration))
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as handle:
            handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            handle.writeframes(payload)
        atomic_bytes(output_path, buf.getvalue())
        return GenerationResult(
            str(output_path),
            rate,
            time.perf_counter() - start,
            self.config["model"],
            simulated=True,
            metadata={"notice": "Tone only; not speech and never scored"},
        )
