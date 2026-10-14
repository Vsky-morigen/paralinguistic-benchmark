import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
DIMENSIONS = ("gender", "emotion", "stress", "pause", "speaking_rate", "context")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    language: str
    text: str
    targets: dict
    reference_audio: str
    reference_sha256: str
    annotation_decision: str = "validated"
    provenance: dict = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "Sample":
        item = cls(**data)
        if item.schema_version != 1 or not ID_PATTERN.fullmatch(item.sample_id):
            raise ValueError("Invalid schema version or sample_id")
        if item.language not in {"zh", "en"} or not item.text.strip():
            raise ValueError(f"Invalid language/text: {item.sample_id}")
        audio = PurePosixPath(item.reference_audio)
        if (
            audio.is_absolute()
            or ".." in audio.parts
            or "\\" in item.reference_audio
            or ":" in item.reference_audio
        ):
            raise ValueError("reference_audio must be a relative path within the audio root")
        if not re.fullmatch(r"[0-9a-f]{64}", item.reference_sha256):
            raise ValueError("Invalid reference SHA-256")
        required = {"gender", "emotion", "context", "speed", "speed_verified", "stress", "pause"}
        if not isinstance(item.targets, dict) or not required <= item.targets.keys():
            raise ValueError("Incomplete reference labels")
        for name in ("gender", "emotion", "context", "speed"):
            if not isinstance(item.targets[name], str) or not item.targets[name].strip():
                raise ValueError(f"Empty target: {name}")
        if type(item.targets["speed_verified"]) is not bool:
            raise ValueError("speed_verified must be boolean")
        for name in ("stress", "pause"):
            if not isinstance(item.targets[name], dict) or item.targets[name].get("status") not in {
                "annotated",
                "unknown",
            }:
                raise ValueError(f"Invalid annotation status: {name}")
        return item

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UnderstandingRequest:
    request_id: str
    dimension: str
    language: str
    audio_path: Path
    audio_sha256: str
    question: str
    choices: list[str]
    parameters: dict = field(default_factory=dict)
    transcript: str | None = None

    def to_payload(self) -> dict:
        """Portable model input, without file paths or reference answers."""
        value = {
            name: getattr(self, name)
            for name in (
                "request_id",
                "dimension",
                "language",
                "audio_sha256",
                "question",
                "choices",
                "parameters",
            )
        }
        if self.transcript is not None:
            value["transcript"] = self.transcript
        return value


@dataclass
class PredictionResult:
    answer: object
    latency_seconds: float
    model: str
    simulated: bool = False
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        if type(self.simulated) is not bool:
            raise ValueError("simulated must be boolean")
        return asdict(self)


@dataclass
class EvaluationResult:
    dimension: str
    status: str
    score: float | None = None
    passed: bool | None = None
    method: str | None = None
    reason: str | None = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
