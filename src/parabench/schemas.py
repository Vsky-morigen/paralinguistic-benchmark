import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
DIMENSIONS = ("content", "gender", "emotion", "stress", "pause", "speaking_rate", "context")


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
        if audio.is_absolute() or ".." in audio.parts or "\\" in item.reference_audio:
            raise ValueError("reference_audio must be a relative path within the reference root")
        if not re.fullmatch(r"[0-9a-f]{64}", item.reference_sha256):
            raise ValueError("Invalid reference SHA-256")
        required = {"gender", "emotion", "context", "speed", "speed_verified", "stress", "pause"}
        if not isinstance(item.targets, dict) or not required <= item.targets.keys():
            raise ValueError("Incomplete expression targets")
        for k in ["gender", "emotion", "context", "speed"]:
            if not isinstance(item.targets[k], str) or not item.targets[k].strip():
                raise ValueError(f"Empty target: {k}")
        if type(item.targets["speed_verified"]) is not bool:
            raise ValueError("speed_verified must be boolean")
        for k in ["stress", "pause"]:
            if not isinstance(item.targets[k], dict):
                raise ValueError(f"Invalid target: {k}")
            if item.targets[k].get("status") not in {"annotated", "unknown"}:
                raise ValueError(f"Invalid annotation status: {k}")
        return item

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    sample_id: str
    text: str
    language: str
    instructions: str
    generation: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    audio_path: str
    sample_rate: int
    latency_seconds: float
    model: str
    simulated: bool = False
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    dimension: str
    status: str  # scored | skipped | error
    score: float | None = None
    passed: bool | None = None
    method: str | None = None
    reason: str | None = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
