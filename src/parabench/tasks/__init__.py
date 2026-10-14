import random
import unicodedata
from pathlib import Path

from parabench.datasets.annotations import canonical
from parabench.io import fingerprint, read_json, sha256_file
from parabench.schemas import DIMENSIONS, Sample, UnderstandingRequest

PROMPT_VERSION = "understanding_v1"
LABEL_ENGLISH = {
    "男声": "male-sounding voice",
    "女声": "female-sounding voice",
    "中性": "neutral",
    "愉悦": "happy",
    "恼怒": "angry",
    "悲伤": "sad",
    "担忧": "worried",
    "偏慢": "slow",
    "中速": "moderate",
    "偏快": "fast",
}


def label_in_language(label: str, language: str) -> str:
    return LABEL_ENGLISH.get(label, label) if language == "en" else label


def plain_transcript(text: str) -> str:
    # Keep word spaces, but remove punctuation, stress marks and pause symbols.
    return " ".join(
        "".join(
            char for char in unicodedata.normalize("NFKC", text) if char.isalnum() or char.isspace()
        ).split()
    )


def load_choices(release: Path) -> dict:
    manifest = read_json(release / "manifest.json")
    path = release / "choices.json"
    if manifest.get("choices_sha256") != sha256_file(path):
        raise ValueError("choices.json hash mismatch")
    catalog = read_json(path)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported choice schema")
    for values in [*catalog["labels"].values(), *catalog["context"].values()]:
        if (
            not isinstance(values, list)
            or len(values) < 2
            or len(set(values)) != len(values)
            or not all(isinstance(v, str) and v.strip() for v in values)
        ):
            raise ValueError("Each choice set needs at least two distinct nonempty labels")
    return catalog


def build_request(
    sample: Sample, dimension: str, audio: Path, catalog: dict, parameters: dict, seed: int = 0
) -> UnderstandingRequest:
    if dimension not in DIMENSIONS:
        raise ValueError("Unknown dimension")
    chinese = sample.language == "zh"
    questions = {
        "gender": (
            "根据声音听感，选择声音性别。",
            "Choose the perceived voice gender from the audio.",
        ),
        "emotion": (
            "根据实际语气，选择说话者的情绪。",
            "Choose the emotion conveyed by the vocal delivery.",
        ),
        "speaking_rate": (
            "根据实际声音，选择语速类别。",
            "Choose the speaking rate heard in the audio.",
        ),
        "context": (
            "结合声音和说话内容，选择最符合说话者表达意图的情境。",
            "Use the audio and its vocal delivery to choose the most likely communicative intent.",
        ),
    }
    transcript = None
    choices = []
    if dimension in {"stress", "pause"}:
        transcript = plain_transcript(sample.text)
        n = len(canonical(transcript))
        convention = (
            f"规范化台词采用 Unicode NFKC、仅保留字母数字并转小写，共{n}个字符。"
            if chinese
            else f"Normalize the transcript using Unicode NFKC, keep alphanumeric characters and lowercase; there are {n} characters. "
        )
        if dimension == "stress":
            task = (
                "听音频，返回所有重音字符区间。下标从0开始，区间左闭右开。只返回形如 [[0,2]] 的JSON数组；无重音返回[]。"
                if chinese
                else "Listen and return all stressed character spans as a JSON array, e.g. [[0,2]]. Use zero-based half-open intervals; return [] if none."
            )
        else:
            task = (
                "听音频，返回停顿所在的文字边界：数字表示前面有多少个规范化字符。只返回形如 [2] 的JSON数组；无停顿返回[]。无需标开头结尾静音，评分忽略句间停顿。"
                if chinese
                else "Listen and return pause boundaries as a JSON array, e.g. [2]; each number counts normalized characters preceding a pause. Return [] if none. Ignore initial/final silence; intersentence pauses are excluded by the scorer."
            )
        question = convention + task
    else:
        question = questions[dimension][0 if chinese else 1]
        choices = list(
            catalog["context"][sample.provenance["text_id"]]
            if dimension == "context"
            else catalog["labels"][dimension]
        )
        choices = [label_in_language(label, sample.language) for label in choices]
        # Choice order is reproducible and independent of the correct answer.
        random.Random(fingerprint([seed, sample.sample_id, dimension])).shuffle(choices)
        question += (
            "只返回一个候选标签，必须与选项文字完全一致。"
            if chinese
            else " Return exactly one of the supplied labels, without explanation."
        )
    return UnderstandingRequest(
        f"{sample.sample_id}.{dimension}",
        dimension,
        sample.language,
        audio,
        sample.reference_sha256,
        question,
        choices,
        parameters,
        transcript,
    )
