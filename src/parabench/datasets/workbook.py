"""Read the current vertical review-card workbook without changing its cells."""

from pathlib import Path

import openpyxl

FIELD_LABELS = {
    "我的声线": "gender",
    "我的情绪": "emotion",
    "我的语速（选填）": "speed",
    "我的重音标注": "stress",
    "我的停顿标注": "pause",
    "我的情境判断": "context",
    "我的处理意见": "decision",
    "我的质量判断": "quality",
    "我听到的具体意思": "meaning",
    "处理理由／备注": "notes",
}


def read_review_workbook(path: Path) -> dict[str, dict]:
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        sheet = workbook["逐条复核"]
        answers, current = {}, None
        for values in sheet.iter_rows(min_col=1, max_col=4, values_only=True):
            left, value, label, answer = values
            if left == "音频" and isinstance(value, str) and value.startswith("AUDIO_"):
                if value in answers:
                    raise ValueError(f"Duplicate audio card: {value}")
                current = value
                answers[current] = {}
            if current is None or not isinstance(label, str):
                continue
            label = label.replace("（必填）", "")
            if label in FIELD_LABELS:
                key = FIELD_LABELS[label]
                if answer is not None and not isinstance(answer, str):
                    raise ValueError(f"Expected text answer: {current}/{label}")
                answers[current][key] = answer.strip() if isinstance(answer, str) else None
        if not answers:
            raise ValueError("No AUDIO_ cards in review workbook")
        return answers
    finally:
        workbook.close()
