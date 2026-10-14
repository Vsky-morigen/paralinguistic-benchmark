import re
import unicodedata

UNKNOWN = {"无法判断", "听不清", "不确定"}
NONE = {"无", "无重音", "无明显重音", "无停顿", "无句中停顿", "无句内停顿"}


def canonical(text: str) -> str:
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def stress_annotation(value: str, text: str) -> dict:
    value = value.strip()
    if value in UNKNOWN:
        return {"status": "unknown", "display": value, "spans": []}
    if value in NONE or canonical(value) == canonical(text) + "无":
        return {"status": "annotated", "display": "无", "spans": []}
    matches = list(re.finditer(r"【([^【】]+)】", value))
    if value.count("【") != len(matches) or value.count("】") != len(matches):
        raise ValueError("Unbalanced stress annotation brackets")
    plain = value.replace("【", "").replace("】", "")
    if canonical(plain) != canonical(text):
        raise ValueError("Stress annotation changes the transcript")
    spans = []
    for m in matches:
        start = len(canonical(value[: m.start()]))
        end = start + len(canonical(m.group(1)))
        if start == end:
            raise ValueError("Empty stress target")
        spans.append([start, end])
    return {
        "status": "annotated",
        "display": value.replace("|", "").replace("｜", ""),
        "spans": spans,
    }


def pause_annotation(value: str, text: str, sentence_boundaries: set[int]) -> dict:
    value = value.strip()
    if value in UNKNOWN:
        return {"status": "unknown", "display": value, "positions": []}
    if value in NONE:
        return {"status": "annotated", "display": "无句内停顿", "positions": []}
    plain = value.replace("【", "").replace("】", "")
    if canonical(plain) != canonical(text):
        raise ValueError("Pause annotation changes the transcript")
    positions, chars = [], []
    for i, c in enumerate(plain):
        if c not in "|｜":
            chars.append(c)
        else:
            offset = len(canonical(plain[:i]))
            if offset not in sentence_boundaries and 0 < offset < len(canonical(text)):
                positions.append(offset)
                chars.append("｜")
    return {
        "status": "annotated",
        "display": "".join(chars) if positions else "无句内停顿",
        "positions": sorted(set(positions)),
    }
