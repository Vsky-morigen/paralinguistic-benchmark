import csv
import io
from collections import Counter
from pathlib import Path

from parabench.io import (
    atomic_bytes,
    exclusive_lock,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from parabench.runner.evaluation import checked_snapshot
from parabench.schemas import DIMENSIONS


def label_macro_f1(values: list[dict]) -> float | None:
    classes = {v["evidence"]["reference"] for v in values if v["status"] == "scored"}
    if not classes:
        return None
    scores = []
    for label in classes:
        tp = sum(
            v["evidence"]["reference"] == label and v["evidence"]["prediction"] == label
            for v in values
        )
        fp = sum(
            v["evidence"]["reference"] != label and v["evidence"]["prediction"] == label
            for v in values
        )
        fn = sum(
            v["evidence"]["reference"] == label and v["evidence"]["prediction"] != label
            for v in values
        )
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return sum(scores) / len(scores)


def build_report(run: Path) -> dict:
    with exclusive_lock(run.parent / (run.name + ".lock")):
        info, samples, _ = checked_snapshot(run)
        provenance = read_json(run / "evaluation_provenance.json")
        if provenance["run_fingerprint"] != info["fingerprint"]:
            raise ValueError("Run changed after evaluation")
        for filename, key in [
            ("evaluations.jsonl", "evaluations_sha256"),
            ("evaluation_config.json", "evaluation_config_sha256"),
        ]:
            if sha256_file(run / filename) != provenance[key]:
                raise ValueError("Evaluation changed; rerun evaluate")
        records = run / "records"
        current_ids = {
            s.sample_id for s in samples if (records / (s.sample_id + ".json")).is_file()
        }
        if current_ids != set(provenance["record_hashes"]):
            raise ValueError("Prediction records changed; rerun evaluate")
        for sid, digest in provenance["record_hashes"].items():
            if sha256_file(records / (sid + ".json")) != digest:
                raise ValueError("Prediction changed; rerun evaluate")
        for sample in samples:
            audio = run / "inputs" / (sample.sample_id + ".wav")
            if not audio.is_file() or sha256_file(audio) != sample.reference_sha256:
                raise ValueError("Input audio changed; report is invalid")
        rows = read_jsonl(run / "evaluations.jsonl")
        summary = {
            "samples": len(rows),
            "languages": dict(Counter(x["language"] for x in rows)),
            "scores": [],
        }
        for scope in ("all", "zh", "en"):
            selected = [row for row in rows if scope == "all" or row["language"] == scope]
            for name in DIMENSIONS:
                values = [v for row in selected for v in row["results"] if v["dimension"] == name]
                eligible = [v for v in values if v["status"] != "skipped"]
                scored = [v for v in values if v["status"] == "scored"]
                positions = name in {"stress", "pause"}
                group = {
                    "scope": scope,
                    "dimension": name,
                    "selected": len(selected),
                    "eligible": len(eligible),
                    "scored": len(scored),
                    "skipped": sum(v["status"] == "skipped" for v in values),
                    "errors": sum(v["status"] == "error" for v in values),
                    "invalid_answers": sum(
                        not v["evidence"].get("valid_answer", True) for v in scored
                    ),
                    "coverage": len(scored) / len(eligible) if eligible else None,
                    "mean_score": sum(v["score"] for v in scored) / len(eligible)
                    if eligible
                    else None,
                    "completed_mean_score": sum(v["score"] for v in scored) / len(scored)
                    if scored
                    else None,
                    "pass_rate": sum(v["passed"] for v in scored) / len(eligible)
                    if eligible
                    else None,
                    "completed_macro_f1": label_macro_f1(scored)
                    if scored and not positions
                    else None,
                    "mean_precision": sum(v["evidence"]["precision"] for v in scored)
                    / len(eligible)
                    if positions and eligible
                    else None,
                    "mean_recall": sum(v["evidence"]["recall"] for v in scored) / len(eligible)
                    if positions and eligible
                    else None,
                    "input_mode": "audio_with_plain_transcript"
                    if positions
                    else "audio_question_choices",
                }
                summary["scores"].append(group)
        write_json(run / "report.json", summary)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "sample_id",
                "language",
                "dimension",
                "status",
                "score",
                "passed",
                "reason",
                "valid_answer",
                "precision",
                "recall",
                "true_positive",
                "false_positive",
                "false_negative",
            ]
        )
        for row in rows:
            for value in row["results"]:
                writer.writerow(
                    [
                        row["sample_id"],
                        row["language"],
                        value["dimension"],
                        value["status"],
                        value["score"],
                        value["passed"],
                        value["reason"],
                        *[
                            value["evidence"].get(k)
                            for k in (
                                "valid_answer",
                                "precision",
                                "recall",
                                "true_positive",
                                "false_positive",
                                "false_negative",
                            )
                        ],
                    ]
                )
        atomic_bytes(run / "details.csv", ("\ufeff" + buffer.getvalue()).encode("utf-8"))
        lines = [
            "# 副语言理解评测结果",
            "",
            f"音频数量：{len(rows)}",
            "",
            "主分数以全部有效题目为分母；无效回答及调用失败按0计入。未知标签、关闭维度和模拟预测不计分。",
            "",
            "| 范围 | 维度 | 有效题 | 已评分 | 错误 | 主分数 | 全对率 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for group in summary["scores"]:
            if group["selected"]:
                score = "未评分" if group["mean_score"] is None else f"{group['mean_score']:.4f}"
                passed = "—" if group["pass_rate"] is None else f"{group['pass_rate']:.4f}"
                lines.append(
                    f"| {group['scope']} | {group['dimension']} | {group['eligible']} | {group['scored']} | {group['errors']} | {score} | {passed} |"
                )
        atomic_bytes(run / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
        return summary
