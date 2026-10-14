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
from parabench.schemas import DIMENSIONS


def build_report(run: Path) -> dict:
    with exclusive_lock(run.parent / (run.name + ".lock")):
        provenance = read_json(run / "evaluation_provenance.json")
        if sha256_file(run / "evaluations.jsonl") != provenance["evaluations_sha256"]:
            raise ValueError("Evaluations changed after scoring")
        for sample_id, digest in provenance["generation_record_hashes"].items():
            record_path = run / "records" / (sample_id + ".json")
            if sha256_file(record_path) != digest:
                raise ValueError(
                    "Generation changed after evaluation; rerun evaluate before report"
                )
            record = read_json(record_path)
            if record["status"] == "success":
                audio = (run / record["generation"]["audio_path"]).resolve()
                if (
                    not audio.is_relative_to(run.resolve())
                    or sha256_file(audio) != record["audio_sha256"]
                ):
                    raise ValueError("Generated audio changed after evaluation")
        rows = read_jsonl(run / "evaluations.jsonl")
        summary = {
            "samples": len(rows),
            "languages": dict(Counter(x["language"] for x in rows)),
            "generation_status": dict(Counter(x["generation_status"] for x in rows)),
            "simulated_samples": sum(x["simulated"] for x in rows),
            "scores": [],
            "overall_score": None,
            "overall_note": "No overall score; incomplete dimensions and unlike methods are not pooled",
        }
        for scope in ["all", "zh", "en"]:
            selected = [x for x in rows if scope == "all" or x["language"] == scope]
            for dimension in DIMENSIONS:
                values = [r for x in selected for r in x["results"] if r["dimension"] == dimension]
                scored = [r for r in values if r["status"] == "scored"]
                methods = sorted({r["method"] for r in scored}) or [None]
                for method in methods:
                    group = [r for r in scored if r["method"] == method]
                    summary["scores"].append(
                        {
                            "scope": scope,
                            "dimension": dimension,
                            "method": method,
                            "selected": len(selected),
                            "scored": len(group),
                            "skipped": sum(r["status"] == "skipped" for r in values),
                            "errors": sum(r["status"] == "error" for r in values),
                            "coverage": len(group) / len(selected) if selected else None,
                            "mean_score": sum(r["score"] for r in group) / len(group)
                            if group
                            else None,
                            "pass_rate": sum(r["passed"] for r in group) / len(group)
                            if group
                            else None,
                            "mean_precision": sum(r["evidence"]["precision"] for r in group)
                            / len(group)
                            if group and all("precision" in r["evidence"] for r in group)
                            else None,
                            "mean_recall": sum(r["evidence"]["recall"] for r in group) / len(group)
                            if group and all("recall" in r["evidence"] for r in group)
                            else None,
                            "metric_versions": sorted(
                                {
                                    r["evidence"]["metric_version"]
                                    for r in group
                                    if "metric_version" in r["evidence"]
                                }
                            ),
                        }
                    )
        write_json(run / "report.json", summary)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "sample_id",
                "language",
                "generation_status",
                "simulated",
                "dimension",
                "status",
                "method",
                "score",
                "passed",
                "reason",
                "precision",
                "recall",
                "true_positive",
                "false_positive",
                "false_negative",
                "metric_version",
            ]
        )
        for x in rows:
            for r in x["results"]:
                writer.writerow(
                    [
                        x["sample_id"],
                        x["language"],
                        x["generation_status"],
                        x["simulated"],
                        r["dimension"],
                        r["status"],
                        r["method"],
                        r["score"],
                        r["passed"],
                        r["reason"],
                        *[
                            r["evidence"].get(k)
                            for k in [
                                "precision",
                                "recall",
                                "true_positive",
                                "false_positive",
                                "false_negative",
                                "metric_version",
                            ]
                        ],
                    ]
                )
        atomic_bytes(run / "details.csv", buffer.getvalue().encode("utf-8-sig"))
        lines = [
            "# Benchmark run report",
            "",
            f"Samples: {len(rows)}; simulated: {summary['simulated_samples']}.",
            "No overall score. Missing evaluations are reported separately.",
            "",
            "| Scope | Dimension | Method | Scored / selected | Skipped | Errors | Mean score | Precision | Recall |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary["scores"]:
            score = "n.a." if row["mean_score"] is None else f"{row['mean_score']:.4f}"
            precision = "n.a." if row["mean_precision"] is None else f"{row['mean_precision']:.4f}"
            recall = "n.a." if row["mean_recall"] is None else f"{row['mean_recall']:.4f}"
            lines.append(
                f"| {row['scope']} | {row['dimension']} | {row['method'] or 'unconfigured'} | {row['scored']} / {row['selected']} | {row['skipped']} | {row['errors']} | {score} | {precision} | {recall} |"
            )
        atomic_bytes(run / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
        return summary
