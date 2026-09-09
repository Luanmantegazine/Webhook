#!/usr/bin/env python3
"""Evaluate Hydra's rule classifier from cached aggregate-document JSON files.

Manifest columns:
    sample_id,rvl_label,target_family,document_path,page_sizes_path

``page_sizes_path`` is optional. Paths are resolved relative to the manifest.
The script writes ``predictions.csv`` and ``report.json`` without rerunning
YOLO, docTR, translation, or template reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.document.rules_classifier_core import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    DOCUMENT_FAMILIES,
    classify_with_rules,
    extract_classification_features,
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _metrics(records: list[dict]) -> dict:
    labels = list(DOCUMENT_FAMILIES)
    confusion = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for record in records:
        truth = record["target_family"]
        prediction = record["predicted_family"]
        if truth not in confusion:
            confusion[truth] = {candidate: 0 for candidate in labels}
        if prediction not in confusion[truth]:
            confusion[truth][prediction] = 0
        confusion[truth][prediction] += 1

    per_class = {}
    for label in labels:
        tp = sum(1 for r in records if r["target_family"] == label and r["predicted_family"] == label)
        fp = sum(1 for r in records if r["target_family"] != label and r["predicted_family"] == label)
        fn = sum(1 for r in records if r["target_family"] == label and r["predicted_family"] != label)
        support = sum(1 for r in records if r["target_family"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    supported = [value for value in per_class.values() if value["support"] > 0]
    accepted = [record for record in records if record["decision"] == "classified"]
    correct = sum(r["target_family"] == r["predicted_family"] for r in records)
    accepted_correct = sum(r["target_family"] == r["predicted_family"] for r in accepted)
    classifier_times = [float(record["classifier_time_ms"]) for record in records]
    feature_times = [float(record["feature_extraction_time_ms"]) for record in records]

    return {
        "sample_count": len(records),
        "accuracy": round(correct / len(records), 4) if records else 0.0,
        "macro_precision": round(statistics.fmean(item["precision"] for item in supported), 4) if supported else 0.0,
        "macro_recall": round(statistics.fmean(item["recall"] for item in supported), 4) if supported else 0.0,
        "macro_f1": round(statistics.fmean(item["f1"] for item in supported), 4) if supported else 0.0,
        "coverage": round(len(accepted) / len(records), 4) if records else 0.0,
        "abstention_or_fallback_rate": round(1.0 - len(accepted) / len(records), 4) if records else 0.0,
        "accuracy_on_accepted": round(accepted_correct / len(accepted), 4) if accepted else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "timing_ms": {
            "feature_mean": round(statistics.fmean(feature_times), 4) if feature_times else 0.0,
            "classifier_mean": round(statistics.fmean(classifier_times), 4) if classifier_times else 0.0,
            "classifier_p50": round(_percentile(classifier_times, 0.50), 4),
            "classifier_p95": round(_percentile(classifier_times, 0.95), 4),
            "classifier_p99": round(_percentile(classifier_times, 0.99), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("output/rules-benchmark"))
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--min-score-margin", type=float, default=DEFAULT_MIN_SCORE_MARGIN)
    parser.add_argument(
        "--min-recognized-characters", type=int, default=DEFAULT_MIN_RECOGNIZED_CHARACTERS
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    base = manifest.parent
    results = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            document_path = (base / row["document_path"]).resolve()
            page_sizes_path = row.get("page_sizes_path", "").strip()
            page_sizes = _read_json((base / page_sizes_path).resolve()) if page_sizes_path else None

            feature_started = perf_counter()
            features = extract_classification_features(_read_json(document_path), page_sizes=page_sizes)
            feature_time = (perf_counter() - feature_started) * 1000.0

            classifier_started = perf_counter()
            prediction = classify_with_rules(
                features,
                confidence_threshold=args.confidence_threshold,
                min_score_margin=args.min_score_margin,
                min_recognized_characters=args.min_recognized_characters,
                mode="evaluate",
            )
            classifier_time = (perf_counter() - classifier_started) * 1000.0
            results.append(
                {
                    "sample_id": row["sample_id"],
                    "rvl_label": row["rvl_label"],
                    "target_family": row["target_family"],
                    "predicted_family": prediction["document_family"],
                    "top_candidate": prediction["top_candidate"],
                    "confidence": prediction["confidence"],
                    "score_margin": prediction["score_margin"],
                    "decision": prediction["decision"],
                    "reason": prediction["reason"],
                    "feature_extraction_time_ms": round(feature_time, 4),
                    "classifier_time_ms": round(classifier_time, 4),
                    "rules_triggered": json.dumps(prediction["evidence"]["rules_triggered"]),
                    "candidate_scores": json.dumps(prediction["candidate_scores"]),
                }
            )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_directory / "predictions.csv"
    report_path = args.output_directory / "report.json"
    fieldnames = list(results[0].keys()) if results else ["sample_id"]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    report = {
        "configuration": {
            "confidence_threshold": args.confidence_threshold,
            "min_score_margin": args.min_score_margin,
            "min_recognized_characters": args.min_recognized_characters,
        },
        "metrics": _metrics(results),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({"predictions": str(predictions_path), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

