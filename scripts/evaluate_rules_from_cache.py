#!/usr/bin/env python3
"""Evaluate Hydra's rule classifier from cached aggregate-document JSON files.

Manifest columns:
    sample_id,rvl_label,target_family,document_path,page_sizes_path

``page_sizes_path`` is optional. Paths are resolved relative to the manifest.
The script writes ``predictions.csv`` and ``report.json`` without rerunning
YOLO, docTR, translation, or template reconstruction.

Metric design
-------------
``other`` is both a real document family and the sink for every abstention and
fallback, and in ``evaluate`` mode the classifier can *never* positively
predict it: ``apply_decision_policy`` returns a family only on the
``classified`` path, and that family always comes from ``SCORED_FAMILIES``.
Every ``other`` a run produces is therefore a refusal, not a prediction.

Scoring those refusals as predictions — which the previous revision did — hands
the classifier a true positive for ``other`` every time it declines to answer a
document that happens to be a file folder or a handwritten page. That inflates
``other`` precision and recall, and since ``other`` enters the macro average it
inflates the headline number too.

Refusals are consequently kept out of the classification metrics and reported
as coverage instead, which is the standard selective-classification framing the
prototype's risk-coverage protocol already assumes. Three views are reported and
each answers a different question:

``selective``
    Precision/recall/F1 over the predictions the classifier actually made.
    This is the quality of the rules.
``end_to_end``
    Accuracy over every sample. ``declined_as_error`` treats a refusal as
    wrong; ``declined_as_other`` treats it as a routing decision to the
    fallback template, which is the deployment view and the number the previous
    revision reported without naming it.
``confusion_matrix``
    Every sample, with refusals in an explicit ``<declined>`` column instead of
    folded into ``other``.

Run with ``--mode observe`` to disable abstention entirely and obtain the
full-coverage confusion matrix that isolates rule quality from the rejection
policy.
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
    apply_decision_policy,
    classify_with_rules,
    extract_classification_features,
)

#: Confusion-matrix column for samples the classifier refused to answer. Not a
#: family, and deliberately not spellable as one.
DECLINED = "<declined>"

#: Decisions that carry an actual prediction. ``observed`` is ``observe`` mode's
#: argmax, which is a prediction by construction; ``fallback`` and ``abstained``
#: are refusals.
ACCEPTING_DECISIONS = frozenset({"classified", "observed"})


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _ranked_from_scores(candidate_scores: dict) -> list[tuple[str, float]]:
    """Rebuild the ranking ``classify_with_rules`` produced.

    ``other`` is dropped: it is appended to ``candidate_scores`` for reporting
    but is not one of the scored families, and leaving it in would let a
    zero-scoring residual class occupy the runner-up slot and corrupt the
    margin. The sort key matches the core's exactly so a swept decision is the
    decision the classifier would have made.
    """
    return sorted(
        ((family, float(score)) for family, score in candidate_scores.items() if family != "other"),
        key=lambda item: (-item[1], item[0]),
    )


def _per_class(records: list[dict], labels: list[str]) -> dict:
    """Precision/recall/F1 per family over records carrying a prediction."""
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
            "predicted": tp + fp,
        }
    return per_class


def _selective_metrics(accepted: list[dict], labels: list[str]) -> dict:
    """Classification quality over the predictions the classifier made.

    The macro average covers families with non-zero support, which is the
    standard convention. A family that is predicted but never present cannot be
    averaged over — its recall is undefined rather than zero — so instead of
    silently dropping its false positives the count is surfaced under
    ``predictions_outside_support``. On a benchmark balanced by family the set
    is empty; on an ad-hoc manifest it is exactly the leak that would otherwise
    escape macro precision.
    """
    per_class = _per_class(accepted, labels)
    supported = [value for value in per_class.values() if value["support"] > 0]
    correct = sum(r["target_family"] == r["predicted_family"] for r in accepted)
    outside = {
        label: value["predicted"]
        for label, value in per_class.items()
        if value["support"] == 0 and value["predicted"] > 0
    }
    return {
        "sample_count": len(accepted),
        "accuracy": round(correct / len(accepted), 4) if accepted else 0.0,
        "macro_precision": round(statistics.fmean(item["precision"] for item in supported), 4) if supported else 0.0,
        "macro_recall": round(statistics.fmean(item["recall"] for item in supported), 4) if supported else 0.0,
        "macro_f1": round(statistics.fmean(item["f1"] for item in supported), 4) if supported else 0.0,
        "macro_averaged_over": [label for label, value in per_class.items() if value["support"] > 0],
        "predictions_outside_support": outside,
        "per_class": per_class,
    }


def _confusion_matrix(records: list[dict], labels: list[str]) -> dict:
    """Every sample, with refusals in their own column.

    Rows are ground-truth families; columns are the families plus
    ``<declined>``. Folding refusals into the ``other`` column, as the previous
    revision did, is what let an abstention be read as a correct ``other``.
    """
    columns = labels + [DECLINED]
    confusion = {truth: {column: 0 for column in columns} for truth in labels}
    for record in records:
        truth = record["target_family"]
        if truth not in confusion:
            confusion[truth] = {column: 0 for column in columns}
        column = (
            record["predicted_family"]
            if record["decision"] in ACCEPTING_DECISIONS
            else DECLINED
        )
        if column not in confusion[truth]:
            confusion[truth][column] = 0
        confusion[truth][column] += 1
    return confusion


def _risk_coverage(sweep_inputs: list[dict], args) -> list[dict]:
    """Sweep the confidence threshold over the stored scores.

    ``apply_decision_policy`` is a pure function of the scores and the
    thresholds, so the curve is recomputed without re-running the rule engine
    and is guaranteed to describe the same firings as the reported operating
    point. This needs the recognised-character count as well as the scores,
    which is why ``predictions.csv`` now carries it: without it the
    ``insufficient_ocr_text`` branch cannot be reproduced and the low-coverage
    end of the curve comes out wrong.
    """
    curve = []
    for step in range(0, 21):
        threshold = round(step * 0.05, 2)
        accepted = 0
        correct = 0
        for item in sweep_inputs:
            decision = apply_decision_policy(
                item["ranked"],
                item["alnum_character_count"],
                threshold,
                args.min_score_margin,
                args.min_recognized_characters,
                "evaluate",
            )
            if decision["decision"] in ACCEPTING_DECISIONS:
                accepted += 1
                correct += decision["document_family"] == item["target_family"]
        coverage = accepted / len(sweep_inputs) if sweep_inputs else 0.0
        accuracy = correct / accepted if accepted else 0.0
        curve.append(
            {
                "confidence_threshold": threshold,
                "coverage": round(coverage, 4),
                "accuracy_on_accepted": round(accuracy, 4),
                "risk": round(1.0 - accuracy, 4) if accepted else 0.0,
                "accepted": accepted,
            }
        )
    return curve


def _metrics(records: list[dict]) -> dict:
    labels = list(DOCUMENT_FAMILIES)
    accepted = [record for record in records if record["decision"] in ACCEPTING_DECISIONS]
    declined = [record for record in records if record["decision"] not in ACCEPTING_DECISIONS]

    decision_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for record in records:
        decision_counts[record["decision"]] = decision_counts.get(record["decision"], 0) + 1
        reason_counts[record["reason"]] = reason_counts.get(record["reason"], 0) + 1

    # Declined-as-error: a refusal is never a correct prediction. Declined-as-
    # other: a refusal routes to the fallback template, so it "matches" a
    # true ``other``. The second is a deployment metric, not a measure of rule
    # quality, and must never be reported as accuracy without the qualifier.
    correct_strict = sum(r["target_family"] == r["predicted_family"] for r in accepted)
    correct_routed = correct_strict + sum(r["target_family"] == "other" for r in declined)

    classifier_times = [float(record["classifier_time_ms"]) for record in records]
    feature_times = [float(record["feature_extraction_time_ms"]) for record in records]

    return {
        "sample_count": len(records),
        "decisions": {
            "coverage": round(len(accepted) / len(records), 4) if records else 0.0,
            "abstention_or_fallback_rate": round(len(declined) / len(records), 4) if records else 0.0,
            "counts": dict(sorted(decision_counts.items())),
            "reasons": dict(sorted(reason_counts.items())),
        },
        "selective": _selective_metrics(accepted, labels),
        "end_to_end": {
            "accuracy_declined_as_error": round(correct_strict / len(records), 4) if records else 0.0,
            "accuracy_declined_as_other": round(correct_routed / len(records), 4) if records else 0.0,
        },
        "confusion_matrix": _confusion_matrix(records, labels),
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
        "--mode",
        choices=("evaluate", "observe", "auto"),
        default="evaluate",
        help=(
            "Decision regime. 'evaluate' enforces the thresholds and is the "
            "regime whose risk-coverage curve should be reported. 'observe' "
            "disables abstention for a full-coverage confusion matrix that "
            "isolates rule quality from the rejection policy."
        ),
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--min-score-margin", type=float, default=DEFAULT_MIN_SCORE_MARGIN)
    parser.add_argument(
        "--min-recognized-characters", type=int, default=DEFAULT_MIN_RECOGNIZED_CHARACTERS
    )
    parser.add_argument(
        "--no-risk-coverage",
        action="store_true",
        help="Skip the threshold sweep (it is recomputed from stored scores, so it is cheap)",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    base = manifest.parent
    results = []
    sweep_inputs = []
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
                mode=args.mode,
            )
            classifier_time = (perf_counter() - classifier_started) * 1000.0

            # The OCR-sufficiency gate is part of the decision policy, so a
            # sweep that cannot see this number cannot reproduce the policy.
            recognized_characters = int(features.get("alnum_character_count") or 0)
            sweep_inputs.append(
                {
                    "target_family": row["target_family"],
                    "alnum_character_count": recognized_characters,
                    "ranked": _ranked_from_scores(prediction["candidate_scores"]),
                }
            )
            results.append(
                {
                    "sample_id": row["sample_id"],
                    "rvl_label": row["rvl_label"],
                    "target_family": row["target_family"],
                    "predicted_family": prediction["document_family"],
                    "top_candidate": prediction["top_candidate"],
                    "confidence": prediction["confidence"],
                    "score": prediction["score"],
                    "score_margin": prediction["score_margin"],
                    "decision": prediction["decision"],
                    "reason": prediction["reason"],
                    "alnum_character_count": recognized_characters,
                    "word_count": features.get("word_count"),
                    "measured_page_ratio": features.get("measured_page_ratio"),
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
            "mode": args.mode,
            "confidence_threshold": args.confidence_threshold,
            "min_score_margin": args.min_score_margin,
            "min_recognized_characters": args.min_recognized_characters,
        },
        "metrics": _metrics(results),
    }
    if not args.no_risk_coverage and results:
        report["risk_coverage"] = _risk_coverage(sweep_inputs, args)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({"predictions": str(predictions_path), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
