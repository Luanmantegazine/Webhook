#!/usr/bin/env python3
"""Two perspectives on the same predictions, kept apart on purpose.

**Conventional** asks the question a paper asks: over every document, how often
is the label right? Accuracy, balanced accuracy, macro precision/recall/F1,
weighted F1, a confusion matrix, per-family figures.

**Operational** asks the question a deployment asks: of the documents the
system chose to answer, how often is the answer wrong? Coverage, selective
accuracy, selective risk, accepted-but-wrong counts.

The distinction is not cosmetic. ``fallback``, ``other`` and ``abstained`` are
valid outcomes of this system, not failures: a declined document costs a manual
route, an accepted wrong one costs a wrong route. Only the second is an
operational error, and mixing them produces a number that describes neither.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

ACCEPTED_DECISIONS = frozenset({"classified", "observed"})
DECLINED_DECISIONS = frozenset({"fallback", "abstained"})


def is_accepted(record: dict[str, Any]) -> bool:
    return record.get("decision") in ACCEPTED_DECISIONS


def is_rejection_target(record: dict[str, Any]) -> bool:
    return bool(record.get("is_rejection_target"))


def is_correct(record: dict[str, Any]) -> bool:
    return is_accepted(record) and record.get("predicted_family") == record.get("target_family")


def is_accepted_wrong(record: dict[str, Any]) -> bool:
    """The only operational error: an accepted answer that is wrong.

    A rejection target that gets accepted counts here too — accepting a file
    folder as an invoice is a wrong route, whatever the taxonomy calls it.
    """
    if not is_accepted(record):
        return False
    if is_rejection_target(record):
        return True
    return record.get("predicted_family") != record.get("target_family")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def conventional_metrics(
    records: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    """Every document counts, declined ones included, predicted as ``other``."""
    label_list = list(labels)
    if "other" not in label_list:
        label_list = label_list + ["other"]
    confusion = {
        target: {predicted: 0 for predicted in label_list} for target in label_list
    }
    for record in records:
        target = record.get("target_family") or "other"
        predicted = record.get("predicted_family") if is_accepted(record) else "other"
        if target not in confusion:
            confusion[target] = {name: 0 for name in label_list}
        if predicted not in confusion[target]:
            confusion[target][predicted] = 0
        confusion[target][predicted] += 1

    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    precisions = []
    f1s = []
    supports = []
    weighted_f1_total = 0.0
    correct = 0
    for label in label_list:
        true_positive = confusion.get(label, {}).get(label, 0)
        support = sum(confusion.get(label, {}).values())
        predicted_count = sum(
            confusion.get(target, {}).get(label, 0) for target in confusion
        )
        precision = safe_ratio(true_positive, predicted_count)
        recall = safe_ratio(true_positive, support)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted_count": predicted_count,
        }
        correct += true_positive
        if support:
            supports.append(support)
            weighted_f1_total += f1 * support
        # The macro averages run over classes the corpus actually exercises —
        # present in the ground truth or in the predictions. Including a family
        # with no support and no prediction adds a zero to every average and
        # reports a macro-F1 of 0.4 on a corpus that got every document right.
        if support or predicted_count:
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
    total = sum(sum(row.values()) for row in confusion.values())
    return {
        "sample_count": total,
        "accuracy": round(safe_ratio(correct, total), 4),
        "balanced_accuracy": round(safe_ratio(sum(recalls), len(recalls)), 4),
        "macro_precision": round(safe_ratio(sum(precisions), len(precisions)), 4),
        "macro_recall": round(safe_ratio(sum(recalls), len(recalls)) if recalls else 0.0, 4),
        "macro_f1": round(safe_ratio(sum(f1s), len(f1s)), 4),
        "weighted_f1": round(safe_ratio(weighted_f1_total, sum(supports)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "macro_average_over": sorted(
            label
            for label in label_list
            if per_class[label]["support"] or per_class[label]["predicted_count"]
        ),
        "definition": (
            "Every evaluated document counts; a declined document is scored as a "
            "prediction of 'other'. This is the academic reading, and it treats a "
            "refusal as an error — which the operational block deliberately does not. "
            "Macro averages run over the classes listed in macro_average_over: those "
            "the corpus exercises as truth or as prediction."
        ),
    }


def selective_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The deployment reading: what does an accepted answer cost?"""
    total = len(records)
    accepted = [record for record in records if is_accepted(record)]
    accepted_correct = [record for record in accepted if is_correct(record)]
    accepted_wrong = [record for record in accepted if is_accepted_wrong(record)]
    decisions = Counter(record.get("decision") for record in records)
    rejection_targets = [record for record in records if is_rejection_target(record)]
    rejection_accepted = [record for record in rejection_targets if is_accepted(record)]
    coverage = safe_ratio(len(accepted), total)
    selective_accuracy = safe_ratio(len(accepted_correct), len(accepted))
    return {
        "sample_count": total,
        "accepted_count": len(accepted),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_accuracy, 4),
        "selective_risk": round(1.0 - selective_accuracy if accepted else 0.0, 4),
        "accepted_wrong_count": len(accepted_wrong),
        "operational_error_count": len(accepted_wrong),
        "operational_error_rate": round(safe_ratio(len(accepted_wrong), total), 4),
        "controlled_rejection_count": total - len(accepted),
        "controlled_rejection_rate": round(safe_ratio(total - len(accepted), total), 4),
        "fallback_count": decisions.get("fallback", 0),
        "abstained_count": decisions.get("abstained", 0),
        "rejection_target_count": len(rejection_targets),
        "rejection_target_false_accept_count": len(rejection_accepted),
        "definition": (
            "fallback, other and abstained are valid outcomes, not errors. The only "
            "operational error counted here is an accepted classification that is "
            "wrong, including any accepted rejection target."
        ),
    }


def timing_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    embedding_times = [float(record.get("embedding_time_ms") or 0.0) for record in records]
    classifier_times = [float(record.get("classifier_time_ms") or 0.0) for record in records]
    total_seconds = (sum(embedding_times) + sum(classifier_times)) / 1000.0
    return {
        "embedding_ms_mean": round(safe_ratio(sum(embedding_times), len(embedding_times)), 4),
        "embedding_ms_p95": round(percentile(embedding_times, 0.95), 4),
        "classifier_ms_mean": round(safe_ratio(sum(classifier_times), len(classifier_times)), 4),
        "classifier_ms_p95": round(percentile(classifier_times, 0.95), 4),
        "documents_per_second": round(safe_ratio(len(records), total_seconds), 4)
        if total_seconds
        else 0.0,
        "definition": (
            "Embedding and classification are timed separately: the first is model "
            "inference and dominates, the second is a matrix product over a handful "
            "of centroids."
        ),
    }


def risk_coverage_curve(
    records: Sequence[dict[str, Any]],
    *,
    margin: float = 0.0,
) -> list[dict[str, Any]]:
    """Sweep the similarity threshold; report coverage and risk at each point.

    Declined-by-text documents stay in the denominator at every threshold: a
    curve that drops them reports a coverage the deployment does not have.
    """
    scores = sorted(
        {round(float(record.get("top_similarity") or 0.0), 6) for record in records}
    )
    thresholds = [0.0] + scores + [max(scores) + 0.01 if scores else 1.0]
    curve = []
    for threshold in sorted(set(thresholds)):
        replayed = [_replay(record, threshold, margin) for record in records]
        selective = selective_metrics(replayed)
        curve.append(
            {
                "similarity_threshold": round(float(threshold), 6),
                "minimum_score_margin": round(float(margin), 6),
                "accepted_count": selective["accepted_count"],
                "coverage": selective["coverage"],
                "selective_accuracy": selective["selective_accuracy"],
                "selective_risk": selective["selective_risk"],
                "accepted_wrong_count": selective["accepted_wrong_count"],
                "rejection_target_false_accept_count": selective[
                    "rejection_target_false_accept_count"
                ],
            }
        )
    return curve


def _replay(record: dict[str, Any], threshold: float, margin: float) -> dict[str, Any]:
    """Re-apply the decision policy to a stored row at a new operating point."""
    replayed = dict(record)
    if record.get("decision") == "abstained":
        # Text-level abstention does not move with the threshold.
        return replayed
    similarity = float(record.get("top_similarity") or 0.0)
    score_margin = float(record.get("score_margin") or 0.0)
    if similarity >= float(threshold) and score_margin >= float(margin):
        replayed["decision"] = "classified"
        replayed["predicted_family"] = record.get("top_candidate")
    else:
        replayed["decision"] = "fallback"
        replayed["predicted_family"] = "other"
    return replayed


def operating_point_at_coverage(
    curve: Sequence[dict[str, Any]], target_coverage: float
) -> dict[str, Any] | None:
    """The point whose coverage is closest to a target, for comparisons.

    The gap is reported alongside: a curve can be too coarse to land on the
    requested coverage, and a point half a corpus away from the target would
    otherwise read as "the same operating point".
    """
    if not curve:
        return None
    selected = min(
        curve,
        key=lambda point: (
            abs(point["coverage"] - float(target_coverage)),
            -point["similarity_threshold"],
        ),
    )
    return {
        **selected,
        "requested_coverage": round(float(target_coverage), 6),
        "coverage_gap": round(abs(selected["coverage"] - float(target_coverage)), 6),
    }


def operating_point_at_risk(
    curve: Sequence[dict[str, Any]], target_risk: float
) -> dict[str, Any] | None:
    """The widest coverage whose selective risk stays at or below a target."""
    eligible = [
        point
        for point in curve
        if point["accepted_count"] > 0 and point["selective_risk"] <= float(target_risk)
    ]
    if not eligible:
        return None
    selected = max(
        eligible, key=lambda point: (point["coverage"], -point["similarity_threshold"])
    )
    return {**selected, "requested_selective_risk": round(float(target_risk), 6)}
