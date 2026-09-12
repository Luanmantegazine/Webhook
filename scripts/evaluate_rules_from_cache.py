#!/usr/bin/env python3
"""Evaluate and calibrate Hydra's rules classifier from cached JSON documents.

The evaluator consumes cache manifests pointing to aggregate-document JSON files,
with optional page-word artifacts, without rerunning layout detection, OCR,
translation, or reconstruction.

Reproducibility contract
------------------------
Every run records ``SCHEMA_VERSION``, ``TAXONOMY_VERSION``,
``FEATURE_EXTRACTION_VERSION``, ``CLASSIFIER_VERSION``, ``rule_fingerprint`` and
``feature_fingerprint`` — in ``report.json`` under ``versions`` and on every row
of ``predictions.csv`` — so that no stored result has to be dated by hand to
know what produced it.

Features come from one of exactly two places, and the row says which:

* ``classification_features_path``: the record the benchmark workflow stored,
  scored as it is;
* otherwise re-extracted from ``document``, ``page_sizes``, ``page_words`` and
  ``provenance`` — the same four inputs the workflow feeds to
  ``extract_document_classification_features``.

The run is refused, rather than degraded, when:

* a feature record carries an unsupported schema or feature-extraction version,
  or a fingerprint that does not match its own contents;
* a manifest ``target_family`` is outside the taxonomy;
* a stored artifact names a rule absent from ``RULE_IDS``;
* feature versions or fingerprints are mixed inside one evaluation;
* the shipped taxonomy config contradicts ``tasks/document/rvl_cdip_eval.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.document.rules_classifier_core import (  # noqa: E402
    CLASSIFIER_VERSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    DOCUMENT_FAMILIES,
    FAMILY_CONFIDENCE_THRESHOLDS,
    FAMILY_THRESHOLD_HOLDS,
    FAMILY_THRESHOLD_PROVENANCE,
    FEATURE_EXTRACTION_VERSION,
    FeatureContractError,
    SCORED_FAMILIES,
    RULE_FINGERPRINT,
    RULE_IDS,
    SCHEMA_VERSION,
    TAXONOMY_VERSION,
    available_channels,
    apply_decision_policy,
    classifier_versions,
    effective_family_threshold,
    classify_with_rules,
    extract_classification_features,
    resolve_family_thresholds,
    validate_feature_record,
    validate_rule_ids,
)
from tasks.document.rvl_cdip_eval import (  # noqa: E402
    TaxonomyConfigError,
    is_known_family,
    resolve_evaluation_target,
    taxonomy_descriptor,
    verify_config_file,
)


REQUIRED_MANIFEST_COLUMNS = ("sample_id", "rvl_label", "document_path")
PAGE_WORDS_PATH_COLUMN = "page_words_path"
#: Features stored by the benchmark workflow. When present they are used as
#: they are — the point of caching them is that the evaluation scores exactly
#: what the pipeline produced — after being checked against this classifier's
#: feature contract. When absent, features are re-extracted from ``document``,
#: ``page_sizes``, ``page_words`` and ``provenance``, which is the same input
#: set the workflow hands to ``extract_document_classification_features``.
FEATURES_PATH_COLUMN = "classification_features_path"
PROVENANCE_PATH_COLUMN = "provenance_path"
VERSION_PREDICTION_COLUMNS = [
    "effective_family_threshold",
    "global_confidence_threshold",
    "schema_version",
    "taxonomy_version",
    "feature_extraction_version",
    "classifier_version",
    "rule_fingerprint",
    "feature_fingerprint",
    "feature_source",
    "applied_confidence_threshold",
    "family_gate_status",
    "family_gate_reason",
    "gated_families",
]
SPLIT_COLUMNS = ("source_manifest_split", "source_split", "split")
CACHED_ERROR_COLUMNS = ("cached_manifest_input_error", "cached_input_error")
SOURCE_ERROR_COLUMNS = ("source_manifest_input_error", "source_input_error")
SUPPORTED_ERROR_COLUMNS = frozenset(CACHED_ERROR_COLUMNS + SOURCE_ERROR_COLUMNS)
ACCEPTED_DECISIONS = frozenset({"classified", "observed"})
METRIC_SCHEMA_VERSION = 2
DEFAULT_CONFIDENCE_GRID = "0.45,0.50,0.55,0.60,0.65,0.70"
DEFAULT_MARGIN_GRID = "0.06,0.08,0.10,0.12,0.15"
SCOPABLE_TARGET_FAMILIES = (
    "correspondence",
    "form_structured",
    "research_paper",
    "news_article",
)
GEOMETRY_PREDICTION_COLUMNS = [
    "geometry_page_ratio",
    "geometry_pages",
    "tab_stop_count",
    "wide_gap_line_ratio",
    "label_value_line_ratio",
    "top_band_header_ratio",
    "line_pitch_regularity",
    "body_wide_gap_line_ratio",
    "centered_line_ratio",
    "word_line_count",
    "right_edge_regularity",
    "two_column_ratio",
    # v6: the features the presentation diagnosis turns on. Reading the v5
    # false accepts required joining predictions back to the feature cache,
    # which is exactly the kind of step that does not get taken.
    "narrative_line_ratio",
    "picture_area_ratio",
    "relevant_picture_count",
    "table_area_ratio",
    "landscape_ratio",
]
PREDICTION_CSV_HEADERS = [
    "sample_id", "rvl_label", "target_family", "source_target_family", "target_kind",
    "rejection_label", "is_rejection_target", "in_scope_target", "target_scope_status", "source_split",
    "predicted_family", "top_candidate", "runner_up",
    "confidence", "score", "score_margin", "decision", "reason", "recognized_characters",
    "feature_extraction_time_ms", "classifier_time_ms", "rules_triggered", "rules_by_family",
    "candidate_scores", "rule_indicators", "evidence_channels",
] + VERSION_PREDICTION_COLUMNS + GEOMETRY_PREDICTION_COLUMNS
REVIEW_CSV_HEADERS = [
    "sample_id", "rvl_label", "target_family", "source_target_family", "target_kind",
    "rejection_label", "predicted_family",
    # v6: ``correct`` was one column answering three different questions, so a
    # rejection target that was wrongly accepted and an out-of-scope document
    # that was correctly routed both read as "False" and sorted together.
    "canonical_correct", "scope_correct", "is_unsafe_accept", "is_rejection_false_accept",
    "review_priority", "review_priority_label",
    "decision", "reason", "confidence", "score", "effective_family_threshold",
    "score_margin", "top_candidate", "runner_up", "gate_status", "gate_reason",
    "fired_rules", "review_rank",
]
FAMILY_RISK_COVERAGE_CSV_HEADERS = [
    "target_family", "confidence_threshold", "effective_family_threshold",
    "min_score_margin",
    "min_recognized_characters", "classification_mode", "is_selected_operating_point",
    "positive_count", "negative_count", "negative_other_target_count",
    "negative_control_count", "negative_rejection_count", "included_sample_count",
    "accepted_count", "true_positive_count", "false_positive_count",
    "rejection_false_accept_count",
    "coverage", "precision", "risk",
]
PER_LABEL_CSV_HEADERS = [
    "rvl_label", "target_family", "target_kind", "sample_count", "accepted_count",
    "coverage", "canonical_correct_count", "accepted_correct_count",
    "accepted_routing_accuracy", "unsafe_accept_count", "unsafe_accept_rate",
    "top_predicted_families", "top_reasons",
]
ROUTING_CSV_HEADERS = [
    "family", "effective_family_threshold", "true_target_count",
    "top_candidate_true_positive_count", "top_candidate_false_positive_count",
    "top_candidate_precision", "top_candidate_recall",
    "accepted_true_positive_count", "accepted_false_positive_count",
    "accepted_false_negative_count", "accepted_precision", "accepted_recall",
    "accepted_rejection_false_accept_count",
]
CONFUSION_CSV_HEADERS = [
    "target_family", "source_target_families", "predicted_family", "error_count",
    "top_reasons", "top_fired_rules", "representative_sample_ids",
]
INPUT_ERROR_CSV_HEADERS = [
    "row_number", "sample_id", "rvl_label", "target_family", "source_split", "error_source",
    "error_message",
]
CALIBRATION_CSV_HEADERS = [
    "confidence_threshold", "min_score_margin", "macro_f1",
    "macro_f1_classified_excluding_other", "classified_response_count",
    "classified_response_denominator", "coverage", "accuracy_on_accepted",
    "classified_accuracy", "classified_correct_count", "meets_min_coverage",
    "meets_min_accepted_accuracy",
    "meets_constraints", "rank",
]


class EvaluationError(RuntimeError):
    """Base class for expected evaluation failures."""


class ManifestValidationError(EvaluationError):
    """Invalid manifest schema or records."""


class CalibrationValidationError(EvaluationError):
    """Calibration inputs violate validation-only constraints."""


class BaselineValidationError(EvaluationError):
    """Baseline report has invalid schema or values."""


class VersionCompatibilityError(EvaluationError):
    """Inputs cannot be evaluated together, or cannot be evaluated at all.

    Four distinct situations, all of which produce numbers that look ordinary
    and mean nothing:

    * a feature record written under a different schema or feature-extraction
      version than this classifier reads;
    * a ground-truth family outside the taxonomy;
    * a stored artifact naming a rule this classifier does not declare;
    * a *mixture* of versions or fingerprints inside one evaluation, where the
      aggregate metric describes no single system at all.
    """


@dataclass
class CachedSample:
    row_number: int
    sample_id: str
    rvl_label: str
    target_family: str | None
    source_target_family: str
    is_rejection_target: bool
    source_split: str
    features: dict[str, Any]
    feature_extraction_time_ms: float
    feature_source: str = "reextracted"


class FeatureVersionLedger:
    """Refuse a mixture of feature versions inside one evaluation.

    Pooling records from two feature-extraction versions produces a metric that
    describes neither. The first record fixes the expected descriptor; the
    second that disagrees names both samples and stops the run.
    """

    def __init__(self) -> None:
        self.descriptor: dict[str, Any] | None = None
        self.reference_sample: str = ""
        self.sources: Counter = Counter()

    def record(self, sample_id: str, descriptor: dict[str, Any], source: str) -> None:
        self.sources[source] += 1
        if self.descriptor is None:
            self.descriptor = dict(descriptor)
            self.reference_sample = sample_id
            return
        if dict(descriptor) != self.descriptor:
            differing = sorted(
                key
                for key in set(descriptor) | set(self.descriptor)
                if descriptor.get(key) != self.descriptor.get(key)
            )
            raise VersionCompatibilityError(
                "Mixed feature versions in one evaluation: sample_id="
                f"{sample_id} disagrees with sample_id={self.reference_sample} on "
                f"{', '.join(differing)} "
                f"({ {key: descriptor.get(key) for key in differing} } vs "
                f"{ {key: self.descriptor.get(key) for key in differing} }). "
                "Re-extract the whole cache with one feature-extraction version."
            )


@dataclass
class ManifestInputErrorRecord:
    row_number: int
    sample_id: str
    rvl_label: str
    target_family: str
    source_split: str
    error_source: str
    error_message: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _non_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_non_empty(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = _non_empty(row.get(name, ""))
        if value:
            return value
    return ""


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_target_families(value: str) -> list[str]:
    if not _non_empty(value):
        return []
    normalized = []
    for token in value.split(","):
        candidate = "_".join(_non_empty(token).lower().replace("-", "_").split())
        if not candidate:
            continue
        normalized.append(candidate)
    if not normalized:
        return []
    unknown = sorted(set(normalized) - set(SCOPABLE_TARGET_FAMILIES))
    if unknown:
        raise ManifestValidationError(
            "--target-families only supports: "
            f"{', '.join(SCOPABLE_TARGET_FAMILIES)}. Unknown: {', '.join(unknown)}."
        )
    return sorted(set(normalized))


def _target_scope_status(
    *,
    is_rejection_target: bool,
    target_family: str | None,
    target_families: frozenset[str],
) -> tuple[bool | None, str]:
    if is_rejection_target:
        return (False, "rejection_target") if target_families else (None, "")
    if not target_families:
        return None, ""
    in_scope = bool(target_family and target_family in target_families)
    return in_scope, "in_scope_target" if in_scope else "out_of_scope_target"


def _parse_grid(text: str, label: str) -> list[float]:
    values = []
    for token in text.split(","):
        normalized = token.strip()
        if not normalized:
            continue
        try:
            numeric = float(normalized)
        except ValueError as error:
            raise CalibrationValidationError(
                f"Invalid {label} grid value '{normalized}'. Use comma-separated decimals."
            ) from error
        if numeric < 0.0:
            raise CalibrationValidationError(f"{label} values must be >= 0.0 (got {numeric}).")
        values.append(round(numeric, 6))
    unique_sorted = sorted(set(values))
    if not unique_sorted:
        raise CalibrationValidationError(f"{label} grid must have at least one numeric value.")
    return unique_sorted


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _is_rejection_target(record: dict[str, Any]) -> bool:
    return bool(record.get("is_rejection_target", False))


def _is_correct_rejection_decision(record: dict[str, Any]) -> bool:
    return record.get("decision") in {"abstained", "fallback"}


def _is_accepted(record: dict[str, Any]) -> bool:
    return record.get("decision") in ACCEPTED_DECISIONS


def _is_rejection_false_accept(record: dict[str, Any]) -> bool:
    """A document the classifier was supposed to decline, and did not.

    The most expensive error in the taxonomy: a file folder or a handwritten
    page routed to a family is a document sent somewhere it does not belong,
    with a confidence attached.
    """
    return _is_rejection_target(record) and _is_accepted(record)


def _is_accepted_wrong_family(record: dict[str, Any]) -> bool:
    """An accepted answer that names the wrong family."""
    if _is_rejection_target(record) or not _is_accepted(record):
        return False
    return record.get("predicted_family") != record.get("target_family")


def _is_unsafe_accept(record: dict[str, Any]) -> bool:
    """Any accepted decision that is wrong.

    Unsafe accepts are the number a deployment cares about: a refusal costs
    coverage, an unsafe accept costs trust. Kept distinct from "not correct",
    which also counts refusals.
    """
    return _is_rejection_false_accept(record) or _is_accepted_wrong_family(record)


def _record_is_correct(record: dict[str, Any], target_families: frozenset[str] | None = None) -> bool:
    if _is_rejection_target(record):
        return _is_correct_rejection_decision(record)
    if target_families and record["target_family"] not in target_families:
        return False
    return record["target_family"] == record["predicted_family"]


def _classified_metrics(
    classified_records: list[dict[str, Any]],
    evaluated_sample_count: int,
    labels: list[str],
) -> dict[str, Any]:
    confusion = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for record in classified_records:
        truth = record["target_family"]
        prediction = record["predicted_family"]
        if truth not in confusion:
            confusion[truth] = {candidate: 0 for candidate in labels}
        if prediction not in confusion[truth]:
            confusion[truth][prediction] = 0
        confusion[truth][prediction] += 1

    per_class = {}
    for label in labels:
        tp = sum(
            1
            for r in classified_records
            if r["target_family"] == label and r["predicted_family"] == label
        )
        fp = sum(
            1
            for r in classified_records
            if r["target_family"] != label and r["predicted_family"] == label
        )
        fn = sum(
            1
            for r in classified_records
            if r["target_family"] == label and r["predicted_family"] != label
        )
        support = sum(1 for r in classified_records if r["target_family"] == label)
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
    correct_count = sum(
        record["target_family"] == record["predicted_family"] for record in classified_records
    )
    classified_response_count = len(classified_records)
    coverage = (
        classified_response_count / evaluated_sample_count if evaluated_sample_count else 0.0
    )
    accuracy = correct_count / classified_response_count if classified_response_count else 0.0
    return {
        "labels": labels,
        "classified_response_count": classified_response_count,
        "classified_response_denominator": evaluated_sample_count,
        "coverage": round(coverage, 4),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(statistics.fmean(item["precision"] for item in supported), 4)
        if supported
        else 0.0,
        "macro_recall": round(statistics.fmean(item["recall"] for item in supported), 4)
        if supported
        else 0.0,
        "macro_f1": round(statistics.fmean(item["f1"] for item in supported), 4)
        if supported
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _routing_metrics(
    records: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Routing quality for every scorable family.

    v5 reported detection metrics only for the four scoped families, so the
    families nobody was scoping — the ones quietly absorbing other families'
    documents — had no precision number anywhere in the report. Every scorable
    family is measured here, with rejection targets included as negatives:
    accepting a file folder as ``financial_document`` is a false positive for
    ``financial_document``, and excluding it from that family's denominator is
    how it stayed invisible.
    """
    classifier_eligible = [record for record in records if not _is_rejection_target(record)]
    accepted = [record for record in records if _is_accepted(record)]
    accepted_eligible = [record for record in accepted if not _is_rejection_target(record)]
    accepted_correct = sum(
        record["predicted_family"] == record["target_family"] for record in accepted_eligible
    )
    accepted_wrong_family_count = sum(_is_accepted_wrong_family(record) for record in records)
    rejection_false_accept_count = sum(_is_rejection_false_accept(record) for record in records)
    unsafe_accept_count = accepted_wrong_family_count + rejection_false_accept_count

    per_family: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for family in SCORED_FAMILIES:
        threshold = effective_family_threshold(family, confidence_threshold)
        support = sum(record["target_family"] == family for record in classifier_eligible)
        top_tp = sum(
            record["target_family"] == family and record.get("top_candidate") == family
            for record in classifier_eligible
        )
        top_fp = sum(
            record["target_family"] != family and record.get("top_candidate") == family
            for record in records
        )
        accepted_tp = sum(
            not _is_rejection_target(record)
            and record["target_family"] == family
            and _is_accepted(record)
            and record["predicted_family"] == family
            for record in records
        )
        accepted_fp = sum(
            _is_accepted(record)
            and record["predicted_family"] == family
            and (_is_rejection_target(record) or record["target_family"] != family)
            for record in records
        )
        accepted_fn = sum(
            not _is_rejection_target(record)
            and record["target_family"] == family
            and not (_is_accepted(record) and record["predicted_family"] == family)
            for record in records
        )
        rejection_fp = sum(
            _is_rejection_false_accept(record) and record["predicted_family"] == family
            for record in records
        )
        row = {
            "family": family,
            "effective_family_threshold": threshold,
            "true_target_count": support,
            "top_candidate_true_positive_count": top_tp,
            "top_candidate_false_positive_count": top_fp,
            "top_candidate_precision": round(_safe_ratio(top_tp, top_tp + top_fp), 4),
            "top_candidate_recall": round(_safe_ratio(top_tp, support), 4),
            "accepted_true_positive_count": accepted_tp,
            "accepted_false_positive_count": accepted_fp,
            "accepted_false_negative_count": accepted_fn,
            "accepted_precision": round(_safe_ratio(accepted_tp, accepted_tp + accepted_fp), 4),
            "accepted_recall": round(_safe_ratio(accepted_tp, accepted_tp + accepted_fn), 4),
            "accepted_rejection_false_accept_count": rejection_fp,
        }
        per_family[family] = {key: value for key, value in row.items() if key != "family"}
        csv_rows.append(row)

    payload = {
        "definition": (
            "Routing quality over every scorable family. Positives are documents whose "
            "canonical target is the family; negatives are every other document "
            "including rejection targets, so accepting a file folder counts against the "
            "family that accepted it."
        ),
        "families": list(SCORED_FAMILIES),
        "accepted_count": len(accepted),
        "accepted_classifier_eligible_count": len(accepted_eligible),
        "accepted_routing_accuracy": round(
            _safe_ratio(accepted_correct, len(accepted_eligible)), 4
        ),
        "accepted_correct_count": accepted_correct,
        "accepted_wrong_family_count": accepted_wrong_family_count,
        "rejection_false_accept_count": rejection_false_accept_count,
        "unsafe_accept_count": unsafe_accept_count,
        "unsafe_accept_rate": round(_safe_ratio(unsafe_accept_count, len(accepted)), 4),
        "unsafe_accept_definition": (
            "An accepted decision (classified or observed) that names the wrong family, "
            "plus every rejection target that was accepted at all."
        ),
        "per_family": per_family,
    }
    return payload, csv_rows


def _per_rvl_label_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics per original RVL-CDIP label, not only per mapped family.

    Three labels collapse onto ``correspondence`` and two onto each of several
    other families, so a family-level number hides which *source class* the
    classifier actually handles. A family at 70% built from one label at 100%
    and another at 10% is not a family at 70%.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_non_empty(record.get("rvl_label")) or "<unlabelled>", []).append(record)

    payload: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for label, rows in sorted(grouped.items()):
        accepted = [record for record in rows if _is_accepted(record)]
        is_rejection = bool(rows and _is_rejection_target(rows[0]))
        canonical_correct = sum(_record_is_correct(record) for record in rows)
        accepted_correct = sum(
            not _is_rejection_target(record)
            and record["predicted_family"] == record["target_family"]
            for record in accepted
        )
        unsafe = sum(_is_unsafe_accept(record) for record in rows)
        predicted = Counter(record["predicted_family"] for record in rows)
        reasons = Counter(_non_empty(record.get("reason")) for record in rows if _non_empty(record.get("reason")))
        entry = {
            "target_family": rows[0]["target_family"],
            "target_kind": rows[0]["target_kind"],
            "sample_count": len(rows),
            "accepted_count": len(accepted),
            "coverage": round(_safe_ratio(len(accepted), len(rows)), 4),
            "canonical_correct_count": canonical_correct,
            "accepted_correct_count": accepted_correct,
            "accepted_routing_accuracy": round(_safe_ratio(accepted_correct, len(accepted)), 4),
            "unsafe_accept_count": unsafe,
            "unsafe_accept_rate": round(_safe_ratio(unsafe, len(accepted)), 4),
            "predicted_families": dict(sorted(predicted.items())),
            "reasons": dict(sorted(reasons.items())),
            "correct_criterion": (
                "abstained or fallback" if is_rejection else "predicted_family == target_family"
            ),
        }
        payload[label] = entry
        csv_rows.append(
            {
                "rvl_label": label,
                "target_family": entry["target_family"],
                "target_kind": entry["target_kind"],
                "sample_count": entry["sample_count"],
                "accepted_count": entry["accepted_count"],
                "coverage": entry["coverage"],
                "canonical_correct_count": entry["canonical_correct_count"],
                "accepted_correct_count": entry["accepted_correct_count"],
                "accepted_routing_accuracy": entry["accepted_routing_accuracy"],
                "unsafe_accept_count": entry["unsafe_accept_count"],
                "unsafe_accept_rate": entry["unsafe_accept_rate"],
                "top_predicted_families": "; ".join(
                    f"{name}:{count}" for name, count in predicted.most_common(5)
                ),
                "top_reasons": "; ".join(f"{name}:{count}" for name, count in reasons.most_common(5)),
            }
        )
    return payload, csv_rows


def _scoped_detection_metrics(
    classifier_eligible_records: list[dict[str, Any]],
    target_families: frozenset[str],
) -> dict[str, Any]:
    if not target_families:
        return {
            "enabled": False,
            "target_families": [],
            "note": "Set --target-families to enable scoped detection metrics.",
        }

    ordered_targets = sorted(target_families)
    in_scope_records = [
        record for record in classifier_eligible_records if record["target_family"] in target_families
    ]
    out_of_scope_records = [
        record for record in classifier_eligible_records if record["target_family"] not in target_families
    ]
    in_scope_accepted = [
        record for record in in_scope_records if record["decision"] in ACCEPTED_DECISIONS
    ]
    in_scope_accepted_correct = sum(
        record["predicted_family"] == record["target_family"] for record in in_scope_accepted
    )

    top_tp_total = 0
    top_fp_total = 0
    top_fn_total = 0
    accepted_tp_total = 0
    accepted_fp_total = 0
    accepted_fn_total = 0
    per_family = {}
    for family in ordered_targets:
        support = sum(record["target_family"] == family for record in classifier_eligible_records)
        top_tp = sum(
            record["target_family"] == family and record.get("top_candidate") == family
            for record in classifier_eligible_records
        )
        top_fp = sum(
            record["target_family"] != family and record.get("top_candidate") == family
            for record in classifier_eligible_records
        )
        top_fn = sum(
            record["target_family"] == family and record.get("top_candidate") != family
            for record in classifier_eligible_records
        )
        accepted_tp = sum(
            record["target_family"] == family
            and record["decision"] in ACCEPTED_DECISIONS
            and record["predicted_family"] == family
            for record in classifier_eligible_records
        )
        accepted_fp = sum(
            record["target_family"] != family
            and record["decision"] in ACCEPTED_DECISIONS
            and record["predicted_family"] == family
            for record in classifier_eligible_records
        )
        accepted_fn = sum(
            record["target_family"] == family
            and not (
                record["decision"] in ACCEPTED_DECISIONS and record["predicted_family"] == family
            )
            for record in classifier_eligible_records
        )
        top_tp_total += top_tp
        top_fp_total += top_fp
        top_fn_total += top_fn
        accepted_tp_total += accepted_tp
        accepted_fp_total += accepted_fp
        accepted_fn_total += accepted_fn
        per_family[family] = {
            "true_target_count": support,
            "top_candidate_detection": {
                "true_positive_count": top_tp,
                "false_positive_count": top_fp,
                "false_negative_count": top_fn,
                "predicted_target_count": top_tp + top_fp,
                "precision": round(_safe_ratio(top_tp, top_tp + top_fp), 4),
                "recall": round(_safe_ratio(top_tp, top_tp + top_fn), 4),
            },
            "accepted_detection": {
                "true_positive_count": accepted_tp,
                "false_positive_count": accepted_fp,
                "false_negative_count": accepted_fn,
                "accepted_predicted_target_count": accepted_tp + accepted_fp,
                "precision": round(_safe_ratio(accepted_tp, accepted_tp + accepted_fp), 4),
                "recall": round(_safe_ratio(accepted_tp, accepted_tp + accepted_fn), 4),
            },
        }

    top_macro_precision = statistics.fmean(
        row["top_candidate_detection"]["precision"] for row in per_family.values()
    )
    top_macro_recall = statistics.fmean(
        row["top_candidate_detection"]["recall"] for row in per_family.values()
    )
    accepted_macro_precision = statistics.fmean(
        row["accepted_detection"]["precision"] for row in per_family.values()
    )
    accepted_macro_recall = statistics.fmean(
        row["accepted_detection"]["recall"] for row in per_family.values()
    )

    out_of_scope_top_as_target = sum(
        record.get("top_candidate") in target_families for record in out_of_scope_records
    )
    out_of_scope_accepted_as_target = sum(
        record["decision"] in ACCEPTED_DECISIONS and record["predicted_family"] in target_families
        for record in out_of_scope_records
    )
    return {
        "enabled": True,
        "target_families": ordered_targets,
        "in_scope_target_count": len(in_scope_records),
        "out_of_scope_count": len(out_of_scope_records),
        "rejection_target_count": 0,
        "in_scope_coverage": {
            "accepted_in_scope_count": len(in_scope_accepted),
            "in_scope_target_count": len(in_scope_records),
            "coverage": round(_safe_ratio(len(in_scope_accepted), len(in_scope_records)), 4),
        },
        "in_scope_accepted_conditional_accuracy": {
            "correct_count": in_scope_accepted_correct,
            "accepted_in_scope_count": len(in_scope_accepted),
            "accuracy": round(_safe_ratio(in_scope_accepted_correct, len(in_scope_accepted)), 4),
            "definition": "Correct only when accepted decision predicts the exact in-scope canonical target.",
        },
        "top_candidate_detection": {
            "macro_precision": round(top_macro_precision, 4),
            "macro_recall": round(top_macro_recall, 4),
            "micro_precision": round(_safe_ratio(top_tp_total, top_tp_total + top_fp_total), 4),
            "micro_recall": round(_safe_ratio(top_tp_total, top_tp_total + top_fn_total), 4),
            "true_positive_count": top_tp_total,
            "false_positive_count": top_fp_total,
            "false_negative_count": top_fn_total,
            "definition": (
                "Threshold-independent: positive if top_candidate equals target family."
            ),
        },
        "accepted_detection": {
            "macro_precision": round(accepted_macro_precision, 4),
            "macro_recall": round(accepted_macro_recall, 4),
            "micro_precision": round(
                _safe_ratio(accepted_tp_total, accepted_tp_total + accepted_fp_total), 4
            ),
            "micro_recall": round(
                _safe_ratio(accepted_tp_total, accepted_tp_total + accepted_fn_total), 4
            ),
            "true_positive_count": accepted_tp_total,
            "false_positive_count": accepted_fp_total,
            "false_negative_count": accepted_fn_total,
            "definition": (
                "Positive only for accepted decisions (classified or observed) whose "
                "predicted_family equals target family."
            ),
        },
        "out_of_scope_leakage": {
            "top_candidate_as_target_count": out_of_scope_top_as_target,
            "top_candidate_leakage_fraction": round(
                _safe_ratio(out_of_scope_top_as_target, len(out_of_scope_records)), 4
            ),
            "accepted_as_target_count": out_of_scope_accepted_as_target,
            "accepted_leakage_fraction": round(
                _safe_ratio(out_of_scope_accepted_as_target, len(out_of_scope_records)), 4
            ),
            "denominator_out_of_scope_count": len(out_of_scope_records),
        },
        "per_target_family": per_family,
    }


def _metrics(
    records: list[dict[str, Any]],
    target_families: frozenset[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    scoped_target_families = frozenset(target_families or ())
    labels = sorted(scoped_target_families) if scoped_target_families else list(DOCUMENT_FAMILIES)
    classifier_eligible_records = [
        record for record in records if not _is_rejection_target(record)
    ]
    rejection_records = [record for record in records if _is_rejection_target(record)]
    evaluation_records = (
        [
            record
            for record in classifier_eligible_records
            if record["target_family"] in scoped_target_families
        ]
        if scoped_target_families
        else classifier_eligible_records
    )
    classified = [
        record for record in evaluation_records if record["decision"] in ACCEPTED_DECISIONS
    ]
    classified_metrics = _classified_metrics(classified, len(evaluation_records), labels)
    macro_f1_excluding_other = _classified_metrics(
        classified,
        len(evaluation_records),
        [label for label in labels if label != "other"],
    )
    decision_counter = Counter(record["decision"] for record in evaluation_records)
    rejection_decision_counter = Counter(record["decision"] for record in rejection_records)
    explicit_other = [
        record
        for record in classified
        if record["predicted_family"] == "other" and record["decision"] in ACCEPTED_DECISIONS
    ]
    fallback = decision_counter.get("fallback", 0)
    abstained = decision_counter.get("abstained", 0)
    ranking_canonical_matches = sum(
        record.get("top_candidate") == record["target_family"]
        for record in classifier_eligible_records
    )
    classifier_times = [float(record["classifier_time_ms"]) for record in records]
    feature_times = [float(record["feature_extraction_time_ms"]) for record in records]
    scoped_detection = _scoped_detection_metrics(classifier_eligible_records, scoped_target_families)
    if scoped_target_families:
        scoped_detection["rejection_target_count"] = len(rejection_records)

    ranking_accuracy = {
        "correct_count": ranking_canonical_matches,
        "denominator": len(classifier_eligible_records),
        "accuracy": round(
            _safe_ratio(ranking_canonical_matches, len(classifier_eligible_records)), 4
        ),
        "definition": (
            "Global diagnostic: top_candidate equals canonical target_family, independent "
            "of decision thresholds and excluding rejection targets."
        ),
    }
    if scoped_target_families:
        ranking_accuracy = {
            "canonical_match_count": ranking_canonical_matches,
            "denominator": len(classifier_eligible_records),
            "canonical_match_fraction": round(
                _safe_ratio(ranking_canonical_matches, len(classifier_eligible_records)), 4
            ),
            "role": "diagnostic_only",
            "definition": (
                "Global diagnostic: top_candidate equals the canonical target family, "
                "independent of decision thresholds and excluding rejection targets. "
                "This is not scoped correctness: out-of-scope canonical matches are "
                "forwarded and are not counted as correct classification."
            ),
        }
    else:
        ranking_accuracy["definition"] = (
            "top_candidate equals canonical target_family; independent of decision "
            "thresholds and excludes rejection targets."
        )

    routing, _routing_rows = _routing_metrics(records, confidence_threshold)
    per_rvl_label, _label_rows = _per_rvl_label_metrics(records)

    result = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "sample_count": len(records),
        # Promoted to the top level: these four answer "can an accepted answer
        # be trusted", which no other block in this report answers directly.
        "accepted_routing_accuracy": routing["accepted_routing_accuracy"],
        "accepted_wrong_family_count": routing["accepted_wrong_family_count"],
        "unsafe_accept_count": routing["unsafe_accept_count"],
        "unsafe_accept_rate": routing["unsafe_accept_rate"],
        "routing": routing,
        "per_rvl_label": per_rvl_label,
        "classifier_eligible_sample_count": len(classifier_eligible_records),
        "rejection_target_count": len(rejection_records),
        "classification_coverage": {
            "classified_response_count": classified_metrics["classified_response_count"],
            "evaluated_sample_count": len(evaluation_records),
            "coverage": classified_metrics["coverage"],
        },
        "classified_metrics": classified_metrics,
        "classified_macro_f1_excluding_other": {
            "labels": macro_f1_excluding_other["labels"],
            "classified_response_count": macro_f1_excluding_other["classified_response_count"],
            "classified_response_denominator": macro_f1_excluding_other[
                "classified_response_denominator"
            ],
            "coverage": macro_f1_excluding_other["coverage"],
            "macro_f1": macro_f1_excluding_other["macro_f1"],
        },
        "ranking_accuracy": ranking_accuracy,
        "accuracy": classified_metrics["accuracy"],
        "macro_precision": classified_metrics["macro_precision"],
        "macro_recall": classified_metrics["macro_recall"],
        "macro_f1": classified_metrics["macro_f1"],
        "coverage": classified_metrics["coverage"],
        "abstention_or_fallback_rate": round(
            _safe_ratio(abstained + fallback, len(evaluation_records)), 4
        ),
        "accuracy_on_accepted": classified_metrics["accuracy"],
        "per_class": classified_metrics["per_class"],
        "confusion_matrix": classified_metrics["confusion_matrix"],
        "legacy_field_metadata": {
            "deprecated": True,
            "replacement": "classified_metrics",
            "interpretation": (
                "accuracy, macro_*, per_class, and confusion_matrix are aliases for "
                "classified_metrics and exclude abstained and fallback responses. "
                "Reports without metric_schema_version=2 used different, all-decision semantics."
            ),
        },
        "decision_aware": {
            "accepted_decisions": sorted(ACCEPTED_DECISIONS),
            "decision_counts": dict(sorted(decision_counter.items())),
            "explicit_other_count": len(explicit_other),
            "explicit_other_rate_on_accepted": round(len(explicit_other) / len(classified), 4)
            if classified
            else 0.0,
            "fallback_count": fallback,
            "fallback_rate": round(_safe_ratio(fallback, len(evaluation_records)), 4),
            "abstained_count": abstained,
            "abstained_rate": round(_safe_ratio(abstained, len(evaluation_records)), 4),
        },
        "rejection_quality": {
            "rejection_target_count": len(rejection_records),
            "correct_rejection_count": sum(
                _is_correct_rejection_decision(record) for record in rejection_records
            ),
            "incorrect_rejection_count": sum(
                not _is_correct_rejection_decision(record) for record in rejection_records
            ),
            "accuracy": round(
                sum(_is_correct_rejection_decision(record) for record in rejection_records)
                / len(rejection_records),
                4,
            )
            if rejection_records
            else 0.0,
            "decision_counts": dict(sorted(rejection_decision_counter.items())),
            "criterion": (
                "A file-folder or handwritten rejection target is correct only when "
                "the classifier decision is abstained or fallback; classified and "
                "observed are false accepts. Rejection targets are not classifier families."
            ),
        },
        "timing_ms": {
            "feature_mean": round(statistics.fmean(feature_times), 4) if feature_times else 0.0,
            "classifier_mean": round(statistics.fmean(classifier_times), 4) if classifier_times else 0.0,
            "classifier_p50": round(_percentile(classifier_times, 0.50), 4),
            "classifier_p95": round(_percentile(classifier_times, 0.95), 4),
            "classifier_p99": round(_percentile(classifier_times, 0.99), 4),
        },
    }
    if scoped_target_families:
        result["target_scope"] = {
            "enabled": True,
            "target_families": sorted(scoped_target_families),
            "evaluation_sample_count": len(evaluation_records),
            "out_of_scope_sample_count": len(classifier_eligible_records) - len(evaluation_records),
            "definition": (
                "Classifier accuracy-style metrics evaluate only in-scope target families; "
                "out-of-scope rows are forwarded for leakage diagnostics."
            ),
        }
        result["target_scope_detection"] = scoped_detection
    return result


def _calibration_metrics(records: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return unrounded classified-only objective values for grid selection."""
    classifier_eligible_records = [
        record for record in records if not _is_rejection_target(record)
    ]
    classified = [
        record for record in classifier_eligible_records
        if record["decision"] in ACCEPTED_DECISIONS
    ]
    f1_scores = []
    for label in DOCUMENT_FAMILIES:
        if label == "other":
            continue
        support = sum(record["target_family"] == label for record in classified)
        if not support:
            continue
        true_positive = sum(
            record["target_family"] == label and record["predicted_family"] == label
            for record in classified
        )
        false_positive = sum(
            record["target_family"] != label and record["predicted_family"] == label
            for record in classified
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    macro_f1 = statistics.fmean(f1_scores) if f1_scores else 0.0
    coverage = (
        len(classified) / len(classifier_eligible_records)
        if classifier_eligible_records
        else 0.0
    )
    classified_accuracy = (
        sum(record["target_family"] == record["predicted_family"] for record in classified)
        / len(classified)
        if classified
        else 0.0
    )
    return macro_f1, coverage, classified_accuracy


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_headers: list[str] | None = None) -> None:
    headers = list(preferred_headers or [])
    if not headers:
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    headers.append(key)
                    seen.add(key)
    if not headers:
        headers = ["sample_id"]
    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                serialized[key] = value
        serialized_rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(serialized_rows)


def _serialize_prediction_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    serialised = dict(record)
    for field in (
        "rules_triggered",
        "rules_by_family",
        "candidate_scores",
        "rule_indicators",
        "evidence_channels",
        "gated_families",
    ):
        serialised[field] = json.dumps(serialised[field], ensure_ascii=False, sort_keys=True)
    return serialised


def _predict_samples(
    samples: list[CachedSample],
    confidence_threshold: float,
    min_score_margin: float,
    min_recognized_characters: int,
    mode: str,
    target_families: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    scoped_target_families = frozenset(target_families or ())
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        classifier_started = perf_counter()
        prediction = classify_with_rules(
            sample.features,
            confidence_threshold=confidence_threshold,
            min_score_margin=min_score_margin,
            min_recognized_characters=min_recognized_characters,
            mode=mode,
            include_indicators=True,
        )
        classifier_time = (perf_counter() - classifier_started) * 1000.0
        # An indicator vector naming a rule this build does not declare means
        # the stored artifact and the scorer disagree about the rule set.
        try:
            validate_rule_ids(
                prediction.get("rule_indicators", {}),
                source=f"sample_id={sample.sample_id} rule_indicators",
            )
        except FeatureContractError as error:
            raise VersionCompatibilityError(str(error)) from error
        gates = prediction.get("evidence", {}).get("family_gates", {}) or {}
        # The gate to report is the one belonging to the family that was
        # actually judged: on a refusal ``document_family`` is ``other``, which
        # has no gate, and the interesting record is the family that was
        # refused.
        decided_family = prediction.get("top_candidate") or prediction["document_family"]
        in_scope_target, target_scope_status = _target_scope_status(
            is_rejection_target=sample.is_rejection_target,
            target_family=sample.target_family,
            target_families=scoped_target_families,
        )
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "rvl_label": sample.rvl_label,
                "target_family": sample.target_family,
                "source_target_family": sample.source_target_family,
                "target_kind": "rejection" if sample.is_rejection_target else "classifier_family",
                "rejection_label": sample.rvl_label if sample.is_rejection_target else "",
                "is_rejection_target": sample.is_rejection_target,
                "in_scope_target": in_scope_target,
                "target_scope_status": target_scope_status,
                "source_split": sample.source_split,
                "predicted_family": prediction["document_family"],
                "top_candidate": prediction["top_candidate"],
                "runner_up": prediction.get("runner_up"),
                "confidence": _coerce_float(prediction.get("confidence")),
                "score": _coerce_float(prediction.get("score")),
                "score_margin": _coerce_float(prediction.get("score_margin")),
                "decision": prediction.get("decision", ""),
                "reason": prediction.get("reason", ""),
                "recognized_characters": int(
                    _coerce_float(sample.features.get("alnum_character_count"))
                ),
                "feature_extraction_time_ms": round(sample.feature_extraction_time_ms, 4),
                "classifier_time_ms": round(classifier_time, 4),
                "rules_triggered": prediction.get("evidence", {}).get("rules_triggered", []) or [],
                "rules_by_family": prediction.get("evidence", {}).get("rules_by_family", {}) or {},
                "candidate_scores": prediction.get("candidate_scores", {}) or {},
                "rule_indicators": prediction.get("rule_indicators", {}) or {},
                "evidence_channels": sorted(available_channels(sample.features)),
                "schema_version": prediction.get("schema_version", ""),
                "taxonomy_version": prediction.get("taxonomy_version", ""),
                "feature_extraction_version": prediction.get("feature_extraction_version", ""),
                "classifier_version": prediction.get("classifier_version", ""),
                "rule_fingerprint": prediction.get("rule_fingerprint", ""),
                "feature_fingerprint": prediction.get("feature_fingerprint", ""),
                "feature_source": sample.feature_source,
                "applied_confidence_threshold": _coerce_float(
                    prediction.get("thresholds", {}).get("applied_confidence")
                ),
                "family_gate_status": (
                    gates.get(decided_family, {}).get("status", "not_gated")
                ),
                "family_gate_reason": (
                    gates.get(decided_family, {}).get("reason", "")
                ),
                # Reported beside — never instead of — the global threshold: a
                # family judged at its own bar and reported under the global
                # number is a lower bar that no table shows.
                "effective_family_threshold": _coerce_float(
                    prediction.get("thresholds", {}).get("effective_family_threshold"),
                    confidence_threshold,
                ),
                "global_confidence_threshold": float(confidence_threshold),
                "gated_families": prediction.get("evidence", {}).get("gated_families", []) or [],
                "geometry_page_ratio": _coerce_float(sample.features.get("geometry_page_ratio")),
                "geometry_pages": int(_coerce_float(sample.features.get("geometry_pages"))),
                "tab_stop_count": int(_coerce_float(sample.features.get("tab_stop_count"))),
                "wide_gap_line_ratio": _coerce_float(sample.features.get("wide_gap_line_ratio")),
                "label_value_line_ratio": _coerce_float(sample.features.get("label_value_line_ratio")),
                "top_band_header_ratio": _coerce_float(sample.features.get("top_band_header_ratio")),
                "line_pitch_regularity": _coerce_float(sample.features.get("line_pitch_regularity")),
                "body_wide_gap_line_ratio": _coerce_float(sample.features.get("body_wide_gap_line_ratio")),
                "centered_line_ratio": _coerce_float(sample.features.get("centered_line_ratio")),
                "word_line_count": int(_coerce_float(sample.features.get("word_line_count"))),
                "right_edge_regularity": _coerce_float(sample.features.get("right_edge_regularity")),
                "two_column_ratio": _coerce_float(sample.features.get("two_column_ratio")),
                "narrative_line_ratio": _coerce_float(sample.features.get("narrative_line_ratio")),
                "picture_area_ratio": _coerce_float(sample.features.get("picture_area_ratio")),
                "relevant_picture_count": int(
                    _coerce_float(sample.features.get("relevant_picture_count"))
                ),
                "table_area_ratio": _coerce_float(sample.features.get("table_area_ratio")),
                "landscape_ratio": _coerce_float(sample.features.get("landscape_ratio")),
            }
        )
    return predictions


def _rank_stored_candidate_scores(record: dict[str, Any]) -> list[tuple[str, float]]:
    scores = record.get("candidate_scores")
    if not isinstance(scores, dict):
        raise EvaluationError(
            f"Prediction {record.get('sample_id', '<unknown>')} is missing candidate_scores."
        )
    ranked = sorted(
        (
            (str(family), _coerce_float(score))
            for family, score in scores.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        raise EvaluationError(
            f"Prediction {record.get('sample_id', '<unknown>')} has no candidate_scores."
        )
    return ranked


def _build_family_risk_coverage(
    predictions: list[dict[str, Any]],
    *,
    confidence_threshold: float,
    min_score_margin: float,
    min_recognized_characters: int,
    classification_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build binary family curves by replaying only the stored-score decision policy."""
    classifier_eligible = [
        record for record in predictions if not _is_rejection_target(record)
    ]
    # v6: rejection targets are negatives, not exclusions. A curve computed
    # without them reports a risk the deployment does not have — file folders
    # and handwritten pages are exactly the documents that get accepted by a
    # family with a low bar, and leaving them out of the denominator hides it.
    curve_population = predictions
    rejection_count = len(predictions) - len(classifier_eligible)
    ranked_scores = {
        id(record): _rank_stored_candidate_scores(record)
        for record in curve_population
    }
    top_scores = [ranked[0][1] for ranked in ranked_scores.values()]
    thresholds = sorted(
        {
            0.0,
            round(float(confidence_threshold), 6),
            *(round(score, 6) for score in top_scores),
            *(max(top_scores) + 1.0 for _ in top_scores[:1]),
        }
    )
    curves = []
    csv_rows = []
    scored_families = set(SCORED_FAMILIES)
    for family in SCORED_FAMILIES:
        positives = [
            record
            for record in curve_population
            if record.get("target_family") == family and not _is_rejection_target(record)
        ]
        other_target_negatives = [
            record
            for record in curve_population
            if not _is_rejection_target(record)
            and record.get("target_family") in scored_families
            and record.get("target_family") != family
        ]
        control_negatives = [
            record
            for record in curve_population
            if not _is_rejection_target(record)
            and record.get("target_family") not in scored_families
        ]
        rejection_negatives = [
            record for record in curve_population if _is_rejection_target(record)
        ]
        negative_count = (
            len(other_target_negatives) + len(control_negatives) + len(rejection_negatives)
        )
        common = {
            "target_family": family,
            "positive_count": len(positives),
            "negative_count": negative_count,
            "negative_other_target_count": len(other_target_negatives),
            "negative_control_count": len(control_negatives),
            "negative_rejection_count": len(rejection_negatives),
            "included_sample_count": len(curve_population),
        }
        points = []
        for threshold in thresholds:
            # Declared family thresholds are offsets from the module default,
            # so they move with the swept global threshold. A curve that held
            # them fixed would describe a policy nobody runs.
            swept_family_thresholds = resolve_family_thresholds(threshold)
            accepted_records = []
            for record in curve_population:
                decision = apply_decision_policy(
                    ranked_scores[id(record)],
                    int(_coerce_float(record.get("recognized_characters"))),
                    threshold,
                    min_score_margin,
                    min_recognized_characters,
                    classification_mode,
                    swept_family_thresholds,
                )
                if (
                    decision["decision"] in ACCEPTED_DECISIONS
                    and decision["document_family"] == family
                ):
                    accepted_records.append(record)
            true_positives = sum(
                record.get("target_family") == family and not _is_rejection_target(record)
                for record in accepted_records
            )
            false_positives = len(accepted_records) - true_positives
            rejection_false_accepts = sum(
                _is_rejection_target(record) for record in accepted_records
            )
            point = {
                "confidence_threshold": threshold,
                # The bar this family was actually judged against at this point
                # of the sweep, which is not the global threshold whenever the
                # family declares its own.
                "effective_family_threshold": swept_family_thresholds.get(family, threshold),
                "min_score_margin": min_score_margin,
                "min_recognized_characters": min_recognized_characters,
                "classification_mode": classification_mode,
                "is_selected_operating_point": threshold == round(float(confidence_threshold), 6),
                "accepted_count": len(accepted_records),
                "true_positive_count": true_positives,
                "false_positive_count": false_positives,
                "rejection_false_accept_count": rejection_false_accepts,
                "coverage": round(_safe_ratio(true_positives, len(positives)), 4),
                "precision": round(_safe_ratio(true_positives, len(accepted_records)), 4),
                "risk": round(_safe_ratio(false_positives, len(accepted_records)), 4),
            }
            points.append(point)
            csv_rows.append(common | point)
        curves.append(common | {"points": points})

    payload = {
        "schema_version": 1,
        "definition": (
            "For every scorable family, positives have that canonical target_family; "
            "negatives are all other documents — the other scorable families, controls "
            "outside them, and rejection targets, which are negatives for every family "
            "because accepting one is always an error. "
            "Each point replays apply_decision_policy from stored candidate_scores only; "
            "it does not rerun OCR, feature extraction, or rule evaluation. Coverage is "
            "true-positive acceptance divided by positives; risk is false-positive "
            "acceptance divided by all accepted predictions for the target family."
        ),
        "configuration": {
            "classification_mode": classification_mode,
            "selected_confidence_threshold": confidence_threshold,
            "declared_family_confidence_thresholds": dict(FAMILY_CONFIDENCE_THRESHOLDS),
            "family_confidence_thresholds_at_operating_point": resolve_family_thresholds(
                confidence_threshold
            ),
            "min_score_margin": min_score_margin,
            "min_recognized_characters": min_recognized_characters,
        },
        "population": {
            "target_families": list(SCORED_FAMILIES),
            "included_sample_count": len(curve_population),
            "control_count": sum(
                not _is_rejection_target(record)
                and record.get("target_family") not in scored_families
                for record in curve_population
            ),
            "rejection_target_count": rejection_count,
        },
        "curves": curves,
    }
    return payload, csv_rows


#: Review order. A reviewer's attention is the scarce resource, so the list is
#: ordered by what an error *costs*, not by how confident the classifier was:
#: a wrongly accepted rejection target is a document sent somewhere it does not
#: belong, while a fallback with no rules is only a gap in coverage.
REVIEW_PRIORITIES: tuple[tuple[int, str], ...] = (
    (1, "rejection_target_accepted"),
    (2, "accepted_wrong_family"),
    (3, "high_confidence_false_positive"),
    (4, "false_negative_near_threshold"),
    (5, "fallback_without_rules"),
    (6, "other"),
)
_HIGH_CONFIDENCE_FALSE_POSITIVE = 0.85
_NEAR_THRESHOLD_BAND = 0.15


def _review_priority(record: dict[str, Any], canonical_correct: bool) -> tuple[int, str]:
    if _is_rejection_false_accept(record):
        return REVIEW_PRIORITIES[0]
    if _is_accepted_wrong_family(record):
        # An accepted wrong answer given with high confidence is the same class
        # of error but the more misleading one, so it is split out below by
        # score rather than by being demoted here.
        if _coerce_float(record.get("score")) >= _HIGH_CONFIDENCE_FALSE_POSITIVE:
            return REVIEW_PRIORITIES[2]
        return REVIEW_PRIORITIES[1]
    if not canonical_correct and not _is_accepted(record):
        threshold = _coerce_float(
            record.get("effective_family_threshold"),
            DEFAULT_CONFIDENCE_THRESHOLD,
        )
        score = _coerce_float(record.get("score"))
        if record.get("reason") == "no_rules_matched":
            return REVIEW_PRIORITIES[4]
        if score > 0.0 and (threshold - score) <= _NEAR_THRESHOLD_BAND:
            return REVIEW_PRIORITIES[3]
    return REVIEW_PRIORITIES[5]


def _build_ranked_review(
    predictions: list[dict[str, Any]],
    review_limit: int,
    target_families: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank documents for human review.

    v6 replaced the single ``correct`` column. It answered three questions at
    once — canonical correctness, in-scope correctness, and whether a rejection
    target had been declined — so a correctly routed out-of-scope document and
    a wrongly accepted file folder both read ``False`` and sorted side by side.
    Each question now has its own column, and the sort is by review priority.
    """
    scoped_target_families = frozenset(target_families or ())
    ranked = []
    for record in predictions:
        canonical_correct = _record_is_correct(record)
        scope_correct = (
            _record_is_correct(record, scoped_target_families)
            if scoped_target_families
            else canonical_correct
        )
        priority, priority_label = _review_priority(record, canonical_correct)
        fired_rule_names = [
            str(item.get("rule", ""))
            for item in record["rules_triggered"]
            if isinstance(item, dict) and _non_empty(item.get("rule"))
        ]
        ranked.append(
            {
                "sample_id": record["sample_id"],
                "rvl_label": record["rvl_label"],
                "target_family": record["target_family"],
                "source_target_family": record["source_target_family"],
                "target_kind": record["target_kind"],
                "rejection_label": record["rejection_label"],
                "predicted_family": record["predicted_family"],
                "canonical_correct": bool(canonical_correct),
                "scope_correct": bool(scope_correct),
                "is_unsafe_accept": _is_unsafe_accept(record),
                "is_rejection_false_accept": _is_rejection_false_accept(record),
                "review_priority": priority,
                "review_priority_label": priority_label,
                "decision": record["decision"],
                "reason": record["reason"],
                "confidence": round(_coerce_float(record["confidence"]), 4),
                "score": round(_coerce_float(record.get("score")), 4),
                "effective_family_threshold": round(
                    _coerce_float(record.get("effective_family_threshold")), 4
                ),
                "score_margin": round(_coerce_float(record["score_margin"]), 4),
                "top_candidate": record.get("top_candidate"),
                "runner_up": record.get("runner_up"),
                "gate_status": record.get("family_gate_status", ""),
                "gate_reason": record.get("family_gate_reason", ""),
                "fired_rules": fired_rule_names,
            }
        )

    ranked.sort(
        key=lambda row: (
            row["review_priority"],
            -row["confidence"] if row["is_unsafe_accept"] else row["score_margin"],
            row["confidence"],
            row["sample_id"],
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["review_rank"] = index
    limit = review_limit if review_limit > 0 else len(ranked)
    return ranked[:limit]


def _build_confusion_pairs(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in predictions:
        if _is_rejection_target(record) or record["target_family"] == record["predicted_family"]:
            continue
        key = (str(record["target_family"]), record["predicted_family"])
        grouped.setdefault(key, []).append(record)

    pair_rows = []
    csv_rows = []
    for (truth, predicted), rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: (_coerce_float(item["score_margin"]), _coerce_float(item["confidence"]), item["sample_id"]),
        )
        reasons = Counter(str(item.get("reason", "")) for item in rows if _non_empty(item.get("reason")))
        source_target_families = Counter(
            str(item.get("source_target_family", ""))
            for item in rows
            if _non_empty(item.get("source_target_family"))
        )
        fired_rules = Counter()
        for item in rows:
            for rule in item.get("rules_triggered", []):
                if isinstance(rule, dict):
                    name = _non_empty(rule.get("rule"))
                    if name:
                        fired_rules[name] += 1
        representatives = []
        for candidate in rows_sorted[:5]:
            representatives.append(
                {
                    "sample_id": candidate["sample_id"],
                    "decision": candidate["decision"],
                    "reason": candidate["reason"],
                    "confidence": round(_coerce_float(candidate["confidence"]), 4),
                    "score_margin": round(_coerce_float(candidate["score_margin"]), 4),
                    "fired_rules": [
                        str(entry.get("rule", ""))
                        for entry in candidate.get("rules_triggered", [])
                        if isinstance(entry, dict) and _non_empty(entry.get("rule"))
                    ],
                }
            )
        pair_rows.append(
            {
                "target_family": truth,
                "source_target_families": dict(sorted(source_target_families.items())),
                "predicted_family": predicted,
                "error_count": len(rows),
                "reasons": dict(sorted(reasons.items())),
                "fired_rules": [name for name, _ in fired_rules.most_common(12)],
                "representative_samples": representatives,
            }
        )
        csv_rows.append(
            {
                "target_family": truth,
                "source_target_families": "; ".join(
                    f"{family}:{count}" for family, count in sorted(source_target_families.items())
                ),
                "predicted_family": predicted,
                "error_count": len(rows),
                "top_reasons": "; ".join(f"{name}:{count}" for name, count in reasons.most_common(5)),
                "top_fired_rules": "; ".join(name for name, _ in fired_rules.most_common(8)),
                "representative_sample_ids": ",".join(entry["sample_id"] for entry in representatives),
            }
        )

    pair_rows.sort(key=lambda row: (-row["error_count"], row["target_family"], row["predicted_family"]))
    csv_rows.sort(key=lambda row: (-row["error_count"], row["target_family"], row["predicted_family"]))
    return pair_rows, csv_rows


def _validate_manifest_columns(fieldnames: list[str], manifest_path: Path) -> str | None:
    missing = [name for name in REQUIRED_MANIFEST_COLUMNS if name not in fieldnames]
    if missing:
        raise ManifestValidationError(
            f"Manifest {manifest_path} is missing required column(s): {', '.join(missing)}"
        )

    unsupported_error_columns = [
        name for name in fieldnames if "error" in name.lower() and name not in SUPPORTED_ERROR_COLUMNS
    ]
    if unsupported_error_columns:
        raise ManifestValidationError(
            "Unsupported manifest error column(s): "
            f"{', '.join(sorted(unsupported_error_columns))}. "
            "Supported error columns are: "
            f"{', '.join(sorted(SUPPORTED_ERROR_COLUMNS))}."
        )

    return next((name for name in SPLIT_COLUMNS if name in fieldnames), None)


def _load_manifest_records(
    manifest_path: Path,
) -> tuple[list[dict[str, str]], list[ManifestInputErrorRecord], str | None]:
    classifier_rows: list[dict[str, str]] = []
    input_errors: list[ManifestInputErrorRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestValidationError(f"Manifest {manifest_path} has no header row.")
        split_column = _validate_manifest_columns(reader.fieldnames, manifest_path)

        for row_number, raw_row in enumerate(reader, start=2):
            row = {key: (value or "") for key, value in raw_row.items()}
            sample_id = _non_empty(row.get("sample_id", "")) or f"row_{row_number}"
            rvl_label = _non_empty(row.get("rvl_label", ""))
            target_family = _non_empty(row.get("target_family", ""))
            split_value = _first_non_empty(row, SPLIT_COLUMNS)
            cached_error = _first_non_empty(row, CACHED_ERROR_COLUMNS)
            source_error = _first_non_empty(row, SOURCE_ERROR_COLUMNS)

            if rvl_label:
                try:
                    resolve_evaluation_target(rvl_label)
                except ValueError as error:
                    raise ManifestValidationError(
                        f"Manifest {manifest_path} row {row_number} "
                        f"(sample_id={sample_id}) has {error}."
                    ) from error

            # A ``target_family`` outside the taxonomy is a ground-truth error
            # that presents as a classifier error: the family can never be
            # predicted, so its recall reads as zero for a reason that has
            # nothing to do with the rules.
            if target_family and not is_known_family(target_family):
                raise VersionCompatibilityError(
                    f"Manifest {manifest_path} row {row_number} (sample_id={sample_id}) "
                    f"declares target_family={target_family!r}, which is outside taxonomy "
                    f"{TAXONOMY_VERSION}. Known families: {', '.join(DOCUMENT_FAMILIES)}."
                )

            if cached_error:
                input_errors.append(
                    ManifestInputErrorRecord(
                        row_number=row_number,
                        sample_id=sample_id,
                        rvl_label=rvl_label,
                        target_family=target_family,
                        source_split=split_value,
                        error_source="cached_manifest_input_error",
                        error_message=cached_error,
                    )
                )
                continue
            if source_error:
                input_errors.append(
                    ManifestInputErrorRecord(
                        row_number=row_number,
                        sample_id=sample_id,
                        rvl_label=rvl_label,
                        target_family=target_family,
                        source_split=split_value,
                        error_source="source_manifest_input_error",
                        error_message=source_error,
                    )
                )
                continue

            missing_values = [
                column for column in REQUIRED_MANIFEST_COLUMNS if not _non_empty(row.get(column, ""))
            ]
            if missing_values:
                raise ManifestValidationError(
                    f"Manifest {manifest_path} row {row_number} (sample_id={sample_id}) "
                    f"is missing required value(s): {', '.join(missing_values)}"
                )
            row["_manifest_row_number"] = str(row_number)
            classifier_rows.append(row)
    return classifier_rows, input_errors, split_column


def _resolve_optional_path(
    manifest_directory: Path, value: str, sample_id: str, column: str
) -> Path | None:
    if not value:
        return None
    path = (manifest_directory / value).resolve()
    if not path.is_file():
        raise ManifestValidationError(
            f"{column} does not exist for sample_id={sample_id}: {path}"
        )
    return path


def _load_cached_samples(
    manifest_path: Path,
    rows: list[dict[str, str]],
    ledger: FeatureVersionLedger,
) -> list[CachedSample]:
    samples = []
    manifest_directory = manifest_path.parent
    for row_number, row in enumerate(rows, start=2):
        manifest_row_number = int(_non_empty(row.get("_manifest_row_number")) or row_number)
        sample_id = _non_empty(row.get("sample_id")) or f"row_{row_number}"
        document_path_value = _non_empty(row.get("document_path"))
        page_sizes_path_value = _non_empty(row.get("page_sizes_path"))
        page_words_path_value = _non_empty(row.get(PAGE_WORDS_PATH_COLUMN))
        document_path = (manifest_directory / document_path_value).resolve()
        if not document_path.is_file():
            raise ManifestValidationError(
                f"Cached document JSON does not exist for sample_id={sample_id}: {document_path}"
            )
        page_sizes = None
        if page_sizes_path_value:
            page_sizes_path = (manifest_directory / page_sizes_path_value).resolve()
            if not page_sizes_path.is_file():
                raise ManifestValidationError(
                    f"page_sizes_path does not exist for sample_id={sample_id}: {page_sizes_path}"
                )
            page_sizes = _read_json(page_sizes_path)

        page_words = None
        if page_words_path_value:
            page_words_path = (manifest_directory / page_words_path_value).resolve()
            if not page_words_path.is_file():
                raise ManifestValidationError(
                    f"{PAGE_WORDS_PATH_COLUMN} does not exist for sample_id={sample_id}: "
                    f"{page_words_path}"
                )
            page_words = _read_json(page_words_path)

        stored_features_path = _resolve_optional_path(
            manifest_directory,
            _non_empty(row.get(FEATURES_PATH_COLUMN)),
            sample_id,
            FEATURES_PATH_COLUMN,
        )
        provenance_path = _resolve_optional_path(
            manifest_directory,
            _non_empty(row.get(PROVENANCE_PATH_COLUMN)),
            sample_id,
            PROVENANCE_PATH_COLUMN,
        )

        feature_started = perf_counter()
        if stored_features_path is not None:
            # Path 1: score exactly what the workflow stored.
            features = _read_json(stored_features_path)
            feature_source = "workflow_cache"
        else:
            # Path 2: re-extract from the same four inputs the workflow feeds
            # to extract_document_classification_features. ``provenance`` is
            # part of that input set: the features are entirely determined by
            # the extraction chain, so a record without its identity is not
            # reproducible even when every number in it is right.
            document = _read_json(document_path)
            provenance = None
            if provenance_path is not None:
                provenance = _read_json(provenance_path)
            elif isinstance(document, dict) and isinstance(document.get("provenance"), dict):
                provenance = document["provenance"]
            try:
                features = extract_classification_features(
                    document,
                    page_sizes=page_sizes,
                    page_words=page_words,
                    provenance=provenance,
                )
            except Exception as error:
                raise ManifestValidationError(
                    f"Failed feature extraction for sample_id={sample_id}: {error}"
                ) from error
            feature_source = "reextracted"
        feature_time = (perf_counter() - feature_started) * 1000.0

        try:
            descriptor = validate_feature_record(
                features, source=f"sample_id={sample_id}"
            )
        except FeatureContractError as error:
            raise VersionCompatibilityError(str(error)) from error
        ledger.record(sample_id, descriptor, feature_source)
        rvl_label = _non_empty(row.get("rvl_label"))
        try:
            evaluation_target = resolve_evaluation_target(rvl_label)
        except ValueError as error:
            raise ManifestValidationError(
                f"Manifest {manifest_path} row {manifest_row_number} "
                f"(sample_id={sample_id}) has {error}."
            ) from error
        samples.append(
            CachedSample(
                row_number=manifest_row_number,
                sample_id=sample_id,
                rvl_label=rvl_label,
                target_family=evaluation_target.family,
                source_target_family=_non_empty(row.get("target_family")),
                is_rejection_target=evaluation_target.is_rejection_target,
                source_split=_first_non_empty(row, SPLIT_COLUMNS),
                features=features,
                feature_extraction_time_ms=feature_time,
                feature_source=feature_source,
            )
        )
    return samples


def _validation_only_guard(
    samples: list[CachedSample],
    split_column: str | None,
    input_errors: list[ManifestInputErrorRecord],
) -> None:
    if not split_column:
        raise CalibrationValidationError(
            "Calibration requires a source split column in the cache manifest. "
            f"Accepted split column names: {', '.join(SPLIT_COLUMNS)}."
        )
    if not samples:
        raise CalibrationValidationError("Calibration requires at least one non-error classifier sample.")

    non_validation = []
    missing_split = []
    for sample in samples:
        split = _non_empty(sample.source_split).lower()
        if not split:
            missing_split.append(sample.sample_id)
            continue
        if split != "validation":
            non_validation.append((sample.sample_id, sample.source_split))
    for error_record in input_errors:
        if error_record.rvl_label:
            try:
                if resolve_evaluation_target(error_record.rvl_label).is_rejection_target:
                    continue
            except ValueError:
                pass
        split = _non_empty(error_record.source_split).lower()
        if not split:
            missing_split.append(error_record.sample_id)
            continue
        if split != "validation":
            non_validation.append((error_record.sample_id, error_record.source_split))
    if missing_split or non_validation:
        preview = []
        for sample_id in missing_split[:5]:
            preview.append(f"{sample_id}=<missing>")
        for sample_id, split_value in non_validation[:5]:
            preview.append(f"{sample_id}={split_value}")
        raise CalibrationValidationError(
            "Calibration is validation-only. Every classifier sample must have split=validation "
            f"in column '{split_column}'. Violations: {', '.join(preview)}."
        )


def _run_calibration(
    samples: list[CachedSample],
    classification_mode: str,
    min_recognized_characters: int,
    confidence_grid: list[float],
    margin_grid: list[float],
    min_coverage: float,
    min_accepted_accuracy: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if classification_mode != "evaluate":
        raise CalibrationValidationError("Calibration can only be used with --classification-mode evaluate.")

    leaderboard: list[dict[str, Any]] = []
    for confidence_threshold in confidence_grid:
        for min_score_margin in margin_grid:
            predictions = _predict_samples(
                samples=samples,
                confidence_threshold=confidence_threshold,
                min_score_margin=min_score_margin,
                min_recognized_characters=min_recognized_characters,
                mode=classification_mode,
            )
            macro_f1, coverage, accuracy_on_accepted = _calibration_metrics(predictions)
            classified_response_count = sum(
                prediction["decision"] in ACCEPTED_DECISIONS for prediction in predictions
            )
            classified_correct_count = sum(
                prediction["target_family"] == prediction["predicted_family"]
                for prediction in predictions
                if prediction["decision"] in ACCEPTED_DECISIONS
            )
            meets_coverage = coverage >= min_coverage
            meets_accepted_accuracy = accuracy_on_accepted >= min_accepted_accuracy
            leaderboard.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "min_score_margin": min_score_margin,
                    "macro_f1": round(macro_f1, 6),
                    "macro_f1_classified_excluding_other": round(macro_f1, 6),
                    "classified_response_count": classified_response_count,
                    "classified_response_denominator": len(predictions),
                    "coverage": round(coverage, 6),
                    "accuracy_on_accepted": round(accuracy_on_accepted, 6),
                    "classified_accuracy": round(accuracy_on_accepted, 6),
                    "classified_correct_count": classified_correct_count,
                    "_selection_macro_f1": macro_f1,
                    "_selection_coverage": coverage,
                    "_selection_accuracy_on_accepted": accuracy_on_accepted,
                    "meets_min_coverage": meets_coverage,
                    "meets_min_accepted_accuracy": meets_accepted_accuracy,
                    "meets_constraints": bool(meets_coverage and meets_accepted_accuracy),
                }
            )

    constrained = sorted(
        [row for row in leaderboard if row["meets_constraints"]],
        key=lambda row: (
            row["_selection_macro_f1"],
            row["_selection_coverage"],
            row["_selection_accuracy_on_accepted"],
            -row["confidence_threshold"],
            -row["min_score_margin"],
        ),
        reverse=True,
    )
    if not constrained:
        best = max(
            leaderboard,
            key=lambda row: (
                row["_selection_macro_f1"],
                row["_selection_coverage"],
                row["_selection_accuracy_on_accepted"],
                -row["confidence_threshold"],
                -row["min_score_margin"],
            ),
        )
        raise CalibrationValidationError(
            "No calibration configuration met constraints "
            f"(min_coverage={min_coverage}, min_accepted_accuracy={min_accepted_accuracy}). "
            f"Best observed config: confidence_threshold={best['confidence_threshold']}, "
            f"min_score_margin={best['min_score_margin']}, "
            f"macro_f1_classified_excluding_other={best['macro_f1_classified_excluding_other']}, "
            f"coverage={best['coverage']}, accuracy_on_accepted={best['accuracy_on_accepted']}."
        )

    selected = constrained[0]
    ranked = sorted(
        leaderboard,
        key=lambda row: (
            0 if row["meets_constraints"] else 1,
            -row["_selection_macro_f1"],
            -row["_selection_coverage"],
            -row["_selection_accuracy_on_accepted"],
            row["confidence_threshold"],
            row["min_score_margin"],
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        for key in (
            "_selection_macro_f1",
            "_selection_coverage",
            "_selection_accuracy_on_accepted",
        ):
            row.pop(key)
    return ranked, selected


def _baseline_extract_metrics(report: dict[str, Any], baseline_path: Path) -> dict[str, float]:
    if not isinstance(report, dict):
        raise BaselineValidationError(f"Baseline report {baseline_path} must be a JSON object.")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise BaselineValidationError(f"Baseline report {baseline_path} is missing metrics object.")

    def require_number(container: dict[str, Any], key: str, context: str) -> float:
        if key not in container:
            raise BaselineValidationError(
                f"Baseline report {baseline_path} is missing {context}.{key}."
            )
        value = container[key]
        if not isinstance(value, (int, float)):
            raise BaselineValidationError(
                f"Baseline report {baseline_path} expected numeric {context}.{key}, got {type(value).__name__}."
            )
        return float(value)

    macro_f1 = require_number(metrics, "macro_f1", "metrics")
    coverage = require_number(metrics, "coverage", "metrics")
    timing_container = metrics.get("timing_ms")
    timing_context = "metrics.timing_ms"
    if not isinstance(timing_container, dict):
        timing_container = metrics.get("timing")
        timing_context = "metrics.timing"
    if not isinstance(timing_container, dict):
        raise BaselineValidationError(
            f"Baseline report {baseline_path} must contain metrics.timing_ms or metrics.timing object."
        )
    classifier_p95 = require_number(timing_container, "classifier_p95", timing_context)

    input_errors = metrics.get("input_errors", {})
    if input_errors is None:
        input_errors = {}
    if not isinstance(input_errors, dict):
        raise BaselineValidationError(
            f"Baseline report {baseline_path} metrics.input_errors must be an object when present."
        )
    cached_input_errors_value = input_errors.get("cached_manifest_input_errors", 0)
    if not isinstance(cached_input_errors_value, (int, float)):
        raise BaselineValidationError(
            f"Baseline report {baseline_path} expected numeric metrics.input_errors.cached_manifest_input_errors."
        )
    cached_input_errors = float(cached_input_errors_value)
    return {
        "macro_f1": macro_f1,
        "coverage": coverage,
        "classifier_p95": classifier_p95,
        "cached_manifest_input_errors": cached_input_errors,
    }


def _baseline_version_drift(
    current_versions: dict[str, Any],
    baseline_report: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    """Compare identities with the baseline, and refuse unknown rule ids.

    Drift itself is reported, not refused: comparing a new rule set against an
    older baseline is the whole point of a regression gate. What *is* refused is
    a baseline naming rules this build does not declare — those cannot be read
    as "never fired" without inventing a result.
    """
    baseline_versions = baseline_report.get("versions")
    if not isinstance(baseline_versions, dict):
        return {
            "baseline_versions_present": False,
            "note": (
                "Baseline report predates version recording; it cannot be checked for "
                "schema, taxonomy, feature or rule-set compatibility."
            ),
        }

    baseline_rule_ids = baseline_versions.get("rule_ids")
    if baseline_rule_ids is not None:
        try:
            validate_rule_ids(baseline_rule_ids, source=f"baseline {baseline_path} rule_ids")
        except FeatureContractError as error:
            raise VersionCompatibilityError(str(error)) from error

    tracked = (
        "schema_version",
        "taxonomy_version",
        "feature_extraction_version",
        "classifier_version",
        "rule_fingerprint",
        "feature_fingerprint",
    )
    differences = {
        key: {"baseline": baseline_versions.get(key), "current": current_versions.get(key)}
        for key in tracked
        if baseline_versions.get(key) != current_versions.get(key)
    }
    return {
        "baseline_versions_present": True,
        "identical": not differences,
        "differences": differences,
        "interpretation": (
            "A differing rule_fingerprint or feature_extraction_version means the two "
            "reports describe different systems; the regression gate still runs, but the "
            "comparison is between systems, not between runs of one system."
        ),
    }


def _build_baseline_comparison(
    current_report: dict[str, Any],
    baseline_report: dict[str, Any],
    baseline_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    current_metrics = current_report["metrics"]
    current = {
        "macro_f1": float(current_metrics["macro_f1"]),
        "coverage": float(current_metrics["coverage"]),
        "classifier_p95": float(current_metrics["timing_ms"]["classifier_p95"]),
        "cached_manifest_input_errors": float(
            current_metrics.get("input_errors", {}).get("cached_manifest_input_errors", 0)
        ),
    }
    baseline = _baseline_extract_metrics(baseline_report, baseline_path)

    gates = []
    failures = []

    macro_f1_regression = baseline["macro_f1"] - current["macro_f1"]
    gates.append(
        {
            "name": "macro_f1",
            "baseline": round(baseline["macro_f1"], 6),
            "current": round(current["macro_f1"], 6),
            "regression": round(macro_f1_regression, 6),
            "max_allowed_regression": args.max_macro_f1_regression,
            "passed": macro_f1_regression <= args.max_macro_f1_regression,
        }
    )
    if macro_f1_regression > args.max_macro_f1_regression:
        failures.append(
            f"macro_f1 regression {macro_f1_regression:.4f} exceeds allowed {args.max_macro_f1_regression:.4f}"
        )

    coverage_regression = baseline["coverage"] - current["coverage"]
    gates.append(
        {
            "name": "coverage",
            "baseline": round(baseline["coverage"], 6),
            "current": round(current["coverage"], 6),
            "regression": round(coverage_regression, 6),
            "max_allowed_regression": args.max_coverage_regression,
            "passed": coverage_regression <= args.max_coverage_regression,
        }
    )
    if coverage_regression > args.max_coverage_regression:
        failures.append(
            f"coverage regression {coverage_regression:.4f} exceeds allowed {args.max_coverage_regression:.4f}"
        )

    p95_regression = current["classifier_p95"] - baseline["classifier_p95"]
    gates.append(
        {
            "name": "classifier_p95_ms",
            "baseline": round(baseline["classifier_p95"], 6),
            "current": round(current["classifier_p95"], 6),
            "regression": round(p95_regression, 6),
            "max_allowed_regression": args.max_classifier_p95_regression_ms,
            "passed": p95_regression <= args.max_classifier_p95_regression_ms,
        }
    )
    if p95_regression > args.max_classifier_p95_regression_ms:
        failures.append(
            "classifier_p95 regression "
            f"{p95_regression:.4f}ms exceeds allowed {args.max_classifier_p95_regression_ms:.4f}ms"
        )

    cached_errors_regression = current["cached_manifest_input_errors"] - baseline["cached_manifest_input_errors"]
    gates.append(
        {
            "name": "cached_manifest_input_errors",
            "baseline": int(round(baseline["cached_manifest_input_errors"])),
            "current": int(round(current["cached_manifest_input_errors"])),
            "regression": int(round(cached_errors_regression)),
            "max_allowed_regression": args.max_cached_input_error_increase,
            "passed": cached_errors_regression <= args.max_cached_input_error_increase,
        }
    )
    if cached_errors_regression > args.max_cached_input_error_increase:
        failures.append(
            "cached_manifest_input_errors increase "
            f"{cached_errors_regression:.0f} exceeds allowed {args.max_cached_input_error_increase}"
        )

    return {
        "baseline_report": str(baseline_path),
        "overall_passed": len(failures) == 0,
        "version_drift": _baseline_version_drift(
            current_report.get("versions", {}), baseline_report, baseline_path
        ),
        "gates": gates,
    }, failures


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.description = (
        "Evaluate Hydra rule classification from cached aggregate-document JSONs. "
        "Calibration mode is validation-only and requires split=validation."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("output/rules-benchmark"))
    parser.add_argument("--classification-mode", choices=("evaluate", "observe"), default="evaluate")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--min-score-margin", type=float, default=DEFAULT_MIN_SCORE_MARGIN)
    parser.add_argument(
        "--min-recognized-characters",
        type=int,
        default=DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    )
    parser.add_argument(
        "--calibrate-validation",
        action="store_true",
        help="Sweep confidence/margin grids on validation-only input and select the best config.",
    )
    parser.add_argument("--confidence-grid", default=DEFAULT_CONFIDENCE_GRID)
    parser.add_argument("--margin-grid", default=DEFAULT_MARGIN_GRID)
    parser.add_argument("--calibration-min-coverage", type=float, default=0.0)
    parser.add_argument("--calibration-min-accepted-accuracy", type=float, default=0.0)
    parser.add_argument(
        "--target-families",
        default="",
        help=(
            "Comma-separated in-scope evaluation families. Supported: "
            f"{', '.join(SCOPABLE_TARGET_FAMILIES)}."
        ),
    )
    parser.add_argument("--review-limit", type=int, default=250)

    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--max-macro-f1-regression", type=float, default=0.0)
    parser.add_argument("--max-coverage-regression", type=float, default=0.0)
    parser.add_argument("--max-classifier-p95-regression-ms", type=float, default=0.0)
    parser.add_argument("--max-cached-input-error-increase", type=int, default=0)
    return parser


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise ManifestValidationError(f"Manifest does not exist: {manifest}")
    scoped_target_families = frozenset(_parse_target_families(args.target_families))

    # The shipped taxonomy config and the taxonomy module must agree before a
    # single row is read: a divergence means ground truth is ambiguous, and
    # every metric downstream would silently inherit the ambiguity.
    try:
        taxonomy_check = verify_config_file()
    except TaxonomyConfigError as error:
        raise VersionCompatibilityError(str(error)) from error

    classifier_rows, input_errors, split_column = _load_manifest_records(manifest)
    feature_versions = FeatureVersionLedger()
    samples = _load_cached_samples(manifest, classifier_rows, feature_versions)
    classifier_eligible_samples = [
        sample for sample in samples if not sample.is_rejection_target
    ]

    calibration_artifacts = {}
    selected_config = {
        "confidence_threshold": float(args.confidence_threshold),
        "min_score_margin": float(args.min_score_margin),
        "min_recognized_characters": int(args.min_recognized_characters),
    }

    if args.calibrate_validation:
        _validation_only_guard(classifier_eligible_samples, split_column, input_errors)
        confidence_grid = _parse_grid(args.confidence_grid, "confidence")
        margin_grid = _parse_grid(args.margin_grid, "margin")
        if args.calibration_min_coverage < 0.0 or args.calibration_min_coverage > 1.0:
            raise CalibrationValidationError("--calibration-min-coverage must be within [0, 1].")
        if args.calibration_min_accepted_accuracy < 0.0 or args.calibration_min_accepted_accuracy > 1.0:
            raise CalibrationValidationError("--calibration-min-accepted-accuracy must be within [0, 1].")

        leaderboard, selected = _run_calibration(
            samples=classifier_eligible_samples,
            classification_mode=args.classification_mode,
            min_recognized_characters=int(args.min_recognized_characters),
            confidence_grid=confidence_grid,
            margin_grid=margin_grid,
            min_coverage=float(args.calibration_min_coverage),
            min_accepted_accuracy=float(args.calibration_min_accepted_accuracy),
        )
        selected_config["confidence_threshold"] = float(selected["confidence_threshold"])
        selected_config["min_score_margin"] = float(selected["min_score_margin"])

        calibration_artifacts = {
            "leaderboard_rows": leaderboard,
            "selected": selected,
            "classifier_eligible_rows": len(classifier_eligible_samples),
            "rejection_target_rows": len(samples) - len(classifier_eligible_samples),
            "constraints": {
                "min_coverage": float(args.calibration_min_coverage),
                "min_accepted_accuracy": float(args.calibration_min_accepted_accuracy),
            },
            "objective": {
                "field": "macro_f1_classified_excluding_other",
                "definition": (
                    "Macro-F1 over classified or observed responses only, excluding the "
                    "'other' label from the macro average."
                ),
                "zero_classified_response_value": 0.0,
            },
            "grids": {
                "confidence_threshold": confidence_grid,
                "min_score_margin": margin_grid,
            },
        }

    predictions = _predict_samples(
        samples=samples,
        confidence_threshold=selected_config["confidence_threshold"],
        min_score_margin=selected_config["min_score_margin"],
        min_recognized_characters=selected_config["min_recognized_characters"],
        mode=args.classification_mode,
        target_families=scoped_target_families,
    )
    metrics = _metrics(
        predictions,
        scoped_target_families,
        confidence_threshold=selected_config["confidence_threshold"],
    )
    metrics["input_errors"] = {
        "cached_manifest_input_errors": sum(
            1 for record in input_errors if record.error_source == "cached_manifest_input_error"
        ),
        "source_manifest_input_errors": sum(
            1 for record in input_errors if record.error_source == "source_manifest_input_error"
        ),
        "total_manifest_input_errors": len(input_errors),
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_directory = args.output_directory.resolve()
    predictions_path = output_directory / "predictions.csv"
    report_path = output_directory / "report.json"
    review_json_path = output_directory / "diagnostics_review.json"
    review_csv_path = output_directory / "diagnostics_review.csv"
    confusion_json_path = output_directory / "diagnostics_confusion_pairs.json"
    confusion_csv_path = output_directory / "diagnostics_confusion_pairs.csv"
    input_errors_json_path = output_directory / "manifest_input_errors.json"
    input_errors_csv_path = output_directory / "manifest_input_errors.csv"
    family_risk_coverage_json_path = output_directory / "family_risk_coverage_curve.json"
    family_risk_coverage_csv_path = output_directory / "family_risk_coverage_curve.csv"
    routing_csv_path = output_directory / "routing_by_family.csv"
    per_label_csv_path = output_directory / "metrics_by_rvl_label.csv"

    prediction_rows_for_csv = [_serialize_prediction_for_csv(item) for item in predictions]
    _write_csv(predictions_path, prediction_rows_for_csv, PREDICTION_CSV_HEADERS)
    family_risk_coverage, family_risk_coverage_rows = _build_family_risk_coverage(
        predictions,
        confidence_threshold=selected_config["confidence_threshold"],
        min_score_margin=selected_config["min_score_margin"],
        min_recognized_characters=selected_config["min_recognized_characters"],
        classification_mode=args.classification_mode,
    )
    _write_json(family_risk_coverage_json_path, family_risk_coverage)
    _write_csv(
        family_risk_coverage_csv_path,
        family_risk_coverage_rows,
        FAMILY_RISK_COVERAGE_CSV_HEADERS,
    )

    _routing_payload, routing_rows = _routing_metrics(
        predictions, selected_config["confidence_threshold"]
    )
    _per_label_payload, per_label_rows = _per_rvl_label_metrics(predictions)
    _write_csv(routing_csv_path, routing_rows, ROUTING_CSV_HEADERS)
    _write_csv(per_label_csv_path, per_label_rows, PER_LABEL_CSV_HEADERS)

    ranked_review = _build_ranked_review(
        predictions,
        int(args.review_limit),
        target_families=scoped_target_families,
    )
    confusion_rows, confusion_rows_csv = _build_confusion_pairs(predictions)
    _write_json(review_json_path, ranked_review)
    _write_csv(review_csv_path, ranked_review, REVIEW_CSV_HEADERS)
    _write_json(confusion_json_path, confusion_rows)
    _write_csv(confusion_csv_path, confusion_rows_csv, CONFUSION_CSV_HEADERS)

    manifest_error_payload = [
        {
            "row_number": row.row_number,
            "sample_id": row.sample_id,
            "rvl_label": row.rvl_label,
            "target_family": row.target_family,
            "source_split": row.source_split,
            "error_source": row.error_source,
            "error_message": row.error_message,
        }
        for row in input_errors
    ]
    _write_json(input_errors_json_path, manifest_error_payload)
    _write_csv(input_errors_csv_path, manifest_error_payload, INPUT_ERROR_CSV_HEADERS)

    report_configuration = {
        "classification_mode": args.classification_mode,
        "confidence_threshold": selected_config["confidence_threshold"],
        "family_confidence_thresholds": resolve_family_thresholds(
            selected_config["confidence_threshold"]
        ),
        "declared_family_confidence_thresholds": dict(FAMILY_CONFIDENCE_THRESHOLDS),
        "family_threshold_provenance": dict(FAMILY_THRESHOLD_PROVENANCE),
        "family_threshold_holds": dict(FAMILY_THRESHOLD_HOLDS),
        "min_score_margin": selected_config["min_score_margin"],
        "min_recognized_characters": selected_config["min_recognized_characters"],
        "calibrate_validation": bool(args.calibrate_validation),
        "evaluation_target_source": "tasks.document.rvl_cdip_eval.CLASS_TO_FAMILY",
    }

    # Everything a later reader needs to know whether two reports may be
    # compared at all. Recorded whether or not a baseline is supplied.
    observed = feature_versions.descriptor or {}
    versions_block = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "feature_fingerprint": observed.get("feature_fingerprint", ""),
        "observed_feature_versions": dict(observed),
        "feature_sources": dict(sorted(feature_versions.sources.items())),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "rule_count": len(RULE_IDS),
        "rule_ids": list(RULE_IDS),
        "classifier": classifier_versions(),
        "taxonomy": taxonomy_descriptor(),
        "taxonomy_config_check": taxonomy_check,
        "refusals": (
            "The evaluator refuses: feature records under an unsupported schema or "
            "feature-extraction version; a target_family outside the taxonomy; rule ids "
            "absent from RULE_IDS; and any mixture of feature versions or fingerprints "
            "within one evaluation."
        ),
    }
    if scoped_target_families:
        report_configuration["target_families"] = sorted(scoped_target_families)

    report = {
        "configuration": report_configuration,
        "versions": versions_block,
        "metrics": metrics,
        "manifest": {
            "path": str(manifest),
            "split_column_detected": split_column,
            "rows_total": len(classifier_rows) + len(input_errors),
            "classifier_rows": len(classifier_rows),
            "classifier_eligible_rows": len(classifier_eligible_samples),
            "rejection_target_rows": len(samples) - len(classifier_eligible_samples),
            "input_error_rows": len(input_errors),
            "input_error_sources": sorted(SUPPORTED_ERROR_COLUMNS),
        },
        "diagnostics": {
            "review_limit": int(args.review_limit),
            "review_json": str(review_json_path),
            "review_csv": str(review_csv_path),
            "confusion_pairs_json": str(confusion_json_path),
            "confusion_pairs_csv": str(confusion_csv_path),
            "manifest_input_errors_json": str(input_errors_json_path),
            "manifest_input_errors_csv": str(input_errors_csv_path),
            "family_risk_coverage_json": str(family_risk_coverage_json_path),
            "family_risk_coverage_csv": str(family_risk_coverage_csv_path),
            "routing_by_family_csv": str(routing_csv_path),
            "metrics_by_rvl_label_csv": str(per_label_csv_path),
            "review_priority_order": [label for _rank, label in REVIEW_PRIORITIES],
        },
    }
    if scoped_target_families:
        report["configuration"]["target_families"] = sorted(scoped_target_families)

    artifact_paths = {
        "predictions": str(predictions_path),
        "report": str(report_path),
        "diagnostics_review_json": str(review_json_path),
        "diagnostics_review_csv": str(review_csv_path),
        "diagnostics_confusion_pairs_json": str(confusion_json_path),
        "diagnostics_confusion_pairs_csv": str(confusion_csv_path),
        "manifest_input_errors_json": str(input_errors_json_path),
        "manifest_input_errors_csv": str(input_errors_csv_path),
        "family_risk_coverage_json": str(family_risk_coverage_json_path),
        "family_risk_coverage_csv": str(family_risk_coverage_csv_path),
        "routing_by_family_csv": str(routing_csv_path),
        "metrics_by_rvl_label_csv": str(per_label_csv_path),
    }

    if calibration_artifacts:
        leaderboard_json_path = output_directory / "calibration_leaderboard.json"
        leaderboard_csv_path = output_directory / "calibration_leaderboard.csv"
        selected_json_path = output_directory / "calibration_selected_config.json"
        _write_json(leaderboard_json_path, calibration_artifacts["leaderboard_rows"])
        _write_csv(
            leaderboard_csv_path,
            calibration_artifacts["leaderboard_rows"],
            CALIBRATION_CSV_HEADERS,
        )
        selected_payload = {
            "selected_configuration": {
                "confidence_threshold": selected_config["confidence_threshold"],
                "min_score_margin": selected_config["min_score_margin"],
                "min_recognized_characters": selected_config["min_recognized_characters"],
            },
            "selected_row": calibration_artifacts["selected"],
            "classifier_eligible_rows": calibration_artifacts["classifier_eligible_rows"],
            "rejection_target_rows": calibration_artifacts["rejection_target_rows"],
            "constraints": calibration_artifacts["constraints"],
            "objective": calibration_artifacts["objective"],
            "grids": calibration_artifacts["grids"],
        }
        _write_json(selected_json_path, selected_payload)
        report["calibration"] = selected_payload | {
            "leaderboard_json": str(leaderboard_json_path),
            "leaderboard_csv": str(leaderboard_csv_path),
        }
        artifact_paths.update(
            {
                "calibration_leaderboard_json": str(leaderboard_json_path),
                "calibration_leaderboard_csv": str(leaderboard_csv_path),
                "calibration_selected_config_json": str(selected_json_path),
            }
        )

    baseline_failures: list[str] = []
    if args.baseline_report:
        baseline_path = args.baseline_report.resolve()
        if not baseline_path.is_file():
            raise BaselineValidationError(f"Baseline report does not exist: {baseline_path}")
        baseline_payload = _read_json(baseline_path)
        comparison, baseline_failures = _build_baseline_comparison(report, baseline_payload, baseline_path, args)
        baseline_comparison_path = output_directory / "baseline_comparison.json"
        _write_json(baseline_comparison_path, comparison)
        report["baseline_comparison"] = comparison
        artifact_paths["baseline_comparison"] = str(baseline_comparison_path)

    _write_json(report_path, report)

    status = 0 if not baseline_failures else 3
    if baseline_failures:
        artifact_paths["failure"] = "; ".join(baseline_failures)
    return status, artifact_paths


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        status_code, payload = _run(args)
    except VersionCompatibilityError as error:
        print(f"ERROR: refusing to evaluate: {error}", file=sys.stderr)
        return 2
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"ERROR: Unexpected failure: {error}", file=sys.stderr)
        return 2

    if status_code != 0:
        print(f"ERROR: baseline regression gate failed: {payload['failure']}", file=sys.stderr)
    print(json.dumps(payload, ensure_ascii=False))
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
