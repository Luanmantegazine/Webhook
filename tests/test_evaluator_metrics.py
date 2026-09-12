"""Tests for the offline evaluator's metric layer.

These functions carry the experimental methodology, so they are tested
independently of the cache-reading plumbing around them.

The module was rewritten for v6. It had been importing ``DECLINED``,
``_selective_metrics`` and ``_ranked_from_scores`` — an evaluator API that no
longer existed, so the whole file raised ImportError on collection and every
assertion in it had silently stopped running. The cases it was protecting are
kept: a refusal is not a prediction, and refusals must not earn the classifier
true positives for declining to answer.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_rules_from_cache import (  # noqa: E402
    REVIEW_PRIORITIES,
    _build_family_risk_coverage,
    _build_ranked_review,
    _is_accepted_wrong_family,
    _is_rejection_false_accept,
    _is_unsafe_accept,
    _metrics,
    _per_rvl_label_metrics,
    _routing_metrics,
)
from tasks.document.rules_classifier_core import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    SCORED_FAMILIES,
)


def record(
    sample_id,
    rvl_label,
    target,
    predicted,
    decision,
    *,
    reason="",
    is_rejection=False,
    score=0.8,
    margin=0.2,
    candidate_scores=None,
    **extra,
):
    base = {
        "sample_id": sample_id,
        "rvl_label": rvl_label,
        "target_family": target,
        "source_target_family": "",
        "target_kind": "rejection" if is_rejection else "classifier_family",
        "rejection_label": rvl_label if is_rejection else "",
        "is_rejection_target": is_rejection,
        "predicted_family": predicted,
        "top_candidate": predicted if predicted != "other" else None,
        "runner_up": None,
        "decision": decision,
        "reason": reason or decision,
        "confidence": min(score, 1.0),
        "score": score,
        "score_margin": margin,
        "recognized_characters": 500,
        "effective_family_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "classifier_time_ms": 1.0,
        "feature_extraction_time_ms": 1.0,
        "rules_triggered": [],
        "candidate_scores": candidate_scores
        or {family: (score if family == predicted else 0.0) for family in SCORED_FAMILIES},
    }
    base.update(extra)
    return base


def population():
    """One correct accept, one wrong accept, one accepted file folder, two refusals."""
    return [
        record("s1", "invoice", "financial_document", "financial_document", "classified"),
        record("s2", "form", "form_structured", "correspondence", "classified", score=0.95),
        record(
            "s3",
            "file folder",
            "other",
            "presentation_marketing",
            "classified",
            is_rejection=True,
        ),
        record(
            "s4",
            "handwritten",
            "other",
            "other",
            "abstained",
            reason="insufficient_ocr_text",
            is_rejection=True,
            score=0.0,
        ),
        record(
            "s5",
            "letter",
            "correspondence",
            "other",
            "fallback",
            reason="no_rules_matched",
            score=0.0,
        ),
    ]


class UnsafeAcceptTests(unittest.TestCase):
    """A refusal costs coverage; an accepted wrong answer costs trust."""

    def setUp(self):
        self.records = population()

    def test_accepted_wrong_family_is_identified(self):
        self.assertTrue(_is_accepted_wrong_family(self.records[1]))
        self.assertFalse(_is_accepted_wrong_family(self.records[0]))
        # A refusal is wrong, but it is not an unsafe *accept*.
        self.assertFalse(_is_accepted_wrong_family(self.records[4]))

    def test_accepted_rejection_target_is_identified(self):
        self.assertTrue(_is_rejection_false_accept(self.records[2]))
        self.assertFalse(_is_rejection_false_accept(self.records[3]))

    def test_unsafe_accepts_are_the_union(self):
        unsafe = [item["sample_id"] for item in self.records if _is_unsafe_accept(item)]
        self.assertEqual(unsafe, ["s2", "s3"])

    def test_metrics_report_the_four_v6_fields(self):
        metrics = _metrics(self.records)
        self.assertEqual(metrics["accepted_wrong_family_count"], 1)
        self.assertEqual(metrics["unsafe_accept_count"], 2)
        # Three accepted decisions, two of them unsafe.
        self.assertAlmostEqual(metrics["unsafe_accept_rate"], round(2 / 3, 4))
        self.assertAlmostEqual(metrics["accepted_routing_accuracy"], 0.5)

    def test_refusals_do_not_count_as_accepted_answers(self):
        """The case the previous revision of this module existed to protect."""
        metrics = _metrics(self.records)
        routing = metrics["routing"]
        self.assertEqual(routing["accepted_count"], 3)
        self.assertEqual(routing["accepted_correct_count"], 1)
        # s4 declined a rejection target correctly: it earns no true positive.
        self.assertEqual(
            metrics["rejection_quality"]["correct_rejection_count"], 1
        )


class RoutingMetricTests(unittest.TestCase):
    def setUp(self):
        self.routing, self.rows = _routing_metrics(
            population(), DEFAULT_CONFIDENCE_THRESHOLD
        )

    def test_every_scorable_family_is_measured(self):
        self.assertEqual(set(self.routing["per_family"]), set(SCORED_FAMILIES))
        self.assertEqual([row["family"] for row in self.rows], list(SCORED_FAMILIES))

    def test_accepted_rejection_target_counts_against_the_accepting_family(self):
        entry = self.routing["per_family"]["presentation_marketing"]
        self.assertEqual(entry["accepted_false_positive_count"], 1)
        self.assertEqual(entry["accepted_rejection_false_accept_count"], 1)
        self.assertEqual(entry["accepted_precision"], 0.0)

    def test_effective_threshold_is_reported_per_family(self):
        self.assertEqual(
            self.routing["per_family"]["form_structured"]["effective_family_threshold"],
            0.40,
        )
        self.assertEqual(
            self.routing["per_family"]["news_article"]["effective_family_threshold"],
            DEFAULT_CONFIDENCE_THRESHOLD,
        )


class PerLabelMetricTests(unittest.TestCase):
    def test_metrics_are_reported_per_original_label(self):
        payload, rows = _per_rvl_label_metrics(population())
        self.assertEqual(
            set(payload), {"invoice", "form", "file folder", "handwritten", "letter"}
        )
        self.assertEqual(payload["form"]["unsafe_accept_count"], 1)
        self.assertEqual(payload["file folder"]["unsafe_accept_count"], 1)
        self.assertEqual(payload["handwritten"]["canonical_correct_count"], 1)
        self.assertEqual(len(rows), 5)

    def test_rejection_labels_declare_their_own_criterion(self):
        payload, _rows = _per_rvl_label_metrics(population())
        self.assertEqual(
            payload["file folder"]["correct_criterion"], "abstained or fallback"
        )
        self.assertEqual(
            payload["invoice"]["correct_criterion"], "predicted_family == target_family"
        )


class ReviewOrderingTests(unittest.TestCase):
    def setUp(self):
        self.review = _build_ranked_review(population(), review_limit=0)
        self.by_id = {row["sample_id"]: row for row in self.review}

    def test_the_ambiguous_correct_column_is_gone(self):
        self.assertNotIn("correct", self.review[0])
        for field in (
            "canonical_correct",
            "scope_correct",
            "is_unsafe_accept",
            "is_rejection_false_accept",
        ):
            self.assertIn(field, self.review[0])

    def test_accepted_rejection_target_is_reviewed_first(self):
        self.assertEqual(self.review[0]["sample_id"], "s3")
        self.assertEqual(self.review[0]["review_priority_label"], "rejection_target_accepted")

    def test_high_confidence_wrong_accept_outranks_a_plain_one(self):
        """s2 is accepted, wrong, and confident: the most misleading kind."""
        self.assertEqual(
            self.by_id["s2"]["review_priority_label"], "high_confidence_false_positive"
        )
        self.assertLess(self.by_id["s2"]["review_rank"], self.by_id["s5"]["review_rank"])

    def test_fallback_without_rules_is_reviewed_last_of_the_errors(self):
        self.assertEqual(
            self.by_id["s5"]["review_priority_label"], "fallback_without_rules"
        )

    def test_declined_rejection_target_is_not_an_error(self):
        self.assertTrue(self.by_id["s4"]["canonical_correct"])
        self.assertFalse(self.by_id["s4"]["is_unsafe_accept"])

    def test_priority_order_is_the_declared_one(self):
        self.assertEqual(
            [label for _rank, label in REVIEW_PRIORITIES],
            [
                "rejection_target_accepted",
                "accepted_wrong_family",
                "high_confidence_false_positive",
                "false_negative_near_threshold",
                "fallback_without_rules",
                "other",
            ],
        )


class RiskCoverageTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.rows = _build_family_risk_coverage(
            population(),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            min_score_margin=0.10,
            min_recognized_characters=20,
            classification_mode="evaluate",
        )

    def test_curves_cover_every_scorable_family(self):
        self.assertEqual(
            [curve["target_family"] for curve in self.payload["curves"]],
            list(SCORED_FAMILIES),
        )

    def test_rejection_targets_are_negatives_not_exclusions(self):
        for curve in self.payload["curves"]:
            with self.subTest(family=curve["target_family"]):
                self.assertEqual(curve["negative_rejection_count"], 2)
                self.assertEqual(curve["included_sample_count"], 5)

    def test_points_report_the_effective_family_threshold(self):
        curve = next(
            item for item in self.payload["curves"] if item["target_family"] == "form_structured"
        )
        for point in curve["points"]:
            with self.subTest(threshold=point["confidence_threshold"]):
                self.assertIn("effective_family_threshold", point)
                # The family's bar tracks the swept global threshold at its
                # declared offset, and clamps at zero rather than going
                # negative — so the two coincide only at the bottom of a sweep.
                expected = max(
                    0.0,
                    point["confidence_threshold"] - (DEFAULT_CONFIDENCE_THRESHOLD - 0.40),
                )
                self.assertAlmostEqual(point["effective_family_threshold"], expected)


if __name__ == "__main__":
    unittest.main()
