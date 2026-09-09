"""Tests for the offline evaluator's metric layer.

These functions carry the experimental methodology, so they are tested
independently of the cache-reading plumbing around them. The case that matters
most is the one the previous revision got wrong: ``other`` is both a real
family and the sink for every refusal, and in ``evaluate`` mode the classifier
never positively predicts it, so counting a refusal as an ``other`` prediction
hands the classifier free true positives for declining to answer.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_rules_from_cache import (  # noqa: E402
    DECLINED,
    _confusion_matrix,
    _metrics,
    _ranked_from_scores,
    _selective_metrics,
)


def record(target, predicted, decision, reason="", **extra):
    base = {
        "target_family": target,
        "predicted_family": predicted,
        "decision": decision,
        "reason": reason or decision,
        "classifier_time_ms": 1.0,
        "feature_extraction_time_ms": 1.0,
    }
    base.update(extra)
    return base


class RefusalAccountingTests(unittest.TestCase):
    """A refusal is not a prediction."""

    def setUp(self):
        # Two illegible file folders the classifier declined, plus one correct
        # invoice. Ground truth for the declined pair really is ``other``, which
        # is exactly the configuration that used to award them true positives.
        self.records = [
            record("financial_document", "financial_document", "classified"),
            record("other", "other", "abstained", "insufficient_ocr_text"),
            record("other", "other", "fallback", "no_rules_matched"),
        ]

    def test_declined_samples_are_excluded_from_selective_metrics(self):
        selective = _selective_metrics(
            [r for r in self.records if r["decision"] == "classified"],
            ["financial_document", "other"],
        )
        self.assertEqual(selective["sample_count"], 1)
        self.assertEqual(selective["per_class"]["other"]["support"], 0)
        self.assertEqual(selective["per_class"]["other"]["predicted"], 0)

    def test_other_does_not_earn_recall_from_refusals(self):
        metrics = _metrics(self.records)
        self.assertEqual(metrics["selective"]["per_class"]["other"]["recall"], 0.0)
        self.assertEqual(metrics["selective"]["per_class"]["other"]["precision"], 0.0)
        self.assertNotIn("other", metrics["selective"]["macro_averaged_over"])

    def test_refusals_land_in_their_own_confusion_column(self):
        confusion = _confusion_matrix(self.records, ["financial_document", "other"])
        self.assertEqual(confusion["other"][DECLINED], 2)
        self.assertEqual(confusion["other"]["other"], 0)

    def test_end_to_end_separates_the_two_readings(self):
        metrics = _metrics(self.records)
        # Strict: only the invoice is a correct prediction.
        self.assertAlmostEqual(metrics["end_to_end"]["accuracy_declined_as_error"], 1 / 3, places=4)
        # Routing: declining a true ``other`` sends it to the fallback template,
        # which is the right outcome for the pipeline. This is the number the
        # previous revision reported as plain "accuracy".
        self.assertEqual(metrics["end_to_end"]["accuracy_declined_as_other"], 1.0)
        self.assertNotEqual(
            metrics["end_to_end"]["accuracy_declined_as_error"],
            metrics["end_to_end"]["accuracy_declined_as_other"],
        )

    def test_coverage_reports_the_refusal_rate(self):
        metrics = _metrics(self.records)
        self.assertAlmostEqual(metrics["decisions"]["coverage"], 1 / 3, places=4)
        self.assertAlmostEqual(metrics["decisions"]["abstention_or_fallback_rate"], 2 / 3, places=4)
        self.assertEqual(metrics["decisions"]["counts"], {"abstained": 1, "classified": 1, "fallback": 1})


class ObserveModeTests(unittest.TestCase):
    def test_observed_decisions_count_as_predictions(self):
        records = [
            record("other", "presentation_marketing", "observed", "argmax_without_abstention"),
            record("resume", "resume", "observed", "argmax_without_abstention"),
        ]
        metrics = _metrics(records)
        self.assertEqual(metrics["decisions"]["coverage"], 1.0)
        self.assertEqual(metrics["selective"]["sample_count"], 2)
        confusion = _confusion_matrix(records, ["other", "resume", "presentation_marketing"])
        self.assertEqual(confusion["other"][DECLINED], 0)
        self.assertEqual(confusion["other"]["presentation_marketing"], 1)


class MacroAverageTests(unittest.TestCase):
    def test_false_positives_on_a_zero_support_class_are_surfaced(self):
        """Such a class cannot be averaged over, so it must be reported instead.

        Its recall is undefined rather than zero, so including it would drag the
        macro down for a reason that is not a modelling error; excluding it
        silently is what let its false positives escape macro precision.
        """
        records = [
            record("resume", "presentation_marketing", "classified"),
            record("resume", "resume", "classified"),
        ]
        selective = _selective_metrics(records, ["resume", "presentation_marketing"])
        self.assertEqual(selective["macro_averaged_over"], ["resume"])
        self.assertEqual(selective["predictions_outside_support"], {"presentation_marketing": 1})

    def test_no_leak_reported_when_every_class_has_support(self):
        records = [
            record("resume", "resume", "classified"),
            record("news_article", "news_article", "classified"),
        ]
        selective = _selective_metrics(records, ["resume", "news_article"])
        self.assertEqual(selective["predictions_outside_support"], {})
        self.assertEqual(selective["macro_f1"], 1.0)


class RankRebuildTests(unittest.TestCase):
    def test_other_is_dropped_from_the_rebuilt_ranking(self):
        """``other`` is reported in candidate_scores but is not a scored family.

        Leaving it in would let a zero-scoring residual class take the
        runner-up slot and corrupt the margin the sweep applies.
        """
        ranked = _ranked_from_scores({"resume": 0.8, "other": 0.0, "news_article": 0.3})
        self.assertEqual([family for family, _ in ranked], ["resume", "news_article"])

    def test_ties_break_on_family_name_like_the_core(self):
        ranked = _ranked_from_scores({"resume": 0.5, "correspondence": 0.5, "other": 0.0})
        self.assertEqual(ranked, [("correspondence", 0.5), ("resume", 0.5)])


if __name__ == "__main__":
    unittest.main()
