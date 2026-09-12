from __future__ import annotations

import json
from pathlib import Path
import unittest

from tasks.document.rules_classifier_core import (
    DOCUMENT_FAMILIES,
    TAXONOMY_VERSION,
    classify_with_rules,
    extract_classification_features,
)

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "rvl_cdip_taxonomy.json"


def unmatched_document() -> dict:
    """A document long enough to clear the OCR gate but matching no rule.

    200 filler words in a single ``Text`` region: too long for the
    presentation-marketing text-density rules, too short and too unstructured
    for the business-report narrative rule, and lexically empty.
    """
    body = " ".join(f"lorem{index}" for index in range(200))
    return {
        "pages": [
            {"regions": [{"class_name": "Text", "text": body, "bbox": [10, 10, 1100, 1500]}]}
        ],
        "full_text": body,
    }


def make_document(text: str, classes: list[str] | None = None) -> dict:
    selected_classes = classes or ["Title", "Text", "Text", "Text"]
    regions = []
    for index, class_name in enumerate(selected_classes):
        x0 = 80 if index % 2 == 0 else 650
        x1 = 550 if index % 2 == 0 else 1120
        regions.append(
            {
                "class_name": class_name,
                "text": text if index == 0 else "Supporting content for this document.",
                "bbox": [x0, 80 + index * 160, x1, 190 + index * 160],
                "table_data": {},
            }
        )
    return {
        "total_pages": 1,
        "pages": [{"page_number": 1, "regions": regions}],
        "full_text": text,
    }


def classify(document: dict, page_size=None):
    features = extract_classification_features(document, page_sizes=[page_size or [1200, 1600]])
    return features, classify_with_rules(features)


class FeatureExtractionTests(unittest.TestCase):
    def test_uses_measured_page_size_for_orientation(self):
        features = extract_classification_features(
            make_document("Agenda and project presentation"),
            page_sizes=[[1600, 900]],
        )
        self.assertEqual(features["landscape_ratio"], 1.0)

    def test_extracts_layout_counts_and_form_signals(self):
        document = make_document(
            "APPLICATION FORM\nName: _____\nDate: _____\n[ ] Yes [ ] No",
            ["Title", "Text", "Text", "List-item", "Table"],
        )
        features = extract_classification_features(document, page_sizes=[[1200, 1600]])
        self.assertEqual(features["class_counts"]["Table"], 1)
        self.assertGreaterEqual(features["checkbox_count"], 2)
        self.assertGreaterEqual(features["blank_field_count"], 2)


class TaxonomyConfigTests(unittest.TestCase):
    """The shipped taxonomy and the classifier must agree.

    They drifted once: the config stayed at the 7-family v1 while the core
    moved to 10 families, so every letter, memo, e-mail, resume and news
    article carried an ``other`` ground-truth label that the classifier could
    never predict. That silently floors recall for four families instead of
    failing loudly.
    """

    def setUp(self):
        with TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
            self.taxonomy = json.load(handle)

    def test_taxonomy_version_matches_classifier(self):
        self.assertEqual(self.taxonomy["taxonomy_version"], TAXONOMY_VERSION)

    def test_supported_families_match_classifier(self):
        self.assertEqual(tuple(self.taxonomy["supported_families"]), DOCUMENT_FAMILIES)

    def test_every_mapped_target_is_a_known_family(self):
        unknown = set(self.taxonomy["label_mapping"].values()) - set(DOCUMENT_FAMILIES)
        self.assertEqual(unknown, set())

    def test_every_scored_family_is_reachable_from_some_rvl_label(self):
        mapped = set(self.taxonomy["label_mapping"].values())
        unreachable = set(DOCUMENT_FAMILIES) - mapped
        self.assertEqual(unreachable, set())


class DecisionPolicyTests(unittest.TestCase):
    def test_observe_mode_reports_a_score_when_no_rule_matches(self):
        """``observe`` used to raise KeyError('score') on this path.

        It is the default mode of the task wrapper and of both workflows, so
        any document matching no rule aborted the pipeline.
        """
        features = extract_classification_features(
            unmatched_document(), page_sizes=[[1200, 1600]]
        )
        result = classify_with_rules(features, mode="observe")
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["decision"], "observed")
        self.assertEqual(result["reason"], "no_rules_matched")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["confidence"], 0.0)

    def test_every_mode_returns_the_full_result_contract(self):
        features = extract_classification_features(
            unmatched_document(), page_sizes=[[1200, 1600]]
        )
        required = {
            "schema_version",
            "taxonomy_version",
            "classifier_version",
            "document_family",
            "confidence",
            "score",
            "decision",
            "reason",
            "score_margin",
            "candidate_scores",
            "evidence",
            "thresholds",
        }
        for mode in ("observe", "evaluate", "auto"):
            with self.subTest(mode=mode):
                result = classify_with_rules(features, mode=mode)
                self.assertEqual(required - set(result), set())


class ClassificationTests(unittest.TestCase):
    def test_invoice(self):
        _, result = classify(make_document(
            "INVOICE # 9381\nBill To: Example Ltd\nSubtotal $100.00\nTax $10.00\nTotal Due $110.00",
            ["Title", "Text", "Table", "Text"],
        ))
        self.assertEqual(result["document_family"], "financial_document")
        self.assertEqual(result["decision"], "classified")

    def test_research_paper(self):
        _, result = classify(make_document(
            "Abstract: We present a study. DOI 10.1000/example.123\n"
            "Methodology\nExperimental Results [1] [2]\nReferences",
            ["Title", "Text", "Section-header", "Text", "Formula", "Text"],
        ))
        self.assertEqual(result["document_family"], "research_paper")

    def test_technical_report(self):
        _, result = classify(make_document(
            "TECHNICAL SPECIFICATION\n1. Scope\n2. System Requirements\n"
            "The component shall comply with the test procedure.",
            ["Title", "Section-header", "Text", "Table"],
        ))
        self.assertEqual(result["document_family"], "technical_report")

    def test_business_report_is_never_predicted(self):
        """``business_report`` is in the taxonomy but excluded from scoring.

        No canonical RVL-CDIP target maps to it, so any prediction of it would
        be a false positive by construction. This test asserted the opposite
        and had been failing since the family was excluded — a red test nobody
        could make green, which is worse than no test: it hid the two genuine
        failures beside it.
        """
        _, result = classify(make_document(
            "ANNUAL REPORT\nExecutive Summary\nFiscal year revenue and expenses\n"
            "Budget variance and KPI forecast",
            ["Title", "Section-header", "Text", "Table"],
        ))
        self.assertNotIn("business_report", result["candidate_scores"])
        self.assertNotEqual(result["document_family"], "business_report")

    def test_form(self):
        _, result = classify(make_document(
            "REGISTRATION FORM\nName: _____\nAddress: _____\nDate: _____\n[ ] Yes [ ] No",
            ["Title", "Text", "Text", "Text", "List-item", "List-item"],
        ))
        self.assertEqual(result["document_family"], "form_structured")

    def test_presentation(self):
        """Deck vocabulary in heading position, corroborated by pictures.

        ``Agenda`` is now required to stand as a heading line rather than to
        appear anywhere on the page, and the pictures corroborate the case
        rather than making it: see ``tests/test_family_gates.py`` for the
        refusal side of the same rule.
        """
        _, result = classify(
            make_document(
                "Agenda\nMarket overview\nProduct roadmap",
                ["Title", "Picture", "Picture", "List-item", "List-item"],
            ),
            page_size=[1600, 900],
        )
        self.assertEqual(result["document_family"], "presentation_marketing")

    def test_unmatched_document_falls_back_to_other(self):
        """A document matching no rule is refused, and says why.

        The reason changed with the v5 gates: the only rule this text used to
        fire was ``two_column_layout``, and layout alone no longer opens a
        decision, so the refusal is now ``no_rules_matched`` rather than a
        score below the bar. Both are refusals to ``other``; the test asserts
        the refusal and accepts either reason for it.
        """
        _, result = classify(make_document(
            "Hello team. The meeting has moved to Tuesday afternoon. Please confirm attendance."
        ))
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["decision"], "fallback")
        self.assertIn(result["reason"], {"no_rules_matched", "score_below_threshold"})

    def test_insufficient_ocr_abstains(self):
        _, result = classify(make_document("x", ["Text"]))
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["reason"], "insufficient_ocr_text")


if __name__ == "__main__":
    unittest.main()
