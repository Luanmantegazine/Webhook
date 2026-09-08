from __future__ import annotations

import unittest

from tasks.document.rules_classifier_core import (
    classify_with_rules,
    extract_classification_features,
)


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

    def test_business_report(self):
        _, result = classify(make_document(
            "ANNUAL REPORT\nExecutive Summary\nFiscal year revenue and expenses\n"
            "Budget variance and KPI forecast",
            ["Title", "Section-header", "Text", "Table"],
        ))
        self.assertEqual(result["document_family"], "business_report")

    def test_form(self):
        _, result = classify(make_document(
            "REGISTRATION FORM\nName: _____\nAddress: _____\nDate: _____\n[ ] Yes [ ] No",
            ["Title", "Text", "Text", "Text", "List-item", "List-item"],
        ))
        self.assertEqual(result["document_family"], "form_structured")

    def test_presentation(self):
        _, result = classify(
            make_document(
                "Project Presentation\nAgenda\n[ ] Market\n[ ] Product",
                ["Title", "Picture", "Picture", "List-item", "List-item"],
            ),
            page_size=[1600, 900],
        )
        self.assertEqual(result["document_family"], "presentation_marketing")

    def test_unmatched_document_falls_back_to_other(self):
        _, result = classify(make_document(
            "Hello team. The meeting has moved to Tuesday afternoon. Please confirm attendance."
        ))
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["decision"], "fallback")
        self.assertEqual(result["reason"], "score_below_threshold")

    def test_insufficient_ocr_abstains(self):
        _, result = classify(make_document("x", ["Text"]))
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["reason"], "insufficient_ocr_text")


if __name__ == "__main__":
    unittest.main()
