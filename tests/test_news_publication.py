"""Tests for the v8 news publication family.

The family's public key stays ``news_article`` — it is the RVL-CDIP class name
and the key every workflow reads — while the documents it must recognise now
include whole newspaper issues as well as single clipped articles.

Every fixture here is built from structure, never from a document's wording: a
nameplate over an edition line, a column grid, headlines with articles under
them. No test references a publication name, and the Portuguese fixtures use
ordinary municipal-news vocabulary for that reason.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.document.rules_classifier_core import (  # noqa: E402
    RULE_IDS,
    classify_with_rules,
    extract_classification_features,
)
from tasks.document.rvl_cdip_eval import (  # noqa: E402
    DOCUMENT_SUBTYPES,
    FAMILY_CONCEPTUAL_NAMES,
    FAMILY_SUBTYPES,
    is_known_subtype,
)

PAGE_WIDTH, PAGE_HEIGHT = 1240, 1750
COLUMNS = [(80, 460), (470, 850), (860, 1160)]

PT_LEAD = (
    "O prefeito afirmou que “as obras começam em março” durante entrevista. "
    "Segundo a administração, o investimento será dividido em etapas. O secretário "
    "disse que o cronograma foi revisado e acrescentou que a população será consultada."
)
PT_SECOND = (
    "Outra matéria da coluna trata do transporte público e do novo terminal. O diretor "
    "informou que a licitação foi concluída e destacou o prazo de entrega da obra."
)
EN_LEAD = (
    "The mayor said “the works begin in March” during an interview. According to the "
    "administration the investment will be staged. The secretary stated the schedule was "
    "revised and added that residents will be consulted about the project."
)
EN_SECOND = (
    "Another story in this column covers public transport and the new terminal. The "
    "director reported that the tender closed and commented on the delivery deadline."
)


def region(text, class_name, bbox):
    return {"class_name": class_name, "text": text, "bbox": bbox}


def newspaper_page(page_number, *, language="pt", masthead=True, running_header=True):
    """One newspaper page: optional nameplate, then a three-column grid."""
    regions = []
    if masthead and page_number == 1:
        nameplate = "O DIÁRIO REGIONAL" if language == "pt" else "THE REGIONAL DAILY"
        metadata = (
            "Ano XXXI  Edição 717  São Paulo, 24 de janeiro de 2026  "
            "R$ 1,00  www.diarioregional.com.br"
            if language == "pt"
            else "Year 118  Issue 717  January 24, 2026  $1.00  www.regionaldaily.com"
        )
        regions.append(region(nameplate, "Title", [80, 40, 1160, 190]))
        regions.append(region(metadata, "Text", [80, 200, 1160, 240]))
    elif running_header:
        header = (
            "O DIÁRIO REGIONAL - 24 de janeiro de 2026"
            if language == "pt"
            else "THE REGIONAL DAILY - January 24, 2026"
        )
        regions.append(region(header, "Text", [80, 40, 1160, 90]))

    lead, second = (PT_LEAD, PT_SECOND) if language == "pt" else (EN_LEAD, EN_SECOND)
    top = 300
    for index, (left, right) in enumerate(COLUMNS):
        headline = (
            f"Prefeitura anuncia obras na zona {page_number}{index}"
            if language == "pt"
            else f"Council approves new works in district {page_number}{index}"
        )
        regions.append(region(headline, "Section-header", [left, top, right, top + 60]))
        regions.append(region(lead, "Text", [left, top + 70, right, top + 900]))
        regions.append(region(second, "Text", [left, top + 920, right, top + 1350]))
    # Newspapers carry photographs and advertisements; neither is evidence.
    regions.append(region("", "Picture", [80, 1670, 600, 1740]))
    return {"page_number": page_number, "regions": regions}


def newspaper_issue(pages=4, **kwargs):
    return {
        "total_pages": pages,
        "pages": [newspaper_page(number, **kwargs) for number in range(1, pages + 1)],
        "full_text": "",
    }


def single_article(language="pt"):
    byline = "Por João Silva" if language == "pt" else "By Jane Roberts"
    headline = (
        "Prefeitura anuncia novo terminal"
        if language == "pt"
        else "Council announces new terminal"
    )
    body = PT_LEAD if language == "pt" else EN_LEAD
    return {
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    region(headline, "Title", [80, 60, 1160, 160]),
                    region(byline, "Text", [80, 180, 600, 220]),
                    region(body, "Text", [80, 240, 1160, 1200]),
                ],
            }
        ],
        "full_text": "",
    }


def classify(document, pages=None):
    page_count = pages or document["total_pages"]
    features = extract_classification_features(
        document, page_sizes=[[PAGE_WIDTH, PAGE_HEIGHT]] * page_count
    )
    return features, classify_with_rules(features, include_indicators=True)


def news_gate(result):
    return result["evidence"]["family_gates"]["news_article"]


class NewspaperIssueTests(unittest.TestCase):
    def test_portuguese_newspaper_is_classified_as_a_newspaper_issue(self):
        _features, result = classify(newspaper_issue(language="pt"))
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")
        self.assertEqual(news_gate(result)["reason"], "path_newspaper_issue_masthead")

    def test_english_newspaper_is_classified_the_same_way(self):
        _features, result = classify(newspaper_issue(language="en"))
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")

    def test_a_running_header_substitutes_for_a_missing_masthead(self):
        """A scan whose first page is missing is still a newspaper."""
        _features, result = classify(newspaper_issue(masthead=False))
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(
            news_gate(result)["reason"], "path_newspaper_issue_running_header"
        )

    def test_the_structure_features_describe_the_issue(self):
        features, _result = classify(newspaper_issue())
        self.assertEqual(features["estimated_column_count_by_page"], [3, 3, 3, 3])
        self.assertEqual(features["multi_column_page_ratio"], 1.0)
        self.assertGreaterEqual(features["headline_count"], 4)
        self.assertGreaterEqual(features["article_cluster_count"], 3)
        self.assertGreaterEqual(features["publication_masthead_count"], 1)
        self.assertGreaterEqual(features["issue_metadata_count"], 1)
        self.assertGreaterEqual(features["newspaper_page_count"], 2)

    def test_three_or_more_columns_are_represented(self):
        """``two_column_ratio`` cannot say "three"; the column count can."""
        features, _result = classify(newspaper_issue())
        self.assertEqual(features["max_narrative_column_count"], 3)

    def test_the_gate_record_explains_the_decision(self):
        _features, result = classify(newspaper_issue())
        gate = news_gate(result)
        self.assertEqual(gate["status"], "satisfied")
        self.assertEqual(gate["document_subtype"], "newspaper_issue")
        self.assertIn("newspaper_issue_masthead", gate["satisfied_paths"])
        self.assertIn("publication_masthead", gate["satisfied_requirements"])
        self.assertIn("multi_column_page_ratio", gate["metrics"])
        self.assertEqual(
            sorted(gate["available_subtypes"]),
            ["newspaper_issue", "single_news_article"],
        )


class SingleArticleTests(unittest.TestCase):
    def test_portuguese_byline_is_recognised(self):
        """v7 only knew ``By First Last``; a Portuguese article had no byline."""
        _features, result = classify(single_article("pt"))
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "single_news_article")
        self.assertEqual(news_gate(result)["reason"], "path_byline_with_reporting")

    def test_english_byline_still_works(self):
        _features, result = classify(single_article("en"))
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "single_news_article")


class NewspaperFalsePositiveTests(unittest.TestCase):
    """Everything that has columns and headings but is not journalism."""

    def catalogue(self):
        pages = []
        for number in range(1, 5):
            regions = [
                region("SPRING CATALOGUE", "Title", [80, 40, 1160, 190]),
                region(
                    "Edition 3  January 24, 2026  www.shop.example.com",
                    "Text",
                    [80, 200, 1160, 240],
                ),
            ]
            top = 300
            for index, (left, right) in enumerate(COLUMNS):
                regions.append(
                    region(f"Product line {index}", "Section-header", [left, top, right, top + 60])
                )
                regions.append(
                    region(
                        "Available in three finishes with a two year warranty and free "
                        "delivery on every order. Dimensions and colours vary by region "
                        "and by available stock in each store.",
                        "Text",
                        [left, top + 70, right, top + 900],
                    )
                )
            pages.append({"page_number": number, "regions": regions})
        return {"total_pages": 4, "pages": pages, "full_text": ""}

    def journal_article(self):
        pages = []
        for number in range(1, 5):
            regions = [
                region("Deep Structural Analysis of Porous Media", "Title", [80, 40, 1160, 200]),
                region(
                    "Journal of Applied Physics  Vol. 12  No. 3  March 14, 2021  "
                    "doi.org/10.1000/example",
                    "Text",
                    [80, 210, 1160, 250],
                ),
            ]
            top = 300
            for left, right in [(80, 600), (640, 1160)]:
                regions.append(region("Methodology", "Section-header", [left, top, right, top + 60]))
                regions.append(
                    region(
                        "Abstract: we study the effect. References [1] [2] show the results "
                        "are consistent. The experiment was repeated and the conclusions hold.",
                        "Text",
                        [left, top + 70, right, top + 900],
                    )
                )
            pages.append({"page_number": number, "regions": regions})
        return {"total_pages": 4, "pages": pages, "full_text": ""}

    def test_a_catalogue_is_not_a_newspaper(self):
        """Identity, structure and layout without reporting is a catalogue."""
        _features, result = classify(self.catalogue())
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertIsNone(result["document_subtype"])
        self.assertEqual(result["candidate_scores"]["news_article"], 0.0)
        self.assertIn(
            "requires one of attribution_quotes/multilingual_reporting",
            news_gate(result)["missing_requirements"],
        )

    def test_a_two_column_journal_article_is_not_a_newspaper(self):
        _features, result = classify(self.journal_article())
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertEqual(news_gate(result)["status"], "blocked")
        self.assertIn("research_publication_evidence", news_gate(result)["blocked_by"])


class FinancialGuardTests(unittest.TestCase):
    """A newspaper is full of numbers that are not accounting."""

    def test_dialling_codes_are_not_accounting_negatives(self):
        document = {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region(
                            "Ligue (011) 3333-4444 ou (11) 99999-1234 para anúncios. "
                            "Ver nota (3) na página seguinte.",
                            "Text",
                            [80, 80, 1160, 400],
                        )
                    ],
                }
            ],
            "full_text": "",
        }
        features, _result = classify(document)
        self.assertEqual(features["accounting_negative_count"], 0)

    def test_parenthesised_money_still_counts(self):
        document = {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region(
                            "Ledger balance (1,234.00) and subtotal ($2,500.50) with "
                            "disbursements recorded against the account.",
                            "Text",
                            [80, 80, 1160, 400],
                        )
                    ],
                }
            ],
            "full_text": "",
        }
        features, _result = classify(document)
        self.assertGreaterEqual(features["accounting_negative_count"], 2)

    def test_a_newspaper_is_not_a_financial_document(self):
        _features, result = classify(newspaper_issue())
        self.assertEqual(result["candidate_scores"].get("financial_document", 0.0), 0.0)

    def test_news_does_not_block_financial_evidence(self):
        """A financial statement still wins on its own evidence."""
        pages = []
        for number in range(1, 4):
            pages.append(
                {
                    "page_number": number,
                    "regions": [
                        region("ANNUAL FINANCIAL STATEMENTS", "Title", [80, 40, 1160, 190]),
                        region("Year 2025  Edition 1  January 24, 2026", "Text", [80, 200, 1160, 240]),
                        {
                            "class_name": "Table",
                            "text": "",
                            "table_data": {
                                "text_repr": (
                                    "January 1,200.00 (1,340.00)\nFebruary 2,400.00 (2,100.00)\n"
                                    "March 3,100.00 (900.00)\nBalance sheet ledger subtotal "
                                    "$12,400.00 amount due"
                                )
                            },
                            "bbox": [80, 300, 1160, 1400],
                        },
                    ],
                }
            )
        _features, result = classify({"total_pages": 3, "pages": pages, "full_text": ""}, pages=3)
        self.assertEqual(result["document_family"], "financial_document")

    def test_month_names_alone_are_not_a_monthly_series(self):
        indicators = {rule_id: False for rule_id in RULE_IDS}
        del indicators  # the rule is exercised through the document below
        document = {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region(
                            "As obras começam em janeiro, seguem em fevereiro e março, "
                            "e terminam em abril segundo a prefeitura, que informou o "
                            "cronograma completo aos moradores da região nesta semana.",
                            "Text",
                            [80, 80, 1160, 600],
                        )
                    ],
                }
            ],
            "full_text": "",
        }
        _features, result = classify(document)
        self.assertFalse(result["rule_indicators"]["financial_document.monthly_series"])


class SubtypeTaxonomyTests(unittest.TestCase):
    def test_subtypes_are_declared_for_the_legacy_key(self):
        self.assertEqual(
            FAMILY_SUBTYPES["news_article"], ("newspaper_issue", "single_news_article")
        )
        self.assertEqual(FAMILY_CONCEPTUAL_NAMES["news_article"], "news_publication")
        self.assertEqual(
            DOCUMENT_SUBTYPES, frozenset({"newspaper_issue", "single_news_article"})
        )

    def test_a_subtype_is_only_valid_for_its_own_family(self):
        self.assertTrue(is_known_subtype("news_article", "newspaper_issue"))
        self.assertTrue(is_known_subtype("news_article", None))
        self.assertFalse(is_known_subtype("financial_document", "newspaper_issue"))
        self.assertFalse(is_known_subtype("news_article", "magazine_issue"))

    def test_the_public_family_key_is_unchanged(self):
        _features, result = classify(newspaper_issue())
        self.assertEqual(result["document_family"], "news_article")
        self.assertIn("news_article", result["candidate_scores"])


if __name__ == "__main__":
    unittest.main()
