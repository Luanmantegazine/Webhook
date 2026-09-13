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


class CompositeDocumentTests(unittest.TestCase):
    """A newspaper contains other kinds of document without becoming them.

    The family's guards were written for a single article — advertising copy, a
    salutation, a form field, a reference list each say "not journalism" about
    one document with one subject. An issue is a container: the advertisements
    pay for it, the letters page is correspondence, the coupon is a form, the
    book review carries citations. Evaluated over the whole document they read
    its contents as proof that the container is something else.
    """

    def issue_containing(self, *extra_regions, pages=4):
        """A structurally valid issue with extra material dropped into page 2."""
        document = newspaper_issue(pages=pages)
        document["pages"][1]["regions"].extend(extra_regions)
        return document

    def assert_still_a_newspaper(self, document, pages=4, blocked_guard=None):
        _features, result = classify(document, pages=pages)
        gate = news_gate(result)
        self.assertEqual(result["document_family"], "news_article", gate)
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")
        if blocked_guard is not None:
            # Without this the test could pass because the inserted content
            # triggered nothing at all, which would prove nothing.
            self.assertIn(
                blocked_guard,
                gate["path_evaluations"]["byline_with_reporting"]["blocked_by"],
                f"{blocked_guard} did not fire; the fixture proves nothing",
            )
            self.assertEqual(
                gate["path_evaluations"]["newspaper_issue_masthead"]["blocked_by"], []
            )
        return result

    def test_an_advertisement_inside_an_issue_does_not_block_it(self):
        """The regression this change exists for."""
        result = self.assert_still_a_newspaper(
            self.issue_containing(
                region(
                    "SPECIAL OFFER - limited time only. Order now and get a "
                    "money-back guarantee on your first purchase.",
                    "Text",
                    [860, 1450, 1160, 1650],
                )
            )
        )
        gate = news_gate(result)
        # The advertisement did veto the single-article reading, and that is
        # the point: the guard still applies where it was written to apply.
        self.assertIn(
            "advertisement_evidence",
            gate["path_evaluations"]["byline_with_reporting"]["blocked_by"],
        )
        self.assertEqual(
            gate["path_evaluations"]["newspaper_issue_masthead"]["blocked_by"], []
        )

    def test_a_letters_page_does_not_block_the_issue(self):
        self.assert_still_a_newspaper(
            self.issue_containing(
                region(
                    "Dear Editor,\nI write to object to the proposed timetable "
                    "for the works in the central district.\nSincerely yours,\nA reader",
                    "Text",
                    [860, 1450, 1160, 1650],
                )
            ),
            blocked_guard="correspondence_evidence",
        )

    def test_a_coupon_does_not_block_the_issue(self):
        # "Registration form" rather than "subscription form": the point of the
        # test is that ``form_evidence`` *fires* and is not applied to the
        # publication, so the heading has to be one the form lexicon knows.
        self.assert_still_a_newspaper(
            self.issue_containing(
                region("REGISTRATION FORM", "Section-header", [80, 1450, 460, 1500]),
                region(
                    "Name: ______\nAddress: ______\nCity: ______\n[ ] One year  [ ] Two years",
                    "Text",
                    [80, 1510, 460, 1650],
                ),
            ),
            blocked_guard="form_evidence",
        )

    def test_a_book_review_with_citations_does_not_block_the_issue(self):
        self.assert_still_a_newspaper(
            self.issue_containing(
                region(
                    "Abstract: the author revisits the archive. References [1] [2] "
                    "are discussed at length, and the conclusions follow the method "
                    "set out by the researchers. doi.org/10.1000/example",
                    "Text",
                    [470, 1450, 850, 1650],
                )
            ),
            blocked_guard="research_publication_evidence",
        )

    def test_all_four_together_still_do_not_block_the_issue(self):
        self.assert_still_a_newspaper(
            self.issue_containing(
                region(
                    "SPECIAL OFFER - limited time. Order now, money-back guarantee.",
                    "Text",
                    [860, 1450, 1160, 1560],
                ),
                region(
                    "Dear Editor,\nI object to the timetable.\nSincerely yours,\nA reader",
                    "Text",
                    [860, 1570, 1160, 1650],
                ),
                region("REGISTRATION FORM", "Section-header", [80, 1450, 460, 1500]),
                region(
                    "Name: ______\nAddress: ______\n[ ] One year  [ ] Two years",
                    "Text",
                    [80, 1510, 460, 1650],
                ),
                region(
                    "Abstract: References [1] [2] and doi.org/10.1000/example",
                    "Text",
                    [470, 1450, 850, 1650],
                ),
            )
        )

    def test_a_press_release_inside_an_issue_does_not_block_it(self):
        """Reprinting one is not being one."""
        self.assert_still_a_newspaper(
            self.issue_containing(
                region(
                    "FOR IMMEDIATE RELEASE - the company announced its results "
                    "for the quarter ending in March.",
                    "Text",
                    [860, 1450, 1160, 1650],
                )
            ),
            blocked_guard="press_release_evidence",
        )

    def test_a_press_release_on_its_own_is_still_not_an_article(self):
        """The guard is preserved exactly where it was written to apply."""
        document = single_article("en")
        document["pages"][0]["regions"].insert(
            0, region("FOR IMMEDIATE RELEASE", "Title", [80, 20, 600, 55])
        )
        _features, result = classify(document)
        gate = news_gate(result)
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertEqual(gate["status"], "blocked")
        self.assertIn(
            "press_release_evidence",
            gate["path_evaluations"]["byline_with_reporting"]["blocked_by"],
        )

    def test_an_advertising_circular_is_still_not_a_newspaper(self):
        """Dominance, not presence: the guard that survives on the issue paths."""
        pages = []
        for number in range(1, 3):
            regions = [
                region("WEEKLY DEALS", "Title", [80, 40, 1160, 190]),
                region(
                    "Issue 42  January 24, 2026  www.deals.example.com",
                    "Text",
                    [80, 200, 1160, 240],
                ),
            ]
            top = 300
            for index, (left, right) in enumerate(COLUMNS):
                regions.append(
                    region(f"Deal of the day {index}", "Section-header", [left, top, right, top + 60])
                )
                regions.append(
                    region(
                        "SPECIAL OFFER, limited time. Order now! Free trial and a "
                        "money-back guarantee. Buy one, discount applies. Sale ends "
                        "soon, satisfaction guaranteed, call today.",
                        "Text",
                        [left, top + 70, right, top + 900],
                    )
                )
            pages.append({"page_number": number, "regions": regions})
        document = {"total_pages": 2, "pages": pages, "full_text": ""}
        _features, result = classify(document, pages=2)
        self.assertNotEqual(result["document_family"], "news_article")

    def test_the_gate_records_which_blockers_each_path_evaluated(self):
        _features, result = classify(newspaper_issue())
        gate = news_gate(result)
        single = gate["path_evaluations"]["byline_with_reporting"]
        issue = gate["path_evaluations"]["newspaper_issue_masthead"]
        self.assertTrue(single["blockers_inherited_from_family"])
        self.assertIn("press_release_evidence", single["blockers_evaluated"])
        self.assertFalse(issue["blockers_inherited_from_family"])
        self.assertEqual(
            issue["blockers_evaluated"], ["predominantly_advertising_evidence"]
        )
        self.assertIn("blockers_evaluated", gate)
        self.assertEqual(issue["subtype"], "newspaper_issue")


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


class ScannedNewsprintConventionTests(unittest.TestCase):
    """Conventions a real scanned front page uses that v8 first did not read.

    Each of these was found by running the classifier over an actual newspaper
    page rather than over a fixture, and each is a general newspaper or scanning
    convention — not a property of the page that exposed it.
    """

    def test_bylines_are_set_in_capitals(self):
        """``BY JONATHAN MARTIN`` is the ordinary newspaper setting."""
        from tasks.document.rules_classifier_core import _BYLINE_RE

        for line in (
            "BY JONATHAN MARTIN",
            "BY JIM RUTENBERG AND NICK CORASANITI",
            "POR JOÃO SILVA",
            "By Jane Roberts",
            "Por João Silva",
        ):
            with self.subTest(line=line):
                self.assertTrue(_BYLINE_RE.search(line), line)

    def test_a_lowercase_by_in_prose_is_not_a_byline(self):
        from tasks.document.rules_classifier_core import _BYLINE_RE

        self.assertIsNone(
            _BYLINE_RE.search("by fostering confusion and distrust among voters")
        )

    def test_issue_numbers_carry_a_thousands_separator(self):
        """A daily passes ten thousand issues and keeps printing the number."""
        from tasks.document.rules_classifier_core import _ISSUE_METADATA_RE

        for line in ("Issue Number No. 42,812", "No. 42,812", "Nº 42", "Edição 717"):
            with self.subTest(line=line):
                self.assertTrue(_ISSUE_METADATA_RE.search(line), line)
        self.assertIsNone(_ISSUE_METADATA_RE.search("see page 42 for more"))

    def test_ocr_of_newsprint_drops_inter_word_spaces(self):
        """``NOVEMBER6,2020`` is a correct date read by an imperfect scanner."""
        from tasks.document.rules_classifier_core import _PUBLICATION_DATE_RE

        for line in (
            "INTERNATIONALEDITION |FRIDAY,NOVEMBER6,2020",
            "FRIDAY, NOVEMBER 6, 2020",
            "24 de janeiro de 2026",
        ):
            with self.subTest(line=line):
                self.assertTrue(_PUBLICATION_DATE_RE.search(line), line)
        self.assertIsNone(_PUBLICATION_DATE_RE.search("November 2020 was the month"))

    def test_a_masthead_needs_more_than_a_dominant_title(self):
        """The nameplate alone is a cover; the metadata beside it is the identity."""
        from tasks.document.rules_classifier_core import _detect_masthead

        nameplate = region("THE REGIONAL DAILY", "Title", [80, 60, 1100, 230])
        self.assertFalse(_detect_masthead([nameplate], 1240, 1750))
        with_price_only = [nameplate, region("$1.00", "Text", [80, 250, 300, 290])]
        self.assertFalse(_detect_masthead(with_price_only, 1240, 1750))
        with_identity = [
            nameplate,
            region("Issue Number No. 42,812", "Text", [80, 250, 700, 290]),
        ]
        self.assertTrue(_detect_masthead(with_identity, 1240, 1750))


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
