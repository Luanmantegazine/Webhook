"""v11: the weakest reading of "this publication reports", and its limits.

The case is a scanned broadsheet front page. Dense newsprint is where OCR
fails hardest: the small-caps ``BY`` above a name is read into the name or
lost, names fragment, the edition line and the date corrupt, and the quotation
marks ``attribution_quotes`` counts come back as apostrophes. What survives is
the body — reporting verbs in a page-length text.

``issue_reporting_language`` is deliberately weak evidence, so it is admitted
by exactly one path, ``newspaper_issue_masthead``, where five independent
clauses are already satisfied. The tests below are in three parts: the real
document, the rule's numeric boundaries, and the documents the widened clause
must still refuse.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks.document.rules_classifier_core import (  # noqa: E402
    FAMILY_GATES,
    DEFAULT_WEIGHTS,
    classify_with_rules,
    evaluate_rules,
    extract_classification_features,
)

FIXTURE = ROOT / "tests" / "fixtures" / "nyt_front_page_features.json"

PAGE_WIDTH, PAGE_HEIGHT = 1240, 1750
COLUMNS = [(80, 460), (470, 850), (860, 1160)]


def region(text, class_name, bbox):
    return {"class_name": class_name, "text": text, "bbox": bbox}


def classify(document, pages=None):
    page_count = pages or document["total_pages"]
    features = extract_classification_features(
        document, page_sizes=[[PAGE_WIDTH, PAGE_HEIGHT]] * page_count
    )
    return features, classify_with_rules(features, include_indicators=True)


def news_gate(result):
    return result["evidence"]["family_gates"]["news_article"]


class RealFrontPageRegressionTests(unittest.TestCase):
    """A. The reported false negative, on the record the workflow produced.

    The fixture is the feature record reported by the production run, not a
    document written here: `provenance` in the JSON says which keys were
    reported, which were declared to make the record loadable, and why the page
    text is not carried. Running v9's classifier on it reproduces the reported
    result exactly — ``other`` / ``fallback`` with `news_article` at 1.0588 —
    which is what makes it a regression rather than an illustration.
    """

    def setUp(self):
        payload = json.loads(FIXTURE.read_text("utf-8"))
        self.provenance = payload["provenance"]
        self.features = dict(payload["features"])
        self.result = classify_with_rules(dict(self.features), include_indicators=True)

    def test_the_fixture_carries_the_reported_ocr_failure(self):
        """D. The signature of the failure: the strong markers are all absent."""
        for key, value in self.provenance["reproduces"].items():
            self.assertEqual(self.features[key], value, key)
        self.assertEqual(self.features["byline_count"], 0)
        self.assertEqual(self.features["publication_date_count"], 0)
        self.assertEqual(self.features["issue_metadata_count"], 0)
        self.assertEqual(self.features["reporting_verb_count"], 2)

    def test_it_is_a_newspaper_issue(self):
        self.assertEqual(self.result["document_family"], "news_article")
        self.assertEqual(self.result["document_subtype"], "newspaper_issue")
        self.assertEqual(self.result["decision"], "classified")

    def test_the_masthead_path_is_what_opens(self):
        gate = news_gate(self.result)
        self.assertEqual(gate["status"], "satisfied")
        self.assertIn("newspaper_issue_masthead", gate["satisfied_paths"])
        self.assertEqual(gate["reason"], "path_newspaper_issue_masthead")

    def test_the_new_rule_is_what_carries_the_reporting_clause(self):
        self.assertTrue(self.result["rule_indicators"]["news_article.issue_reporting_language"])
        for absent in ("byline", "wire_service", "attribution_quotes", "multilingual_reporting"):
            self.assertFalse(
                self.result["rule_indicators"][f"news_article.{absent}"], absent
            )


class RuleBoundaryTests(unittest.TestCase):
    """B. The rule's limits, read straight off the indicator vector."""

    def indicators(self, **features):
        record = {"total_pages": 1, "measured_page_ratio": 1.0, "geometry_page_ratio": 1.0}
        record.update(features)
        return evaluate_rules(record)

    def test_one_verb_in_a_long_page_does_not_fire(self):
        fired = self.indicators(reporting_verb_count=1, word_count=800)
        self.assertFalse(fired["news_article.issue_reporting_language"])

    def test_two_verbs_just_below_the_length_floor_do_not_fire(self):
        fired = self.indicators(reporting_verb_count=2, word_count=499)
        self.assertFalse(fired["news_article.issue_reporting_language"])

    def test_two_verbs_at_the_length_floor_fire(self):
        fired = self.indicators(reporting_verb_count=2, word_count=500)
        self.assertTrue(fired["news_article.issue_reporting_language"])

    def test_the_length_is_the_longer_of_the_two_readings(self):
        """The region stream can hold a fraction of the words the OCR read."""
        fired = self.indicators(
            reporting_verb_count=2, word_count=120, word_token_count=500
        )
        self.assertTrue(fired["news_article.issue_reporting_language"])

    def test_four_verbs_still_fire_multilingual_reporting(self):
        fired = self.indicators(reporting_verb_count=4, word_count=800)
        self.assertTrue(fired["news_article.multilingual_reporting"])
        self.assertTrue(fired["news_article.issue_reporting_language"])

    def test_multilingual_reporting_keeps_its_own_floor(self):
        """Not relaxed: it is still used by paths that have less around them."""
        fired = self.indicators(reporting_verb_count=3, word_count=800)
        self.assertFalse(fired["news_article.multilingual_reporting"])

    def test_the_two_readings_share_one_group_weight(self):
        self.assertEqual(
            DEFAULT_WEIGHTS["news_article.issue_reporting_language"],
            DEFAULT_WEIGHTS["news_article.multilingual_reporting"],
        )

    def test_only_one_contribution_enters_the_score(self):
        """Both fired scores exactly what one fired scores."""
        base = dict(json.loads(FIXTURE.read_text("utf-8"))["features"])
        one = classify_with_rules(dict(base, reporting_verb_count=2))
        both = classify_with_rules(dict(base, reporting_verb_count=4), include_indicators=True)
        self.assertTrue(both["rule_indicators"]["news_article.multilingual_reporting"])
        self.assertTrue(both["rule_indicators"]["news_article.issue_reporting_language"])
        self.assertEqual(one["score"], both["score"])
        self.assertEqual(one["decision_mass"], both["decision_mass"])
        self.assertEqual(one["available_mass"], both["available_mass"])

    def test_the_alternative_is_admitted_by_one_path_only(self):
        """C. The gate, not the rule, is what keeps this safe."""
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        admitted = {
            path.name
            for path in news.paths
            for pool in path.any_of
            if "issue_reporting_language" in pool
        }
        self.assertEqual(admitted, {"newspaper_issue_masthead"})
        for path in news.paths:
            self.assertNotIn("issue_reporting_language", path.all_of)


def _filler(seed, repeats=8):
    words = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor"
    ).split()
    return " ".join(f"{word}{seed}" for word in words * repeats)


REPORTING = (
    "The director said the programme would continue. The manager stated that the "
    "schedule was revised. "
)


class GateIsolationTests(unittest.TestCase):
    """C. Documents with the language but not the shape of an issue."""

    def assertNotNews(self, result):
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertIsNone(result["document_subtype"])
        self.assertNotIn(
            "newspaper_issue_masthead",
            news_gate(result).get("satisfied_paths") or (),
        )

    def test_a_long_reporting_text_without_a_masthead_is_not_a_newspaper(self):
        document = {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region(REPORTING + _filler(1, 14), "Text", [80, 80, 1160, 900]),
                        region(_filler(2, 14), "Text", [80, 920, 1160, 1600]),
                    ],
                }
            ],
            "full_text": "",
        }
        features, result = classify(document)
        self.assertGreaterEqual(features["reporting_verb_count"], 2)
        self.assertGreaterEqual(features["word_count"], 500)
        self.assertEqual(features["publication_masthead_count"], 0)
        self.assertNotNews(result)

    def test_a_long_report_with_two_reporting_verbs_is_not_a_newspaper(self):
        pages = []
        for number in range(1, 4):
            pages.append(
                {
                    "page_number": number,
                    "regions": [
                        region(
                            f"Technical Report TR-2026-0{number}", "Title", [80, 60, 1160, 160]
                        ),
                        region("1. Methodology", "Section-header", [80, 200, 1160, 250]),
                        region(
                            REPORTING
                            + "The specification, the apparatus and the procedure are "
                            "described below; the analysis and the conclusion follow. "
                            + _filler(number),
                            "Text",
                            [80, 270, 1160, 1400],
                        ),
                    ],
                }
            )
        features, result = classify({"total_pages": 3, "pages": pages, "full_text": ""})
        self.assertGreaterEqual(features["reporting_verb_count"], 2)
        self.assertGreaterEqual(features["word_count"], 500)
        self.assertNotNews(result)

    def test_a_product_catalogue_is_still_not_a_newspaper(self):
        pages = []
        counter = 0
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
            for left, right in COLUMNS:
                counter += 1
                regions.append(
                    region(f"Product line {counter}", "Section-header", [left, top, right, top + 60])
                )
                regions.append(
                    region(
                        f"Product {counter}: available in three finishes with a two "
                        f"year warranty and free delivery. " + _filler(counter, 2),
                        "Text",
                        [left, top + 70, right, top + 900],
                    )
                )
            pages.append({"page_number": number, "regions": regions})
        features, result = classify({"total_pages": 4, "pages": pages, "full_text": ""})
        self.assertGreaterEqual(features["word_count"], 500)
        self.assertEqual(features["reporting_verb_count"], 0)
        self.assertNotNews(result)

    @unittest.expectedFailure
    def test_known_cost_a_catalogue_that_quotes_its_own_managers(self):
        """Documents a precision cost the v11 relaxation introduces.

        Two reporting verbs in 800 words is weak evidence, and a catalogue with
        a large nameplate, a domain, a column grid and one block of quoted
        sales copy satisfies every clause of `newspaper_issue_masthead`. This
        test asserts the behaviour we *want* and is expected to fail until that
        boundary is decided; it is recorded rather than left undiscovered, and
        it will report an unexpected success the moment a guard closes it.
        """
        pages = []
        counter = 0
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
            for left, right in COLUMNS:
                counter += 1
                copy = REPORTING if counter == 1 else ""
                regions.append(
                    region(f"Product line {counter}", "Section-header", [left, top, right, top + 60])
                )
                regions.append(
                    region(
                        copy
                        + f"Product {counter}: available in three finishes with a two "
                        f"year warranty. " + _filler(counter, 2),
                        "Text",
                        [left, top + 70, right, top + 900],
                    )
                )
            pages.append({"page_number": number, "regions": regions})
        _features, result = classify({"total_pages": 4, "pages": pages, "full_text": ""})
        self.assertNotNews(result)

    def test_a_press_release_is_still_blocked(self):
        document = {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region("FOR IMMEDIATE RELEASE", "Title", [80, 40, 1160, 160]),
                        region(
                            REPORTING
                            + "The company announced its results for the quarter. "
                            + _filler(3),
                            "Text",
                            [80, 200, 1160, 1400],
                        ),
                    ],
                }
            ],
            "full_text": "",
        }
        _features, result = classify(document)
        self.assertNotNews(result)
        self.assertEqual(news_gate(result)["status"], "blocked")

    def test_an_event_announcement_is_not_a_newspaper(self):
        from test_event_announcement import call_for_papers

        _features, result = classify(call_for_papers())
        self.assertEqual(result["document_family"], "presentation_marketing")
        self.assertNotNews(result)

    def test_an_academic_article_that_reports_is_not_a_newspaper(self):
        """Two columns, section headings, a running title — and the new rule fires."""
        pages = []
        for number in range(1, 4):
            # "reported" and "according to": exactly two reporting verbs, the
            # floor of the new rule, and short of multilingual_reporting's four.
            lead = (
                "It was reported that the effect persists, and according to [4] the "
                "measurement is stable. "
                if number == 1
                else ""
            )
            pages.append(
                {
                    "page_number": number,
                    "regions": [
                        region(
                            "Deep Structural Analysis of Porous Media",
                            "Title",
                            [80, 40, 1160, 200],
                        ),
                        region("Abstract", "Section-header", [80, 230, 600, 280]),
                        region(
                            lead
                            + "References [1] [2] show the results are consistent "
                            "with the model. " + _filler(number),
                            "Text",
                            [80, 300, 600, 1400],
                        ),
                        region("Methodology", "Section-header", [640, 230, 1160, 280]),
                        region(
                            "The experiment was repeated and the conclusions hold. "
                            + _filler(number + 10),
                            "Text",
                            [640, 300, 1160, 1400],
                        ),
                    ],
                }
            )
        features, result = classify({"total_pages": 3, "pages": pages, "full_text": ""})
        self.assertEqual(features["reporting_verb_count"], 2)
        self.assertGreaterEqual(features["word_count"], 500)
        # The rule fires; the gate is what refuses the document.
        self.assertTrue(result["rule_indicators"]["news_article.issue_reporting_language"])
        self.assertNotNews(result)
        self.assertIn("research_publication_evidence", news_gate(result)["blocked_by"])


if __name__ == "__main__":
    unittest.main()
