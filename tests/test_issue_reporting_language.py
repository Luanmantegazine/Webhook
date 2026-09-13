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

    def test_the_fixture_declares_what_kind_of_record_it_is(self):
        """Not a capture: some fields were declared and no page text was kept."""
        self.assertEqual(
            self.provenance["fixture_kind"], "reconstructed_from_production_metrics"
        )
        self.assertIn("replace_when", self.provenance)
        self.assertNotIn("classification_text", self.features)

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

    def test_the_recovery_path_is_what_opens(self):
        """v12: the strong path is not satisfied — the recovery path is."""
        gate = news_gate(self.result)
        self.assertEqual(gate["status"], "satisfied")
        self.assertNotIn("newspaper_issue_masthead", gate["satisfied_paths"])
        self.assertIn("newspaper_issue_masthead_ocr_recovery", gate["satisfied_paths"])
        self.assertEqual(gate["reason"], "path_newspaper_issue_masthead_ocr_recovery")

    def test_the_recovery_path_rests_on_measured_newsprint_geometry(self):
        indicators = self.result["rule_indicators"]
        self.assertTrue(indicators["news_article.issue_reporting_language"])
        self.assertTrue(indicators["news_article.newspaper_column_geometry"])

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

    def test_the_alternative_belongs_to_one_path_only(self):
        """C. The gate, not the rule, is what keeps this safe."""
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        required_by = {
            path.name for path in news.paths if "issue_reporting_language" in path.all_of
        }
        offered_by = {
            path.name
            for path in news.paths
            for pool in path.any_of
            if "issue_reporting_language" in pool
        }
        self.assertEqual(required_by, {"newspaper_issue_masthead_ocr_recovery"})
        self.assertEqual(offered_by, set())

    def test_the_recovery_path_cannot_take_columns_for_geometry(self):
        """``multi_column_publication`` is not an alternative there — a catalogue has one."""
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        recovery = next(
            path
            for path in news.paths
            if path.name == "newspaper_issue_masthead_ocr_recovery"
        )
        self.assertIn("newspaper_column_geometry", recovery.all_of)
        for pool in recovery.any_of:
            self.assertNotIn("multi_column_publication", pool)
            self.assertNotIn("multi_column_body", pool)

    def test_both_masthead_paths_report_the_same_subtype(self):
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        subtypes = dict(news.path_subtypes)
        self.assertEqual(subtypes["newspaper_issue_masthead"], "newspaper_issue")
        self.assertEqual(
            subtypes["newspaper_issue_masthead_ocr_recovery"], "newspaper_issue"
        )

    def test_the_paths_are_ordered_strong_recovery_running_header_single(self):
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        self.assertEqual(
            [path.name for path in news.paths],
            [
                "newspaper_issue_masthead",
                "newspaper_issue_masthead_ocr_recovery",
                "newspaper_issue_running_header",
                "byline_with_reporting",
                "wire_and_dateline_with_reporting",
            ],
        )


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

    def test_a_catalogue_that_quotes_its_own_managers_is_not_a_newspaper(self):
        """The v11 cost, closed in v12 by the geometry clause.

        This document has everything the masthead path's structural clauses
        ask for — a nameplate, a domain, four pages of headed blocks in a
        column grid — and two sentences of quoted sales copy are enough to fire
        ``issue_reporting_language``. What it does not have is newsprint
        column geometry, and that is now what the recovery path requires.
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
        indicators = result["rule_indicators"]

        # Non-vacuous: everything except the geometry clause is in place.
        self.assertTrue(indicators["news_article.issue_reporting_language"])
        self.assertTrue(indicators["news_article.publication_masthead"])
        self.assertTrue(indicators["news_article.publication_url"])
        self.assertTrue(indicators["news_article.multiple_headlines"])
        self.assertTrue(indicators["news_article.multiple_article_clusters"])
        self.assertTrue(indicators["news_article.multi_column_publication"])
        self.assertFalse(indicators["news_article.newspaper_column_geometry"])

        evaluation = news_gate(result)["path_evaluations"][
            "newspaper_issue_masthead_ocr_recovery"
        ]
        self.assertFalse(evaluation["satisfied"])
        self.assertIn(
            "requires newspaper_column_geometry", evaluation["unmet_requirements"]
        )
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


class RunningHeaderPathTests(unittest.TestCase):
    """v12: a repeated academic title is not a publication identity."""

    def academic_journal_article(self):
        pages = []
        for number in range(1, 4):
            pages.append(
                {
                    "page_number": number,
                    "regions": [
                        # The running title a journal prints on every page.
                        region(
                            "Deep Structural Analysis of Porous Media",
                            "Title",
                            [80, 40, 1160, 200],
                        ),
                        region("Abstract", "Section-header", [80, 230, 600, 280]),
                        region(
                            "It was reported that the effect persists, and according to "
                            "[4] the measurement is stable. References [1] [2] show the "
                            "results are consistent. " + _filler(number),
                            "Text",
                            [80, 300, 600, 1400],
                        ),
                        region("Methodology", "Section-header", [640, 230, 1160, 280]),
                        region(
                            "The authors said the apparatus was recalibrated and stated "
                            "that the conclusions hold. doi.org/10.1000/example "
                            + _filler(number + 10),
                            "Text",
                            [640, 300, 1160, 1400],
                        ),
                    ],
                }
            )
        return {"total_pages": 3, "pages": pages, "full_text": ""}

    def test_an_academic_running_header_is_not_a_newspaper_issue(self):
        features, result = classify(self.academic_journal_article())
        gate = news_gate(result)
        evaluation = gate["path_evaluations"]["newspaper_issue_running_header"]

        # The structural requirements are met; the guard is what refuses it.
        self.assertGreaterEqual(features["reporting_verb_count"], 4)
        self.assertTrue(result["rule_indicators"]["news_article.repeated_publication_header"])
        self.assertTrue(evaluation["requirements_met"])
        self.assertIn("research_publication_evidence", evaluation["blocked_by"])
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertIsNone(result["document_subtype"])

    def test_the_guard_is_on_the_path_and_not_on_the_masthead_paths(self):
        news = next(item for item in FAMILY_GATES if item.family == "news_article")
        by_name = {path.name: path for path in news.paths}
        self.assertIn(
            "research_publication_evidence",
            by_name["newspaper_issue_running_header"].blockers,
        )
        for name in ("newspaper_issue_masthead", "newspaper_issue_masthead_ocr_recovery"):
            self.assertNotIn("research_publication_evidence", by_name[name].blockers)

    def test_a_newspaper_without_a_masthead_still_classifies(self):
        """Positive control: the path still does the job it was written for."""
        from test_news_publication import classify as classify_news, newspaper_issue

        _features, result = classify_news(
            newspaper_issue(pages=4, language="en", masthead=False)
        )
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")
        self.assertEqual(
            news_gate(result)["reason"], "path_newspaper_issue_running_header"
        )


class GeometryControlTests(unittest.TestCase):
    """v12: the recovery path turns on measured geometry and nothing else."""

    def record(self, **overrides):
        base = dict(json.loads(FIXTURE.read_text("utf-8"))["features"])
        base.update(overrides)
        return base

    def paths(self, record):
        result = classify_with_rules(dict(record), include_indicators=True)
        return result, news_gate(result).get("satisfied_paths") or []

    def test_without_newsprint_geometry_the_recovery_path_is_refused(self):
        result, satisfied = self.paths(self.record(newspaper_column_geometry_ratio=0.0))
        self.assertFalse(result["rule_indicators"]["news_article.newspaper_column_geometry"])
        self.assertNotIn("newspaper_issue_masthead_ocr_recovery", satisfied)
        self.assertNotEqual(result["document_family"], "news_article")

    def test_with_newsprint_geometry_the_recovery_path_is_accepted(self):
        """The same vector, one field changed."""
        result, satisfied = self.paths(self.record(newspaper_column_geometry_ratio=0.50))
        self.assertTrue(result["rule_indicators"]["news_article.newspaper_column_geometry"])
        self.assertIn("newspaper_issue_masthead_ocr_recovery", satisfied)
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")

    def test_the_two_vectors_differ_in_exactly_one_field(self):
        without = self.record(newspaper_column_geometry_ratio=0.0)
        with_geometry = self.record(newspaper_column_geometry_ratio=0.50)
        differing = {
            key
            for key in set(without) | set(with_geometry)
            if without.get(key) != with_geometry.get(key)
        }
        self.assertEqual(differing, {"newspaper_column_geometry_ratio"})

    def test_the_strong_path_needs_no_geometry_and_no_weak_reading(self):
        """4. A byline still opens the original path, with the new rule false."""
        record = self.record(
            reporting_verb_count=0,
            newspaper_column_geometry_ratio=0.0,
            byline_count=1,
            issue_metadata_count=1,
        )
        result, satisfied = self.paths(record)
        indicators = result["rule_indicators"]
        self.assertFalse(indicators["news_article.issue_reporting_language"])
        self.assertFalse(indicators["news_article.newspaper_column_geometry"])
        self.assertTrue(indicators["news_article.byline"])
        self.assertIn("newspaper_issue_masthead", satisfied)
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")


if __name__ == "__main__":
    unittest.main()
