"""Institutional event announcements, and the documents they must not swallow.

A call for papers is a promotional document that looks nothing like a slide
deck: it is dense, typeset in columns, and its only pictures are society logos
too small to count as illustration. Under v9 the presentation family had no
path that described it, so a real one fell to ``other``/fallback while the
financial family scored 0.3878 on it — the word *ledger* inside "distributed
ledger technology", and a run of submission deadlines read as a monthly series.

Nothing in this module keys on a conference name, an organiser or a venue: the
fixtures are built from the structure an announcement has (a call, an identified
event, its logistics, its shape) and the negatives from the structure the
documents that merely *mention* an event have instead.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks.document.rules_classifier_core import (  # noqa: E402
    FAMILY_GATES,
    classify_with_rules,
    extract_classification_features,
)

PAGE_WIDTH, PAGE_HEIGHT = 1240, 1750


def region(text, class_name, bbox):
    return {"class_name": class_name, "text": text, "bbox": bbox}


def document(regions, pages=1):
    page = {"page_number": 1, "regions": regions}
    return {
        "total_pages": pages,
        "pages": [page] + [
            {"page_number": number, "regions": regions}
            for number in range(2, pages + 1)
        ],
        "full_text": "\n".join(item["text"] for item in regions),
    }


def classify(doc, pages=None):
    page_count = pages or doc["total_pages"]
    features = extract_classification_features(
        doc, page_sizes=[[PAGE_WIDTH, PAGE_HEIGHT]] * page_count
    )
    return features, classify_with_rules(features, include_indicators=True)


def gate(result, family):
    return result["evidence"]["family_gates"][family]


def fired(result, family):
    entry = result["evidence"]["rules_by_family"].get(family) or []
    return {item["rule"].split(".", 1)[1] for item in entry}


def call_for_papers():
    """The real document, rebuilt from its layout.

    A one-page call for papers: a row of society logos, the conference title
    over a date/venue line, a call, a topic list, an important-dates list,
    submission instructions, the organising committee, and registration.
    """
    return document(
        [
            # Society and sponsor logos. Each covers well under 2% of the page,
            # which is exactly why ``relevant_picture_count`` sees none of them
            # and ``event_announcement_layout`` counts them itself.
            region("", "Picture", [80, 40, 220, 110]),
            region("", "Picture", [260, 40, 400, 110]),
            region("", "Picture", [440, 40, 580, 110]),
            region(
                "2026 International Conference on Blockchain and Cryptocurrency",
                "Title",
                [80, 140, 1160, 230],
            ),
            region(
                "May 18-22, 2026  |  Lisbon, Portugal  |  https://icbc2026.example.org",
                "Text",
                [80, 240, 1160, 285],
            ),
            region("CALL FOR PAPERS", "Section-header", [80, 310, 1160, 370]),
            region(
                "The organizing committee invites authors to submit original, "
                "unpublished research contributions to the conference. Accepted papers "
                "will be presented at the conference and published in the conference "
                "proceedings. Papers are solicited on all aspects of distributed ledger "
                "technology and its applications.",
                "Text",
                [80, 380, 1160, 500],
            ),
            region("Scope and Topics", "Section-header", [80, 520, 600, 565]),
            region(
                "- Consensus protocols and scalability\n"
                "- Smart contract languages and verification\n"
                "- Distributed ledger technology for supply chains\n"
                "- Privacy, anonymity and regulation\n"
                "- Tokenomics and market microstructure\n"
                "- Interoperability across ledgers\n"
                "- Security analysis and audit of deployed systems\n"
                "- Energy cost of proof-of-work networks",
                "List-item",
                [80, 575, 600, 900],
            ),
            region("Important Dates", "Section-header", [640, 520, 1160, 565]),
            region(
                "- Paper submission deadline: January 12, 2026\n"
                "- Author notification: March 2, 2026\n"
                "- Camera-ready version: April 6, 2026\n"
                "- Registration deadline: April 20, 2026\n"
                "- Conference dates: May 18-22, 2026",
                "List-item",
                [640, 575, 1160, 800],
            ),
            region("Paper Submission", "Section-header", [80, 930, 600, 975]),
            region(
                "Manuscripts must be submitted through EasyChair in two-column format. "
                "The page limit is 8 pages including references. Submissions are "
                "reviewed double-blind; formatting guidelines and the template are "
                "available on the conference website. The camera-ready version must "
                "follow the same template.",
                "Text",
                [80, 985, 600, 1180],
            ),
            region("Organizing Committee", "Section-header", [640, 930, 1160, 975]),
            region(
                "General Chairs: A. Moreau, K. Sato\n"
                "Technical Program Chairs: L. Fernandes, R. Okafor\n"
                "Publicity Chair: M. Haddad\n"
                "Finance Chair: S. Petrova\n"
                "Local Organizing Committee: University of Lisbon",
                "Text",
                [640, 985, 1160, 1180],
            ),
            region("Registration", "Section-header", [80, 1210, 1160, 1255]),
            region(
                "Registration opens February 2, 2026. Early registration for students "
                "is available and a limited travel budget supports student authors; "
                "receipts are issued after payment. The venue is the Lisbon Congress "
                "Centre; travel and accommodation information is published on the "
                "conference website.",
                "Text",
                [80, 1265, 1160, 1400],
            ),
            region(
                "Sponsored by the Communications Society and the Blockchain Technical "
                "Community. Contact: icbc2026@example.org",
                "Text",
                [80, 1430, 1160, 1500],
            ),
        ]
    )


class CallForPapersRegressionTests(unittest.TestCase):
    """The reported failure, asserted end to end."""

    def setUp(self):
        self.features, self.result = classify(call_for_papers())

    def test_it_is_classified_as_presentation_marketing(self):
        self.assertEqual(self.result["document_family"], "presentation_marketing")
        self.assertEqual(self.result["decision"], "classified")

    def test_the_event_announcement_path_is_the_one_that_opens(self):
        presentation = gate(self.result, "presentation_marketing")
        self.assertEqual(presentation["status"], "satisfied")
        self.assertIn("institutional_event_announcement", presentation["satisfied_paths"])

    def test_news_article_stays_blocked(self):
        news = gate(self.result, "news_article")
        self.assertNotEqual(news["status"], "satisfied")
        self.assertEqual(self.result["candidate_scores"]["news_article"], 0.0)

    def test_ledger_technology_is_not_accounting_evidence(self):
        self.assertNotIn("accounting_vocabulary", fired(self.result, "financial_document"))
        self.assertEqual(self.features["accounting_distinct_term_count"], 0)

    def test_a_run_of_deadlines_is_not_a_monthly_series(self):
        self.assertNotIn("monthly_series", fired(self.result, "financial_document"))
        self.assertGreaterEqual(self.features["deadline_sequence_count"], 3)

    def test_the_logos_are_below_the_relevant_picture_floor(self):
        # The point of a separate count: the illustration rules see nothing.
        self.assertEqual(self.features["relevant_picture_count"], 0)
        self.assertGreaterEqual(self.features["logo_picture_count"], 1)

    def test_marketing_copy_is_not_what_opened_the_gate(self):
        self.assertNotIn("marketing_copy", fired(self.result, "presentation_marketing"))


class EventPathConfigurationTests(unittest.TestCase):
    def test_marketing_copy_appears_in_no_path(self):
        presentation = next(
            item for item in FAMILY_GATES if item.family == "presentation_marketing"
        )
        for path in presentation.paths:
            self.assertNotIn("marketing_copy", path.all_of)
            for group in path.any_of:
                self.assertNotIn("marketing_copy", group)

    def test_the_announcement_path_requires_a_call_and_an_identity(self):
        presentation = next(
            item for item in FAMILY_GATES if item.family == "presentation_marketing"
        )
        path = next(
            item
            for item in presentation.paths
            if item.name == "institutional_event_announcement"
        )
        self.assertEqual(
            set(path.all_of), {"event_call_to_action", "event_identity"}
        )
        self.assertEqual(len(path.any_of), 2)


class EventAnnouncementNegativeTests(unittest.TestCase):
    """Documents that talk about an event without announcing one."""

    def assertNotAnAnnouncement(self, result):
        presentation = gate(result, "presentation_marketing")
        self.assertNotIn(
            "institutional_event_announcement",
            presentation.get("satisfied_paths") or (),
        )
        self.assertNotEqual(result["document_family"], "presentation_marketing")

    def test_a_paper_published_at_a_conference_is_not_an_announcement(self):
        _features, result = classify(
            document(
                [
                    region(
                        "Byzantine Agreement under Partial Synchrony",
                        "Title",
                        [80, 80, 1160, 180],
                    ),
                    region(
                        "Proceedings of the 2026 International Conference on "
                        "Blockchain and Cryptocurrency, Lisbon, Portugal",
                        "Text",
                        [80, 190, 1160, 240],
                    ),
                    region("Abstract", "Section-header", [80, 270, 600, 310]),
                    region(
                        "We present a consensus protocol that tolerates partial "
                        "synchrony. Our evaluation on a distributed ledger testbed "
                        "shows a throughput improvement over prior work [3], [7].",
                        "Text",
                        [80, 320, 600, 900],
                    ),
                    region("1. Introduction", "Section-header", [640, 270, 1160, 310]),
                    region(
                        "Consensus in the presence of faults has been studied since "
                        "[1]. Recent systems [4], [8], [11] adopt leader rotation; we "
                        "extend that line of work and report the methodology, the "
                        "experimental results and the conclusion below.",
                        "Text",
                        [640, 320, 1160, 900],
                    ),
                    region("References", "Section-header", [80, 930, 600, 970]),
                    region(
                        "[1] L. Lamport, Communications of the ACM, 1978.\n"
                        "[2] M. Fischer et al., Journal of the ACM, 1985.\n"
                        "[3] D. Dolev, SIAM Journal on Computing, 1983.",
                        "Text",
                        [80, 980, 600, 1400],
                    ),
                ]
            )
        )
        self.assertNotAnAnnouncement(result)

    def test_a_proceedings_volume_is_not_an_announcement(self):
        _features, result = classify(
            document(
                [
                    region(
                        "Proceedings of the 2026 International Conference on "
                        "Blockchain and Cryptocurrency",
                        "Title",
                        [80, 80, 1160, 200],
                    ),
                    region("Table of Contents", "Section-header", [80, 230, 1160, 280]),
                    region(
                        "Byzantine Agreement under Partial Synchrony .......... 1\n"
                        "Smart Contract Verification at Scale ................ 17\n"
                        "Privacy-Preserving Settlement ....................... 33\n"
                        "Interoperability across Ledgers ..................... 49\n"
                        "Energy Cost of Proof-of-Work Networks ............... 65",
                        "List-item",
                        [80, 300, 1160, 800],
                    ),
                    region(
                        "This volume collects the papers presented at the conference. "
                        "Each contribution was reviewed by the technical program "
                        "committee and revised by its authors before publication.",
                        "Text",
                        [80, 830, 1160, 1100],
                    ),
                ],
                pages=6,
            )
        )
        self.assertNotAnAnnouncement(result)

    def test_a_technical_report_mentioning_a_conference_is_not_an_announcement(self):
        _features, result = classify(
            document(
                [
                    region(
                        "Technical Report TR-2026-04: Throughput of Permissioned "
                        "Ledgers",
                        "Title",
                        [80, 80, 1160, 200],
                    ),
                    region("1. Objective", "Section-header", [80, 230, 1160, 275]),
                    region(
                        "This report documents the methodology and the experimental "
                        "results of a benchmark carried out for the platform team. An "
                        "earlier version of this material was presented at the 2026 "
                        "International Conference on Blockchain and Cryptocurrency.",
                        "Text",
                        [80, 285, 1160, 560],
                    ),
                    region("2. Methodology", "Section-header", [80, 590, 1160, 635]),
                    region(
                        "The specification of the test harness, the apparatus and the "
                        "procedure are described below. Measurements were repeated "
                        "across three deployments.",
                        "Text",
                        [80, 645, 1160, 950],
                    ),
                    region("3. Results", "Section-header", [80, 980, 1160, 1025]),
                    region(
                        "Throughput scaled linearly to eight nodes. The analysis and "
                        "the conclusion follow in section 4; appendix A lists the "
                        "configuration.",
                        "Text",
                        [80, 1035, 1160, 1350],
                    ),
                ],
                pages=3,
            )
        )
        self.assertNotAnAnnouncement(result)

    def test_a_meeting_agenda_is_not_an_announcement(self):
        _features, result = classify(
            document(
                [
                    region("Steering Group Meeting - Agenda", "Title", [80, 80, 1160, 180]),
                    region(
                        "Tuesday, March 3, 2026, 10:00-12:00, Room 4B",
                        "Text",
                        [80, 190, 1160, 240],
                    ),
                    region(
                        "1. Approval of the previous minutes\n"
                        "2. Status of the platform migration\n"
                        "3. Staffing for the next quarter\n"
                        "4. Any other business",
                        "List-item",
                        [80, 270, 1160, 700],
                    ),
                    region(
                        "Please send additions to the agenda to the chair before the "
                        "meeting. Minutes will be circulated afterwards.",
                        "Text",
                        [80, 730, 1160, 900],
                    ),
                ]
            )
        )
        self.assertNotAnAnnouncement(result)

    def test_a_financial_calendar_is_not_an_announcement(self):
        features, result = classify(
            document(
                [
                    region("Financial Calendar 2026", "Title", [80, 80, 1160, 180]),
                    region(
                        "January - Q4 revenue reported\n"
                        "February - Balance sheet audit begins\n"
                        "March - Annual report and income statement published\n"
                        "April - Budget approved; expenses reforecast\n"
                        "May - Q1 cash flow statement\n"
                        "June - Tax filing; amount due settled",
                        "List-item",
                        [80, 210, 1160, 900],
                    ),
                    region(
                        "Figures are stated in thousands. The subtotal for each "
                        "quarter reconciles to the trial balance.",
                        "Text",
                        [80, 930, 1160, 1100],
                    ),
                ]
            )
        )
        self.assertNotAnAnnouncement(result)
        # The financial rules must still read a genuine financial period.
        self.assertIn("monthly_series", fired(result, "financial_document"))
        self.assertGreaterEqual(features["financial_period_term_count"], 2)

    def test_a_newspaper_carrying_a_conference_advertisement_stays_a_newspaper(self):
        from test_news_publication import newspaper_issue, region as news_region

        issue = newspaper_issue(pages=4, language="en")
        # A quarter-page advertisement in the bottom corner of the front page.
        issue["pages"][0]["regions"].extend(
            [
                news_region(
                    "CALL FOR PAPERS - 2026 International Conference on Blockchain "
                    "and Cryptocurrency",
                    "Section-header",
                    [80, 1420, 600, 1480],
                ),
                news_region(
                    "Submit your paper by January 12, 2026. Author notification "
                    "March 2, 2026. Registration deadline April 20, 2026. "
                    "Organizing committee: University of Lisbon.",
                    "Text",
                    [80, 1490, 600, 1650],
                ),
            ]
        )
        _features, result = classify(issue)
        self.assertEqual(result["document_family"], "news_article")
        self.assertEqual(result["document_subtype"], "newspaper_issue")
        self.assertEqual(result["decision"], "classified")
        # Non-vacuous: the announcement's own requirements *are* met by the
        # advertisement; it is the reporting around it that holds the path shut.
        presentation = gate(result, "presentation_marketing")
        evaluation = presentation["path_evaluations"]["institutional_event_announcement"]
        self.assertTrue(evaluation["requirements_met"])
        self.assertEqual(evaluation["blocked_by"], ["reported_speech_evidence"])
        self.assertNotIn(
            "institutional_event_announcement",
            presentation.get("satisfied_paths") or (),
        )

    def test_an_invoice_with_a_registration_due_date_is_not_an_announcement(self):
        features, result = classify(
            document(
                [
                    region("INVOICE 2026-0412", "Title", [80, 80, 1160, 180]),
                    region(
                        "Bill to: Department of Computer Science\n"
                        "Invoice date: February 10, 2026\n"
                        "Registration due date: April 20, 2026",
                        "Text",
                        [80, 200, 1160, 360],
                    ),
                    region(
                        "Description            Qty   Unit price   Amount\n"
                        "Conference registration  2      450.00     900.00\n"
                        "Workshop registration    1      150.00     150.00\n"
                        "Subtotal                                  1,050.00\n"
                        "Tax (23%)                                   241.50\n"
                        "Amount due                                1,291.50",
                        "Table",
                        [80, 400, 1160, 900],
                    ),
                    region(
                        "Remittance to the account below. Accounts payable queries to "
                        "the finance office.",
                        "Text",
                        [80, 930, 1160, 1050],
                    ),
                ]
            )
        )
        self.assertNotAnAnnouncement(result)
        self.assertIn("accounting_vocabulary", fired(result, "financial_document"))
        self.assertGreaterEqual(features["accounting_distinct_term_count"], 3)


if __name__ == "__main__":
    unittest.main()
