"""Tests for the family gates.

Each test here encodes a failure the development run actually produced, so
that a later change which reopens one of them fails loudly rather than
quietly costing precision again. The presentation cases in particular are the
v5 diagnosis written down: 41 documents accepted, 6 correct, and every false
accept firing ``visual_layout`` together with ``visual_dominance`` for a score
of 0.6333.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.document.rules_classifier_core import (  # noqa: E402
    BLOCKER_PREDICATES,
    CLASSIFIER_VERSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    DEFAULT_WEIGHTS,
    FAMILY_CONFIDENCE_THRESHOLDS,
    FAMILY_GATES,
    FAMILY_THRESHOLD_HOLDS,
    FAMILY_THRESHOLD_PROVENANCE,
    RULE_FINGERPRINT,
    RULE_IDS,
    RULES,
    SCORED_FAMILIES,
    FAMILIES_NOT_RELEASED_FOR_ROUTING,
    ROUTING_RELEASE_STATUS,
    apply_decision_policy,
    classify_with_rules,
    effective_family_threshold,
    evaluate_family_gates,
    extract_classification_features,
    resolve_family_thresholds,
    score_families,
)

PAGE = [1200, 1600]


def region(text, class_name="Text", bbox=None, index=0):
    return {
        "class_name": class_name,
        "text": text,
        "bbox": bbox or [80, 80 + index * 120, 1120, 180 + index * 120],
    }


def document(regions):
    return {"total_pages": 1, "pages": [{"page_number": 1, "regions": regions}], "full_text": ""}


def classify(regions, page_size=None, **kwargs):
    features = extract_classification_features(
        document(regions), page_sizes=[page_size or PAGE]
    )
    return classify_with_rules(features, include_indicators=True, **kwargs)


def gate_of(result, family):
    return result["evidence"]["family_gates"][family]


GATE_BY_FAMILY = {gate.family: gate for gate in FAMILY_GATES}


def presentation_indicators(*names):
    """Only the named presentation rules fire; nothing else does."""
    return {
        rule_id: rule_id in {f"presentation_marketing.{name}" for name in names}
        for rule_id in RULE_IDS
    }


#: Features with every channel available and enough recognised text that no
#: OCR-sufficiency guard fires, so a probe measures the gate and nothing else.
PROBE_FEATURES = {
    "word_count": 120,
    "alnum_character_count": 600,
    "measured_page_ratio": 1.0,
    "geometry_page_ratio": 1.0,
    "total_pages": 1,
}


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def picture_page_without_primary_evidence():
    """The v5 false-accept shape: two pictures and some ordinary prose."""
    return [
        region("", "Picture", [50, 50, 1150, 700]),
        region("", "Picture", [50, 750, 1150, 1400]),
        region(
            "The committee reviewed the proposal at length and asked for a revised "
            "schedule before the end of the current reporting period.",
            "Text",
            [50, 1420, 1150, 1560],
        ),
    ]


def slide_with_visual_support():
    return [
        region("Agenda", "Title", [50, 40, 600, 110]),
        region("", "Picture", [50, 150, 1150, 800]),
        region("Market outlook", "List-item", [50, 850, 700, 910]),
        region("Product roadmap", "List-item", [50, 930, 700, 990]),
        region("Next quarter targets", "List-item", [50, 1010, 700, 1070]),
    ]


def advertisement_with_visual_support():
    return [
        region("SPECIAL OFFER", "Title", [50, 40, 600, 110]),
        region("", "Picture", [50, 150, 1150, 900]),
        region(
            "Limited time only. Order now and get a money-back guarantee on every "
            "purchase made today.",
            "Text",
            [50, 950, 1150, 1100],
        ),
    ]


def deck_vocabulary_without_corroboration():
    """Deck vocabulary on a page with no visual or structural shape at all."""
    return [
        region("Agenda", "Title", [50, 40, 600, 110]),
        region(
            "The committee met to review the schedule and agreed that the revised "
            "timetable would be circulated to every department before the end of the "
            "month, with comments returned in writing by the following Friday so that "
            "the final version can be approved at the next ordinary sitting.",
            "Text",
            [50, 150, 1150, 700],
        ),
        region(
            "A second paragraph of ordinary running prose, long enough that the page is "
            "narrative rather than a sparse arrangement of short blocks, and wide enough "
            "that no column or centring signal is produced by it either.",
            "Text",
            [50, 720, 1150, 1200],
        ),
    ]


def near_empty_picture_page():
    return [
        region("Agenda", "Title", [50, 40, 600, 110]),
        region("", "Picture", [50, 150, 1150, 1400]),
    ]


#: The five development-set form shapes that paths A and B recovered.
FORM_DOCUMENTS = {
    "questionnaire_with_labels": [
        region("EMPLOYEE QUESTIONNAIRE", "Title", index=0),
        region(
            "Name: John Alpha\nDepartment: Sales\nSupervisor: B. Gamma\nLocation: Plant 4",
            index=1,
        ),
    ],
    "questionnaire_with_blanks": [
        region("Survey", "Title", index=0),
        region("Position held: ______\nYears of service: ______\nUnit: ______", index=1),
    ],
    "checkboxes_with_labels": [
        region("Please complete and return this section to the office.", "Title", index=0),
        region("[ ] Yes  [ ] No  [ ] Not applicable", index=1),
        # Four labels, not three: ``Date`` reads as a correspondence header, so
        # it is deliberately not counted towards the form-field signal.
        region(
            "Name: John Alpha\nUnit: Plant 2\nShift: Nights\nBadge: 4471\nDate: 4 May",
            index=2,
        ),
    ],
    "checkboxes_with_blanks": [
        region("Return the completed section to the personnel office.", "Title", index=0),
        region("[X] Approved   [ ] Denied", index=1),
        region("Signature: ______\nDate: ______\nTitle: ______", index=2),
    ],
    "form_heading_with_two_corroborations": [
        region("APPLICATION FORM", "Title", index=0),
        region("Name: ______\nAddress: ______\nCity: ______", index=1),
        region("Phone: __\nFax: __\nUnit: __\nRoom: __\nCode: __", index=2),
    ],
}

#: Documents that must never be accepted as forms. Every one of them carries
#: form-shaped structure — labelled lines, short regions, a colon grid.
FORM_NEGATIVES = {
    "invoice": [
        region("INVOICE 8812", "Title", index=0),
        region("Bill To: Acme\nDate: ____\nPO Number: ____\nTerms: ____", index=1),
        region("Subtotal $50.00\nTax $5.00\nTotal Due $55.00", index=2),
    ],
    "specification": [
        region("MATERIAL SPECIFICATION", "Title", index=0),
        region("Part Number: ____\nRevision A\nTolerance: ____", index=1),
        region("The assembly shall comply with MIL-STD-810. Drawing No. 55-2", index=2),
    ],
    "resume": [
        region("CURRICULUM VITAE", "Title", index=0),
        region("Name: ____\nAddress: ____\nTelephone: ____", index=1),
        region("Work Experience\nEducation\nSkills\n1990-1994 Analyst", index=2),
    ],
    "budget": [
        region("BUDGET 1987 fiscal year", "Title", index=0),
        region("Salaries: 120,000\nSupplies: 32,400\nTravel: 18,900", index=1),
        region(
            "Subtotal 171,300 Total $171,300 estimate ledger receipts audit expenditures",
            index=2,
        ),
    ],
    "labelled_listing": [
        region(
            "Departmental listing of current assignments and responsible officers.",
            "Title",
            index=0,
        ),
        region(
            "Name: John Alpha\nDepartment: Sales\nSupervisor: B. Gamma\nLocation: Plant 4",
            index=1,
        ),
    ],
}


def news_with_byline_and_attribution():
    return [
        region("City Council Approves Budget", "Title", index=0),
        region(
            "By Jane Roberts\nOfficials said “the measure is final”, and according "
            "to the mayor it takes effect at once. A spokesman said the vote was decisive "
            "and commented further on the outcome.",
            index=1,
        ),
    ]


def news_with_byline_and_columns_only():
    return [
        region("Quarterly Overview", "Title", [80, 40, 560, 120]),
        region("By Jane Roberts", "Text", [80, 140, 560, 200]),
        region(
            "Body text in the left column of this page, running on for long enough that "
            "the column detector and the word-count condition both have something to "
            "work with when they are evaluated.",
            "Text",
            [80, 220, 560, 900],
        ),
        region(
            "More body text in the right column of this page, also running on for long "
            "enough that the two-column detector has a second column to find here.",
            "Text",
            [640, 220, 1120, 900],
        ),
        region(
            "Continued left column text carrying the discussion further along the page.",
            "Text",
            [80, 920, 560, 1400],
        ),
        region(
            "Continued right column text carrying the discussion further along as well.",
            "Text",
            [640, 920, 1120, 1400],
        ),
    ]


def news_with_wire_and_attribution_only():
    return [
        region("Associated Press", "Title", index=0),
        region(
            "Officials said “the plan proceeds” and according to sources it was "
            "commented on widely, said one person close to the matter.",
            index=1,
        ),
    ]


# --------------------------------------------------------------------------
# presentation_marketing
# --------------------------------------------------------------------------


class PresentationGateTests(unittest.TestCase):
    def test_visual_evidence_alone_is_blocked_with_score_zero(self):
        """The v5 false accept: two readings of one picture-heavy page.

        Not "scores a little lower" — refused, at any threshold, with the
        reason naming what was wrong with the evidence.
        """
        result = classify(picture_page_without_primary_evidence())
        gate = gate_of(result, "presentation_marketing")
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["reason"], "visual_evidence_only")
        self.assertEqual(result["candidate_scores"]["presentation_marketing"], 0.0)
        self.assertNotEqual(result["document_family"], "presentation_marketing")

    def test_visual_and_structural_rules_share_one_scoring_group(self):
        """No two of them may sum into the decision mass.

        ``slide_structure`` joined the group in v7: as an independent primary it
        let an invoice and a form reach 0.6562 while the only true positive sat
        at 0.625.
        """
        visual = {
            rule.name
            for rule in RULES
            if rule.family == "presentation_marketing" and rule.group == "visual_evidence"
        }
        self.assertEqual(
            visual,
            {
                "visual_layout",
                "landscape_layout",
                "visual_dominance",
                "sparse_centered",
                "slide_structure",
            },
        )

    def test_deck_vocabulary_with_visual_support_is_accepted(self):
        result = classify(slide_with_visual_support())
        gate = gate_of(result, "presentation_marketing")
        self.assertEqual(gate["status"], "satisfied")
        self.assertEqual(result["document_family"], "presentation_marketing")
        self.assertEqual(result["decision"], "classified")

    def test_marketing_copy_does_not_open_a_path(self):
        """v7: on the development set ``marketing_copy`` fired once, for an invoice.

        It still contributes score once a case is open; it may no longer be the
        case. The cost is real — an advertisement carrying no deck vocabulary is
        now refused — and it is taken deliberately.
        """
        result = classify(advertisement_with_visual_support())
        gate = gate_of(result, "presentation_marketing")
        self.assertNotEqual(gate["status"], "satisfied")
        self.assertNotEqual(result["document_family"], "presentation_marketing")
        self.assertEqual(result["candidate_scores"]["presentation_marketing"], 0.0)
        self.assertNotIn("marketing_copy", GATE_BY_FAMILY["presentation_marketing"].primary)

    def test_deck_vocabulary_without_any_corroboration_is_refused(self):
        result = classify(deck_vocabulary_without_corroboration())
        gate = gate_of(result, "presentation_marketing")
        self.assertEqual(gate["status"], "insufficient_evidence")
        self.assertEqual(gate["reason"], "missing_independent_presentation_evidence")
        self.assertEqual(result["candidate_scores"]["presentation_marketing"], 0.0)

    def test_near_empty_ocr_is_refused_for_insufficient_text(self):
        result = classify(near_empty_picture_page())
        gate = gate_of(result, "presentation_marketing")
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["reason"], "insufficient_presentation_text")

    def test_all_three_declared_reasons_are_reachable(self):
        reasons = {
            gate_of(classify(picture_page_without_primary_evidence()), "presentation_marketing")[
                "reason"
            ],
            gate_of(
                classify(deck_vocabulary_without_corroboration()), "presentation_marketing"
            )["reason"],
            gate_of(classify(near_empty_picture_page()), "presentation_marketing")["reason"],
        }
        self.assertEqual(
            reasons,
            {
                "visual_evidence_only",
                "missing_independent_presentation_evidence",
                "insufficient_presentation_text",
            },
        )

    def test_presentation_is_not_rescued_by_a_lower_threshold(self):
        """Requirement: the family's bar is not lowered to fix its precision."""
        self.assertNotIn("presentation_marketing", FAMILY_CONFIDENCE_THRESHOLDS)
        self.assertIn("presentation_marketing", FAMILY_THRESHOLD_HOLDS)
        self.assertEqual(
            effective_family_threshold("presentation_marketing", DEFAULT_CONFIDENCE_THRESHOLD),
            DEFAULT_CONFIDENCE_THRESHOLD,
        )

    def test_removed_rules_are_gone(self):
        """``bullet_layout`` measured what ``slide_structure`` now measures."""
        self.assertNotIn("presentation_marketing.bullet_layout", RULE_IDS)
        self.assertNotIn("presentation_marketing.bullet_layout", DEFAULT_WEIGHTS)


class PresentationV7ProbeTests(unittest.TestCase):
    """The four probes the v7 change is defined by.

    Driven from indicator vectors rather than documents, so each probe tests
    exactly one combination of fired rules and nothing else. The
    document-shaped versions of the same cases are in
    :class:`PresentationFalsePositiveShapeTests` below.
    """

    def probe(self, *rules):
        indicators = presentation_indicators(*rules)
        gates = evaluate_family_gates(PROBE_FEATURES, indicators)
        breakdown = score_families(PROBE_FEATURES, indicators)
        return gates["presentation_marketing"], breakdown["presentation_marketing"]

    def test_visual_layout_with_slide_structure_is_blocked(self):
        gate, _score = self.probe("visual_layout", "slide_structure")
        self.assertNotEqual(gate["status"], "satisfied")
        self.assertEqual(gate["reason"], "visual_evidence_only")

    def test_visual_dominance_with_slide_structure_is_blocked(self):
        gate, _score = self.probe("visual_dominance", "slide_structure")
        self.assertNotEqual(gate["status"], "satisfied")
        self.assertEqual(gate["reason"], "visual_evidence_only")

    def test_slide_structure_with_sparse_centered_is_blocked(self):
        gate, _score = self.probe("slide_structure", "sparse_centered")
        self.assertNotEqual(gate["status"], "satisfied")
        self.assertEqual(gate["reason"], "visual_evidence_only")

    def test_visual_layout_with_presentation_terms_is_satisfied(self):
        gate, _score = self.probe("visual_layout", "presentation_terms")
        self.assertEqual(gate["status"], "satisfied")
        self.assertEqual(gate["reason"], "path_deck_vocabulary_with_visual_support")

    def test_two_group_members_never_outscore_the_accepting_case(self):
        """The v6 arithmetic that no threshold could fix.

        Two visual/structural rules used to sum to 0.6562 while the only true
        positive — deck vocabulary with one visual signal — sat at 0.625.
        """
        _gate, false_positive = self.probe("visual_layout", "slide_structure")
        _gate, true_positive = self.probe("visual_layout", "presentation_terms")
        self.assertLess(false_positive["score"], true_positive["score"])
        self.assertGreaterEqual(true_positive["score"], DEFAULT_CONFIDENCE_THRESHOLD)

    def test_every_pair_without_deck_vocabulary_is_refused(self):
        visual = (
            "visual_layout",
            "landscape_layout",
            "visual_dominance",
            "sparse_centered",
            "slide_structure",
        )
        for first in visual:
            for second in visual:
                if first >= second:
                    continue
                with self.subTest(pair=(first, second)):
                    gate, score = self.probe(first, second)
                    self.assertNotEqual(gate["status"], "satisfied")
                    self.assertLess(score["score"], DEFAULT_CONFIDENCE_THRESHOLD)

    def test_marketing_copy_alone_does_not_open_the_gate(self):
        for companion in ("visual_layout", "slide_structure"):
            with self.subTest(companion=companion):
                gate, _score = self.probe("marketing_copy", companion)
                self.assertNotEqual(gate["status"], "satisfied")

    def test_threshold_is_unchanged_at_the_global_value(self):
        self.assertNotIn("presentation_marketing", FAMILY_CONFIDENCE_THRESHOLDS)
        self.assertEqual(
            effective_family_threshold(
                "presentation_marketing", DEFAULT_CONFIDENCE_THRESHOLD
            ),
            0.60,
        )


class PresentationFalsePositiveShapeTests(unittest.TestCase):
    """The two documents v6 still got wrong, reproduced as shapes.

    Named for what the pages look like, never for the samples: the classifier
    must not be able to tell which corpus document it is looking at, and a test
    keyed to a sample id would be testing a lookup rather than a rule.
    """

    def assert_not_a_presentation(self, regions, page_size=None):
        result = classify(regions, page_size=page_size)
        gate = gate_of(result, "presentation_marketing")
        self.assertNotEqual(result["document_family"], "presentation_marketing")
        self.assertNotEqual(gate["status"], "satisfied")
        self.assertEqual(result["candidate_scores"]["presentation_marketing"], 0.0)
        return result

    def test_illustrated_invoice_shape_is_not_a_presentation(self):
        self.assert_not_a_presentation(
            [
                region("INVOICE 4417", "Title", [50, 40, 600, 110]),
                region("", "Picture", [50, 150, 1150, 700]),
                region("", "Picture", [50, 720, 1150, 1180]),
                region("Bill To: Acme", "Text", [50, 1200, 560, 1250]),
                region("Amount Due $412.00", "Text", [50, 1260, 560, 1310]),
                region("Terms: net 30", "Text", [50, 1320, 560, 1370]),
            ]
        )

    def test_illustrated_form_shape_is_not_a_presentation(self):
        self.assert_not_a_presentation(
            [
                region("REQUEST FORM", "Title", [50, 40, 600, 110]),
                region("", "Picture", [50, 150, 1150, 700]),
                region("", "Picture", [50, 720, 1150, 1180]),
                region("Name: ______", "Text", [50, 1200, 560, 1250]),
                region("Unit: ______", "Text", [50, 1260, 560, 1310]),
                region("[ ] Yes  [ ] No", "List-item", [50, 1320, 560, 1370]),
            ]
        )

    def test_handwritten_and_file_folder_shapes_are_not_presentations(self):
        shapes = {
            "handwritten_note": [
                region("note to self about tomorrow morning and the meeting", "Text"),
            ],
            "file_folder_tab": [
                region("", "Picture", [50, 50, 1150, 1400]),
                region("CORRESPONDENCE 1978 GENERAL FILE", "Title", [50, 1420, 800, 1500]),
            ],
        }
        for name, regions in shapes.items():
            with self.subTest(document=name):
                result = classify(regions)
                self.assertNotEqual(result["document_family"], "presentation_marketing")


# --------------------------------------------------------------------------
# form_structured
# --------------------------------------------------------------------------


class FormGateTests(unittest.TestCase):
    def test_development_set_form_shapes_are_recovered(self):
        """Regression test for the five development-set forms.

        This records what the two paths recovered on the documents that were
        available; it is deliberately not a generalisation claim, and the
        holdout is what would make it one.
        """
        for name, regions in FORM_DOCUMENTS.items():
            with self.subTest(document=name):
                result = classify(regions)
                self.assertEqual(
                    gate_of(result, "form_structured")["status"], "satisfied", result
                )
                self.assertEqual(result["document_family"], "form_structured")

    def test_form_shaped_negatives_are_still_refused(self):
        for name, regions in FORM_NEGATIVES.items():
            with self.subTest(document=name):
                result = classify(regions)
                self.assertNotEqual(result["document_family"], "form_structured")

    def test_structure_alone_never_opens_the_gate(self):
        """No geometry-only path, and no ``field_grid`` by another name."""
        result = classify(FORM_NEGATIVES["labelled_listing"])
        gate = gate_of(result, "form_structured")
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["reason"], "form_structure_only")
        self.assertEqual(result["candidate_scores"]["form_structured"], 0.0)
        self.assertNotIn("form_structured.field_grid", RULE_IDS)

    def test_the_three_paths_are_the_declared_ones(self):
        gate = next(item for item in FAMILY_GATES if item.family == "form_structured")
        self.assertEqual(
            [path.name for path in gate.paths],
            [
                "A_questionnaire_with_structure",
                "B_checkboxes_with_structure",
                "C_form_heading_with_two_corroborations",
            ],
        )

    def test_each_path_is_the_one_that_fires(self):
        expected = {
            "questionnaire_with_labels": "path_A_questionnaire_with_structure",
            "checkboxes_with_labels": "path_B_checkboxes_with_structure",
            "form_heading_with_two_corroborations": (
                "path_C_form_heading_with_two_corroborations"
            ),
        }
        for name, reason in expected.items():
            with self.subTest(path=name):
                result = classify(FORM_DOCUMENTS[name])
                self.assertEqual(gate_of(result, "form_structured")["reason"], reason)

    def test_development_threshold_is_declared_as_a_candidate(self):
        self.assertEqual(FAMILY_CONFIDENCE_THRESHOLDS["form_structured"], 0.40)
        self.assertEqual(
            FAMILY_THRESHOLD_PROVENANCE["form_structured"],
            "development_set_candidate_requires_holdout",
        )


# --------------------------------------------------------------------------
# news_article
# --------------------------------------------------------------------------


class NewsGateTests(unittest.TestCase):
    def test_byline_with_reported_speech_is_accepted(self):
        result = classify(news_with_byline_and_attribution())
        self.assertEqual(gate_of(result, "news_article")["status"], "satisfied")
        self.assertEqual(result["document_family"], "news_article")

    def test_byline_with_columns_alone_is_refused(self):
        result = classify(news_with_byline_and_columns_only())
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertEqual(result["candidate_scores"]["news_article"], 0.0)

    def test_wire_service_with_attribution_alone_is_refused(self):
        result = classify(news_with_wire_and_attribution_only())
        self.assertNotEqual(result["document_family"], "news_article")
        self.assertEqual(result["candidate_scores"]["news_article"], 0.0)

    def test_headline_body_was_removed(self):
        """Six firings, zero news articles: the name claimed what it never measured."""
        self.assertNotIn("news_article.headline_body", RULE_IDS)
        self.assertNotIn("news_article.headline_body", DEFAULT_WEIGHTS)

    def test_news_keeps_the_global_threshold(self):
        self.assertNotIn("news_article", FAMILY_CONFIDENCE_THRESHOLDS)
        self.assertIn("news_article", FAMILY_THRESHOLD_HOLDS)


# --------------------------------------------------------------------------
# correspondence
# --------------------------------------------------------------------------


class CorrespondenceThresholdTests(unittest.TestCase):
    """v7 moves the family bar from 0.40 to 0.42. Rules and gate are untouched."""

    def setUp(self):
        self.family_thresholds = resolve_family_thresholds(DEFAULT_CONFIDENCE_THRESHOLD)

    def decide(self, score, margin=0.30):
        return apply_decision_policy(
            [("correspondence", score), ("other", round(score - margin, 4))],
            500,
            DEFAULT_CONFIDENCE_THRESHOLD,
            DEFAULT_MIN_SCORE_MARGIN,
            DEFAULT_MIN_RECOGNIZED_CHARACTERS,
            "evaluate",
            self.family_thresholds,
        )

    def test_declared_threshold_is_the_v7_value(self):
        self.assertEqual(FAMILY_CONFIDENCE_THRESHOLDS["correspondence"], 0.42)
        self.assertEqual(
            FAMILY_THRESHOLD_PROVENANCE["correspondence"],
            "development_set_candidate_requires_holdout",
        )
        self.assertEqual(self.family_thresholds["correspondence"], 0.42)

    def test_score_below_the_threshold_is_refused(self):
        decision = self.decide(0.41)
        self.assertEqual(decision["decision"], "fallback")
        self.assertEqual(decision["reason"], "score_below_threshold")
        self.assertEqual(decision["document_family"], "other")
        self.assertEqual(decision["applied_confidence_threshold"], 0.42)

    def test_score_at_the_threshold_with_margin_is_accepted(self):
        decision = self.decide(0.42)
        self.assertEqual(decision["decision"], "classified")
        self.assertEqual(decision["document_family"], "correspondence")
        self.assertEqual(decision["applied_confidence_threshold"], 0.42)

    def test_score_above_the_threshold_without_margin_still_abstains(self):
        """The margin is not relaxed along with the threshold."""
        decision = self.decide(0.50, margin=0.02)
        self.assertEqual(decision["decision"], "abstained")
        self.assertEqual(decision["reason"], "ambiguous_rule_scores")

    def test_an_ordinary_letter_is_still_accepted(self):
        result = classify(
            [
                region("Dear Mr Smith,", "Title", index=0),
                region(
                    "Thank you for your note about the shipment schedule for the coming "
                    "quarter. We will confirm the revised dates in writing next week.",
                    index=1,
                ),
                region("Sincerely yours,\nJ. Doe", index=2),
            ]
        )
        self.assertEqual(result["document_family"], "correspondence")
        self.assertEqual(result["decision"], "classified")
        self.assertGreaterEqual(result["score"], 0.42)


# --------------------------------------------------------------------------
# Configuration invariants
# --------------------------------------------------------------------------


class GateConfigurationTests(unittest.TestCase):
    def test_every_rule_of_a_gated_family_is_classified(self):
        rules_by_family: dict[str, set[str]] = {}
        for rule in RULES:
            rules_by_family.setdefault(rule.family, set()).add(rule.name)
        for gate in FAMILY_GATES:
            with self.subTest(family=gate.family):
                self.assertEqual(
                    set(gate.primary) | set(gate.corroborating),
                    rules_by_family[gate.family],
                )
                self.assertFalse(set(gate.primary) & set(gate.corroborating))

    def test_every_declared_blocker_exists(self):
        for gate in FAMILY_GATES:
            for name in gate.blockers:
                with self.subTest(family=gate.family, blocker=name):
                    self.assertIn(name, BLOCKER_PREDICATES)

    def test_declared_thresholds_are_the_development_candidates(self):
        self.assertEqual(
            FAMILY_CONFIDENCE_THRESHOLDS,
            {
                "correspondence": 0.42,
                "form_structured": 0.40,
                "research_paper": 0.43,
                "resume": 0.30,
                "technical_report": 0.31,
            },
        )
        self.assertEqual(
            set(FAMILY_THRESHOLD_PROVENANCE), set(FAMILY_CONFIDENCE_THRESHOLDS)
        )
        self.assertTrue(
            all(
                value == "development_set_candidate_requires_holdout"
                for value in FAMILY_THRESHOLD_PROVENANCE.values()
            )
        )

    def test_precision_limited_families_are_held_at_the_global_threshold(self):
        for family in ("presentation_marketing", "news_article", "financial_document"):
            with self.subTest(family=family):
                self.assertNotIn(family, FAMILY_CONFIDENCE_THRESHOLDS)
                self.assertIn(family, FAMILY_THRESHOLD_HOLDS)

    def test_declared_thresholds_resolve_to_their_values_at_the_default(self):
        resolved = resolve_family_thresholds(DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(resolved, FAMILY_CONFIDENCE_THRESHOLDS)

    def test_declared_thresholds_move_with_the_global_threshold(self):
        """Offsets, not absolutes: a swept curve must describe the real policy."""
        resolved = resolve_family_thresholds(DEFAULT_CONFIDENCE_THRESHOLD + 0.10)
        self.assertAlmostEqual(resolved["form_structured"], 0.50)
        self.assertAlmostEqual(resolved["resume"], 0.40)

    def test_thresholds_are_declared_only_for_scorable_families(self):
        self.assertTrue(set(FAMILY_CONFIDENCE_THRESHOLDS) <= set(SCORED_FAMILIES))

    def test_rule_fingerprint_is_stable_across_processes(self):
        """A fingerprint that differs per run cannot support a claim."""
        command = (
            "import sys; sys.path.insert(0, %r);"
            "from tasks.document.rules_classifier_core import RULE_FINGERPRINT;"
            "print(RULE_FINGERPRINT)" % str(Path(__file__).resolve().parents[1])
        )
        output = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, check=True
        )
        self.assertEqual(output.stdout.strip(), RULE_FINGERPRINT)


if __name__ == "__main__":
    unittest.main()
