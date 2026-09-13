"""Contract tests: identity, reproducibility, and the shape of what is returned.

These are the properties a benchmark result rests on but that no metric would
ever reveal as broken: a fingerprint that misses a threshold, a classification
that changes when you ask it for its own indicator vector, a task wrapper that
hands the runner a one-element tuple.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tasks.document import rules_classifier_core as core  # noqa: E402
from tasks.document.rules_classifier_core import (  # noqa: E402
    CLASSIFIER_VERSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    FAMILIES_NOT_RELEASED_FOR_ROUTING,
    FEATURE_EXTRACTION_VERSION,
    ROUTING_RELEASE_STATUS,
    RULE_FINGERPRINT,
    SCHEMA_VERSION,
    TAXONOMY_VERSION,
    classify_with_rules,
    extract_classification_features,
)


def sample_documents():
    """A handful of documents spanning several families and both decisions."""
    def page(regions):
        return {
            "total_pages": 1,
            "pages": [{"page_number": 1, "regions": regions}],
            "full_text": "",
        }

    def region(text, class_name="Text", index=0):
        return {
            "class_name": class_name,
            "text": text,
            "bbox": [80, 80 + index * 120, 1120, 180 + index * 120],
        }

    return [
        page(
            [
                region("INVOICE 9381", "Title", 0),
                region("Bill To: Example Ltd", index=1),
                region("Subtotal $100.00\nTax $10.00\nTotal Due $110.00", index=2),
            ]
        ),
        page(
            [
                region("Dear Mr Smith,", "Title", 0),
                region(
                    "Thank you for your note about the shipment schedule for the coming "
                    "quarter; we will confirm the revised dates next week.",
                    index=1,
                ),
                region("Sincerely yours,\nJ. Doe", index=2),
            ]
        ),
        page(
            [
                region("APPLICATION FORM", "Title", 0),
                region("Name: ______\nAddress: ______\nCity: ______", index=1),
                region("[ ] Yes  [ ] No", "List-item", 2),
            ]
        ),
        page([region("a handwritten scrawl about nothing in particular", "Text", 0)]),
    ]


def features_for(document):
    return extract_classification_features(document, page_sizes=[[1200, 1600]])


class VersionContractTests(unittest.TestCase):
    def test_v9_identity(self):
        self.assertTrue(CLASSIFIER_VERSION.startswith("rules-rvl-cdip-v9+"))
        self.assertTrue(CLASSIFIER_VERSION.endswith(RULE_FINGERPRINT))

    def test_contract_versions(self):
        """v8 adds features and subtypes; the external record contract is unchanged.

        ``SCHEMA_VERSION`` stays at 2.1 because ``document_subtype`` is optional
        and additive — a consumer that ignores it reads the same record it read
        before. ``FEATURE_EXTRACTION_VERSION`` moves because
        ``accounting_negative_count`` was *redefined*, not merely joined by new
        keys, and ``TAXONOMY_VERSION`` moves because a subtype vocabulary now
        exists to be reported.
        """
        self.assertEqual(SCHEMA_VERSION, "2.1")
        self.assertEqual(TAXONOMY_VERSION, "rvl-cdip-2.1")
        self.assertEqual(FEATURE_EXTRACTION_VERSION, "2.5")

    def test_feature_record_carries_the_current_extraction_version(self):
        features = features_for(sample_documents()[0])
        self.assertEqual(features["feature_extraction_version"], "2.5")
        self.assertEqual(
            features["feature_fingerprint"], core.feature_fingerprint(features)
        )

    def test_result_reports_every_identity(self):
        result = classify_with_rules(features_for(sample_documents()[0]))
        features = features_for(sample_documents()[0])
        self.assertEqual(result["classifier_version"], CLASSIFIER_VERSION)
        self.assertEqual(result["rule_fingerprint"], RULE_FINGERPRINT)
        self.assertEqual(result["feature_fingerprint"], features["feature_fingerprint"])
        self.assertEqual(result["feature_extraction_version"], "2.5")

    def test_subtype_is_present_and_optional(self):
        """Additive: the key is always there, ``None`` when it does not apply."""
        for document in sample_documents():
            result = classify_with_rules(features_for(document))
            self.assertIn("document_subtype", result)
            self.assertIsNone(result["document_subtype"])


class RoutingReleaseTests(unittest.TestCase):
    def test_families_withheld_from_automatic_routing(self):
        """v9 adds news_article: its newspaper paths have no holdout."""
        self.assertEqual(
            FAMILIES_NOT_RELEASED_FOR_ROUTING,
            frozenset({"financial_document", "presentation_marketing", "news_article"}),
        )

    def test_every_withheld_family_records_why(self):
        from tasks.document.rules_classifier_core import ROUTING_RELEASE_NOTES

        for family in FAMILIES_NOT_RELEASED_FOR_ROUTING:
            with self.subTest(family=family):
                self.assertTrue(ROUTING_RELEASE_NOTES.get(family))
        self.assertIn("holdout", ROUTING_RELEASE_NOTES["news_article"])

    def test_no_family_is_marked_production_ready(self):
        self.assertTrue(
            all(
                status in {"development_only", "not_released_for_automatic_routing"}
                for status in ROUTING_RELEASE_STATUS.values()
            )
        )

    def test_every_result_says_it_is_not_released(self):
        for document in sample_documents():
            with self.subTest(document=document["pages"][0]["regions"][0]["text"][:20]):
                result = classify_with_rules(features_for(document))
                self.assertFalse(result["released_for_automatic_routing"])
                self.assertIn("routing_release_status", result)


class GatePathReachabilityTests(unittest.TestCase):
    """Every declared path must be able to reach the bar it is judged against.

    A gate that opens on a combination the scorer cannot carry to the threshold
    is a path that exists on paper and classifies nothing: the document passes
    the gate, scores below the bar, and is refused with a reason that describes
    the score rather than the cause. v8 shipped exactly that for
    ``byline + multilingual_reporting`` — 0.5882 against 0.60.
    """

    #: Paths known to be unreachable, with the numbers, so that a *new* one
    #: fails this test instead of joining a silent backlog. Both pre-date v9
    #: and both need a decision — a weight change or a threshold change — in a
    #: family this round was explicitly scoped away from, so neither is fixed
    #: here rather than adjusted without measurement.
    KNOWN_UNREACHABLE = {
        # citations (0.14) + two_column_layout (0.08) = 0.22 / 0.69 = 0.3188
        # against a 0.43 development-set threshold.
        ("research_paper", "two_independent_academic_groups"),
        # presentation_terms (0.18) + sparse_centered (0.12) = 0.30 / 0.62
        # = 0.4839 against the global 0.60.
        ("presentation_marketing", "deck_vocabulary_with_visual_support"),
    }

    def setUp(self):
        self.rows = core.check_gate_path_reachability()

    def test_every_news_article_path_is_reachable(self):
        for row in self.rows:
            if row["family"] != "news_article":
                continue
            with self.subTest(path=row["path"]):
                self.assertTrue(
                    row["reachable"],
                    f"{row['path']} minimally scores {row['score']} "
                    f"against {row['threshold']}: {row['minimal_rules']}",
                )

    def test_no_new_path_is_unreachable(self):
        unreachable = {
            (row["family"], row["path"]) for row in self.rows if not row["reachable"]
        }
        self.assertEqual(
            unreachable,
            self.KNOWN_UNREACHABLE,
            "a gate path changed reachability; fix it or record it deliberately",
        )

    def test_the_check_reports_a_minimal_combination_that_clears(self):
        row = next(
            item for item in self.rows if item["path"] == "byline_with_reporting"
        )
        # The cheapest satisfying set, not the strongest: the weakest accepting
        # combination is the one a real document presents. Which member of the
        # reporting group it names is a tie-break — they now weigh the same —
        # so the assertion is on the shape and the outcome.
        self.assertIn("byline", row["minimal_rules"])
        self.assertEqual(len(row["minimal_rules"]), 2)
        self.assertGreaterEqual(row["score"], row["threshold"])

    def test_the_v8_failing_combination_now_clears(self):
        """``byline + multilingual_reporting`` scored 0.5882 against 0.60."""
        indicators = {
            rule_id: rule_id
            in {"news_article.byline", "news_article.multilingual_reporting"}
            for rule_id in core.RULE_IDS
        }
        score = core.score_families(
            {"measured_page_ratio": 1.0, "geometry_page_ratio": 1.0, "total_pages": 1},
            indicators,
        )["news_article"]["score"]
        self.assertGreaterEqual(score, DEFAULT_CONFIDENCE_THRESHOLD)


class ScoreContractTests(unittest.TestCase):
    """One documented contract: unclipped ratio for ranking, clipped confidence.

    The evaluator ranks on ``score`` and computes margins from
    ``candidate_scores``; both are evidence ratios and both may exceed 1 when a
    document carries more than a sufficient case. Clipping them would make two
    strongly evidenced families meet at the ceiling and produce a zero margin,
    which is the ambiguity the margin exists to detect. ``confidence`` is the
    reporting value and is clipped to [0, 1].
    """

    def well_evidenced_document(self):
        def region(text, class_name, index):
            return {
                "class_name": class_name,
                "text": text,
                "bbox": [80, 80 + index * 140, 1120, 200 + index * 140],
            }

        return {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "regions": [
                        region("INVOICE 9381", "Title", 0),
                        region("Bill To: Example Ltd\nRemit to: Example Bank", "Text", 1),
                        region(
                            "Subtotal $100.00\nTax $10.00\nTotal Due $110.00\n"
                            "Amount due on receipt. Balance sheet, ledger and "
                            "disbursements recorded; accounts payable audited.",
                            "Text",
                            2,
                        ),
                        region(
                            "Invoices, receipts and expenditures for January, "
                            "February and March are attached, with unit price and "
                            "subtotal per line: $12.00, $18.50, $44.25.",
                            "Text",
                            3,
                        ),
                    ],
                }
            ],
            "full_text": "",
        }

    def test_score_may_exceed_one_while_confidence_may_not(self):
        result = classify_with_rules(features_for(self.well_evidenced_document()))
        self.assertGreater(result["score"], 1.0)
        self.assertGreater(max(result["candidate_scores"].values()), 1.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(result["confidence"], min(round(result["score"], 4), 1.0))

    def test_the_module_documents_the_contract(self):
        self.assertIn("may exceed 1", core.__doc__)
        self.assertIn("confidence", core.__doc__)


class FingerprintSensitivityTests(unittest.TestCase):
    """Anything that can change a decision must change the fingerprint."""

    def assert_fingerprint_changes(self, attribute, value):
        with mock.patch.object(core, attribute, value):
            changed = core._rule_fingerprint()
        self.assertNotEqual(changed, RULE_FINGERPRINT, attribute)
        # And it must come back: the patch is the only difference.
        self.assertEqual(core._rule_fingerprint(), RULE_FINGERPRINT)

    def test_a_family_threshold_change_changes_the_fingerprint(self):
        self.assert_fingerprint_changes(
            "FAMILY_CONFIDENCE_THRESHOLDS",
            {**core.FAMILY_CONFIDENCE_THRESHOLDS, "correspondence": 0.40},
        )

    def test_a_global_threshold_change_changes_the_fingerprint(self):
        self.assert_fingerprint_changes("DEFAULT_CONFIDENCE_THRESHOLD", 0.55)

    def test_a_margin_change_changes_the_fingerprint(self):
        self.assert_fingerprint_changes("DEFAULT_MIN_SCORE_MARGIN", 0.12)

    def test_a_recognized_character_minimum_change_changes_the_fingerprint(self):
        self.assert_fingerprint_changes("DEFAULT_MIN_RECOGNIZED_CHARACTERS", 30)

    def test_a_decision_group_count_change_changes_the_fingerprint(self):
        self.assert_fingerprint_changes("DECISION_GROUP_COUNT", 2)

    def test_a_weight_change_changes_the_fingerprint(self):
        weights = dict(core.DEFAULT_WEIGHTS)
        weights["correspondence.salutation"] = 0.99
        self.assert_fingerprint_changes("DEFAULT_WEIGHTS", weights)

    def test_a_gate_change_changes_the_fingerprint(self):
        gates = []
        for gate in core.FAMILY_GATES:
            if gate.family == "presentation_marketing":
                gate = gate._replace(reason_no_path="something_else")
            gates.append(gate)
        self.assert_fingerprint_changes("FAMILY_GATES", tuple(gates))

    def test_a_substitutable_group_change_changes_the_fingerprint(self):
        rules = []
        for rule in core.RULES:
            if rule.rule_id == "presentation_marketing.slide_structure":
                rule = rule._replace(group="slide_structure")
            rules.append(rule)
        self.assert_fingerprint_changes("RULES", tuple(rules))

    def test_fingerprint_is_identical_in_three_separate_processes(self):
        command = (
            "import sys; sys.path.insert(0, %r);"
            "from tasks.document.rules_classifier_core import RULE_FINGERPRINT;"
            "print(RULE_FINGERPRINT)" % str(REPOSITORY_ROOT)
        )
        observed = {
            subprocess.run(
                [sys.executable, "-c", command],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(3)
        }
        self.assertEqual(observed, {RULE_FINGERPRINT})


class IndicatorEquivalenceTests(unittest.TestCase):
    """Asking for the indicator vector must not change the answer.

    The latency benchmark measures the two modes separately; if they could
    disagree, the cheaper measurement would be describing a different
    classifier from the one that runs.
    """

    DECISION_FIELDS = (
        "document_family",
        "decision",
        "reason",
        "confidence",
        "score",
        "score_margin",
        "candidate_scores",
        "top_candidate",
        "runner_up",
        "decision_mass",
        "available_mass",
    )

    def test_decisions_are_identical_with_and_without_indicators(self):
        for document in sample_documents():
            features = features_for(document)
            for mode in ("observe", "evaluate", "auto"):
                with self.subTest(mode=mode):
                    without = classify_with_rules(
                        features, mode=mode, include_indicators=False
                    )
                    with_indicators = classify_with_rules(
                        features, mode=mode, include_indicators=True
                    )
                    for field in self.DECISION_FIELDS:
                        self.assertEqual(
                            without[field], with_indicators[field], f"{mode}/{field}"
                        )
                    self.assertNotIn("rule_indicators", without)
                    self.assertIn("rule_indicators", with_indicators)

    def test_evidence_is_identical_apart_from_the_indicator_vector(self):
        features = features_for(sample_documents()[0])
        without = classify_with_rules(features, include_indicators=False)
        with_indicators = classify_with_rules(features, include_indicators=True)
        self.assertEqual(without["evidence"], with_indicators["evidence"])
        self.assertEqual(
            set(with_indicators) - set(without), {"rule_indicators"}
        )


class TaskWrapperContractTests(unittest.TestCase):
    """Single-output wrappers return a dict, never a one-element tuple.

    FabricFlow's ``core.task`` is not vendored here, so it is stubbed: the
    decorator is identity for this purpose, and what is under test is the
    function body's return value, not the framework's registration.
    """

    def setUp(self):
        def task(*_args, **_kwargs):
            def decorate(function):
                return function

            return decorate

        core_module = types.ModuleType("core")
        task_module = types.ModuleType("core.task")
        task_module.task = task
        core_module.task = task_module
        self._patcher = mock.patch.dict(
            sys.modules, {"core": core_module, "core.task": task_module}
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_feature_extraction_wrapper_returns_a_dict(self):
        from tasks.document.extract_document_classification_features import (
            extract_document_classification_features,
        )

        features = extract_document_classification_features(
            sample_documents()[0], page_sizes=[[1200, 1600]]
        )
        self.assertIsInstance(features, dict)
        self.assertNotIsInstance(features, tuple)
        self.assertEqual(features["schema_version"], SCHEMA_VERSION)

    def test_classification_wrapper_returns_a_dict(self):
        from tasks.document.classify_document_rules import classify_document_rules

        result = classify_document_rules(features_for(sample_documents()[0]))
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, tuple)
        self.assertEqual(result["classifier_version"], CLASSIFIER_VERSION)
        self.assertIn("execution_time_ms", result)

    def test_classification_wrapper_forwards_include_indicators(self):
        from tasks.document.classify_document_rules import classify_document_rules

        features = features_for(sample_documents()[0])
        self.assertNotIn(
            "rule_indicators", classify_document_rules(features)
        )
        self.assertIn(
            "rule_indicators",
            classify_document_rules(features, include_indicators=True),
        )


class LatencyBenchmarkHelperTests(unittest.TestCase):
    """The benchmark's own machinery, tested away from any timing.

    Timings are not asserted on — a test that asserts a duration is a test that
    fails on a loaded machine — but the equivalence check and the statistics
    are pure functions and are.
    """

    def setUp(self):
        sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
        import benchmark_classifier_latency as benchmark

        self.benchmark = benchmark

    def test_summary_reports_every_required_statistic(self):
        summary = self.benchmark.summarise([float(value) for value in range(1, 101)])
        self.assertEqual(
            set(summary),
            {"call_count", "mean", "median", "p90", "p95", "p99", "stdev", "min", "max"},
        )
        self.assertEqual(summary["call_count"], 100)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 100.0)
        self.assertEqual(summary["p95"], 95.0)
        self.assertEqual(summary["median"], 50.5)

    def test_decision_signature_ignores_the_indicator_vector(self):
        features = features_for(sample_documents()[0])
        without = classify_with_rules(features, include_indicators=False)
        with_indicators = classify_with_rules(features, include_indicators=True)
        self.assertEqual(
            self.benchmark.decision_signature(without),
            self.benchmark.decision_signature(with_indicators),
        )

    def test_decision_signature_notices_a_different_answer(self):
        documents = sample_documents()
        first = classify_with_rules(features_for(documents[0]))
        second = classify_with_rules(features_for(documents[1]))
        self.assertNotEqual(
            self.benchmark.decision_signature(first),
            self.benchmark.decision_signature(second),
        )

    def test_equivalence_check_passes_on_real_feature_records(self):
        records = [features_for(document) for document in sample_documents()]
        options = self.benchmark.Options(
            repetitions=20,
            trials=3,
            warmup_repetitions=1,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            min_score_margin=DEFAULT_MIN_SCORE_MARGIN,
            min_recognized_characters=DEFAULT_MIN_RECOGNIZED_CHARACTERS,
            classification_mode="evaluate",
        )
        equivalent, differences = self.benchmark.check_equivalence(records, options)
        self.assertTrue(equivalent)
        self.assertEqual(differences, [])


class DefaultOperatingPointTests(unittest.TestCase):
    def test_defaults_are_unchanged_in_v7(self):
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.60)
        self.assertEqual(DEFAULT_MIN_SCORE_MARGIN, 0.10)
        self.assertEqual(DEFAULT_MIN_RECOGNIZED_CHARACTERS, 20)


if __name__ == "__main__":
    unittest.main()
