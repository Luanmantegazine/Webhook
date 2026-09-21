"""Leakage detection, cache reuse and the evaluator's two perspectives.

The single most expensive mistake available in this increment is measuring a
threshold on the documents it was chosen on. These tests drive the real
scripts — build, calibrate, evaluate — over a temporary corpus with an injected
fake embedder, and assert that every overlap stops the run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_doubles import DirectionalEmbedder, feature_record  # noqa: E402
from scripts import build_embedding_reference, calibrate_embedding_thresholds  # noqa: E402
from scripts import evaluate_embeddings_from_cache  # noqa: E402
from scripts.embedding_corpus import (  # noqa: E402
    CorpusError,
    LeakageError,
    check_split_disjointness,
    index_execution_features,
    load_feature_record,
    read_manifest,
    require_disjoint_splits,
)
from scripts.embedding_metrics import (  # noqa: E402
    conventional_metrics,
    is_accepted_wrong,
    operating_point_at_coverage,
    operating_point_at_risk,
    risk_coverage_curve,
    selective_metrics,
)

LABEL_MARKERS = {
    "invoice": "invoice",
    "letter": "letter",
    "news article": "newspaper",
}
FAMILY_OF = {
    "invoice": "financial_document",
    "letter": "correspondence",
    "news article": "news_article",
}
BODY = "page of running text with several ordinary words printed on it, number"


class CorpusFixture:
    """A miniature corpus on disk: manifests plus stored feature records."""

    def __init__(self, root: Path):
        self.root = root
        self.executions = root / "executions"
        self.executions.mkdir(parents=True, exist_ok=True)

    def write_split(self, name: str, *, per_label: int, start: int, rejection: int = 0):
        rows = []
        for label, marker in LABEL_MARKERS.items():
            for offset in range(per_label):
                index = start + offset
                sample_id = f"{name}-{marker}-{index:03d}"
                text = f"{marker} {BODY} {index}"
                self._write_features(sample_id, text)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "rvl_label": label,
                        "image_path": f"images/{sample_id}.tif",
                    }
                )
        for offset in range(rejection):
            sample_id = f"{name}-handwritten-{start + offset:03d}"
            self._write_features(sample_id, f"scribbled note {BODY} {offset}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "rvl_label": "handwritten",
                    "image_path": f"images/{sample_id}.tif",
                }
            )
        manifest = self.root / f"{name}_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "rvl_label", "image_path")
            )
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def copy_rows(self, name: str, source: Path, limit: int = 2):
        """A second manifest naming documents the first one already names."""
        with source.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[:limit]
        manifest = self.root / f"{name}_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "rvl_label", "image_path")
            )
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def sparse_label(self, name: str, label: str, marker: str, count: int, start: int):
        """Append a label with too few documents to support a centroid."""
        manifest = self.root / f"{name}_manifest.csv"
        with manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for offset in range(count):
            sample_id = f"{name}-{marker}-sparse-{start + offset:03d}"
            self._write_features(sample_id, f"{marker} {BODY} {start + offset}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "rvl_label": label,
                    "image_path": f"images/{sample_id}.tif",
                }
            )
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "rvl_label", "image_path")
            )
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def _write_features(self, sample_id: str, text: str):
        directory = self.executions / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "classification_features.json").write_text(
            json.dumps(feature_record(text, sample_id=sample_id)), encoding="utf-8"
        )


def embedder():
    return DirectionalEmbedder({"invoice": 0, "letter": 1, "newspaper": 2}, dimension=8)


class SplitIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.corpus = CorpusFixture(self.root)
        self.reference = self.corpus.write_split("reference", per_label=4, start=0)
        self.calibration = self.corpus.write_split("calibration", per_label=4, start=100)
        self.evaluation = self.corpus.write_split(
            "validation", per_label=4, start=200, rejection=2
        )

    def test_disjoint_splits_are_accepted(self):
        report = require_disjoint_splits(
            {
                "reference": read_manifest(self.reference),
                "calibration": read_manifest(self.calibration),
                "evaluation": read_manifest(self.evaluation),
            }
        )
        self.assertTrue(report["disjoint"])
        self.assertEqual(report["overlaps"], [])

    def test_a_shared_sample_id_stops_the_run(self):
        overlapping = self.corpus.copy_rows("overlap", self.reference, limit=3)
        with self.assertRaises(LeakageError) as error:
            require_disjoint_splits(
                {
                    "reference": read_manifest(self.reference),
                    "evaluation": read_manifest(overlapping),
                }
            )
        self.assertIn("not disjoint", str(error.exception))

    def test_a_shared_image_path_stops_the_run_even_with_new_ids(self):
        with self.evaluation.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        renamed = self.root / "renamed_manifest.csv"
        with renamed.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "rvl_label", "image_path")
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "sample_id": f"renamed-{row['sample_id']}"})
        report = check_split_disjointness(
            {"a": read_manifest(self.evaluation), "b": read_manifest(renamed)}
        )
        self.assertFalse(report["disjoint"])
        self.assertGreater(report["overlaps"][0]["shared_image_path_count"], 0)
        self.assertEqual(report["overlaps"][0]["shared_sample_id_count"], 0)

    def test_a_repeated_sample_id_inside_one_manifest_is_refused(self):
        duplicated = self.root / "duplicated.csv"
        duplicated.write_text(
            "sample_id,rvl_label\nx,invoice\nx,letter\n", encoding="utf-8"
        )
        with self.assertRaises(CorpusError):
            read_manifest(duplicated)

    def test_an_unknown_label_is_refused_rather_than_mapped_to_other(self):
        bad = self.root / "bad.csv"
        bad.write_text("sample_id,rvl_label\nx,tax return\n", encoding="utf-8")
        with self.assertRaises(CorpusError) as error:
            read_manifest(bad)
        self.assertIn("unknown rvl_label", str(error.exception))


class CacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.corpus = CorpusFixture(self.root)
        self.reference = self.corpus.write_split("reference", per_label=4, start=0)

    def test_stored_features_are_found_recursively_by_sample_id(self):
        found = index_execution_features(self.corpus.executions)
        rows = read_manifest(self.reference)
        for row in rows:
            self.assertIn(row.sample_id, found)

    def test_a_stored_record_is_read_and_never_re_extracted(self):
        found = index_execution_features(self.corpus.executions)
        path = next(iter(found.values()))
        record = load_feature_record(path)
        self.assertIn("classification_text", record)
        self.assertEqual(record["schema_version"], "2.1")

    def test_a_workflow_output_wrapper_is_unwrapped(self):
        wrapped = self.corpus.executions / "wrapped"
        wrapped.mkdir()
        (wrapped / "classification_features.json").write_text(
            json.dumps({"outputs": {"classification_features": feature_record("invoice body")}}),
            encoding="utf-8",
        )
        record = load_feature_record(wrapped / "classification_features.json")
        self.assertIn("classification_text", record)

    def test_a_file_without_features_is_refused(self):
        path = self.root / "not_features.json"
        path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
        with self.assertRaises(CorpusError) as error:
            load_feature_record(path)
        self.assertIn("never re-runs OCR", str(error.exception))


class PipelineTests(unittest.TestCase):
    """Build, calibrate and evaluate, driven as the scripts really run."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.corpus = CorpusFixture(self.root)
        self.reference_manifest = self.corpus.write_split("reference", per_label=5, start=0)
        self.calibration_manifest = self.corpus.write_split(
            "calibration", per_label=5, start=100, rejection=2
        )
        self.evaluation_manifest = self.corpus.write_split(
            "validation", per_label=5, start=200, rejection=2
        )
        self.index_directory = self.root / "index"
        self.embedder = embedder()

    def build(self, **overrides):
        args = argparse.Namespace(
            manifest=str(self.reference_manifest),
            output=str(self.index_directory),
            executions=str(self.corpus.executions),
            config="",
            model_name="fake/encoder",
            model_revision="",
            max_tokens=10,
            overlap_tokens=4,
            minimum_examples=1,
            disjoint_from=[],
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return build_embedding_reference.run(args, embedder=self.embedder)

    def calibrate(self, **overrides):
        args = argparse.Namespace(
            manifest=str(self.calibration_manifest),
            reference=str(self.index_directory),
            output=str(self.root / "thresholds.json"),
            executions=str(self.corpus.executions),
            config="",
            target_selective_accuracy=0.95,
            similarity_grid_start=0.0,
            similarity_grid_stop=1.0,
            similarity_grid_step=0.05,
            margin_grid_start=0.0,
            margin_grid_stop=0.05,
            margin_grid_step=0.05,
            minimum_family_support=3,
            disjoint_from=[],
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return calibrate_embedding_thresholds.run(args, embedder=self.embedder)

    def evaluate(self, **overrides):
        args = argparse.Namespace(
            manifest=str(self.evaluation_manifest),
            reference=str(self.index_directory),
            output=str(self.root / "report"),
            executions=str(self.corpus.executions),
            config="",
            thresholds=str(self.root / "thresholds.json"),
            classification_mode="evaluate",
            reference_manifest=str(self.reference_manifest),
            calibration_manifest=str(self.calibration_manifest),
            baseline_coverage=0.1481,
            baseline_selective_risk=0.05,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return evaluate_embeddings_from_cache.run(args, embedder=self.embedder)

    def test_the_index_has_one_centroid_per_label_and_none_for_rejections(self):
        _status, summary = self.build()
        self.assertEqual(sorted(summary["labels"]), ["invoice", "letter", "news article"])
        self.assertNotIn("handwritten", summary["labels"])
        self.assertEqual(summary["example_counts"]["invoice"], 5)
        self.assertTrue((self.index_directory / "embedding_reference.npz").is_file())

    def test_a_label_below_the_minimum_gets_no_centroid_and_is_reported(self):
        """A thin label is skipped and named, not quietly averaged from two pages."""
        self.corpus.sparse_label("reference", "resume", "letter", count=2, start=900)
        _status, summary = self.build(minimum_examples=5)
        self.assertNotIn("resume", summary["labels"])
        self.assertEqual(summary["skipped_labels"], {"resume": 2})
        self.assertIn("invoice", summary["labels"])

    def test_no_label_reaching_the_minimum_is_an_error_not_an_empty_index(self):
        with self.assertRaises(Exception) as error:
            self.build(minimum_examples=99)
        self.assertIn("minimum_examples", str(error.exception))

    def test_calibration_produces_a_calibrated_threshold(self):
        self.build()
        _status, payload = self.calibrate()
        thresholds = payload["thresholds"]
        self.assertEqual(thresholds["calibration_status"], "calibrated")
        self.assertGreaterEqual(payload["development_metrics"]["selective_accuracy"], 0.95)
        self.assertGreater(payload["development_metrics"]["coverage"], 0.0)
        # The rejection targets are declined rather than accepted: on this
        # corpus it is the margin, not the similarity floor, that refuses them.
        self.assertEqual(
            payload["development_metrics"]["rejection_target_false_accept_count"], 0
        )
        self.assertGreater(thresholds["minimum_score_margin"], 0.0)

    def test_calibration_is_deterministic(self):
        self.build()
        _first, first_payload = self.calibrate()
        _second, second_payload = self.calibrate()
        self.assertEqual(first_payload["thresholds"], second_payload["thresholds"])

    def test_an_uncalibratable_split_is_marked_uncalibrated_rather_than_fitted(self):
        """No grid point meeting the target must not become a silent threshold."""
        self.build()
        _status, payload = self.calibrate(target_selective_accuracy=1.01)
        self.assertEqual(payload["thresholds"]["calibration_status"], "uncalibrated")
        self.assertIn("accuracy target", payload["thresholds"]["calibration_note"])

    def test_calibration_refuses_a_split_that_overlaps_the_reference_documents(self):
        self.build()
        with self.assertRaises(LeakageError) as error:
            self.calibrate(manifest=str(self.reference_manifest))
        self.assertIn("reference documents", str(error.exception))

    def test_calibration_refuses_a_split_that_overlaps_the_evaluation(self):
        self.build()
        with self.assertRaises(LeakageError):
            self.calibrate(disjoint_from=[str(self.calibration_manifest)])

    def test_evaluation_refuses_an_evaluation_split_inside_the_index(self):
        self.build()
        self.calibrate()
        with self.assertRaises(LeakageError) as error:
            self.evaluate(manifest=str(self.reference_manifest), reference_manifest="")
        self.assertIn("not a measurement", str(error.exception))

    def test_evaluation_writes_every_artifact(self):
        self.build()
        self.calibrate()
        _status, payload = self.evaluate()
        for key, path in payload["artifacts"].items():
            with self.subTest(artifact=key):
                self.assertTrue(Path(path).is_file(), path)

    def test_the_report_keeps_both_perspectives_apart(self):
        self.build()
        self.calibrate()
        _status, payload = self.evaluate()
        metrics = payload["report"]["metrics"]
        self.assertIn("conventional", metrics)
        self.assertIn("operational", metrics)
        self.assertIn("confusion_matrix", metrics["conventional"])
        self.assertIn("selective_accuracy", metrics["operational"])
        self.assertIn("balanced_accuracy", metrics["conventional"])
        # Timings are reported separately, never as one number.
        self.assertIn("embedding_ms_mean", metrics["timing_ms"])
        self.assertIn("classifier_ms_mean", metrics["timing_ms"])

    def test_the_report_compares_at_the_rules_operating_points(self):
        self.build()
        self.calibrate()
        _status, payload = self.evaluate()
        comparison = payload["report"]["comparison_with_rules_baseline"]
        self.assertEqual(comparison["rules_baseline"]["coverage"], 0.1481)
        self.assertEqual(comparison["rules_baseline"]["selective_risk"], 0.05)
        self.assertIn("embeddings_at_calibrated_threshold", comparison)
        self.assertIn("embeddings_at_rules_coverage", comparison)
        self.assertIn("embeddings_at_rules_selective_risk", comparison)

    def test_predictions_carry_the_versions_and_no_vector(self):
        self.build()
        self.calibrate()
        _status, payload = self.evaluate()
        with Path(payload["artifacts"]["predictions"]).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["taxonomy_version"], "rvl-cdip-2.1")
            self.assertTrue(row["classifier_version"].startswith("embeddings-rvl-cdip-v1+"))
            self.assertNotIn("vector", row)
            self.assertNotIn("embedding_vector", row)

    def test_rejection_targets_are_not_classified_as_a_family(self):
        self.build()
        self.calibrate()
        _status, payload = self.evaluate()
        with Path(payload["artifacts"]["predictions"]).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rejections = [row for row in rows if row["rvl_label"] == "handwritten"]
        self.assertTrue(rejections)
        for row in rejections:
            self.assertEqual(row["predicted_family"], "other")


class MetricSeparationTests(unittest.TestCase):
    """fallback, other and abstained are outcomes; only accepted-wrong is an error."""

    def records(self):
        return [
            {"target_family": "financial_document", "predicted_family": "financial_document", "decision": "classified", "top_candidate": "financial_document", "top_similarity": 0.9, "score_margin": 0.2, "is_rejection_target": False},
            {"target_family": "correspondence", "predicted_family": "other", "decision": "fallback", "top_candidate": "correspondence", "top_similarity": 0.4, "score_margin": 0.01, "is_rejection_target": False},
            {"target_family": "news_article", "predicted_family": "other", "decision": "abstained", "top_candidate": None, "top_similarity": 0.0, "score_margin": 0.0, "is_rejection_target": False},
            {"target_family": "resume", "predicted_family": "correspondence", "decision": "classified", "top_candidate": "correspondence", "top_similarity": 0.8, "score_margin": 0.15, "is_rejection_target": False},
            {"target_family": "other", "predicted_family": "other", "decision": "fallback", "top_candidate": "resume", "top_similarity": 0.3, "score_margin": 0.02, "is_rejection_target": True},
        ]

    def test_only_the_accepted_wrong_answer_is_an_operational_error(self):
        metrics = selective_metrics(self.records())
        self.assertEqual(metrics["accepted_count"], 2)
        self.assertEqual(metrics["coverage"], 0.4)
        self.assertEqual(metrics["operational_error_count"], 1)
        self.assertEqual(metrics["selective_accuracy"], 0.5)
        self.assertEqual(metrics["fallback_count"], 2)
        self.assertEqual(metrics["abstained_count"], 1)

    def test_an_accepted_rejection_target_counts_against_the_system(self):
        record = dict(self.records()[4], decision="classified", predicted_family="resume")
        self.assertTrue(is_accepted_wrong(record))

    def test_macro_averages_skip_families_the_corpus_never_exercises(self):
        """A family with no truth and no prediction must not add a zero."""
        metrics = conventional_metrics(
            self.records(),
            ["financial_document", "correspondence", "news_article", "resume", "other", "form_structured"],
        )
        self.assertNotIn("form_structured", metrics["macro_average_over"])
        self.assertIn("financial_document", metrics["macro_average_over"])
        perfect = [
            {"target_family": "resume", "predicted_family": "resume", "decision": "classified", "top_candidate": "resume", "top_similarity": 0.9, "score_margin": 0.3, "is_rejection_target": False},
        ]
        perfect_metrics = conventional_metrics(perfect, ["resume", "correspondence", "other"])
        self.assertEqual(perfect_metrics["accuracy"], 1.0)
        self.assertEqual(perfect_metrics["macro_f1"], 1.0)

    def test_an_operating_point_reports_how_far_it_missed_the_target(self):
        curve = risk_coverage_curve(self.records())
        point = operating_point_at_coverage(curve, 0.1481)
        self.assertIn("requested_coverage", point)
        self.assertIn("coverage_gap", point)
        self.assertGreaterEqual(point["coverage_gap"], 0.0)

    def test_the_conventional_view_counts_a_refusal_as_other(self):
        metrics = conventional_metrics(self.records(), ["financial_document", "correspondence", "news_article", "resume", "other"])
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["confusion_matrix"]["correspondence"]["other"], 1)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("weighted_f1", metrics)

    def test_the_risk_coverage_curve_spans_from_full_to_no_coverage(self):
        curve = risk_coverage_curve(self.records())
        coverages = [point["coverage"] for point in curve]
        self.assertAlmostEqual(max(coverages), 0.8)  # the abstention never accepts
        self.assertEqual(min(coverages), 0.0)
        self.assertEqual(curve, sorted(curve, key=lambda point: point["similarity_threshold"]))

    def test_operating_points_are_selectable_by_coverage_and_by_risk(self):
        curve = risk_coverage_curve(self.records())
        at_coverage = operating_point_at_coverage(curve, 0.4)
        self.assertIsNotNone(at_coverage)
        self.assertLessEqual(abs(at_coverage["coverage"] - 0.4), 0.4)
        at_risk = operating_point_at_risk(curve, 0.0)
        self.assertIsNotNone(at_risk)
        self.assertEqual(at_risk["selective_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
