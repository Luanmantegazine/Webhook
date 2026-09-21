"""Contracts: versions, fingerprints, index persistence, wrappers, isolation.

Two of these matter more than the rest. ``NoModelDownloadTests`` asserts that
importing and exercising this code never reaches for a model or the network —
if that ever regresses, every other test in the suite becomes a network test.
``RulesClassifierUntouchedTests`` asserts that the embedding work changed
nothing about the v12 rules baseline it is measured against.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_doubles import FakeEmbedder, feature_record  # noqa: E402
from tasks.document import embedding_classifier_core as core  # noqa: E402
from tasks.document.embedding_classifier_core import (  # noqa: E402
    CANDIDATE_FAMILIES,
    CANDIDATE_LABELS,
    EMBEDDING_CLASSIFIER_VERSION,
    EMBEDDING_EXTRACTION_VERSION,
    EMBEDDING_FINGERPRINT,
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingClassifierConfig,
    EmbeddingContractError,
    EmbeddingModelConfig,
    EmbeddingReferenceIndex,
    build_reference_index,
    classify_features_with_embeddings,
    classify_with_embeddings,
    embed_document,
    embedding_fingerprint,
    load_classifier_config,
    reference_fingerprint,
)
from tasks.document.rvl_cdip_eval import TAXONOMY_VERSION  # noqa: E402


def model_config(**overrides) -> EmbeddingModelConfig:
    base = {"max_tokens": 10, "overlap_tokens": 4, "model_name": "fake/encoder"}
    base.update(overrides)
    return EmbeddingModelConfig(**base)


def build_index(embedder=None, config=None, labels=("invoice", "letter")) -> EmbeddingReferenceIndex:
    encoder = embedder or FakeEmbedder(dimension=8)
    configuration = config or model_config()
    vectors = {
        label: np.vstack(
            [
                embed_document(f"{label} reference document number {index} body text", encoder, configuration)[0]
                for index in range(3)
            ]
        )
        for label in labels
    }
    return build_reference_index(vectors, model_config=configuration, minimum_examples=1)


class VersionContractTests(unittest.TestCase):
    def test_the_embedding_classifier_has_its_own_versions(self):
        self.assertEqual(EMBEDDING_SCHEMA_VERSION, "2.1")
        self.assertEqual(TAXONOMY_VERSION, "rvl-cdip-2.1")
        self.assertEqual(EMBEDDING_EXTRACTION_VERSION, "1.0")
        self.assertTrue(EMBEDDING_CLASSIFIER_VERSION.startswith("embeddings-rvl-cdip-v1+"))
        self.assertTrue(EMBEDDING_CLASSIFIER_VERSION.endswith(EMBEDDING_FINGERPRINT))

    def test_the_embedding_version_is_not_a_rules_version(self):
        """Two classifiers, two identities; a shared string would pool them."""
        from tasks.document.rules_classifier_core import CLASSIFIER_VERSION, RULE_FINGERPRINT

        self.assertNotEqual(EMBEDDING_CLASSIFIER_VERSION, CLASSIFIER_VERSION)
        self.assertNotEqual(EMBEDDING_FINGERPRINT, RULE_FINGERPRINT)
        self.assertNotIn("rules", EMBEDDING_CLASSIFIER_VERSION)

    def test_the_result_contract_carries_every_identifier(self):
        index = build_index()
        result = classify_features_with_embeddings(
            feature_record("invoice reference document number 9 body text with more words"),
            FakeEmbedder(dimension=8),
            index,
            EmbeddingClassifierConfig(),
            model_config(),
        )
        for key in (
            "schema_version",
            "taxonomy_version",
            "classifier_version",
            "classifier",
            "mode",
            "document_family",
            "document_subtype",
            "confidence",
            "confidence_kind",
            "decision",
            "reason",
            "top_candidate",
            "runner_up",
            "top_similarity",
            "score_margin",
            "candidate_scores",
            "nearest_prototypes",
            "embedding",
            "thresholds",
            "recommended_template",
            "fallback_template",
            "execution_time_ms",
        ):
            self.assertIn(key, result, key)
        self.assertEqual(result["classifier"], "embeddings")
        self.assertIsNone(result["recommended_template"])
        self.assertEqual(result["fallback_template"], "clean_article")
        self.assertEqual(result["thresholds"]["calibration_status"], "uncalibrated")


class FingerprintTests(unittest.TestCase):
    def test_the_embedding_fingerprint_is_stable_within_a_process(self):
        self.assertEqual(embedding_fingerprint(), embedding_fingerprint())

    def test_the_embedding_fingerprint_is_stable_across_processes(self):
        """No memory addresses: two interpreters must agree."""
        import subprocess

        command = [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r);"
            "from tasks.document.embedding_classifier_core import EMBEDDING_FINGERPRINT;"
            "print(EMBEDDING_FINGERPRINT)" % str(ROOT),
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        second = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(first, second)
        self.assertEqual(first, EMBEDDING_FINGERPRINT)

    def test_the_fingerprint_moves_with_the_model(self):
        self.assertNotEqual(
            embedding_fingerprint(model_config(model_name="other/encoder")),
            embedding_fingerprint(model_config()),
        )

    def test_the_fingerprint_moves_with_chunking_and_pooling_inputs(self):
        baseline = embedding_fingerprint(model_config())
        self.assertNotEqual(baseline, embedding_fingerprint(model_config(max_tokens=256)))
        self.assertNotEqual(baseline, embedding_fingerprint(model_config(overlap_tokens=8)))
        self.assertNotEqual(
            baseline, embedding_fingerprint(model_config(text_prefix="passage: "))
        )

    def test_the_reference_fingerprint_moves_with_the_vectors(self):
        index = build_index()
        altered = np.array(index.centroids, copy=True)
        altered[0][0] += 0.25
        self.assertNotEqual(
            index.reference_fingerprint,
            reference_fingerprint(
                labels=index.labels,
                centroids=altered,
                example_counts=index.example_counts,
                metadata=index.metadata,
            ),
        )


class ReferenceIndexPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))
        self.index = build_index()

    def test_round_trip_preserves_labels_counts_and_vectors(self):
        self.index.save(self.directory)
        loaded = EmbeddingReferenceIndex.load(self.directory)
        self.assertEqual(loaded.labels, self.index.labels)
        self.assertEqual(loaded.example_counts, self.index.example_counts)
        np.testing.assert_allclose(loaded.centroids, self.index.centroids, atol=1e-6)
        self.assertEqual(loaded.reference_fingerprint, self.index.reference_fingerprint)

    def test_the_npz_holds_only_numeric_arrays_and_loads_without_pickle(self):
        self.index.save(self.directory)
        with np.load(self.directory / "embedding_reference.npz", allow_pickle=False) as payload:
            self.assertEqual(sorted(payload.files), ["centroids", "example_counts"])
            for name in payload.files:
                self.assertIn(payload[name].dtype.kind, "fiu", name)

    def test_no_vector_is_written_to_json(self):
        self.index.save(self.directory)
        metadata = json.loads(
            (self.directory / "embedding_reference.metadata.json").read_text("utf-8")
        )
        for value in metadata.values():
            self.assertNotIsInstance(value, list) if isinstance(value, float) else None
        self.assertNotIn("centroids", metadata)
        self.assertNotIn("vectors", metadata)

    def test_an_edited_index_is_refused_on_load(self):
        self.index.save(self.directory)
        path = self.directory / "embedding_reference.metadata.json"
        metadata = json.loads(path.read_text("utf-8"))
        metadata["model"] = "someone/else"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(EmbeddingContractError) as error:
            EmbeddingReferenceIndex.load(self.directory)
        self.assertIn("reference_fingerprint", str(error.exception))

    def test_a_missing_artifact_is_refused(self):
        self.index.save(self.directory)
        (self.directory / "embedding_reference.npz").unlink()
        with self.assertRaises(EmbeddingContractError):
            EmbeddingReferenceIndex.load(self.directory)


class IndexCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_a_different_model_is_refused(self):
        with self.assertRaises(EmbeddingContractError) as error:
            self.index.ensure_compatible(model_config(model_name="other/encoder"))
        self.assertIn("model", str(error.exception))

    def test_a_different_revision_is_refused(self):
        index = build_index(config=model_config(model_revision="aaaa"))
        with self.assertRaises(EmbeddingContractError):
            index.ensure_compatible(model_config(model_revision="bbbb"))

    def test_a_different_prefix_or_chunking_is_refused(self):
        for override in (
            {"text_prefix": "passage: "},
            {"max_tokens": 32},
            {"overlap_tokens": 2},
        ):
            with self.subTest(override=override):
                with self.assertRaises(EmbeddingContractError):
                    self.index.ensure_compatible(model_config(**override))

    def test_a_dimension_mismatch_is_refused(self):
        with self.assertRaises(EmbeddingContractError) as error:
            self.index.ensure_compatible(model_config(), document_dimension=384)
        self.assertIn("dimension", str(error.exception))

    def test_classifying_a_vector_of_the_wrong_width_is_refused(self):
        with self.assertRaises(EmbeddingContractError):
            classify_with_embeddings(np.zeros(3), self.index, EmbeddingClassifierConfig())

    def test_a_stale_taxonomy_is_refused(self):
        stale = EmbeddingReferenceIndex(
            labels=self.index.labels,
            centroids=self.index.centroids,
            example_counts=self.index.example_counts,
            metadata={**self.index.metadata, "taxonomy_version": "rvl-cdip-1.0"},
        )
        with self.assertRaises(EmbeddingContractError) as error:
            stale.ensure_compatible(model_config())
        self.assertIn("taxonomy", str(error.exception))

    def test_a_stale_extraction_version_is_refused(self):
        stale = EmbeddingReferenceIndex(
            labels=self.index.labels,
            centroids=self.index.centroids,
            example_counts=self.index.example_counts,
            metadata={**self.index.metadata, "embedding_extraction_version": "0.9"},
        )
        with self.assertRaises(EmbeddingContractError):
            stale.ensure_compatible(model_config())


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index()
        self.embedder = FakeEmbedder(dimension=8)

    def classify(self, record):
        return classify_features_with_embeddings(
            record, self.embedder, self.index, EmbeddingClassifierConfig(), model_config()
        )

    def test_a_foreign_schema_version_is_refused(self):
        record = feature_record("invoice body text long enough to classify", schema_version="1.9")
        with self.assertRaises(EmbeddingContractError) as error:
            self.classify(record)
        self.assertIn("schema_version", str(error.exception))

    def test_a_foreign_taxonomy_version_is_refused(self):
        record = feature_record("invoice body text long enough", taxonomy_version="rvl-cdip-2.0")
        with self.assertRaises(EmbeddingContractError):
            self.classify(record)

    def test_a_record_without_classification_text_is_refused(self):
        record = feature_record("invoice body text long enough")
        del record["classification_text"]
        with self.assertRaises(EmbeddingContractError) as error:
            self.classify(record)
        self.assertIn("does not re-run OCR", str(error.exception))

    def test_a_record_without_provenance_is_refused(self):
        record = feature_record("invoice body text long enough")
        record["provenance"] = None
        with self.assertRaises(EmbeddingContractError):
            self.classify(record)

    def test_insufficient_text_abstains_rather_than_raising(self):
        record = feature_record("ab", alnum_character_count=2)
        result = self.classify(record)
        self.assertEqual(
            (result["document_family"], result["decision"], result["reason"]),
            ("other", "abstained", "insufficient_ocr_text"),
        )


class ConfigurationTests(unittest.TestCase):
    def test_the_shipped_config_declares_the_documented_defaults(self):
        model, classifier = load_classifier_config()
        self.assertEqual(model.model_name, "intfloat/multilingual-e5-base")
        self.assertEqual(model.text_prefix, "query: ")
        self.assertEqual(model.max_tokens, 384)
        self.assertEqual(model.overlap_tokens, 64)
        self.assertEqual(model.pooling, "normalized_chunk_mean")
        self.assertEqual(model.device_order, ("cuda", "mps", "cpu"))
        self.assertEqual(classifier.calibration_status, "uncalibrated")

    def test_the_config_path_is_relative_to_the_repository(self):
        self.assertTrue(str(core.DEFAULT_CONFIG_PATH).endswith(
            "config/document_embedding_classifier.json"
        ))
        self.assertNotIn("/home/", str(core.DEFAULT_CONFIG_PATH.relative_to(ROOT)))

    def test_the_config_does_not_restate_the_taxonomy(self):
        payload = json.loads(core.DEFAULT_CONFIG_PATH.read_text("utf-8"))
        for forbidden in ("label_mapping", "families", "taxonomy", "class_to_family"):
            self.assertNotIn(forbidden, payload)

    def test_a_family_threshold_for_an_unknown_family_is_refused(self):
        with self.assertRaises(EmbeddingContractError):
            EmbeddingClassifierConfig(family_similarity_thresholds={"not_a_family": 0.5})

    def test_candidate_families_come_from_the_taxonomy(self):
        from tasks.document.rvl_cdip_eval import CLASS_TO_FAMILY, REJECTION_LABELS

        expected = sorted(
            {family for label, family in CLASS_TO_FAMILY.items() if label not in REJECTION_LABELS}
        )
        self.assertEqual(list(CANDIDATE_FAMILIES), expected)
        self.assertEqual(
            sorted(CANDIDATE_LABELS),
            sorted(label for label in CLASS_TO_FAMILY if label not in REJECTION_LABELS),
        )


class NoModelDownloadTests(unittest.TestCase):
    """The suite must not reach for a model, an import or the network."""

    def test_sentence_transformers_is_not_imported_by_this_module(self):
        self.assertNotIn("sentence_transformers", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_classifying_never_constructs_a_real_encoder(self):
        index = build_index()
        with mock.patch.object(
            core.SentenceTransformerEmbedder, "load", side_effect=AssertionError("loaded a model")
        ):
            result = classify_features_with_embeddings(
                feature_record("invoice reference document number 9 body text words"),
                FakeEmbedder(dimension=8),
                index,
                EmbeddingClassifierConfig(),
                model_config(),
            )
        self.assertIn(result["decision"], {"classified", "fallback", "abstained"})

    def test_the_real_embedder_only_imports_on_load(self):
        embedder = core.SentenceTransformerEmbedder(model_config())
        self.assertIsNone(embedder._model)
        self.assertNotIn("sentence_transformers", sys.modules)


class TaskWrapperContractTests(unittest.TestCase):
    """Single-output wrappers return a dict, never a one-element tuple."""

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
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))
        self.index = build_index()
        self.index.save(self.directory)

    def test_the_embedding_wrapper_returns_a_dict(self):
        from tasks.document.extract_document_embedding import extract_document_embedding

        with mock.patch.object(
            core.SentenceTransformerEmbedder, "load", autospec=True
        ) as load:
            fake = FakeEmbedder(dimension=8)
            load.side_effect = lambda self: _as_fake(self, fake)
            output = extract_document_embedding(
                feature_record("invoice reference document number 9 body text words"),
                model_name="fake/encoder",
                max_tokens=10,
                overlap_tokens=4,
            )
        self.assertIsInstance(output, dict)
        self.assertNotIsInstance(output, tuple)
        self.assertTrue(output["embedded"])
        self.assertEqual(len(output["vector"]), 8)

    def test_the_embedding_wrapper_declines_unreadable_text_without_a_model(self):
        from tasks.document.extract_document_embedding import extract_document_embedding

        output = extract_document_embedding(feature_record("ab", alnum_character_count=2))
        self.assertIsInstance(output, dict)
        self.assertFalse(output["embedded"])
        self.assertEqual(output["reason"], "insufficient_ocr_text")
        self.assertEqual(output["vector"], [])

    def test_the_classification_wrapper_returns_a_dict(self):
        from tasks.document.classify_document_embeddings import classify_document_embeddings

        embedder = FakeEmbedder(dimension=8)
        vector, metadata = embed_document(
            "invoice reference document number 9 body text words", embedder, model_config()
        )
        result = classify_document_embeddings(
            {
                "vector": [float(value) for value in vector],
                "embedded": True,
                "recognized_characters": 400,
                "metadata": metadata,
            },
            reference_index_path=str(self.directory),
        )
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, tuple)
        self.assertEqual(result["classifier_version"], EMBEDDING_CLASSIFIER_VERSION)
        self.assertIn("execution_time_ms", result)

    def test_the_classification_wrapper_abstains_on_an_unembedded_document(self):
        from tasks.document.classify_document_embeddings import classify_document_embeddings

        result = classify_document_embeddings(
            {"vector": [], "embedded": False, "recognized_characters": 3, "metadata": {}},
            reference_index_path=str(self.directory),
        )
        self.assertEqual(result["decision"], "abstained")
        self.assertEqual(result["document_family"], "other")


def _as_fake(embedder, fake):
    embedder._model = fake
    embedder._device = "cpu"
    embedder._resolved_revision = None
    type(embedder).tokenizer = property(lambda self: fake.tokenizer)
    type(embedder).encode = lambda self, texts: fake.encode(texts)
    return embedder


class RulesClassifierUntouchedTests(unittest.TestCase):
    """The v12 baseline this work is measured against must be untouched."""

    def test_the_rules_classifier_still_reports_v12(self):
        from tasks.document.rules_classifier_core import (
            CLASSIFIER_VERSION,
            FEATURE_EXTRACTION_VERSION,
            RULE_FINGERPRINT,
            SCHEMA_VERSION,
        )

        self.assertEqual(CLASSIFIER_VERSION, "rules-rvl-cdip-v12+0605c7dd2e78")
        self.assertEqual(RULE_FINGERPRINT, "0605c7dd2e78")
        self.assertEqual(SCHEMA_VERSION, "2.1")
        self.assertEqual(FEATURE_EXTRACTION_VERSION, "2.6")

    def test_the_embedding_module_does_not_import_the_rules_classifier(self):
        """Prose may mention it; code may not depend on it."""
        import ast

        tree = ast.parse(
            (ROOT / "tasks" / "document" / "embedding_classifier_core.py").read_text("utf-8")
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertNotIn("tasks.document.rules_classifier_core", imported)
        self.assertIn("tasks.document.rvl_cdip_eval", imported)

    def test_both_classifiers_read_one_taxonomy(self):
        from tasks.document import rules_classifier_core as rules

        self.assertEqual(core.TAXONOMY_VERSION, rules.TAXONOMY_VERSION)
        self.assertIs(core.CLASS_TO_FAMILY, __import__(
            "tasks.document.rvl_cdip_eval", fromlist=["CLASS_TO_FAMILY"]
        ).CLASS_TO_FAMILY)


if __name__ == "__main__":
    unittest.main()
