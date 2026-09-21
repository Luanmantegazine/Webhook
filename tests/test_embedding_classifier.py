"""The embedding baseline's own machinery: chunking, pooling, similarity, decision.

Every test here runs offline against a deterministic fake embedder. Nothing in
this module imports ``sentence_transformers``, opens a socket or touches a
model: a unit suite that downloads 1.1 GB of weights is not a unit suite, and
one that silently falls back to a stub when the download fails is worse.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_doubles import (  # noqa: E402
    DirectionalEmbedder,
    FakeEmbedder,
    FakeTokenizer,
    feature_record,
)
from tasks.document.embedding_classifier_core import (  # noqa: E402
    CANDIDATE_LABELS,
    EmbeddingClassifierConfig,
    EmbeddingContractError,
    EmbeddingModelConfig,
    EmbeddingReferenceIndex,
    REJECTION_LABELS,
    aggregate_family_scores,
    apply_decision_policy,
    build_reference_index,
    chunk_classification_text,
    classify_features_with_embeddings,
    classify_with_embeddings,
    cosine_similarity,
    embed_document,
    encode_tokens,
    l2_normalize,
    normalise_classification_text,
    rank_families,
)


def small_model_config(**overrides) -> EmbeddingModelConfig:
    base = {"max_tokens": 10, "overlap_tokens": 4, "model_name": "fake/encoder"}
    base.update(overrides)
    return EmbeddingModelConfig(**base)


def reference_index(embedder, *, labels_to_texts, model_config=None) -> EmbeddingReferenceIndex:
    config = model_config or small_model_config()
    vectors = {
        label: np.vstack(
            [embed_document(text, embedder, config)[0] for text in texts]
        )
        for label, texts in labels_to_texts.items()
    }
    return build_reference_index(vectors, model_config=config, minimum_examples=1)


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_a_short_document_is_one_chunk(self):
        chunks = chunk_classification_text(
            "invoice total amount due", self.tokenizer, max_tokens=10, overlap_tokens=4
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "invoice total amount due")

    def test_a_long_document_is_several_chunks_in_order(self):
        text = " ".join(f"token{index}" for index in range(25))
        chunks = chunk_classification_text(
            text, self.tokenizer, max_tokens=10, overlap_tokens=4
        )
        self.assertGreater(len(chunks), 1)
        # Order preserved: the first chunk starts at the start, the last ends
        # at the end. A silent truncation would drop the tail entirely.
        self.assertTrue(chunks[0].startswith("token0"))
        self.assertTrue(chunks[-1].endswith("token24"))

    def test_no_token_is_lost_between_chunks(self):
        text = " ".join(f"token{index}" for index in range(47))
        chunks = chunk_classification_text(
            text, self.tokenizer, max_tokens=10, overlap_tokens=4
        )
        seen = []
        for chunk in chunks:
            seen.extend(chunk.split())
        self.assertEqual(set(seen), {f"token{index}" for index in range(47)})

    def test_consecutive_chunks_share_exactly_the_overlap(self):
        text = " ".join(f"token{index}" for index in range(40))
        chunks = chunk_classification_text(
            text, self.tokenizer, max_tokens=10, overlap_tokens=4
        )
        for first, second in zip(chunks, chunks[1:]):
            tail = first.split()[-4:]
            head = second.split()[:4]
            self.assertEqual(tail, head)

    def test_chunking_is_deterministic(self):
        text = " ".join(f"token{index}" for index in range(33))
        first = chunk_classification_text(text, self.tokenizer, max_tokens=10, overlap_tokens=4)
        second = chunk_classification_text(text, FakeTokenizer(), max_tokens=10, overlap_tokens=4)
        self.assertEqual(first, second)

    def test_an_overlap_at_or_above_the_window_is_refused(self):
        for overlap in (10, 11, 50):
            with self.subTest(overlap=overlap):
                with self.assertRaises(EmbeddingContractError):
                    chunk_classification_text(
                        "a b c", self.tokenizer, max_tokens=10, overlap_tokens=overlap
                    )

    def test_a_negative_overlap_or_empty_window_is_refused(self):
        with self.assertRaises(EmbeddingContractError):
            chunk_classification_text("a b", self.tokenizer, max_tokens=10, overlap_tokens=-1)
        with self.assertRaises(EmbeddingContractError):
            chunk_classification_text("a b", self.tokenizer, max_tokens=0, overlap_tokens=0)

    def test_no_chunk_is_empty(self):
        text = "\n\n\n   invoice   \n\n\n total \n\n"
        chunks = chunk_classification_text(text, self.tokenizer, max_tokens=4, overlap_tokens=1)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_normalisation_keeps_titles_tables_and_punctuation(self):
        text = "INVOICE #42\n\n\nItem | Qty | Price\nWidget |  2 | $3.00"
        normalised = normalise_classification_text(text)
        self.assertIn("INVOICE #42", normalised)
        self.assertIn("Item | Qty | Price", normalised)
        self.assertIn("$3.00", normalised)
        self.assertNotIn("\n\n\n", normalised)


class PoolingTests(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder(dimension=8)
        self.config = small_model_config()

    def test_the_document_vector_is_the_normalised_mean_of_normalised_chunks(self):
        text = " ".join(f"token{index}" for index in range(30))
        vector, metadata = embed_document(text, self.embedder, self.config)
        chunks = chunk_classification_text(
            text, self.embedder.tokenizer, max_tokens=10, overlap_tokens=4
        )
        expected_chunks = np.vstack(
            [self.embedder._vector(f"{self.config.text_prefix}{chunk}") for chunk in chunks]
        )
        expected = l2_normalize(l2_normalize(expected_chunks).mean(axis=0))
        np.testing.assert_allclose(vector, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(metadata["chunk_count"], len(chunks))
        self.assertEqual(metadata["pooling"], "normalized_chunk_mean")

    def test_the_document_vector_is_unit_length(self):
        vector, _metadata = embed_document("invoice total due", self.embedder, self.config)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)

    def test_the_same_prefix_is_applied_to_every_chunk(self):
        embed_document("alpha beta gamma delta", self.embedder, self.config)
        batch = self.embedder.encoded_batches[-1]
        for text in batch:
            self.assertTrue(text.startswith("query: "))

    def test_metadata_reports_chunks_tokens_and_dimension(self):
        text = " ".join(f"token{index}" for index in range(30))
        _vector, metadata = embed_document(text, self.embedder, self.config)
        self.assertEqual(metadata["embedding_dimension"], 8)
        self.assertEqual(metadata["token_count"], 30)
        self.assertGreater(metadata["chunk_count"], 1)

    def test_a_zero_vector_normalises_to_zero_rather_than_nan(self):
        normalised = l2_normalize(np.zeros(4))
        self.assertTrue(np.all(np.isfinite(normalised)))
        self.assertEqual(float(np.linalg.norm(normalised)), 0.0)


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_of_identical_directions_is_one(self):
        vector = np.array([0.3, 0.4, 0.5])
        similarity = cosine_similarity(vector, np.vstack([vector, -vector]))
        self.assertAlmostEqual(float(similarity[0]), 1.0, places=12)
        self.assertAlmostEqual(float(similarity[1]), -1.0, places=12)

    def test_cosine_similarity_ignores_magnitude(self):
        vector = np.array([1.0, 0.0, 0.0])
        similarity = cosine_similarity(vector, np.vstack([vector * 17.0]))
        self.assertAlmostEqual(float(similarity[0]), 1.0, places=12)

    def test_a_family_scores_the_strongest_of_its_labels(self):
        scores = {"email": 0.40, "letter": 0.81, "memo": 0.20, "invoice": 0.55}
        families, winners = aggregate_family_scores(scores)
        # 0.81, not the 0.47 mean of the three correspondence labels: averaging
        # a heterogeneous family is how it stops winning anything.
        self.assertAlmostEqual(families["correspondence"], 0.81)
        self.assertEqual(winners["correspondence"], "letter")
        self.assertAlmostEqual(families["financial_document"], 0.55)

    def test_ranking_breaks_ties_by_family_name(self):
        ranked = rank_families({"resume": 0.5, "correspondence": 0.5, "news_article": 0.5})
        self.assertEqual([family for family, _ in ranked], ["correspondence", "news_article", "resume"])

    def test_a_tie_between_labels_of_one_family_is_broken_by_label_name(self):
        _families, winners = aggregate_family_scores({"email": 0.7, "letter": 0.7, "memo": 0.7})
        self.assertEqual(winners["correspondence"], "email")


class DecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.ranked = [("financial_document", 0.82), ("correspondence", 0.70)]

    def test_above_threshold_and_margin_classifies(self):
        config = EmbeddingClassifierConfig(similarity_threshold=0.75, min_score_margin=0.05)
        decision = apply_decision_policy(self.ranked, 500, config)
        self.assertEqual(decision["decision"], "classified")
        self.assertEqual(decision["document_family"], "financial_document")
        self.assertEqual(decision["reason"], "similarity_above_threshold")

    def test_below_threshold_falls_back_to_other(self):
        config = EmbeddingClassifierConfig(similarity_threshold=0.90)
        decision = apply_decision_policy(self.ranked, 500, config)
        self.assertEqual(decision["decision"], "fallback")
        self.assertEqual(decision["document_family"], "other")
        self.assertEqual(decision["reason"], "similarity_below_threshold")

    def test_below_margin_falls_back_to_other(self):
        config = EmbeddingClassifierConfig(similarity_threshold=0.50, min_score_margin=0.20)
        decision = apply_decision_policy(self.ranked, 500, config)
        self.assertEqual(decision["decision"], "fallback")
        self.assertEqual(decision["reason"], "score_margin_below_minimum")

    def test_insufficient_text_abstains(self):
        config = EmbeddingClassifierConfig(min_recognized_characters=20)
        decision = apply_decision_policy(self.ranked, 19, config)
        self.assertEqual(decision, {
            "document_family": "other",
            "decision": "abstained",
            "reason": "insufficient_ocr_text",
        })

    def test_a_family_threshold_overrides_the_global_one(self):
        config = EmbeddingClassifierConfig(
            similarity_threshold=0.50,
            family_similarity_thresholds={"financial_document": 0.90},
        )
        self.assertEqual(
            apply_decision_policy(self.ranked, 500, config)["decision"], "fallback"
        )
        self.assertEqual(
            apply_decision_policy(
                [("correspondence", 0.60), ("resume", 0.10)], 500, config
            )["decision"],
            "classified",
        )

    def test_no_class_is_ever_forced(self):
        """Every refusal lands on other; nothing promotes a runner-up."""
        config = EmbeddingClassifierConfig(similarity_threshold=0.99)
        for ranked in ([("resume", 0.98)], [("resume", 0.5), ("memo", 0.5)], []):
            with self.subTest(ranked=ranked):
                decision = apply_decision_policy(ranked, 500, config)
                self.assertEqual(decision["document_family"], "other")
                self.assertIn(decision["decision"], {"fallback", "abstained"})


class RejectionLabelTests(unittest.TestCase):
    def test_rejection_labels_have_no_centroid_slot(self):
        for label in REJECTION_LABELS:
            self.assertNotIn(label, CANDIDATE_LABELS)

    def test_building_an_index_with_a_rejection_label_is_refused(self):
        embedder = FakeEmbedder()
        config = small_model_config()
        vector, _metadata = embed_document("handwritten note", embedder, config)
        with self.assertRaises(EmbeddingContractError) as error:
            build_reference_index(
                {"handwritten": np.vstack([vector])}, model_config=config
            )
        self.assertIn("negatives", str(error.exception))

    def test_other_is_never_a_predicted_class_with_a_prototype(self):
        embedder = FakeEmbedder()
        index = reference_index(
            embedder,
            labels_to_texts={
                "invoice": ["invoice total amount due subtotal tax"],
                "letter": ["dear sir yours sincerely regards"],
            },
        )
        self.assertNotIn("other", [family for _label, family in index.label_family_pairs()])


class EndToEndTests(unittest.TestCase):
    """Three families with planted structure: the integration check."""

    #: Long enough to clear the 20-character text floor, which is a decision of
    #: the classifier and not something a fixture should sidestep.
    BODY = "page of running text with several ordinary words on it, number"

    @classmethod
    def document(cls, marker: str, index: int) -> str:
        return f"{marker} {cls.BODY} {index}"

    def setUp(self):
        self.embedder = DirectionalEmbedder(
            {"invoice": 0, "letter": 1, "newspaper": 2}, dimension=8
        )
        self.config = small_model_config()
        self.index = reference_index(
            self.embedder,
            labels_to_texts={
                "invoice": [self.document("invoice", index) for index in range(4)],
                "letter": [self.document("letter", index) for index in range(4)],
                "news article": [self.document("newspaper", index) for index in range(4)],
            },
            model_config=self.config,
        )

    def classify(self, text, **config_kwargs):
        record = feature_record(text)
        return classify_features_with_embeddings(
            record,
            self.embedder,
            self.index,
            EmbeddingClassifierConfig(**config_kwargs),
            self.config,
        )

    def test_each_family_is_recovered(self):
        for text, family in (
            (self.document("invoice", 99), "financial_document"),
            (self.document("letter", 99), "correspondence"),
            (self.document("newspaper", 99), "news_article"),
        ):
            with self.subTest(family=family):
                result = self.classify(text, similarity_threshold=0.5)
                self.assertEqual(result["document_family"], family)
                self.assertEqual(result["decision"], "classified")

    def test_an_unlike_document_falls_back_rather_than_being_forced(self):
        result = self.classify("nothing resembling any prototype in this reference index at all", similarity_threshold=0.9)
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["decision"], "fallback")
        self.assertEqual(result["reason"], "similarity_below_threshold")

    def test_a_document_with_no_text_abstains(self):
        record = feature_record("x", alnum_character_count=3)
        result = classify_features_with_embeddings(
            record, self.embedder, self.index, EmbeddingClassifierConfig(), self.config
        )
        self.assertEqual(result["decision"], "abstained")
        self.assertEqual(result["reason"], "insufficient_ocr_text")
        self.assertEqual(result["document_family"], "other")
        self.assertEqual(result["embedding"]["chunk_count"], 0)

    def test_the_result_reports_similarity_not_probability(self):
        result = self.classify(self.document("invoice", 99), similarity_threshold=0.5)
        self.assertEqual(result["confidence_kind"], "cosine_similarity_not_probability")
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertIn("top_similarity", result)

    def test_the_result_never_carries_the_vector(self):
        result = self.classify(self.document("invoice", 99), similarity_threshold=0.5)
        serialised = repr(result)
        self.assertNotIn("vector", result)
        self.assertNotIn("document_embedding", result)
        self.assertLess(len(serialised), 8000)

    def test_nearest_prototypes_are_labels_with_their_families(self):
        result = self.classify(self.document("invoice", 99), similarity_threshold=0.5)
        nearest = result["nearest_prototypes"][0]
        self.assertEqual(nearest["label"], "invoice")
        self.assertEqual(nearest["family"], "financial_document")
        self.assertIn("example_count", nearest)


if __name__ == "__main__":
    unittest.main()
