"""Optional: the real encoder, off by default.

This is the only module that may download weights, and it refuses to run
unless it is asked for explicitly:

    HYDRA_EMBEDDING_REAL_MODEL=1 python -m unittest tests.test_embedding_real_model -v

Without that variable every test here skips, so ``python -m unittest discover
-s tests`` stays offline. What it checks is the handful of things a fake
embedder cannot: that the real tokenizer chunks a long document into several
windows, that the encoder produces the dimension the config declares, and that
a document embeds to a unit vector.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.document.embedding_classifier_core import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSION,
    EmbeddingModelConfig,
    SentenceTransformerEmbedder,
    chunk_classification_text,
    embed_document,
    load_classifier_config,
)

ENABLED = os.environ.get("HYDRA_EMBEDDING_REAL_MODEL", "").strip() not in ("", "0", "false")


@unittest.skipUnless(ENABLED, "set HYDRA_EMBEDDING_REAL_MODEL=1 to run against real weights")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_config, _classifier = load_classifier_config()
        cls.embedder = SentenceTransformerEmbedder(cls.model_config).load()

    def test_the_device_is_one_of_the_declared_order(self):
        self.assertIn(self.embedder.device, self.model_config.device_order)

    def test_a_long_document_chunks_into_several_windows(self):
        text = " ".join(f"palavra{index}" for index in range(4000))
        chunks = chunk_classification_text(
            text,
            self.embedder.tokenizer,
            max_tokens=self.model_config.max_tokens,
            overlap_tokens=self.model_config.overlap_tokens,
        )
        self.assertGreater(len(chunks), 1)

    def test_the_document_vector_has_the_declared_dimension_and_unit_norm(self):
        vector, metadata = embed_document(
            "Fatura número 42. Total a pagar R$ 1.234,00. Vencimento em 10/03/2026.",
            self.embedder,
            self.model_config,
        )
        self.assertEqual(vector.shape[0], DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(metadata["embedding_dimension"], DEFAULT_EMBEDDING_DIMENSION)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_the_loaded_revision_is_recorded(self):
        self.assertTrue(
            self.embedder.revision is None or isinstance(self.embedder.revision, str)
        )


if __name__ == "__main__":
    unittest.main()
