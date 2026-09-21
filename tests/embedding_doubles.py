"""Deterministic test doubles for the embedding classifier.

No network, no model download, no ``sentence_transformers`` import. Every
vector here is a function of the text, so a test that passes once passes for a
reason rather than by luck.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

WORD = re.compile(r"\w+|[^\w\s]")


class FakeTokenizer:
    """Word-level tokenizer with a stable vocabulary built on first sight.

    Real tokenizers map ids to sub-words; for chunking what matters is that
    ``encode`` and ``decode`` round-trip and that ids are stable, both of which
    this satisfies without loading anything.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def _id_of(self, token: str) -> int:
        if token not in self._vocabulary:
            index = len(self._vocabulary) + 1
            self._vocabulary[token] = index
            self._inverse[index] = token
        return self._vocabulary[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id_of(token) for token in WORD.findall(str(text))]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._inverse.get(int(value), "") for value in ids).strip()


class FakeEmbedder:
    """Hash-based embedder: same text in, same vector out, always.

    Vectors are *not* normalised on the way out, deliberately: the module under
    test is responsible for normalising, and a double that pre-normalises would
    hide a missing normalisation step.
    """

    def __init__(self, dimension: int = 8, scale: float = 3.0) -> None:
        self.dimension = int(dimension)
        self.scale = float(scale)
        self.tokenizer = FakeTokenizer()
        self.encoded_batches: list[list[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(
            (digest * (self.dimension // len(digest) + 1))[: self.dimension],
            dtype=np.uint8,
        ).astype(np.float64)
        return (raw / 255.0 - 0.5) * self.scale

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        batch = list(texts)
        self.encoded_batches.append(batch)
        return np.vstack([self._vector(text) for text in batch])


class DirectionalEmbedder:
    """An embedder with planted structure, for end-to-end similarity tests.

    Each family owns an axis. A document's vector is that axis plus a small,
    deterministic perturbation, so documents of one family really do cluster
    and a centroid really does sit among them — which is what makes an
    integration test over three families meaningful instead of circular.
    """

    def __init__(self, axes: dict[str, int], dimension: int = 8, noise: float = 0.05) -> None:
        self.axes = dict(axes)
        self.dimension = int(dimension)
        self.noise = float(noise)
        self.tokenizer = FakeTokenizer()

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        marker = next((name for name in self.axes if name in text), None)
        if marker is None:
            vector[:] = 0.1
            return vector
        vector[self.axes[marker]] = 1.0
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        jitter = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype(np.float64)
        return vector + (jitter / 255.0 - 0.5) * self.noise

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])


def feature_record(
    text: str,
    *,
    sample_id: str = "sample",
    schema_version: str = "2.1",
    taxonomy_version: str = "rvl-cdip-2.1",
    alnum_character_count: int | None = None,
    provenance: dict | None = None,
) -> dict:
    """A minimal ``classification_features`` record, as the contract defines it."""
    record = {
        "schema_version": schema_version,
        "taxonomy_version": taxonomy_version,
        "classification_text": text,
        "alnum_character_count": (
            len([character for character in text if character.isalnum()])
            if alnum_character_count is None
            else int(alnum_character_count)
        ),
        "provenance": {"sample_id": sample_id, "ocr_engine": "doctr"}
        if provenance is None
        else provenance,
        "sample_id": sample_id,
        "feature_extraction_version": "2.6",
    }
    return record
