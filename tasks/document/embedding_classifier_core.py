"""Embedding baseline for document classification: centroids and cosine similarity.

What this is
------------
The rules classifier (``rules_classifier_core``) decides with hand-written
evidence. This module decides with distance to a labelled centroid, over
exactly the same inputs: the ``classification_features`` the existing workflow
already produces. Nothing here re-runs OCR, opens an image, or reads a PDF —
the only field it consumes is ``classification_text``.

It is a *baseline*, deliberately: one text encoder, one vector per document,
one centroid per RVL-CDIP label, cosine similarity, a threshold and a margin.
No fine-tuning, no image or multimodal encoder, no vector database, no
reranker, no k-NN, and no combination with the rules classifier. Those are
later increments and none of them is anticipated in this code.

What it shares with the rules classifier, and what it does not
-------------------------------------------------------------
Shared, by import rather than by copy: the taxonomy
(:mod:`tasks.document.rvl_cdip_eval`), the family list, the rejection labels
and the decision vocabulary (``classified`` / ``fallback`` / ``abstained``).

Not shared: versions and fingerprints. This classifier carries its own
:data:`EMBEDDING_SCHEMA_VERSION`, :data:`EMBEDDING_EXTRACTION_VERSION`,
:data:`EMBEDDING_CLASSIFIER_VERSION`, :data:`EMBEDDING_FINGERPRINT` and, per
index, a ``reference_fingerprint``. A rules fingerprint and an embedding
fingerprint describe different machinery and must never be compared or pooled.

``other`` is not a class here
-----------------------------
``file folder`` and ``handwritten`` get no centroid. They are what the
classifier is expected to *decline*, so they can only ever be reached through
``fallback``/``other`` — never predicted positively. A document that looks like
one of them, and resembles no family strongly enough, falls back; that is the
correct outcome, not an error.

Cosine similarity is not a probability
--------------------------------------
``confidence`` is reported alongside ``confidence_kind =
"cosine_similarity_not_probability"``, and the unclipped cosine is kept in
``top_similarity``. Nothing in this module calibrates a probability, and
nothing downstream should read one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from tasks.document.rvl_cdip_eval import (
    CLASS_TO_FAMILY,
    DOCUMENT_FAMILIES,
    REJECTION_LABELS,
    SCORED_FAMILIES,
    TAXONOMY_VERSION,
    taxonomy_descriptor,
)

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

#: Version of the *record contract* this classifier reads and writes. It
#: deliberately equals the rules classifier's schema, because the input record
#: and the output record are the same shape the evaluator already consumes —
#: the two classifiers are interchangeable to a consumer, which is the whole
#: point of the comparison.
EMBEDDING_SCHEMA_VERSION = "2.1"

#: Feature records this classifier can read.
SUPPORTED_FEATURE_SCHEMA_VERSIONS = frozenset({"2.1"})

#: How a document becomes a vector: normalisation, chunking, pooling. It moves
#: whenever a stored embedding produced under the old definition would differ
#: from one produced now — which makes a cached vector refusable rather than
#: silently mixed with fresh ones.
EMBEDDING_EXTRACTION_VERSION = "1.0"

#: Public decision vocabulary, shared with the rules classifier so one
#: evaluator can read both.
ACCEPTED_DECISIONS = frozenset({"classified", "observed"})
DECISIONS = frozenset({"classified", "fallback", "abstained", "observed"})

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
#: Pinned at build time and recorded in the index metadata. ``None`` means "not
#: pinned", which the index records honestly rather than inventing a value.
DEFAULT_MODEL_REVISION: str | None = None
#: E5 is trained with instruction prefixes. The document-classification task is
#: *symmetric* — a document is compared against reference documents, not
#: against a query — so the same prefix is applied to both sides. Using
#: ``query:`` on one side and ``passage:`` on the other would place the two in
#: different regions of the space and make the cosine meaningless.
DEFAULT_TEXT_PREFIX = "query: "
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_MAX_TOKENS = 384
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_POOLING = "normalized_chunk_mean"
#: Probed in this order; the first available wins.
DEFAULT_DEVICE_ORDER: tuple[str, ...] = ("cuda", "mps", "cpu")

#: Decision defaults. ``0.0`` is *not* a calibrated operating point and the
#: output says so through ``calibration_status``: an uncalibrated run accepts
#: whatever the top candidate is, which is a diagnostic, not a deployment.
DEFAULT_SIMILARITY_THRESHOLD = 0.0
DEFAULT_MIN_SCORE_MARGIN = 0.0
#: Same floor the rules classifier uses, for the same reason: below it there is
#: no text to classify, only OCR noise.
DEFAULT_MIN_RECOGNIZED_CHARACTERS = 20

CALIBRATION_STATUSES = frozenset({"calibrated", "uncalibrated"})

FALLBACK_TEMPLATE = "clean_article"

_WHITESPACE_RUN = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class EmbeddingContractError(RuntimeError):
    """An input, index or configuration this classifier must not score."""


class InsufficientTextError(EmbeddingContractError):
    """Not enough recognised text to embed — an abstention, not a failure."""


# ---------------------------------------------------------------------------
# Label candidates: built from the taxonomy, never redeclared
# ---------------------------------------------------------------------------

#: RVL-CDIP labels that get a centroid. Rejection labels deliberately do not:
#: ``other`` stays a refusal outcome rather than becoming a class with a
#: prototype of its own. Sorted so that index row order is reproducible.
CANDIDATE_LABELS: tuple[str, ...] = tuple(
    sorted(label for label in CLASS_TO_FAMILY if label not in REJECTION_LABELS)
)

#: Family -> the labels whose centroids represent it. A family scores the
#: *maximum* over its labels, never the mean: ``correspondence`` covers email,
#: letter and memo, three genuinely different page shapes, and averaging them
#: would dilute a family into never winning anything.
FAMILY_TO_LABELS: dict[str, tuple[str, ...]] = {
    family: tuple(
        sorted(label for label in CANDIDATE_LABELS if CLASS_TO_FAMILY[label] == family)
    )
    for family in DOCUMENT_FAMILIES
    if any(CLASS_TO_FAMILY.get(label) == family for label in CANDIDATE_LABELS)
}

#: Families this classifier can actually predict.
CANDIDATE_FAMILIES: tuple[str, ...] = tuple(sorted(FAMILY_TO_LABELS))


def family_of_label(label: str) -> str:
    """The family a candidate label belongs to, from the taxonomy module."""
    try:
        return CLASS_TO_FAMILY[label]
    except KeyError as error:
        raise EmbeddingContractError(
            f"unknown RVL-CDIP label {label!r}; known candidates are: "
            f"{', '.join(CANDIDATE_LABELS)}"
        ) from error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Everything that decides what vector a document becomes."""

    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str | None = DEFAULT_MODEL_REVISION
    text_prefix: str = DEFAULT_TEXT_PREFIX
    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    pooling: str = DEFAULT_POOLING
    device_order: tuple[str, ...] = DEFAULT_DEVICE_ORDER
    normalize: bool = True

    def __post_init__(self) -> None:
        validate_chunk_configuration(self.max_tokens, self.overlap_tokens)
        if not str(self.model_name).strip():
            raise EmbeddingContractError("model_name must not be empty")
        if self.pooling != DEFAULT_POOLING:
            raise EmbeddingContractError(
                f"unsupported pooling {self.pooling!r}; this baseline implements "
                f"{DEFAULT_POOLING!r} only"
            )

    def descriptor(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "revision": self.model_revision,
            "text_prefix": self.text_prefix,
            "max_tokens": int(self.max_tokens),
            "overlap_tokens": int(self.overlap_tokens),
            "pooling": self.pooling,
            "normalize": bool(self.normalize),
            "extraction_version": EMBEDDING_EXTRACTION_VERSION,
        }


@dataclass(frozen=True)
class EmbeddingClassifierConfig:
    """Everything that decides what a vector is classified as."""

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    min_score_margin: float = DEFAULT_MIN_SCORE_MARGIN
    min_recognized_characters: int = DEFAULT_MIN_RECOGNIZED_CHARACTERS
    family_similarity_thresholds: dict[str, float] = field(default_factory=dict)
    family_min_score_margins: dict[str, float] = field(default_factory=dict)
    calibration_status: str = "uncalibrated"
    calibration_note: str = ""
    mode: str = "evaluate"

    def __post_init__(self) -> None:
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise EmbeddingContractError(
                f"calibration_status must be one of {sorted(CALIBRATION_STATUSES)}, "
                f"got {self.calibration_status!r}"
            )
        for mapping, name in (
            (self.family_similarity_thresholds, "family_similarity_thresholds"),
            (self.family_min_score_margins, "family_min_score_margins"),
        ):
            for family in mapping:
                if family not in CANDIDATE_FAMILIES:
                    raise EmbeddingContractError(
                        f"{name} names {family!r}, which is not a candidate family; "
                        f"candidates are: {', '.join(CANDIDATE_FAMILIES)}"
                    )

    def threshold_for(self, family: str) -> float:
        return float(
            self.family_similarity_thresholds.get(family, self.similarity_threshold)
        )

    def margin_for(self, family: str) -> float:
        return float(self.family_min_score_margins.get(family, self.min_score_margin))

    def descriptor(self) -> dict[str, Any]:
        return {
            "similarity": round(float(self.similarity_threshold), 6),
            "minimum_score_margin": round(float(self.min_score_margin), 6),
            "minimum_recognized_characters": int(self.min_recognized_characters),
            "family_similarity_thresholds": {
                family: round(float(value), 6)
                for family, value in sorted(self.family_similarity_thresholds.items())
            },
            "family_minimum_score_margins": {
                family: round(float(value), 6)
                for family, value in sorted(self.family_min_score_margins.items())
            },
            "calibration_status": self.calibration_status,
            "calibration_note": self.calibration_note,
        }


def load_classifier_config(path: str | Path | None = None) -> tuple[EmbeddingModelConfig, EmbeddingClassifierConfig]:
    """Read the shipped JSON configuration.

    Absent file -> documented defaults, marked ``uncalibrated``. No absolute or
    user-specific path is ever baked in: the default is resolved relative to
    this repository.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    payload: dict[str, Any] = {}
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EmbeddingContractError(
                f"cannot read embedding classifier config {config_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise EmbeddingContractError(
                f"embedding classifier config {config_path} must be a JSON object"
            )
    model_payload = payload.get("model") or {}
    decision_payload = payload.get("decision") or {}
    model = EmbeddingModelConfig(
        model_name=str(model_payload.get("name", DEFAULT_MODEL_NAME)),
        model_revision=model_payload.get("revision") or None,
        text_prefix=str(model_payload.get("text_prefix", DEFAULT_TEXT_PREFIX)),
        max_tokens=int(model_payload.get("max_tokens", DEFAULT_MAX_TOKENS)),
        overlap_tokens=int(model_payload.get("overlap_tokens", DEFAULT_OVERLAP_TOKENS)),
        pooling=str(model_payload.get("pooling", DEFAULT_POOLING)),
        device_order=tuple(model_payload.get("device_order", DEFAULT_DEVICE_ORDER)),
    )
    classifier = EmbeddingClassifierConfig(
        similarity_threshold=float(
            decision_payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        ),
        min_score_margin=float(
            decision_payload.get("min_score_margin", DEFAULT_MIN_SCORE_MARGIN)
        ),
        min_recognized_characters=int(
            decision_payload.get(
                "min_recognized_characters", DEFAULT_MIN_RECOGNIZED_CHARACTERS
            )
        ),
        family_similarity_thresholds=dict(
            decision_payload.get("family_similarity_thresholds") or {}
        ),
        family_min_score_margins=dict(
            decision_payload.get("family_min_score_margins") or {}
        ),
        calibration_status=str(decision_payload.get("calibration_status", "uncalibrated")),
        calibration_note=str(decision_payload.get("calibration_note", "")),
    )
    return model, classifier


def load_thresholds_file(
    path: str | Path, base: EmbeddingClassifierConfig | None = None
) -> EmbeddingClassifierConfig:
    """Apply a calibration artifact on top of a configuration."""
    payload = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(payload, dict):
        raise EmbeddingContractError(f"threshold file {path} must be a JSON object")
    thresholds = payload.get("thresholds") or payload
    current = base or EmbeddingClassifierConfig()
    return replace(
        current,
        similarity_threshold=float(
            thresholds.get("similarity", current.similarity_threshold)
        ),
        min_score_margin=float(
            thresholds.get("minimum_score_margin", current.min_score_margin)
        ),
        min_recognized_characters=int(
            thresholds.get(
                "minimum_recognized_characters", current.min_recognized_characters
            )
        ),
        family_similarity_thresholds=dict(
            thresholds.get("family_similarity_thresholds") or {}
        ),
        family_min_score_margins=dict(
            thresholds.get("family_minimum_score_margins") or {}
        ),
        calibration_status=str(
            thresholds.get("calibration_status", current.calibration_status)
        ),
        calibration_note=str(thresholds.get("calibration_note", current.calibration_note)),
    )


# ---------------------------------------------------------------------------
# Text normalisation and chunking
# ---------------------------------------------------------------------------


def normalise_classification_text(text: Any) -> str:
    """Collapse runs of spaces and blank lines. Nothing else.

    Titles, tables, punctuation and casing are left exactly as the OCR read
    them: they carry most of what separates an invoice from a memo, and a
    "cleaning" step that strips them is a silent feature change.
    """
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = _WHITESPACE_RUN.sub(" ", value)
    value = "\n".join(line.strip() for line in value.split("\n"))
    value = _BLANK_LINES.sub("\n\n", value)
    return value.strip()


def validate_chunk_configuration(max_tokens: int, overlap_tokens: int) -> None:
    if int(max_tokens) <= 0:
        raise EmbeddingContractError(f"max_tokens must be positive, got {max_tokens!r}")
    if int(overlap_tokens) < 0:
        raise EmbeddingContractError(
            f"overlap_tokens must not be negative, got {overlap_tokens!r}"
        )
    if int(overlap_tokens) >= int(max_tokens):
        raise EmbeddingContractError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than max_tokens "
            f"({max_tokens}); otherwise the window never advances"
        )


@runtime_checkable
class TokenizerProtocol(Protocol):
    """The two methods this module needs from a tokenizer."""

    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...


def encode_tokens(tokenizer: Any, text: str) -> list[int]:
    """Token ids without special tokens, whichever tokenizer API is available."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def decode_tokens(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(ids))


def chunk_classification_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split a document into overlapping windows of the model's own tokens.

    Pure: same text, same tokenizer, same window -> same chunks, in order.

    The document is *never* silently truncated at the first window. A scanned
    newspaper page runs to thousands of tokens, and an encoder that reads only
    the first 384 of them classifies the masthead, not the document — the kind
    of failure that produces a plausible number and no error.

    The window advances by ``max_tokens - overlap_tokens``, so every token
    appears in at least one chunk and consecutive chunks share exactly
    ``overlap_tokens`` tokens. Chunks that decode to nothing are dropped rather
    than embedded as empty strings.
    """
    validate_chunk_configuration(max_tokens, overlap_tokens)
    normalised = normalise_classification_text(text)
    if not normalised:
        return []
    token_ids = encode_tokens(tokenizer, normalised)
    if not token_ids:
        return []
    step = int(max_tokens) - int(overlap_tokens)
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + int(max_tokens)]
        if not window:
            break
        decoded = decode_tokens(tokenizer, window).strip()
        if decoded:
            chunks.append(decoded)
        if start + int(max_tokens) >= len(token_ids):
            break
    return chunks


def count_tokens(text: str, tokenizer: Any) -> int:
    return len(encode_tokens(tokenizer, normalise_classification_text(text)))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbedderProtocol(Protocol):
    """What :func:`embed_document` needs: a tokenizer and batch encoding."""

    tokenizer: Any

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length; a zero row stays zero rather than NaN."""
    array = np.asarray(vectors, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    normalised = array / safe
    return normalised[0] if single else normalised


def embed_document(
    text: str,
    embedder: Any,
    config: EmbeddingModelConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chunk, embed, normalise, mean, normalise again.

    Returns the document vector and the metadata that describes how it was
    produced. The vector itself is never part of a workflow's public output —
    768 floats per document in a JSON report is not a report.
    """
    model_config = config or EmbeddingModelConfig()
    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is None:
        raise EmbeddingContractError("embedder must expose a .tokenizer")
    chunks = chunk_classification_text(
        text,
        tokenizer,
        max_tokens=model_config.max_tokens,
        overlap_tokens=model_config.overlap_tokens,
    )
    if not chunks:
        raise InsufficientTextError("no non-empty chunk to embed")
    prefixed = [f"{model_config.text_prefix}{chunk}" for chunk in chunks]
    raw = np.asarray(embedder.encode(prefixed), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[0] != len(prefixed):
        raise EmbeddingContractError(
            f"embedder returned {raw.shape[0]} vectors for {len(prefixed)} chunks"
        )
    # Normalise each chunk before averaging: without it a single long chunk
    # with a large norm decides the document by magnitude rather than by
    # direction, which is not what cosine similarity is being asked.
    per_chunk = l2_normalize(raw)
    pooled = l2_normalize(per_chunk.mean(axis=0))
    metadata = {
        "chunk_count": len(chunks),
        "token_count": count_tokens(text, tokenizer),
        "embedding_dimension": int(pooled.shape[0]),
        "pooling": model_config.pooling,
        "model": model_config.model_name,
        "revision": model_config.model_revision,
        "text_prefix": model_config.text_prefix,
        "max_tokens": int(model_config.max_tokens),
        "overlap_tokens": int(model_config.overlap_tokens),
        "extraction_version": EMBEDDING_EXTRACTION_VERSION,
    }
    return pooled.astype(np.float64), metadata


class SentenceTransformerEmbedder:
    """Lazy wrapper around ``sentence-transformers``.

    The import happens inside :meth:`load`, never at module import: the unit
    tests must run with no model, no network and no ``sentence_transformers``
    installed, and an import at module scope would make that impossible.
    """

    def __init__(self, config: EmbeddingModelConfig | None = None) -> None:
        self.config = config or EmbeddingModelConfig()
        self._model: Any = None
        self._device: str | None = None
        self._resolved_revision: str | None = self.config.model_revision

    @staticmethod
    def detect_device(order: Iterable[str] = DEFAULT_DEVICE_ORDER) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        for candidate in order:
            if candidate == "cuda" and torch.cuda.is_available():
                return "cuda"
            if candidate == "mps" and getattr(torch.backends, "mps", None) is not None:
                if torch.backends.mps.is_available():
                    return "mps"
            if candidate == "cpu":
                return "cpu"
        return "cpu"

    def load(self) -> "SentenceTransformerEmbedder":
        if self._model is not None:
            return self
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise EmbeddingContractError(
                "sentence-transformers is required to embed documents; install it, "
                "or inject an embedder implementing EmbedderProtocol"
            ) from error
        self._device = self.detect_device(self.config.device_order)
        kwargs: dict[str, Any] = {"device": self._device}
        if self.config.model_revision:
            kwargs["revision"] = self.config.model_revision
        self._model = SentenceTransformer(self.config.model_name, **kwargs)
        self._resolved_revision = self.config.model_revision or _resolve_model_revision(
            self._model
        )
        return self

    @property
    def device(self) -> str:
        return self._device or "cpu"

    @property
    def revision(self) -> str | None:
        return self._resolved_revision

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._model.tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=8,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )


def _resolve_model_revision(model: Any) -> str | None:  # pragma: no cover - needs a model
    """Best-effort commit hash of the loaded weights, or ``None``."""
    for module in getattr(model, "_modules", {}).values():
        auto_model = getattr(module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


# ---------------------------------------------------------------------------
# Reference index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingReferenceIndex:
    """Label centroids, their provenance, and the compatibility it enforces."""

    labels: tuple[str, ...]
    centroids: np.ndarray
    example_counts: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.labels) != self.centroids.shape[0]:
            raise EmbeddingContractError(
                f"index has {len(self.labels)} labels and "
                f"{self.centroids.shape[0]} centroid rows"
            )
        if len(self.example_counts) != len(self.labels):
            raise EmbeddingContractError("example_counts must be one per label")
        unknown = [label for label in self.labels if label not in CANDIDATE_LABELS]
        if unknown:
            raise EmbeddingContractError(
                f"index carries non-candidate labels {unknown}; rejection labels "
                f"({', '.join(sorted(REJECTION_LABELS))}) must not have centroids"
            )

    @property
    def dimension(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({family_of_label(label) for label in self.labels}))

    @property
    def reference_fingerprint(self) -> str:
        return str(self.metadata.get("reference_fingerprint", ""))

    def label_family_pairs(self) -> list[tuple[str, str]]:
        return [(label, family_of_label(label)) for label in self.labels]

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``embedding_reference.npz`` and its metadata JSON.

        The ``.npz`` holds numeric arrays only — labels live in the JSON — so
        it can be, and is, loaded with ``allow_pickle=False``. Vectors are
        never written to JSON.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        array_path = target / "embedding_reference.npz"
        metadata_path = target / "embedding_reference.metadata.json"
        np.savez(
            array_path,
            centroids=np.asarray(self.centroids, dtype=np.float32),
            example_counts=np.asarray(self.example_counts, dtype=np.int32),
        )
        metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"arrays": array_path, "metadata": metadata_path}

    @classmethod
    def load(cls, directory: str | Path) -> "EmbeddingReferenceIndex":
        source = Path(directory)
        if source.is_file():
            source = source.parent
        array_path = source / "embedding_reference.npz"
        metadata_path = source / "embedding_reference.metadata.json"
        for path in (array_path, metadata_path):
            if not path.is_file():
                raise EmbeddingContractError(f"reference index is missing {path}")
        metadata = json.loads(metadata_path.read_text("utf-8"))
        # allow_pickle stays False: an index is data, and an index that can
        # execute code on load is not data.
        with np.load(array_path, allow_pickle=False) as payload:
            centroids = np.asarray(payload["centroids"], dtype=np.float64)
            example_counts = tuple(int(value) for value in payload["example_counts"])
        labels = tuple(str(label) for label in metadata.get("labels", ()))
        index = cls(
            labels=labels,
            centroids=centroids,
            example_counts=example_counts,
            metadata=metadata,
        )
        expected = reference_fingerprint(
            labels=index.labels,
            centroids=index.centroids,
            example_counts=index.example_counts,
            metadata=index.metadata,
        )
        stored = index.reference_fingerprint
        if stored and stored != expected:
            raise EmbeddingContractError(
                f"reference_fingerprint {stored!r} does not match the index contents "
                f"(recomputed {expected!r}); the index was edited after it was built"
            )
        return index

    # -- compatibility ------------------------------------------------------

    def ensure_compatible(
        self,
        model_config: EmbeddingModelConfig,
        *,
        document_dimension: int | None = None,
        source: str = "reference index",
    ) -> None:
        """Refuse an index that describes different machinery.

        Every one of these mismatches produces a cosine that looks perfectly
        ordinary and means nothing: a different encoder, a different prefix, a
        different chunking, a different taxonomy.
        """
        stored_model = self.metadata.get("model")
        if stored_model != model_config.model_name:
            raise EmbeddingContractError(
                f"{source}: built with model {stored_model!r}, classifier configured "
                f"for {model_config.model_name!r}"
            )
        stored_revision = self.metadata.get("revision")
        if (
            model_config.model_revision
            and stored_revision
            and stored_revision != model_config.model_revision
        ):
            raise EmbeddingContractError(
                f"{source}: built with revision {stored_revision!r}, classifier "
                f"configured for {model_config.model_revision!r}"
            )
        stored_taxonomy = self.metadata.get("taxonomy_version")
        if stored_taxonomy != TAXONOMY_VERSION:
            raise EmbeddingContractError(
                f"{source}: built under taxonomy {stored_taxonomy!r}, current "
                f"taxonomy is {TAXONOMY_VERSION!r}"
            )
        stored_extraction = self.metadata.get("embedding_extraction_version")
        if stored_extraction != EMBEDDING_EXTRACTION_VERSION:
            raise EmbeddingContractError(
                f"{source}: built under embedding extraction version "
                f"{stored_extraction!r}, current is {EMBEDDING_EXTRACTION_VERSION!r}"
            )
        for key, expected, actual in (
            ("text_prefix", self.metadata.get("text_prefix"), model_config.text_prefix),
            ("pooling", self.metadata.get("pooling"), model_config.pooling),
            ("max_tokens", self.metadata.get("max_tokens"), int(model_config.max_tokens)),
            (
                "overlap_tokens",
                self.metadata.get("overlap_tokens"),
                int(model_config.overlap_tokens),
            ),
        ):
            if expected != actual:
                raise EmbeddingContractError(
                    f"{source}: built with {key}={expected!r}, classifier configured "
                    f"for {key}={actual!r}"
                )
        stored_dimension = self.metadata.get("dimension")
        if stored_dimension is not None and int(stored_dimension) != self.dimension:
            raise EmbeddingContractError(
                f"{source}: metadata declares dimension {stored_dimension}, arrays "
                f"carry {self.dimension}"
            )
        if document_dimension is not None and int(document_dimension) != self.dimension:
            raise EmbeddingContractError(
                f"{source}: document embedding has dimension {document_dimension}, "
                f"index centroids have {self.dimension}"
            )


def build_reference_index(
    embeddings_by_label: dict[str, np.ndarray],
    *,
    model_config: EmbeddingModelConfig,
    sample_ids_by_label: dict[str, Sequence[str]] | None = None,
    manifest_hash: str = "",
    manifest_path: str = "",
    minimum_examples: int = 1,
    generated_at: str | None = None,
) -> EmbeddingReferenceIndex:
    """Average the reference documents of each label, then normalise.

    Labels are the unit, not families: a family's score is later the maximum
    over its labels, so ``correspondence`` keeps three prototypes rather than
    one blurred average of email, letter and memo.
    """
    labels = sorted(embeddings_by_label)
    unknown = [label for label in labels if label not in CANDIDATE_LABELS]
    if unknown:
        raise EmbeddingContractError(
            f"reference documents carry labels without a centroid slot: {unknown}; "
            f"rejection labels ({', '.join(sorted(REJECTION_LABELS))}) are negatives, "
            "not classes"
        )
    if not labels:
        raise EmbeddingContractError("no reference documents to build centroids from")
    centroids = []
    counts = []
    kept_labels = []
    skipped: dict[str, int] = {}
    for label in labels:
        matrix = np.asarray(embeddings_by_label[label], dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] < int(minimum_examples):
            skipped[label] = int(matrix.shape[0])
            continue
        centroid = l2_normalize(l2_normalize(matrix).mean(axis=0))
        centroids.append(centroid)
        counts.append(int(matrix.shape[0]))
        kept_labels.append(label)
    if not kept_labels:
        raise EmbeddingContractError(
            f"no label reached minimum_examples={minimum_examples}"
        )
    dimensions = {row.shape[0] for row in centroids}
    if len(dimensions) != 1:
        raise EmbeddingContractError(f"reference embeddings have mixed dimensions: {dimensions}")
    matrix = np.vstack(centroids)
    sample_ids = {
        label: sorted(str(value) for value in (sample_ids_by_label or {}).get(label, ()))
        for label in kept_labels
    }
    metadata: dict[str, Any] = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "embedding_extraction_version": EMBEDDING_EXTRACTION_VERSION,
        "embedding_fingerprint": EMBEDDING_FINGERPRINT,
        "model": model_config.model_name,
        "revision": model_config.model_revision,
        "dimension": int(matrix.shape[1]),
        "text_prefix": model_config.text_prefix,
        "pooling": model_config.pooling,
        "max_tokens": int(model_config.max_tokens),
        "overlap_tokens": int(model_config.overlap_tokens),
        "labels": list(kept_labels),
        "label_families": {label: family_of_label(label) for label in kept_labels},
        "example_counts": {label: count for label, count in zip(kept_labels, counts)},
        "sample_ids": sample_ids,
        "skipped_labels": skipped,
        "minimum_examples": int(minimum_examples),
        "manifest_hash": manifest_hash,
        "manifest_path": manifest_path,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxonomy": taxonomy_descriptor(),
    }
    metadata["reference_fingerprint"] = reference_fingerprint(
        labels=tuple(kept_labels),
        centroids=matrix,
        example_counts=tuple(counts),
        metadata=metadata,
    )
    return EmbeddingReferenceIndex(
        labels=tuple(kept_labels),
        centroids=matrix,
        example_counts=tuple(counts),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Similarity and decision
# ---------------------------------------------------------------------------


def cosine_similarity(document_embedding: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine of the angle to every centroid. Both sides are re-normalised."""
    document = l2_normalize(np.asarray(document_embedding, dtype=np.float64))
    matrix = l2_normalize(np.asarray(centroids, dtype=np.float64))
    return matrix @ document


def aggregate_family_scores(
    label_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    """Family score = the strongest of its labels, with the label that won it."""
    families: dict[str, float] = {}
    winners: dict[str, str] = {}
    for label in sorted(label_scores):
        family = family_of_label(label)
        score = float(label_scores[label])
        # Strictly greater, walking labels in sorted order, so a tie is broken
        # by label name and the result does not depend on dict ordering.
        if family not in families or score > families[family]:
            families[family] = score
            winners[family] = label
    return families, winners


def rank_families(family_scores: dict[str, float]) -> list[tuple[str, float]]:
    """Deterministic ranking: score descending, then family name ascending."""
    return sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))


def apply_decision_policy(
    ranked: Sequence[tuple[str, float]],
    recognized_characters: int,
    config: EmbeddingClassifierConfig,
) -> dict[str, Any]:
    """Threshold, then margin, then accept. Never force a class."""
    if recognized_characters < int(config.min_recognized_characters):
        return {
            "document_family": "other",
            "decision": "abstained",
            "reason": "insufficient_ocr_text",
        }
    if not ranked:
        return {
            "document_family": "other",
            "decision": "fallback",
            "reason": "no_reference_centroids",
        }
    top_family, top_score = ranked[0]
    runner_up, runner_up_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    margin = float(top_score) - float(runner_up_score) if runner_up else float(top_score)
    threshold = config.threshold_for(top_family)
    minimum_margin = config.margin_for(top_family)
    if float(top_score) < threshold:
        return {
            "document_family": "other",
            "decision": "fallback",
            "reason": "similarity_below_threshold",
        }
    if margin < minimum_margin:
        return {
            "document_family": "other",
            "decision": "fallback",
            "reason": "score_margin_below_minimum",
        }
    return {
        "document_family": top_family,
        "decision": "classified",
        "reason": "similarity_above_threshold",
    }


def classify_with_embeddings(
    document_embedding: np.ndarray,
    reference_index: EmbeddingReferenceIndex,
    config: EmbeddingClassifierConfig | None = None,
    *,
    embedding_metadata: dict[str, Any] | None = None,
    model_config: EmbeddingModelConfig | None = None,
    nearest_prototype_count: int = 5,
) -> dict[str, Any]:
    """Classify one document vector against the label centroids.

    Pure and deterministic: no I/O, no model, no clock.
    """
    decision_config = config or EmbeddingClassifierConfig()
    vector = np.asarray(document_embedding, dtype=np.float64).reshape(-1)
    if vector.shape[0] != reference_index.dimension:
        raise EmbeddingContractError(
            f"document embedding has dimension {vector.shape[0]}, index centroids "
            f"have {reference_index.dimension}"
        )
    similarities = cosine_similarity(vector, reference_index.centroids)
    label_scores = {
        label: float(score) for label, score in zip(reference_index.labels, similarities)
    }
    family_scores, winning_labels = aggregate_family_scores(label_scores)
    ranked = rank_families(family_scores)
    recognized = int(
        (embedding_metadata or {}).get("recognized_characters", DEFAULT_MIN_RECOGNIZED_CHARACTERS)
    )
    decision = apply_decision_policy(ranked, recognized, decision_config)

    top_family, top_score = ranked[0] if ranked else ("other", 0.0)
    runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    margin = float(top_score) - float(runner_up_score) if runner_up else float(top_score)
    nearest = [
        {
            "label": label,
            "family": family_of_label(label),
            "similarity": round(float(score), 6),
            "example_count": int(count),
        }
        for label, score, count in sorted(
            zip(reference_index.labels, similarities, reference_index.example_counts),
            key=lambda item: (-float(item[1]), item[0]),
        )[: max(0, int(nearest_prototype_count))]
    ]
    return {
        "document_family": decision["document_family"],
        "decision": decision["decision"],
        "reason": decision["reason"],
        "top_candidate": top_family if ranked else None,
        "runner_up": runner_up,
        "top_similarity": round(float(top_score), 6),
        "score_margin": round(float(margin), 6),
        "candidate_scores": {
            family: round(float(score), 6) for family, score in sorted(family_scores.items())
        },
        "label_scores": {label: round(float(score), 6) for label, score in sorted(label_scores.items())},
        "winning_labels": winning_labels,
        "nearest_prototypes": nearest,
        "ranked": [(family, round(float(score), 6)) for family, score in ranked],
    }


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def validate_embedding_input(
    features: Any, *, source: str = "classification_features"
) -> dict[str, Any]:
    """Refuse a record this classifier must not read; report what it accepted.

    Deliberately narrower than the rules classifier's contract: this classifier
    reads ``classification_text`` and nothing else, so it does not require the
    layout feature set or its fingerprint. What it does require is the schema
    and taxonomy it was built against, and the provenance that says which
    pipeline produced the text.
    """
    if not isinstance(features, dict):
        raise EmbeddingContractError(f"{source}: feature record must be a dict")
    schema_version = features.get("schema_version")
    if schema_version not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
        raise EmbeddingContractError(
            f"{source}: incompatible schema_version {schema_version!r}; this "
            f"classifier reads {sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS)}"
        )
    taxonomy_version = features.get("taxonomy_version")
    if taxonomy_version != TAXONOMY_VERSION:
        raise EmbeddingContractError(
            f"{source}: features carry taxonomy_version {taxonomy_version!r}, "
            f"classifier is {TAXONOMY_VERSION!r}"
        )
    if "classification_text" not in features:
        raise EmbeddingContractError(
            f"{source}: no classification_text; this classifier does not re-run OCR "
            "and will not reconstruct text from images"
        )
    if not isinstance(features.get("provenance"), dict):
        raise EmbeddingContractError(
            f"{source}: provenance must be a dict; without it a stored vector cannot "
            "be traced to the pipeline that produced its text"
        )
    return {
        "schema_version": schema_version,
        "taxonomy_version": taxonomy_version,
        # Recorded, never required: the embedding classifier does not read the
        # layout features, but a report that cannot say which extraction
        # produced its text is not reproducible.
        "feature_extraction_version": features.get("feature_extraction_version"),
        "feature_fingerprint": features.get("feature_fingerprint"),
        "recognized_characters": int(features.get("alnum_character_count") or 0),
    }


def has_sufficient_text(features: dict[str, Any], config: EmbeddingClassifierConfig) -> bool:
    return int(features.get("alnum_character_count") or 0) >= int(
        config.min_recognized_characters
    )


def insufficient_text_result(
    features: dict[str, Any],
    config: EmbeddingClassifierConfig,
    *,
    model_config: EmbeddingModelConfig | None = None,
    reference_index: EmbeddingReferenceIndex | None = None,
) -> dict[str, Any]:
    """The abstention contract, in the same shape as any other result."""
    return build_result(
        classification={
            "document_family": "other",
            "decision": "abstained",
            "reason": "insufficient_ocr_text",
            "top_candidate": None,
            "runner_up": None,
            "top_similarity": 0.0,
            "score_margin": 0.0,
            "candidate_scores": {},
            "nearest_prototypes": [],
        },
        embedding_metadata={
            "chunk_count": 0,
            "token_count": 0,
            "embedding_dimension": 0,
            "pooling": (model_config or EmbeddingModelConfig()).pooling,
        },
        config=config,
        model_config=model_config or EmbeddingModelConfig(),
        reference_index=reference_index,
        recognized_characters=int(features.get("alnum_character_count") or 0),
    )


def build_result(
    *,
    classification: dict[str, Any],
    embedding_metadata: dict[str, Any],
    config: EmbeddingClassifierConfig,
    model_config: EmbeddingModelConfig,
    reference_index: EmbeddingReferenceIndex | None = None,
    recognized_characters: int = 0,
    execution_time_ms: float = 0.0,
) -> dict[str, Any]:
    """Assemble the public result record.

    The document vector is not in it, by design. ``confidence`` is the cosine
    clipped into ``[0, 1]`` for consumers that require that range, and
    ``confidence_kind`` states plainly that it is not a probability;
    ``top_similarity`` keeps the raw value.
    """
    similarity = float(classification.get("top_similarity") or 0.0)
    accepted = classification.get("decision") in ACCEPTED_DECISIONS
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "classifier_version": EMBEDDING_CLASSIFIER_VERSION,
        "classifier": "embeddings",
        "mode": config.mode,
        "document_family": classification.get("document_family", "other"),
        # No embedding subtype exists: subtypes are a rules-gate concept, and
        # inventing one here would put a value in a field nothing computed.
        "document_subtype": None,
        "confidence": round(min(1.0, max(0.0, similarity)), 6),
        "confidence_kind": "cosine_similarity_not_probability",
        "decision": classification.get("decision", "fallback"),
        "reason": classification.get("reason", ""),
        "top_candidate": classification.get("top_candidate"),
        "runner_up": classification.get("runner_up"),
        "top_similarity": round(similarity, 6),
        "score_margin": round(float(classification.get("score_margin") or 0.0), 6),
        "candidate_scores": dict(classification.get("candidate_scores") or {}),
        "nearest_prototypes": list(classification.get("nearest_prototypes") or []),
        "embedding": {
            "model": model_config.model_name,
            "revision": model_config.model_revision,
            "dimension": int(embedding_metadata.get("embedding_dimension") or 0),
            "chunk_count": int(embedding_metadata.get("chunk_count") or 0),
            "token_count": int(embedding_metadata.get("token_count") or 0),
            "pooling": embedding_metadata.get("pooling", model_config.pooling),
            "text_prefix": model_config.text_prefix,
            "extraction_version": EMBEDDING_EXTRACTION_VERSION,
            "embedding_fingerprint": EMBEDDING_FINGERPRINT,
            "reference_fingerprint": (
                reference_index.reference_fingerprint if reference_index else None
            ),
        },
        "thresholds": {
            **config.descriptor(),
            "effective_similarity_threshold": round(
                config.threshold_for(str(classification.get("top_candidate") or "")), 6
            )
            if classification.get("top_candidate")
            else round(float(config.similarity_threshold), 6),
        },
        "recognized_characters": int(recognized_characters),
        "recommended_template": None,
        "fallback_template": FALLBACK_TEMPLATE,
        "accepted": bool(accepted),
        "execution_time_ms": round(float(execution_time_ms), 4),
    }


def classify_features_with_embeddings(
    features: dict[str, Any],
    embedder: Any,
    reference_index: EmbeddingReferenceIndex,
    config: EmbeddingClassifierConfig | None = None,
    model_config: EmbeddingModelConfig | None = None,
) -> dict[str, Any]:
    """End-to-end: validate a feature record, embed its text, classify it."""
    decision_config = config or EmbeddingClassifierConfig()
    embedding_config = model_config or EmbeddingModelConfig()
    descriptor = validate_embedding_input(features)
    reference_index.ensure_compatible(embedding_config)
    if not has_sufficient_text(features, decision_config):
        return insufficient_text_result(
            features,
            decision_config,
            model_config=embedding_config,
            reference_index=reference_index,
        )
    vector, metadata = embed_document(
        features.get("classification_text"), embedder, embedding_config
    )
    reference_index.ensure_compatible(
        embedding_config, document_dimension=int(vector.shape[0])
    )
    metadata["recognized_characters"] = descriptor["recognized_characters"]
    classification = classify_with_embeddings(
        vector,
        reference_index,
        decision_config,
        embedding_metadata=metadata,
        model_config=embedding_config,
    )
    return build_result(
        classification=classification,
        embedding_metadata=metadata,
        config=decision_config,
        model_config=embedding_config,
        reference_index=reference_index,
        recognized_characters=descriptor["recognized_characters"],
    )


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _source_signature(value: Any) -> str:
    """Source text of a definition, whitespace-normalised.

    Source rather than bytecode, and never ``repr`` of a code object: a repr
    embeds memory addresses, which makes a fingerprint differ between two runs
    of identical code — exactly the bug this project already fixed once in the
    rules classifier.
    """
    try:
        return " ".join(inspect.getsource(value).split())
    except (OSError, TypeError):  # pragma: no cover - builtins only
        return f"<unavailable:{getattr(value, '__name__', repr(value))}>"


def embedding_fingerprint(
    model_config: EmbeddingModelConfig | None = None,
) -> str:
    """Identity of the decision machinery, not of any particular index.

    Moves when chunking, pooling, normalisation, the similarity computation,
    the family aggregation, the decision policy, the model identity or the
    taxonomy changes. Deterministic across processes: it is derived from source
    text and configuration values only.
    """
    config = model_config or EmbeddingModelConfig()
    parts = [
        f"schema:{EMBEDDING_SCHEMA_VERSION}",
        f"extraction:{EMBEDDING_EXTRACTION_VERSION}",
        f"taxonomy:{TAXONOMY_VERSION}",
        f"model:{config.model_name}",
        f"revision:{config.model_revision or ''}",
        f"prefix:{config.text_prefix}",
        f"chunking:{int(config.max_tokens)}/{int(config.overlap_tokens)}",
        f"pooling:{config.pooling}",
        f"normalize:{bool(config.normalize)}",
        f"candidate_labels:{list(CANDIDATE_LABELS)}",
        f"candidate_families:{list(CANDIDATE_FAMILIES)}",
        f"family_labels:{sorted((family, list(labels)) for family, labels in FAMILY_TO_LABELS.items())}",
        f"scored_families:{list(SCORED_FAMILIES)}",
        f"min_recognized_characters:{int(DEFAULT_MIN_RECOGNIZED_CHARACTERS)}",
    ]
    for function in (
        normalise_classification_text,
        chunk_classification_text,
        embed_document,
        l2_normalize,
        cosine_similarity,
        aggregate_family_scores,
        rank_families,
        apply_decision_policy,
        classify_with_embeddings,
        build_result,
    ):
        parts.append(f"{function.__name__}:{_source_signature(function)}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def reference_fingerprint(
    *,
    labels: Sequence[str],
    centroids: np.ndarray,
    example_counts: Sequence[int],
    metadata: dict[str, Any],
) -> str:
    """Identity of one built index: its vectors, its labels and its provenance."""
    digest = hashlib.sha1()
    digest.update(f"labels:{list(labels)}".encode("utf-8"))
    digest.update(f"counts:{[int(value) for value in example_counts]}".encode("utf-8"))
    # float32, the persisted precision, so the fingerprint of an index equals
    # the fingerprint of the same index after a save/load round trip.
    digest.update(np.ascontiguousarray(np.asarray(centroids, dtype=np.float32)).tobytes())
    for key in (
        "model",
        "revision",
        "dimension",
        "text_prefix",
        "pooling",
        "max_tokens",
        "overlap_tokens",
        "taxonomy_version",
        "embedding_extraction_version",
        "embedding_fingerprint",
        "manifest_hash",
    ):
        digest.update(f"{key}={metadata.get(key)!r}".encode("utf-8"))
    return digest.hexdigest()[:12]


def manifest_hash(path: str | Path) -> str:
    """Content hash of a manifest file, so an index names the split it read."""
    data = Path(path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:12]


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "document_embedding_classifier.json"
)

#: Identity of this classifier's machinery, computed once at import.
EMBEDDING_FINGERPRINT = embedding_fingerprint()

#: ``embeddings-rvl-cdip-v1+<fingerprint>``. Never comparable with a
#: ``rules-rvl-cdip-*`` version: different inputs, different decision, different
#: failure modes.
EMBEDDING_CLASSIFIER_VERSION = f"embeddings-rvl-cdip-v1+{EMBEDDING_FINGERPRINT}"


def classifier_versions(
    reference_index: EmbeddingReferenceIndex | None = None,
    model_config: EmbeddingModelConfig | None = None,
    config: EmbeddingClassifierConfig | None = None,
) -> dict[str, Any]:
    """The version block every report and every row carries."""
    model = model_config or EmbeddingModelConfig()
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "embedding_extraction_version": EMBEDDING_EXTRACTION_VERSION,
        "classifier_version": EMBEDDING_CLASSIFIER_VERSION,
        "classifier": "embeddings",
        "embedding_fingerprint": EMBEDDING_FINGERPRINT,
        "reference_fingerprint": (
            reference_index.reference_fingerprint if reference_index else None
        ),
        "model": model.descriptor(),
        "decision": (config or EmbeddingClassifierConfig()).descriptor(),
        "candidate_labels": list(CANDIDATE_LABELS),
        "candidate_families": list(CANDIDATE_FAMILIES),
        "rejection_labels": sorted(REJECTION_LABELS),
        "taxonomy": taxonomy_descriptor(),
    }
