"""FabricFlow task wrapper: turn classification features into one document vector.

This task never opens an image, a PDF or an OCR engine. It reads the
``classification_text`` that ``extract_document_classification_features``
already produced, chunks it with the encoder's own tokenizer, embeds every
chunk and pools them. The full vector travels to the classifier task; the
workflow's public output carries only the metadata that describes it.
"""

from __future__ import annotations

from time import perf_counter

from core.task import task

from tasks.document.embedding_classifier_core import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TEXT_PREFIX,
    EMBEDDING_EXTRACTION_VERSION,
    EMBEDDING_FINGERPRINT,
    EmbeddingModelConfig,
    SentenceTransformerEmbedder,
    embed_document,
    validate_embedding_input,
)


@task(
    outputs={
        "document_embedding": {
            "type": "dict",
            "description": "Pooled document vector, its chunking metadata, and the model identity that produced it",
        }
    },
    display_name="Extract Document Embedding",
    description="Embed the OCR text of an aggregated document with a sentence-transformers encoder",
    category="document",
    parameters={
        "classification_features": {
            "type": "dict",
            "required": True,
            "description": "Output of extract_document_classification_features",
        },
        "model_name": {
            "type": "str",
            "required": False,
            "default": DEFAULT_MODEL_NAME,
            "description": "sentence-transformers model identifier",
        },
        "model_revision": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Pinned model revision; empty means the hub default, recorded as unpinned",
        },
        "text_prefix": {
            "type": "str",
            "required": False,
            "default": DEFAULT_TEXT_PREFIX,
            "description": "Instruction prefix applied to documents and reference examples alike",
        },
        "max_tokens": {
            "type": "int",
            "required": False,
            "default": DEFAULT_MAX_TOKENS,
            "description": "Chunk size in model tokens",
        },
        "overlap_tokens": {
            "type": "int",
            "required": False,
            "default": DEFAULT_OVERLAP_TOKENS,
            "description": "Token overlap between consecutive chunks; must be smaller than max_tokens",
        },
        "min_recognized_characters": {
            "type": "int",
            "required": False,
            "default": DEFAULT_MIN_RECOGNIZED_CHARACTERS,
            "description": "Below this many OCR characters the document is not embedded at all",
        },
    },
)
def extract_document_embedding(
    classification_features: dict,
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str = "",
    text_prefix: str = DEFAULT_TEXT_PREFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_recognized_characters: int = DEFAULT_MIN_RECOGNIZED_CHARACTERS,
) -> dict:
    started = perf_counter()
    descriptor = validate_embedding_input(classification_features)
    config = EmbeddingModelConfig(
        model_name=model_name,
        model_revision=model_revision or None,
        text_prefix=text_prefix,
        max_tokens=int(max_tokens),
        overlap_tokens=int(overlap_tokens),
    )
    recognized = descriptor["recognized_characters"]
    if recognized < int(min_recognized_characters):
        # Not an error and not an empty vector: a document with no readable
        # text has no position in the space, and saying so is the honest
        # output. The classifier turns this into an abstention.
        return {
            "vector": [],
            "embedded": False,
            "reason": "insufficient_ocr_text",
            "recognized_characters": recognized,
            "metadata": {
                "chunk_count": 0,
                "token_count": 0,
                "embedding_dimension": 0,
                "pooling": config.pooling,
                "model": config.model_name,
                "revision": config.model_revision,
                "extraction_version": EMBEDDING_EXTRACTION_VERSION,
                "embedding_fingerprint": EMBEDDING_FINGERPRINT,
            },
            "execution_time_ms": round((perf_counter() - started) * 1000.0, 4),
        }
    embedder = SentenceTransformerEmbedder(config).load()
    vector, metadata = embed_document(
        classification_features.get("classification_text"), embedder, config
    )
    metadata["revision"] = metadata.get("revision") or embedder.revision
    metadata["device"] = embedder.device
    metadata["embedding_fingerprint"] = EMBEDDING_FINGERPRINT
    metadata["recognized_characters"] = recognized
    # Single-output task: the declared ``document_embedding`` output is this
    # dict, never a singleton tuple — a one-element tuple is unpacked
    # positionally by the runner and arrives downstream unreadable.
    return {
        "vector": [float(value) for value in vector],
        "embedded": True,
        "reason": "",
        "recognized_characters": recognized,
        "metadata": metadata,
        "execution_time_ms": round((perf_counter() - started) * 1000.0, 4),
    }
