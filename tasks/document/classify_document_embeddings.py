"""FabricFlow task wrapper: classify a document vector against label centroids."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from core.task import task

from tasks.document.embedding_classifier_core import (
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    DEFAULT_SIMILARITY_THRESHOLD,
    EmbeddingClassifierConfig,
    EmbeddingContractError,
    EmbeddingModelConfig,
    EmbeddingReferenceIndex,
    build_result,
    classify_with_embeddings,
    insufficient_text_result,
    load_classifier_config,
    load_thresholds_file,
)


@task(
    outputs={
        "classification": {
            "type": "dict",
            "description": "Document family, cosine similarity, decision, nearest prototypes, and classifier latency",
        }
    },
    display_name="Classify Document With Embeddings",
    description="Classify an embedded document by cosine similarity to RVL-CDIP label centroids",
    category="document",
    parameters={
        "document_embedding": {
            "type": "dict",
            "required": True,
            "description": "Output of extract_document_embedding",
        },
        "reference_index_path": {
            "type": "str",
            "required": True,
            "description": "Directory holding embedding_reference.npz and its metadata",
        },
        "config_path": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Classifier configuration JSON; empty uses the shipped config",
        },
        "thresholds_path": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Calibration artifact overriding the configured thresholds",
        },
        "similarity_threshold": {
            "type": "float",
            "required": False,
            "default": -1.0,
            "description": "Explicit similarity threshold; negative keeps the configured value",
        },
        "min_score_margin": {
            "type": "float",
            "required": False,
            "default": -1.0,
            "description": "Explicit minimum margin between the two leading families; negative keeps the configured value",
        },
        "min_recognized_characters": {
            "type": "int",
            "required": False,
            "default": DEFAULT_MIN_RECOGNIZED_CHARACTERS,
            "description": "Minimum OCR alphanumeric characters required for a decision",
        },
        "classification_mode": {
            "type": "str",
            "required": False,
            "default": "evaluate",
            "description": "Execution mode recorded in the result: evaluate or observe",
        },
    },
)
def classify_document_embeddings(
    document_embedding: dict,
    reference_index_path: str,
    config_path: str = "",
    thresholds_path: str = "",
    similarity_threshold: float = -1.0,
    min_score_margin: float = -1.0,
    min_recognized_characters: int = DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    classification_mode: str = "evaluate",
) -> dict:
    started = perf_counter()
    if not isinstance(document_embedding, dict):
        raise EmbeddingContractError("document_embedding must be a dict")

    model_config, classifier_config = load_classifier_config(config_path or None)
    if thresholds_path:
        classifier_config = load_thresholds_file(thresholds_path, classifier_config)
    overrides: dict = {
        "min_recognized_characters": int(min_recognized_characters),
        "mode": classification_mode,
    }
    if similarity_threshold >= 0.0:
        overrides["similarity_threshold"] = float(similarity_threshold)
    if min_score_margin >= 0.0:
        overrides["min_score_margin"] = float(min_score_margin)
    classifier_config = replace(classifier_config, **overrides)

    index = EmbeddingReferenceIndex.load(Path(reference_index_path))
    metadata = dict(document_embedding.get("metadata") or {})
    recognized = int(document_embedding.get("recognized_characters") or 0)

    if not document_embedding.get("embedded"):
        # An abstention compares nothing against nothing: there is no vector,
        # no cosine and no centroid involved, so it must not be made to depend
        # on the index agreeing with a model that was never run.
        result = insufficient_text_result(
            {"alnum_character_count": recognized},
            classifier_config,
            model_config=model_config,
            reference_index=index,
        )
        result["execution_time_ms"] = round((perf_counter() - started) * 1000.0, 4)
        return result

    # The model that built the index and the model that embedded this document
    # must be the same one, configured the same way. Two encoders produce two
    # geometries, and a cosine across them is a number with no meaning.
    stored_model = metadata.get("model")
    if stored_model:
        model_config = replace(
            model_config,
            model_name=str(stored_model),
            model_revision=metadata.get("revision") or model_config.model_revision,
            text_prefix=str(metadata.get("text_prefix", model_config.text_prefix)),
            max_tokens=int(metadata.get("max_tokens", model_config.max_tokens)),
            overlap_tokens=int(metadata.get("overlap_tokens", model_config.overlap_tokens)),
        )
    index.ensure_compatible(model_config)

    vector = np.asarray(document_embedding.get("vector") or [], dtype=np.float64)
    index.ensure_compatible(model_config, document_dimension=int(vector.shape[0]))
    metadata["recognized_characters"] = recognized
    classification = classify_with_embeddings(
        vector,
        index,
        classifier_config,
        embedding_metadata=metadata,
        model_config=model_config,
    )
    result = build_result(
        classification=classification,
        embedding_metadata=metadata,
        config=classifier_config,
        model_config=model_config,
        reference_index=index,
        recognized_characters=recognized,
    )
    result["execution_time_ms"] = round((perf_counter() - started) * 1000.0, 4)
    # Single-output task: a dict, never a singleton tuple.
    return result
