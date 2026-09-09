"""FabricFlow task wrapper for the RVL-CDIP-oriented rule classifier."""

from __future__ import annotations

from time import perf_counter

from core.task import task

from tasks.document.rules_classifier_core import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    classify_with_rules,
)


@task(
    outputs={
        "classification": {
            "type": "dict",
            "description": "Document family, confidence, decision, rule evidence, and classifier latency",
        }
    },
    display_name="Classify Document With Rules",
    description="Classify an aggregated scanned document using deterministic textual and layout rules",
    category="document",
    parameters={
        "classification_features": {
            "type": "dict",
            "required": True,
            "description": "Output of extract_document_classification_features",
        },
        "confidence_threshold": {
            "type": "float",
            "required": False,
            "default": DEFAULT_CONFIDENCE_THRESHOLD,
            "min": 0.0,
            "max": 1.0,
            "description": "Minimum winning rule score required for classification",
        },
        "min_score_margin": {
            "type": "float",
            "required": False,
            "default": DEFAULT_MIN_SCORE_MARGIN,
            "min": 0.0,
            "max": 1.0,
            "description": "Minimum score gap between the two leading families",
        },
        "min_recognized_characters": {
            "type": "int",
            "required": False,
            "default": DEFAULT_MIN_RECOGNIZED_CHARACTERS,
            "description": "Minimum OCR alphanumeric characters required for a semantic decision",
        },
        "classification_mode": {
            "type": "str",
            "required": False,
            "default": "observe",
            "description": "Execution mode: observe, evaluate, or auto",
        },
    },
)
def classify_document_rules(
    classification_features: dict,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_score_margin: float = DEFAULT_MIN_SCORE_MARGIN,
    min_recognized_characters: int = DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    classification_mode: str = "observe",
) -> dict:
    started = perf_counter()
    result = classify_with_rules(
        classification_features,
        confidence_threshold=confidence_threshold,
        min_score_margin=min_score_margin,
        min_recognized_characters=min_recognized_characters,
        mode=classification_mode,
    )
    result["execution_time_ms"] = round((perf_counter() - started) * 1000.0, 4)
    return result

