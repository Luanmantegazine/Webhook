"""FabricFlow task wrapper for shared classification feature extraction."""

from core.task import task

from tasks.document.rules_classifier_core import extract_classification_features as _extract


@task(
    outputs={
        "classification_features": {
            "type": "dict",
            "description": "Normalized textual, OCR-quality, and layout features used by document classifiers",
        }
    },
    display_name="Extract Document Classification Features",
    description="Build reusable document-classification features from aggregate_document_content and measured page sizes",
    category="document",
    parameters={
        "document": {
            "type": "dict",
            "required": True,
            "description": "Output document from aggregate_document_content",
        },
        "page_sizes": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Measured [width_px, height_px] for every rendered page",
        },
        "max_text_chars": {
            "type": "int",
            "required": False,
            "default": 20000,
            "description": "Maximum number of normalized OCR characters evaluated",
        },
    },
)
def extract_document_classification_features(
    document: dict,
    page_sizes: list | None = None,
    max_text_chars: int = 20_000,
) -> dict:
    return _extract(
        document,
        page_sizes=page_sizes,
        max_text_chars=max_text_chars,
    )