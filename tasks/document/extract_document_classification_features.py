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
        "page_words": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Per-page word-level OCR results from detect_and_extract_layout_doctr",
        },
        "provenance": {
            "type": "dict",
            "required": False,
            "default": None,
            "description": "Optional explicit extraction provenance override",
        },
        "model_repository": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Layout model repository identifier",
        },
        "model_filename": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Layout model filename",
        },
        "doctr_det_arch": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "docTR detection architecture",
        },
        "doctr_reco_arch": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "docTR recognition architecture",
        },
        "table_detection_model_name": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Table detection model name",
        },
        "table_structure_model_name": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Table structure model name",
        },
        "dpi": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Render DPI used during extraction",
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
    page_words: list | None = None,
    provenance: dict | None = None,
    model_repository: str = "",
    model_filename: str = "",
    doctr_det_arch: str = "",
    doctr_reco_arch: str = "",
    table_detection_model_name: str = "",
    table_structure_model_name: str = "",
    dpi: int = 0,
    max_text_chars: int = 20_000,
) -> dict:
    inferred_provenance = {
        key: value
        for key, value in {
            "layout_model_repository": model_repository,
            "layout_model_filename": model_filename,
            "ocr_engine": "doctr" if doctr_det_arch or doctr_reco_arch else "",
            "ocr_det_arch": doctr_det_arch,
            "ocr_reco_arch": doctr_reco_arch,
            "table_detection_model": table_detection_model_name,
            "table_structure_model": table_structure_model_name,
            "render_dpi": int(dpi) if dpi else None,
        }.items()
        if value not in ("", None)
    }
    resolved_provenance = dict(provenance) if isinstance(provenance, dict) else inferred_provenance
    return _extract(
        document,
        page_sizes=page_sizes,
        page_words=page_words,
        max_text_chars=max_text_chars,
        provenance=resolved_provenance,
    )
