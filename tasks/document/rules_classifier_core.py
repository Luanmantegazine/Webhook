"""Pure rule-based document classification for Hydra.

This module deliberately has no dependency on FabricFlow's ``@task`` decorator.
It can therefore be unit-tested and reused by task wrappers, benchmark scripts,
and offline calibration notebooks.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
TAXONOMY_VERSION = "rvl-cdip-1.0"
CLASSIFIER_VERSION = "rules-rvl-cdip-v1"

DOCUMENT_FAMILIES = (
    "research_paper",
    "technical_report",
    "business_report",
    "financial_document",
    "form_structured",
    "presentation_marketing",
    "other",
)

TEXT_CLASSES = frozenset(
    {
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "Caption",
        "Footnote",
    }
)

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_CURRENCY_RE = re.compile(
    r"(?:[$€£¥]\s?\d|\b(?:usd|eur|gbp|brl|cad|aud|jpy)\b|\bR\$\s?\d)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(?:\d{1,3}(?:\s*[,;-]\s*\d{1,3})*)\]")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+){0,3}|[IVXLC]+|[A-Z])[.)]?\s+[A-Z][^\n]{2,80}$"
)
_FIELD_LABEL_RE = re.compile(
    r"(?m)^\s*[A-Za-z][A-Za-z0-9 /_-]{1,35}:\s*(?:$|[_\.]{2,}|\S.{0,25}$)"
)
_CHECKBOX_RE = re.compile(r"(?:\[\s?[xX]?\s?\]|☐|☑|□|■|\(\s?\))")
_BLANK_FIELD_RE = re.compile(r"(?:_{3,}|\.{5,})")


def _normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    lines = [" ".join(line.split()) for line in normalized.splitlines()]
    return "\n".join(line for line in lines if line)


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _flatten_table_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        cleaned = _normalise_text(value)
        if cleaned:
            yield cleaned
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"image", "image_base64"}:
                yield from _flatten_table_text(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_table_text(child)


def _page_size_for(
    page_index: int,
    regions: list[dict],
    page_sizes: list | None,
) -> tuple[float, float] | None:
    if isinstance(page_sizes, list) and page_index < len(page_sizes):
        candidate = page_sizes[page_index]
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                width, height = float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                width, height = 0.0, 0.0
            if width > 0 and height > 0:
                return width, height

    boxes = [_valid_bbox(region.get("bbox")) for region in regions]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    left_margin = min(box[0] for box in boxes)
    top_margin = min(box[1] for box in boxes)
    width = max(box[2] for box in boxes) + max(0.0, left_margin)
    height = max(box[3] for box in boxes) + max(0.0, top_margin)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _looks_two_column(regions: list[dict], page_width: float) -> bool:
    if page_width <= 0:
        return False
    centers: list[float] = []
    for region in regions:
        if region.get("class_name") not in TEXT_CLASSES | {"Table", "Picture"}:
            continue
        bbox = _valid_bbox(region.get("bbox"))
        if bbox is None:
            continue
        if (bbox[2] - bbox[0]) >= 0.60 * page_width:
            continue
        centers.append((bbox[0] + bbox[2]) / 2.0)

    centers.sort()
    if len(centers) < 4:
        return False

    gaps = [
        (centers[index + 1] - centers[index], index)
        for index in range(len(centers) - 1)
    ]
    best_gap, split_index = max(gaps)
    split = (centers[split_index] + centers[split_index + 1]) / 2.0
    left_count = split_index + 1
    right_count = len(centers) - left_count
    min_side = max(2, math.ceil(0.25 * len(centers)))
    return (
        best_gap >= 0.12 * page_width
        and 0.30 * page_width <= split <= 0.65 * page_width
        and left_count >= min_side
        and right_count >= min_side
    )


def extract_classification_features(
    document: dict,
    page_sizes: list | None = None,
    max_text_chars: int = 20_000,
) -> dict:
    """Build JSON-serializable textual and layout features.

    ``page_sizes`` should be the measured output of
    ``detect_and_extract_layout_doctr``. Bbox-based inference is only a
    compatibility fallback for older cached documents.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dict")
    try:
        text_limit = max(1_000, int(max_text_chars))
    except (TypeError, ValueError):
        text_limit = 20_000

    pages = document.get("pages")
    if not isinstance(pages, list):
        pages = []

    class_counts: Counter[str] = Counter()
    text_parts: list[str] = []
    text_region_count = 0
    nonempty_text_regions = 0
    valid_bbox_count = 0
    page_orientations: list[str] = []
    two_column_pages = 0

    for page_index, page in enumerate(pages):
        regions = page.get("regions", []) if isinstance(page, dict) else []
        if not isinstance(regions, list):
            regions = []
        typed_regions = [region for region in regions if isinstance(region, dict)]
        size = _page_size_for(page_index, typed_regions, page_sizes)
        if size is not None:
            width, height = size
            page_orientations.append("landscape" if width > height else "portrait")
            if _looks_two_column(typed_regions, width):
                two_column_pages += 1

        for region in typed_regions:
            class_name = str(region.get("class_name") or "Unknown")
            class_counts[class_name] += 1
            if _valid_bbox(region.get("bbox")) is not None:
                valid_bbox_count += 1

            if class_name in TEXT_CLASSES:
                text_region_count += 1
                region_text = _normalise_text(region.get("text"))
                if region_text:
                    nonempty_text_regions += 1
                    text_parts.append(region_text)
            elif class_name == "Table":
                region_text = _normalise_text(region.get("text"))
                if region_text:
                    text_parts.append(region_text)
                else:
                    text_parts.extend(_flatten_table_text(region.get("table_data", {})))

    if not text_parts:
        fallback_text = _normalise_text(document.get("full_text"))
        if fallback_text:
            text_parts.append(fallback_text)

    text = "\n".join(text_parts)
    text = text[:text_limit]
    words = _WORD_RE.findall(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    alnum_characters = sum(character.isalnum() for character in text)
    total_regions = sum(class_counts.values())
    total_pages = len(pages)
    measured_pages = len(page_orientations)

    currency_matches = len(_CURRENCY_RE.findall(text))
    citation_matches = len(_CITATION_RE.findall(text))
    doi_matches = len(_DOI_RE.findall(text))
    checkbox_matches = len(_CHECKBOX_RE.findall(text))
    blank_field_matches = len(_BLANK_FIELD_RE.findall(text))
    field_label_matches = len(_FIELD_LABEL_RE.findall(text))
    numbered_heading_matches = len(_NUMBERED_HEADING_RE.findall(text))

    def density(class_name: str) -> float:
        return class_counts.get(class_name, 0) / total_regions if total_regions else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "classification_text": text,
        "total_pages": total_pages,
        "total_regions": total_regions,
        "class_counts": dict(sorted(class_counts.items())),
        "word_count": len(words),
        "alnum_character_count": alnum_characters,
        "line_count": len(lines),
        "average_words_per_page": len(words) / max(1, total_pages),
        "average_words_per_text_region": len(words) / max(1, nonempty_text_regions),
        "text_region_count": text_region_count,
        "nonempty_text_region_count": nonempty_text_regions,
        "empty_text_region_ratio": (
            (text_region_count - nonempty_text_regions) / text_region_count
            if text_region_count
            else 1.0
        ),
        "valid_bbox_ratio": valid_bbox_count / total_regions if total_regions else 0.0,
        "table_density": density("Table"),
        "picture_density": density("Picture"),
        "formula_density": density("Formula"),
        "list_density": density("List-item"),
        "title_density": density("Title"),
        "section_header_density": density("Section-header"),
        "landscape_ratio": (
            page_orientations.count("landscape") / measured_pages
            if measured_pages
            else 0.0
        ),
        "two_column_ratio": two_column_pages / measured_pages if measured_pages else 0.0,
        "currency_match_count": currency_matches,
        "currency_matches_per_1000_words": currency_matches * 1000 / max(1, len(words)),
        "citation_match_count": citation_matches,
        "doi_match_count": doi_matches,
        "checkbox_count": checkbox_matches,
        "blank_field_count": blank_field_matches,
        "field_label_count": field_label_matches,
        "numbered_heading_count": numbered_heading_matches,
    }


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.MULTILINE) is not None


def _add_rule(
    scores: dict[str, float],
    evidence: dict[str, list[dict]],
    family: str,
    rule: str,
    weight: float,
    matched: bool,
) -> None:
    if not matched:
        return
    scores[family] += weight
    evidence[family].append({"rule": rule, "weight": weight})


def classify_with_rules(
    features: dict,
    confidence_threshold: float = 0.45,
    min_score_margin: float = 0.08,
    min_recognized_characters: int = 20,
    mode: str = "observe",
) -> dict:
    """Classify extracted Hydra features with explainable weighted rules."""
    if not isinstance(features, dict):
        raise TypeError("features must be a dict")
    if mode not in {"observe", "evaluate", "auto"}:
        raise ValueError("mode must be one of: observe, evaluate, auto")

    text = str(features.get("classification_text") or "")
    word_count = int(features.get("word_count") or 0)
    recognized = int(features.get("alnum_character_count") or 0)
    scores = {family: 0.0 for family in DOCUMENT_FAMILIES if family != "other"}
    evidence = {family: [] for family in scores}

    # Research paper
    _add_rule(scores, evidence, "research_paper", "abstract_heading", 0.25,
              _contains(text, r"(?:^|\n)\s*abstract\b|\babstract\s*[:—-]"))
    _add_rule(scores, evidence, "research_paper", "references_heading", 0.22,
              _contains(text, r"(?:^|\n)\s*(?:references|bibliography)\s*$"))
    _add_rule(scores, evidence, "research_paper", "doi", 0.18,
              int(features.get("doi_match_count") or 0) > 0)
    _add_rule(scores, evidence, "research_paper", "citations", 0.14,
              int(features.get("citation_match_count") or 0) >= 2)
    _add_rule(scores, evidence, "research_paper", "academic_sections", 0.16,
              _contains(text, r"\b(?:methodology|methods|experimental results|related work|conclusion)\b"))
    _add_rule(scores, evidence, "research_paper", "formula_layout", 0.08,
              float(features.get("formula_density") or 0.0) >= 0.03)
    _add_rule(scores, evidence, "research_paper", "two_column_layout", 0.08,
              float(features.get("two_column_ratio") or 0.0) >= 0.5)

    # Technical report / specification
    _add_rule(scores, evidence, "technical_report", "technical_report_phrase", 0.30,
              _contains(text, r"\btechnical (?:report|specification)\b"))
    _add_rule(scores, evidence, "technical_report", "requirements_language", 0.22,
              _contains(text, r"\b(?:system requirements?|functional requirements?|shall comply|specification)\b"))
    _add_rule(scores, evidence, "technical_report", "engineering_sections", 0.18,
              _contains(text, r"\b(?:scope|architecture|implementation|validation|test procedure)\b"))
    _add_rule(scores, evidence, "technical_report", "numbered_headings", 0.14,
              int(features.get("numbered_heading_count") or 0) >= 2)
    _add_rule(scores, evidence, "technical_report", "structured_tables", 0.08,
              float(features.get("table_density") or 0.0) >= 0.08)

    # Business report / budget
    _add_rule(scores, evidence, "business_report", "executive_summary", 0.25,
              _contains(text, r"\bexecutive summary\b"))
    _add_rule(scores, evidence, "business_report", "reporting_period", 0.22,
              _contains(text, r"\b(?:quarterly|annual report|fiscal year|year ended|q[1-4])\b"))
    _add_rule(scores, evidence, "business_report", "business_metrics", 0.22,
              _contains(text, r"\b(?:revenue|expenses?|forecast|budget|variance|key performance indicators?|kpis?)\b"))
    _add_rule(scores, evidence, "business_report", "business_tables", 0.10,
              float(features.get("table_density") or 0.0) >= 0.08)
    _add_rule(scores, evidence, "business_report", "multi_section_report", 0.08,
              float(features.get("section_header_density") or 0.0) >= 0.08)

    # Invoice / transactional financial document
    _add_rule(scores, evidence, "financial_document", "invoice_identifier", 0.42,
              _contains(text, r"\b(?:invoice|invoice\s*(?:no\.?|number|#))\b"))
    _add_rule(scores, evidence, "financial_document", "billing_parties", 0.20,
              _contains(text, r"\b(?:bill to|ship to|sold to|remit to|vendor)\b"))
    _add_rule(scores, evidence, "financial_document", "amount_due", 0.24,
              _contains(text, r"\b(?:amount due|balance due|total due|payment due)\b"))
    _add_rule(scores, evidence, "financial_document", "subtotal_tax_total", 0.20,
              _contains(text, r"\bsubtotal\b") and _contains(text, r"\b(?:tax|vat)\b")
              and _contains(text, r"\btotal\b"))
    _add_rule(scores, evidence, "financial_document", "currency_values", 0.12,
              int(features.get("currency_match_count") or 0) >= 2)
    _add_rule(scores, evidence, "financial_document", "transaction_table", 0.10,
              float(features.get("table_density") or 0.0) >= 0.08)

    # Form / questionnaire
    _add_rule(scores, evidence, "form_structured", "form_heading", 0.28,
              _contains(text, r"\b(?:application|registration|request|survey|questionnaire) form\b"))
    _add_rule(scores, evidence, "form_structured", "questionnaire_heading", 0.26,
              _contains(text, r"\b(?:questionnaire|survey)\b"))
    _add_rule(scores, evidence, "form_structured", "checkboxes", 0.22,
              int(features.get("checkbox_count") or 0) >= 2)
    _add_rule(scores, evidence, "form_structured", "blank_fields", 0.18,
              int(features.get("blank_field_count") or 0) >= 2)
    _add_rule(scores, evidence, "form_structured", "field_labels", 0.22,
              int(features.get("field_label_count") or 0) >= 3)
    _add_rule(scores, evidence, "form_structured", "short_field_regions", 0.10,
              int(features.get("text_region_count") or 0) >= 5
              and float(features.get("average_words_per_text_region") or 0.0) <= 8.0)

    # Presentation or advertisement. Layout signals are intentionally central
    # because promotional pages may have very little OCR text.
    _add_rule(scores, evidence, "presentation_marketing", "landscape_layout", 0.30,
              float(features.get("landscape_ratio") or 0.0) >= 0.5)
    _add_rule(scores, evidence, "presentation_marketing", "visual_layout", 0.20,
              float(features.get("picture_density") or 0.0) >= 0.15)
    _add_rule(scores, evidence, "presentation_marketing", "low_text_density", 0.16,
              word_count <= 120)
    _add_rule(scores, evidence, "presentation_marketing", "bullet_layout", 0.14,
              float(features.get("list_density") or 0.0) >= 0.15)
    _add_rule(scores, evidence, "presentation_marketing", "presentation_terms", 0.24,
              _contains(text, r"\b(?:agenda|presentation|our products?|limited time|special offer)\b"))

    ranked = sorted(scores.items(), key=lambda item: (-min(item[1], 1.0), item[0]))
    top_family, raw_top_score = ranked[0]
    runner_up_family, raw_runner_up_score = ranked[1]
    top_score = round(min(raw_top_score, 1.0), 4)
    runner_up_score = round(min(raw_runner_up_score, 1.0), 4)
    margin = round(top_score - runner_up_score, 4)

    if recognized < int(min_recognized_characters):
        selected_family = "other"
        decision = "abstained"
        reason = "insufficient_ocr_text"
        confidence = 0.0
    elif top_score == 0.0:
        selected_family = "other"
        decision = "fallback"
        reason = "no_rules_matched"
        confidence = 0.0
    elif top_score < float(confidence_threshold):
        selected_family = "other"
        decision = "fallback"
        reason = "score_below_threshold"
        confidence = top_score
    elif margin < float(min_score_margin):
        selected_family = "other"
        decision = "abstained"
        reason = "ambiguous_rule_scores"
        confidence = top_score
    else:
        selected_family = top_family
        decision = "classified"
        reason = "high_confidence_rule_match"
        confidence = top_score

    candidate_scores = {
        family: round(min(score, 1.0), 4)
        for family, score in sorted(scores.items())
    }
    candidate_scores["other"] = 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "classifier": "rules",
        "mode": mode,
        "document_family": selected_family,
        "confidence": confidence,
        "decision": decision,
        "reason": reason,
        "top_candidate": top_family if top_score > 0.0 else None,
        "runner_up": runner_up_family if runner_up_score > 0.0 else None,
        "score_margin": margin,
        "candidate_scores": candidate_scores,
        "evidence": {
            "rules_triggered": evidence[top_family],
            "top_features": {
                key: features.get(key)
                for key in (
                    "word_count",
                    "table_density",
                    "picture_density",
                    "formula_density",
                    "list_density",
                    "landscape_ratio",
                    "two_column_ratio",
                    "currency_match_count",
                    "citation_match_count",
                    "field_label_count",
                    "checkbox_count",
                )
            },
        },
        "thresholds": {
            "confidence": float(confidence_threshold),
            "minimum_score_margin": float(min_score_margin),
            "minimum_recognized_characters": int(min_recognized_characters),
        },
        "recommended_template": None,
        "fallback_template": "clean_article",
    }
