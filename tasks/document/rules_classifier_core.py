"""Pure rule-based document classification for Hydra.

This module deliberately has no dependency on FabricFlow's ``@task`` decorator.
It can therefore be unit-tested and reused by task wrappers, benchmark scripts,
and offline calibration notebooks.

Design contract
---------------
Four responsibilities are kept strictly separate so that each can be inspected,
replaced or calibrated on its own:

1. **Feature extraction** (:func:`extract_classification_features`) turns a
   layout-annotated document into a JSON-serialisable, unit-free record of
   textual and geometric observations. It never decides anything.
2. **Rule evaluation** (:func:`evaluate_rules`) turns that record into a binary
   indicator vector. Rules are declared as data (:data:`RULES`), so the set of
   observable predicates is auditable and stable independent of how it is
   weighted.
3. **Scoring** (:func:`score_families`) maps indicators to a per-family score in
   ``[0, 1]``. Evidence that is *substitutable* is grouped, and only the
   strongest member of a group contributes; the normalising denominator is the
   evidence mass actually available for the document, so families with
   different numbers of rules remain comparable under a single threshold.
4. **Decision policy** (:func:`apply_decision_policy`) turns scores into a
   family, an abstention, or a fallback. It is a pure function of the scores and
   the thresholds, which is what allows a risk-coverage curve to be swept
   offline without re-running the rules.

The weights shipped in :data:`DEFAULT_WEIGHTS` are a documented prior, not a
calibrated model. Any experimental claim should either calibrate them on a
development split or compare them against weights fitted from labelled data;
both paths consume the indicator vector, not the rule definitions.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any, Callable, Iterable, NamedTuple


SCHEMA_VERSION = "2.0"
TAXONOMY_VERSION = "rvl-cdip-2.0"
CLASSIFIER_VERSION = "rules-rvl-cdip-v3"

#: Provisional operating point, and the single source of truth for it. The task
#: wrapper and the offline evaluator import these instead of restating them:
#: the previous revision carried a v1 operating point (0.45/0.08) in both while
#: the scorer had already moved to the normalised v2 ratio, so the thresholds
#: the benchmark reported were not the thresholds the module documented. These
#: values are *not* calibrated — tune them on the validation split and freeze
#: them before the test split is touched.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_MIN_SCORE_MARGIN = 0.10
DEFAULT_MIN_RECOGNIZED_CHARACTERS = 20

#: Document families. ``other`` is both the residual class and the destination
#: of every abstention; :func:`classify_with_rules` reports ``decision`` and
#: ``reason`` so the two can be told apart downstream. Evaluation code must
#: never treat them as the same outcome.
DOCUMENT_FAMILIES = (
    "research_paper",
    "technical_report",
    "business_report",
    "financial_document",
    "form_structured",
    "presentation_marketing",
    "correspondence",
    "resume",
    "news_article",
    "other",
)

SCORED_FAMILIES = tuple(family for family in DOCUMENT_FAMILIES if family != "other")

FAMILY_TEMPLATES = {
    "research_paper": "two_column_academic",
    "technical_report": "numbered_technical",
    "business_report": "sectioned_report",
    "financial_document": "tabular_financial",
    "form_structured": "field_grid",
    "presentation_marketing": "visual_landscape",
    "correspondence": "letter_block",
    "resume": "cv_columns",
    "news_article": "news_columns",
}

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

#: Region classes that carry the running text of a page. Used only for the
#: column-detection geometry, where captions and tables are noise.
TEXT_FLOW_CLASSES = frozenset({"Text", "Title", "List-item", "Footnote", "Section-header"})
#: Region classes excluded from column detection because their placement is
#: driven by the figure they annotate rather than by the text grid. ``Caption``
#: appears here *and* in :data:`TEXT_CLASSES`: it contributes text but not
#: layout evidence.
NON_FLOW_CLASSES = frozenset({"Table", "Picture", "Caption", "Formula"})


# --------------------------------------------------------------------------
# Lexicons and surface patterns
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_CURRENCY_RE = re.compile(
    r"(?:[$€£¥]\s?\d|\b(?:usd|eur|gbp|brl|cad|aud|jpy)\b|\bR\$\s?\d)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(?:\d{1,3}(?:\s*[,;-]\s*\d{1,3})*)\]")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

# A heading is a *numbered* line whose title has at least two words. Single
# capital letters and single roman characters are excluded: they matched every
# lettered list item in the previous revision.
_NUMBERED_HEADING_RE = re.compile(
    # A hierarchical number may stand alone ("3.1 Scope"); a bare number or a
    # roman numeral must carry a separator, otherwise any line opening with a
    # quantity would qualify.
    r"(?m)^\s*(?:\d{1,2}(?:\.\d{1,2}){1,3}[.)]?|(?:\d{1,2}|[IVX]{2,6})[.)])\s+"
    r"[A-Z][A-Za-z-]+(?:\s+[\w,'()-]+){1,10}\s*$"
)

# Captures the label so that correspondence headers (To/From/Subject/...) can be
# separated from genuine form fields; the two are surface-identical but are
# evidence for different families.
_FIELD_LABEL_RE = re.compile(
    r"(?m)^\s*([A-Za-z][A-Za-z0-9 /_-]{0,35}):[ \t]*(?:$|[_\.]{2,}|\S.{0,25}$)"
)
_CORRESPONDENCE_LABELS = frozenset(
    {
        "to",
        "from",
        "subject",
        "re",
        "cc",
        "bcc",
        "date",
        "sent",
        "attn",
        "attention",
        "fwd",
        "reply to",
        "copies to",
        "distribution",
    }
)

_CHECKBOX_RE = re.compile(r"(?:\[\s?[xX]?\s?\]|☐|☑|□|■|\(\s?\))")
_BLANK_FIELD_RE = re.compile(r"(?:_{3,}|\.{5,})")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\(?\d[\d,]*(?:\.\d+)?\)?(?![\w.])")
_ACCOUNTING_NEGATIVE_RE = re.compile(r"\(\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*\)")
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

# The two commercial lexicons are disjoint on purpose. Sharing tokens between
# them made every budget-like document fire both the business and the financial
# vocabulary rules, and the family with the larger evidence mass always won.
_ACCOUNTING_TERM_RE = re.compile(
    r"\b(?:balance|balance sheet|ledger|remittance|disbursements?|subtotal|"
    r"invoices?|accounts (?:payable|receivable)|expenditures?|receipts?|"
    r"income statement|cash flow|financial statements?|audit|"
    r"budget|estimate|amount due|unit price)\b",
    re.IGNORECASE,
)
_BUSINESS_TERM_RE = re.compile(
    r"\b(?:revenues?|expenses?|forecasts?|variance|market share|objectives?|"
    r"strategy|strategic|key performance indicators?|kpis?|sales volume|"
    r"profitability|growth|performance review|action items?)\b",
    re.IGNORECASE,
)

_CORRESPONDENCE_HEADER_RE = re.compile(
    r"(?m)^\s*(to|from|subject|re|cc|bcc|date|sent|attn|attention)\s*:", re.IGNORECASE
)
_SALUTATION_RE = re.compile(
    r"(?m)^\s*(?:dear\b|to whom it may concern\b|gentlemen\s*[:,])", re.IGNORECASE
)
_CLOSING_RE = re.compile(
    r"\b(?:sincerely(?:\s+yours)?|yours (?:truly|sincerely|very truly)|"
    r"best regards|kind regards|cordially|respectfully submitted)\b",
    re.IGNORECASE,
)
_MEMO_RE = re.compile(
    r"\b(?:inter[- ]?office\s+)?memorand(?:um|a)\b|^\s*memo\b",
    re.IGNORECASE | re.MULTILINE,
)
_EMAIL_MARKER_RE = re.compile(
    r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b|-{2,}\s*original message\s*-{2,}|^\s*sent:\s",
    re.IGNORECASE | re.MULTILINE,
)

_RESUME_HEADING_RE = re.compile(
    r"\bcurriculum vitae\b|^\s*(?:r[eé]sum[eé]|c\.?\s?v\.?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EXPERIENCE_SECTION_RE = re.compile(
    r"(?m)^\s*(?:work experience|professional experience|employment(?: history)?|"
    r"experience|positions? held|career (?:summary|history))\b",
    re.IGNORECASE,
)
_EDUCATION_SECTION_RE = re.compile(
    r"(?m)^\s*(?:education|academic (?:background|training)|degrees?|"
    r"qualifications)\b",
    re.IGNORECASE,
)
_RESUME_EXTRA_RE = re.compile(
    r"(?m)^\s*(?:skills|objective|honou?rs|awards|publications|"
    r"professional (?:affiliations|memberships)|references)\b",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:(?:19|20)\d{2}|present)\b", re.IGNORECASE
)

# Case is carried explicitly rather than by IGNORECASE: the capitalisation of
# the two following names is the discriminating part of the pattern.
_BYLINE_RE = re.compile(r"^\s*[Bb]y\s+[A-Z][a-zA-Z.'-]+\s+[A-Z][a-zA-Z.'-]+", re.MULTILINE)
_WIRE_SERVICE_RE = re.compile(
    r"\b(?:associated press|reuters|united press international|\(ap\)|\(upi\)|"
    r"staff (?:writer|reporter)|special to the|wire services?)\b",
    re.IGNORECASE,
)
_DATELINE_RE = re.compile(
    r"(?m)^\s*[A-Z][A-Z .]{2,25},\s*[A-Z][a-z]{2,9}\.?\s+\d{1,2}\b"
)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:said|says|told reporters|according to|commented)\b", re.IGNORECASE
)


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


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


def _table_data_text(table_data: Any) -> str:
    if not isinstance(table_data, dict):
        return ""
    columns = table_data.get("columns")
    rows = table_data.get("data")
    if not isinstance(columns, list) or not columns or not isinstance(rows, list):
        return "\n".join(_flatten_table_text(table_data))
    column_names = [str(column) for column in columns]
    lines = [
        " | ".join(column_names),
        " | ".join("---" for _ in column_names),
    ]
    for row in rows:
        if isinstance(row, dict):
            lines.append(" | ".join(str(row.get(column, "")) for column in columns))
        elif isinstance(row, (list, tuple)):
            lines.append(" | ".join(str(value) for value in row))
    return "\n".join(lines)


def _select_table_text(table_data: Any) -> str:
    """Select one stable textual representation for a table."""
    if not isinstance(table_data, dict):
        return ""
    for key in ("text_repr", "markdown", "tsv", "csv"):
        value = table_data.get(key)
        if (
            isinstance(value, str)
            and value.strip()
            and not value.strip().startswith("> [Table not extracted")
        ):
            return _normalise_text(value)
    return _normalise_text(_table_data_text(table_data))


def _text_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _overlap_over_smaller(first: list[float], second: list[float]) -> float:
    """Intersection over the smaller box.

    The containment measure, not IoU: a detector that emits both a fragment and
    the merged block around it produces boxes whose IoU is low but whose smaller
    box is almost entirely inside the larger one.
    """
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    smaller = min(
        (first[2] - first[0]) * (first[3] - first[1]),
        (second[2] - second[0]) * (second[3] - second[1]),
    )
    return intersection / smaller if smaller > 0 else 0.0


def _suppress_redundant_regions(
    regions: list[dict], overlap_threshold: float = 0.70
) -> list[dict]:
    """Drop detections that repeat text already covered by an overlapping region.

    A layout detector that fails to suppress overlapping candidates emits the
    same text two or three times: once as a fragment and once inside a merged
    block. That is an artefact of detection, not repetition in the document, and
    counting it inflates every per-occurrence feature. Suppression is decided
    geometrically **and** textually — overlapping boxes whose text is a repeat or
    a substring — so genuinely repeated, spatially distinct regions (the
    identical option pairs of a form, the repeated header of a table) still
    count once each, which is the signal the previous revision erased.
    """
    ordered = sorted(
        enumerate(regions),
        key=lambda item: -len(_normalise_text(item[1].get("text")) or ""),
    )
    dropped: set[int] = set()
    for position, (index, region) in enumerate(ordered):
        if index in dropped:
            continue
        bbox = _valid_bbox(region.get("bbox"))
        text = _text_key(_normalise_text(region.get("text")))
        if bbox is None or not text:
            continue
        for other_index, other in ordered[position + 1 :]:
            if other_index in dropped:
                continue
            other_bbox = _valid_bbox(other.get("bbox"))
            other_text = _text_key(_normalise_text(other.get("text")))
            if other_bbox is None or not other_text:
                continue
            if other_text in text and _overlap_over_smaller(bbox, other_bbox) >= overlap_threshold:
                dropped.add(other_index)
    return [region for index, region in enumerate(regions) if index not in dropped]


def _page_size_for(
    page_index: int,
    regions: list[dict],
    page_sizes: list | None,
) -> tuple[float, float] | None:
    """Return the measured page size, or ``None`` when it is unknown.

    Bbox-based inference was removed as a silent fallback: it underestimates the
    page by the unknown right and bottom margins, which inflates every area
    ratio without any signal that the value is unreliable. Pages without a
    measured size now contribute text but no geometry, and
    ``measured_page_ratio`` reports how much of the document that affects.
    """
    if isinstance(page_sizes, list) and page_index < len(page_sizes):
        candidate = page_sizes[page_index]
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                width, height = float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                return None
            if width > 0 and height > 0 and math.isfinite(width) and math.isfinite(height):
                return width, height
    return None


def _looks_two_column(
    regions: list[dict], page_width: float, page_height: float
) -> bool:
    """Detect a two-column text grid from region centroids.

    Both page dimensions are taken from the measured page size, so the area
    ratios computed here agree with the ones reported as features.
    """
    if page_width <= 0 or page_height <= 0:
        return False
    page_area = page_width * page_height
    centers: list[float] = []
    text_blocks: list[tuple[str, list[float]]] = []
    table_area = 0.0

    for region in regions:
        class_name = region.get("class_name")
        bbox = _valid_bbox(region.get("bbox"))
        if bbox is None:
            continue
        if class_name == "Table":
            table_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            continue
        if class_name in NON_FLOW_CLASSES:
            continue
        if class_name not in TEXT_FLOW_CLASSES:
            continue
        if not _normalise_text(region.get("text")):
            continue
        text_blocks.append((str(class_name), bbox))

    if table_area / page_area >= 0.35:
        return False
    if len(text_blocks) < 4:
        return False
    if sum(class_name == "Text" for class_name, _ in text_blocks) < 2:
        return False

    for _class_name, bbox in text_blocks:
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
    min_side = max(2, math.ceil(0.30 * len(centers)))
    return (
        best_gap >= 0.12 * page_width
        and 0.30 * page_width <= split <= 0.65 * page_width
        and left_count >= min_side
        and right_count >= min_side
    )


def _count_field_labels(text: str) -> tuple[int, int]:
    """Split ``Label:`` lines into form fields and correspondence headers.

    ``To:``/``From:``/``Date:``/``Subject:`` are surface-identical to form field
    labels but are evidence for a different family. Counting them as form fields
    made every memorandum, letter and e-mail in the corpus fire the form rules.
    Correspondence headers are counted as *distinct* labels so a repeated header
    cannot inflate the signal.
    """
    form_labels = 0
    correspondence_labels: set[str] = set()
    for match in _FIELD_LABEL_RE.finditer(text):
        label = " ".join(match.group(1).split()).casefold()
        if label in _CORRESPONDENCE_LABELS:
            correspondence_labels.add(label)
        else:
            form_labels += 1
    return form_labels, len(correspondence_labels)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------


def extract_classification_features(
    document: dict,
    page_sizes: list | None = None,
    max_text_chars: int = 20_000,
    provenance: dict | None = None,
) -> dict:
    """Build JSON-serializable textual and layout features.

    ``page_sizes`` must be the measured output of the layout detector, in the
    same coordinate system as the region bounding boxes. Pages without a
    measured size contribute text but no geometry; ``measured_page_ratio``
    reports the fraction of pages for which layout features are meaningful, and
    layout-dependent rules are excluded from scoring when it is zero.

    ``provenance`` should carry the identity of the extraction chain (layout
    detector and OCR engine plus their versions). It is echoed verbatim into the
    classification result: the features are entirely determined by that chain,
    so a result without it is not reproducible.
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
    raw_text_parts: list[str] = []
    text_keys: set[str] = set()
    text_region_count = 0
    nonempty_text_regions = 0
    valid_bbox_count = 0
    page_orientations: list[str] = []
    two_column_pages = 0
    picture_area = 0.0
    relevant_picture_count = 0
    table_area = 0.0
    page_area_total = 0.0
    region_confidences: list[float] = []
    suppressed_region_count = 0

    for page_index, page in enumerate(pages):
        regions = page.get("regions", []) if isinstance(page, dict) else []
        if not isinstance(regions, list):
            regions = []
        typed_regions = [region for region in regions if isinstance(region, dict)]
        deduplicated = _suppress_redundant_regions(typed_regions)
        suppressed_region_count += len(typed_regions) - len(deduplicated)
        typed_regions = deduplicated
        size = _page_size_for(page_index, typed_regions, page_sizes)
        page_area = size[0] * size[1] if size is not None else None

        if size is not None:
            width, height = size
            page_area_total += page_area
            page_orientations.append("landscape" if width > height else "portrait")
            if _looks_two_column(typed_regions, width, height):
                two_column_pages += 1

        for region in typed_regions:
            class_name = str(region.get("class_name") or "Unknown")
            class_counts[class_name] += 1
            # The detector's own confidence is evidence about the evidence. On a
            # low-resolution scan every downstream feature can be unreliable
            # while looking perfectly well-formed; discarding this left the
            # classifier unable to tell a clean document from a guess.
            try:
                confidence = float(region.get("confidence"))
            except (TypeError, ValueError):
                confidence = float("nan")
            if math.isfinite(confidence):
                region_confidences.append(confidence)
            bbox = _valid_bbox(region.get("bbox"))
            if bbox is not None:
                valid_bbox_count += 1
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else 0.0

            region_text = ""
            if class_name in TEXT_CLASSES:
                text_region_count += 1
                region_text = _normalise_text(region.get("text"))
                if region_text:
                    nonempty_text_regions += 1
            elif class_name == "Table":
                region_text = _select_table_text(region.get("table_data"))
                if not region_text:
                    region_text = _normalise_text(region.get("text"))

            if region_text:
                # The de-duplicated stream feeds the lexical rules, where a
                # repeated boilerplate line is noise. The raw stream feeds the
                # form counters, where repetition *is* the signal: identical
                # blank fields, checkboxes and short labels are exactly what
                # de-duplication used to erase.
                raw_text_parts.append(region_text)
                key = _text_key(region_text)
                if key not in text_keys:
                    text_keys.add(key)
                    text_parts.append(region_text)

            # Area ratios are only defined for pages with a measured size.
            if page_area:
                if class_name == "Table" and bbox is not None:
                    table_area += area
                elif class_name == "Picture" and bbox is not None:
                    picture_area += area
                    if area / page_area >= 0.02:
                        relevant_picture_count += 1

    if not text_parts:
        fallback_text = _normalise_text(document.get("full_text"))
        if fallback_text:
            text_parts.append(fallback_text)
            raw_text_parts.append(fallback_text)

    text = "\n".join(text_parts)[:text_limit]
    raw_text = "\n".join(raw_text_parts)[:text_limit]

    words = _WORD_RE.findall(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    alnum_characters = sum(character.isalnum() for character in text)
    total_regions = sum(class_counts.values())
    total_pages = len(pages)
    measured_pages = len(page_orientations)

    field_label_count, correspondence_header_count = _count_field_labels(raw_text)
    # A correspondence header block may be a single region, in which case the
    # de-duplicated and raw streams agree; take the stronger of the two counts.
    correspondence_header_count = max(
        correspondence_header_count,
        len({match.group(1).casefold() for match in _CORRESPONDENCE_HEADER_RE.finditer(text)}),
    )

    def density(class_name: str) -> float:
        return class_counts.get(class_name, 0) / total_regions if total_regions else 0.0

    def area_ratio(value: float) -> float:
        return value / page_area_total if page_area_total > 0 else 0.0

    accounting_term_count = len(_ACCOUNTING_TERM_RE.findall(text))
    currency_matches = len(_CURRENCY_RE.findall(text))

    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "provenance": dict(provenance) if isinstance(provenance, dict) else {},
        "classification_text": text,
        "total_pages": total_pages,
        "measured_page_count": measured_pages,
        "measured_page_ratio": measured_pages / total_pages if total_pages else 0.0,
        "total_regions": total_regions,
        "suppressed_region_count": suppressed_region_count,
        "layout_confidence_mean": (
            round(sum(region_confidences) / len(region_confidences), 4)
            if region_confidences
            else None
        ),
        "layout_confidence_min": round(min(region_confidences), 4) if region_confidences else None,
        "low_confidence_region_ratio": (
            round(sum(1 for value in region_confidences if value < 0.35) / len(region_confidences), 4)
            if region_confidences
            else None
        ),
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
            else 0.0
        ),
        "valid_bbox_ratio": valid_bbox_count / total_regions if total_regions else 0.0,
        "table_density": density("Table"),
        "picture_density": density("Picture"),
        "picture_area_ratio": area_ratio(picture_area),
        "relevant_picture_count": relevant_picture_count,
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
        "numeric_token_count": len(_NUMBER_RE.findall(text)),
        "month_match_count": len(_MONTH_RE.findall(text)),
        "accounting_term_count": accounting_term_count,
        # Retained under the previous name so existing consumers keep working;
        # the lexicon behind it is now disjoint from the business lexicon.
        "financial_term_count": accounting_term_count,
        "business_term_count": len(_BUSINESS_TERM_RE.findall(text)),
        "accounting_negative_count": len(_ACCOUNTING_NEGATIVE_RE.findall(text)),
        "table_area_ratio": area_ratio(table_area),
        "citation_match_count": len(_CITATION_RE.findall(text)),
        "doi_match_count": len(_DOI_RE.findall(text)),
        "checkbox_count": len(_CHECKBOX_RE.findall(raw_text)),
        "blank_field_count": len(_BLANK_FIELD_RE.findall(raw_text)),
        "field_label_count": field_label_count,
        "correspondence_header_count": correspondence_header_count,
        "numbered_heading_count": len(_NUMBERED_HEADING_RE.findall(text)),
        "year_range_count": len(_YEAR_RANGE_RE.findall(text)),
        "attribution_count": len(_ATTRIBUTION_RE.findall(text)),
    }


# --------------------------------------------------------------------------
# Rule declarations
# --------------------------------------------------------------------------


class _Context:
    """Read-only accessor passed to every rule predicate."""

    __slots__ = ("features", "text")

    def __init__(self, features: dict) -> None:
        self.features = features
        self.text = str(features.get("classification_text") or "")

    def has(self, pattern: re.Pattern[str]) -> bool:
        return pattern.search(self.text) is not None

    def num(self, key: str) -> int:
        try:
            return int(self.features.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    def flt(self, key: str) -> float:
        try:
            return float(self.features.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0


class RuleSpec(NamedTuple):
    """One observable predicate attached to one family.

    ``group`` marks *substitutable* evidence: within a group only the
    strongest fired rule contributes to the score, and the group contributes its
    strongest weight once to the normalising mass. This is what stops the same
    underlying observation (a dense table, a citation marker, a visual page)
    from being counted two or three times for the same family.

    ``requires`` names the evidence channel the rule depends on. A rule whose
    channel is unavailable for a document is excluded from both the score and
    the mass, so a corpus without that channel does not silently deflate the
    families that rely on it.
    """

    family: str
    name: str
    group: str
    requires: str  # "text" | "layout" | "multipage"
    predicate: Callable[[_Context], bool]

    @property
    def rule_id(self) -> str:
        return f"{self.family}.{self.name}"


def _rule(family: str, name: str, predicate, *, group: str = "", requires: str = "text") -> RuleSpec:
    return RuleSpec(family, name, group or name, requires, predicate)


# Patterns used only inside predicates.
_ABSTRACT_RE = re.compile(r"(?:^|\n)\s*abstract\b|\babstract\s*[:—-]", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"(?m)^\s*(?:references|bibliography)\s*$", re.IGNORECASE)
_ACADEMIC_SECTION_RE = re.compile(
    r"\b(?:methodology|methods|experimental results|related work|conclusions?)\b",
    re.IGNORECASE,
)
_TECH_REPORT_RE = re.compile(r"\btechnical (?:report|specification)\b", re.IGNORECASE)
_REQUIREMENTS_RE = re.compile(
    r"\b(?:system requirements?|functional requirements?|shall comply|specification)\b",
    re.IGNORECASE,
)
_ENGINEERING_SECTION_RE = re.compile(
    r"\b(?:scope|architecture|implementation|validation|test procedure)\b",
    re.IGNORECASE,
)
_EXEC_SUMMARY_RE = re.compile(r"\bexecutive summary\b", re.IGNORECASE)
_REPORTING_PERIOD_RE = re.compile(
    r"\b(?:quarterly|annual report|fiscal year|year ended|q[1-4])\b", re.IGNORECASE
)
_INVOICE_RE = re.compile(r"\binvoice\b", re.IGNORECASE)
_BILLING_PARTY_RE = re.compile(
    r"\b(?:bill to|ship to|sold to|remit to|vendor)\b", re.IGNORECASE
)
_AMOUNT_DUE_RE = re.compile(
    r"\b(?:amount due|balance due|total due|payment due)\b", re.IGNORECASE
)
_SUBTOTAL_RE = re.compile(r"\bsubtotal\b", re.IGNORECASE)
_TAX_RE = re.compile(r"\b(?:tax|vat)\b", re.IGNORECASE)
_TOTAL_RE = re.compile(r"\btotal\b", re.IGNORECASE)
_STATEMENT_RE = re.compile(
    r"\b(?:financial statements?|balance sheet|income statement|cash flow)\b",
    re.IGNORECASE,
)
_BEGINNING_BALANCE_RE = re.compile(r"\bbeginning balance\b", re.IGNORECASE)
_ENDING_BALANCE_RE = re.compile(r"\bending balance\b", re.IGNORECASE)
_FINANCIAL_UNITS_RE = re.compile(
    r"\b(?:dollars|amounts?)\s+in\s+(?:thousands?|millions?)\b", re.IGNORECASE
)
_FORM_HEADING_RE = re.compile(
    r"\b(?:application|registration|request|survey|questionnaire) form\b", re.IGNORECASE
)
_QUESTIONNAIRE_RE = re.compile(r"\b(?:questionnaire|survey)\b", re.IGNORECASE)
_PRESENTATION_TERM_RE = re.compile(
    r"\b(?:agenda|presentation|our products?|limited time|special offer)\b",
    re.IGNORECASE,
)


def _tables_dominate(ctx: _Context) -> bool:
    """Guard for families whose layout evidence is meaningless on table pages."""
    return ctx.flt("table_area_ratio") >= 0.35


def _visual_page(ctx: _Context) -> bool:
    return ctx.num("relevant_picture_count") >= 2 or ctx.flt("picture_area_ratio") >= 0.35


def _has_primary_correspondence_signal(ctx: _Context) -> bool:
    return (
        ctx.num("correspondence_header_count") >= 2
        or ctx.has(_SALUTATION_RE)
        or ctx.has(_CLOSING_RE)
        or ctx.has(_MEMO_RE)
        or ctx.has(_EMAIL_MARKER_RE)
    )


RULES: tuple[RuleSpec, ...] = (
    # ---- research paper -------------------------------------------------
    _rule("research_paper", "abstract_heading", lambda c: c.has(_ABSTRACT_RE)),
    _rule("research_paper", "references_heading", lambda c: c.has(_REFERENCES_RE)),
    _rule("research_paper", "doi", lambda c: c.num("doi_match_count") > 0, group="citation_evidence"),
    _rule("research_paper", "citations", lambda c: c.num("citation_match_count") >= 2, group="citation_evidence"),
    _rule("research_paper", "academic_sections", lambda c: c.has(_ACADEMIC_SECTION_RE)),
    _rule("research_paper", "formula_layout", lambda c: c.flt("formula_density") >= 0.03),
    _rule(
        "research_paper",
        "two_column_layout",
        lambda c: c.flt("two_column_ratio") >= 0.5,
        requires="layout",
    ),
    # ---- technical report ----------------------------------------------
    _rule("technical_report", "technical_report_phrase", lambda c: c.has(_TECH_REPORT_RE)),
    _rule("technical_report", "requirements_language", lambda c: c.has(_REQUIREMENTS_RE)),
    _rule("technical_report", "engineering_sections", lambda c: c.has(_ENGINEERING_SECTION_RE)),
    _rule("technical_report", "numbered_headings", lambda c: c.num("numbered_heading_count") >= 2),
    _rule("technical_report", "structured_tables", lambda c: c.flt("table_density") >= 0.08),
    # ---- business report ------------------------------------------------
    _rule("business_report", "executive_summary", lambda c: c.has(_EXEC_SUMMARY_RE)),
    _rule("business_report", "reporting_period", lambda c: c.has(_REPORTING_PERIOD_RE)),
    _rule("business_report", "business_vocabulary", lambda c: c.num("business_term_count") >= 3),
    _rule(
        "business_report",
        "business_tables",
        lambda c: c.flt("table_density") >= 0.08 and c.flt("table_area_ratio") < 0.35,
    ),
    _rule("business_report", "multi_section_report", lambda c: c.flt("section_header_density") >= 0.08),
    # Replaces the former ``narrative_balance`` penalty, whose sign contradicted
    # the family it was attached to: a sectioned narrative body is positive
    # evidence for a business report, not a reason to discount one.
    _rule(
        "business_report",
        "narrative_body",
        lambda c: c.num("word_count") >= 250
        and c.flt("section_header_density") >= 0.05
        and c.flt("table_area_ratio") < 0.25,
    ),
    # ---- financial document ---------------------------------------------
    _rule("financial_document", "invoice_identifier", lambda c: c.has(_INVOICE_RE)),
    _rule("financial_document", "billing_parties", lambda c: c.has(_BILLING_PARTY_RE)),
    _rule("financial_document", "amount_due", lambda c: c.has(_AMOUNT_DUE_RE), group="totals_block"),
    _rule(
        "financial_document",
        "subtotal_tax_total",
        lambda c: c.has(_SUBTOTAL_RE) and c.has(_TAX_RE) and c.has(_TOTAL_RE),
        group="totals_block",
    ),
    _rule("financial_document", "currency_values", lambda c: c.num("currency_match_count") >= 2),
    _rule("financial_document", "financial_statement_terms", lambda c: c.has(_STATEMENT_RE), group="statement_block"),
    _rule(
        "financial_document",
        "balance_period",
        lambda c: c.has(_BEGINNING_BALANCE_RE) and c.has(_ENDING_BALANCE_RE),
        group="statement_block",
    ),
    _rule("financial_document", "financial_units", lambda c: c.has(_FINANCIAL_UNITS_RE)),
    _rule("financial_document", "accounting_vocabulary", lambda c: c.num("accounting_term_count") >= 4),
    _rule("financial_document", "monthly_series", lambda c: c.num("month_match_count") >= 3),
    # The three former table rules collapsed into one: they tested the same
    # ``table_density`` observation and tripled its weight for this family.
    _rule(
        "financial_document",
        "numeric_table",
        lambda c: c.flt("table_density") >= 0.08
        and c.num("numeric_token_count") >= 8
        and (c.num("accounting_term_count") >= 3 or c.num("currency_match_count") >= 2),
    ),
    _rule("financial_document", "accounting_negatives", lambda c: c.num("accounting_negative_count") >= 2),
    # ---- form / questionnaire -------------------------------------------
    _rule("form_structured", "form_heading", lambda c: c.has(_FORM_HEADING_RE)),
    _rule("form_structured", "questionnaire_heading", lambda c: c.has(_QUESTIONNAIRE_RE)),
    _rule("form_structured", "checkboxes", lambda c: c.num("checkbox_count") >= 2),
    _rule("form_structured", "blank_fields", lambda c: c.num("blank_field_count") >= 2, group="labelled_blanks"),
    _rule("form_structured", "field_labels", lambda c: c.num("field_label_count") >= 3, group="labelled_blanks"),
    _rule(
        "form_structured",
        "short_field_regions",
        lambda c: c.num("text_region_count") >= 5
        and c.flt("average_words_per_text_region") <= 8.0,
    ),
    # ---- presentation / advertisement -----------------------------------
    # Landscape orientation and picture dominance are alternative expressions of
    # the same "visual page" evidence, so they share a group: a corpus of
    # portrait-only scans no longer deflates this family's attainable score.
    _rule(
        "presentation_marketing",
        "visual_layout",
        lambda c: not _tables_dominate(c) and _visual_page(c),
        group="visual_page",
        requires="layout",
    ),
    _rule(
        "presentation_marketing",
        "landscape_layout",
        lambda c: not _tables_dominate(c)
        and c.flt("landscape_ratio") >= 0.5
        and (c.num("relevant_picture_count") >= 2 or c.flt("picture_area_ratio") >= 0.25),
        group="visual_page",
        requires="layout",
    ),
    _rule(
        "presentation_marketing",
        "low_text_density",
        lambda c: not _tables_dominate(c) and c.num("word_count") <= 120,
    ),
    _rule(
        "presentation_marketing",
        "bullet_layout",
        lambda c: not _tables_dominate(c)
        and c.flt("list_density") >= 0.15
        and c.num("word_count") <= 180,
    ),
    _rule(
        "presentation_marketing",
        "presentation_terms",
        lambda c: not _tables_dominate(c) and c.has(_PRESENTATION_TERM_RE),
    ),
    # ---- correspondence (letter / memo / e-mail) -------------------------
    _rule("correspondence", "header_block", lambda c: c.num("correspondence_header_count") >= 2),
    _rule("correspondence", "salutation", lambda c: c.has(_SALUTATION_RE)),
    _rule("correspondence", "closing", lambda c: c.has(_CLOSING_RE)),
    _rule("correspondence", "memo_heading", lambda c: c.has(_MEMO_RE)),
    _rule("correspondence", "email_markers", lambda c: c.has(_EMAIL_MARKER_RE)),
    # Corroborating, never standing alone: "short, single-page, no tables"
    # describes most scanned documents, so on its own it is not evidence of
    # anything. It may only add to a case that a primary signal has opened.
    _rule(
        "correspondence",
        "letter_body",
        lambda c: _has_primary_correspondence_signal(c)
        and c.num("total_pages") <= 2
        and 40 <= c.num("word_count") <= 600
        and c.flt("table_density") < 0.05,
        requires="multipage",
    ),
    # ---- resume ----------------------------------------------------------
    _rule("resume", "resume_heading", lambda c: c.has(_RESUME_HEADING_RE)),
    _rule("resume", "experience_section", lambda c: c.has(_EXPERIENCE_SECTION_RE)),
    _rule("resume", "education_section", lambda c: c.has(_EDUCATION_SECTION_RE)),
    _rule("resume", "year_ranges", lambda c: c.num("year_range_count") >= 3),
    _rule("resume", "cv_sections", lambda c: c.has(_RESUME_EXTRA_RE)),
    # ---- news article ----------------------------------------------------
    _rule("news_article", "byline", lambda c: c.has(_BYLINE_RE)),
    _rule("news_article", "dateline", lambda c: c.has(_DATELINE_RE)),
    _rule("news_article", "wire_service", lambda c: c.has(_WIRE_SERVICE_RE)),
    _rule("news_article", "attribution_quotes", lambda c: c.num("attribution_count") >= 3),
    _rule(
        "news_article",
        "multi_column_body",
        lambda c: c.flt("two_column_ratio") >= 0.5 and c.num("word_count") >= 200,
        requires="layout",
    ),
)

RULE_IDS: tuple[str, ...] = tuple(rule.rule_id for rule in RULES)

#: Hand-set prior weights. These are *not* calibrated: the accompanying
#: evaluation module fits weights from labelled data and compares the two.
DEFAULT_WEIGHTS: dict[str, float] = {
    "research_paper.abstract_heading": 0.25,
    "research_paper.references_heading": 0.22,
    "research_paper.doi": 0.18,
    "research_paper.citations": 0.14,
    "research_paper.academic_sections": 0.16,
    "research_paper.formula_layout": 0.08,
    "research_paper.two_column_layout": 0.08,
    "technical_report.technical_report_phrase": 0.30,
    "technical_report.requirements_language": 0.22,
    "technical_report.engineering_sections": 0.18,
    "technical_report.numbered_headings": 0.14,
    "technical_report.structured_tables": 0.08,
    "business_report.executive_summary": 0.25,
    "business_report.reporting_period": 0.22,
    "business_report.business_vocabulary": 0.22,
    "business_report.business_tables": 0.12,
    "business_report.multi_section_report": 0.08,
    "business_report.narrative_body": 0.10,
    "financial_document.invoice_identifier": 0.42,
    "financial_document.billing_parties": 0.20,
    "financial_document.amount_due": 0.24,
    "financial_document.subtotal_tax_total": 0.20,
    "financial_document.currency_values": 0.12,
    "financial_document.financial_statement_terms": 0.24,
    "financial_document.balance_period": 0.32,
    "financial_document.financial_units": 0.18,
    "financial_document.accounting_vocabulary": 0.20,
    "financial_document.monthly_series": 0.18,
    "financial_document.numeric_table": 0.22,
    "financial_document.accounting_negatives": 0.10,
    "form_structured.form_heading": 0.28,
    "form_structured.questionnaire_heading": 0.26,
    "form_structured.checkboxes": 0.22,
    "form_structured.blank_fields": 0.18,
    "form_structured.field_labels": 0.22,
    "form_structured.short_field_regions": 0.10,
    "presentation_marketing.visual_layout": 0.22,
    "presentation_marketing.landscape_layout": 0.16,
    "presentation_marketing.low_text_density": 0.08,
    "presentation_marketing.bullet_layout": 0.12,
    "presentation_marketing.presentation_terms": 0.24,
    "correspondence.header_block": 0.30,
    "correspondence.salutation": 0.24,
    "correspondence.closing": 0.18,
    "correspondence.memo_heading": 0.26,
    "correspondence.email_markers": 0.22,
    "correspondence.letter_body": 0.08,
    "resume.resume_heading": 0.34,
    "resume.experience_section": 0.24,
    "resume.education_section": 0.22,
    "resume.year_ranges": 0.16,
    "resume.cv_sections": 0.12,
    "news_article.byline": 0.26,
    "news_article.dateline": 0.22,
    "news_article.wire_service": 0.20,
    "news_article.attribution_quotes": 0.16,
    "news_article.multi_column_body": 0.16,
}


def available_channels(features: dict) -> frozenset[str]:
    """Evidence channels that this document can actually supply."""
    channels = {"text"}
    try:
        if float(features.get("measured_page_ratio") or 0.0) > 0.0:
            channels.add("layout")
        if int(features.get("total_pages") or 0) > 0:
            channels.add("multipage")
    except (TypeError, ValueError):
        pass
    return frozenset(channels)


def evaluate_rules(features: dict) -> dict[str, bool]:
    """Evaluate every rule into a binary indicator vector.

    The vector is the interface between the rule set and any scorer, hand-set or
    fitted. Rules whose evidence channel is unavailable are reported as
    ``False``; :func:`score_families` excludes them from the denominator so
    their absence is not read as negative evidence.
    """
    if not isinstance(features, dict):
        raise TypeError("features must be a dict")
    ctx = _Context(features)
    channels = available_channels(features)
    indicators: dict[str, bool] = {}
    for rule in RULES:
        if rule.requires not in channels:
            indicators[rule.rule_id] = False
            continue
        try:
            indicators[rule.rule_id] = bool(rule.predicate(ctx))
        except Exception:  # a malformed feature must not abort classification
            indicators[rule.rule_id] = False
    return indicators


#: How many of a family's strongest evidence groups constitute a sufficient
#: case. The normalising denominator is the mass of that many groups, not the
#: family's total mass: dividing by the total would make a family with many
#: weak corroborating rules structurally unable to reach a decision on its two
#: or three decisive ones. Raising this makes every family harder to accept.
DECISION_GROUP_COUNT = 3


def score_families(
    features: dict,
    indicators: dict[str, bool],
    weights: dict[str, float] | None = None,
    decision_group_count: int = DECISION_GROUP_COUNT,
) -> dict[str, dict]:
    """Aggregate indicators into a comparable score per family.

    Three steps, each addressing one way the previous scorer was incomparable
    across families:

    * **Grouping.** Within a group only the strongest fired rule contributes, so
      one observation cannot be counted twice for the same family.
    * **Channel filtering.** Groups whose evidence channel the document cannot
      supply are dropped from both the numerator and the denominator, so a
      single-page portrait corpus does not silently penalise the families that
      rely on multi-page or landscape geometry.
    * **Normalisation by decisive mass.** The score is ``fired / decision_mass``
      where ``decision_mass`` is the mass of the family's strongest available
      groups. All families therefore reach 1.0 under a comparable amount of
      evidence, and a single threshold means the same thing for each.

    The score is an evidence ratio, deliberately **not** clipped here and
    deliberately not a probability: it may exceed 1.0 when a document carries
    more than a sufficient case, and ranking on the unclipped value is what
    keeps two strongly-evidenced families from tying at the ceiling and
    collapsing the margin. Calibrated probabilities come from fitting weights on
    labelled data, not from this function.
    """
    effective = DEFAULT_WEIGHTS if weights is None else weights
    channels = available_channels(features)

    groups: dict[tuple[str, str], dict] = {}
    for rule in RULES:
        if rule.requires not in channels:
            continue
        weight = float(effective.get(rule.rule_id, 0.0))
        slot = groups.setdefault(
            (rule.family, rule.group),
            {"capacity": 0.0, "fired_weight": 0.0, "winner": None, "members": []},
        )
        slot["members"].append(rule.rule_id)
        slot["capacity"] = max(slot["capacity"], weight)
        if indicators.get(rule.rule_id) and weight > slot["fired_weight"]:
            slot["fired_weight"] = weight
            slot["winner"] = rule.rule_id

    result: dict[str, dict] = {
        family: {
            "score": 0.0,
            "raw": 0.0,
            "available_mass": 0.0,
            "decision_mass": 0.0,
            "capacities": [],
            "counted": [],
            "suppressed": [],
        }
        for family in SCORED_FAMILIES
    }
    for (family, _group), slot in groups.items():
        entry = result[family]
        entry["available_mass"] += slot["capacity"]
        entry["capacities"].append(slot["capacity"])
        entry["raw"] += slot["fired_weight"]
        if slot["winner"]:
            entry["counted"].append(
                {"rule": slot["winner"], "weight": round(slot["fired_weight"], 4)}
            )
            entry["suppressed"].extend(
                member
                for member in slot["members"]
                if member != slot["winner"] and indicators.get(member)
            )

    top_n = max(1, int(decision_group_count))
    for entry in result.values():
        decision_mass = sum(sorted(entry["capacities"], reverse=True)[:top_n])
        entry["decision_mass"] = round(decision_mass, 4)
        entry["score"] = round(entry["raw"] / decision_mass, 4) if decision_mass > 0 else 0.0
        entry["raw"] = round(entry["raw"], 4)
        entry["available_mass"] = round(entry["available_mass"], 4)
        entry["counted"].sort(key=lambda item: (-item["weight"], item["rule"]))
        del entry["capacities"]
    return result


def apply_decision_policy(
    ranked: list[tuple[str, float]],
    recognized_characters: int,
    confidence_threshold: float,
    min_score_margin: float,
    min_recognized_characters: int,
    mode: str,
) -> dict:
    """Turn a ranked score list into a decision.

    Ranking, the threshold test and the margin all operate on the *unclipped*
    evidence ratio; only the reported ``confidence`` is clipped to ``[0, 1]``.
    Clipping before ranking was what let two well-evidenced families meet at the
    ceiling, produce a zero margin, and abstain at full confidence.

    Pure and cheap by design: a risk-coverage sweep re-runs only this function
    over stored scores, never the rule engine, so the curve is guaranteed to
    describe the same firings that produced the reported operating point.
    """
    top_family, top_score = ranked[0]
    runner_up_family, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    margin = round(top_score - runner_up_score, 4)
    reported = round(min(top_score, 1.0), 4)

    if mode == "observe":
        # Full-coverage mode: no abstention, so the confusion matrix is complete
        # and the abstention policy can be evaluated separately from the rules.
        if top_score <= 0.0:
            return {
                "document_family": "other",
                "decision": "observed",
                "reason": "no_rules_matched",
                "confidence": 0.0,
                "score": top_score,
                "score_margin": margin,
            }
        return {
            "document_family": top_family,
            "decision": "observed",
            "reason": "argmax_without_abstention",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        }

    if recognized_characters < int(min_recognized_characters):
        return {
            "document_family": "other",
            "decision": "abstained",
            "reason": "insufficient_ocr_text",
            "confidence": 0.0,
            "score": top_score,
            "score_margin": margin,
        }
    if top_score <= 0.0:
        return {
            "document_family": "other",
            "decision": "fallback",
            "reason": "no_rules_matched",
            "confidence": 0.0,
            "score": 0.0,
            "score_margin": margin,
        }
    if top_score < float(confidence_threshold):
        return {
            "document_family": "other",
            "decision": "fallback",
            "reason": "score_below_threshold",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        }
    if margin < float(min_score_margin):
        return {
            "document_family": "other",
            "decision": "abstained",
            "reason": "ambiguous_rule_scores",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        }
    return {
        "document_family": top_family,
        "decision": "classified",
        "reason": "score_above_threshold",
        "confidence": reported,
        "score": top_score,
        "score_margin": margin,
    }


def classify_with_rules(
    features: dict,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_score_margin: float = DEFAULT_MIN_SCORE_MARGIN,
    min_recognized_characters: int = DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    mode: str = "evaluate",
    weights: dict[str, float] | None = None,
    include_indicators: bool = False,
) -> dict:
    """Classify extracted Hydra features with explainable weighted rules.

    ``mode`` selects the decision regime, and each regime exists for a distinct
    purpose in the experimental protocol:

    ``observe``
        No abstention. The argmax family is always reported, which yields a
        full-coverage confusion matrix and isolates rule quality from the
        rejection policy.
    ``evaluate``
        Thresholds enforced; abstentions and fallbacks are reported with their
        reason. This is the regime whose risk-coverage curve should be reported.
    ``auto``
        As ``evaluate``, and additionally resolves the reconstruction template
        for the accepted family.

    ``confidence_threshold`` applies to the unclipped evidence ratio, where 1.0
    means "as much evidence as the family's strongest groups can supply". The
    default thresholds are provisional: normalisation makes them comparable
    across families, but their values must still be calibrated on a development
    split before any reported result depends on them.
    """
    if not isinstance(features, dict):
        raise TypeError("features must be a dict")
    if mode not in {"observe", "evaluate", "auto"}:
        raise ValueError("mode must be one of: observe, evaluate, auto")

    indicators = evaluate_rules(features)
    breakdown = score_families(features, indicators, weights)

    ranked = sorted(
        ((family, entry["score"]) for family, entry in breakdown.items()),
        key=lambda item: (-item[1], item[0]),
    )
    decision = apply_decision_policy(
        ranked,
        int(features.get("alnum_character_count") or 0),
        confidence_threshold,
        min_score_margin,
        min_recognized_characters,
        mode,
    )

    top_family, top_score = ranked[0]
    runner_up_family, runner_up_score = ranked[1]
    selected = decision["document_family"]

    candidate_scores = {family: entry["score"] for family, entry in sorted(breakdown.items())}
    candidate_scores["other"] = 0.0

    result = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "classifier": "rules",
        "mode": mode,
        "provenance": features.get("provenance") or {},
        "document_family": selected,
        # ``confidence`` is the reporting value, clipped to [0, 1] for
        # downstream consumers; ``score`` is the unclipped evidence ratio the
        # thresholds are actually applied to. They are not probabilities.
        "confidence": decision["confidence"],
        "score": decision["score"],
        "decision": decision["decision"],
        "reason": decision["reason"],
        "top_candidate": top_family if top_score > 0.0 else None,
        "runner_up": runner_up_family if runner_up_score > 0.0 else None,
        "score_margin": decision["score_margin"],
        "candidate_scores": candidate_scores,
        "decision_mass": {
            family: entry["decision_mass"] for family, entry in sorted(breakdown.items())
        },
        "available_mass": {
            family: entry["available_mass"] for family, entry in sorted(breakdown.items())
        },
        "evidence": {
            # Fired rules for every family, not only the winner: error analysis
            # needs to see what the losing families had.
            "rules_by_family": {
                family: entry["counted"]
                for family, entry in sorted(breakdown.items())
                if entry["counted"]
            },
            "rules_triggered": breakdown[top_family]["counted"],
            "suppressed_by_grouping": {
                family: entry["suppressed"]
                for family, entry in sorted(breakdown.items())
                if entry["suppressed"]
            },
            "top_features": {
                key: features.get(key)
                for key in (
                    "word_count",
                    "measured_page_ratio",
                    "layout_confidence_mean",
                    "low_confidence_region_ratio",
                    "suppressed_region_count",
                    "table_density",
                    "picture_density",
                    "formula_density",
                    "list_density",
                    "landscape_ratio",
                    "two_column_ratio",
                    "currency_match_count",
                    "numeric_token_count",
                    "month_match_count",
                    "accounting_term_count",
                    "business_term_count",
                    "table_area_ratio",
                    "picture_area_ratio",
                    "relevant_picture_count",
                    "citation_match_count",
                    "field_label_count",
                    "correspondence_header_count",
                    "checkbox_count",
                )
            },
        },
        "thresholds": {
            "confidence": float(confidence_threshold),
            "minimum_score_margin": float(min_score_margin),
            "minimum_recognized_characters": int(min_recognized_characters),
        },
        "recommended_template": (
            FAMILY_TEMPLATES.get(selected) if mode == "auto" and decision["decision"] == "classified" else None
        ),
        "fallback_template": "clean_article",
    }
    if include_indicators:
        result["rule_indicators"] = {
            rule_id: bool(indicators.get(rule_id)) for rule_id in RULE_IDS
        }
    return result