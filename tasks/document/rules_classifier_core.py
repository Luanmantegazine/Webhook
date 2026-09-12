"""Pure rule-based document classification for Hydra.

This module deliberately has no dependency on FabricFlow's ``@task`` decorator.
It can therefore be unit-tested and reused by task wrappers, benchmark scripts,
and offline calibration notebooks.

Design contract
---------------
Five responsibilities are kept strictly separate so that each can be inspected,
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
4. **Family gates** (:data:`FAMILY_GATES`, :func:`evaluate_family_gates`) state,
   as data, what *kind* of evidence may decide a family: which rules are primary
   and which merely corroborate, how many independent groups a decision needs,
   and which cross-family guards veto it. A family whose gate is not satisfied
   is removed from contention before ranking; the pre-gate score and the gate's
   own reasoning are both reported. Gates replace negative weights, which
   suppressed a family and silently rescaled every score around it at the same
   time, with no way for a reader to tell the two effects apart.
5. **Decision policy** (:func:`apply_decision_policy`) turns scores into a
   family, an abstention, or a fallback. It is a pure function of the scores and
   the thresholds, which is what allows a risk-coverage curve to be swept
   offline without re-running the rules. Families may declare their own
   operating point (:data:`FAMILY_CONFIDENCE_THRESHOLDS`), resolved as an offset
   from the global threshold so a sweep keeps describing the policy that runs.

Reproducibility
---------------
Every result carries the identity of what produced it: ``schema_version``,
``taxonomy_version``, ``feature_extraction_version``, ``classifier_version``,
``rule_fingerprint`` and ``feature_fingerprint``. The rule fingerprint covers
rules, weights, groups, channels, gates, blockers, family thresholds and the
decision-group count — everything that can change a decision — and is stable
across processes and interpreter versions.

The weights shipped in :data:`DEFAULT_WEIGHTS` are a documented prior, not a
calibrated model. Any experimental claim should either calibrate them on a
development split or compare them against weights fitted from labelled data;
both paths consume the indicator vector, not the rule definitions.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import math
import re
from types import CodeType
import unicodedata
from typing import Any, Callable, Iterable, NamedTuple

from tasks.document.rvl_cdip_eval import (
    DOCUMENT_FAMILIES,
    PRESS_RELEASE_POLICY,
    SCORED_FAMILIES,
    TAXONOMY_VERSION,
    taxonomy_descriptor,
)
from tasks.document.word_geometry import extract_geometry_features

#: External contract version of the feature record and of the classification
#: result. It changes only when a *consumer* would have to change: adding a
#: feature key or a reporting field does not move it, removing or redefining
#: one does. ``feature_extraction_version`` and the fingerprints below carry
#: the finer-grained identity.
SCHEMA_VERSION = "2.1"

#: Feature records produced under any other schema version cannot be pooled
#: with these. The evaluator refuses them rather than silently coercing.
SUPPORTED_FEATURE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# Bump whenever a derived feature's definition changes. The fingerprint then
# invalidates stale cached features even when the emitted key set is unchanged.
# 2.3: narrative_line_ratio, academic/editorial metadata counts, quote and
# press-release markers, marketing lexicon, technical-strength counters.
FEATURE_EXTRACTION_VERSION = "2.3"

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

#: ``DOCUMENT_FAMILIES``, ``SCORED_FAMILIES`` and ``TAXONOMY_VERSION`` are
#: re-exported from :mod:`tasks.document.rvl_cdip_eval`, which is the single
#: source of truth for the taxonomy. They are *not* redefined here: the label
#: mapping, the family list and the classifier used to declare the taxonomy
#: independently, and any one of them could change without the others.

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
    # [ \t] rather than \s prevents a list item and the next line becoming one
    # false heading.
    r"(?m)^[ \t]*(?:\d{1,2}(?:\.\d{1,2}){1,3}[.)]?|(?:\d{1,2}|[IVX]{2,6})[.)])[ \t]+"
    r"[A-Z][A-Za-z-]+(?:[ \t]+[\w,'()-]+){1,10}[ \t]*$"
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
# Straight and typographic quotation marks. A news body quotes sources; an
# attribution verb without a quotation is a narrative verb like any other, and
# on its own says nothing about the family.
_QUOTE_RE = re.compile(r"[\u201c\u201d\u00ab\u00bb]|(?<![\w\"])\"(?=[A-Za-z])")

# ``For immediate release`` is the defining surface of a press release. RVL-CDIP
# has no press-release class, so whether these documents may be accepted as
# news_article is a taxonomy decision, declared once in
# :mod:`tasks.document.rvl_cdip_eval` and consumed by exactly one gate.
_PRESS_RELEASE_RE = re.compile(
    r"\bfor immediate release\b|\bpress release\b|\bnews release\b|"
    r"\bfor release\b.{0,40}\b(?:a\.m\.|p\.m\.|immediately)\b",
    re.IGNORECASE,
)

# Author and affiliation block of an academic paper. Deliberately institutional:
# a personal name alone is not evidence of anything, an institutional address
# under a title is.
_AFFILIATION_RE = re.compile(
    r"\b(?:university|universidade|institute|instituto|laborator(?:y|ies|io)|"
    r"department of|dept\.? of|faculty of|school of|academy of|research (?:center|centre|group)|"
    r"college of)\b",
    re.IGNORECASE,
)
_CORRESPONDING_AUTHOR_RE = re.compile(
    r"\b(?:corresponding author|e-?mail address|\*\s*corresponding)\b", re.IGNORECASE
)

# Marketing surface, kept separate from the presentation lexicon: these are the
# words of an advertisement, and they are used as a *guard* for other families
# rather than as standalone evidence for this one.
_MARKETING_TERM_RE = re.compile(
    r"\b(?:special offer|limited time|free trial|call (?:now|today)|order now|"
    r"money[- ]back|discount|sale ends|buy (?:one|now)|satisfaction guaranteed|"
    r"new and improved|advertisement)\b",
    re.IGNORECASE,
)

# Specification vocabulary strong enough to be a guard. ``shall``/``must``
# requirement language plus clause numbering is what an engineering
# specification looks like regardless of whether it also has labelled fields.
_SPEC_STRENGTH_RE = re.compile(
    r"\b(?:shall (?:be|comply|conform|not|provide|have)|in accordance with|"
    r"per (?:mil|ansi|astm|iso|ieee)[- ]?\w*|tolerance|specification no\.?|"
    r"revision [a-z0-9]|drawing no\.?|part number)\b",
    re.IGNORECASE,
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


def _feature_fingerprint(features: dict[str, Any]) -> str:
    keys = sorted(str(key) for key in features.keys())
    digest = hashlib.sha1(
        (f"{FEATURE_EXTRACTION_VERSION}\n" + "\n".join(keys)).encode("utf-8")
    ).hexdigest()
    return f"ff-{digest[:12]}"


def feature_fingerprint(features: dict[str, Any]) -> str:
    """The fingerprint a feature record *should* carry.

    Public so that a consumer of a stored record can recompute it instead of
    trusting the value stored beside it. The extraction version is inside the
    digest, so a redefined feature invalidates the fingerprint even when the
    emitted key set is unchanged.
    """
    if not isinstance(features, dict):
        raise TypeError("features must be a dict")
    return _feature_fingerprint({key: value for key, value in features.items() if key != "feature_fingerprint"})


class FeatureContractError(ValueError):
    """A stored feature record cannot be scored by this classifier."""


def validate_feature_record(features: Any, *, source: str = "features") -> dict[str, Any]:
    """Refuse feature records this classifier cannot score.

    Stale cached features are the quietest failure in the whole pipeline: they
    score, they produce a number, and nothing in the output says the number
    describes a different feature definition than the one being reported. So
    every incompatibility is an error here, never a coercion.

    Returns the record's version descriptor, which the caller can use to
    detect a *mixture* of versions across a corpus — equally silent, equally
    fatal to comparability.
    """
    if not isinstance(features, dict):
        raise FeatureContractError(f"{source}: feature record must be a dict")

    schema_version = features.get("schema_version")
    if schema_version not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
        raise FeatureContractError(
            f"{source}: incompatible schema_version {schema_version!r}; "
            f"this classifier reads {sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS)}"
        )
    taxonomy_version = features.get("taxonomy_version")
    if taxonomy_version != TAXONOMY_VERSION:
        raise FeatureContractError(
            f"{source}: features carry taxonomy_version {taxonomy_version!r}, "
            f"classifier is {TAXONOMY_VERSION!r}"
        )
    extraction_version = features.get("feature_extraction_version")
    if extraction_version != FEATURE_EXTRACTION_VERSION:
        raise FeatureContractError(
            f"{source}: features were extracted by feature_extraction_version "
            f"{extraction_version!r}, classifier expects {FEATURE_EXTRACTION_VERSION!r}; "
            "re-extract from document, page_sizes, page_words and provenance"
        )
    stored = features.get("feature_fingerprint")
    expected = feature_fingerprint(features)
    if stored != expected:
        raise FeatureContractError(
            f"{source}: feature_fingerprint {stored!r} does not match the record "
            f"(recomputed {expected!r}); the feature set was edited after extraction"
        )
    return {
        "schema_version": schema_version,
        "taxonomy_version": taxonomy_version,
        "feature_extraction_version": extraction_version,
        "feature_fingerprint": stored,
    }


def validate_rule_ids(rule_ids: Any, *, source: str = "rule_ids") -> tuple[str, ...]:
    """Refuse rule identifiers this classifier does not declare.

    A stored artifact naming a rule that no longer exists was produced by a
    different rule set. Reading it as though the rule simply never fired turns
    a version mismatch into a plausible-looking zero.
    """
    if isinstance(rule_ids, dict):
        candidates = [str(key) for key in rule_ids]
    elif isinstance(rule_ids, (list, tuple, set, frozenset)):
        candidates = [str(item) for item in rule_ids]
    else:
        raise FeatureContractError(f"{source}: expected a collection of rule ids")
    unknown = sorted(set(candidates) - set(RULE_IDS))
    if unknown:
        raise FeatureContractError(
            f"{source}: rule id(s) absent from RULE_IDS: {', '.join(unknown)}"
        )
    return tuple(candidates)


def classifier_versions() -> dict[str, Any]:
    """Everything needed to reproduce a decision, in one block."""
    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "rule_count": len(RULE_IDS),
        "decision_group_count": DECISION_GROUP_COUNT,
        "family_confidence_thresholds": dict(FAMILY_CONFIDENCE_THRESHOLDS),
        "taxonomy": taxonomy_descriptor(),
    }


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
    page_words: list | None = None,
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
            # Recovered docTR text is a non-detection, not a zero-confidence
            # layout detection; keep it out of layout confidence statistics.
            if region.get("source") != "doctr_fallback":
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

    extracted = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
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
        "quote_mark_count": len(_QUOTE_RE.findall(text)),
        # Academic surface split into structure and vocabulary. The two were
        # previously only observable as one boolean each, which made it
        # impossible to state "a heading, not merely the word" as a rule.
        "academic_section_heading_count": len(_ACADEMIC_SECTION_HEADING_RE.findall(text)),
        "academic_vocabulary_count": len(_ACADEMIC_SECTION_RE.findall(text)),
        "affiliation_marker_count": len(_AFFILIATION_RE.findall(text)),
        "corresponding_author_count": len(_CORRESPONDING_AUTHOR_RE.findall(text)),
        "press_release_marker_count": len(_PRESS_RELEASE_RE.findall(text)),
        "marketing_term_count": len(_MARKETING_TERM_RE.findall(text)),
        "specification_strength_count": len(_SPEC_STRENGTH_RE.findall(text)),
    }
    extracted.update(
        extract_geometry_features(
            page_words,
            page_sizes=page_sizes,
            page_count=total_pages,
        )
    )
    extracted["feature_fingerprint"] = _feature_fingerprint(extracted)
    return extracted


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

    def opt(self, key: str) -> float | None:
        """A float that may legitimately be absent.

        OCR-quality means are ``None`` when the channel produced no
        measurement. Coercing that to ``0.0`` would read "no measurement" as
        "worst possible quality", which is how an unmeasured document ends up
        being treated as illegible.
        """
        value = self.features.get(key)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None


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
    requires: str  # "text" | "layout" | "multipage" | "geometry"
    predicate: Callable[[_Context], bool]

    @property
    def rule_id(self) -> str:
        return f"{self.family}.{self.name}"


def _rule(family: str, name: str, predicate, *, group: str = "", requires: str = "text") -> RuleSpec:
    return RuleSpec(family, name, group or name, requires, predicate)


# Patterns used only inside predicates.
_ABSTRACT_RE = re.compile(
    r"(?:(?:^|\n)\s*abstract\b|\babstract\s*[:—-])"
    # The first branch stops at ``abstract`` while the second consumes a
    # delimiter. Include delimiters here so the guard applies to both forms.
    r"(?![ \t:—-]*(?:form|sheet|submission|blank)\b)",
    re.IGNORECASE,
)
_REFERENCES_RE = re.compile(r"(?m)^\s*(?:references|bibliography)\s*$", re.IGNORECASE)
_ACADEMIC_SECTION_RE = re.compile(
    r"\b(?:methodology|methods|experimental results|related work|conclusions?)\b",
    re.IGNORECASE,
)
_ACADEMIC_SECTION_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(?:methodology|methods|experimental results|related work|conclusions?)"
    r"\s*[:.]?\s*$",
    re.IGNORECASE,
)
_EDITORIAL_DATES_RE = re.compile(
    r"\breceived\b.{0,100}\bacce(?:pt|pl)ed\b|"
    r"\bacce(?:pt|pl)ed\b.{0,100}\breceived\b",
    re.IGNORECASE | re.DOTALL,
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
# ``questionnaire`` is specific enough anywhere on the page; ``survey`` is not —
# it is an ordinary word in research papers and reports — so it only counts as a
# heading: a short line that ends with it.
_QUESTIONNAIRE_RE = re.compile(
    r"\bquestionnaire\b|^[ \t]*[\w ,'()-]{0,40}\bsurvey\b[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_PRESENTATION_TERM_RE = re.compile(
    r"\b(?:agenda|presentation|our products?|limited time|special offer)\b",
    re.IGNORECASE,
)


def _tables_dominate(ctx: _Context) -> bool:
    """Guard for families whose layout evidence is meaningless on table pages."""
    return ctx.flt("table_area_ratio") >= 0.35


def _visual_page(ctx: _Context) -> bool:
    return ctx.num("relevant_picture_count") >= 2 or ctx.flt("picture_area_ratio") >= 0.35


def _typeset_body(ctx: _Context) -> bool:
    return (ctx.flt('right_edge_regularity') >= 0.75 and ctx.flt('line_pitch_regularity') >= 0.70 and ctx.flt('wide_gap_line_ratio') <= 0.15 and ctx.num('word_line_count') >= 12)


def _has_primary_correspondence_signal(ctx: _Context) -> bool:
    return (ctx.num('correspondence_header_count') >= 2 or ctx.has(_SALUTATION_RE) or ctx.has(_CLOSING_RE) or ctx.has(_MEMO_RE) or ctx.has(_EMAIL_MARKER_RE))


RULES: tuple[RuleSpec, ...] = (
    # ---- research paper -------------------------------------------------
    _rule("research_paper", "abstract_heading", lambda c: c.has(_ABSTRACT_RE)),
    _rule("research_paper", "references_heading", lambda c: c.has(_REFERENCES_RE)),
    _rule("research_paper", "doi", lambda c: c.num("doi_match_count") > 0, group="citation_evidence"),
    _rule("research_paper", "citations", lambda c: c.num("citation_match_count") >= 2, group="citation_evidence"),
    _rule(
        "research_paper",
        "academic_section_headings",
        lambda c: c.num("academic_section_heading_count") >= 1,
        group="academic_sections",
    ),
    # Corroborating only, and no longer satisfiable by one generic word:
    # "methods", "results", "study" and "report" occur in every kind of
    # document, so a single occurrence is not evidence of anything. It is
    # excluded from the family gate, so it can never open a decision on its own.
    _rule(
        "research_paper",
        "academic_vocabulary",
        lambda c: c.num("academic_vocabulary_count") >= 2,
        group="academic_sections",
    ),
    _rule(
        "research_paper",
        "editorial_dates",
        lambda c: c.has(_EDITORIAL_DATES_RE),
        group="editorial_metadata",
    ),
    _rule(
        "research_paper",
        "authors_affiliations",
        lambda c: c.num("affiliation_marker_count") >= 2
        or (
            c.num("affiliation_marker_count") >= 1
            and c.num("corresponding_author_count") >= 1
        ),
        group="editorial_metadata",
    ),
    # The three typographic signals of an academic page are substitutable
    # expressions of one observation, so they share a group and cannot be
    # counted three times for a document that happens to be well typeset.
    _rule(
        "research_paper",
        "formula_layout",
        lambda c: c.flt("formula_density") >= 0.03,
        group="academic_layout",
    ),
    _rule(
        "research_paper",
        "two_column_layout",
        lambda c: c.flt("two_column_ratio") >= 0.5,
        group="academic_layout",
        requires="layout",
    ),
    _rule(
        "research_paper",
        "justified_body",
        _typeset_body,
        group="academic_layout",
        requires="geometry",
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
    # ``blank_fields`` is primary evidence and ``field_labels`` is corroborating,
    # so they are no longer substitutable members of one group: a printed form
    # with both should score higher than one with labels alone, and the family
    # gate must be able to tell the two apart.
    _rule("form_structured", "blank_fields", lambda c: c.num("blank_field_count") >= 2),
    _rule("form_structured", "field_labels", lambda c: c.num("field_label_count") >= 3),
    _rule(
        "form_structured",
        "short_field_regions",
        lambda c: c.num("text_region_count") >= 5
        and c.flt("average_words_per_text_region") <= 8.0,
    ),
    # The three geometric expressions of a field layout share one group: they
    # are the same observation seen through label-value spacing, tab stops and
    # line regularity. None of them can open a decision — see FAMILY_GATES.
    # The former broad ``field_grid`` rule (any repeated alignment columns) is
    # deliberately *not* reintroduced: prose, tables and columned reports all
    # satisfy it.
    _rule(
        "form_structured",
        "label_value_lines",
        lambda c: c.flt("label_value_line_ratio") >= 0.30,
        group="field_geometry",
        requires="geometry",
    ),
    _rule(
        "form_structured",
        "tab_stop_alignment",
        lambda c: c.num("tab_stop_count") >= 2
        and (c.num("blank_field_count") >= 1 or c.num("field_label_count") >= 2),
        group="field_geometry",
        requires="geometry",
    ),
    _rule(
        "form_structured",
        "field_geometry_regularity",
        lambda c: c.flt("body_wide_gap_line_ratio") >= 0.35
        and c.flt("line_pitch_regularity") >= 0.55
        and c.flt("narrative_line_ratio") <= 0.35,
        group="field_geometry",
        requires="geometry",
    ),
    # ---- presentation / advertisement -----------------------------------
    # Presentation classification requires positive visual, list, or lexical
    # evidence. Landscape orientation and picture dominance are alternative
    # expressions of the same "visual page" evidence, so they share a group:
    # a corpus of portrait-only scans no longer deflates this family's
    # attainable score.
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
    # "A short title over lists" — the slide shape. A title region is now
    # required: list density with no title is a table of contents, an index, or
    # an itemised form, none of which are presentations.
    _rule(
        "presentation_marketing",
        "bullet_layout",
        lambda c: not _tables_dominate(c)
        and c.flt("list_density") >= 0.15
        and c.num("word_count") <= 180
        and c.flt("title_density") >= 0.02,
    ),
    _rule(
        "presentation_marketing",
        "slide_structure",
        lambda c: not _tables_dominate(c)
        and c.flt("landscape_ratio") >= 0.5
        and c.flt("title_density") >= 0.05
        and (
            c.flt("list_density") >= 0.08
            or c.flt("average_words_per_text_region") <= 20.0
        ),
        requires="layout",
    ),
    # Positive visual evidence stated directly: a large share of the page is
    # picture *and* the text that is there is not narrative. Sparse text alone
    # is not part of this rule — that was the path by which a short or
    # low-quality OCR of any document became a "presentation".
    _rule(
        "presentation_marketing",
        "visual_dominance",
        lambda c: not _tables_dominate(c)
        and c.flt("picture_area_ratio") >= 0.25
        and c.flt("narrative_line_ratio") <= 0.25,
        group="visual_density",
        requires="geometry",
    ),
    # Corroborating vocabulary. Excluded from the family gate: an "agenda" line
    # or a "special offer" is not on its own a reason to call a scan a slide.
    _rule(
        "presentation_marketing",
        "presentation_terms",
        lambda c: not _tables_dominate(c) and c.has(_PRESENTATION_TERM_RE),
    ),
    # Restricted: centred sparse text describes a title page, a handwritten
    # note, a certificate and a failed OCR just as well as a slide, so it now
    # requires the landscape or pictorial evidence that distinguishes them.
    _rule(
        "presentation_marketing",
        "sparse_centered",
        lambda c: c.flt("centered_line_ratio") >= 0.30
        and c.num("word_line_count") <= 25
        and (c.flt("landscape_ratio") >= 0.5 or c.num("relevant_picture_count") >= 1),
        group="visual_density",
        requires="geometry",
    ),
    # ---- correspondence (letter / memo / e-mail) -------------------------
    _rule("correspondence", "header_block", lambda c: c.num("correspondence_header_count") >= 2),
    _rule("correspondence", "salutation", lambda c: c.has(_SALUTATION_RE)),
    _rule("correspondence", "closing", lambda c: c.has(_CLOSING_RE)),
    _rule("correspondence", "memo_heading", lambda c: c.has(_MEMO_RE)),
    _rule("correspondence", "email_markers", lambda c: c.has(_EMAIL_MARKER_RE)),
    _rule(
        "correspondence",
        "letter_geometry",
        lambda c: c.flt("top_band_header_ratio") >= 0.40
        and c.flt("line_pitch_regularity") >= 0.60
        and c.flt("body_wide_gap_line_ratio") <= 0.20,
        requires="geometry",
    ),
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
    # Attribution verbs without quotations are ordinary narrative verbs; a
    # reported speech block is an attribution verb *and* a quotation.
    _rule(
        "news_article",
        "attribution_quotes",
        lambda c: c.num("attribution_count") >= 3 and c.num("quote_mark_count") >= 2,
    ),
    _rule(
        "news_article",
        "multi_column_body",
        lambda c: c.flt("two_column_ratio") >= 0.5 and c.num("word_count") >= 200,
        group="news_layout",
        requires="layout",
    ),
    _rule(
        "news_article",
        "justified_body",
        _typeset_body,
        group="news_layout",
        requires="geometry",
    ),
    # Headline over a long running body: a titled region on a page whose body
    # is narrative and neither tabular nor pictorial.
    _rule(
        "news_article",
        "headline_body",
        lambda c: c.flt("title_density") >= 0.02
        and c.num("word_count") >= 250
        and c.flt("table_area_ratio") < 0.25
        and c.flt("picture_area_ratio") < 0.35,
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
    "research_paper.academic_section_headings": 0.16,
    "research_paper.academic_vocabulary": 0.05,
    "research_paper.editorial_dates": 0.22,
    "research_paper.authors_affiliations": 0.14,
    "research_paper.formula_layout": 0.08,
    "research_paper.two_column_layout": 0.08,
    "research_paper.justified_body": 0.14,
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
    # Primary evidence outweighs corroborating evidence by construction, so the
    # score agrees with the gate instead of contradicting it: the previous
    # weights made ``field_labels`` and ``label_value_lines`` — both
    # corroborating — as heavy as a form heading.
    "form_structured.form_heading": 0.30,
    "form_structured.questionnaire_heading": 0.26,
    "form_structured.checkboxes": 0.24,
    "form_structured.blank_fields": 0.22,
    "form_structured.field_labels": 0.12,
    "form_structured.short_field_regions": 0.08,
    "form_structured.label_value_lines": 0.14,
    "form_structured.tab_stop_alignment": 0.10,
    "form_structured.field_geometry_regularity": 0.08,
    "presentation_marketing.visual_layout": 0.22,
    "presentation_marketing.landscape_layout": 0.16,
    "presentation_marketing.slide_structure": 0.18,
    "presentation_marketing.visual_dominance": 0.16,
    "presentation_marketing.bullet_layout": 0.12,
    "presentation_marketing.presentation_terms": 0.20,
    "presentation_marketing.sparse_centered": 0.12,
    "correspondence.header_block": 0.30,
    "correspondence.salutation": 0.24,
    "correspondence.closing": 0.18,
    "correspondence.memo_heading": 0.26,
    "correspondence.email_markers": 0.22,
    "correspondence.letter_geometry": 0.22,
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
    "news_article.justified_body": 0.16,
    "news_article.headline_body": 0.12,
}



#: How many of a family's strongest evidence groups constitute a sufficient
#: case. The normalising denominator is the mass of that many groups, not the
#: family's total mass: dividing by the total would make a family with many
#: weak corroborating rules structurally unable to reach a decision on its two
#: or three decisive ones. Raising this makes every family harder to accept.
DECISION_GROUP_COUNT = 3


# --------------------------------------------------------------------------
# Family gates: declarative acceptance requirements and cross-family guards
# --------------------------------------------------------------------------
#
# A gate answers a question a weighted sum cannot: *is this the kind of
# evidence that may decide this family at all?* Negative weights were the
# previous answer, and they are close to uninterpretable — a large negative
# weight both suppresses the family and silently re-scales every score around
# it, and no reader of the output can tell which of the two happened.
#
# A gate is data, not code: it names the rules that form each independent
# evidence group, how many groups a decision needs, which rules are merely
# corroborating, and which named guards veto the family outright. Adding a
# family requirement means adding a row here, never a conditional inside the
# scorer or the decision policy.


class Blocker(NamedTuple):
    """A named, reusable "this is another family" guard."""

    name: str
    predicate: Callable[[_Context], bool]
    description: str


def _invoice_evidence(ctx: _Context) -> bool:
    signals = (
        ctx.has(_INVOICE_RE),
        ctx.has(_AMOUNT_DUE_RE),
        ctx.has(_SUBTOTAL_RE) and ctx.has(_TOTAL_RE),
        ctx.has(_BILLING_PARTY_RE),
        ctx.num("currency_match_count") >= 3,
    )
    return sum(bool(signal) for signal in signals) >= 2


def _technical_specification_evidence(ctx: _Context) -> bool:
    return (
        ctx.has(_TECH_REPORT_RE)
        or ctx.num("specification_strength_count") >= 2
        or (ctx.has(_REQUIREMENTS_RE) and ctx.num("numbered_heading_count") >= 2)
    )


def _news_reporting_evidence(ctx: _Context) -> bool:
    markers = (ctx.has(_BYLINE_RE), ctx.has(_DATELINE_RE), ctx.has(_WIRE_SERVICE_RE))
    if sum(bool(marker) for marker in markers) >= 2:
        return True
    return (
        any(markers)
        and ctx.num("attribution_count") >= 3
        and ctx.num("quote_mark_count") >= 2
    )


def _resume_evidence(ctx: _Context) -> bool:
    return ctx.has(_RESUME_HEADING_RE) or (
        ctx.has(_EXPERIENCE_SECTION_RE) and ctx.has(_EDUCATION_SECTION_RE)
    )


def _budget_evidence(ctx: _Context) -> bool:
    """A budget is a labelled grid of money, which is form-shaped but is not a form."""
    return (ctx.num("accounting_term_count") >= 4 and ctx.num("numeric_token_count") >= 12) or (
        ctx.has(_REPORTING_PERIOD_RE)
        and ctx.num("accounting_term_count") >= 3
        and ctx.num("currency_match_count") >= 2
    )


def _advertisement_evidence(ctx: _Context) -> bool:
    return ctx.num("marketing_term_count") >= 2 or (
        ctx.num("marketing_term_count") >= 1 and _visual_page(ctx)
    )


def _research_publication_evidence(ctx: _Context) -> bool:
    groups = (
        ctx.has(_ABSTRACT_RE) or ctx.has(_REFERENCES_RE),
        ctx.num("doi_match_count") > 0 or ctx.num("citation_match_count") >= 2,
        ctx.has(_EDITORIAL_DATES_RE) or ctx.num("affiliation_marker_count") >= 2,
    )
    return sum(bool(group) for group in groups) >= 2


def _correspondence_evidence(ctx: _Context) -> bool:
    return (
        ctx.has(_MEMO_RE)
        or ctx.num("correspondence_header_count") >= 2
        or (ctx.has(_SALUTATION_RE) and ctx.has(_CLOSING_RE))
    )


def _form_evidence(ctx: _Context) -> bool:
    return (ctx.has(_FORM_HEADING_RE) or ctx.has(_QUESTIONNAIRE_RE)) and (
        ctx.num("checkbox_count") >= 2 or ctx.num("blank_field_count") >= 2
    )


def _press_release_evidence(ctx: _Context) -> bool:
    """Press releases are news-shaped but are not journalism.

    Whether they may be accepted as ``news_article`` is a taxonomy decision,
    declared once as ``press_release_policy`` in
    :mod:`tasks.document.rvl_cdip_eval`. Under the default, this guard is
    active; under ``news_article`` it is inert.
    """
    if PRESS_RELEASE_POLICY != "not_news_article":
        return False
    return ctx.num("press_release_marker_count") >= 1


def _sparse_or_low_quality_ocr(ctx: _Context) -> bool:
    """Little text, or text the OCR is not confident about.

    This is emphatically *not* evidence for any family. A handwritten page, a
    dark scan and a failed binarisation all produce it, and the previous rule
    set let it act as positive evidence for ``presentation_marketing`` because
    slides also happen to be short. Used only as a guard.
    """
    sparse = ctx.num("word_count") <= 60 or ctx.num("alnum_character_count") <= 200
    word_confidence = ctx.opt("word_confidence_mean")
    layout_confidence = ctx.opt("layout_confidence_mean")
    low_quality = (word_confidence is not None and word_confidence < 0.55) or (
        layout_confidence is not None and layout_confidence < 0.35
    )
    if not (sparse or low_quality):
        return False
    return not _visual_page(ctx) and ctx.flt("landscape_ratio") < 0.5


BLOCKERS: tuple[Blocker, ...] = (
    Blocker("invoice_evidence", _invoice_evidence, "Billing document: at least two invoice signals."),
    Blocker(
        "technical_specification_evidence",
        _technical_specification_evidence,
        "Engineering specification: requirement language, clause numbering or spec identifiers.",
    ),
    Blocker(
        "news_reporting_evidence",
        _news_reporting_evidence,
        "Journalistic reporting: two of byline/dateline/wire, or one plus reported speech.",
    ),
    Blocker("resume_evidence", _resume_evidence, "CV heading, or experience and education sections."),
    Blocker("budget_evidence", _budget_evidence, "Accounting vocabulary over a dense numeric grid."),
    Blocker("advertisement_evidence", _advertisement_evidence, "Marketing copy, or marketing copy on a visual page."),
    Blocker(
        "research_publication_evidence",
        _research_publication_evidence,
        "Two independent academic evidence groups.",
    ),
    Blocker(
        "correspondence_evidence",
        _correspondence_evidence,
        "Memorandum heading, routing header block, or salutation with closing.",
    ),
    Blocker("form_evidence", _form_evidence, "Form or questionnaire heading over blanks or checkboxes."),
    Blocker(
        "press_release_evidence",
        _press_release_evidence,
        "Press-release markers; active only under press_release_policy=not_news_article.",
    ),
    Blocker(
        "sparse_or_low_quality_ocr",
        _sparse_or_low_quality_ocr,
        "Short or low-confidence OCR with no visual evidence; never positive evidence.",
    ),
)

BLOCKER_PREDICATES: dict[str, Blocker] = {blocker.name: blocker for blocker in BLOCKERS}


class FamilyGate(NamedTuple):
    """Declarative acceptance requirement for one family.

    ``groups``
        Independent evidence groups, ``(group_name, rule_names)``. Two rules in
        the same group are two ways of seeing one thing and count once.
    ``min_groups``
        How many distinct groups must fire for the family to be decidable.
    ``sufficient_rules``
        Rules strong enough to satisfy the gate alone.
    ``required_any_groups``
        When non-empty, at least one of these groups must be among those that
        fired, whatever ``min_groups`` says.
    ``corroborating``
        Rules that add score but can never open a decision.
    ``blockers``
        Named guards; any one of them firing vetoes the family outright.
    ``confidence_threshold``
        Optional family-specific operating point. Declared here, resolved by
        :func:`resolve_family_thresholds`, and reported in the result. It is an
        offset from the global threshold, never a replacement for it.
    """

    family: str
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    min_groups: int
    sufficient_rules: frozenset[str]
    required_any_groups: frozenset[str]
    corroborating: tuple[str, ...]
    blockers: tuple[str, ...]
    confidence_threshold: float | None
    rationale: str


FAMILY_GATES: tuple[FamilyGate, ...] = (
    FamilyGate(
        family="form_structured",
        groups=(
            ("form_heading", ("form_heading",)),
            ("questionnaire_heading", ("questionnaire_heading",)),
            ("checkboxes", ("checkboxes",)),
            ("blank_fields", ("blank_fields",)),
        ),
        min_groups=2,
        sufficient_rules=frozenset({"form_heading", "questionnaire_heading"}),
        required_any_groups=frozenset(),
        corroborating=(
            "field_labels",
            "short_field_regions",
            "label_value_lines",
            "tab_stop_alignment",
            "field_geometry_regularity",
        ),
        blockers=(
            "invoice_evidence",
            "technical_specification_evidence",
            "news_reporting_evidence",
            "advertisement_evidence",
            "budget_evidence",
            "resume_evidence",
        ),
        confidence_threshold=None,
        rationale=(
            "One strong primary (a form or questionnaire heading) or two independent "
            "primaries (checkboxes, blank fields). Labelled lines, short regions, tab "
            "stops and geometric regularity corroborate only: every one of them is also "
            "produced by invoices, specifications, budgets, resumes, advertisements and "
            "columned news pages, which is where the family's false positives came from."
        ),
    ),
    FamilyGate(
        family="correspondence",
        # Each signal is its own group: a salutation at the top of a page and a
        # closing at the bottom are two independent observations, not two
        # spellings of one. Grouping them cost exactly the coverage this family
        # is meant to gain — an ordinary "Dear ... Sincerely" letter carrying no
        # routing header fired one group and was refused.
        groups=(
            ("header_block", ("header_block",)),
            ("email_markers", ("email_markers",)),
            ("salutation", ("salutation",)),
            ("closing", ("closing",)),
            ("memo_heading", ("memo_heading",)),
            ("letter_geometry", ("letter_geometry",)),
            ("letter_body", ("letter_body",)),
        ),
        min_groups=2,
        sufficient_rules=frozenset(),
        required_any_groups=frozenset(),
        corroborating=(),
        blockers=("form_evidence", "news_reporting_evidence"),
        confidence_threshold=0.50,
        rationale=(
            "Two independent signals accept, even when each is individually weak — a "
            "salutation with a closing, or an e-mail marker with letter geometry, is a "
            "letter. Coverage is bought with the family threshold declared here, not by "
            "lowering the global threshold for every family. ``letter_body`` cannot "
            "reach the bar on its own: it only fires when a primary signal is already "
            "present, and a letter carrying nothing but a routing header and the shape "
            "of a letter still scores below the family threshold."
        ),
    ),
    FamilyGate(
        family="research_paper",
        groups=(
            (
                "academic_structure",
                ("abstract_heading", "references_heading", "academic_section_headings"),
            ),
            ("citation_evidence", ("doi", "citations")),
            ("editorial_metadata", ("editorial_dates", "authors_affiliations")),
            ("academic_layout", ("two_column_layout", "justified_body", "formula_layout")),
        ),
        min_groups=2,
        sufficient_rules=frozenset(),
        required_any_groups=frozenset(),
        corroborating=("academic_vocabulary",),
        blockers=("news_reporting_evidence", "invoice_evidence", "form_evidence"),
        confidence_threshold=None,
        rationale=(
            "Two independent groups: structure with citations, structure with editorial "
            "metadata, or citations with academic layout. Generic vocabulary is "
            "corroborating only, so 'results', 'method', 'study' or a bare date can never "
            "decide the family."
        ),
    ),
    FamilyGate(
        family="news_article",
        groups=(
            ("journalistic_source", ("byline", "wire_service")),
            ("dateline", ("dateline",)),
            ("reported_speech", ("attribution_quotes",)),
            ("news_layout", ("multi_column_body", "justified_body")),
            ("headline_body", ("headline_body",)),
        ),
        min_groups=2,
        sufficient_rules=frozenset(),
        required_any_groups=frozenset({"journalistic_source", "dateline", "news_layout"}),
        corroborating=(),
        blockers=(
            "research_publication_evidence",
            "advertisement_evidence",
            "form_evidence",
            "correspondence_evidence",
            "press_release_evidence",
        ),
        confidence_threshold=None,
        rationale=(
            "A byline, an attribution or a column count alone decides nothing: two groups "
            "are required, one of which must be a journalistic source, a dateline or the "
            "news layout. Guards keep scientific publications, advertisements, forms, "
            "institutional correspondence and press releases out of the family."
        ),
    ),
    FamilyGate(
        family="presentation_marketing",
        groups=(
            ("visual_page", ("visual_layout", "landscape_layout")),
            ("slide_structure", ("slide_structure",)),
            ("title_with_lists", ("bullet_layout",)),
            ("visual_density", ("visual_dominance", "sparse_centered")),
        ),
        min_groups=1,
        sufficient_rules=frozenset(),
        required_any_groups=frozenset(
            {"visual_page", "slide_structure", "title_with_lists", "visual_density"}
        ),
        corroborating=("presentation_terms",),
        blockers=("sparse_or_low_quality_ocr", "form_evidence", "invoice_evidence"),
        confidence_threshold=None,
        rationale=(
            "Positive visual evidence is required: relevant pictures, a landscape slide "
            "structure, a short title over lists, or high visual area against low "
            "narrative density. Short text, sparse text and low-quality OCR are guards, "
            "never evidence, so a handwritten page is not a presentation for having "
            "little text on it."
        ),
    ),
)

GATED_FAMILIES: frozenset[str] = frozenset(gate.family for gate in FAMILY_GATES)

#: Family-specific operating points, declared once and reported in every
#: result. They are *offsets* from the global threshold (see
#: :func:`resolve_family_thresholds`), so sweeping the global threshold still
#: moves every family and the risk-coverage curve stays meaningful.
FAMILY_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    gate.family: float(gate.confidence_threshold)
    for gate in FAMILY_GATES
    if gate.confidence_threshold is not None
}


def _validate_gate_configuration() -> None:
    """Refuse to import a gate table that does not describe the rule set.

    A gate naming a rule that no longer exists silently stops constraining the
    family it was written for, which is the failure mode this check exists to
    make impossible.
    """
    rules_by_family: dict[str, set[str]] = {}
    for rule in RULES:
        rules_by_family.setdefault(rule.family, set()).add(rule.name)
    for gate in FAMILY_GATES:
        known = rules_by_family.get(gate.family)
        if not known:
            raise ValueError(f"gate declared for unknown family {gate.family!r}")
        grouped = [name for _group, names in gate.groups for name in names]
        if len(grouped) != len(set(grouped)):
            raise ValueError(f"{gate.family}: a rule appears in more than one gate group")
        covered = set(grouped) | set(gate.corroborating)
        unknown = sorted(covered - known)
        if unknown:
            raise ValueError(f"{gate.family}: gate references unknown rule(s) {unknown}")
        uncovered = sorted(known - covered)
        if uncovered:
            raise ValueError(
                f"{gate.family}: rule(s) {uncovered} are neither gate evidence nor "
                "corroborating; every rule of a gated family must be classified"
            )
        overlap = sorted(set(grouped) & set(gate.corroborating))
        if overlap:
            raise ValueError(f"{gate.family}: rule(s) {overlap} are both primary and corroborating")
        unknown_sufficient = sorted(gate.sufficient_rules - set(grouped))
        if unknown_sufficient:
            raise ValueError(
                f"{gate.family}: sufficient rule(s) {unknown_sufficient} are not gate evidence"
            )
        group_names = {name for name, _rules in gate.groups}
        unknown_required = sorted(gate.required_any_groups - group_names)
        if unknown_required:
            raise ValueError(f"{gate.family}: required group(s) {unknown_required} do not exist")
        if gate.min_groups < 1 or gate.min_groups > len(gate.groups):
            raise ValueError(f"{gate.family}: min_groups is outside the declared groups")
        unknown_blockers = sorted(set(gate.blockers) - set(BLOCKER_PREDICATES))
        if unknown_blockers:
            raise ValueError(f"{gate.family}: unknown blocker(s) {unknown_blockers}")


_validate_gate_configuration()


def evaluate_family_gates(
    features: dict,
    indicators: dict[str, bool],
) -> dict[str, dict]:
    """Evaluate every declared gate into an auditable per-family record.

    Pure and separate from scoring: the returned record says *why* a family may
    or may not be decided, and :func:`classify_with_rules` is the only place
    that acts on it.
    """
    ctx = _Context(features)
    channels = available_channels(features)
    report: dict[str, dict] = {}
    for gate in FAMILY_GATES:
        fired_groups: list[str] = []
        fired_rules: list[str] = []
        for group_name, rule_names in gate.groups:
            hits = [name for name in rule_names if indicators.get(f"{gate.family}.{name}")]
            if hits:
                fired_groups.append(group_name)
                fired_rules.extend(hits)
        sufficient = sorted(
            name for name in gate.sufficient_rules if indicators.get(f"{gate.family}.{name}")
        )
        corroborating = sorted(
            name for name in gate.corroborating if indicators.get(f"{gate.family}.{name}")
        )
        blocked_by = []
        for name in gate.blockers:
            blocker = BLOCKER_PREDICATES[name]
            try:
                if blocker.predicate(ctx):
                    blocked_by.append(name)
            except Exception:  # a malformed feature must not abort classification
                continue

        meets_required = (
            not gate.required_any_groups
            or bool(gate.required_any_groups & set(fired_groups))
        )
        satisfied = bool(sufficient) or (
            len(fired_groups) >= gate.min_groups and meets_required
        )
        if blocked_by:
            status = "blocked"
        elif satisfied:
            status = "satisfied"
        else:
            status = "insufficient_evidence"
        report[gate.family] = {
            "status": status,
            "required_group_count": gate.min_groups,
            "fired_groups": fired_groups,
            "fired_primary_rules": sorted(fired_rules),
            "sufficient_rules_fired": sufficient,
            "corroborating_fired": corroborating,
            "required_any_groups": sorted(gate.required_any_groups),
            "blocked_by": blocked_by,
            "available_channels": sorted(channels),
            "family_confidence_threshold": gate.confidence_threshold,
            "rationale": gate.rationale,
        }
    return report


def resolve_family_thresholds(
    confidence_threshold: float,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Turn declared family thresholds into thresholds at this operating point.

    A declared family threshold is an operating point stated *relative to the
    module default*: it is applied as the same offset wherever the global
    threshold is moved. That is what keeps a swept risk-coverage curve honest —
    an absolute family threshold would sit still while the curve moved around
    it, and the curve would then describe a policy nobody runs.
    """
    declared = FAMILY_CONFIDENCE_THRESHOLDS if overrides is None else overrides
    global_threshold = float(confidence_threshold)
    resolved: dict[str, float] = {}
    for family, value in declared.items():
        offset = float(value) - DEFAULT_CONFIDENCE_THRESHOLD
        resolved[family] = round(max(0.0, global_threshold + offset), 6)
    return resolved


def _code_signature(code: CodeType) -> str:
    """Deterministic rendering of a code object.

    ``repr`` of a nested code object embeds its memory address, so hashing
    ``co_consts`` directly produced a *different fingerprint on every process*
    — the one thing a fingerprint must never do. Any predicate containing a
    generator expression or a comprehension has such a nested object, which is
    most of them. Nested code is therefore rendered recursively.
    """
    parts = [str(code.co_argcount), code.co_code.hex()]
    for const in code.co_consts:
        if isinstance(const, CodeType):
            parts.append(f"<code:{_code_signature(const)}>")
        else:
            parts.append(repr(const))
    parts.append(",".join(code.co_names))
    parts.append(",".join(code.co_varnames))
    return "|".join(parts)


def _definition_signature(value: Any) -> str | None:
    """Identity of one definition a rule depends on.

    Source is preferred over bytecode: it is stable across interpreter
    versions, so a fingerprint does not change merely because the benchmark ran
    on a different Python. Bytecode is the fallback for definitions whose
    source is unavailable.
    """
    if isinstance(value, re.Pattern):
        return f"re:{value.pattern!r}/{value.flags}"
    if callable(value) and hasattr(value, "__code__"):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError, IndexError):
            return f"code:{_code_signature(value.__code__)}"
        return f"src:{' '.join(source.split())}"
    return None


def _rule_fingerprint() -> str:
    parts = []
    for rule in RULES:
        weight = float(DEFAULT_WEIGHTS.get(rule.rule_id, 0.0))
        predicate = rule.predicate
        referenced_definitions = []
        for name in sorted(predicate.__code__.co_names):
            signature = _definition_signature(predicate.__globals__.get(name))
            if signature is not None:
                referenced_definitions.append(f"{name}={signature}")
        parts.append(
            f"{rule.rule_id}|{rule.group}|{rule.requires}|{weight:.6f}|"
            f"{_definition_signature(predicate)}|{'|'.join(referenced_definitions)}"
        )
    # Gates, family thresholds and the decision-group count are policy, and
    # policy changes the classifier as surely as a weight does. They are part
    # of the fingerprint so that two runs carrying the same fingerprint really
    # did apply the same rules *and* the same acceptance requirements.
    for gate in FAMILY_GATES:
        parts.append(
            f"gate:{gate.family}|{gate.groups}|{gate.min_groups}|"
            f"{sorted(gate.sufficient_rules)}|{sorted(gate.required_any_groups)}|"
            f"{gate.corroborating}|{gate.blockers}|{gate.confidence_threshold}"
        )
    for blocker in BLOCKERS:
        parts.append(f"blocker:{blocker.name}|{_definition_signature(blocker.predicate)}")
    parts.append(f"decision_group_count:{DECISION_GROUP_COUNT}")
    parts.append(f"family_thresholds:{sorted(FAMILY_CONFIDENCE_THRESHOLDS.items())}")
    parts.append(f"press_release_policy:{PRESS_RELEASE_POLICY}")
    parts.append(f"scored_families:{list(SCORED_FAMILIES)}")
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


#: Identity of the rule set *and* of the policy applied to it. Any change to a
#: rule, a weight, a gate, a blocker or a family threshold moves it.
RULE_FINGERPRINT = _rule_fingerprint()

#: v5: primary/corroborating separation, declarative family gates, per-family
#: operating points.
CLASSIFIER_VERSION = f"rules-rvl-cdip-v5+{RULE_FINGERPRINT}"


def available_channels(features: dict) -> frozenset[str]:
    """Evidence channels that this document can actually supply."""
    channels = {"text"}
    try:
        if float(features.get("measured_page_ratio") or 0.0) > 0.0:
            channels.add("layout")
        if int(features.get("total_pages") or 0) > 0:
            channels.add("multipage")
        if float(features["geometry_page_ratio"] or 0) > 0:
            channels.add("geometry")
    except (KeyError, TypeError, ValueError):
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
        if rule.family not in SCORED_FAMILIES or rule.requires not in channels:
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
    family_thresholds: dict[str, float] | None = None,
) -> dict:
    """Turn a ranked score list into a decision.

    ``family_thresholds`` is the resolved output of
    :func:`resolve_family_thresholds`: a per-family operating point that
    *replaces* the global threshold for the families that declare one, and is
    reported alongside the decision as ``applied_confidence_threshold`` so no
    reader has to infer which threshold was applied.

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
    resolved_thresholds = family_thresholds or {}
    applied_threshold = float(
        resolved_thresholds.get(top_family, confidence_threshold)
    )

    def _decision(payload: dict) -> dict:
        """Every outcome reports the threshold it was actually judged against."""
        return {"applied_confidence_threshold": round(applied_threshold, 6), **payload}

    if mode == "observe":
        # Full-coverage mode: no abstention, so the confusion matrix is complete
        # and the abstention policy can be evaluated separately from the rules.
        if top_score <= 0.0:
            return _decision({
                "document_family": "other",
                "decision": "observed",
                "reason": "no_rules_matched",
                "confidence": 0.0,
                "score": top_score,
                "score_margin": margin,
            })
        return _decision({
            "document_family": top_family,
            "decision": "observed",
            "reason": "argmax_without_abstention",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        })

    if recognized_characters < int(min_recognized_characters):
        return _decision({
            "document_family": "other",
            "decision": "abstained",
            "reason": "insufficient_ocr_text",
            "confidence": 0.0,
            "score": top_score,
            "score_margin": margin,
        })
    if top_score <= 0.0:
        return _decision({
            "document_family": "other",
            "decision": "fallback",
            "reason": "no_rules_matched",
            "confidence": 0.0,
            "score": 0.0,
            "score_margin": margin,
        })
    if top_score < applied_threshold:
        return _decision({
            "document_family": "other",
            "decision": "fallback",
            "reason": "score_below_threshold",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        })
    if margin < float(min_score_margin):
        return _decision({
            "document_family": "other",
            "decision": "abstained",
            "reason": "ambiguous_rule_scores",
            "confidence": reported,
            "score": top_score,
            "score_margin": margin,
        })
    return _decision({
        "document_family": top_family,
        "decision": "classified",
        "reason": "score_above_threshold",
        "confidence": reported,
        "score": top_score,
        "score_margin": margin,
    })


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
    gates = evaluate_family_gates(features, indicators)

    # A gate that is not satisfied removes the family from contention: its
    # evidence exists but is not of a kind that may decide the family. The
    # pre-gate score is retained for error analysis, and the gated score is the
    # one that ranks, that is reported as ``candidate_scores``, and that a
    # risk-coverage replay re-reads — so the curve describes the policy that
    # actually ran. Gates apply in every mode, including ``observe``: they are
    # part of what counts as evidence, not part of the rejection policy.
    pre_gate_scores = {family: entry["score"] for family, entry in breakdown.items()}
    for family, entry in breakdown.items():
        gate = gates.get(family)
        if gate is not None and gate["status"] != "satisfied":
            entry["score"] = 0.0

    ranked = sorted(
        ((family, entry["score"]) for family, entry in breakdown.items()),
        key=lambda item: (-item[1], item[0]),
    )
    family_thresholds = resolve_family_thresholds(confidence_threshold)
    decision = apply_decision_policy(
        ranked,
        int(features.get("alnum_character_count") or 0),
        confidence_threshold,
        min_score_margin,
        min_recognized_characters,
        mode,
        family_thresholds,
    )

    top_family, top_score = ranked[0]
    runner_up_family, runner_up_score = ranked[1]
    selected = decision["document_family"]

    candidate_scores = {family: entry["score"] for family, entry in sorted(breakdown.items())}
    candidate_scores["other"] = 0.0

    result = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "feature_extraction_version": features.get("feature_extraction_version"),
        "classifier_version": CLASSIFIER_VERSION,
        # The two identities a stored result needs to be reproducible: which
        # rules and policy produced it, and which feature definitions it read.
        "rule_fingerprint": RULE_FINGERPRINT,
        "feature_fingerprint": features.get("feature_fingerprint"),
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
            # Why each gated family could or could not be decided. This is the
            # auditable half of the gate: the score alone cannot say whether a
            # family was out of contention for lack of evidence or because a
            # guard vetoed it.
            "family_gates": gates,
            "pre_gate_scores": {
                family: score for family, score in sorted(pre_gate_scores.items())
            },
            "gated_families": sorted(
                family
                for family, gate in gates.items()
                if gate["status"] != "satisfied" and pre_gate_scores.get(family, 0.0) > 0.0
            ),
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
            # Declared per-family operating points, resolved at this global
            # threshold. Exposed so that a lower bar for one family is visible
            # in the output rather than hidden in the module.
            "family_confidence": family_thresholds,
            "declared_family_confidence": dict(FAMILY_CONFIDENCE_THRESHOLDS),
            "applied_confidence": decision["applied_confidence_threshold"],
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