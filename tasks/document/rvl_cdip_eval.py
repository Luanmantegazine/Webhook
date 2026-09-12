"""Canonical document taxonomy and RVL-CDIP evaluation targets.

Why this module exists
----------------------
The label-to-family mapping used to live in three places at once: the shipped
``config/rvl_cdip_taxonomy.json``, the evaluator's own reading of the manifest,
and the classifier's family tuple. Three copies of one decision is three ways
for the taxonomy to change silently — and a taxonomy that changes silently
makes every number reported before and after it incomparable without saying so.

This module is the single source of truth. The classifier imports its family
tuple from here, the evaluator resolves every manifest row through
:func:`resolve_evaluation_target`, and the shipped JSON is *verified* against
this module rather than read as a second opinion (:func:`verify_config_file`).

Documented mapping decisions
----------------------------
``scientific publication -> research_paper``
    Settled. RVL-CDIP's "scientific publication" class is the peer-reviewed
    article: abstract, references, citations, two-column typesetting.

``scientific report -> technical_report`` (default) *or* ``research_paper``
    **Not settled.** RVL-CDIP's "scientific report" class mixes laboratory and
    institutional reports (closer to ``technical_report``) with preprint-shaped
    documents (closer to ``research_paper``). The choice is a taxonomy
    decision, not a rule-quality question, so it is declared here, exposed in
    every report through :func:`taxonomy_descriptor`, and configurable in one
    place via ``scientific_report_family`` in the taxonomy config file. The
    default preserves the mapping shipped so far; changing it changes ground
    truth and therefore every metric that depends on it.

``press release -> (no RVL-CDIP class)``
    RVL-CDIP has no press-release class, but press releases do appear inside
    the news-article and correspondence classes and carry news-shaped surface
    signals (dateline, attribution quotes, wire-service names). Whether a press
    release should be accepted as ``news_article`` is a taxonomy decision, so it
    is declared here as ``press_release_policy`` and consumed by exactly one
    classifier gate. The default, ``not_news_article``, treats press-release
    markers as a blocker for ``news_article`` — an institutional announcement
    is not journalism.

Rejection targets
-----------------
``handwritten`` and ``file folder`` are not classifier families. They are
targets the classifier is expected to *decline*: the only correct outcome is
``abstained`` or ``fallback``. They are reported separately from classification
accuracy so that declining them is never scored as a correct prediction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple


class TaxonomyConfigError(RuntimeError):
    """The shipped taxonomy config contradicts this module."""


#: Bumped whenever the family set or the label mapping changes. Any stored
#: result carrying a different value was produced against a different ground
#: truth and must not be pooled with the current one.
TAXONOMY_VERSION = "rvl-cdip-2.0"

#: Public family taxonomy. ``other`` is both the residual class and the
#: destination of every abstention; ``business_report`` is retained for
#: downstream compatibility but is not an RVL-CDIP classification candidate.
DOCUMENT_FAMILIES: tuple[str, ...] = (
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

#: Families excluded from scoring: ``other`` because it is the refusal sink,
#: ``business_report`` because no canonical RVL-CDIP target maps to it, so it
#: could only ever be a false positive in the canonical evaluation.
UNSCORED_FAMILIES = frozenset({"other", "business_report"})

SCORED_FAMILIES: tuple[str, ...] = tuple(
    family for family in DOCUMENT_FAMILIES if family not in UNSCORED_FAMILIES
)

#: Families for which RVL-CDIP provides no reliable equivalent.
NOT_EVALUATED_WITH_RVL_CDIP: tuple[str, ...] = ("legal_document", "manual_procedure")

#: Corpus classes that are targets for *refusal*, not for classification.
REJECTION_LABELS = frozenset({"handwritten", "file folder"})

#: The configurable half of the mapping. See the module docstring.
SCIENTIFIC_REPORT_FAMILY_CHOICES: tuple[str, ...] = ("technical_report", "research_paper")
DEFAULT_SCIENTIFIC_REPORT_FAMILY = "technical_report"

PRESS_RELEASE_POLICY_CHOICES: tuple[str, ...] = ("not_news_article", "news_article")
DEFAULT_PRESS_RELEASE_POLICY = "not_news_article"

#: The settled half of the mapping: every RVL-CDIP class except the one whose
#: destination is still open. ``scientific report`` is added by
#: :func:`_build_mapping` from the configured choice.
_SETTLED_LABEL_MAPPING: dict[str, str] = {
    "letter": "correspondence",
    "form": "form_structured",
    "email": "correspondence",
    "handwritten": "other",
    "advertisement": "presentation_marketing",
    "scientific publication": "research_paper",
    "specification": "technical_report",
    "file folder": "other",
    "news article": "news_article",
    "budget": "business_report",
    "invoice": "financial_document",
    "presentation": "presentation_marketing",
    "questionnaire": "form_structured",
    "resume": "resume",
    "memo": "correspondence",
}

#: Location of the shipped declarative copy. It is verified against this
#: module, never read as an independent mapping.
CONFIG_PATH = Path(
    os.environ.get("HYDRA_TAXONOMY_CONFIG")
    or (Path(__file__).resolve().parents[2] / "config" / "rvl_cdip_taxonomy.json")
)


def _normalise_label(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("-", " ").replace("_", " ")
    return " ".join(text.split()).casefold()


def _read_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise TaxonomyConfigError(f"Cannot read taxonomy config {CONFIG_PATH}: {error}") from error
    if not isinstance(payload, dict):
        raise TaxonomyConfigError(f"Taxonomy config {CONFIG_PATH} must be a JSON object.")
    return payload


def _resolve_choice(payload: dict[str, Any], key: str, choices: tuple[str, ...], default: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        return default
    candidate = _normalise_label(value).replace(" ", "_")
    if candidate not in choices:
        raise TaxonomyConfigError(
            f"Taxonomy config {CONFIG_PATH} sets {key}={value!r}; "
            f"supported values are: {', '.join(choices)}."
        )
    return candidate


_CONFIG = _read_config()

#: The open taxonomy decisions, resolved once, here.
SCIENTIFIC_REPORT_FAMILY = _resolve_choice(
    _CONFIG,
    "scientific_report_family",
    SCIENTIFIC_REPORT_FAMILY_CHOICES,
    DEFAULT_SCIENTIFIC_REPORT_FAMILY,
)
PRESS_RELEASE_POLICY = _resolve_choice(
    _CONFIG,
    "press_release_policy",
    PRESS_RELEASE_POLICY_CHOICES,
    DEFAULT_PRESS_RELEASE_POLICY,
)

#: RVL-CDIP class -> Hydra family. The only mapping in the codebase.
CLASS_TO_FAMILY: dict[str, str] = dict(
    sorted({**_SETTLED_LABEL_MAPPING, "scientific report": SCIENTIFIC_REPORT_FAMILY}.items())
)

#: Decisions that are declared but not settled. Reported verbatim so a reader
#: of any benchmark output can see what is still open.
PENDING_TAXONOMY_DECISIONS: tuple[dict[str, Any], ...] = (
    {
        "decision": "scientific_report_family",
        "status": "pending",
        "selected": SCIENTIFIC_REPORT_FAMILY,
        "choices": list(SCIENTIFIC_REPORT_FAMILY_CHOICES),
        "affects": ["ground_truth"],
        "note": (
            "RVL-CDIP 'scientific report' mixes institutional/laboratory reports with "
            "preprint-shaped documents. Changing this changes ground truth, so metrics "
            "computed under different selections must not be pooled."
        ),
    },
    {
        "decision": "press_release_policy",
        "status": "pending",
        "selected": PRESS_RELEASE_POLICY,
        "choices": list(PRESS_RELEASE_POLICY_CHOICES),
        "affects": ["news_article_gate"],
        "note": (
            "RVL-CDIP has no press-release class. Under 'not_news_article' press-release "
            "markers block the news_article gate; under 'news_article' they do not."
        ),
    },
)


class EvaluationTarget(NamedTuple):
    """The evaluation target a corpus label resolves to.

    ``family`` is the canonical ground-truth family. ``kind`` separates the two
    reasons a document can carry ``other``: a genuine residual family versus a
    target the classifier is expected to decline.
    """

    label: str
    family: str
    kind: str  # "classifier_family" | "rejection"

    @property
    def is_rejection_target(self) -> bool:
        return self.kind == "rejection"


def known_labels() -> tuple[str, ...]:
    return tuple(sorted(CLASS_TO_FAMILY))


def is_known_family(family: Any) -> bool:
    return _normalise_label(family).replace(" ", "_") in set(DOCUMENT_FAMILIES)


def resolve_evaluation_target(label: Any) -> EvaluationTarget:
    """Resolve a corpus label into its canonical evaluation target.

    Raises ``ValueError`` for any label outside the taxonomy: an unknown label
    silently mapped to ``other`` is a ground-truth error that reads as a
    classifier error, which is exactly the confusion this module exists to stop.
    """
    normalised = _normalise_label(label)
    if not normalised:
        raise ValueError("an empty rvl_label; every evaluated row needs a corpus label")
    if normalised not in CLASS_TO_FAMILY:
        raise ValueError(
            f"an unknown rvl_label {str(label)!r}; "
            f"known labels are: {', '.join(known_labels())}"
        )
    return EvaluationTarget(
        label=normalised,
        family=CLASS_TO_FAMILY[normalised],
        kind="rejection" if normalised in REJECTION_LABELS else "classifier_family",
    )


def taxonomy_descriptor() -> dict[str, Any]:
    """The taxonomy as it is actually configured, for embedding in reports."""
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "source_module": f"{__name__}.CLASS_TO_FAMILY",
        "config_path": str(CONFIG_PATH),
        "config_present": CONFIG_PATH.is_file(),
        "document_families": list(DOCUMENT_FAMILIES),
        "scored_families": list(SCORED_FAMILIES),
        "unscored_families": sorted(UNSCORED_FAMILIES),
        "rejection_labels": sorted(REJECTION_LABELS),
        "not_evaluated_with_rvl_cdip": list(NOT_EVALUATED_WITH_RVL_CDIP),
        "label_mapping": dict(CLASS_TO_FAMILY),
        "scientific_report_family": SCIENTIFIC_REPORT_FAMILY,
        "press_release_policy": PRESS_RELEASE_POLICY,
        "pending_decisions": [dict(item) for item in PENDING_TAXONOMY_DECISIONS],
    }


def verify_config_file(path: Path | None = None) -> dict[str, Any]:
    """Check that the shipped JSON says exactly what this module says.

    The JSON is a declarative copy for readers and for downstream tooling. If
    it disagrees with this module, one of the two has been edited alone, and
    the taxonomy has changed in a way nobody declared — so this raises rather
    than picking a winner.
    """
    config_path = Path(path) if path is not None else CONFIG_PATH
    if not config_path.is_file():
        return {"verified": False, "reason": "config_absent", "config_path": str(config_path)}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise TaxonomyConfigError(f"Cannot read taxonomy config {config_path}: {error}") from error

    problems: list[str] = []
    if payload.get("taxonomy_version") != TAXONOMY_VERSION:
        problems.append(
            f"taxonomy_version={payload.get('taxonomy_version')!r} != {TAXONOMY_VERSION!r}"
        )
    supported = payload.get("supported_families")
    if tuple(supported or ()) != DOCUMENT_FAMILIES:
        problems.append("supported_families differs from DOCUMENT_FAMILIES")
    mapping = payload.get("label_mapping")
    if not isinstance(mapping, dict):
        problems.append("label_mapping is missing or not an object")
    else:
        normalised = {_normalise_label(key): value for key, value in mapping.items()}
        if normalised != CLASS_TO_FAMILY:
            missing = sorted(set(CLASS_TO_FAMILY) - set(normalised))
            extra = sorted(set(normalised) - set(CLASS_TO_FAMILY))
            changed = sorted(
                label
                for label in set(CLASS_TO_FAMILY) & set(normalised)
                if normalised[label] != CLASS_TO_FAMILY[label]
            )
            problems.append(
                "label_mapping differs from CLASS_TO_FAMILY "
                f"(missing={missing}, unexpected={extra}, changed={changed})"
            )
    declared_rejections = payload.get("rejection_labels")
    if declared_rejections is not None and set(declared_rejections) != set(REJECTION_LABELS):
        problems.append("rejection_labels differs from REJECTION_LABELS")

    if problems:
        raise TaxonomyConfigError(
            f"Taxonomy config {config_path} contradicts {__name__}: " + "; ".join(problems)
        )
    return {"verified": True, "config_path": str(config_path)}
