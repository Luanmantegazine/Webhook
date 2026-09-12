# Rules-Based Document Classification Prototype

## Scope

This prototype adds deterministic document-family classification to Hydra while
reusing the existing scan-processing engine. It targets RVL-CDIP images and
runs after `aggregate_document_content`, before translated text or template
rendering can alter the source signals.

The prototype supports:

- `research_paper`
- `technical_report`
- `business_report`
- `financial_document`
- `form_structured`
- `presentation_marketing`
- `correspondence`
- `resume`
- `news_article`
- `other`

RVL-CDIP does not provide reliable equivalents for `legal_document` or
`manual_procedure`; those families are intentionally not evaluated in this
version.

`other` is the residual class *and* the destination of every abstention and
fallback. `handwritten` and `file folder` map to it; letters, memos and e-mails
map to `correspondence`, `resume` to `resume`, and `news article` to
`news_article`. Keeping those five RVL labels in `other` — as taxonomy v1 did —
gives four families a ground-truth label the classifier can never predict, so
their recall reads as zero for a reason that has nothing to do with rule
quality.

## Taxonomy: one module, and what is still open

`tasks/document/rvl_cdip_eval.py` is the single source of truth for the family
list, the RVL-CDIP label mapping and the rejection targets. The classifier
imports `DOCUMENT_FAMILIES`, `SCORED_FAMILIES` and `TAXONOMY_VERSION` from it;
the evaluator resolves every manifest row through `resolve_evaluation_target`;
`config/rvl_cdip_taxonomy.json` is a declarative copy that `verify_config_file()`
checks on every evaluation run and refuses to reconcile silently.

| Label | Family | Status |
| --- | --- | --- |
| `scientific publication` | `research_paper` | Settled |
| `scientific report` | `technical_report` (default) or `research_paper` | **Open** — `scientific_report_family` |
| press releases (no RVL class) | not `news_article` (default) | **Open** — `press_release_policy` |

Both open decisions are declared in one place, reported under
`versions.taxonomy.pending_decisions` in every `report.json`, and configurable
through the same keys in `config/rvl_cdip_taxonomy.json`. Changing
`scientific_report_family` changes ground truth: metrics computed under
different selections must not be pooled. `press_release_policy` changes only
the `news_article` gate, not the labels.

## Versions and fingerprints

| Identifier | Moves when |
| --- | --- |
| `SCHEMA_VERSION` | The external contract of the feature record or the result changes. Adding a field does not move it. |
| `TAXONOMY_VERSION` | The family set or the label mapping changes. |
| `FEATURE_EXTRACTION_VERSION` | Any derived feature's definition changes. |
| `feature_fingerprint` | The emitted feature key set or the extraction version changes. Recomputable from the record itself. |
| `CLASSIFIER_VERSION` / `rule_fingerprint` | Any rule, weight, group, channel, gate, blocker, family threshold or decision-group count changes. |

The rule fingerprint is computed from rule *source* where available rather than
from bytecode reprs: nested code objects render with their memory address, so
the previous digest differed on every process — a fingerprint that cannot be
compared across runs cannot support a reproducibility claim.

## Design

The solution has three layers:

1. `convert_scanned_image_to_pdf` adapts TIFF/PNG/JPEG scans to the PDF entry
   point already used by Hydra.
2. `extract_document_classification_features` turns the aggregated document
   and measured page sizes into a shared feature contract.
3. `classify_document_rules` applies weighted, explainable rules and returns a
   confidence-gated decision.

Classification runs as a parallel observer in the translation workflow. It
does not select a template in this version.

## Files to copy into Hydra

| Prototype path | Hydra target |
| --- | --- |
| `tasks/document/rvl_cdip_eval.py` | `tasks/document/rvl_cdip_eval.py` |
| `tasks/document/rules_classifier_core.py` | `tasks/document/rules_classifier_core.py` |
| `tasks/document/word_geometry.py` | `tasks/document/word_geometry.py` |
| `tasks/document/extract_document_classification_features.py` | `tasks/document/extract_document_classification_features.py` |
| `tasks/document/classify_document_rules.py` | `tasks/document/classify_document_rules.py` |
| `tasks/dataset/convert_scanned_image_to_pdf.py` | `tasks/dataset/convert_scanned_image_to_pdf.py` |
| `workflows/document_translated_with_templates_rules_observe.json` | `workflows/document_translated_with_templates.json` after review |
| `workflows/document_classification_rules_benchmark.json` | `workflows/document_classification_rules_benchmark.json` |
| `config/rvl_cdip_taxonomy.json` | Suggested classification config directory |

Task discovery must include `tasks/dataset` and the two new modules in
`tasks/document`. If Hydra uses an explicit task registry, add those imports to
the registry.

## Workflow integration

The updated translation workflow makes one correction and adds two tasks:

- Declares the existing `page_sizes` output of
  `detect_and_extract_layout_doctr`.
- Runs `extract_document_classification_features` after aggregation.
- Runs `classify_document_rules` in `observe` mode.

The new workflow outputs are:

```json
{
  "classification_features": "tasks.extract_classification_features.classification_features",
  "classification": "tasks.classify_document.classification"
}
```

`reconstruct_with_template` remains unchanged. The only installed template is
still `clean_article`, so `recommended_template` is `null` and
`fallback_template` is `clean_article`.

## Classification contract

Example result:

```json
{
  "schema_version": "2.1",
  "taxonomy_version": "rvl-cdip-2.0",
  "feature_extraction_version": "2.3",
  "classifier_version": "rules-rvl-cdip-v5+<rule_fingerprint>",
  "rule_fingerprint": "<12 hex>",
  "feature_fingerprint": "ff-<12 hex>",
  "classifier": "rules",
  "mode": "evaluate",
  "provenance": {},
  "document_family": "financial_document",
  "confidence": 0.9,
  "score": 0.9,
  "decision": "classified",
  "reason": "score_above_threshold",
  "top_candidate": "financial_document",
  "runner_up": "business_report",
  "score_margin": 0.59,
  "candidate_scores": {},
  "decision_mass": {},
  "available_mass": {},
  "evidence": {
    "family_gates": {},
    "pre_gate_scores": {},
    "gated_families": [],
    "rules_by_family": {},
    "rules_triggered": [
      {"rule": "financial_document.invoice_identifier", "weight": 0.42},
      {"rule": "financial_document.amount_due", "weight": 0.24}
    ],
    "suppressed_by_grouping": {},
    "top_features": {}
  },
  "thresholds": {
    "confidence": 0.6,
    "family_confidence": {"correspondence": 0.5},
    "declared_family_confidence": {"correspondence": 0.5},
    "applied_confidence": 0.6,
    "minimum_score_margin": 0.1,
    "minimum_recognized_characters": 20
  },
  "execution_time_ms": 1.4,
  "recommended_template": null,
  "fallback_template": "clean_article"
}
```

`confidence` is `score` clipped to `[0, 1]` for downstream consumers; `score`
is the unclipped evidence ratio the thresholds are actually applied to. Neither
is a probability. `recommended_template` is populated only in `auto` mode.

Possible decisions:

| Decision | Meaning |
| --- | --- |
| `classified` | Score and margin passed their thresholds |
| `fallback` | No family reached the minimum score; result is `other` |
| `abstained` | OCR was insufficient or the leading categories were ambiguous |
| `observed` | `observe` mode only: argmax reported with no abstention |

`observe` exists to separate rule quality from the rejection policy: it yields a
full-coverage confusion matrix. `evaluate` is the regime whose risk-coverage
curve should be reported.

## Family gates

Scoring answers "how much evidence is there"; a gate answers the question a
weighted sum cannot — *is this the kind of evidence that may decide this family
at all?* Gates are declared as data in `FAMILY_GATES`, evaluated by
`evaluate_family_gates`, and applied in exactly one place: a family whose gate
is not satisfied has its score set to `0.0` before ranking. The pre-gate score
is kept in `evidence.pre_gate_scores`, and `evidence.family_gates` records, per
family, which evidence groups fired, which rules merely corroborated, and which
guard vetoed it.

Negative weights were the alternative, and they are close to unreadable: a large
negative weight both suppresses a family and rescales every score around it, and
no reader of the output can tell which of the two happened.

| Family | Accepts when | Corroborating only | Guards against |
| --- | --- | --- | --- |
| `form_structured` | one strong primary (`form_heading`, `questionnaire_heading`) **or** two primaries (`checkboxes`, `blank_fields`) | `field_labels`, `short_field_regions`, `label_value_lines`, `tab_stop_alignment`, `field_geometry_regularity` | invoices, specifications, news, advertisements, budgets, resumes |
| `correspondence` | two independent signals among header block, e-mail markers, salutation, closing, memo heading, letter geometry, letter body | — | forms, news reporting |
| `research_paper` | two of: academic structure, citations, editorial metadata, academic layout | `academic_vocabulary` | news reporting, invoices, forms |
| `news_article` | two groups, one of which must be a journalistic source, a dateline, or the news layout | — | scientific publications, advertisements, forms, institutional correspondence, press releases |
| `presentation_marketing` | one positive visual evidence: relevant pictures, landscape slide structure, short title over lists, or high visual area against low narrative density | `presentation_terms` | sparse or low-quality OCR with no visual evidence, forms, invoices |

Three consequences worth stating explicitly, because they were the family's
failure modes:

- `field_labels` alone, `label_value_lines` alone, and geometry alone can never
  classify a form. The old broad `field_grid` rule is deliberately not
  reintroduced: prose, tables and columned reports all satisfy it.
- Generic academic vocabulary (`results`, `method`, `study`, `report`) and a
  bare date can never classify a research paper.
- Short text, sparse text and low OCR confidence are guards, never evidence. A
  handwritten page is not a presentation for having little text on it.

### Per-family operating points

`correspondence` is accepted at a lower threshold than the rest — the family's
signals are individually weak and jointly decisive — and this is declared in
`FAMILY_CONFIDENCE_THRESHOLDS`, not bought by lowering the global threshold for
every family. A declared value is an *offset* from the module default, resolved
by `resolve_family_thresholds` at whatever global threshold is in force, so a
swept risk-coverage curve keeps describing the policy that actually runs. Every
result reports `thresholds.family_confidence` and the
`applied_confidence_threshold` of its own decision.

## Default thresholds

```json
{
  "classification_confidence_threshold": 0.60,
  "classification_min_score_margin": 0.10,
  "classification_min_recognized_characters": 20,
  "classification_max_text_chars": 20000
}
```

These live in `tasks/document/rules_classifier_core.py` as
`DEFAULT_CONFIDENCE_THRESHOLD`, `DEFAULT_MIN_SCORE_MARGIN` and
`DEFAULT_MIN_RECOGNIZED_CHARACTERS`; the task wrapper and
`scripts/evaluate_rules_from_cache.py` import them rather than restating them.
Change the operating point in one place only.

These are starting values, not final calibrated probabilities. Rule confidence
is a bounded evidence score — under v2 normalisation a score of 1.0 means "as
much evidence as the family's strongest groups can supply", so the v1 values
(0.45 / 0.08) no longer mean what they did. Thresholds must be tuned on the
RVL-CDIP validation split and frozen before the test split is evaluated.

## Preparing the RVL-CDIP benchmark

Create a balanced subset by target Hydra family, not by the original RVL
label. The 16 RVL labels collapse unevenly onto the 10 families — three map to
`correspondence`, two each to `form_structured`, `technical_report`,
`presentation_marketing` and `other` — so a subset balanced by RVL label is not
balanced by family.

Recommended first run:

- 100 validation images per Hydra family for rule development.
- 100 test images per Hydra family for the final report.
- Preserve `sample_id`, original RVL label, split, and target family.

The included preparation script streams and balances the subset:

```bash
python scripts/prepare_rvl_cdip_subset.py \
  --split validation \
  --per-family 100 \
  --output-directory data/rvl_cdip_subset
```

Repeat with `--split test` for the locked final benchmark. The two runs use the
same seed but independent official splits.

Run every scan through `document_classification_rules_benchmark.json`. Cache
the resulting `document`, `page_sizes`, and `classification_features`. Rule
calibration can then use the cached outputs without rerunning YOLO or docTR.

For offline evaluation, create a CSV manifest:

```csv
sample_id,rvl_label,target_family,document_path,page_sizes_path,page_words_path,classification_features_path,provenance_path,source_manifest_split
rvl-001,invoice,financial_document,cache/rvl-001/document.json,cache/rvl-001/page_sizes.json,cache/rvl-001/page_words.json,cache/rvl-001/features.json,cache/rvl-001/provenance.json,validation
rvl-002,form,form_structured,cache/rvl-002/document.json,cache/rvl-002/page_sizes.json,,,,validation
```

Features come from one of exactly two places, and `predictions.csv` records
which in `feature_source`:

- `classification_features_path` — the record the workflow stored, scored as it
  is (`feature_source=workflow_cache`);
- otherwise re-extracted from `document`, `page_sizes`, `page_words` and
  `provenance` (`feature_source=reextracted`) — the same four inputs the
  workflow feeds to `extract_document_classification_features`. When
  `provenance_path` is absent, a `provenance` object inside the cached document
  is used.

The run is **refused**, not degraded, when a feature record carries an
unsupported `schema_version` or `feature_extraction_version` or a fingerprint
that disagrees with its own contents; when a manifest `target_family` is outside
the taxonomy; when a stored artifact names a rule absent from `RULE_IDS`; when
feature versions or fingerprints are mixed inside one evaluation; or when
`config/rvl_cdip_taxonomy.json` contradicts `tasks/document/rvl_cdip_eval.py`.
Each of those produces numbers that look ordinary and describe nothing.

`report.json` carries a `versions` block with `SCHEMA_VERSION`,
`TAXONOMY_VERSION`, `FEATURE_EXTRACTION_VERSION`, `CLASSIFIER_VERSION`,
`rule_fingerprint`, `feature_fingerprint`, the full `rule_ids` list and the
resolved taxonomy — including its pending decisions. Every row of
`predictions.csv` carries the same six identifiers, so a prediction can be
traced to the system that produced it without consulting the run that wrote it.

Then run:

```bash
python scripts/evaluate_rules_from_cache.py benchmark_manifest.csv \
  --output-directory output/rules-benchmark
```

Generated results:

- `predictions.csv`
- `report.json`

### Reading the report

`other` is a real family *and* the sink for every abstention and fallback, and
in `evaluate` mode the classifier can never positively predict it: a family is
only ever returned on the `classified` path, and that family always comes from
the scored nine. **Every `other` in an `evaluate` run is a refusal, not a
prediction.** Scoring refusals as predictions gives the classifier a true
positive for `other` each time it declines to answer a file folder or a
handwritten page, which inflates `other` precision and recall and, through the
macro average, the headline number too.

The report therefore separates three questions:

| Block | Question it answers |
| --- | --- |
| `decisions` | How often did it answer at all? Coverage, refusal rate, and the reason breakdown. |
| `selective` | How good are the answers it gave? P/R/F1 over accepted predictions only — this is rule quality. |
| `end_to_end` | `accuracy_declined_as_error` treats a refusal as wrong. `accuracy_declined_as_other` treats it as routing to the fallback template — the deployment view, and the number older reports called plain "accuracy". |
| `confusion_matrix` | Every sample, with refusals in an explicit `<declined>` column rather than folded into `other`. |

`selective.macro_*` averages over families with non-zero support. A family that
is predicted but never present cannot be averaged over — its recall is
undefined, not zero — so its false positives are reported under
`predictions_outside_support` instead of vanishing from macro precision.

### Full-coverage mode

```bash
python scripts/evaluate_rules_from_cache.py benchmark_manifest.csv --mode observe
```

`observe` disables abstention, so the argmax is always reported and the
confusion matrix is complete. Run it alongside the `evaluate` run: comparing
the two is what separates rule quality from the rejection policy. It is also
where families that quietly absorb OCR failures become visible — an illegible
scan still has an argmax.

### Risk-coverage curve

Every run sweeps the confidence threshold and writes a `risk_coverage` block
(disable with `--no-risk-coverage`). The sweep re-runs only
`apply_decision_policy` over the stored scores, never the rule engine, so the
curve is guaranteed to describe the same firings as the reported operating
point.

This needs `alnum_character_count`, which `predictions.csv` now carries: the
OCR-sufficiency gate is part of the decision policy, and a sweep that cannot
see that count silently mis-reports the whole low-threshold end of the curve as
higher coverage than the classifier would really give.

## Important evaluation boundaries

- Never report `end_to_end.accuracy_declined_as_other` as "accuracy" without
  the qualifier. It credits the classifier for refusing to answer.
- Report coverage next to every selective metric. A high `selective.accuracy`
  at low coverage is a classifier that answers only the easy documents.
- Tune rules and thresholds only on RVL-CDIP validation data.
- Use the RVL-CDIP test split once for the final reported result.
- Report classifier latency separately from shared OCR/layout preprocessing.
- Retain misclassified OCR text and triggered rules for error analysis.
- Do not treat `advertisement` and `presentation` as semantically identical;
  they are temporarily grouped because the candidate Hydra taxonomy gives them
  the same presentation profile.
- Do not map `specification` to manuals or contracts merely to fill missing
  categories.

## Tests

From the prototype root:

```bash
python -m unittest discover -s tests -v
```

`tests/test_rules_classifier.py` covers feature extraction, page orientation,
invoice, research paper, technical report, business report, form, presentation,
fallback, insufficient-OCR abstention, the decision contract in all three
modes, and agreement between `config/rvl_cdip_taxonomy.json` and the
classifier's own family list.

`tests/test_evaluator_metrics.py` covers the evaluator's metric layer, which is
where the experimental methodology lives: that refusals stay out of the
classification metrics, that they land in the `<declined>` confusion column,
that both end-to-end readings are reported and differ, and that false positives
on a zero-support family are surfaced rather than dropped.
