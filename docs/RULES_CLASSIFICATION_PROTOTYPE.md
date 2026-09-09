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
fallback. `config/rvl_cdip_taxonomy.json` maps only `handwritten` and
`file folder` to it; letters, memos and e-mails map to `correspondence`,
`resume` to `resume`, and `news article` to `news_article`. Keeping those five
RVL labels in `other` — as taxonomy v1 did — gives four families a ground-truth
label the classifier can never predict, so their recall reads as zero for a
reason that has nothing to do with rule quality. `tests/test_rules_classifier.py`
asserts that the config and `DOCUMENT_FAMILIES` agree.

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
| `tasks/document/rules_classifier_core.py` | `tasks/document/rules_classifier_core.py` |
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
  "schema_version": "2.0",
  "taxonomy_version": "rvl-cdip-2.0",
  "classifier_version": "rules-rvl-cdip-v3",
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
    "rules_by_family": {},
    "rules_triggered": [
      {"rule": "financial_document.invoice_identifier", "weight": 0.42},
      {"rule": "financial_document.amount_due", "weight": 0.24}
    ],
    "suppressed_by_grouping": {},
    "top_features": {}
  },
  "thresholds": {},
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
sample_id,rvl_label,target_family,document_path,page_sizes_path
rvl-001,invoice,financial_document,cache/rvl-001/document.json,cache/rvl-001/page_sizes.json
rvl-002,form,form_structured,cache/rvl-002/document.json,cache/rvl-002/page_sizes.json
```

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
