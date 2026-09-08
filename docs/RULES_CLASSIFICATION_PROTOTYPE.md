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
- `other`

RVL-CDIP does not provide reliable equivalents for `legal_document` or
`manual_procedure`; those families are intentionally not evaluated in this
version.

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
  "schema_version": "1.0",
  "taxonomy_version": "rvl-cdip-1.0",
  "classifier_version": "rules-rvl-cdip-v1",
  "classifier": "rules",
  "mode": "observe",
  "document_family": "financial_document",
  "confidence": 0.9,
  "decision": "classified",
  "reason": "high_confidence_rule_match",
  "top_candidate": "financial_document",
  "runner_up": "business_report",
  "score_margin": 0.59,
  "candidate_scores": {},
  "evidence": {
    "rules_triggered": [
      {"rule": "invoice_identifier", "weight": 0.42},
      {"rule": "amount_due", "weight": 0.24}
    ],
    "top_features": {}
  },
  "execution_time_ms": 1.4,
  "recommended_template": null,
  "fallback_template": "clean_article"
}
```

Possible decisions:

| Decision | Meaning |
| --- | --- |
| `classified` | Score and margin passed their thresholds |
| `fallback` | No family reached the minimum score; result is `other` |
| `abstained` | OCR was insufficient or the leading categories were ambiguous |

## Default thresholds

```json
{
  "classification_confidence_threshold": 0.45,
  "classification_min_score_margin": 0.08,
  "classification_min_recognized_characters": 20,
  "classification_max_text_chars": 20000
}
```

These are starting values, not final calibrated probabilities. Rule confidence
is a bounded evidence score. Thresholds must be tuned on the RVL-CDIP
validation split and frozen before the test split is evaluated.

## Preparing the RVL-CDIP benchmark

Create a balanced subset by target Hydra family, not by the original RVL label.
Otherwise the seven original labels mapped to `other` will dominate the
benchmark.

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

The report includes accuracy, macro precision/recall/F1, coverage, accuracy on
accepted predictions, class metrics, confusion matrix, and classifier latency
at mean/P50/P95/P99.

## Important evaluation boundaries

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

The test suite covers feature extraction, page orientation, invoice, research
paper, technical report, business report, form, presentation, fallback, and
insufficient-OCR abstention.
