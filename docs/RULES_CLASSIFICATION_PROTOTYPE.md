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

## Taxonomy: one module, and the decisions it records

`tasks/document/rvl_cdip_eval.py` is the single source of truth for the family
list, the RVL-CDIP label mapping and the rejection targets. The classifier
imports `DOCUMENT_FAMILIES`, `SCORED_FAMILIES` and `TAXONOMY_VERSION` from it;
the evaluator resolves every manifest row through `resolve_evaluation_target`;
`config/rvl_cdip_taxonomy.json` is a declarative copy that `verify_config_file()`
checks on every evaluation run and refuses to reconcile silently.

| Label | Family | Status |
| --- | --- | --- |
| `scientific publication` | `research_paper` | Settled |
| `scientific report` | `technical_report` | **Accepted in v6** — `scientific_report_family` |
| press releases (no RVL class) | not `news_article` | **Accepted in v6** — `press_release_policy` |

Both decisions are reported under `versions.taxonomy.decisions` in every
`report.json` and remain configurable through the same keys in
`config/rvl_cdip_taxonomy.json`. They are inputs the benchmark rests on, not
open questions carried alongside the numbers. `scientific_report_family` is
ground truth: results from before and after any change to it must not be
pooled. `press_release_policy` changes only the `news_article` gate, never a
label.

## Versions and fingerprints

| Identifier | Moves when |
| --- | --- |
| `SCHEMA_VERSION` | The external contract of the feature record or the result changes. Adding a field does not move it. |
| `TAXONOMY_VERSION` | The family set or the label mapping changes. |
| `FEATURE_EXTRACTION_VERSION` | Any derived feature's definition changes. |
| `feature_fingerprint` | The emitted feature key set or the extraction version changes. Recomputable from the record itself. |
| `CLASSIFIER_VERSION` / `rule_fingerprint` | Any rule, weight, substitutable group, channel, gate, blocker, family threshold, decision-group count, **or global operating point** (threshold, minimum margin, minimum recognised characters) changes. |

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
  "classifier_version": "rules-rvl-cdip-v7+<rule_fingerprint>",
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
stays in `evidence.pre_gate_scores`, and `evidence.family_gates` records which
acceptance path opened, what the unsatisfied paths were missing, which guard
vetoed the family, and the declared refusal `reason`.

A gate is a set of **acceptance paths**. Each path may require named rules
(`all_of`), one rule from each of several pools (`any_of`), or *n* distinct
evidence units (`min_units`, where a unit is a group of rules that are
alternative readings of one observation). Several paths per family is the
point: "a questionnaire heading with one structural signal" and "a form heading
with two corroborating signals" are different cases with different evidence
bars, and collapsing them into one count over a bag of rules loses exactly that
distinction.

| Family | Acceptance paths | Corroborating only | Guards against |
| --- | --- | --- | --- |
| `form_structured` | **A** questionnaire heading + 1 structural signal · **B** checkboxes + 1 structural signal · **C** form heading + 2 structural signals | `field_labels`, `label_value_lines`, `tab_stop_alignment`, `field_geometry_regularity`, `short_field_regions`, `blank_fields` | invoices, specifications, news, advertisements, budgets, resumes |
| `correspondence` | two independent signals among header block, e-mail markers, salutation, closing, memo heading, letter geometry, letter body | — | forms, news reporting |
| `research_paper` | two of: academic structure, citations, editorial metadata, academic layout | `academic_vocabulary` | news reporting, invoices, forms |
| `news_article` | **A** byline + (attribution quotes or justified body) · **B** wire service + dateline + (attribution quotes or justified body) | `attribution_quotes`, `multi_column_body`, `justified_body` | scientific publications, advertisements, forms, institutional correspondence, press releases |
| `presentation_marketing` | deck vocabulary (`presentation_terms`) **and** one visual/structural corroboration | `visual_layout`, `landscape_layout`, `visual_dominance`, `sparse_centered`, `slide_structure`, `marketing_copy` | near-empty OCR, low-confidence OCR, forms, invoices |

Consequences worth stating explicitly, because each was a measured failure:

- **Visual and structural evidence cannot decide a presentation.** All five of
  `visual_layout`, `landscape_layout`, `visual_dominance`, `sparse_centered` and
  `slide_structure` share one scoring group, so a page contributes that evidence
  once however many ways it is measured, and a document firing only those rules
  is refused with `visual_evidence_only` at any threshold. This was tightened
  twice on measurement: v5 accepted 41 and got 6 right, every false accept
  firing `visual_layout` with `visual_dominance` at `0.6333`; v6 still accepted
  an invoice and a form, both firing `visual_layout` with `slide_structure` at
  `0.6562` — **higher than the only true positive at `0.625`**, which is why no
  threshold could separate them and why the fix had to be structural.
- **Advertising copy cannot decide one either.** `marketing_copy` corroborates
  but opens no path: on the development set it fired exactly once, for an
  invoice.
- **Structure alone cannot decide a form.** `field_labels`, `label_value_lines`
  and geometry corroborate and never open a path. The broad `field_grid` rule
  is not reintroduced under any name.
- **A byline and a column count are not a news article.** `byline +
  multi_column_body` and `wire_service + attribution_quotes` satisfy no path:
  both admitted advertisements and scientific publications.
- **Generic academic vocabulary** (`results`, `method`, `study`, `report`) and a
  bare date can never decide a research paper.
- **Short text is not a family.** Near-empty OCR is refused with
  `insufficient_presentation_text`; the bar is deliberately low, because slides
  *are* short and what keeps short picture pages out is the demand for
  independent primary evidence, not a word count.

### Rules removed in v6

| Rule | Development-set behaviour | Disposition |
| --- | --- | --- |
| `news_article.headline_body` | 6 firings, 0 news articles | Removed. "A titled region over 250 words that is neither table nor picture" describes most typed pages; the name claimed evidence the implementation never measured. Nothing replaces it until the feature record carries headline typography. |
| `presentation_marketing.bullet_layout` | 1 firing, 0 positives | Folded into `slide_structure`, which measures the same observation. Two rules for one observation summed into the decision twice. |

### Rules regrouped in v7

| Rule | Change | Why |
| --- | --- | --- |
| `presentation_marketing.slide_structure` | primary → member of the `visual_evidence` group | As an independent primary it let an invoice and a form reach `0.6562`. "A titled page of short text blocks" describes both of those as well as it describes a slide. |
| `presentation_marketing.marketing_copy` | primary → corroborating | Fired once on the development set, for an invoice. It still adds score once a case is open; it may no longer be the case. |

No predicate, weight or threshold in `form_structured`, `research_paper`,
`news_article`, `resume`, `technical_report` or `financial_document` was touched
in v7.

`presentation_marketing.presentation_terms` was rewritten rather than removed:
it had fired 6 times with 0 positives because it matched the bare word
"presentation" and mixed slide vocabulary with marketing copy. It is now deck
vocabulary in heading position, and the marketing half became `marketing_copy`.
`slide_structure` fired 0 times because it required landscape orientation on a
portrait-scanned corpus — a rule named for slide structure that was in fact
measuring page orientation — and now measures the slide shape itself.
`landscape_layout` also fired 0 times, but it measures what its name says and is
kept for corpora that carry landscape pages; grouped with the other visual
rules, it adds no decision mass.

### Per-family operating points

The global threshold stays at `0.60`. Five families declare their own:

| Family | Threshold | Status |
| --- | --- | --- |
| `resume` | 0.30 | development-set candidate |
| `technical_report` | 0.31 | development-set candidate |
| `correspondence` | 0.42 | development-set candidate (v7: was 0.40) |
| `form_structured` | 0.40 | development-set candidate |
| `research_paper` | 0.43 | development-set candidate |

**These are candidates read off the development split, not calibrated
thresholds**, and `FAMILY_THRESHOLD_PROVENANCE` carries that status
(`development_set_candidate_requires_holdout`) into every report so no reader
can mistake them for a calibration result. A holdout is what would make them
one.

The v7 move of `correspondence` from 0.40 to 0.42 was read off the *same* 270
documents as the v6 value, so it is the same kind of candidate and not a firmer
one. Do not re-tune it on those documents again: a threshold fitted twice to one
split is fitted to that split, whatever the second reading shows.

`presentation_marketing`, `news_article` and `financial_document` are held at
the global threshold on purpose, recorded in `FAMILY_THRESHOLD_HOLDS` with the
reason: their problem was precision, and a lower bar is the one change that
cannot help it.

A declared value is an *offset* from the module default, resolved by
`resolve_family_thresholds` at whatever global threshold is in force, so a swept
risk-coverage curve keeps describing the policy that actually runs. Every
prediction row reports `effective_family_threshold` beside
`global_confidence_threshold`: a family judged at its own bar and reported under
the global number is a lower bar that no table shows.

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

Every run writes, in addition to `predictions.csv` and `report.json`:

| Artifact | Contents |
| --- | --- |
| `routing_by_family.csv` | Routing precision and recall for **every** scorable family, not only a scoped subset. Rejection targets count as negatives, so accepting a file folder is a false positive for the family that accepted it. |
| `metrics_by_rvl_label.csv` | The same questions per original RVL-CDIP label. Three labels collapse onto `correspondence`: a family at 70% built from one label at 100% and another at 10% is not a family at 70%. |
| `family_risk_coverage_curve.*` | One curve per scorable family, with rejection targets as negatives and `effective_family_threshold` on every point. |
| `diagnostics_review.*` | Ordered by what an error costs, not by confidence. |

Four fields answer "can an accepted answer be trusted", and sit at the top of
`metrics`:

| Field | Meaning |
| --- | --- |
| `accepted_routing_accuracy` | Of the accepted answers, the share naming the right family. |
| `accepted_wrong_family_count` | Accepted answers naming the wrong family. |
| `unsafe_accept_count` | The above, plus every rejection target that was accepted at all. |
| `unsafe_accept_rate` | Unsafe accepts over all accepted decisions. |

An unsafe accept is not the same as "not correct": a refusal costs coverage, an
unsafe accept costs trust.

`diagnostics_review` no longer carries a single `correct` column — it answered
three questions at once, so a correctly routed out-of-scope document and a
wrongly accepted file folder both read `False` and sorted together. It now
carries `canonical_correct`, `scope_correct`, `is_unsafe_accept` and
`is_rejection_false_accept`, and rows are ordered by review priority:

1. `rejection_target_accepted` — a document that should have been declined;
2. `accepted_wrong_family` — an accepted answer naming the wrong family;
3. `high_confidence_false_positive` — the same error, told confidently;
4. `false_negative_near_threshold` — a refusal that just missed its bar;
5. `fallback_without_rules` — a gap in coverage, not a wrong answer.


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

## Routing readiness

No family is production-ready: every number here comes from a development split
with no holdout, so `ROUTING_RELEASE_STATUS` marks each scorable family
`development_only` and every result carries
`released_for_automatic_routing: false`.

Two families are withheld from automatic routing even in development, and stay
withheld in v7:

| Family | Reason |
| --- | --- |
| `presentation_marketing` | Gate rebuilt for the second consecutive version; its precision has not been measured since. |
| `financial_document` | Nothing in the development run examined its precision. |

## Classifier latency

`scripts/benchmark_classifier_latency.py` measures `classify_with_rules` and
nothing else:

```bash
python scripts/benchmark_classifier_latency.py benchmark_manifest.csv \
  --output output/classifier_latency.json --repetitions 20 --trials 3
```

Every feature record is loaded and validated before timing starts, so module
import, JSON reading, OCR, layout detection and feature extraction are outside
the measured region entirely. A warm-up pass runs first. The two modes are timed
separately and three independent trials run end to end; the report gives mean,
median, p90, p95, p99, standard deviation, min and max per trial, plus the
median of the three p95 values.

The report's `measurement` block names what was measured and what was not.
`indicators_off` is the operational series: `include_indicators=True` builds a
diagnostic vector production never asks for, and quoting its latency as the
pipeline's overstates it. The script also verifies that both modes produce the
same decision, family, scores and margins, and exits non-zero if they do not —
a cheaper measurement of a different classifier is not a measurement of this
one.

This is classifier time only. End-to-end latency is dominated by OCR and layout
detection, which this script does not measure at all.

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

`tests/test_family_gates.py` covers the v6 gates. Each test encodes a failure
the development run actually produced: visual-only evidence refused with
`visual_evidence_only` and a zero score, the three declared presentation refusal
reasons all reachable, the five development-set form shapes recovered by paths A
and B while six form-shaped negatives stay refused, `byline + multi_column_body`
and `wire_service + attribution_quotes` refused, the removed rules absent from
`RULE_IDS`, the declared thresholds and their provenance, and the rule
fingerprint stable across processes.

`tests/test_evaluator_metrics.py` covers the evaluator's metric layer, which is
where the experimental methodology lives: that refusals stay out of the accepted
metrics and earn no true positives, that unsafe accepts are counted and separated
from ordinary errors, that routing is measured for every scorable family, that
per-label metrics exist, that rejection targets are negatives in the curves, and
that the review order puts an accepted rejection target first. It was rewritten
in v6: it had been importing an evaluator API that no longer existed, so the
whole module raised `ImportError` on collection and every assertion in it had
silently stopped running.
