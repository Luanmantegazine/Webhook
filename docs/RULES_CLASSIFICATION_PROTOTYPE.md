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
| `TAXONOMY_VERSION` | The family set, the label mapping or the subtype vocabulary changes. Now **rvl-cdip-2.1**: subtypes exist. No label was remapped, so ground truth is unchanged. |
| `FEATURE_EXTRACTION_VERSION` | Any derived feature's definition changes. Now **2.4**: v8 adds the newspaper-structure features and *redefines* `accounting_negative_count`, so a 2.3 cache is refused rather than pooled. |
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
  "feature_extraction_version": "2.4",
  "classifier_version": "rules-rvl-cdip-v8+<rule_fingerprint>",
  "rule_fingerprint": "<12 hex>",
  "feature_fingerprint": "ff-<12 hex>",
  "classifier": "rules",
  "mode": "evaluate",
  "provenance": {},
  "document_family": "financial_document",
  "document_subtype": null,
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
| `news_article` | **A** byline + reporting · **B** wire service + dateline + reporting · **C** masthead + issue metadata + editorial structure + layout + reporting · **D** running header + editorial structure + layout + reporting | `attribution_quotes`, `multilingual_reporting`, `justified_body`, `multi_column_body`, `multi_column_publication`, `newspaper_column_geometry` | scientific publications, advertisements, forms, institutional correspondence, press releases |
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

### The news publication family (v8)

`news_article` is a **legacy public key**. It is the RVL-CDIP class name and the
key every workflow reads, so it does not change. Conceptually the family is a
*news publication*, and it covers two document shapes that a consumer may well
want to treat differently:

| Subtype | Shape |
| --- | --- |
| `single_news_article` | A clipped article: one headline, a byline, a body. |
| `newspaper_issue` | A whole newspaper: a masthead, a column grid, many headlines with articles under them, photographs and advertisements. |

The shape is reported as `document_subtype` on the result and in
`evidence.family_gates.news_article`. The field is optional and additive —
`null` for every family that declares no subtypes and for every refusal — which
is why `SCHEMA_VERSION` does not move.

An issue was unreachable under v7, and every reason was a property of the rules
rather than of the document:

| v7 behaviour | v8 |
| --- | --- |
| The byline pattern matched only `By First Last`. | `By`, `Por`, `Por <role> <name>`, `Da redação`, `Reportagem de …`. |
| The dateline pattern was English-only. | `SÃO PAULO, 24 de janeiro de 2026` alongside `WASHINGTON, Apr. 4`. |
| Attribution was `said` / `according to`. | A multilingual reporting lexicon, and `attribution_quotes` now requires a verb *attached to a quotation*. |
| `two_column_ratio` answered a yes/no question about two columns. | Narrative columns are counted per page — two, three, four or more — from region left edges and, independently, from word geometry. |
| A whole issue has no byline and no single dateline of its own. | The issue is recognised by what it *is*: publication identity, editorial structure and a column grid. |

The newspaper paths require all of:

1. **Publication identity** — a masthead with issue metadata, or a running
   header repeated across pages. The masthead test is a short, visually
   dominant title in the top band *plus* an identifying signal: a domain, an
   issue number, a publication date. **A cover price is never one of them**: a
   price beside a title is a magazine cover, a flyer or a menu just as often.
2. **Editorial structure** — several headlines, each with an article under it.
   A page of headings alone is a table of contents, so what is counted is the
   headline *with a body beneath it in the same column*.
3. **Newspaper layout** — a multi-column grid, by region geometry or by word
   geometry.
4. **Reporting language** — quoted sources or a density of attribution verbs.

Point 4 is an addition to the specified formula, and it is there because
identity, structure and layout alone are *also* satisfied by a product
catalogue — a nameplate, an edition line, headings over blurbs, three columns.
What makes a publication journalistic is that it reports. `tests/test_news_publication.py`
holds the catalogue as a regression case.

Photographs, prices, short text and columns open nothing on their own.

### Blockers are per path, because an issue is a container

The family's guards — `advertisement_evidence`, `correspondence_evidence`,
`form_evidence`, `research_publication_evidence`, `press_release_evidence` —
are evaluated over the whole document. That is the right question for a single
article, which has one subject, and the wrong one for a publication. A perfectly
ordinary newspaper carries the advertisements that pay for it, a letters page
that is correspondence, a subscription coupon that is a form, and a book review
that carries citations. Applied document-wide, each of those reads as proof
that the container is something else.

Measured on a structurally valid four-page issue, before the fix:

| Inserted on page 2 | v8 before | v8 now |
| --- | --- | --- |
| An advertisement | refused, `blocked_by_advertisement_evidence` | `news_article` / `newspaper_issue` |
| A letters page | **routed to `correspondence`** | `news_article` / `newspaper_issue` |
| A subscription coupon | **routed to `form_structured`** | `news_article` / `newspaper_issue` |
| A book review with citations | **routed to `research_paper`** | `news_article` / `newspaper_issue` |

`GatePath` now carries its own `blockers`. `None` inherits the family's list —
what a path describing a whole document wants — and an explicit tuple overrides
it. The single-article paths inherit the full list unchanged; the newspaper
paths declare one guard:

| Path | Guards |
| --- | --- |
| `byline_with_reporting` | inherited: research, advertisement, form, correspondence, press release |
| `wire_and_dateline_with_reporting` | inherited: the same five |
| `newspaper_issue_masthead` | `predominantly_advertising_evidence` |
| `newspaper_issue_running_header` | `predominantly_advertising_evidence` |

`predominantly_advertising_evidence` asks a different question from
`advertisement_evidence`: not *is advertising present* — it always is — but *is
advertising the document*. It requires marketing copy at a density of at least
four terms per thousand words, and journalistic structure rebuts it: six
article clusters, or articles across two pages, or reported speech. An
advertising circular with a nameplate and an edition line is still refused;
`test_an_advertising_circular_is_still_not_a_newspaper` holds that line.

`press_release_evidence` is deliberately kept on the single-article paths and
dropped from the newspaper paths: a newspaper reprinting a press release is
still a newspaper, while a document that *is* a press release is not an article.

A satisfied path now wins over a guard that vetoed a *different* path. The
guards still fire — they are recorded, per path, under
`evidence.family_gates.news_article.path_evaluations`:

```json
{
  "byline_with_reporting": {
    "satisfied": false,
    "blockers_evaluated": ["research_publication_evidence", "advertisement_evidence", "..."],
    "blockers_inherited_from_family": true,
    "blocked_by": ["advertisement_evidence"]
  },
  "newspaper_issue_masthead": {
    "satisfied": true,
    "blockers_evaluated": ["predominantly_advertising_evidence"],
    "blockers_inherited_from_family": false,
    "blocked_by": []
  }
}
```

so a reader can see that a guard written for a single article fired and was not
applied to the publication.

### Verified against a real newspaper page

The family was checked on an actual scanned broadsheet front page (a daily's
international edition, one page, 4112 × 6566 at 300 dpi) rather than only on
fixtures. Result:

```json
{
  "document_family": "news_article",
  "document_subtype": "newspaper_issue",
  "decision": "classified",
  "score": 1.3235,
  "reason": "path_newspaper_issue_masthead"
}
```

with the gate metrics:

| Metric | Value |
| --- | --- |
| `publication_masthead_count` | 1 |
| `issue_metadata_count` | 1 |
| `publication_date_count` | 1 |
| `headline_count` | 29 |
| `article_cluster_count` | 25 |
| `estimated_column_count_by_page` | `[6]` |
| `multi_column_page_ratio` | 1.0 |
| `reporting_verb_count` | 6 |

**What this does and does not establish.** The repository's OCR and layout stages
(docTR, YOLO DocLayNet) are not installable in the development sandbox, so the
`document`, `page_sizes` and `page_words` inputs were reconstructed offline with
a stand-in OCR and a size-based region typer. The *rules* are therefore
verified against a real newspaper's structure and wording; the pipeline
end-to-end is not. Two numbers in the table are visibly limited by the stand-in
rather than by the classifier — `quotation_attribution_count` is 0 and only
about a quarter of the page's words were recovered — and the decision holds
anyway, which is the useful part: the newspaper path does not depend on a
complete transcription.

Three pattern gaps were found by that run, each a general convention rather
than a property of the page:

| Gap | Fix |
| --- | --- |
| `BY JONATHAN MARTIN` — bylines are set in **capitals** | The case of the introducer is no longer discriminating; the case of the name after it still is. |
| `Issue Number No. 42,812` — issue numbers carry a **thousands separator** | `No. 42,812` and `Edição 1.717` parse. `see page 42` still does not. |
| `NOVEMBER6,2020` — OCR of newsprint drops **inter-word spaces** | The date pattern tolerates missing spaces between month, day and year. `November 2020 was` is still not a publication date. |

The byline rule did **not** fire on that page: the byline lines did not survive
reconstruction cleanly. The issue was recognised anyway, through publication
identity, editorial structure, layout and reporting language — which is the
reason the newspaper path exists separately from the single-article paths.

### Financial guards in v8

A newspaper is full of numbers that are not accounting, and three of them were
being read as financial evidence:

| Signal | v7 | v8 |
| --- | --- | --- |
| `accounting_negative_count` | Any parenthesised number, so `(011)` and `(11)` — dialling codes in the classifieds — were negative balances. | A parenthesised number counts only with a currency symbol, inside a table, or with accounting vocabulary within a short window. Telephone patterns are excluded first; bare two-to-four digit integers (codes, footnotes, years) never count. |
| `monthly_series` | Three month names anywhere. | Month names **and** a table, accounting vocabulary, or a real density of money. Four months across four articles is not a series. |
| `currency_values` | Two currency matches anywhere. | Two matches **and** a density of at least 1.5 per thousand words. A financial document is dense in money; a newspaper prints a few advertised prices across thousands of words. |

`news_article` does **not** block `financial_document`: a real financial
statement still competes and still wins on its own evidence, which
`FinancialGuardTests` asserts directly.

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

`tests/test_news_publication.py` covers the v8 news publication family: a
Portuguese and an English newspaper issue accepted with
`document_subtype: newspaper_issue`, a running header standing in for a missing
masthead, the structure features that describe the issue, Portuguese and English
bylines on single articles, a product catalogue and a two-column journal article
refused, and the financial guards — dialling codes not counted as accounting
negatives, parenthesised money still counted, and a real financial statement
still winning its own case.

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
