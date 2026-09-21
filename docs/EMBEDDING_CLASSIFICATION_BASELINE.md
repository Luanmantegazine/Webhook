# Embedding baseline (v1)

The rules classifier decides with hand-written evidence. This decides with
distance to a labelled centroid, over **exactly the same inputs**: the
`classification_features` the existing workflow already produces. Conversion,
layout detection, OCR, aggregation and the evaluation taxonomy are untouched;
only the decision mechanism is replaced.

It is a baseline, deliberately. Not in this increment: fine-tuning, image or
multimodal encoders, FAISS or any vector database, rerankers, k-NN,
combination with the rules classifier, and automatic template selection.

## Architecture

```
aggregate_document_content
  └─ extract_document_classification_features   (unchanged, shared)
       ├─ classify_document_rules               (v12 — untouched baseline)
       └─ extract_document_embedding            (new)
            └─ classify_document_embeddings     (new)
```

| Layer | File | Responsibility |
| --- | --- | --- |
| Core | `tasks/document/embedding_classifier_core.py` | Chunking, pooling, index, similarity, decision policy, versions, fingerprints. Pure; imports numpy only. |
| Task | `tasks/document/extract_document_embedding.py` | Text → one pooled vector, with the model identity that produced it. |
| Task | `tasks/document/classify_document_embeddings.py` | Vector → family, decision, nearest prototypes. |
| Script | `scripts/build_embedding_reference.py` | Reference split → label centroids. |
| Script | `scripts/calibrate_embedding_thresholds.py` | Development split → thresholds. |
| Script | `scripts/evaluate_embeddings_from_cache.py` | Cached runs → predictions, report, confusion matrix, risk/coverage. |
| Shared | `scripts/embedding_corpus.py` | Manifests, feature lookup, split-integrity checks. |
| Shared | `scripts/embedding_metrics.py` | The two metric perspectives. |
| Config | `config/document_embedding_classifier.json` | Model and decision configuration. No taxonomy. |
| Workflow | `workflows/document_classification_embeddings_benchmark.json` | Same pipeline as the rules benchmark, different decision stage. |

The taxonomy is **not** redeclared anywhere: `tasks/document/rvl_cdip_eval.py`
remains the single source, and `CANDIDATE_LABELS`, `FAMILY_TO_LABELS` and
`CANDIDATE_FAMILIES` are derived from it at import.

## Model

`intfloat/multilingual-e5-base` via `sentence-transformers`
(<https://huggingface.co/intfloat/multilingual-e5-base>), 768 dimensions.
Name and revision are configurable and recorded; the device is probed in the
order `cuda`, `mps`, `cpu`.

The task is **symmetric** — a document is compared against reference documents,
not against a query — so the same prefix, `query: ` by default, is applied to
both sides. Using `query:` on one side and `passage:` on the other would put
the two in different regions of the space and make the cosine meaningless.

Unit tests never load the model: they inject a deterministic fake embedder.
Only `tests/test_embedding_real_model.py` touches real weights, and it skips
unless `HYDRA_EMBEDDING_REAL_MODEL=1`.

## Input contract

```json
{
  "schema_version": "2.1",
  "taxonomy_version": "rvl-cdip-2.1",
  "classification_text": "...",
  "alnum_character_count": 100,
  "provenance": {}
}
```

Refused, not degraded: a foreign `schema_version`, a foreign
`taxonomy_version`, a missing `classification_text`, a missing `provenance`.
Below `min_recognized_characters` (20) the answer is the abstention contract:

```json
{"document_family": "other", "decision": "abstained", "reason": "insufficient_ocr_text"}
```

No OCR is run and no text is reconstructed from images inside the classifier.

## Chunking and pooling

`chunk_classification_text(text, tokenizer, max_tokens=384, overlap_tokens=64)`
is pure and deterministic. Only runs of spaces and blank lines are normalised —
titles, tables and punctuation are left alone, because they carry most of what
separates an invoice from a memo.

The document is never silently truncated at the first window: the window
advances by `max_tokens - overlap_tokens`, so every token appears in at least
one chunk and consecutive chunks share exactly `overlap_tokens`. Empty chunks
are dropped; `overlap_tokens >= max_tokens` is an error, not a clamp.

Each chunk is embedded, **normalised**, averaged, and the mean is normalised
again (`normalized_chunk_mean`). Normalising before the mean is what stops one
long chunk from deciding the document by magnitude rather than direction.

Reported metadata: `chunk_count`, `token_count`, `embedding_dimension`,
`pooling`. The vector itself is never in a workflow's public output.

## Reference index contract

Centroids are built per **RVL-CDIP label**, not per family, and a family scores
the **maximum** over its labels — averaging `email`, `letter` and `memo` into
one `correspondence` prototype dilutes a heterogeneous family into never
winning anything.

`file folder` and `handwritten` get **no centroid**. They are rejection
targets: negatives during calibration, and reachable only through
`fallback`/`other`. Building an index that names one is an error.

Persistence:

| File | Contents |
| --- | --- |
| `embedding_reference.npz` | `centroids` (float32) and `example_counts` (int32) — numeric arrays only, loaded with `allow_pickle=False` |
| `embedding_reference.metadata.json` | model, revision, dimension, prefix, pooling, chunk size and overlap, schema, taxonomy, embedding extraction version, manifest hash, sample ids, labels, per-label counts, `reference_fingerprint`, generation timestamp |

No vector is ever written to JSON, and nothing is pickled. The index is refused
on load or on use when the model, revision, taxonomy, extraction version,
prefix, chunking, dimension or fingerprint disagrees with the classifier.

## Decision contract

1. cosine similarity against every centroid;
2. aggregate to families by maximum;
3. rank deterministically (score descending, then family name);
4. take top-1 and top-2, compute `score_margin`;
5. apply the threshold, then the margin.

| Situation | Outcome |
| --- | --- |
| text below the character floor | `abstained` / `other` |
| `top_similarity` below threshold | `fallback` / `other` |
| `score_margin` below minimum | `fallback` / `other` |
| otherwise | `classified` / top candidate |

No class is ever forced. Thresholds and margins are global with optional
per-family overrides.

`confidence` is the cosine clipped into `[0, 1]` and is reported next to
`confidence_kind: "cosine_similarity_not_probability"`; `top_similarity` keeps
the raw value. Nothing here calibrates a probability.

## Calibration

Objective: **maximise coverage subject to `selective_accuracy >= 0.95`** — the
rules classifier's operating promise, so the two are compared at the same bar.
The grid is a fixed arithmetic sweep evaluated in a fixed order, so two runs on
one split agree exactly.

Refusals: a calibration split sharing a `sample_id` or an `image_path` with the
reference or evaluation split; a calibration split containing documents the
index was built from. A family with fewer than `--minimum-family-support`
development documents gets **no** threshold of its own and is listed under
`insufficient_support`; it keeps the global value rather than an invented one.
If no grid point reaches the target, the artifact is written with
`calibration_status: "uncalibrated"` rather than a silently fitted number.

## Evaluation

Consumes the stored `classification_features` of the 270 validation documents,
found either through a manifest `classification_features_path` column or by
searching `--executions` recursively. It never re-runs TIFF conversion, PDF
loading, layout detection or docTR.

Artifacts: `predictions.csv`, `report.json`, `confusion_matrix.{json,csv}`,
`risk_coverage_curve.{json,csv}`, `artifacts.json`.

Two perspectives, reported side by side and never mixed:

* **Conventional** — accuracy, balanced accuracy, macro precision/recall/F1,
  weighted F1, confusion matrix, per-family figures. A refusal counts as a
  prediction of `other`, i.e. as an error.
* **Operational** — coverage, selective accuracy, selective risk, accepted
  wrong answers, controlled rejection rate, fallback and abstention counts,
  embedding time, classifier time, throughput. `fallback`, `other` and
  `abstained` are valid outcomes; **the only operational error is an accepted
  classification that is wrong**, including any accepted rejection target.

For comparison the report replays the stored similarities at three points: the
calibrated threshold, the rules classifier's **coverage (14.81%)**, and the
rules classifier's **selective risk (5%)**. The v12 figures are quoted from its
recorded benchmark; this script never runs the rules classifier and never
writes to its artifacts.

## Versions and fingerprints

| Name | Value / meaning |
| --- | --- |
| `EMBEDDING_SCHEMA_VERSION` | `2.1` — the record contract, shared with the rules classifier so one evaluator reads both |
| `EMBEDDING_EXTRACTION_VERSION` | `1.0` — normalisation, chunking, pooling |
| `EMBEDDING_CLASSIFIER_VERSION` | `embeddings-rvl-cdip-v1+<embedding_fingerprint>` |
| `embedding_fingerprint` | source of the chunking, pooling, similarity, aggregation and decision functions, plus model, revision, prefix, chunking, taxonomy and candidate set |
| `reference_fingerprint` | the built index: labels, counts, centroid bytes (float32), model identity, manifest hash |

Both are derived from source text and configuration values, never from `repr`
of a code object, so they are identical across processes. A rules fingerprint
and an embedding fingerprint describe different machinery and must never be
compared or pooled.

## Commands

```bash
# 1. Reference centroids (train split; disjoint from calibration and validation)
uv run python scripts/build_embedding_reference.py \
  --manifest data/rvl_cdip_subset/train/reference_manifest.csv \
  --executions output/executions/train \
  --output output/embedding_reference \
  --disjoint-from data/rvl_cdip_subset/validation/rvl_cdip_manifest.csv

# 2. Thresholds (development split only; never the 270 evaluated documents)
uv run python scripts/calibrate_embedding_thresholds.py \
  --manifest data/rvl_cdip_subset/train/calibration_manifest.csv \
  --executions output/executions/train \
  --reference output/embedding_reference \
  --output config/embedding_thresholds.json \
  --disjoint-from data/rvl_cdip_subset/validation/rvl_cdip_manifest.csv

# 3. The 270-document benchmark, over the cached runs
uv run python scripts/evaluate_embeddings_from_cache.py \
  --manifest data/rvl_cdip_subset/validation/rvl_cdip_manifest.csv \
  --executions output/executions/validation \
  --reference output/embedding_reference \
  --thresholds config/embedding_thresholds.json \
  --reference-manifest data/rvl_cdip_subset/train/reference_manifest.csv \
  --calibration-manifest data/rvl_cdip_subset/train/calibration_manifest.csv \
  --output output/rvl_cdip_embeddings_v1

# 4. Tests (offline; no model, no network)
uv run python -m unittest discover -s tests -p "test_embedding*.py" -v

# 5. Optional, real weights
HYDRA_EMBEDDING_REAL_MODEL=1 uv run python -m unittest tests.test_embedding_real_model -v
```

Recommended reference split: 20 documents per RVL-CDIP label across the 16
labels (~320 documents), with calibration and evaluation disjoint from it and
from each other.

Extra dependencies live in `requirements-embeddings.txt`; the rules classifier
and the whole unit suite run without them.

## Known limitations

- **No benchmark numbers yet.** `sentence-transformers` is not installed in
  this environment and no reference split has been embedded, so no coverage or
  selective accuracy is reported here. The machinery is tested end to end
  against a deterministic fake encoder; the real comparison against v12 needs a
  run with weights.
- **Uncalibrated by default.** The shipped thresholds are 0.0 and say so.
- **`business_report` has a centroid** (from `budget`) but is excluded from the
  rules evaluator's `SCORED_FAMILIES`; the embedding report scores it as its
  own family, which is the taxonomy's mapping for that label.
- **Text only.** A page whose evidence is visual — a slide, a form grid, a
  newspaper's column geometry — reaches this classifier as OCR text alone.
- **One vector per document.** A long document is pooled into a single point,
  so a newspaper issue is a mean of many articles.
- **Cosine similarities are not comparable across models.** A new model or a
  changed prefix invalidates every stored threshold; the fingerprints make that
  refusable rather than silent.
