# Hydra Rules Classifier Prototype

Rules-only document-family classification for scanned RVL-CDIP documents,
integrated with Hydra's existing OCR/layout aggregation pipeline.

Start with `docs/RULES_CLASSIFICATION_PROTOTYPE.md` for installation,
contracts, benchmark preparation, execution, and evaluation guidance.

## Layout

| Path | Contents |
| --- | --- |
| `tasks/document/rvl_cdip_eval.py` | Single source of truth for the taxonomy: family list, RVL-CDIP label mapping, rejection targets, and the open mapping decisions. |
| `tasks/document/rules_classifier_core.py` | The classifier: feature extraction, rules, scoring, family gates, decision policy. No FabricFlow dependency, so it is unit-testable on its own. |
| `tasks/document/` | FabricFlow `@task` wrappers around the core. |
| `tasks/dataset/` | Scan-to-PDF adapter for the RVL-CDIP entry point. |
| `workflows/` | Benchmark workflow, and the translation workflow with classification as a parallel observer. |
| `config/rvl_cdip_taxonomy.json` | Declarative copy of the taxonomy. Verified against `tasks/document/rvl_cdip_eval.py` on every evaluation run. |
| `scripts/` | RVL-CDIP subset preparation and offline evaluation from cached OCR output. |
| `tests/` | Unit tests. |

## Tests

```bash
python -m unittest discover -s tests -v
```

The core module and the test suite depend only on the standard library. The
task wrappers additionally require FabricFlow's `core.task`, `scripts/` requires
`datasets`, and the scan adapter requires `Pillow` — none of which are needed to
run the tests.
