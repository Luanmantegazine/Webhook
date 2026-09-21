#!/usr/bin/env python3
"""Evaluate the embedding baseline over already-extracted workflow runs.

Consumes the ``classification_features`` the benchmark workflow already stored
for the 270 validation documents. It does **not** re-run TIFF conversion, PDF
loading, layout detection or docTR: the only stage replaced is the decision.

Two perspectives are reported side by side and never mixed (see
``scripts/embedding_metrics.py``): the conventional one, where a refusal counts
as an error, and the operational one, where ``fallback``, ``other`` and
``abstained`` are valid outcomes and the only error is an accepted wrong
answer.

For comparison against the rules baseline the report also replays the stored
similarities at two fixed operating points — the rules classifier's coverage
and its selective risk — so the two systems are read at the same promise
rather than at whichever point each happens to sit.

    uv run python scripts/evaluate_embeddings_from_cache.py \
      --manifest data/rvl_cdip_subset/validation/rvl_cdip_manifest.csv \
      --executions output/executions/validation \
      --reference output/embedding_reference \
      --thresholds config/embedding_thresholds.json \
      --output output/rvl_cdip_embeddings_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embedding_corpus import (  # noqa: E402
    CorpusError,
    LeakageError,
    index_execution_features,
    load_feature_record,
    read_manifest,
    require_disjoint_splits,
    resolve_features_path,
    write_csv,
    write_json,
)
from scripts.embedding_metrics import (  # noqa: E402
    conventional_metrics,
    operating_point_at_coverage,
    operating_point_at_risk,
    risk_coverage_curve,
    selective_metrics,
    timing_metrics,
)
from tasks.document.embedding_classifier_core import (  # noqa: E402
    EMBEDDING_CLASSIFIER_VERSION,
    EMBEDDING_EXTRACTION_VERSION,
    EMBEDDING_FINGERPRINT,
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingClassifierConfig,
    EmbeddingContractError,
    EmbeddingModelConfig,
    EmbeddingReferenceIndex,
    SentenceTransformerEmbedder,
    build_result,
    classify_with_embeddings,
    classifier_versions,
    embed_document,
    insufficient_text_result,
    load_classifier_config,
    load_thresholds_file,
    validate_embedding_input,
)
from tasks.document.rvl_cdip_eval import (  # noqa: E402
    DOCUMENT_FAMILIES,
    TAXONOMY_VERSION,
    TaxonomyConfigError,
    taxonomy_descriptor,
    verify_config_file,
)

#: The rules baseline this run is compared against. Recorded, never recomputed:
#: these numbers come from the v12 benchmark report and are reported as the
#: reference operating point, not as something this script measured.
RULES_BASELINE = {
    "classifier_version": "rules-rvl-cdip-v12+0605c7dd2e78",
    "sample_count": 270,
    "coverage": 0.1481,
    "selective_accuracy": 0.95,
    "selective_risk": 0.05,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="Evaluation manifest CSV")
    parser.add_argument("--reference", required=True, help="Reference index directory")
    parser.add_argument("--output", required=True, help="Output directory for the report")
    parser.add_argument(
        "--executions",
        default="",
        help="Directory of stored workflow runs, searched recursively for features",
    )
    parser.add_argument("--config", default="", help="Embedding classifier config JSON")
    parser.add_argument("--thresholds", default="", help="Calibration artifact")
    parser.add_argument("--classification-mode", default="evaluate", choices=("evaluate", "observe"))
    parser.add_argument(
        "--reference-manifest",
        default="",
        help="Reference split manifest, checked for overlap with the evaluation split",
    )
    parser.add_argument(
        "--calibration-manifest",
        default="",
        help="Calibration split manifest, checked for overlap with the evaluation split",
    )
    parser.add_argument(
        "--baseline-coverage",
        type=float,
        default=RULES_BASELINE["coverage"],
        help="Rules coverage to replay the embedding scores at",
    )
    parser.add_argument(
        "--baseline-selective-risk",
        type=float,
        default=RULES_BASELINE["selective_risk"],
        help="Rules selective risk to replay the embedding scores at",
    )
    return parser


def _predict(rows, *, manifest_path, executions_index, index, model_config, classifier_config, embedder):
    predictions: list[dict] = []
    for row in rows:
        features_path = resolve_features_path(
            row, manifest_path=manifest_path, executions_index=executions_index
        )
        record = load_feature_record(features_path)
        descriptor = validate_embedding_input(record, source=str(features_path))
        recognized = descriptor["recognized_characters"]

        embedding_started = perf_counter()
        if recognized < classifier_config.min_recognized_characters:
            embedding_ms = (perf_counter() - embedding_started) * 1000.0
            classifier_started = perf_counter()
            result = insufficient_text_result(
                record,
                classifier_config,
                model_config=model_config,
                reference_index=index,
            )
            classifier_ms = (perf_counter() - classifier_started) * 1000.0
        else:
            vector, metadata = embed_document(
                record.get("classification_text"), embedder, model_config
            )
            embedding_ms = (perf_counter() - embedding_started) * 1000.0
            metadata["recognized_characters"] = recognized
            classifier_started = perf_counter()
            classification = classify_with_embeddings(
                vector,
                index,
                classifier_config,
                embedding_metadata=metadata,
                model_config=model_config,
            )
            result = build_result(
                classification=classification,
                embedding_metadata=metadata,
                config=classifier_config,
                model_config=model_config,
                reference_index=index,
                recognized_characters=recognized,
            )
            classifier_ms = (perf_counter() - classifier_started) * 1000.0

        predictions.append(
            {
                "sample_id": row.sample_id,
                "rvl_label": row.rvl_label,
                "target_family": row.target_family,
                "is_rejection_target": row.is_rejection_target,
                "predicted_family": result["document_family"],
                "decision": result["decision"],
                "reason": result["reason"],
                "top_candidate": result["top_candidate"],
                "runner_up": result["runner_up"],
                "top_similarity": result["top_similarity"],
                "score_margin": result["score_margin"],
                "confidence": result["confidence"],
                "confidence_kind": result["confidence_kind"],
                "candidate_scores": result["candidate_scores"],
                "nearest_prototypes": result["nearest_prototypes"],
                "recognized_characters": recognized,
                "chunk_count": result["embedding"]["chunk_count"],
                "token_count": result["embedding"]["token_count"],
                "embedding_time_ms": round(embedding_ms, 4),
                "classifier_time_ms": round(classifier_ms, 4),
                "features_path": str(features_path),
                "feature_extraction_version": descriptor["feature_extraction_version"],
                "feature_fingerprint": descriptor["feature_fingerprint"],
                "schema_version": EMBEDDING_SCHEMA_VERSION,
                "taxonomy_version": TAXONOMY_VERSION,
                "classifier_version": EMBEDDING_CLASSIFIER_VERSION,
                "embedding_fingerprint": EMBEDDING_FINGERPRINT,
                "reference_fingerprint": index.reference_fingerprint,
            }
        )
    return predictions


def _csv_row(record: dict) -> dict:
    row = {
        key: value
        for key, value in record.items()
        if key not in ("candidate_scores", "nearest_prototypes")
    }
    row["candidate_scores"] = json.dumps(record.get("candidate_scores") or {}, sort_keys=True)
    top = (record.get("nearest_prototypes") or [{}])[0]
    row["nearest_label"] = top.get("label", "")
    row["nearest_similarity"] = top.get("similarity", "")
    return row


def run(args: argparse.Namespace, embedder=None) -> tuple[int, dict]:
    verify_config_file()
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)

    splits = {"evaluation": rows}
    if args.reference_manifest:
        splits["reference"] = read_manifest(args.reference_manifest)
    if args.calibration_manifest:
        splits["calibration"] = read_manifest(args.calibration_manifest)
    integrity = require_disjoint_splits(splits)

    index = EmbeddingReferenceIndex.load(args.reference)
    model_config, classifier_config = load_classifier_config(args.config or None)
    if args.thresholds:
        classifier_config = load_thresholds_file(args.thresholds, classifier_config)
    classifier_config = replace(classifier_config, mode=args.classification_mode)
    model_config = replace(
        model_config,
        model_name=str(index.metadata.get("model", model_config.model_name)),
        model_revision=index.metadata.get("revision") or model_config.model_revision,
        text_prefix=str(index.metadata.get("text_prefix", model_config.text_prefix)),
        max_tokens=int(index.metadata.get("max_tokens", model_config.max_tokens)),
        overlap_tokens=int(index.metadata.get("overlap_tokens", model_config.overlap_tokens)),
    )
    index.ensure_compatible(model_config)

    # A document that helped build a centroid cannot also be evaluated against
    # it. This catches the case the manifests alone cannot: the index carries
    # the sample ids it was built from.
    reference_ids = {
        sample
        for samples in (index.metadata.get("sample_ids") or {}).values()
        for sample in samples
    }
    shared = sorted(reference_ids & {row.sample_id for row in rows})
    if shared:
        raise LeakageError(
            f"{len(shared)} evaluated sample(s) are reference documents of this index "
            f"({', '.join(shared[:5])}...); their similarity to their own centroid is "
            "not a measurement."
        )

    executions_index = index_execution_features(args.executions) if args.executions else {}
    encoder = embedder if embedder is not None else SentenceTransformerEmbedder(model_config).load()

    predictions = _predict(
        rows,
        manifest_path=manifest_path,
        executions_index=executions_index,
        index=index,
        model_config=model_config,
        classifier_config=classifier_config,
        embedder=encoder,
    )

    conventional = conventional_metrics(predictions, list(DOCUMENT_FAMILIES))
    operational = selective_metrics(predictions)
    timings = timing_metrics(predictions)
    curve = risk_coverage_curve(predictions, margin=classifier_config.min_score_margin)
    at_baseline_coverage = operating_point_at_coverage(curve, args.baseline_coverage)
    at_baseline_risk = operating_point_at_risk(curve, args.baseline_selective_risk)

    per_family = {}
    for family in DOCUMENT_FAMILIES:
        family_records = [row for row in predictions if row["target_family"] == family]
        if not family_records:
            continue
        per_family[family] = selective_metrics(family_records)

    feature_versions = Counter(
        str(row.get("feature_extraction_version")) for row in predictions
    )
    report = {
        "versions": {
            **classifier_versions(index, model_config, classifier_config),
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "embedding_extraction_version": EMBEDDING_EXTRACTION_VERSION,
            "observed_feature_extraction_versions": dict(sorted(feature_versions.items())),
            "reference_index": {
                "labels": list(index.labels),
                "example_counts": dict(zip(index.labels, index.example_counts)),
                "dimension": index.dimension,
                "manifest_hash": index.metadata.get("manifest_hash"),
                "generated_at": index.metadata.get("generated_at"),
            },
        },
        "manifest": {
            "path": str(manifest_path),
            "sample_count": len(rows),
            "label_counts": dict(sorted(Counter(row.rvl_label for row in rows).items())),
        },
        "split_integrity": integrity,
        "metrics": {
            "conventional": conventional,
            "operational": operational,
            "per_family_operational": per_family,
            "timing_ms": timings,
        },
        "comparison_with_rules_baseline": {
            "rules_baseline": RULES_BASELINE,
            "note": (
                "The rules figures are the recorded v12 benchmark result, quoted "
                "here; this script does not rerun the rules classifier and does not "
                "modify any of its artifacts."
            ),
            "embeddings_at_calibrated_threshold": {
                "similarity_threshold": classifier_config.similarity_threshold,
                "minimum_score_margin": classifier_config.min_score_margin,
                "calibration_status": classifier_config.calibration_status,
                "coverage": operational["coverage"],
                "selective_accuracy": operational["selective_accuracy"],
                "selective_risk": operational["selective_risk"],
                "accepted_wrong_count": operational["accepted_wrong_count"],
            },
            "embeddings_at_rules_coverage": at_baseline_coverage,
            "embeddings_at_rules_selective_risk": at_baseline_risk,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxonomy": taxonomy_descriptor(),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", [_csv_row(row) for row in predictions])
    write_json(output / "report.json", report)
    write_json(output / "confusion_matrix.json", conventional["confusion_matrix"])
    write_csv(
        output / "confusion_matrix.csv",
        [
            {"target_family": target, **{f"predicted_{name}": count for name, count in row.items()}}
            for target, row in sorted(conventional["confusion_matrix"].items())
        ],
    )
    write_csv(output / "risk_coverage_curve.csv", curve)
    write_json(output / "risk_coverage_curve.json", curve)
    artifacts = {
        "predictions": str(output / "predictions.csv"),
        "report": str(output / "report.json"),
        "confusion_matrix_json": str(output / "confusion_matrix.json"),
        "confusion_matrix_csv": str(output / "confusion_matrix.csv"),
        "risk_coverage_csv": str(output / "risk_coverage_curve.csv"),
        "risk_coverage_json": str(output / "risk_coverage_curve.json"),
    }
    write_json(output / "artifacts.json", artifacts)
    return 0, {"artifacts": artifacts, "report": report}


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        status, payload = run(args)
    except LeakageError as error:
        print(json.dumps({"error": "leakage", "detail": str(error)}, indent=2))
        return 2
    except (CorpusError, EmbeddingContractError, TaxonomyConfigError) as error:
        print(json.dumps({"error": type(error).__name__, "detail": str(error)}, indent=2))
        return 1
    print(json.dumps(payload["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
