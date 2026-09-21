#!/usr/bin/env python3
"""Build the label centroids the embedding classifier compares against.

Reads a *reference* manifest — a split that is neither calibrated on nor
evaluated on — embeds each document's stored ``classification_features`` text,
averages per RVL-CDIP label, and writes ``embedding_reference.npz`` plus its
metadata.

``file folder`` and ``handwritten`` rows are read and counted, but they get no
centroid: they are negatives for calibration, and a rejection label with a
prototype would turn "I decline this" into a positive prediction.

    uv run python scripts/build_embedding_reference.py \
      --manifest data/rvl_cdip_subset/train/reference_manifest.csv \
      --output output/embedding_reference
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    ManifestRow,
    index_execution_features,
    load_feature_record,
    read_manifest,
    require_disjoint_splits,
    resolve_features_path,
    write_json,
)
from tasks.document.embedding_classifier_core import (  # noqa: E402
    CANDIDATE_LABELS,
    EMBEDDING_CLASSIFIER_VERSION,
    EMBEDDING_FINGERPRINT,
    EmbeddingContractError,
    EmbeddingModelConfig,
    SentenceTransformerEmbedder,
    build_reference_index,
    embed_document,
    load_classifier_config,
    manifest_hash,
    validate_embedding_input,
)
from tasks.document.rvl_cdip_eval import REJECTION_LABELS  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="Reference split manifest CSV")
    parser.add_argument("--output", required=True, help="Directory for the index artifacts")
    parser.add_argument(
        "--executions",
        default="",
        help="Directory of stored workflow runs, searched for classification_features",
    )
    parser.add_argument("--config", default="", help="Embedding classifier config JSON")
    parser.add_argument("--model-name", default="", help="Override the configured model")
    parser.add_argument("--model-revision", default="", help="Pin the model revision")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--overlap-tokens", type=int, default=-1)
    parser.add_argument(
        "--minimum-examples",
        type=int,
        default=5,
        help="Labels with fewer reference documents get no centroid and are reported",
    )
    parser.add_argument(
        "--disjoint-from",
        action="append",
        default=[],
        help="Manifest that must not share samples with the reference split; repeatable",
    )
    return parser


def _resolve_model_config(args: argparse.Namespace) -> EmbeddingModelConfig:
    model_config, _classifier_config = load_classifier_config(args.config or None)
    overrides: dict = {}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.model_revision:
        overrides["model_revision"] = args.model_revision
    if args.max_tokens:
        overrides["max_tokens"] = int(args.max_tokens)
    if args.overlap_tokens >= 0:
        overrides["overlap_tokens"] = int(args.overlap_tokens)
    if overrides:
        from dataclasses import replace

        model_config = replace(model_config, **overrides)
    return model_config


def run(args: argparse.Namespace, embedder=None) -> tuple[int, dict]:
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    splits = {"reference": rows}
    for other in args.disjoint_from:
        splits[Path(other).name] = read_manifest(other)
    integrity = require_disjoint_splits(splits)

    executions_index = (
        index_execution_features(args.executions) if args.executions else {}
    )
    model_config = _resolve_model_config(args)
    encoder = embedder if embedder is not None else SentenceTransformerEmbedder(model_config).load()

    embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    sample_ids: dict[str, list[str]] = defaultdict(list)
    skipped: list[dict] = []
    rejection_rows: list[str] = []
    started = perf_counter()

    for row in rows:
        if row.rvl_label in REJECTION_LABELS:
            # Read, counted, and deliberately given no centroid.
            rejection_rows.append(row.sample_id)
            continue
        if row.rvl_label not in CANDIDATE_LABELS:
            raise CorpusError(
                f"{row.sample_id}: label {row.rvl_label!r} has no centroid slot"
            )
        features_path = resolve_features_path(
            row, manifest_path=manifest_path, executions_index=executions_index
        )
        record = load_feature_record(features_path)
        try:
            descriptor = validate_embedding_input(record, source=str(features_path))
            vector, _metadata = embed_document(
                record.get("classification_text"), encoder, model_config
            )
        except EmbeddingContractError as error:
            skipped.append({"sample_id": row.sample_id, "reason": str(error)})
            continue
        if descriptor["recognized_characters"] <= 0:
            skipped.append({"sample_id": row.sample_id, "reason": "no recognised text"})
            continue
        embeddings[row.rvl_label].append(vector)
        sample_ids[row.rvl_label].append(row.sample_id)

    matrices = {
        label: np.vstack(vectors) for label, vectors in sorted(embeddings.items())
    }
    index = build_reference_index(
        matrices,
        model_config=model_config,
        sample_ids_by_label=sample_ids,
        manifest_hash=manifest_hash(manifest_path),
        manifest_path=str(manifest_path),
        minimum_examples=int(args.minimum_examples),
    )
    index.metadata["rejection_label_sample_ids"] = sorted(rejection_rows)
    index.metadata["skipped_samples"] = skipped
    index.metadata["split_integrity"] = integrity
    index.metadata["build_seconds"] = round(perf_counter() - started, 3)
    # The fingerprint covers the vectors and the model identity, not the
    # bookkeeping above, so it is recomputed after the bookkeeping is attached
    # only if the covered fields changed — they did not.
    paths = index.save(args.output)
    summary = {
        "classifier_version": EMBEDDING_CLASSIFIER_VERSION,
        "embedding_fingerprint": EMBEDDING_FINGERPRINT,
        "reference_fingerprint": index.reference_fingerprint,
        "arrays": str(paths["arrays"]),
        "metadata": str(paths["metadata"]),
        "labels": list(index.labels),
        "example_counts": dict(zip(index.labels, index.example_counts)),
        "skipped_labels": index.metadata.get("skipped_labels", {}),
        "skipped_samples": skipped,
        "rejection_label_sample_count": len(rejection_rows),
        "dimension": index.dimension,
        "split_integrity": integrity,
    }
    write_json(Path(args.output) / "build_summary.json", summary)
    return 0, summary


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        status, summary = run(args)
    except LeakageError as error:
        print(json.dumps({"error": "leakage", "detail": str(error)}, indent=2))
        return 2
    except (CorpusError, EmbeddingContractError) as error:
        print(json.dumps({"error": type(error).__name__, "detail": str(error)}, indent=2))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
