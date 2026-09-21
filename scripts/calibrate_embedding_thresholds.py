#!/usr/bin/env python3
"""Calibrate the similarity threshold and margin on a development split.

Objective, by default: **maximise coverage subject to selective accuracy
>= 0.95** — the rules classifier's operating contract, so the two are compared
under the same promise.

Two things this script refuses to do:

* calibrate on the evaluation corpus. Every manifest it is given is checked
  against every other for shared ``sample_id`` or ``image_path``, and an
  overlap stops the run. A threshold chosen on the documents it is later
  measured on reports its own training accuracy;
* invent a per-family threshold for a family with too few development
  examples. Such a family is listed under ``insufficient_support`` and keeps
  the global value.

The grid is deterministic: a fixed arithmetic sweep, evaluated in a fixed
order, so two runs on the same split produce the same thresholds.

    uv run python scripts/calibrate_embedding_thresholds.py \
      --manifest data/rvl_cdip_subset/train/calibration_manifest.csv \
      --reference output/embedding_reference \
      --output config/embedding_thresholds.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

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
    write_json,
)
from scripts.embedding_metrics import selective_metrics  # noqa: E402
from tasks.document.embedding_classifier_core import (  # noqa: E402
    CANDIDATE_FAMILIES,
    EMBEDDING_CLASSIFIER_VERSION,
    EMBEDDING_FINGERPRINT,
    EmbeddingClassifierConfig,
    EmbeddingContractError,
    EmbeddingModelConfig,
    EmbeddingReferenceIndex,
    SentenceTransformerEmbedder,
    classify_with_embeddings,
    embed_document,
    load_classifier_config,
    manifest_hash,
    validate_embedding_input,
)

DEFAULT_TARGET_ACCURACY = 0.95
DEFAULT_MINIMUM_FAMILY_SUPPORT = 10


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="Development split manifest CSV")
    parser.add_argument("--reference", required=True, help="Reference index directory")
    parser.add_argument("--output", required=True, help="Where to write the thresholds JSON")
    parser.add_argument("--executions", default="", help="Directory of stored workflow runs")
    parser.add_argument("--config", default="", help="Embedding classifier config JSON")
    parser.add_argument(
        "--target-selective-accuracy",
        type=float,
        default=DEFAULT_TARGET_ACCURACY,
        help="Accuracy the accepted decisions must hold (default 0.95)",
    )
    parser.add_argument("--similarity-grid-start", type=float, default=0.60)
    parser.add_argument("--similarity-grid-stop", type=float, default=0.95)
    parser.add_argument("--similarity-grid-step", type=float, default=0.005)
    parser.add_argument("--margin-grid-start", type=float, default=0.0)
    parser.add_argument("--margin-grid-stop", type=float, default=0.10)
    parser.add_argument("--margin-grid-step", type=float, default=0.005)
    parser.add_argument(
        "--minimum-family-support",
        type=int,
        default=DEFAULT_MINIMUM_FAMILY_SUPPORT,
        help="Below this many development documents a family gets no own threshold",
    )
    parser.add_argument(
        "--disjoint-from",
        action="append",
        default=[],
        help="Manifest that must not share samples with the calibration split; repeatable",
    )
    return parser


def arithmetic_grid(start: float, stop: float, step: float) -> list[float]:
    """A deterministic sweep. No randomness, no adaptive search, no shuffling."""
    if step <= 0:
        raise ValueError("grid step must be positive")
    count = int(round((stop - start) / step)) + 1
    return [round(start + index * step, 6) for index in range(max(0, count))]


def score_development_set(rows, *, manifest_path, executions_index, index, model_config, embedder):
    """Embed and score every development document once; thresholds sweep after."""
    scored = []
    for row in rows:
        features_path = resolve_features_path(
            row, manifest_path=manifest_path, executions_index=executions_index
        )
        record = load_feature_record(features_path)
        descriptor = validate_embedding_input(record, source=str(features_path))
        recognized = descriptor["recognized_characters"]
        entry = {
            "sample_id": row.sample_id,
            "rvl_label": row.rvl_label,
            "target_family": row.target_family,
            "is_rejection_target": row.is_rejection_target,
            "recognized_characters": recognized,
        }
        if recognized < DEFAULT_MINIMUM_CHARACTERS:
            entry.update(
                {
                    "decision": "abstained",
                    "predicted_family": "other",
                    "top_candidate": None,
                    "top_similarity": 0.0,
                    "score_margin": 0.0,
                }
            )
            scored.append(entry)
            continue
        vector, metadata = embed_document(
            record.get("classification_text"), embedder, model_config
        )
        classification = classify_with_embeddings(
            vector, index, EmbeddingClassifierConfig(), embedding_metadata={
                "recognized_characters": recognized
            }
        )
        entry.update(
            {
                "decision": "classified",
                "predicted_family": classification["top_candidate"],
                "top_candidate": classification["top_candidate"],
                "top_similarity": classification["top_similarity"],
                "score_margin": classification["score_margin"],
                "candidate_scores": classification["candidate_scores"],
            }
        )
        scored.append(entry)
    return scored


DEFAULT_MINIMUM_CHARACTERS = EmbeddingClassifierConfig().min_recognized_characters


def _apply(entry: dict, threshold: float, margin: float) -> dict:
    replayed = dict(entry)
    if entry["decision"] == "abstained":
        return replayed
    if entry["top_similarity"] >= threshold and entry["score_margin"] >= margin:
        replayed["decision"] = "classified"
        replayed["predicted_family"] = entry["top_candidate"]
    else:
        replayed["decision"] = "fallback"
        replayed["predicted_family"] = "other"
    return replayed


def search_global_operating_point(
    scored: list[dict],
    *,
    similarity_grid: list[float],
    margin_grid: list[float],
    target_accuracy: float,
) -> dict:
    """Maximise coverage subject to the accuracy floor; ties broken determinedly."""
    best: dict | None = None
    evaluated = []
    for threshold in similarity_grid:
        for margin in margin_grid:
            replayed = [_apply(entry, threshold, margin) for entry in scored]
            metrics = selective_metrics(replayed)
            point = {
                "similarity_threshold": threshold,
                "minimum_score_margin": margin,
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
                "selective_risk": metrics["selective_risk"],
                "accepted_count": metrics["accepted_count"],
                "accepted_wrong_count": metrics["accepted_wrong_count"],
                "meets_target": bool(
                    metrics["accepted_count"] > 0
                    and metrics["selective_accuracy"] >= float(target_accuracy)
                ),
            }
            evaluated.append(point)
            if not point["meets_target"]:
                continue
            key = (point["coverage"], point["selective_accuracy"], -threshold, -margin)
            if best is None or key > (
                best["coverage"],
                best["selective_accuracy"],
                -best["similarity_threshold"],
                -best["minimum_score_margin"],
            ):
                best = point
    return {"best": best, "evaluated": evaluated}


def search_family_thresholds(
    scored: list[dict],
    *,
    similarity_grid: list[float],
    global_point: dict,
    target_accuracy: float,
    minimum_support: int,
) -> tuple[dict[str, float], dict[str, dict]]:
    """A per-family threshold only where the split actually supports one."""
    support = Counter(
        entry["target_family"] for entry in scored if not entry["is_rejection_target"]
    )
    family_thresholds: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    for family in CANDIDATE_FAMILIES:
        candidates = [entry for entry in scored if entry["top_candidate"] == family]
        family_support = support.get(family, 0)
        if family_support < int(minimum_support) or not candidates:
            diagnostics[family] = {
                "status": "insufficient_support",
                "development_support": family_support,
                "minimum_support": int(minimum_support),
                "threshold_used": global_point["similarity_threshold"],
                "note": (
                    "Too few development documents to choose a threshold for this "
                    "family; it keeps the global value rather than receiving an "
                    "invented one."
                ),
            }
            continue
        best = None
        for threshold in similarity_grid:
            replayed = [
                _apply(entry, threshold if entry["top_candidate"] == family else global_point["similarity_threshold"], global_point["minimum_score_margin"])
                for entry in scored
            ]
            family_records = [
                record
                for record in replayed
                if record["top_candidate"] == family
            ]
            metrics = selective_metrics(family_records)
            if metrics["accepted_count"] == 0:
                continue
            if metrics["selective_accuracy"] < float(target_accuracy):
                continue
            key = (metrics["coverage"], metrics["selective_accuracy"], -threshold)
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        if best is None:
            diagnostics[family] = {
                "status": "no_threshold_meets_target",
                "development_support": family_support,
                "threshold_used": global_point["similarity_threshold"],
            }
            continue
        _key, threshold, metrics = best
        family_thresholds[family] = threshold
        diagnostics[family] = {
            "status": "calibrated",
            "development_support": family_support,
            "threshold": threshold,
            "coverage": metrics["coverage"],
            "selective_accuracy": metrics["selective_accuracy"],
        }
    return family_thresholds, diagnostics


def run(args: argparse.Namespace, embedder=None) -> tuple[int, dict]:
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    splits = {"calibration": rows}
    for other in args.disjoint_from:
        splits[Path(other).name] = read_manifest(other)
    integrity = require_disjoint_splits(splits)

    index = EmbeddingReferenceIndex.load(args.reference)
    model_config, base_config = load_classifier_config(args.config or None)
    model_config = replace(
        model_config,
        model_name=str(index.metadata.get("model", model_config.model_name)),
        model_revision=index.metadata.get("revision") or model_config.model_revision,
        text_prefix=str(index.metadata.get("text_prefix", model_config.text_prefix)),
        max_tokens=int(index.metadata.get("max_tokens", model_config.max_tokens)),
        overlap_tokens=int(index.metadata.get("overlap_tokens", model_config.overlap_tokens)),
    )
    index.ensure_compatible(model_config)
    # The reference documents must not be in the calibration split either: a
    # document sitting inside its own centroid is trivially similar to it.
    reference_ids = {
        sample
        for samples in (index.metadata.get("sample_ids") or {}).values()
        for sample in samples
    }
    shared_with_reference = sorted(reference_ids & {row.sample_id for row in rows})
    if shared_with_reference:
        raise LeakageError(
            f"{len(shared_with_reference)} calibration sample(s) are also reference "
            f"documents in the index ({', '.join(shared_with_reference[:5])}...); a "
            "document compared against a centroid built from itself is not evidence."
        )

    encoder = embedder if embedder is not None else SentenceTransformerEmbedder(model_config).load()
    executions_index = index_execution_features(args.executions) if args.executions else {}
    scored = score_development_set(
        rows,
        manifest_path=manifest_path,
        executions_index=executions_index,
        index=index,
        model_config=model_config,
        embedder=encoder,
    )

    similarity_grid = arithmetic_grid(
        args.similarity_grid_start, args.similarity_grid_stop, args.similarity_grid_step
    )
    margin_grid = arithmetic_grid(
        args.margin_grid_start, args.margin_grid_stop, args.margin_grid_step
    )
    search = search_global_operating_point(
        scored,
        similarity_grid=similarity_grid,
        margin_grid=margin_grid,
        target_accuracy=args.target_selective_accuracy,
    )
    calibrated = search["best"] is not None
    global_point = search["best"] or {
        "similarity_threshold": base_config.similarity_threshold,
        "minimum_score_margin": base_config.min_score_margin,
        "coverage": 0.0,
        "selective_accuracy": 0.0,
        "selective_risk": 0.0,
        "accepted_count": 0,
        "accepted_wrong_count": 0,
    }
    family_thresholds, family_diagnostics = (
        search_family_thresholds(
            scored,
            similarity_grid=similarity_grid,
            global_point=global_point,
            target_accuracy=args.target_selective_accuracy,
            minimum_support=args.minimum_family_support,
        )
        if calibrated
        else ({}, {})
    )

    final = [
        _apply(
            entry,
            family_thresholds.get(entry["top_candidate"], global_point["similarity_threshold"]),
            global_point["minimum_score_margin"],
        )
        for entry in scored
    ]
    development_metrics = selective_metrics(final)

    payload = {
        "thresholds": {
            "similarity": global_point["similarity_threshold"],
            "minimum_score_margin": global_point["minimum_score_margin"],
            "minimum_recognized_characters": base_config.min_recognized_characters,
            "family_similarity_thresholds": family_thresholds,
            "family_minimum_score_margins": {},
            "calibration_status": "calibrated" if calibrated else "uncalibrated",
            "calibration_note": (
                f"Maximised coverage subject to selective_accuracy >= "
                f"{args.target_selective_accuracy} on {manifest_path.name} "
                f"({len(rows)} documents)."
                if calibrated
                else (
                    "No grid point reached the accuracy target on this split; the "
                    "configured defaults are kept and the status stays uncalibrated."
                )
            ),
        },
        "objective": {
            "maximise": "coverage",
            "subject_to": f"selective_accuracy >= {args.target_selective_accuracy}",
            "grid": {
                "similarity": [
                    args.similarity_grid_start,
                    args.similarity_grid_stop,
                    args.similarity_grid_step,
                ],
                "margin": [
                    args.margin_grid_start,
                    args.margin_grid_stop,
                    args.margin_grid_step,
                ],
                "deterministic": True,
            },
        },
        "development_split": {
            "manifest": str(manifest_path),
            "manifest_hash": manifest_hash(manifest_path),
            "sample_count": len(rows),
            "label_counts": dict(sorted(Counter(row.rvl_label for row in rows).items())),
            "rejection_target_count": sum(1 for row in rows if row.is_rejection_target),
        },
        "development_metrics": development_metrics,
        "family_thresholds": family_diagnostics,
        "insufficient_support": sorted(
            family
            for family, entry in family_diagnostics.items()
            if entry["status"] != "calibrated"
        ),
        "split_integrity": integrity,
        "versions": {
            "classifier_version": EMBEDDING_CLASSIFIER_VERSION,
            "embedding_fingerprint": EMBEDDING_FINGERPRINT,
            "reference_fingerprint": index.reference_fingerprint,
            "model": model_config.descriptor(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(args.output, payload)
    return 0, payload


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        status, payload = run(args)
    except LeakageError as error:
        print(json.dumps({"error": "leakage", "detail": str(error)}, indent=2))
        return 2
    except (CorpusError, EmbeddingContractError) as error:
        print(json.dumps({"error": type(error).__name__, "detail": str(error)}, indent=2))
        return 1
    print(json.dumps(payload["thresholds"], ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
