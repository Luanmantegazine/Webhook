#!/usr/bin/env python3
"""Manifest reading, feature location and split-integrity checks, shared.

The three embedding scripts all have to answer the same three questions —
which samples am I reading, where is each one's ``classification_features``,
and do these splits overlap — and answering them three times is how a leak
gets into exactly one of them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.document.rvl_cdip_eval import (  # noqa: E402
    resolve_evaluation_target,
)

REQUIRED_COLUMNS = ("sample_id", "rvl_label")
FEATURES_COLUMN = "classification_features_path"
IMAGE_COLUMN = "image_path"
FEATURES_FILENAMES = (
    "classification_features.json",
    "features.json",
)


class CorpusError(RuntimeError):
    """A manifest, a cache or a split that must not be used as given."""


class LeakageError(CorpusError):
    """A sample appears in two splits that must stay disjoint.

    Calibrating a threshold on a document that is also in the benchmark makes
    the benchmark report the threshold's training accuracy. It is not a
    warning: the run stops.
    """


class ManifestRow(NamedTuple):
    sample_id: str
    rvl_label: str
    target_family: str
    is_rejection_target: bool
    image_path: str
    features_path: str
    row: dict[str, str]


def read_manifest(path: str | Path, *, require_label: bool = True) -> list[ManifestRow]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise CorpusError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise CorpusError(
                f"{manifest_path}: missing required column(s) {', '.join(missing)}; "
                f"found {', '.join(fieldnames) or '(none)'}"
            )
        rows: list[ManifestRow] = []
        for line_number, raw in enumerate(reader, start=2):
            sample_id = (raw.get("sample_id") or "").strip()
            if not sample_id:
                raise CorpusError(f"{manifest_path}:{line_number}: empty sample_id")
            label = (raw.get("rvl_label") or "").strip()
            if not label and not require_label:
                target_family, rejection = "", False
            else:
                try:
                    target = resolve_evaluation_target(label)
                except ValueError as error:
                    raise CorpusError(f"{manifest_path}:{line_number}: {error}") from error
                target_family, rejection = target.family, target.is_rejection_target
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    rvl_label=label,
                    target_family=target_family,
                    is_rejection_target=rejection,
                    image_path=(raw.get(IMAGE_COLUMN) or "").strip(),
                    features_path=(raw.get(FEATURES_COLUMN) or "").strip(),
                    row=dict(raw),
                )
            )
    duplicates = _duplicates(row.sample_id for row in rows)
    if duplicates:
        raise CorpusError(
            f"{manifest_path}: repeated sample_id(s) {', '.join(sorted(duplicates))}"
        )
    return rows


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def index_execution_features(executions_directory: str | Path) -> dict[str, Path]:
    """Find every stored ``classification_features`` under a directory.

    A workflow run leaves one directory per sample; this walks them and keys
    the records by ``sample_id`` as the record itself reports it, falling back
    to the directory name. It never re-extracts and never runs OCR.
    """
    root = Path(executions_directory)
    if not root.is_dir():
        raise CorpusError(f"executions directory not found: {root}")
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name in FEATURES_FILENAMES:
            sample_id = _sample_id_of(path)
            if sample_id:
                found.setdefault(sample_id, path)
            continue
        # A run that stores the whole workflow output in one JSON is also read,
        # as long as the classification_features are inside it.
        if path.name.endswith("classification_features.json"):
            sample_id = _sample_id_of(path)
            if sample_id:
                found.setdefault(sample_id, path)
    return found


def _sample_id_of(path: Path) -> str:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    record = extract_feature_record(payload)
    if isinstance(record, dict):
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id:
            return sample_id
    provenance = (record or {}).get("provenance") if isinstance(record, dict) else None
    if isinstance(provenance, dict):
        sample_id = str(provenance.get("sample_id") or "").strip()
        if sample_id:
            return sample_id
    return path.parent.name


def extract_feature_record(payload: Any) -> dict[str, Any] | None:
    """Unwrap a stored workflow output into the feature record itself."""
    if not isinstance(payload, dict):
        return None
    if "classification_text" in payload:
        return payload
    nested = payload.get("classification_features")
    if isinstance(nested, dict):
        return nested
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        return extract_feature_record(outputs)
    return None


def load_feature_record(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read feature record {source}: {error}") from error
    record = extract_feature_record(payload)
    if record is None:
        raise CorpusError(
            f"{source}: no classification_features in this file; the embedding "
            "classifier reads stored features and never re-runs OCR"
        )
    return record


def resolve_features_path(
    row: ManifestRow,
    *,
    manifest_path: Path,
    executions_index: dict[str, Path] | None = None,
) -> Path:
    if row.features_path:
        candidate = Path(row.features_path)
        if not candidate.is_absolute():
            candidate = (manifest_path.parent / candidate).resolve()
        if candidate.is_file():
            return candidate
        raise CorpusError(
            f"{row.sample_id}: {FEATURES_COLUMN} points at {candidate}, which does not exist"
        )
    if executions_index and row.sample_id in executions_index:
        return executions_index[row.sample_id]
    raise CorpusError(
        f"{row.sample_id}: no classification_features found; give the manifest a "
        f"{FEATURES_COLUMN} column or pass --executions pointing at the stored runs"
    )


def check_split_disjointness(
    splits: dict[str, list[ManifestRow]],
) -> dict[str, Any]:
    """Refuse overlapping splits, by ``sample_id`` and by ``image_path``.

    Both are checked because they fail differently: a re-generated manifest can
    give the same page a new id, and a copied corpus can give the same id a new
    path. Either one is the same document evaluated twice.
    """
    report: dict[str, Any] = {"splits": {}, "overlaps": []}
    names = sorted(splits)
    for name in names:
        rows = splits[name]
        report["splits"][name] = {
            "sample_count": len(rows),
            "labels": sorted({row.rvl_label for row in rows if row.rvl_label}),
        }
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            left_ids = {row.sample_id for row in splits[left]}
            right_ids = {row.sample_id for row in splits[right]}
            shared_ids = sorted(left_ids & right_ids)
            left_images = {row.image_path for row in splits[left] if row.image_path}
            right_images = {row.image_path for row in splits[right] if row.image_path}
            shared_images = sorted(left_images & right_images)
            if shared_ids or shared_images:
                report["overlaps"].append(
                    {
                        "splits": [left, right],
                        "shared_sample_ids": shared_ids[:50],
                        "shared_sample_id_count": len(shared_ids),
                        "shared_image_paths": shared_images[:50],
                        "shared_image_path_count": len(shared_images),
                    }
                )
    report["disjoint"] = not report["overlaps"]
    return report


def require_disjoint_splits(splits: dict[str, list[ManifestRow]]) -> dict[str, Any]:
    report = check_split_disjointness(splits)
    if not report["disjoint"]:
        details = "; ".join(
            f"{overlap['splits'][0]} and {overlap['splits'][1]} share "
            f"{overlap['shared_sample_id_count']} sample_id(s) and "
            f"{overlap['shared_image_path_count']} image path(s)"
            for overlap in report["overlaps"]
        )
        raise LeakageError(
            f"splits are not disjoint: {details}. Calibrating or building "
            "reference centroids on evaluated documents makes the benchmark "
            "report its own training accuracy."
        )
    return report


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = headers or sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
