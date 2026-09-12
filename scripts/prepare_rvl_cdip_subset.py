from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASET_ID = "Chan814/rvl-cdip"

SPLIT_ALIASES = {
    "validation": "val",
}

# Fallback order used by the Parquet mirror when the label feature does not
# expose ClassLabel names. It is intentionally not the order used by the
# legacy aharley/rvl_cdip.py loader.
PARQUET_MIRROR_LABELS = (
    "advertisement",
    "budget",
    "email",
    "file folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific publication",
    "scientific report",
    "specification",
)


def _normalize_rvl_label_name(label: str) -> str:
    normalized = " ".join(label.strip().lower().replace("_", " ").split())
    # The current Parquet mirror's dataset card contains this typo.
    if normalized == "advertissement":
        normalized = "advertisement"
    return normalized


def _resolve_rvl_label(value, label_feature=None) -> tuple[int, str]:
    """Resolve RVL-CDIP class id/name without assuming storage representation."""
    if label_feature is not None and hasattr(label_feature, "int2str"):
        label_index = int(value)
        return label_index, _normalize_rvl_label_name(label_feature.int2str(label_index))

    if isinstance(value, str):
        normalized = _normalize_rvl_label_name(value)
        if normalized not in PARQUET_MIRROR_LABELS:
            raise ValueError(f"Unknown RVL-CDIP label: {value!r}")
        return PARQUET_MIRROR_LABELS.index(normalized), normalized

    label_index = int(value)
    if label_index < 0 or label_index >= len(PARQUET_MIRROR_LABELS):
        raise ValueError(f"Unknown RVL-CDIP label id: {label_index}")
    return label_index, _normalize_rvl_label_name(PARQUET_MIRROR_LABELS[label_index])


def _validate_class_quotas(raw_quotas: dict) -> dict[str, int]:
    """Normalize and validate explicit quotas for official RVL-CDIP labels."""
    if not isinstance(raw_quotas, dict) or not raw_quotas:
        raise ValueError("class quotas must be a non-empty JSON object or set of LABEL=COUNT entries")

    quotas: dict[str, int] = {}
    for raw_label, quota in raw_quotas.items():
        if not isinstance(raw_label, str):
            raise ValueError("class quota labels must be strings")
        label = _normalize_rvl_label_name(raw_label)
        if label not in PARQUET_MIRROR_LABELS:
            raise ValueError(f"Unknown official RVL-CDIP label: {raw_label!r}")
        if type(quota) is not int or quota < 1:
            raise ValueError(f"Quota for {raw_label!r} must be a positive integer")
        if label in quotas:
            raise ValueError(f"Duplicate class quota for {label!r}")
        quotas[label] = quota
    return dict(sorted(quotas.items()))


def _class_quotas_from_entries(entries: list[str]) -> dict[str, int]:
    raw_quotas: dict[str, int] = {}
    for entry in entries:
        label, separator, raw_quota = entry.rpartition("=")
        if not separator or not label.strip() or not raw_quota.strip():
            raise ValueError(f"Invalid class quota {entry!r}; use LABEL=COUNT")
        try:
            quota = int(raw_quota)
        except ValueError as exc:
            raise ValueError(
                f"Invalid quota {raw_quota!r} for {label!r}; use a positive integer"
            ) from exc
        if label in raw_quotas:
            raise ValueError(f"Duplicate class quota for {label!r}")
        raw_quotas[label] = quota
    return _validate_class_quotas(raw_quotas)


def _class_quotas_from_file(path: Path) -> dict[str, int]:
    try:
        raw_quotas = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read class quotas file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in class quotas file {path}: {exc.msg}") from exc
    return _validate_class_quotas(raw_quotas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "val", "test", "train"), default="validation")
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--output-directory", type=Path, default=Path("data/rvl_cdip_subset"))
    class_quota_group = parser.add_mutually_exclusive_group()
    class_quota_group.add_argument(
        "--class-quotas",
        type=Path,
        metavar="JSON_PATH",
        help="JSON object mapping official RVL-CDIP labels to positive integer quotas",
    )
    class_quota_group.add_argument(
        "--class-quota",
        action="append",
        default=[],
        metavar="LABEL=COUNT",
        help="Repeatable explicit quota for one official RVL-CDIP label",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Parquet-based RVL-CDIP repository; legacy script repositories are unsupported",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub revision for a Parquet branch",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "rvl_cdip_taxonomy.json",
    )
    args = parser.parse_args()

    if args.per_family < 1:
        parser.error("--per-family must be positive")

    try:
        requested_quotas = (
            _class_quotas_from_file(args.class_quotas)
            if args.class_quotas
            else _class_quotas_from_entries(args.class_quota)
            if args.class_quota
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    if requested_quotas is not None and args.split not in SPLIT_ALIASES:
        parser.error("explicit class quotas are supported only for the validation development split")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the 'datasets' package before preparing RVL-CDIP") from exc

    if requested_quotas is None:
        with args.taxonomy.open("r", encoding="utf-8") as handle:
            taxonomy = json.load(handle)
        label_mapping = taxonomy["label_mapping"]
        target_families = tuple(taxonomy["supported_families"])
        counts = {family: 0 for family in target_families}
    else:
        label_mapping = None
        counts = {label: 0 for label in requested_quotas}
    counts_by_label: dict[str, int] = {}

    dataset_split = SPLIT_ALIASES.get(args.split, args.split)

    output_directory = args.output_directory.resolve() / args.split
    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "rvl_cdip_manifest.csv"

    dataset_kwargs = {
        "path": args.dataset_id,
        "split": dataset_split,
        "streaming": True,
    }
    if args.revision:
        dataset_kwargs["revision"] = args.revision
    dataset = load_dataset(**dataset_kwargs)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    label_feature = dataset.features.get("label") if dataset.features else None

    rows = []
    for sample_index, sample in enumerate(dataset):
        try:
            label_index, rvl_label = _resolve_rvl_label(sample["label"], label_feature)
        except (TypeError, ValueError):
            continue
        if requested_quotas is None:
            if rvl_label not in label_mapping:
                raise ValueError(
                    f"Dataset label {rvl_label!r} is absent from the taxonomy mapping"
                )
            target_family = label_mapping[rvl_label]
            if counts[target_family] >= args.per_family:
                continue
        else:
            if rvl_label not in requested_quotas or counts[rvl_label] >= requested_quotas[rvl_label]:
                continue

        label_count = counts_by_label.get(rvl_label, 0)
        label_slug = rvl_label.replace(" ", "_")
        sample_id = f"rvl-{args.split}-c{label_index:02d}-{label_slug}-{label_count:04d}"
        image_path = images_directory / f"{sample_id}.tif"
        sample["image"].convert("L").save(image_path, format="TIFF", compression="tiff_deflate")
        rows.append(
            {
                "sample_id": sample_id,
                "split": args.split,
                "label_index": label_index,
                "rvl_label": rvl_label,
                "image_path": str(image_path.relative_to(output_directory)),
                "source_stream_index": sample_index,
            }
        )
        counts_by_label[rvl_label] = label_count + 1
        if requested_quotas is None:
            counts[target_family] += 1
            quotas_reached = all(count >= args.per_family for count in counts.values())
        else:
            counts[rvl_label] += 1
            quotas_reached = all(
                counts[label] >= requested_quotas[label] for label in requested_quotas
            )
        if quotas_reached:
            break

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "split",
                "label_index",
                "rvl_label",
                "image_path",
                "source_stream_index",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    if requested_quotas is not None:
        missing = {
            label: requested_quotas[label] - count
            for label, count in counts.items()
            if count < requested_quotas[label]
        }
        summary_path = output_directory / "rvl_cdip_manifest_summary.json"
        summary = {
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "manifest": str(manifest_path),
            "summary_manifest": str(summary_path),
            "seed": args.seed,
            "split": args.split,
            "requested_quotas": requested_quotas,
            "collected_counts": counts,
            "missing_quotas": missing,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        missing = {
            family: args.per_family - count
            for family, count in counts.items()
            if count < args.per_family
        }
        summary = {
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "manifest": str(manifest_path),
            "counts": counts,
            "counts_by_label": dict(sorted(counts_by_label.items())),
            "missing": missing,
        }
    print(json.dumps(summary))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
