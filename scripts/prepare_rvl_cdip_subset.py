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


def _normalize_rvl_label(value, label_feature=None) -> str:
    """Resolve integer/String/ClassLabel values without assuming label order."""
    if isinstance(value, str):
        label = value
    elif label_feature is not None and hasattr(label_feature, "int2str"):
        label = label_feature.int2str(int(value))
    else:
        label_index = int(value)
        if label_index < 0 or label_index >= len(PARQUET_MIRROR_LABELS):
            raise ValueError(f"Unknown RVL-CDIP label id: {label_index}")
        label = PARQUET_MIRROR_LABELS[label_index]

    normalized = " ".join(label.strip().lower().replace("_", " ").split())
    # The current Parquet mirror's dataset card contains this typo.
    if normalized == "advertissement":
        normalized = "advertisement"
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "val", "test", "train"), default="validation")
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--output-directory", type=Path, default=Path("data/rvl_cdip_subset"))
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
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the 'datasets' package before preparing RVL-CDIP") from exc

    with args.taxonomy.open("r", encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    label_mapping = taxonomy["label_mapping"]
    target_families = tuple(taxonomy["supported_families"])
    counts = {family: 0 for family in target_families}

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
            rvl_label = _normalize_rvl_label(sample["label"], label_feature)
        except (TypeError, ValueError):
            continue
        if rvl_label not in label_mapping:
            raise ValueError(
                f"Dataset label {rvl_label!r} is absent from the taxonomy mapping"
            )
        target_family = label_mapping[rvl_label]
        if counts[target_family] >= args.per_family:
            continue

        family_index = counts[target_family]
        sample_id = f"rvl-{args.split}-{target_family}-{family_index:04d}"
        image_path = images_directory / f"{sample_id}.tif"
        sample["image"].convert("L").save(image_path, format="TIFF", compression="tiff_deflate")
        rows.append(
            {
                "sample_id": sample_id,
                "split": args.split,
                "rvl_label": rvl_label,
                "target_family": target_family,
                "image_path": str(image_path.relative_to(output_directory)),
                "source_stream_index": sample_index,
            }
        )
        counts[target_family] += 1
        if all(count >= args.per_family for count in counts.values()):
            break

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "split",
                "rvl_label",
                "target_family",
                "image_path",
                "source_stream_index",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    missing = {
        family: args.per_family - count
        for family, count in counts.items()
        if count < args.per_family
    }
    print(json.dumps({
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "manifest": str(manifest_path),
        "counts": counts,
        "missing": missing,
    }))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())