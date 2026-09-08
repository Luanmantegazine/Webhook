#!/usr/bin/env python3
"""Create a balanced Hydra-family subset from the RVL-CDIP dataset.

The script streams a shuffled Hugging Face split, writes lossless TIFF scans,
and creates the manifest required before executing Hydra's rules benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RVL_LABELS = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test", "train"), default="validation")
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--output-directory", type=Path, default=Path("data/rvl_cdip_subset"))
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

    output_directory = args.output_directory.resolve() / args.split
    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "rvl_cdip_manifest.csv"

    dataset = load_dataset("aharley/rvl_cdip", split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rows = []
    for sample_index, sample in enumerate(dataset):
        label_value = int(sample["label"])
        if label_value < 0 or label_value >= len(RVL_LABELS):
            continue
        rvl_label = RVL_LABELS[label_value]
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
    print(json.dumps({"manifest": str(manifest_path), "counts": counts, "missing": missing}))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

