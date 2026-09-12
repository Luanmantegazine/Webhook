#!/usr/bin/env python3
"""Measure ``classify_with_rules`` alone, on features already in memory.

Why this is a separate script
-----------------------------
The evaluator's ``classifier_p95`` is measured inside a loop that has just
finished reading a JSON file and extracting features for the same document, so
it carries the cache and allocator state of that work. It is a reasonable
number for the evaluator and a poor one for a latency claim: it answers "how
long did the classifier take inside the evaluation" rather than "how long does
classification take".

This script answers the second question, and nothing else:

* every feature record is loaded and validated **before** timing starts;
* module import, JSON reading, OCR and feature extraction are outside the
  measured region entirely;
* a warm-up pass runs first, so the first-call costs of interpreter caches and
  regex compilation are not counted as classification;
* ``include_indicators=False`` and ``include_indicators=True`` are measured
  separately, because the indicator vector is a diagnostic that production does
  not ask for — reporting its cost as the operational latency overstates the
  pipeline;
* three independent trials run end to end, and the median of their p95 values
  is reported alongside each trial, because a single p95 from a single trial is
  one draw from a noisy distribution;
* both modes are checked to produce the same decision, family, scores and
  margins, because a cheaper measurement of a different classifier is not a
  measurement of this one.

``measurement`` in the report records all of that, so a number taken from this
file cannot be quoted without the conditions that produced it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rules_from_cache import (  # noqa: E402
    EvaluationError,
    FeatureVersionLedger,
    _load_cached_samples,
    _load_manifest_records,
)
from tasks.document.rules_classifier_core import (  # noqa: E402
    CLASSIFIER_VERSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_RECOGNIZED_CHARACTERS,
    DEFAULT_MIN_SCORE_MARGIN,
    FEATURE_EXTRACTION_VERSION,
    RULE_FINGERPRINT,
    SCHEMA_VERSION,
    TAXONOMY_VERSION,
    classify_with_rules,
)

#: The fields that make two results the same decision. ``rule_indicators`` is
#: deliberately absent — it is the one field the two modes are meant to differ
#: in — and so is any timing field.
DECISION_FIELDS = (
    "document_family",
    "decision",
    "reason",
    "confidence",
    "score",
    "score_margin",
    "top_candidate",
    "runner_up",
    "candidate_scores",
    "decision_mass",
    "available_mass",
)


def decision_signature(result: dict[str, Any]) -> tuple:
    """A hashable rendering of everything that counts as "the same answer"."""
    return tuple(
        json.dumps(result.get(field), sort_keys=True, default=str)
        for field in DECISION_FIELDS
    )


def _percentile(ordered: list[float], percentile: float) -> float:
    """Nearest-rank percentile over an already sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def summarise(durations_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(durations_ms)
    return {
        "call_count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4) if ordered else 0.0,
        "median": round(statistics.median(ordered), 4) if ordered else 0.0,
        "p90": round(_percentile(ordered, 0.90), 4),
        "p95": round(_percentile(ordered, 0.95), 4),
        "p99": round(_percentile(ordered, 0.99), 4),
        "stdev": round(statistics.pstdev(ordered), 4) if len(ordered) > 1 else 0.0,
        "min": round(ordered[0], 4) if ordered else 0.0,
        "max": round(ordered[-1], 4) if ordered else 0.0,
    }


@dataclass
class Options:
    repetitions: int
    trials: int
    warmup_repetitions: int
    confidence_threshold: float
    min_score_margin: float
    min_recognized_characters: int
    classification_mode: str


def _classify(features: dict, options: Options, include_indicators: bool) -> dict:
    return classify_with_rules(
        features,
        confidence_threshold=options.confidence_threshold,
        min_score_margin=options.min_score_margin,
        min_recognized_characters=options.min_recognized_characters,
        mode=options.classification_mode,
        include_indicators=include_indicators,
    )


def run_trial(
    feature_records: list[dict],
    options: Options,
    include_indicators: bool,
) -> dict[str, Any]:
    """One trial: ``repetitions`` passes over every feature record."""
    durations_ms: list[float] = []
    for _repetition in range(options.repetitions):
        for features in feature_records:
            started = perf_counter()
            _classify(features, options, include_indicators)
            durations_ms.append((perf_counter() - started) * 1000.0)
    return summarise(durations_ms)


def warm_up(feature_records: list[dict], options: Options) -> int:
    """Run both modes before measuring, and report how many calls that was.

    Regex objects, the rule tuple and the gate table are all built at import
    time, but their first *use* still pays for cold branch prediction and
    allocator growth. Counting that as classification latency reports a cost
    the second document never pays.
    """
    calls = 0
    for _repetition in range(max(1, options.warmup_repetitions)):
        for features in feature_records:
            for include_indicators in (False, True):
                _classify(features, options, include_indicators)
                calls += 1
    return calls


def check_equivalence(
    feature_records: list[dict], options: Options
) -> tuple[bool, list[dict[str, Any]]]:
    """Both modes must produce the same decision for every record."""
    differences: list[dict[str, Any]] = []
    for index, features in enumerate(feature_records):
        without = _classify(features, options, False)
        with_indicators = _classify(features, options, True)
        if decision_signature(without) != decision_signature(with_indicators):
            differences.append(
                {
                    "record_index": index,
                    "fields": [
                        field
                        for field in DECISION_FIELDS
                        if without.get(field) != with_indicators.get(field)
                    ],
                }
            )
    return (not differences), differences


def benchmark(feature_records: list[dict], options: Options) -> dict[str, Any]:
    equivalent, differences = check_equivalence(feature_records, options)
    warmup_calls = warm_up(feature_records, options)

    results: dict[str, Any] = {}
    for label, include_indicators in (("indicators_off", False), ("indicators_on", True)):
        trials = [
            run_trial(feature_records, options, include_indicators)
            for _trial in range(options.trials)
        ]
        p95_values = [trial["p95"] for trial in trials]
        results[label] = {
            "include_indicators": include_indicators,
            "trials": trials,
            "p95_by_trial": p95_values,
            "median_p95_across_trials": round(statistics.median(p95_values), 4),
            "p95_spread": round(max(p95_values) - min(p95_values), 4),
        }

    return {
        "sample_count": len(feature_records),
        "repetitions": options.repetitions,
        "trials": options.trials,
        "warmup_completed": warmup_calls > 0,
        "warmup_calls": warmup_calls,
        "indicators_off": results["indicators_off"],
        "indicators_on": results["indicators_on"],
        "decision_outputs_equivalent": equivalent,
        "decision_output_differences": differences,
        "measurement": {
            "measures": "tasks.document.rules_classifier_core.classify_with_rules",
            "excluded_from_measurement": [
                "module import",
                "manifest and JSON reading",
                "OCR",
                "layout detection",
                "feature extraction",
                "feature contract validation",
            ],
            "features_preloaded_in_memory": True,
            "warmup_before_measurement": True,
            "timing_unit": "milliseconds_per_classify_call",
            "percentile_method": "nearest_rank",
            "operational_latency_series": "indicators_off",
            "operational_latency_note": (
                "indicators_on includes the diagnostic rule-indicator vector, which "
                "production does not request. Quoting it as the operational latency "
                "overstates the pipeline; quote indicators_off, and say so."
            ),
            "not_a_pipeline_latency": (
                "This is classifier time only. End-to-end latency is dominated by OCR "
                "and layout detection, which are not measured here at all."
            ),
        },
        "configuration": {
            "classification_mode": options.classification_mode,
            "confidence_threshold": options.confidence_threshold,
            "min_score_margin": options.min_score_margin,
            "min_recognized_characters": options.min_recognized_characters,
        },
        "versions": {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
            "classifier_version": CLASSIFIER_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "feature_fingerprint": (
                feature_records[0].get("feature_fingerprint") if feature_records else ""
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }


def load_features(manifest: Path, max_samples: int) -> list[dict]:
    """Read and validate every feature record *before* any timing starts."""
    classifier_rows, _input_errors, _split_column = _load_manifest_records(manifest)
    samples = _load_cached_samples(manifest, classifier_rows, FeatureVersionLedger())
    records = [sample.features for sample in samples]
    if max_samples > 0:
        records = records[:max_samples]
    if not records:
        raise EvaluationError(f"Manifest {manifest} yielded no feature records.")
    return records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="the same cache manifest the evaluator reads")
    parser.add_argument("--output", type=Path, default=Path("output/classifier_latency.json"))
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="passes over the full feature set per trial (minimum 20)",
    )
    parser.add_argument("--trials", type=int, default=3, help="independent trials (minimum 3)")
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every sample")
    parser.add_argument("--classification-mode", choices=("evaluate", "observe"), default="evaluate")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--min-score-margin", type=float, default=DEFAULT_MIN_SCORE_MARGIN)
    parser.add_argument(
        "--min-recognized-characters", type=int, default=DEFAULT_MIN_RECOGNIZED_CHARACTERS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.repetitions < 20:
        print("ERROR: --repetitions must be at least 20.", file=sys.stderr)
        return 2
    if args.trials < 3:
        print("ERROR: --trials must be at least 3.", file=sys.stderr)
        return 2

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        print(f"ERROR: manifest does not exist: {manifest}", file=sys.stderr)
        return 2

    try:
        feature_records = load_features(manifest, args.max_samples)
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    options = Options(
        repetitions=args.repetitions,
        trials=args.trials,
        warmup_repetitions=args.warmup_repetitions,
        confidence_threshold=args.confidence_threshold,
        min_score_margin=args.min_score_margin,
        min_recognized_characters=args.min_recognized_characters,
        classification_mode=args.classification_mode,
    )
    report = benchmark(feature_records, options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "report": str(args.output.resolve()),
                "sample_count": report["sample_count"],
                "decision_outputs_equivalent": report["decision_outputs_equivalent"],
                "operational_median_p95_ms": report["indicators_off"][
                    "median_p95_across_trials"
                ],
                "indicators_on_median_p95_ms": report["indicators_on"][
                    "median_p95_across_trials"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["decision_outputs_equivalent"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
