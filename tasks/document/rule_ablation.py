"""Per-rule ablation directly from a predictions CSV — no fixtures required.

Why this exists
---------------
The ablation was previously described as a step of the fixture battery, which
made it look as though it needed a fixture corpus that does not exist. It does
not. What an ablation needs is the **rule indicator vector** per document plus
the label, and both fit in the CSV the evaluation already produces. Removing the
fixture dependency removes an artifact class from the critical path.

What it still needs
-------------------
A split that is not the one the study reports. That is an experimental
decision, not something a tool can infer, so the caller declares it with
``--reported-split`` and only that split is refused. Under RVL-CDIP's official
three-way partition a study that reports on ``test`` may tune freely on
``validation`` — tuning is what a validation split is for. A study that reports
on ``validation`` needs its development sample from ``train`` instead. Both
choices are legitimate; leaving the choice implicit is not, so the declaration
is written into the result.

Required columns
----------------
``rvl_label``            corpus class (used to derive the family)
``rule_indicators``      JSON object {rule_id: bool} — add ``include_indicators=True``
                         to the classification call to emit it
``evidence_channels``    JSON array of channel names available for the document
                         (optional; see the note on the fallback below)
``source_split``         the split the document came from

The ablation is exact only with the indicator vector: from ``rules_by_family``
alone a rule suppressed by grouping is invisible, so removing the group winner
would appear to remove the whole group's contribution rather than fall back to
the runner-up.

Metric definitions
------------------
``marginal_evidence`` forces one stored rule indicator false but retains all
default weights, including its normalising mass. ``marginal_deletion`` sets its
weight to zero, changing that mass exactly as deleting the rule would. A
never-fired rule must have zero marginal evidence, yet can have a nonzero
marginal deletion through its declared denominator mass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

# Make ``python tasks/document/rule_ablation.py ...`` work as well as importing
# this module from the repository root.  The scorer imports ``tasks.document``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tasks.document import rules_classifier_core as core
from tasks.document import rvl_cdip_eval as ev

DEFAULT_CHANNELS = ("text", "layout", "multipage")
SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


def _family_for(row: dict) -> str | None:
    label = row.get("rvl_label") or ""
    if not label:
        return None
    try:
        return ev.resolve_evaluation_target(label).family
    except ValueError:
        return None


def load_rows(path: str, require_indicators: bool = True) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    if require_indicators and "rule_indicators" not in rows[0]:
        raise SystemExit(
            f"{path} has no 'rule_indicators' column.\n"
            "Pass include_indicators=True to classify_with_rules and re-emit the CSV.\n"
            "Without it the ablation cannot see rules suppressed by grouping, and its\n"
            "numbers would be wrong in a way that is not visible in the output."
        )
    return rows


def _normalise_split(value: str | None) -> str:
    name = (value or "").strip().lower()
    return SPLIT_ALIASES.get(name, name)


def guard_split(rows: Sequence[dict], declared: str | None, reported: str) -> str:
    """Refuse to ablate on the split the study reports.

    The earlier version of this guard refused any split named ``validation``,
    which over-constrained the experiment: it assumed the write-up would quote
    validation numbers. Under the standard three-way protocol the opposite is
    true — validation exists to be tuned on, and the held-out claim rests on
    ``test``. The caller therefore declares which split is reported, and only
    that one is off limits. The decision is recorded in the output so a later
    reader can check that the tuning split and the reported split really were
    different.
    """
    splits = {_normalise_split(row.get("source_split")) for row in rows}
    splits.discard("")
    if len(splits) > 1:
        raise SystemExit(f"CSV mixes splits {sorted(splits)}; ablate one split at a time.")
    sourced = next(iter(splits), "")
    name = _normalise_split(declared) or sourced
    reported = _normalise_split(reported)
    if not name:
        raise SystemExit("cannot determine the split: pass --split or add a source_split column")
    if sourced and declared and name != sourced:
        raise SystemExit(
            f"--split {name!r} disagrees with CSV source_split {sourced!r}; "
            "do not relabel the source split."
        )
    if name == reported:
        raise SystemExit(
            f"Refusing to ablate on source split '{name}' because it is also the reported split "
            f"(--reported-split {reported}).\n\n"
            "Choosing which rules to keep by their effect on the reported split makes\n"
            "every number on it fitted rather than held out.\n\n"
            "Either tune on a different split, or — if the study really does report\n"
            f"'{reported}' — say which split the tuning uses instead."
        )
    return name


def _channels(row: dict) -> frozenset[str]:
    raw = row.get("evidence_channels")
    if raw:
        try:
            return frozenset(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return frozenset(DEFAULT_CHANNELS)


def ranking_accuracy(
    rows: Sequence[dict],
    weights: dict[str, float] | None = None,
    forced_false: set[str] | None = None,
) -> tuple[int, int]:
    """Argmax-correct count, recomputed from the indicators under given weights.

    ``forced_false`` forces named indicators to ``False`` while retaining
    ``DEFAULT_WEIGHTS`` (unless explicit ``weights`` are supplied), and therefore
    leaves every family's normalising mass untouched.
    """
    correct = total = 0
    for row in rows:
        want = _family_for(row)
        if want is None:  # rejection target: no family to rank
            continue
        total += 1
        indicators = json.loads(row["rule_indicators"] or "{}")
        if forced_false:
            indicators = dict(indicators)
            indicators.update({rule_id: False for rule_id in forced_false})
        # score_families reads the channels off the features dict; the ablation
        # only needs those two keys, so a minimal stand-in is enough and avoids
        # carrying whole feature records through the CSV.
        available = _channels(row)
        features = {
            "measured_page_ratio": 1.0 if "layout" in available else 0.0,
            "geometry_page_ratio": 1.0 if "geometry" in available else 0.0,
            "total_pages": 1 if "multipage" in available else 0,
        }
        breakdown = core.score_families(features, indicators, weights)
        ranked = sorted(
            ((family, entry["score"]) for family, entry in breakdown.items()),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked and ranked[0][1] > 0 and ranked[0][0] == want:
            correct += 1
    return correct, total


def ablate(rows: Sequence[dict]) -> list[dict]:
    """Calculate two distinct marginal metrics for every rule.

    ``marginal_evidence`` forces just that rule's stored indicator to ``False``
    while retaining :data:`core.DEFAULT_WEIGHTS`.  Its family therefore retains
    the same normalising mass.  This is the marginal value of the rule's
    *observed evidence*.

    ``marginal_deletion`` zeroes the rule's weight, which both removes it from
    the score and changes the family's normalising mass.  This is the marginal
    value of deleting the rule from the scorer.

    The two measurements answer different questions and must not be
    interchanged.  Deletion can be nonzero for a rule that never fires: its
    declared weight can still affect the denominator.  Conversely, a
    never-fired rule always has zero ``marginal_evidence``.

    The two diverge whenever a rule's group sits inside the family's decisive
    mass, and the divergence is not a subtlety — in the first development run,
    ``form_structured.form_heading`` fired **zero times** and still showed a
    deletion value of +0.0148. A rule that never fires cannot contribute
    evidence; the entire effect was the denominator moving, i.e. the family's
    effective threshold being re-calibrated by the removal.

    Reading them together: high evidence value means the rule earns its place.
    Near-zero evidence value with a large deletion value means the rule is
    acting as threshold ballast, and the honest fix is to set the family's
    decisive mass deliberately rather than to keep a dead rule for its side
    effect.
    """
    base_correct, total = ranking_accuracy(rows)
    base = base_correct / total if total else 0.0
    results = []
    for rule_id in core.RULE_IDS:
        if not core.DEFAULT_WEIGHTS.get(rule_id):
            continue
        weights = dict(core.DEFAULT_WEIGHTS)
        weights[rule_id] = 0.0
        deleted, _ = ranking_accuracy(rows, weights)
        evidence_removed, _ = ranking_accuracy(rows, forced_false={rule_id})
        results.append(
            {
                "rule": rule_id,
                "fires": sum(
                    1 for row in rows if json.loads(row["rule_indicators"] or "{}").get(rule_id)
                ),
                "ranking_if_deleted": round(deleted / total, 4) if total else 0.0,
                "marginal_deletion": round(base - deleted / total, 4) if total else 0.0,
                "marginal_evidence": round(base - evidence_removed / total, 4)
                if total
                else 0.0,
            }
        )
    results.sort(key=lambda item: (-item["marginal_evidence"], -item["marginal_deletion"], item["rule"]))
    return results


def ablate_family(rows: Sequence[dict], family: str) -> dict:
    """Effect of removing a whole family from the taxonomy."""
    base_correct, total = ranking_accuracy(rows)
    weights = {
        rule.rule_id: (0.0 if rule.family == family else core.DEFAULT_WEIGHTS[rule.rule_id])
        for rule in core.RULES
    }
    correct, _ = ranking_accuracy(rows, weights)
    return {
        "family": family,
        "ranking_before": round(base_correct / total, 4) if total else 0.0,
        "ranking_after": round(correct / total, 4) if total else 0.0,
        "marginal_value": round((base_correct - correct) / total, 4) if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", help="predictions CSV carrying rule_indicators")
    parser.add_argument("--split", help="declared split name; inferred from the CSV if absent")
    parser.add_argument(
        "--reported-split",
        required=True,
        help=(
            "the split whose numbers the write-up will quote; ablating on it is refused. "
            "Set it to 'validation' only if the study reports validation, in which case "
            "tuning must happen on train."
        ),
    )
    parser.add_argument("--family", action="append", default=[], help="also ablate a whole family")
    parser.add_argument("--json", help="write the full result here")
    args = parser.parse_args(argv)

    rows = load_rows(args.predictions)
    split = guard_split(rows, args.split, args.reported_split)
    correct, total = ranking_accuracy(rows)
    baseline = correct / total if total else 0.0
    reported_split = _normalise_split(args.reported_split)
    print(f"source_split={split} · reported_split={reported_split}")
    print(f"documentos classificáveis={total}")
    print(f"acurácia de ranking (base) = {correct}/{total} = {baseline:.4f}\n")

    families = [ablate_family(rows, family) for family in args.family]
    for entry in families:
        print(
            f"[família] {entry['family']:24s} {entry['ranking_before']:.4f} -> "
            f"{entry['ranking_after']:.4f}  ({-entry['marginal_value']:+.4f})"
        )
    if families:
        print()

    results = ablate(rows)
    print(f"{'marginal_evidence':>18s} {'marginal_deletion':>18s} {'dispara':>8s}  regra")
    for entry in results:
        if entry["marginal_deletion"] == 0 and entry["marginal_evidence"] == 0 and entry["fires"] == 0:
            continue
        flag = ""
        if entry["fires"] and entry["marginal_evidence"] < 0:
            flag = "   <- evidência prejudicial"
        elif entry["marginal_evidence"] == 0 and entry["marginal_deletion"] != 0:
            flag = "   <- só lastro de limiar"
        print(
            f"{entry['marginal_evidence']:+18.4f} {entry['marginal_deletion']:+18.4f} "
            f"{entry['fires']:8d}  {entry['rule']}{flag}"
        )

    dead = [entry for entry in results if entry["fires"] == 0]
    print(f"\nregras que nunca disparam neste split: {len(dead)}")
    for entry in dead:
        print(f"    {entry['rule']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tuning_split": split,
                    "source_split": split,
                    "reported_split": reported_split,
                    "baseline": baseline,
                    "families": families,
                    "rules": results,
                },
                handle,
                indent=2,
            )
        print(f"\nescrito em {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())