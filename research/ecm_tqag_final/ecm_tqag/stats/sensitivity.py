"""Sensitivity floor for the secondary perturbation probe.

The protocol gates the whole perturbation probe behind a fixed floor:

* five positive controls and five text-sufficient negative controls,
* required separation of at least 8/10 for *each* answerer family,
* control-replicate disagreement not exceeding 10%.

On failure the probe is labeled insensitive and stops, and no ``Delta_perm``
estimate may be reported. This module returns that decision as data so
``stats.bootstrap.delta_perm_interval`` and the report layer can enforce it
mechanically instead of relying on an analyst remembering the rule.

"Separation" is scored as the number of controls the family got right in the
direction the control was designed to test: a positive control must be answered
correctly *with* the image, a negative (text-sufficient) control must be
answerable without it. Both are supplied as booleans by the caller; this module
does not infer them.
"""
from __future__ import annotations

from typing import Any, Sequence

from .validate import (
    blocked,
    require_bool_sequence,
    require_positive_int,
    require_same_length,
    require_sequence,
)

N_POSITIVE_CONTROLS = 5
N_NEGATIVE_CONTROLS = 5
N_CONTROLS_TOTAL = N_POSITIVE_CONTROLS + N_NEGATIVE_CONTROLS
MIN_SEPARATION = 8
MAX_REPLICATE_DISAGREEMENT = 0.10


def family_sensitivity(positive_correct: Sequence[bool],
                       negative_correct: Sequence[bool],
                       replicate_agreement: Sequence[bool],
                       *,
                       family: str,
                       min_separation: int = MIN_SEPARATION,
                       max_disagreement: float = MAX_REPLICATE_DISAGREEMENT
                       ) -> dict[str, Any]:
    """Evaluate one answerer family against the fixed floor.

    ``replicate_agreement[i]`` is True when the control replicate reproduced the
    original control outcome for item ``i``. Disagreement is the complement rate.
    The control counts are enforced exactly (5 and 5): a floor evaluated on a
    different number of controls is not the preregistered floor.
    """
    if not isinstance(family, str) or not family.strip():
        raise blocked("family_invalid")
    min_separation = require_positive_int("min_separation", min_separation)
    if isinstance(max_disagreement, bool) or not isinstance(max_disagreement, (int, float)):
        raise blocked("max_disagreement_not_numeric")
    max_disagreement = float(max_disagreement)
    if not 0.0 <= max_disagreement <= 1.0:
        raise blocked("max_disagreement_out_of_range")

    positives = require_bool_sequence("positive_correct", positive_correct)
    negatives = require_bool_sequence("negative_correct", negative_correct)
    replicates = require_bool_sequence("replicate_agreement", replicate_agreement)

    if len(positives) != N_POSITIVE_CONTROLS:
        raise blocked(f"positive_control_count:{len(positives)}!={N_POSITIVE_CONTROLS}")
    if len(negatives) != N_NEGATIVE_CONTROLS:
        raise blocked(f"negative_control_count:{len(negatives)}!={N_NEGATIVE_CONTROLS}")
    if len(replicates) != N_CONTROLS_TOTAL:
        raise blocked(f"replicate_count:{len(replicates)}!={N_CONTROLS_TOTAL}")

    separation = sum(positives) + sum(negatives)
    n_disagreements = sum(1 for ok in replicates if not ok)
    disagreement_rate = n_disagreements / float(len(replicates))

    separation_ok = separation >= min_separation
    # Strict '>' so exactly 10% passes, matching "must not exceed 10%".
    replicate_ok = disagreement_rate <= max_disagreement + 1e-12
    passed = bool(separation_ok and replicate_ok)

    failures: list[str] = []
    if not separation_ok:
        failures.append(f"separation_below_floor:{separation}/{N_CONTROLS_TOTAL}")
    if not replicate_ok:
        failures.append(f"replicate_disagreement_exceeded:{disagreement_rate:.3f}")

    return {
        "family": family,
        "n_positive_controls": len(positives),
        "n_negative_controls": len(negatives),
        "n_controls": N_CONTROLS_TOTAL,
        "positive_correct": sum(positives),
        "negative_correct": sum(negatives),
        "separation": separation,
        "min_separation": min_separation,
        "separation_ok": separation_ok,
        "n_replicate_disagreements": n_disagreements,
        "replicate_disagreement_rate": disagreement_rate,
        "max_replicate_disagreement": max_disagreement,
        "replicate_ok": replicate_ok,
        "passed": passed,
        "failures": failures,
    }


def sensitivity_floor(families: Sequence[dict[str, Any]], *,
                      min_families: int = 2) -> dict[str, Any]:
    """Combine per-family verdicts into the probe-level gate.

    The floor is conjunctive: **every** family must pass. One insensitive family
    is enough to label the probe insensitive, because the protocol requires the
    separation "for each family". At least two families must be evaluated.
    """
    min_families = require_positive_int("min_families", min_families)
    records = require_sequence("families", families)
    if len(records) < min_families:
        raise blocked(f"too_few_families:{len(records)}<{min_families}")

    names: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise blocked(f"family_record_not_object:{index}")
        for key in ("family", "passed"):
            if key not in record:
                raise blocked(f"family_record_missing_key:{index}:{key}")
        if not isinstance(record["passed"], bool):
            raise blocked(f"family_passed_not_boolean:{index}")
        names.append(str(record["family"]))
    if len(set(names)) != len(names):
        raise blocked("duplicate_family")

    failed = [str(r["family"]) for r in records if not r["passed"]]
    passed = not failed

    return {
        "check": "sensitivity_floor",
        "n_families": len(records),
        "families": [dict(r) for r in records],
        "family_names": names,
        "failed_families": failed,
        "passed": passed,
        # The protocol's own vocabulary: on failure the probe is INSENSITIVE and
        # stops, and Delta_perm must not be reported.
        "probe_status": "sensitive" if passed else "insensitive",
        "delta_perm_reportable": passed,
        "stop_probe": not passed,
    }
