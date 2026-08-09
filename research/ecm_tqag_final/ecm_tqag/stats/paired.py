"""Primary confirmatory endpoint: two-sided **exact** McNemar over paired chunks.

The preregistered primary contrast is full ECM--TQAG versus the caption-mediated
baseline on gate-valid yield per chunk, over the frozen census of 16 paired
chunks. This module implements that test and nothing else claims to be primary.

Design commitments encoded here rather than left to the caller:

* The unit of analysis is the chunk. ``PRIMARY_N_CHUNKS`` is enforced by default
  so a run that silently lost or duplicated a cell cannot produce a primary
  p-value.
* The test is the *exact* conditional binomial test on the discordant pairs, not
  a chi-square with or without continuity correction. For n = 16 the asymptotic
  approximation is not licensed.
* Pair identity is checked by *ordered* identifier equality when identifiers are
  supplied. Same-set-different-order silently rebuilds a wrong 2x2 table.
* A non-significant result is reported as ``inconclusive``, never as
  equivalence, and the prespecified minimum discordance is reported alongside so
  an underpowered table is visible in the artifact itself.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .intervals import clopper_pearson
from .validate import (
    blocked,
    require_aligned_ids,
    require_alpha,
    require_bool_sequence,
    require_expected_n,
    require_same_length,
    require_unit_ids,
)

# The frozen evaluation population: 16 image-bearing, text-sufficient chunks.
PRIMARY_N_CHUNKS = 16

PRIMARY_ENDPOINT = "gate_valid_yield_per_chunk"
PRIMARY_TEST = "mcnemar_exact_two_sided"


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool], *,
                  alpha: float = 0.05,
                  expected_n: int | None = None,
                  left_ids: Sequence[str] | None = None,
                  right_ids: Sequence[str] | None = None,
                  left_label: str = "left",
                  right_label: str = "right") -> dict[str, Any]:
    """Two-sided exact McNemar test on paired boolean outcomes.

    The exact two-sided p-value is the conditional binomial test on the
    discordant count ``d = b + c`` under ``p = 1/2``:

        p = min(1, 2 * P(X <= min(b, c) | d, 1/2))

    The ``min(1, ...)`` clamp is what makes this the standard exact two-sided
    test; doubling alone can exceed 1 when the table is near-balanced.

    ``d == 0`` yields ``p = 1.0`` by definition (no evidence either way), and is
    flagged via ``no_discordance`` so it cannot be mistaken for a tested null.
    """
    alpha = require_alpha(alpha)
    left_values = require_bool_sequence(left_label, left)
    right_values = require_bool_sequence(right_label, right)
    require_same_length(left_label, left_values, right_label, right_values)

    n = len(left_values)
    if expected_n is not None:
        require_expected_n(n, expected_n, label="n_pairs")

    pair_ids: list[str] | None = None
    if left_ids is not None or right_ids is not None:
        if left_ids is None or right_ids is None:
            raise blocked("pair_ids_incomplete")
        ids_a = require_unit_ids(f"{left_label}_ids", left_ids, n)
        ids_b = require_unit_ids(f"{right_label}_ids", right_ids, n)
        pair_ids = require_aligned_ids(f"{left_label}_ids", ids_a,
                                       f"{right_label}_ids", ids_b)

    both = sum(1 for x, y in zip(left_values, right_values) if x and y)
    neither = sum(1 for x, y in zip(left_values, right_values) if not x and not y)
    b = sum(1 for x, y in zip(left_values, right_values) if x and not y)
    c = sum(1 for x, y in zip(left_values, right_values) if not x and y)
    d = b + c

    if d == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        lower_tail = math.fsum(math.comb(d, i) for i in range(k + 1)) / float(2 ** d)
        p_value = min(1.0, 2.0 * lower_tail)

    required = minimum_unidirectional_discordance(alpha)
    significant = p_value <= alpha

    return {
        "test": PRIMARY_TEST,
        "endpoint": PRIMARY_ENDPOINT,
        "unit": "chunk",
        "n_pairs": n,
        "n": n,
        "pair_ids": pair_ids,
        "table": {
            "both": both,
            f"{left_label}_only": b,
            f"{right_label}_only": c,
            "neither": neither,
        },
        "left_only": b,
        "right_only": c,
        "discordant": d,
        "p_two_sided_exact": p_value,
        "alpha": alpha,
        "significant": significant,
        # A null result is explicitly NOT equivalence. The protocol forbids that
        # reading, so the artifact carries the verdict word itself.
        "verdict": "difference_detected" if significant else "inconclusive",
        "equivalence_claimed": False,
        "min_unidirectional_discordance_required": required,
        "discordance_target_met": d >= required,
        "no_discordance": d == 0,
        f"{left_label}_successes": sum(left_values),
        f"{right_label}_successes": sum(right_values),
        f"{left_label}_rate": clopper_pearson(sum(left_values), n, alpha),
        f"{right_label}_rate": clopper_pearson(sum(right_values), n, alpha),
    }


def minimum_unidirectional_discordance(alpha: float = 0.05) -> int:
    """Smallest all-one-direction discordant count that can reach ``alpha``.

    With ``b = d`` and ``c = 0`` the exact two-sided p-value is
    ``min(1, 2 * 0.5**d)``, so this returns the smallest ``d`` for which that is
    at or below alpha. At alpha = 0.05 the answer is 6, which is the figure the
    preregistration quotes as the minimum required discordance.
    """
    alpha = require_alpha(alpha)
    d = 1
    while min(1.0, 2.0 * (0.5 ** d)) > alpha:
        d += 1
        if d > 4096:  # unreachable for any alpha in (0, 1); guards a typo
            raise blocked("discordance_search_diverged")
    return d


def primary_contrast(full: Sequence[bool], caption_mediated: Sequence[bool], *,
                     alpha: float = 0.05,
                     chunk_ids: Sequence[str] | None = None,
                     expected_n: int | None = PRIMARY_N_CHUNKS) -> dict[str, Any]:
    """The single confirmatory contrast, with the census size enforced.

    Defaults to requiring all 16 chunks. Passing ``expected_n=None`` is possible
    for exploratory reuse, but then the result is tagged ``census_enforced:
    False`` so a partial table can never be presented as the primary analysis.
    """
    result = exact_mcnemar(
        full,
        caption_mediated,
        alpha=alpha,
        expected_n=expected_n,
        left_ids=chunk_ids,
        right_ids=chunk_ids,
        left_label="full",
        right_label="caption_mediated",
    )
    result["contrast"] = "full_vs_caption_mediated"
    result["confirmatory"] = True
    result["census_enforced"] = expected_n is not None
    result["expected_n"] = expected_n
    return result


def paired_difference(control: Sequence[bool], perturbed: Sequence[bool], *,
                     alpha: float = 0.05,
                     item_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Paired difference in accuracy between control and perturbed conditions.

    This is the point-estimate half of ``Delta_perm``; the interval half lives in
    ``stats.bootstrap`` because the protocol requires a paired or
    document-clustered bootstrap rather than a closed-form interval. Named
    "paired perturbation sensitivity" and carries no mediation interpretation.
    """
    control_values = require_bool_sequence("control", control)
    perturbed_values = require_bool_sequence("perturbed", perturbed)
    require_same_length("control", control_values, "perturbed", perturbed_values)
    n = len(control_values)

    if item_ids is not None:
        require_unit_ids("item_ids", item_ids, n)

    deltas = [int(a) - int(b) for a, b in zip(control_values, perturbed_values)]
    return {
        "estimand": "Delta_perm",
        "interpretation": "paired_perturbation_sensitivity",
        "mediation_claim": False,
        "n": n,
        "delta_perm": math.fsum(deltas) / float(n),
        "control_correct": sum(control_values),
        "perturbed_correct": sum(perturbed_values),
        "control_rate": clopper_pearson(sum(control_values), n, alpha),
        "perturbed_rate": clopper_pearson(sum(perturbed_values), n, alpha),
        "paired_table": exact_mcnemar(
            control_values, perturbed_values, alpha=alpha,
            left_label="control", right_label="perturbed",
        ),
    }
