"""Quadratic-weighted Cohen's kappa for model-judge agreement.

The protocol makes model-based judging exploratory, with one narrow exception:
answerability may support a secondary claim *only if* final agreement reaches
quadratic-weighted kappa >= 0.60. This module computes that statistic and
returns the eligibility decision as data, so the release artifact records the
gate rather than leaving it to prose.

Quadratic weights are used because judge ratings are ordinal: disagreeing by one
level is not the same defect as disagreeing by three. Weights are

    w[i][j] = 1 - ((i - j) ** 2) / ((k - 1) ** 2)

and kappa = (observed_weighted_agreement - expected) / (1 - expected), with the
expected table formed from the outer product of the marginals.

Degenerate case handled explicitly: if both judges use exactly one identical
category for every item, the expected agreement is 1 and kappa is undefined
(0/0). That fails closed instead of returning a fabricated 1.0 or 0.0, because
"both judges said the same thing every time" carries no information about their
ability to discriminate.
"""
from __future__ import annotations

from typing import Any, Sequence

from .validate import (
    blocked,
    require_positive_int,
    require_same_length,
    require_sequence,
    require_unit_ids,
)

# Answerability is the only model-judge dimension eligible for a secondary claim.
KAPPA_CLAIM_THRESHOLD = 0.60
CLAIM_ELIGIBLE_DIMENSION = "answerability"


def _require_rating_sequence(name: str, value: Any, n_categories: int) -> list[int]:
    """Ratings must be integers on ``0..n_categories-1``.

    ``bool`` is refused even though it is an int subclass: a boolean rating
    silently implies a 2-category scale, which would change the weight matrix.
    """
    items = require_sequence(name, value)
    if not items:
        raise blocked(f"{name}_empty")
    out: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise blocked(f"{name}_not_int:{index}")
        if not 0 <= item < n_categories:
            raise blocked(f"{name}_out_of_range:{index}:{item}")
        out.append(item)
    return out


def quadratic_weight(i: int, j: int, n_categories: int) -> float:
    """Quadratic disagreement weight on an ordinal scale."""
    if n_categories < 2:
        raise blocked("n_categories_lt_2")
    span = float((n_categories - 1) ** 2)
    return 1.0 - ((float(i - j) ** 2) / span)


def quadratic_weighted_kappa(rater_a: Sequence[int], rater_b: Sequence[int], *,
                             n_categories: int,
                             item_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Quadratic-weighted Cohen's kappa between two judges.

    Returns the coefficient plus the observed/expected weighted agreements and
    the confusion table, so a reviewer can recompute the number by hand from the
    artifact alone.
    """
    n_categories = require_positive_int("n_categories", n_categories)
    if n_categories < 2:
        raise blocked("n_categories_lt_2")

    a = _require_rating_sequence("rater_a", rater_a, n_categories)
    b = _require_rating_sequence("rater_b", rater_b, n_categories)
    require_same_length("rater_a", a, "rater_b", b)
    n = len(a)

    if item_ids is not None:
        require_unit_ids("item_ids", item_ids, n)

    observed = [[0 for _ in range(n_categories)] for _ in range(n_categories)]
    for x, y in zip(a, b):
        observed[x][y] += 1

    marg_a = [sum(row) for row in observed]
    marg_b = [sum(observed[i][j] for i in range(n_categories))
              for j in range(n_categories)]

    num = 0.0
    den = 0.0
    for i in range(n_categories):
        for j in range(n_categories):
            w = quadratic_weight(i, j, n_categories)
            expected_ij = (marg_a[i] * marg_b[j]) / float(n)
            num += w * observed[i][j]
            den += w * expected_ij

    observed_weighted = num / float(n)
    expected_weighted = den / float(n)

    denominator = 1.0 - expected_weighted
    if abs(denominator) < 1e-12:
        # Both marginals collapsed onto one shared category: kappa is 0/0.
        return {
            "metric": "quadratic_weighted_kappa",
            "n_items": n,
            "n_categories": n_categories,
            "kappa": None,
            "defined": False,
            "reason": "expected_agreement_is_one",
            "observed_weighted_agreement": observed_weighted,
            "expected_weighted_agreement": expected_weighted,
            "confusion": observed,
            "marginal_a": marg_a,
            "marginal_b": marg_b,
            "n_exact_agreements": sum(observed[i][i] for i in range(n_categories)),
        }

    kappa = (observed_weighted - expected_weighted) / denominator
    return {
        "metric": "quadratic_weighted_kappa",
        "n_items": n,
        "n_categories": n_categories,
        "kappa": kappa,
        "defined": True,
        "reason": None,
        "observed_weighted_agreement": observed_weighted,
        "expected_weighted_agreement": expected_weighted,
        "confusion": observed,
        "marginal_a": marg_a,
        "marginal_b": marg_b,
        "n_exact_agreements": sum(observed[i][i] for i in range(n_categories)),
    }


def judge_agreement_report(rater_a: Sequence[int], rater_b: Sequence[int], *,
                           n_categories: int,
                           dimension: str,
                           item_ids: Sequence[str] | None = None,
                           threshold: float = KAPPA_CLAIM_THRESHOLD) -> dict[str, Any]:
    """Kappa plus the preregistered secondary-claim eligibility decision.

    Eligibility requires all three conditions, evaluated fail-closed:
      * kappa is defined,
      * kappa >= threshold (0.60),
      * the dimension is ``answerability``.

    Any other dimension stays exploratory no matter how high its kappa is, which
    is what the protocol says, and an undefined kappa is never eligible.
    """
    if not isinstance(dimension, str) or not dimension.strip():
        raise blocked("dimension_invalid")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise blocked("threshold_not_numeric")
    threshold = float(threshold)
    if not -1.0 <= threshold <= 1.0:
        raise blocked("threshold_out_of_range")

    result = quadratic_weighted_kappa(rater_a, rater_b,
                                      n_categories=n_categories,
                                      item_ids=item_ids)
    kappa = result["kappa"]
    meets = bool(result["defined"] and kappa is not None and kappa >= threshold)
    dimension_eligible = dimension == CLAIM_ELIGIBLE_DIMENSION

    result.update({
        "dimension": dimension,
        "threshold": threshold,
        "meets_threshold": meets,
        "dimension_claim_eligible": dimension_eligible,
        "secondary_claim_eligible": bool(meets and dimension_eligible),
        "status": "exploratory" if not (meets and dimension_eligible) else "secondary_claim_eligible",
    })
    return result
