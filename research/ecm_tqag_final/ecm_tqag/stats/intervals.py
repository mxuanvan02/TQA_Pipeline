"""Exact Clopper--Pearson binomial intervals (pure stdlib, no approximations).

The protocol requires *exact* intervals, so this module does not use a normal,
Wilson, or Agresti--Coull approximation anywhere.

Clopper--Pearson is defined by inverting the exact binomial tails:

    lower = p such that  P(X >= k | n, p) = alpha / 2      (0 when k == 0)
    upper = p such that  P(X <= k | n, p) = alpha / 2      (1 when k == n)

Both tails are computed as exact finite sums with ``math.comb``, and the
inversion is a monotone bisection. For the corpus size in this protocol (n = 16)
the sums are tiny and carry no series-truncation error, which is why this is
preferred over an incomplete-beta-inverse implementation: there is nothing to
converge, only a monotone root to bracket.

``P(X >= k | p)`` is non-decreasing in p and ``P(X <= k | p)`` is non-increasing
in p, so bisection is guaranteed to bracket a unique root.
"""
from __future__ import annotations

import math

from .validate import blocked, require_alpha

# Bisection budget. Each step halves a bracket of initial width 1, so 200 steps
# is far below double precision resolution; the loop exits on width first.
_MAX_BISECT_STEPS = 200
_BISECT_TOL = 1e-15


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Exact binomial point mass, guarding the 0**0 corner cases."""
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_cdf(n: int, k: int, p: float) -> float:
    """Exact ``P(X <= k)`` as a finite sum; clamped to [0, 1] for round-off."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = math.fsum(binomial_pmf(n, i, p) for i in range(0, k + 1))
    return min(1.0, max(0.0, total))


def binomial_sf_ge(n: int, k: int, p: float) -> float:
    """Exact upper tail ``P(X >= k)`` as a finite sum (not 1 - CDF).

    Summing the upper tail directly avoids catastrophic cancellation when the
    tail is tiny, which matters for the small alpha/2 targets used here.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = math.fsum(binomial_pmf(n, i, p) for i in range(k, n + 1))
    return min(1.0, max(0.0, total))


def _bisect_increasing(fn, target: float) -> float:
    """Solve fn(p) = target on [0, 1] for non-decreasing fn."""
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_BISECT_STEPS):
        if hi - lo < _BISECT_TOL:
            break
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _bisect_decreasing(fn, target: float) -> float:
    """Solve fn(p) = target on [0, 1] for non-increasing fn."""
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_BISECT_STEPS):
        if hi - lo < _BISECT_TOL:
            break
        mid = 0.5 * (lo + hi)
        if fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> dict:
    """Exact two-sided Clopper--Pearson interval for a binomial proportion.

    Returns the point estimate, the interval, and enough provenance for the
    release artifact to be self-describing. ``n == 0`` fails closed rather than
    returning ``[0, 1]``: an empty denominator is a missing cell, not an
    uninformative observation.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise blocked("n_not_int")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise blocked("successes_not_int")
    if n <= 0:
        raise blocked("n_not_positive")
    if not 0 <= successes <= n:
        raise blocked(f"successes_out_of_range:{successes}/{n}")
    alpha = require_alpha(alpha)

    half = alpha / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lower = _bisect_increasing(lambda p: binomial_sf_ge(n, successes, p), half)
    if successes == n:
        upper = 1.0
    else:
        upper = _bisect_decreasing(lambda p: binomial_cdf(n, successes, p), half)

    # Numerical hygiene: the bisection midpoint can land a hair outside the
    # point estimate on the boundary cases, which would look like an interval
    # that excludes its own estimate.
    point = successes / n
    lower = min(lower, point)
    upper = max(upper, point)

    return {
        "method": "clopper_pearson_exact",
        "successes": successes,
        "n": n,
        "point": point,
        "alpha": alpha,
        "conf_level": 1.0 - alpha,
        "lower": lower,
        "upper": upper,
        "interval": [lower, upper],
    }


def proportion_report(successes: int, n: int, alpha: float = 0.05,
                      *, label: str | None = None) -> dict:
    """Clopper--Pearson interval tagged with a human-readable label."""
    out = clopper_pearson(successes, n, alpha)
    if label is not None:
        if not isinstance(label, str) or not label.strip():
            raise blocked("label_invalid")
        out = {"label": label, **out}
    return out
