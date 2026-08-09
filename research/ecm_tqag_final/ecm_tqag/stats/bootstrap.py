"""Paired and document-clustered bootstrap intervals for ``Delta_perm``.

The protocol requires "exact paired or document-clustered bootstrap intervals"
for the paired perturbation sensitivity, so this module provides both and forces
the caller to choose explicitly:

* ``paired_bootstrap`` resamples *paired units* with replacement. The pairing is
  never broken -- each draw takes a whole (control, perturbed) pair -- because
  resampling the two arms independently would destroy the within-item
  correlation the paired design exists to exploit.
* ``clustered_bootstrap`` resamples *documents* (textbooks/chunks) with
  replacement and takes every item inside each drawn cluster. Items from the same
  document are not independent, so an item-level interval would be
  anticonservative. This is the block bootstrap for clustered data.

Determinism: ``seed`` is a required keyword everywhere. There is no default and
no fallback to system entropy, so a published interval can always be replayed
bit-for-bit. Resampling uses ``random.Random(seed)`` (Mersenne Twister, stable
across Python versions for a given seed and call sequence) and draws indices via
an explicit ``randrange`` loop rather than ``random.choices``, whose internal
implementation is not a documented stability guarantee.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

from .validate import (
    blocked,
    require_alpha,
    require_positive_int,
    require_same_length,
    require_seed,
    require_sequence,
    require_unit_sequence,
)

DEFAULT_RESAMPLES = 10000


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted sample.

    ``q`` is a fraction in [0, 1]. This is the same convention as the standard
    "type 7" quantile, chosen because it is the common default and is stable for
    the large resample counts used here.
    """
    if not sorted_values:
        raise blocked("percentile_of_empty_sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[int(position)])
    weight = position - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def _draw_indices(rng: random.Random, size: int, n: int) -> list[int]:
    """Draw ``size`` indices in [0, n) with replacement, deterministically."""
    return [rng.randrange(n) for _ in range(size)]


def _summarise(replicates: list[float], observed: float, alpha: float,
               *, method: str, n_units: int, seed: int,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ordered = sorted(replicates)
    lower = _percentile(ordered, alpha / 2.0)
    upper = _percentile(ordered, 1.0 - alpha / 2.0)
    mean = math.fsum(ordered) / float(len(ordered))
    if len(ordered) > 1:
        variance = math.fsum((value - mean) ** 2 for value in ordered) / float(len(ordered) - 1)
    else:
        variance = 0.0
    out: dict[str, Any] = {
        "method": method,
        "estimate": observed,
        "n_units": n_units,
        "n_resamples": len(ordered),
        "seed": seed,
        "alpha": alpha,
        "conf_level": 1.0 - alpha,
        "interval_type": "percentile",
        "lower": lower,
        "upper": upper,
        "interval": [lower, upper],
        "bootstrap_mean": mean,
        "bootstrap_bias": mean - observed,
        "bootstrap_se": math.sqrt(variance),
        # Whether the interval covers "no sensitivity". Reported rather than
        # converted into a p-value: the protocol asks for an interval here.
        "excludes_zero": not (lower <= 0.0 <= upper),
        "deterministic": True,
    }
    if extra:
        out.update(extra)
    return out


def paired_bootstrap(control: Sequence[float], perturbed: Sequence[float], *,
                     seed: int,
                     alpha: float = 0.05,
                     n_resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Percentile bootstrap CI for the paired mean difference.

    Resamples the vector of *within-unit differences*, which is the paired
    bootstrap: unit ``i`` contributes ``control[i] - perturbed[i]`` as one
    inseparable observation.

    Inputs are unit-level values in [0, 1] so that already-aggregated per-item
    proportions (see ``stats.aggregate``) can be passed directly without
    re-inflating the sample with repeats.
    """
    seed = require_seed(seed)
    alpha = require_alpha(alpha)
    n_resamples = require_positive_int("n_resamples", n_resamples)
    control_values = require_unit_sequence("control", control)
    perturbed_values = require_unit_sequence("perturbed", perturbed)
    require_same_length("control", control_values, "perturbed", perturbed_values)

    deltas = [a - b for a, b in zip(control_values, perturbed_values)]
    n = len(deltas)
    if n < 2:
        # A one-unit bootstrap resamples the same value forever and returns a
        # zero-width interval that looks like infinite precision.
        raise blocked(f"insufficient_units_for_bootstrap:{n}")

    observed = math.fsum(deltas) / float(n)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_resamples):
        idx = _draw_indices(rng, n, n)
        replicates.append(math.fsum(deltas[i] for i in idx) / float(n))

    return _summarise(
        replicates, observed, alpha,
        method="paired_percentile_bootstrap", n_units=n, seed=seed,
        extra={
            "estimand": "Delta_perm",
            "resample_unit": "paired_item",
            "pairing_preserved": True,
        },
    )


def clustered_bootstrap(control: Sequence[float], perturbed: Sequence[float],
                        clusters: Sequence[str], *,
                        seed: int,
                        alpha: float = 0.05,
                        n_resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Document-clustered (block) percentile bootstrap for the paired difference.

    Whole clusters are drawn with replacement and every item inside a drawn
    cluster is taken, so within-document dependence is preserved. The replicate
    statistic is the unweighted mean over the items in the resampled collection,
    matching the observed statistic's definition.
    """
    seed = require_seed(seed)
    alpha = require_alpha(alpha)
    n_resamples = require_positive_int("n_resamples", n_resamples)
    control_values = require_unit_sequence("control", control)
    perturbed_values = require_unit_sequence("perturbed", perturbed)
    require_same_length("control", control_values, "perturbed", perturbed_values)

    cluster_labels = require_sequence("clusters", clusters)
    if len(cluster_labels) != len(control_values):
        raise blocked(
            f"clusters_length_mismatch:{len(cluster_labels)}!={len(control_values)}")
    for index, label in enumerate(cluster_labels):
        if not isinstance(label, str) or not label.strip():
            raise blocked(f"cluster_label_invalid:{index}")

    deltas = [a - b for a, b in zip(control_values, perturbed_values)]

    # Group in first-appearance order so the resampling sequence is a function of
    # the input order alone, never of dict/set iteration chance.
    order: list[str] = []
    grouped: dict[str, list[float]] = {}
    for label, delta in zip(cluster_labels, deltas):
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        grouped[label].append(delta)

    n_clusters = len(order)
    if n_clusters < 2:
        raise blocked(f"insufficient_clusters_for_bootstrap:{n_clusters}")

    observed = math.fsum(deltas) / float(len(deltas))
    blocks = [grouped[label] for label in order]

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_resamples):
        idx = _draw_indices(rng, n_clusters, n_clusters)
        pooled: list[float] = []
        for i in idx:
            pooled.extend(blocks[i])
        replicates.append(math.fsum(pooled) / float(len(pooled)))

    return _summarise(
        replicates, observed, alpha,
        method="document_clustered_percentile_bootstrap",
        n_units=len(deltas), seed=seed,
        extra={
            "estimand": "Delta_perm",
            "resample_unit": "document_cluster",
            "pairing_preserved": True,
            "n_clusters": n_clusters,
            "cluster_sizes": {label: len(grouped[label]) for label in order},
        },
    )


def delta_perm_interval(control: Sequence[float], perturbed: Sequence[float], *,
                        seed: int,
                        clusters: Sequence[str] | None = None,
                        alpha: float = 0.05,
                        n_resamples: int = DEFAULT_RESAMPLES,
                        sensitivity_floor_passed: bool) -> dict[str, Any]:
    """Protocol entry point for a ``Delta_perm`` interval.

    Two protocol rules are enforced here rather than trusted to the caller:

    1. If the sensitivity floor failed, **no** ``Delta_perm`` estimate is
       reported. The function returns a suppressed record instead of a number,
       so an insensitive probe cannot leak an estimate into the artifact.
    2. Clustering is used when cluster labels are supplied, and the choice is
       recorded in the output.
    """
    if not isinstance(sensitivity_floor_passed, bool):
        raise blocked("sensitivity_floor_passed_not_boolean")
    if not sensitivity_floor_passed:
        return {
            "estimand": "Delta_perm",
            "reported": False,
            "suppressed_reason": "sensitivity_floor_failed",
            "interval": None,
            "estimate": None,
        }
    if clusters is None:
        result = paired_bootstrap(control, perturbed, seed=seed, alpha=alpha,
                                  n_resamples=n_resamples)
    else:
        result = clustered_bootstrap(control, perturbed, clusters, seed=seed,
                                     alpha=alpha, n_resamples=n_resamples)
    result["reported"] = True
    result["suppressed_reason"] = None
    return result


def bootstrap_statistic(values: Sequence[float], statistic: Callable[[Sequence[float]], float],
                        *, seed: int, alpha: float = 0.05,
                        n_resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Generic deterministic percentile bootstrap for a scalar statistic.

    Provided for exploratory/diagnostic quantities so they do not grow their own
    ad-hoc resampling loops with their own seeding habits.
    """
    seed = require_seed(seed)
    alpha = require_alpha(alpha)
    n_resamples = require_positive_int("n_resamples", n_resamples)
    if not callable(statistic):
        raise blocked("statistic_not_callable")
    items = require_unit_sequence("values", values)
    n = len(items)
    if n < 2:
        raise blocked(f"insufficient_units_for_bootstrap:{n}")

    observed = float(statistic(items))
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_resamples):
        idx = _draw_indices(rng, n, n)
        replicates.append(float(statistic([items[i] for i in idx])))

    return _summarise(replicates, observed, alpha,
                      method="percentile_bootstrap", n_units=n, seed=seed)
