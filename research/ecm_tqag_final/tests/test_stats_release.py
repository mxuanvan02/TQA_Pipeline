from __future__ import annotations

import math

import pytest

from ecm_tqag.stats.agreement import judge_agreement_report, quadratic_weighted_kappa
from ecm_tqag.stats.aggregate import aggregate_repeats, paired_aggregate
from ecm_tqag.stats.bootstrap import delta_perm_interval, paired_bootstrap
from ecm_tqag.stats.intervals import clopper_pearson
from ecm_tqag.stats.paired import exact_mcnemar, primary_contrast
from ecm_tqag.stats.report import build_primary_statistics, verify_digest
from ecm_tqag.stats.sensitivity import family_sensitivity, sensitivity_floor


def test_clopper_pearson_known_reference_values() -> None:
    zero = clopper_pearson(0, 10)
    half = clopper_pearson(5, 10)
    all_ = clopper_pearson(10, 10)
    assert zero["lower"] == 0.0
    assert zero["upper"] == pytest.approx(0.3084971078187608, abs=1e-12)
    assert half["lower"] == pytest.approx(0.1870860284473985, abs=1e-12)
    assert half["upper"] == pytest.approx(0.8129139715526015, abs=1e-12)
    assert all_["lower"] == pytest.approx(0.6915028921812392, abs=1e-12)
    assert all_["upper"] == 1.0


def test_exact_mcnemar_keeps_legacy_and_release_fields() -> None:
    left = [True] + [True] * 4 + [False] * 3 + [False] * 8
    right = [True] + [False] * 4 + [True] * 3 + [False] * 8
    result = exact_mcnemar(left, right)
    assert result["n"] == result["n_pairs"] == 16
    assert result["left_only"] == 4
    assert result["right_only"] == 3
    assert result["discordant"] == 7
    assert result["p_two_sided_exact"] == 1.0
    assert result["equivalence_claimed"] is False


def test_primary_contrast_fails_closed_on_partial_or_misaligned_census() -> None:
    ids = [f"c{i:02d}" for i in range(16)]
    result = primary_contrast([True] * 16, [False] * 16, chunk_ids=ids)
    assert result["n"] == 16
    with pytest.raises(ValueError, match="BLOCKED_STATS:n_pairs_mismatch"):
        primary_contrast([True] * 15, [False] * 15, chunk_ids=ids[:15])
    with pytest.raises(ValueError, match="BLOCKED_STATS:pairing_misaligned"):
        exact_mcnemar([True, False], [False, True], left_ids=["a", "b"], right_ids=["b", "a"])


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    control = [1.0, 1.0, 0.0, 1.0]
    perturbed = [0.0, 1.0, 0.0, 0.0]
    a = paired_bootstrap(control, perturbed, seed=17, n_resamples=500)
    b = paired_bootstrap(control, perturbed, seed=17, n_resamples=500)
    assert a == b
    assert a["estimate"] == 0.5
    assert a["pairing_preserved"] is True
    assert a["n_units"] == 4


def test_repeats_are_collapsed_with_no_sample_size_inflation() -> None:
    records = [
        {"item_id": "a", "correct": True},
        {"item_id": "a", "correct": False},
        {"item_id": "b", "correct": True},
        {"item_id": "b", "correct": True},
    ]
    agg = aggregate_repeats(records)
    assert agg["n_repeat_records"] == 4
    assert agg["effective_n"] == 2
    assert agg["sample_size_inflation"] is False
    paired = paired_aggregate(records, list(reversed(records)))
    assert paired["effective_n"] == 2
    assert paired["item_ids"] == ["a", "b"]


def test_quadratic_weighted_kappa_reference_and_degenerate_case() -> None:
    perfect = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], n_categories=4)
    assert perfect["defined"] is True
    assert perfect["kappa"] == pytest.approx(1.0)
    degenerate = quadratic_weighted_kappa([2, 2, 2], [2, 2, 2], n_categories=4)
    assert degenerate["defined"] is False
    assert degenerate["kappa"] is None
    exploratory = judge_agreement_report([0, 1, 2, 3], [0, 1, 2, 3], n_categories=4, dimension="style")
    assert exploratory["secondary_claim_eligible"] is False


def _family(name: str, *, separation: int = 8, disagreements: int = 1) -> dict:
    controls = [True] * separation + [False] * (10 - separation)
    replicate = [False] * disagreements + [True] * (10 - disagreements)
    return family_sensitivity(controls[:5], controls[5:], replicate, family=name)


def test_sensitivity_floor_is_conjunctive_and_suppresses_delta_perm() -> None:
    passed = sensitivity_floor([_family("a"), _family("b", separation=10, disagreements=0)])
    assert passed["passed"] is True
    failed = sensitivity_floor([_family("a"), _family("b", separation=7, disagreements=0)])
    assert failed["passed"] is False
    suppressed = delta_perm_interval([1.0, 0.0], [0.0, 0.0], seed=3, sensitivity_floor_passed=False)
    assert suppressed == {
        "estimand": "Delta_perm",
        "reported": False,
        "suppressed_reason": "sensitivity_floor_failed",
        "interval": None,
        "estimate": None,
    }


def test_release_statistics_digest_and_floor_guard() -> None:
    full = [True] * 8 + [False] * 8
    caption = [False] * 8 + [True] * 8
    ids = [f"c{i:02d}" for i in range(16)]
    artifact = build_primary_statistics(full=full, caption_mediated=caption, chunk_ids=ids)
    assert verify_digest(artifact)
    with pytest.raises(ValueError, match="BLOCKED_STATS:delta_perm_reported_without_sensitivity_floor"):
        build_primary_statistics(
            full=full,
            caption_mediated=caption,
            chunk_ids=ids,
            sensitivity={"passed": False},
            delta_perm={"reported": True, "estimate": 0.1},
        )


def test_statistics_reject_non_boolean_primary_cells() -> None:
    with pytest.raises(ValueError, match="BLOCKED_STATS:left_not_boolean"):
        exact_mcnemar([1, 0], [True, False])  # type: ignore[list-item]
