from __future__ import annotations

from ecm_tqag.budget import full_call_plan
from ecm_tqag.protocol import JUDGING_ARMS, fixed_judging_frame


def test_final_budget_counts_prior_and_current_smokes() -> None:
    plan = full_call_plan()
    phases = {row["name"]: row["base_calls"] for row in plan["phases"]}
    assert phases == {
        "role_smoke_current_freeze": 6,
        "structure_and_caption_extraction": 54,
        "construction": 128,
        "sensitivity_floor": 40,
        "full_item_secondary_probes": 160,
        "image_audit": 18,
        "blinded_model_judging": 80,
    }
    assert plan["prior_attempts"] == 6
    assert plan["current_freeze_base_calls"] == 486
    assert plan["study_base_calls"] == 492
    assert plan["retry_reserve"] == 50
    assert plan["study_worst_case_http_calls"] == 542
    assert plan["current_ledger_cap"] == 536
    assert plan["operational_http_cap"] == 550


def test_secondary_budget_is_scoped_to_full_arm_items_only() -> None:
    plan = full_call_plan(maximum_eligible_full_items=4)
    phases = {row["name"]: row["base_calls"] for row in plan["phases"]}
    assert phases["full_item_secondary_probes"] == 40
    assert plan["study_base_calls"] == 372
    assert plan["retry_reserve"] == 38
    assert plan["study_worst_case_http_calls"] == 410
    assert plan["current_ledger_cap"] == 404


def test_fixed_judging_frame_is_deterministic_arm_blinded_and_balanced() -> None:
    candidates = [
        {"item_id": f"{arm}-c{i:02d}", "arm": arm, "chunk_token": f"c{i:02d}"}
        for arm in JUDGING_ARMS for i in range(16)
    ]
    a = fixed_judging_frame(candidates, freeze_sha256="a" * 64)
    b = fixed_judging_frame(list(reversed(candidates)), freeze_sha256="a" * 64)
    assert a == b
    assert len(a["private_frame"]) == 40
    assert len(a["blinded_frame"]) == 40
    assert set(row["arm"] for row in a["private_frame"]) == set(JUDGING_ARMS)
    assert {arm: sum(row["arm"] == arm for row in a["private_frame"])
            for arm in JUDGING_ARMS} == {arm: 8 for arm in JUDGING_ARMS}
    assert all(set(row) == {"judge_item_id", "item_payload_sha256"}
               for row in a["blinded_frame"])
    assert all("arm" not in row and "chunk_token" not in row
               for row in a["blinded_frame"])


def test_fixed_judging_frame_fails_closed_on_pool_shortfall_or_duplicate() -> None:
    candidates = [
        {"item_id": f"{arm}-c{i:02d}", "arm": arm, "chunk_token": f"c{i:02d}"}
        for arm in JUDGING_ARMS for i in range(16)
    ]
    import pytest
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:judging_pool"):
        fixed_judging_frame(candidates[:-1], freeze_sha256="b" * 64)
    duplicate = candidates[:-1] + [dict(candidates[0])]
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:duplicate_item_id"):
        fixed_judging_frame(duplicate, freeze_sha256="b" * 64)


def test_budget_subtracts_only_verified_satisfied_calls_from_remaining_base() -> None:
    plan = full_call_plan(prior_attempts=71, satisfied_calls=25)
    assert plan["gross_current_freeze_base_calls"] == 486
    assert plan["satisfied_calls"] == 25
    assert plan["current_freeze_base_calls"] == 461
    assert plan["study_base_calls"] == 532
    assert plan["retry_reserve"] == 18
    assert plan["study_worst_case_http_calls"] == 550
    assert plan["current_ledger_cap"] == 479
