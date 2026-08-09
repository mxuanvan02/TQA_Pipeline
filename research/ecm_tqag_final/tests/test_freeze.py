from pathlib import Path

import pytest

from ecm_tqag.freeze import (
    build_freeze,
    plan_ledger,
    validate_execution_gate,
    validate_pre_smoke_gate,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "work_ecm24" / "ecm_inputs_final_v2.json"


def test_frozen_plan_has_complete_census_and_call_accounting():
    freeze = build_freeze(MANIFEST)
    assert freeze["census"] == {"chunks": 16, "conditions": 3, "images": 18, "packages": 48}
    rows = plan_ledger(freeze)
    assert len(rows) == 96
    assert sum(r["construction_calls"] for r in rows) == 128
    assert all(r["status"] == "PLANNED" for r in rows)
    assert all("text" not in r and "image_path" not in r for r in rows)


def test_freeze_has_role_and_input_hashes():
    freeze = build_freeze(MANIFEST)
    assert freeze["schema"] == "ecm-tqag.freeze.release"
    assert freeze["manifest_sha256"]
    assert freeze["input_fingerprints"]
    assert freeze["roles"]["generator"]["provider"] == "openrouter"
    assert freeze["roles"]["generator"]["model"] == "qwen/qwen3-vl-32b-instruct"
    assert freeze["roles"]["answerer_a"]["model"] != freeze["roles"]["generator"]["model"]
    assert set(freeze["roles"]) == {
        "generator", "answerer_a", "answerer_b", "image_auditor",
        "model_judge_a", "model_judge_b"
    }
    assert freeze["roles"]["answerer_a"]["model"] == "claude-sonnet-5"
    assert freeze["roles"]["answerer_b"]["model"] == "gpt-5.6-terra"
    assert freeze["roles"]["model_judge_a"]["model"] == "claude-sonnet-5"
    assert freeze["roles"]["model_judge_b"]["model"] == "gpt-5.6-terra"
    assert all(role["approval"] == "APPROVED" for role in freeze["roles"].values())
    assert freeze["full_call_budget"]["prior_attempts"] == 119
    assert freeze["full_call_budget"]["gross_current_freeze_base_calls"] == 486
    assert freeze["full_call_budget"]["satisfied_calls"] == 55
    assert freeze["full_call_budget"]["current_freeze_base_calls"] == 431
    assert freeze["full_call_budget"]["study_base_calls"] == 550
    assert freeze["full_call_budget"]["requested_retry_reserve"] == 55
    assert freeze["full_call_budget"]["retry_reserve"] == 0
    assert freeze["full_call_budget"]["retry_reserve_constrained_by_hard_cap"] is True
    assert freeze["full_call_budget"]["worst_case_http_calls"] == 550
    assert freeze["full_call_budget"]["current_ledger_cap"] == 431
    assert freeze["operational_http_cap"] == 550
    assert freeze["budget_within_cap"] is True
    required_sources = {
        "dry_run.py",
        "ecm_tqag/budget.py",
        "ecm_tqag/freeze.py",
        "ecm_tqag/run/ledger.py",
        "ecm_tqag/run/transport.py",
        "ecm_tqag/structure_reader.py",
        "ecm_tqag/counterfactual_images.py",
        "ecm_tqag/stats/paired.py",
    }
    assert required_sources <= set(freeze["source_sha256"])


def test_pre_smoke_gate_accepts_approved_frozen_instrument():
    freeze = build_freeze(MANIFEST)
    validate_pre_smoke_gate(freeze, execute=True)


def test_pre_smoke_gate_still_requires_explicit_execute():
    freeze = build_freeze(MANIFEST)
    with pytest.raises(ValueError, match="execute_flag_missing"):
        validate_pre_smoke_gate(freeze, execute=False)


def test_execution_gate_rejects_missing_smoke_results():
    freeze = build_freeze(MANIFEST)
    with pytest.raises(ValueError, match="smoke_missing"):
        validate_execution_gate(freeze, execute=True, smoke_passed_roles=set())


def test_execution_gate_accepts_approved_roster_after_all_smokes():
    freeze = build_freeze(MANIFEST)
    validate_execution_gate(
        freeze, execute=True, smoke_passed_roles=set(freeze["roles"])
    )


def test_execution_gate_rejects_source_hash_drift():
    freeze = build_freeze(MANIFEST)
    freeze["source_sha256"]["ecm_tqag/budget.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_hash_drift"):
        validate_execution_gate(
            freeze, execute=True, smoke_passed_roles=set(freeze["roles"])
        )


def test_missing_manifest_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="BLOCKED_INPUT_INTEGRITY"):
        build_freeze(tmp_path / "missing.json")
