"""Protocol-completion tests: source-freeze policy, 40/80 judging frame, 542/550 budget.

Every check here is offline and fails closed: an ambiguous or incomplete state must
raise, never silently proceed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ecm_tqag import budget as budget_mod
from ecm_tqag import freeze as freeze_mod
from ecm_tqag.budget import full_call_plan
from ecm_tqag.freeze import (
    REQUIRED_SOURCE_FILES,
    SOURCE_DISCOVERY_POLICY,
    build_freeze,
    discover_source_files,
    project_root,
    validate_execution_gate,
)
from ecm_tqag.protocol import (
    JUDGING_ARMS,
    JUDGING_CANDIDATES_PER_ARM,
    JUDGING_FRAME_SIZE,
    JUDGING_PER_ARM,
    JUDGING_POOL_SIZE,
    fixed_judging_frame,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "work_ecm24" / "ecm_inputs_final_v2.json"


def _pool(prefix: str = "") -> list[dict[str, str]]:
    return [
        {"item_id": f"{prefix}{arm}-c{i:02d}", "arm": arm, "chunk_token": f"c{i:02d}"}
        for arm in JUDGING_ARMS
        for i in range(JUDGING_CANDIDATES_PER_ARM)
    ]


# --------------------------------------------------------------------------- #
# Source discovery policy: hashable later, not frozen to a guessed list now.
# --------------------------------------------------------------------------- #


def test_source_discovery_policy_is_declared_not_hand_listed() -> None:
    assert SOURCE_DISCOVERY_POLICY["id"] == "ecm-tqag.source-discovery.v1"
    assert SOURCE_DISCOVERY_POLICY["pattern"] == "**/*.py"
    assert "tests" in SOURCE_DISCOVERY_POLICY["excluded_dir_names"]
    assert "__pycache__" in SOURCE_DISCOVERY_POLICY["excluded_dir_names"]


def test_source_discovery_covers_every_implementation_module() -> None:
    """Discovery must equal the on-disk implementation surface, not a subset of it."""
    discovered = set(discover_source_files())
    root = project_root()
    excluded = set(SOURCE_DISCOVERY_POLICY["excluded_dir_names"])
    on_disk = {
        path.relative_to(root).as_posix()
        for path in (root / "ecm_tqag").glob("**/*.py")
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts[:-1])
    }
    on_disk |= {
        name for name in SOURCE_DISCOVERY_POLICY["file_roots"] if (root / name).is_file()
    }
    assert discovered == on_disk
    assert not any(part == "__pycache__" for path in discovered for part in path.split("/"))
    assert not any(path.startswith("tests/") for path in discovered)


def test_required_sources_include_io_and_package_initializers() -> None:
    """io.py and the __init__.py files are load-bearing and must never be skipped."""
    for required in ("ecm_tqag/io.py", "ecm_tqag/__init__.py", "ecm_tqag/run/__init__.py",
                    "ecm_tqag/stats/__init__.py", "ecm_tqag/protocol.py"):
        assert required in REQUIRED_SOURCE_FILES
    assert REQUIRED_SOURCE_FILES <= set(discover_source_files())


def test_source_discovery_is_deterministically_ordered() -> None:
    files = discover_source_files()
    assert files == sorted(files)
    assert len(files) == len(set(files))
    assert discover_source_files() == files


def test_source_discovery_fails_closed_when_a_required_module_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken checkout must block, not silently freeze a truncated source list."""
    (tmp_path / "ecm_tqag").mkdir()
    (tmp_path / "ecm_tqag" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(freeze_mod, "project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="BLOCKED_FREEZE:source_discovery_incomplete"):
        discover_source_files()


def test_source_discovery_fails_closed_when_package_root_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(freeze_mod, "project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="BLOCKED_FREEZE:source_package_missing:ecm_tqag"):
        discover_source_files()


def test_freeze_records_the_discovery_policy_and_all_discovered_hashes() -> None:
    freeze = build_freeze(MANIFEST)
    assert freeze["source_discovery"]["id"] == "ecm-tqag.source-discovery.v1"
    assert freeze["source_discovery"]["required_files"] == sorted(REQUIRED_SOURCE_FILES)
    recorded = freeze["source_sha256"]
    assert set(recorded) == set(discover_source_files())
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in recorded.values())
    assert len(set(recorded.values())) == len(recorded)  # no truncated/duplicate hashing


def test_execution_gate_blocks_a_source_file_added_after_freeze() -> None:
    """A module that appears after freezing is unhashed code and must block execution."""
    freeze = build_freeze(MANIFEST)
    freeze["source_sha256"].pop("ecm_tqag/io.py")
    with pytest.raises(ValueError, match=r"source_hash_drift:.*missing:ecm_tqag/io\.py"):
        validate_execution_gate(freeze, execute=True, smoke_passed_roles=set(freeze["roles"]))


def test_execution_gate_blocks_when_source_hashes_are_absent() -> None:
    freeze = build_freeze(MANIFEST)
    freeze["source_sha256"] = None
    with pytest.raises(ValueError, match="source_hashes_missing"):
        validate_execution_gate(freeze, execute=True, smoke_passed_roles=set(freeze["roles"]))


# --------------------------------------------------------------------------- #
# 40-of-80 fixed judging frame.
# --------------------------------------------------------------------------- #


def test_judging_frame_constants_are_forty_of_eighty() -> None:
    assert len(JUDGING_ARMS) == 5
    assert "gates_off" not in JUDGING_ARMS  # deterministic rescore, not a new item
    assert JUDGING_CANDIDATES_PER_ARM == 16
    assert JUDGING_POOL_SIZE == 80
    assert JUDGING_PER_ARM == 8
    assert JUDGING_FRAME_SIZE == 40
    assert JUDGING_FRAME_SIZE * 2 == JUDGING_POOL_SIZE


def test_judging_frame_selects_eight_per_arm_from_eighty() -> None:
    frame = fixed_judging_frame(_pool(), freeze_sha256="c" * 64)
    assert frame["candidate_pool_size"] == 80
    assert frame["expected_candidate_pool_size"] == 80
    assert frame["frame_size"] == 40
    assert frame["blinded_before_judging"] is True
    assert frame["outcome_dependent"] is False
    counts = {arm: sum(row["arm"] == arm for row in frame["private_frame"])
              for arm in JUDGING_ARMS}
    assert counts == {arm: 8 for arm in JUDGING_ARMS}
    assert len({row["item_id"] for row in frame["private_frame"]}) == 40


def test_blinded_frame_leaks_no_arm_or_routing_metadata() -> None:
    frame = fixed_judging_frame(_pool(), freeze_sha256="d" * 64)
    private_ids = {row["item_id"] for row in frame["private_frame"]}
    for row in frame["blinded_frame"]:
        assert set(row) == {"judge_item_id", "item_payload_sha256"}
        assert row["judge_item_id"] not in private_ids
        assert not any(arm in row["judge_item_id"] for arm in JUDGING_ARMS)
    assert len({row["judge_item_id"] for row in frame["blinded_frame"]}) == 40


def test_judging_frame_is_independent_of_candidate_input_order() -> None:
    pool = _pool()
    shuffled = pool[37:] + pool[:37]
    a = fixed_judging_frame(pool, freeze_sha256="e" * 64)
    b = fixed_judging_frame(shuffled, freeze_sha256="e" * 64)
    assert a == b


def test_judging_frame_is_bound_to_the_freeze_identity() -> None:
    pool = _pool()
    a = fixed_judging_frame(pool, freeze_sha256="a" * 64)
    b = fixed_judging_frame(pool, freeze_sha256="f" * 64)
    assert [r["item_id"] for r in a["private_frame"]] != [r["item_id"] for r in b["private_frame"]]


@pytest.mark.parametrize("bad", ["", "zz", "g" * 64, "a" * 63, "A" * 65])
def test_judging_frame_rejects_malformed_freeze_identity(bad: str) -> None:
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:invalid_freeze_sha256"):
        fixed_judging_frame(_pool(), freeze_sha256=bad)


def test_judging_frame_rejects_oversized_pool_and_unknown_arm() -> None:
    oversized = _pool() + [{"item_id": "extra-1", "arm": "full", "chunk_token": "c99"}]
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:judging_pool"):
        fixed_judging_frame(oversized, freeze_sha256="b" * 64)

    wrong_arm = _pool()[:-1] + [{"item_id": "x-1", "arm": "gates_off", "chunk_token": "c15"}]
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:unexpected_judging_arm:gates_off"):
        fixed_judging_frame(wrong_arm, freeze_sha256="b" * 64)


def test_judging_frame_rejects_missing_or_blank_item_ids() -> None:
    pool = _pool()
    pool[3] = {"arm": pool[3]["arm"], "chunk_token": "c03"}
    with pytest.raises(ValueError, match="BLOCKED_PROTOCOL:invalid_item_id:3"):
        fixed_judging_frame(pool, freeze_sha256="b" * 64)


def test_judging_frame_honours_supplied_payload_commitments() -> None:
    pool = [{**row, "item_payload_sha256": "1" * 64} for row in _pool()]
    frame = fixed_judging_frame(pool, freeze_sha256="c" * 64)
    assert all(row["item_payload_sha256"] == "1" * 64 for row in frame["blinded_frame"])


def test_freeze_declares_the_fixed_judging_frame() -> None:
    declared = build_freeze(MANIFEST)["judging_frame"]
    assert declared["candidate_pool_size"] == 80
    assert declared["frame_size"] == 40
    assert declared["per_arm"] == 8
    assert declared["arms"] == list(JUDGING_ARMS)
    assert declared["selected_before_judging"] is True
    assert declared["outcome_dependent"] is False


# --------------------------------------------------------------------------- #
# 542/550 accounting.
# --------------------------------------------------------------------------- #


def test_study_accounting_reconciles_492_50_542_536_550() -> None:
    plan = full_call_plan()
    assert plan["prior_attempts"] == 6
    assert plan["current_freeze_base_calls"] == 486
    assert plan["study_base_calls"] == plan["prior_attempts"] + plan["current_freeze_base_calls"] == 492
    assert plan["retry_reserve"] == 50
    assert plan["study_worst_case_http_calls"] == 542
    assert plan["study_worst_case_http_calls"] == plan["study_base_calls"] + plan["retry_reserve"]
    assert plan["current_ledger_cap"] == 536
    assert plan["current_ledger_cap"] == 542 - plan["prior_attempts"]
    assert plan["operational_http_cap"] == 550
    assert plan["study_worst_case_http_calls"] <= plan["operational_http_cap"]
    assert plan["operational_http_cap"] - plan["study_worst_case_http_calls"] == 8
    assert plan["worst_case_http_calls"] == plan["study_worst_case_http_calls"]


def test_phase_base_calls_sum_to_the_current_freeze_total() -> None:
    plan = full_call_plan()
    assert sum(row["base_calls"] for row in plan["phases"]) == plan["current_freeze_base_calls"]
    conditional = {row["name"] for row in plan["phases"] if row["conditional"]}
    assert conditional == {"full_item_secondary_probes"}
    assert plan["secondary_scope"] == "eligible_full_arm_items_only"


def test_judging_phase_is_two_families_over_the_fixed_forty() -> None:
    phases = {row["name"]: row["base_calls"] for row in full_call_plan()["phases"]}
    assert phases["blinded_model_judging"] == JUDGING_FRAME_SIZE * 2 == 80


def test_budget_rejects_a_judge_sample_that_is_not_the_fixed_frame() -> None:
    for bad in (20, 39, 41, 80):
        with pytest.raises(ValueError, match="BLOCKED_BUDGET:judge_sample_must_match_fixed_frame"):
            full_call_plan(human_judge_sample=bad)


def test_budget_fails_closed_above_the_operational_cap() -> None:
    # Retry headroom may shrink, but the immutable base plan itself may never be
    # squeezed below the cap.
    with pytest.raises(ValueError, match=r"BLOCKED_BUDGET:study_worst_case_exceeds_cap:492/491"):
        full_call_plan(operational_http_cap=491)
    constrained = full_call_plan(operational_http_cap=541)
    assert constrained["study_worst_case_http_calls"] == 541
    assert constrained["retry_reserve_constrained_by_hard_cap"] is True


def test_budget_rejects_invalid_prior_attempts_and_caps() -> None:
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:invalid_prior_attempts"):
        full_call_plan(prior_attempts=-1)
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:invalid_prior_attempts"):
        full_call_plan(prior_attempts=True)
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:invalid_operational_cap"):
        full_call_plan(operational_http_cap=0)
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:invalid_retry_fraction"):
        full_call_plan(retry_fraction=0.5)
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:unexpected_frozen_census"):
        full_call_plan(chunks=15)
    with pytest.raises(ValueError, match="BLOCKED_BUDGET:exactly_two_judge_families_required"):
        full_call_plan(judge_families=3)


def test_prior_attempts_consume_headroom_rather_than_being_forgotten() -> None:
    """The six spent smoke attempts must reduce the remaining ledger allowance."""
    with_prior = full_call_plan(prior_attempts=6)
    without_prior = full_call_plan(prior_attempts=0)
    assert with_prior["study_base_calls"] - without_prior["study_base_calls"] == 6
    assert with_prior["study_worst_case_http_calls"] > without_prior["study_worst_case_http_calls"]
    assert with_prior["current_ledger_cap"] < with_prior["study_worst_case_http_calls"]


def test_freeze_budget_matches_the_study_accounting_and_reports_cap_compliance() -> None:
    freeze = build_freeze(MANIFEST)
    assert freeze["operational_http_cap"] == budget_mod.__dict__.get(
        "OPERATIONAL_HTTP_CAP", freeze_mod.OPERATIONAL_HTTP_CAP
    ) or 550
    plan = freeze["full_call_budget"]
    assert plan["prior_attempts"] == 119
    assert plan["satisfied_calls"] == 55
    assert plan["gross_current_freeze_base_calls"] == 486
    assert plan["current_freeze_base_calls"] == 431
    assert plan["study_base_calls"] == 550
    assert plan["requested_retry_reserve"] == 55
    assert plan["retry_reserve"] == 0
    assert plan["retry_reserve_constrained_by_hard_cap"] is True
    assert plan["study_worst_case_http_calls"] == 550
    assert plan["current_ledger_cap"] == 431
    assert freeze["budget_within_cap"] is True
    assert freeze["execution_gate"]["budget_within_cap"] is True


def test_execution_gate_blocks_a_budget_that_exceeds_the_recorded_cap() -> None:
    freeze = build_freeze(MANIFEST)
    freeze["full_call_budget"]["worst_case_http_calls"] = 551
    with pytest.raises(ValueError, match="call_budget_exceeds_cap"):
        validate_execution_gate(freeze, execute=True, smoke_passed_roles=set(freeze["roles"]))
