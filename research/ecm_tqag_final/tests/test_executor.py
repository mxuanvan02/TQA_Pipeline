from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.run.executor import PhaseExecutor, ExecutionBlocked


def _plan() -> dict:
    return {
        "schema": "ecm-tqag.phase-plan.v1",
        "secondary_is_conditional": True,
        "phases": ["role_smoke", "construction", "secondary_probes"],
        "counts": {"role_smoke": 1, "construction": 1, "secondary_probes": 1},
        "dependencies": {
            "role_smoke": [],
            "construction": ["role_smoke"],
            "secondary_probes": ["construction"],
        },
        "tasks": [
            {"phase": "role_smoke", "task_id": "smoke:generator", "calls": 1},
            {"phase": "construction", "task_id": "construct:full:c1:1", "calls": 1, "chunk_id": "c1", "arm": "full"},
            {"phase": "secondary_probes", "task_id": "probe:answerer_a:c1:control", "calls": 1, "chunk_id": "c1"},
        ],
    }


def test_executor_enforces_dependencies_and_resumes(tmp_path: Path) -> None:
    calls: list[str] = []
    def worker(task: dict) -> dict:
        calls.append(task["task_id"])
        return {"ok": True, "task_id": task["task_id"], "calls_used": task["calls"]}

    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(), worker=worker)
    with pytest.raises(ExecutionBlocked, match="secondary_requires_floor"):
        ex.run(phase="secondary_probes", floor_passed=False)
    first = ex.run(phase="role_smoke")
    assert first["completed"] == 1
    second = ex.run(phase="role_smoke")
    assert second["skipped"] == 1
    assert calls == ["smoke:generator"]


def test_executor_blocks_later_phase_until_prerequisite_is_complete(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(),
                       worker=lambda t: {"ok": True, "calls_used": t["calls"]})
    with pytest.raises(ExecutionBlocked, match="prerequisite_incomplete:role_smoke"):
        ex.run(phase="construction")


def test_executor_rejects_call_parity_mismatch(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(),
                       worker=lambda t: {"ok": True, "calls_used": 99})
    with pytest.raises(ExecutionBlocked, match="call_parity_mismatch"):
        ex.run(phase="role_smoke")


def test_executor_allows_zero_call_not_applicable_probe(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(),
                       worker=lambda t: {"status": "OK", "calls_used": t["calls"]})
    ex.run(phase="role_smoke")
    ex.run(phase="construction")
    ex.worker = lambda t: {"status": "NOT_APPLICABLE", "reason": "full_item_not_eligible",
                           "calls_used": 0}
    report = ex.run(phase="secondary_probes", floor_passed=True)
    assert report["completed"] == 1


def test_executor_rejects_arbitrary_zero_call_probe(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(),
                       worker=lambda t: {"status": "OK", "calls_used": t["calls"]})
    ex.run(phase="role_smoke")
    ex.run(phase="construction")
    ex.worker = lambda t: {"status": "NOT_APPLICABLE", "reason": "made_up", "calls_used": 0}
    with pytest.raises(ExecutionBlocked, match="call_parity_mismatch"):
        ex.run(phase="secondary_probes", floor_passed=True)


def test_executor_rejects_tampered_result(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(),
                       worker=lambda t: {"ok": True, "calls_used": t["calls"]})
    ex.run(phase="role_smoke")
    result = next(tmp_path.glob("results/*.json"))
    result.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ExecutionBlocked, match="result_hash_mismatch"):
        PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(), worker=lambda t: {})


def test_executor_accepts_paid_schema_rejection_with_normal_call_parity(tmp_path: Path) -> None:
    plan = _plan()
    plan["phases"].append("extraction")
    plan["counts"]["extraction"] = 1
    plan["dependencies"]["extraction"] = []
    plan["tasks"].append({"phase": "extraction", "task_id": "extract:graph:01", "calls": 1})
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan,
        worker=lambda _t: {"status": "SCHEMA_REJECTED", "reason": "grid_bbox_out_of_range",
                           "calls_used": 1, "response_sha256": "a" * 64})
    assert ex.run(phase="extraction")["completed"] == 1


def test_executor_rejects_zero_call_schema_rejection(tmp_path: Path) -> None:
    plan = _plan()
    plan["phases"].append("extraction")
    plan["counts"]["extraction"] = 1
    plan["dependencies"]["extraction"] = []
    plan["tasks"].append({"phase": "extraction", "task_id": "extract:graph:01", "calls": 1})
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan,
        worker=lambda _t: {"status": "SCHEMA_REJECTED", "reason": "grid_bbox_out_of_range",
                           "calls_used": 0, "response_sha256": "a" * 64})
    with pytest.raises(ExecutionBlocked, match="call_parity_mismatch"):
        ex.run(phase="extraction")


def _import_origin(task_id: str) -> dict[str, str]:
    return {
        "origin_freeze_sha256": "a" * 64,
        "origin_run": "paid_origin",
        "origin_task_id": task_id,
        "origin_idempotency_key": "b" * 64,
        "origin_payload_sha256": "c" * 64,
        "origin_response_sha256": "d" * 64,
    }


def test_executor_imports_only_terminal_planner_schema_rejection(tmp_path: Path) -> None:
    plan = _plan()
    task = plan["tasks"][1]
    task["construction_stage"] = "planner"
    task["parent_task_id"] = None
    task_id = task["task_id"]
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan, worker=lambda _t: {})
    result = {
        "status": "SCHEMA_REJECTED",
        "calls_used": 1,
        "reason": "planner_schema_rejected:ValueError",
        "construction_stage": "planner",
    }
    ex.import_completed(
        task_id, result, status="SCHEMA_REJECTED",
        import_origin=_import_origin(task_id),
    )
    assert task_id in ex.completed
    stored = json.loads(next(tmp_path.glob("results/*.json")).read_text(encoding="utf-8"))
    assert stored["status"] == "SCHEMA_REJECTED"
    assert stored["import_origin"]["origin_task_id"] == task_id


@pytest.mark.parametrize("status", ["PLAN_OK", "PARSED", "GUARD_REJECTED"])
def test_executor_rejects_other_construction_imports(tmp_path: Path, status: str) -> None:
    plan = _plan()
    task = plan["tasks"][1]
    task["construction_stage"] = "planner"
    task["parent_task_id"] = None
    task_id = task["task_id"]
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan, worker=lambda _t: {})
    with pytest.raises(ExecutionBlocked, match="import_construction_not_schema_rejection"):
        ex.import_completed(
            task_id, {"status": status, "calls_used": 1}, status=status,
            import_origin=_import_origin(task_id),
        )
