from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ecm_tqag.run.executor import ExecutionBlocked, PhaseExecutor


def _plan() -> dict:
    return {
        "schema": "ecm-tqag.phase-plan.v1",
        "secondary_is_conditional": True,
        "phases": ["role_smoke", "construction", "secondary_probes"],
        "dependencies": {"role_smoke": [], "construction": ["role_smoke"],
                         "secondary_probes": ["construction"]},
        "counts": {"role_smoke": 1, "construction": 1, "secondary_probes": 1},
        "tasks": [
            {"phase": "role_smoke", "task_id": "smoke:generator", "calls": 1},
            {"phase": "construction", "task_id": "construct:full:c1:1", "calls": 1,
             "chunk_id": "c1", "arm": "full"},
            {"phase": "secondary_probes", "task_id": "probe:answerer_a:c1:control", "calls": 1,
             "chunk_id": "c1"},
        ],
    }


def _worker(task: dict) -> dict:
    return {"ok": True, "calls_used": task["calls"]}


def test_executor_rejects_result_path_escape(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(), worker=_worker)
    ex.run(phase="role_smoke")
    checkpoint = tmp_path / "EXECUTION.jsonl"
    record = json.loads(checkpoint.read_text().splitlines()[0])
    record["result_path"] = "../outside.json"
    checkpoint.write_text(json.dumps(record) + "\n")
    with pytest.raises(ExecutionBlocked, match="result_path_invalid"):
        PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(), worker=_worker)


def test_executor_writes_private_sidecar_and_flushes_checkpoint(tmp_path: Path) -> None:
    ex = PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=_plan(), worker=_worker)
    ex.run(phase="role_smoke")
    result = next((tmp_path / "results").glob("*.json"))
    assert result.stat().st_mode & 0o077 == 0
    assert (tmp_path / "EXECUTION.jsonl").stat().st_mode & 0o077 == 0


def test_executor_rejects_invalid_task_call_count(tmp_path: Path) -> None:
    plan = _plan()
    plan["tasks"][0]["calls"] = -1
    with pytest.raises(ExecutionBlocked, match="invalid_task_calls"):
        PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan, worker=_worker)


def test_executor_rejects_phase_count_mismatch(tmp_path: Path) -> None:
    plan = _plan()
    plan["counts"]["role_smoke"] = 9
    with pytest.raises(ExecutionBlocked, match="phase_count_mismatch"):
        PhaseExecutor(tmp_path, freeze_sha256="f" * 64, plan=plan, worker=_worker)
