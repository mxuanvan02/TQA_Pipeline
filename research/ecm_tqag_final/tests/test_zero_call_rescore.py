from __future__ import annotations

from pathlib import Path

from ecm_tqag.run.executor import PhaseExecutor


def test_executor_accepts_zero_call_deterministic_rescore(tmp_path: Path) -> None:
    plan = {
        "schema": "ecm-tqag.phase-plan.v1",
        "phases": ["construction"],
        "dependencies": {"construction": []},
        "counts": {"construction": 0},
        "tasks": [{
            "phase": "construction", "task_id": "construct:gates_off:c01:1",
            "calls": 0, "arm": "gates_off", "deterministic_rescore": True,
        }],
    }
    ex = PhaseExecutor(
        tmp_path, freeze_sha256="f" * 64, plan=plan,
        worker=lambda task: {"status": "OK", "calls_used": 0},
    )
    report = ex.run(phase="construction")
    assert report["completed"] == 1
