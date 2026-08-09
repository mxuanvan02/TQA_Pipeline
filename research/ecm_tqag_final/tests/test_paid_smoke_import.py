from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.freeze import build_freeze
from ecm_tqag.io import sha256_file
from ecm_tqag.run.paid import PaidRunBlocked, build_paid_runner

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"
ROLES = (
    "answerer_a", "answerer_b", "generator", "image_auditor",
    "model_judge_a", "model_judge_b",
)


def _files(tmp_path: Path) -> tuple[Path, Path]:
    freeze_path = tmp_path / "freeze.json"
    freeze = build_freeze(MANIFEST)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps({
        "schema": "ecm-tqag.role-smoke.v1",
        "freeze_sha256": sha256_file(freeze_path),
        "status": "PASS",
        "passed_roles": list(ROLES),
        "http_attempts_used": 6,
        "results": [
            {
                "role": role,
                "provider": freeze["roles"][role]["provider"],
                "model": freeze["roles"][role]["model"],
                "vision_required": freeze["roles"][role]["vision_required"],
                "status": "PASS",
                "idempotency_key": f"key-{role}",
                "response_sha256": "a" * 64,
                "usage": None,
            }
            for role in ROLES
        ],
    }, sort_keys=True), encoding="utf-8")
    return freeze_path, smoke_path


def test_paid_runner_imports_verified_smoke_without_recalling_models(tmp_path: Path) -> None:
    freeze, smoke = _files(tmp_path)
    calls: list[str] = []
    runner = build_paid_runner(
        freeze_path=freeze, smoke_path=smoke, manifest_path=MANIFEST,
        controls_path=CONTROLS, run_dir=tmp_path / "run", execute=True,
        worker=lambda task: calls.append(task["task_id"]) or {
            "status": "DRYRUN", "calls_used": task["calls"]
        },
    )

    report = runner.run_phase("role_smoke")
    assert report == {"phase": "role_smoke", "completed": 0, "skipped": 6}
    assert calls == []

    # Extraction is authorized because all six smoke tasks were imported as
    # externally completed, not because the dependency was silently deleted.
    extraction = runner.run_phase("extraction")
    assert extraction["completed"] == 54
    assert len(calls) == 54

    checkpoint = (tmp_path / "run" / "EXECUTION.jsonl").read_text(encoding="utf-8")
    assert checkpoint.count('"status":"IMPORTED_VERIFIED_SMOKE"') == 6


def test_paid_runner_rejects_smoke_rows_that_do_not_match_frozen_roster(tmp_path: Path) -> None:
    freeze, smoke = _files(tmp_path)
    body = json.loads(smoke.read_text(encoding="utf-8"))
    body["results"][0]["model"] = "wrong-model"
    smoke.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises((ValueError, PaidRunBlocked), match="smoke_results_mismatch"):
        build_paid_runner(
            freeze_path=freeze, smoke_path=smoke, manifest_path=MANIFEST,
            controls_path=CONTROLS, run_dir=tmp_path / "run", execute=True,
            worker=lambda task: {"status": "DRYRUN", "calls_used": task["calls"]},
        )
