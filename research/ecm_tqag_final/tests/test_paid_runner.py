from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.run.paid import build_paid_runner, PaidRunBlocked


def _authorized(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    manifest = root.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
    controls = root / "fixtures" / "sensitivity_controls.json"
    freeze = tmp_path / "freeze.json"
    from ecm_tqag.freeze import build_freeze
    freeze.write_text(json.dumps(build_freeze(manifest), sort_keys=True), encoding="utf-8")
    from ecm_tqag.io import sha256_file
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({
        "schema": "ecm-tqag.role-smoke.v1",
        "freeze_sha256": sha256_file(freeze),
        "status": "PASS",
        "passed_roles": ["answerer_a", "answerer_b", "generator", "image_auditor", "model_judge_a", "model_judge_b"],
    }), encoding="utf-8")
    return freeze, smoke, manifest, controls


def test_build_paid_runner_is_authorized_but_makes_no_network_call(tmp_path: Path) -> None:
    freeze, smoke, manifest, controls = _authorized(tmp_path)
    calls: list[dict] = []
    runner = build_paid_runner(
        freeze_path=freeze, smoke_path=smoke, manifest_path=manifest,
        controls_path=controls, run_dir=tmp_path / "run", execute=True,
        worker=lambda task: calls.append(task) or {"status": "DRYRUN", "calls_used": task["calls"]},
    )
    assert runner.authorization["status"] == "AUTHORIZED"
    assert calls == []
    assert runner.run_phase("role_smoke")["completed"] == 6
    assert len(calls) == 6


def test_paid_runner_blocks_secondary_without_floor(tmp_path: Path) -> None:
    freeze, smoke, manifest, controls = _authorized(tmp_path)
    runner = build_paid_runner(
        freeze_path=freeze, smoke_path=smoke, manifest_path=manifest,
        controls_path=controls, run_dir=tmp_path / "run", execute=True,
        worker=lambda task: {"status": "DRYRUN", "calls_used": task["calls"]},
    )
    with pytest.raises(PaidRunBlocked, match="secondary_requires_floor"):
        runner.run_phase("secondary_probes", floor_passed=False)


def test_paid_runner_rejects_missing_worker(tmp_path: Path) -> None:
    freeze, smoke, manifest, controls = _authorized(tmp_path)
    with pytest.raises(PaidRunBlocked, match="worker_missing"):
        build_paid_runner(
            freeze_path=freeze, smoke_path=smoke, manifest_path=manifest,
            controls_path=controls, run_dir=tmp_path / "run", execute=True,
        )
